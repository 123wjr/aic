# 当前实验脚本流程与结构

本文用于后续模块化重构前理解现有实验流程。目标：把配置、模型差异、prompt、数据、输出从脚本硬编码中逐步拆出来，减少后续调整实验配置时的代码改动。

## 入口结构

```mermaid
flowchart TD
    A[server.env] --> B[run_server.sh]
    B --> C{命令}
    C -->|check| D[检查配置 / query / Python 编译 / 依赖]
    C -->|smoke| E[小样本 LIMIT 跑一次]
    C -->|full / resume| F[完整实验]
    C -->|validate| G[校验 predictions.json]
    C -->|package| H[打包 submission.zip]

    E --> I[run_inference.sh]
    F --> I
    I --> J[src/inference.py]
    J --> K[grounding.runner.run]
```

## 核心推理流程

```mermaid
flowchart TD
    A[inference.py 解析 CLI 参数] --> B[选择 backend]
    B --> B1[QwenBackend]
    B --> B2[LocateAnythingBackend]
    B --> B3[InternVLBackend]

    A --> C[选择 prompt_profile]
    C --> D[load_provider]
    D --> E[RunConfig]

    E --> F[runner.run]
    F --> G[load_query_groups]
    G --> H[按 image_field 分组]
    H --> I[prepare_prompt_results]
    I --> J[PromptCache 读写 prompts.jsonl]

    J --> K[加载断点 partial/output]
    K --> L[过滤已完成 query]
    L --> M[按 batch_size 组 InferenceUnit]
    M --> N[backend.infer]
    N --> O{OOM?}
    O -->|是| P[_infer_resilient 拆 batch / 拆 unit]
    P --> N
    O -->|否| Q[parse_bbox]
    Q --> R[sanitize_bbox]
    R --> S[写 raw.jsonl 审计]
    S --> T[atomic_dump partial]
    T --> U{全部完成?}
    U -->|否| M
    U -->|是| V[写 predictions.json]
    V --> W[删除 partial]
```

## 输出链路

```mermaid
flowchart LR
    A[predictions.json] --> B[validate_submission.py]
    B --> C[package_submission.py]
    C --> D[submission.zip]

    E[raw.jsonl] --> F[审计: prompt/raw_output/parse_status/runtime_stats]
    G[prompts.jsonl] --> H[prompt 缓存]
    I[partial.json] --> J[断点恢复]
    K[*.log] --> L[运行日志]
```

## 多 GPU 分片

```mermaid
flowchart TD
    A[run_inference.sh] --> B{backend == locateanything 且 GPU_IDS 多张?}
    B -->|否| C[单进程 inference.py]
    B -->|是| D[每张 GPU 一个 worker]
    D --> E[CUDA_VISIBLE_DEVICES=gpuN]
    E --> F[--shard_index N --shard_count 总数]
    F --> G[每个 worker 产出 predictions.gpuN.json]
    G --> H[merge_predictions.py]
    H --> I[最终 predictions.json]
```

## 当前模块职责

1. `run_server.sh`
   实验命令入口：`check / smoke / full / resume / validate / package`。读取 `server.env`。

2. `run_inference.sh`
   把环境变量拼成 CLI 参数；处理 LocateAnything 多 GPU 分片；最后校验和打包。

3. `src/inference.py`
   唯一 Python 推理入口。负责 argparse、backend config、prompt provider、`RunConfig`。

4. `grounding/runner.py`
   真正流程编排：加载数据、生成 prompt、断点恢复、batch、OOM 拆分、解析 bbox、写结果。

5. `grounding/backends/*.py`
   模型差异层。当前有 `qwen`、`locateanything`、`internvl`。

## 当前可调项

```mermaid
mindmap
  root((实验配置))
    模型
      MODEL_BACKEND
      MODEL
      dtype
      max_new_tokens
      attention
    数据
      DATA_DIR
      QUERY_FILE
      image_field
      limit
    prompt
      PROMPT_PROFILE
      PROMPT_PROVIDER
      PromptCache namespace
    batch
      BATCH_SIZE
      SAVE_EVERY
      shard_index
      shard_count
    输出
      OUTPUT_ROOT
      RUN_NAME
      raw_output
      partial
      prompt_cache
      submission_zip
```

## 硬编码与模块化重点

1. `server.env.example` 和 `run_inference.sh` 里 backend 默认 profile 重复写了两遍。后续应收敛到一个配置源。

2. `run_inference.sh` 写死 `QUERY_FILE="${DATA_DIR}/queries/queries.json"`。后续应允许实验配置显式声明 query 文件。

3. `PROMPT_PROFILES` 在 `prompts.py` 写死为 tuple。后续可改成 registry/config，避免新增 profile 时改核心代码。

4. `inference.py` 的 argparse 混合了通用参数和各 backend 参数。后续可拆成 `ExperimentConfig + BackendConfig`。

5. backend 选择是 `if/elif/else`。后续可改成 backend registry：`name -> config parser -> backend factory`。

## 建议的目标结构

```mermaid
flowchart TD
    A[experiment.yaml] --> B[ExperimentConfig]
    B --> C[BackendRegistry]
    B --> D[PromptRegistry]
    B --> E[DatasetConfig]
    B --> F[OutputConfig]

    C --> G[BackendFactory]
    D --> H[PromptProvider]
    E --> I[load_query_groups]
    F --> J[RunConfig]

    G --> K[runner.run]
    H --> K
    I --> K
    J --> K
```

最小重构方向：先把 `server.env + run_inference.sh 参数拼接` 合并成一个 `experiment.yaml` 或 JSON 配置加载器；`runner.py` 暂时别动，它已经相对干净。
