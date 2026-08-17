# 多模态查询目标定位工作流

`src/inference.py` 是唯一推理入口。三个 backend 共享数据分组、全量提示词生成与缓存、批处理、OOM 拆分、断点、raw 审计、bbox 解析和提交校验；backend 只消费已经生成的 prompt，不再拼接模型指令。

默认 prompt profile 由 backend 选择，也可以显式设置全局 `PROMPT_PROFILE`/`--prompt_profile`：

```text
qwen_json | locateanything | internvl_json_union | internvl_native_box | internvl_minimal
```

自定义复杂逻辑放在 `--prompt_provider module:function`。标记 `function.__prompt_batch__ = True` 后，函数一次收到完整的 `tuple[PromptRequest, ...]`，返回 `query_id -> prompt`，不会随推理 batch 被重复调用。示例见 `examples/prompt_provider.py`。

LocateAnything 还需要 `opencv-python-headless`、`decord` 和 `lmdb`，并要求 Transformers 4.x；这些约束已经包含在 `requirements.txt` 中。Transformers 5.x 会与官方远程模型代码的 attention API 不兼容。

```bash
cd grounding_workflow
cp server.env.example server.env
bash run_server.sh check
bash run_server.sh smoke
bash run_server.sh full
```

输出 `predictions.json` 只保留查询原字段并新增归一化 `bbox: [x1,y1,x2,y2]`。

LocateAnything 不支持在单个官方 runtime 内自动做模型并行。配置 `GPU_IDS=0,1` 后，脚本会启动两个独立 worker，每个 worker 绑定一张 GPU 并按图像组分片；worker 内遇到 OOM 会自动降低后续 batch size。

InternVL3.5-38B-HF 使用 `--backend internvl`。它默认使用 Transformers 的 `device_map=auto` 让一个进程看到的多张 GPU 共同承载模型，不要给 InternVL 设置 LocateAnything 的多进程分片。backend 会缓存同图的视觉特征，缓存大小由 `INTERNVL_IMAGE_CACHE_SIZE` 控制。
