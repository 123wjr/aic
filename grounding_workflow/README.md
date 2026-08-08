# 多模态查询目标定位工作流

`src/inference.py` 是唯一推理入口。`--backend qwen` 和 `--backend locateanything` 共享数据分组、提示词缓存、批处理、OOM 拆分、断点、raw 审计、bbox 解析和提交校验；模型差异只存在于 `src/grounding/backends/`。

```bash
cd grounding_workflow
cp server.env.example server.env
bash run_server.sh check
bash run_server.sh smoke
bash run_server.sh full
```

输出 `predictions.json` 只保留查询原字段并新增归一化 `bbox: [x1,y1,x2,y2]`。
