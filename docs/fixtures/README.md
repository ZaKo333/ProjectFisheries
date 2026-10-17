# docs/fixtures —— 探针留档与接口快照

这里放的是**一次性探针的输出留档**和**接口的离线快照**，评审时不用重跑就能核对结论；
它们不是给人从头读的说明书（说明书在 docs/ 与 README.md）。

| 文件 | 怎么重新生成 |
|---|---|
| _api_samples.json | .\\.venv\\Scripts\\python.exe scripts\\probes\\_make_api_samples.py |
| _frontend_spec.txt | .\\.venv\\Scripts\\python.exe scripts\\probes\\_probe_frontend_spec.py |
| _doc_example.txt | .\\.venv\\Scripts\\python.exe scripts\\probes\\_make_doc_example.py > docs\\fixtures\\_doc_example.txt |
| _probe_*.txt | 跑 scripts/probes/_probe_*.py 后把输出重定向进来 |
| _sample_api_responses.json / _routes.txt | 早期探针留档；要看当前契约，以 _api_samples.json 与 docs/openapi.json 为准 |

> 不建议手改：scripts/probes/_make_api_samples.py 已把时间 / 耗时 / 随机 id 换成占位符，
> 重跑应逐字节一致；	ests/unit/test_datasets.py 还会检查快照里不得出现本机路径。
