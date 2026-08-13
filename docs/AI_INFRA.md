# KUN AI Infra

KUN AI Infra 把个人知识库中的检索链路变成可运行、可观测、可比较、可回归的本地实验平台。它仍是 Windows-first、单用户、本地优先软件，不宣称企业级生产部署。

## 运行隔离

| 副本 | Git | 数据目录 |
| --- | --- | --- |
| 稳定版 | `codex/baseline-v1` / `kun-v1-baseline-2026-08-13` | `%LOCALAPPDATA%\KUN` |
| AI Infra | `codex/ai-infra-v1` | `%LOCALAPPDATA%\KUN-AI-Infra` |

AI Infra 副本的 `启动-KUN.cmd` 会设置独立数据目录。不要把 API Key、资料、SQLite、向量、索引或实验私有数据提交到 Git。

## Provider 配置

API Key 优先从 Windows Credential Manager 读取，环境变量只用于本地开发。

```powershell
$env:DASHSCOPE_API_KEY="..."
$env:DEEPSEEK_API_KEY="..."
$env:DASHSCOPE_RERANK_BASE_URL="https://{workspace}.cn-beijing.maas.aliyuncs.com/compatible-api/v1"
```

可选配置：

```powershell
$env:DASHSCOPE_EMBEDDING_MODEL="qwen3.7-text-embedding"
$env:DASHSCOPE_EMBEDDING_DIMENSION="1024"
$env:DASHSCOPE_RERANK_MODEL="qwen3-rerank"
$env:DEEPSEEK_MODEL="deepseek-v4-flash"
```

在“设置 → 模型与 API”分别验证 DeepSeek Chat、百炼 Embedding 和百炼 Reranker。绿色状态只代表当前配置生命周期内的真实连接测试成功。

## 基础设施层

### 持久任务运行器

`infra_jobs` 是任务事实来源，线程池只负责执行已经原子领取的任务。任务保存：

- 状态、阶段和 0–100 进度；
- Worker、尝试次数、最大重试次数；
- 5 秒心跳；
- 幂等键；
- 取消请求、失败码和脱敏结果摘要；
- 应用重启后的重新排队。

文档索引、图片分析、索引代次构建、质量实验、候选题生成和性能压测都通过同一运行器执行。

### Trace / Span

`infra_traces` 保存一次检索、索引、评测或性能运行；`infra_spans` 保存内部阶段。常见 Span：

```text
BM25 → Embedding query → FAISS → RRF → Rerank
Load chunks → Cache lookup → Embedding batch → FAISS build
```

持久化前会对 `api_key`、`authorization`、`credential`、`password`、`secret` 等字段脱敏。Trace 只展示可观测事件，不记录模型私有推理过程。

### 不可变索引代次

代次配置 hash 包含：

- Provider、模型和维度；
- Flat/HNSW；
- Chunk size/overlap；
- Parser/Chunker 版本；
- 当前文档内容指纹。

不同 Chunk 配置会重新解析并真正生成代次专属 Chunk 与 FTS，不是只改配置标签。Embedding 缓存按文本 hash、Provider、模型和维度隔离。构建完成后先写临时文件，再原子替换 `index.faiss` 与 `manifest.json`。只有 `ready` 且清单存在的代次才能激活；旧代次保留用于回滚。

构建前 UI 显示 Chunk 数、缓存命中/缺失、预计批次、重发字符数及费用口径。输入“重建此索引”后才入队。

## 质量评测

### Gold 工作流

1. 导入已有人工题或创建版本化数据集。
2. 经用户确认后，把有限文档片段发送给 DeepSeek 生成候选题。
3. 候选题一律保存为 `draft`。
4. 用户核对问题、来源、定位、难度和类型后标记 `accepted` 或 `rejected`。
5. 正式实验只统计 `accepted`。

候选题生成需要开启“云端文档临时理解”，并输入“生成候选题”确认。单次最多生成 20 条，推荐每次审核 10 条。

### 指标定义

| 指标 | 解释 |
| --- | --- |
| Document Recall@1/5/10 | 前 K 条是否包含正确文档 |
| Evidence Recall@1/5/10 | 前 K 条是否命中正确文档和 locator |
| MRR | 首个正确证据排名的倒数均值 |
| nDCG@10 | 使用 Gold relevance 的分级排序质量 |
| Citation resolvable rate | 返回结果是否同时具有可解析 chunk id 和 locator |
| P50/P95/P99 | 单题完整 Retrieval pipeline 延迟 |

实验强制至少取 10 条候选，避免用 Top 5 结果伪算 Recall@10。每题保存排名、阶段详情、Bad Case 分类和 Trace ID。

### Bad Case

- `document_found_evidence_missed`：找到了文档，但没有命中正确证据位置。
- `not_retrieved`：候选结果存在，但没有正确文档。
- `empty_result`：没有任何候选。

## 性能压测

性能轨使用固定 seed 的归一化随机向量，支持 1k、10k、100k：

- Flat/HNSW 构建时间；
- 查询 P50/P95/P99 与 QPS；
- HNSW 相对 Flat Top 10 的 ANN Recall；
- 估算向量内存；
- 1.2 GB 安全阈值。

这些结果只说明本机基础设施性能，不代表真实文档检索质量。UI 和报告都带有 `NO QUALITY CLAIM` 边界。

## Regression Gate

候选与 Baseline 必须使用同一数据集版本并都已成功。当前门禁：

- Evidence Recall@5 绝对下降不得超过 0.01；
- Document Recall@5 绝对下降不得超过 0.01；
- Retrieval P95 增长不得超过 20%；
- 使用固定 seed、1,000 次 paired bootstrap 报告 Evidence Recall 差值 95% 区间。

任一规则失败即阻塞候选配置。门禁是工程决策信号，不代替人工分析 Bad Case。

## 推荐实验顺序

先创建一个可复现 Baseline，再一次只改变一个变量：

```text
BM25
Dense
BM25 + Dense + RRF
Hybrid + qwen3-rerank

qwen3.7-text-embedding / 1024
text-embedding-v4 / 1024
text-embedding-v4 / 256

400 / overlap 80
700 / overlap 120
1000 / overlap 150
```

每次实验后先看总体指标，再用 Retrieval Duel 和 Bad Case 判断变化发生在哪个阶段，最后运行 Regression Gate。

## API 入口

```text
GET/POST  /api/infra/index-generations
POST      /api/infra/retrieval-duel
GET       /api/infra/traces/{trace_id}
GET       /api/infra/jobs
GET/POST  /api/eval/datasets
POST      /api/eval/experiments
POST      /api/eval/compare
GET       /api/eval/experiments/{run_id}/report
POST      /api/infra/performance-benchmarks
```

## 验证

```powershell
npm run build
cd backend
python -m pytest tests -q --basetemp .test-tmp -p no:cacheprovider
```

发布任何指标前，报告必须写明数据集规模、模型与维度、Chunk 配置、机器信息、日期和 Git revision。100 条人工 Gold、Recall@5 ≥ 0.85、引用定位成功率 ≥ 0.95、100k Chunk Retrieval P95 ≤ 1.5 秒仍是发布门禁，而不是当前默认宣称。
