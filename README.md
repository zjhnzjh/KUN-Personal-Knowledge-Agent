# KUN Personal Knowledge Agent

> Knowledge · Understanding · Navigation

KUN 是一个面向 Windows 的本地优先个人知识智能体。它将 PDF、Word、Markdown、Excel 和图片整理为可检索、可引用的个人知识空间，并通过名为“坤坤”的原创助手完成资料问答、图片搜索与可控长期记忆。

项目面向个人学习与知识管理，不宣称企业级多租户、组织权限或合规认证能力。

## 已完成的真实能力

- ChatGPT 式对话：用户消息气泡、流式回答、简短执行状态、安全 Markdown 渲染和完整回答不截断。
- 历史对话：自动保存、标题生成、搜索、重新打开、继续追问和删除。
- 文件导入：支持单个或批量选择、拖放，以及在主输入框中用 `Ctrl+V` 粘贴图片或从资源管理器复制的文件。
- 导入确认：Agent 先生成标题、摘要和标签，用户确认或修改后才建立正式索引。
- 文档解析：PDF、DOCX、Markdown、TXT、XLSX/XLS、CSV 与常见图片格式。
- 后台索引：显示解析、切分和 Embedding 进度；支持失败提示、重试、重启恢复和跨文档向量缓存。
- Hybrid RAG：SQLite FTS5/BM25 与云端 Embedding 召回，经融合后生成带页码引用的回答。
- 来源预览：点击引用可打开文档面板；聊天区与 PDF 区之间可拖动调整宽度。
- 图片理解：确认导入前先用视觉模型生成描述与 OCR，再据此生成标题、摘要和标签；确认后建立图片语义索引。
- 图片搜索：按画面含义、图片文字和标签进行自然语言搜索，返回本地图片与理解结果。
- 可控 Memory：短期 Memory 保留当前对话工作上下文；姓名、所在地、稳定偏好、目标等可形成长期候选，用户确认后才参与回答。
- Skill 中心：查看内置 Skill 的 `SKILL.md`、Tool、权限和超时，并可视化创建符合 frontmatter 规范的个人 Skill。
- Tool 中心：查看真实可用性和运行记录；`web.fetch` 可安全读取用户明确提供的公开 HTTPS 网页，`web.search` 在搜索提供方未配置时明确显示不可用。
- RAG 实验室：创建人工标注问题，实际运行 Recall@K、MRR、nDCG、平均延迟和 P95 延迟，并查看逐题命中排名。
- AI Infra 控制台：持久任务队列、心跳与重启恢复、Trace/Span 瀑布、不可变索引代次、Retrieval Duel、实验矩阵、Bad Case 和回归门禁。
- 可复现实验：真正按 400/700/1000 三种 Chunk 配置重切片，隔离 Embedding 模型与维度缓存，并记录数据集 hash、配置 hash、Git revision 和机器信息。
- 性能压测：使用固定 seed 的 1k/10k/100k 生成向量测量 Flat/HNSW 构建、QPS、P50/P95/P99 与 ANN Recall；与人工 Gold 质量结果严格分开。
- 多知识空间：默认提供“AI Agent 学习”“课程与读书”“个人项目”“求职与成长”，也可新建空间。
- Agent Workflow：LangGraph 负责 Query 理解、Skill/Tool 选择、检索、回答与 Memory 建议。
- Tool System：统一参数校验、权限范围、可用状态、错误与 SQLite 执行记录；不显示虚假成功。
- 密钥安全：API Key 存入 Windows Credential Manager，不进入前端、Git 或普通配置文件。

## 打开 KUN

在桌面打开：

```text
C:\Users\你的用户名\Desktop\KUN-Personal-Knowledge-Agent
```

双击：

```text
启动-KUN.cmd
```

保持启动窗口开启，然后访问：

```text
http://127.0.0.1:3000/
```

关闭启动窗口或按 `Ctrl+C` 会停止本地服务。当前交付是可运行的 Web + 本地服务版本；Tauri 桌面壳已有工程骨架，但还不是可双击安装的 `.exe`。完成首个功能版本后再生成 Windows 安装包，避免把未完成能力包装成成品。

## 索引状态是什么意思

文件资料库中的状态可以点击查看详情：

| 状态 | 含义 |
| --- | --- |
| 正在处理 | 正在解析、切分、生成向量或写入索引 |
| 可检索 | KUN 本地副本存在，知识片段与检索索引均可使用 |
| 处理失败 | 某一步失败，可查看原因并重试 |
| 索引缺失 | 本地副本仍在，但索引记录不完整，需要重新建立 |
| 副本缺失 | KUN 资料库内的独立副本已不存在，无法继续检索 |

AI Infra 开发副本使用 `%LOCALAPPDATA%\KUN-AI-Infra`；稳定版仍使用 `%LOCALAPPDATA%\KUN`。两者的数据库、资料副本和索引互不覆盖。

导入确认后，KUN 会把文件复制到对应数据目录的 `library`。因此，删除电脑原位置的文件不会影响检索；只有删除 KUN 保存的副本或删除资料记录，才会让它失效。

## 数据与隐私边界

- 原文件副本、索引、会话和 Memory 默认保存在本机。
- 云端 Embedding 会发送用于生成向量的文本片段。
- 云端对话模型只接收回答所需的问题、少量对话上下文、已启用 Memory 和检索片段。
- 图片首次理解会发送压缩副本至配置的视觉模型，原始文件仍保存在本机。
- 删除索引或 KUN 副本不会修改用户原位置文件。
- 第三方公开内容仍受著作权和平台条款约束；系统不绕过登录、付费、DRM 或反爬限制。

## 架构

```text
React UI / Tauri shell
        │
        ▼
FastAPI loopback API
        ├── Ingestion ── Parser / Chunker / Embedding / Image understanding
        ├── LangGraph ── Understand / Select Skill / Retrieve / Answer / Memory
        ├── Tool registry ── Schema / Permission / Availability / Trace
        ├── Hybrid RAG ── SQLite FTS5(BM25) + dense vectors + fusion
        ├── AI Infra ── Persisted jobs / Trace spans / FAISS generations
        ├── Eval runner ── Gold datasets / experiments / regression gates
        └── Local storage ── SQLite / copied source files / index metadata
```

更完整的工程说明见 `docs/ARCHITECTURE.md`，AI Infra 操作说明见 `docs/AI_INFRA.md`，Tool 契约见 `docs/TOOLS.md`。

## 开发运行

环境要求：

- Windows 10/11
- Node.js 22.13+
- Python 3.11+

安装前端依赖：

```powershell
npm install
```

安装后端依赖：

```powershell
python -m venv backend\.venv
.\backend\.venv\Scripts\Activate.ps1
python -m pip install -r backend\requirements.txt
```

同时启动前端和后端：

```powershell
npm run dev:all
```

也可以分别运行：

```powershell
uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8765
npm run dev
```

运行前后端验证：

```powershell
npm test
```

## 评估原则

项目不会用几个演示问题夸大准确率。RAG 评估应记录数据集版本、模型版本、Chunk 策略、Recall@5/10、MRR、nDCG@10、引用定位成功率、延迟和费用。正式发布指标前，需要建立经过人工标注的评估集。

质量实验仅统计 `accepted` 的人工 Gold；DeepSeek 生成的问题默认是 `draft`。确定性生成向量只用于基础设施性能压测，不能用于准确率宣传。实验报告可通过 `/api/eval/experiments/{run_id}/report` 导出 Markdown 或 JSON。

## 后续里程碑

- 完成 100 条人工 Gold，并在实际 API 配置下发布第一份可复现实验报告。
- Excel 表格结构、合并单元格和单元格级引用。
- PaddleOCR 本地模式与视觉模型模式切换。
- 视频转写、关键帧/OCR、时间戳引用。
- 配置正式联网搜索提供方，并将网页搜索证据接入带来源回答。
- Tauri 系统托盘、全局快捷键、文件夹监控与 Windows 安装包。

## Portfolio evidence workflow

KUN's RAG lab stores each evaluation run with its dataset size, Top K, retrieval metrics, latency, and per-question results. Export a reproducible Markdown report after running an evaluation:

```powershell
python backend/scripts/export_evaluation_report.py --format markdown --output docs/evaluation/latest-report.md
```

The exporter also records the embedding/chat model names, machine profile, citation-location success rate, and bad cases. Treat the generated values as measured evidence only: do not describe them as production accuracy, and do not publish private source files or API keys.

See `docs/portfolio/AGENT_CASE_STUDY.md` for the interview narrative and `docs/portfolio/DEMO_SCRIPT.md` for the three-minute walkthrough.

## License

本项目目前未授予开源许可证。除非仓库后续明确添加许可证，否则保留全部权利。
