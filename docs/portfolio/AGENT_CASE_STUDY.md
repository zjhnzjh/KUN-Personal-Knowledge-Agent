# KUN：企业级 Agent 应用工程案例

> 面试和简历使用边界：KUN 是 Windows-first、single-user、local-first 的个人知识智能体。它用于展示 Agent 应用工程能力，不应表述为已经完成企业级多租户生产部署。

## 30 秒版本

KUN 解决的是个人 PDF、Word、Excel 和图片资料分散、检索不可追溯的问题。我用 FastAPI、LangGraph 和 SQLite 搭建了从资料导入、解析切分、混合检索、引用回答到 Memory 和 Tool 追踪的完整闭环。重点不是把 LLM 接上，而是把检索、工具权限、敏感信息保护、失败状态和评测做成可观察的工程链路。

## 面试叙事

### 1. 问题

本地资料长期分散在不同文件夹，用户既难以快速找到资料，也无法判断回答来自哪一页。系统需要兼顾本地优先、可追溯回答和可量化检索效果。

### 2. 架构

```text
React / Tauri shell
        ↓
FastAPI loopback API
        ↓
LangGraph: understand → skill/tool selection → retrieve → answer → memory proposal
        ↓
SQLite FTS5/BM25 + dense retrieval + RRF
        ↓
SQLite metadata, copied source files, evaluation runs, tool traces
```

### 3. 工程亮点

- 用文档页码、标题和 Chunk ID 保留引用定位，回答只展示实际使用的证据。
- Tool 先经过 Schema、权限范围、网络范围、超时和确认策略校验，再执行并写入脱敏 Trace。
- Memory 分为当前对话短期 Memory 和用户确认后的长期 Memory；敏感信息不会自动写入。
- RAG 实验室使用人工标注问题运行 Recall@K、MRR、nDCG、平均延迟、P95 和 citation-location success，并保留逐题 Bad Case。
- 模型或 Embedding 不可用时显示明确的降级状态，不把失败伪装成成功。

## 可验证结果

运行评测后，只把报告中的真实数字写入简历或面试材料：

```text
数据集：________ 题
Recall@5：________
MRR：________
nDCG@5：________
引用定位成功率：________
平均检索延迟：________ ms
P95 检索延迟：________ ms
```

不要用一次小规模评测推导“生产准确率”。面试时必须同时说明数据集版本、模型版本、机器环境、Chunk 策略和至少一个 Bad Case。

## 校招简历候选表述

> 下面的数字只允许替换为已运行、可解释的真实结果。

- 独立搭建 Windows 本地优先个人知识 Agent，完成 PDF/Word/Excel/图片导入、解析切分、混合检索、带页码引用问答、Memory 和 Tool Trace 的端到端闭环。
- 基于 LangGraph 编排意图识别、Skill/Tool 路由、检索、回答和 Memory 建议；Tool 执行前统一校验参数、权限、联网范围、超时与确认要求。
- 使用 SQLite FTS5/BM25 与稠密检索进行 RRF 融合，建设人工标注评测集，以 Recall@K、MRR、nDCG、P95 和引用定位成功率追踪检索质量，并保留逐题 Bad Case。

## 明确边界

- 不说“企业级生产部署”，说“企业知识检索/流程自动化方向的 Agent 工程原型”。
- 不说“有真实用户”，除非能给出用户范围、使用周期和可验证数据。
- 不说“提高准确率 X%”，除非能说明对照组、数据集和评测脚本。
