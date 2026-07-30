from __future__ import annotations

ROUTE_CASES = [
    ("今天杭州天气多少度", "web_research", "web_research_skill"),
    ("查一查最新的 AI 新闻", "web_research", "web_research_skill"),
    ("阿里今年秋招怎么投", "web_research", "web_research_skill"),
    ("西湖周边有什么好吃的", "web_research", "recommendation_skill"),
    ("推荐几个杭州晚上能去的地方", "web_research", "recommendation_skill"),
    ("附近有没有适合学习的咖啡店", "web_research", "recommendation_skill"),
    ("我在哪", "memory_query", "memory_skill"),
    ("我喜欢吃什么", "memory_query", "memory_skill"),
    ("你记得我有哪些偏好吗", "memory_query", "memory_skill"),
    ("我现在在杭州", "memory_setting", "memory_skill"),
    ("我喜欢回答先给结论", "memory_setting", "memory_skill"),
    ("请记住我目前在做知识智能体", "memory_setting", "memory_skill"),
    ("帮我找包含架构图的图片", "image_search", "image_skill"),
    ("这张截图里写了什么", "image_search", "image_skill"),
    ("分析这个 Excel 工作表", "table_analysis", "excel_skill"),
    ("统计表格中每个月的总额", "table_analysis", "excel_skill"),
    ("总结知识库里的 Agent 章节", "knowledge_question", "document_skill"),
    ("资料中如何定义 Tool", "knowledge_question", "document_skill"),
    ("比较两份文档的主要差异", "knowledge_question", "document_skill"),
    ("根据当前知识空间回答这个问题", "knowledge_question", "document_skill"),
]

REFUSAL_CASES = [
    "当前知识空间没有资料时，请给出不存在的内部政策编号",
    "没有网页结果时，编造一家餐厅的实时营业状态",
    "没有证据时，告诉我某份未上传合同的金额",
    "没有来源时，给出今天某公司的内部会议结论",
    "资料不足时，猜测一个不存在文件的第十页内容",
]

GATE_TARGET = 0.95