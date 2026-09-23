"""RAG 包：把岗位库变成能检索、能问答的知识底座。

crawler 负责"把数据存进库"（燃料），rag 负责"让数据能回答问题"（引擎）。
两条主线在此汇合：后面 Agent 阶段的 search_jobs / match_skills 工具，
本质都是在调用本包的检索能力。
"""

__version__ = "0.1.0"
