# Agent 岗位情报站（job-intel-agent）

一个从数据采集到智能问答的完整 Agent 项目：采集大模型方向的真实招聘岗位，建成可语义检索的知识库（RAG），用 LangGraph 编排的智能体回答求职问题，带评测与流式 API 服务。

> 示例：问「我会 PyTorch 和分布式训练，适合什么岗位？」→ 智能体自动检索岗位库 → 基于真实 JD 回答并标注来源编号 → 结论可回查原文。

## 架构（五层）

```
① 采集  crawler/    requests 调公开接口 → 统一 schema → SQLite（主键去重，幂等）
② 检索  rag/        JD 切分 → BGE-M3 嵌入（1024 维）→ 余弦检索
③ 智能体 agent/    LangGraph + Function Calling：模型自主决定查什么、查几轮（护栏：步数上限）
④ 评测  evals/      30 条评测集：召回 hit@5 / MRR + 忠实度（LLM 判卷）
⑤ 服务  api/        FastAPI + SSE 流式问答
```

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置 API Key（硅基流动 https://console.siliconflow.cn，注册即送额度）
#    项目根目录创建 .env 文件，写入一行：
#    SILICONFLOW_API_KEY=sk-你的Key

# 3. 采集岗位（多关键词、多页，重复运行自动去重）
python -m crawler --source tencent --keyword "大模型" --max-pages 2
python -m crawler --source tencent --keyword "Agent" --max-pages 2
python -m crawler --stats                      # 查看入库统计

# 4. 建立向量索引（增量：已建过的岗位自动跳过）
python -m rag --index

# 5. 三种用法任选
python -m rag --ask "深圳有哪些大模型岗位？"     # RAG 问答（带来源引用）
python -m agent --ask "我会Python，能投什么岗？"  # Agent：模型自主决定调工具
python -m api                                  # 流式服务 → http://127.0.0.1:8000/docs

# 6. 评测与测试
python -m evals --skip-faithful    # 召回指标（省额度模式）
python -m evals                    # 召回 + 忠实度全量评测
python -m pytest tests/ -v         # 9 项离线单元测试（无需 Key）
```

## 项目结构

```
crawler/           ① 采集：http_client(重试退避) / sources(数据源) / normalize(清洗) / storage(入库)
rag/               ② 检索：chunker(自然边界切分) / embedder(嵌入) / store(向量存储+余弦检索) / llm(带引用回答)
agent/             ③ 智能体：tools(search_jobs/match_skills) / graph(LangGraph 循环+护栏) / llm_client
evals/             ④ 评测：dataset.json(30条) / metrics(纯函数指标) / run_eval(评测脚本)
api/               ⑤ 服务：FastAPI + SSE（sources → token → done 事件流）
tests/             单元测试：爬虫清洗/Agent 图结构/护栏/工具降级/评测集自检
jobs.db            SQLite 数据库（jobs 表 + jd_chunks 向量表）
```

## 评测说明

- **召回（hit_rate@5 / MRR）**：30 条人工标注的评测问题，检验“该找到的岗位有没有找到、排第几”；
- **忠实度（faithfulness）**：LLM 判卷，检验回答里的每个事实是否都能在检索片段中找到依据，从严计分；
- `python -m evals --skip-faithful` 只跑召回（不耗对话额度），改完切分参数/检索策略后跑一遍即可回归对比。

## 关键设计决策

| 决策 | 理由 |
|---|---|
| SQLite 单库存一切（岗位 + 向量） | 数百条数据规模，少一个组件就少一份部署/维护成本；结构化字段与向量同库，事务一致 |
| numpy 暴力余弦而非向量数据库 | 实测 300 条 × 1024 维完整检索平均 3.4ms；检索逻辑封装在 `rag/store.search()` 单函数内，量级上来后函数体换 FAISS，调用方零改动 |
| 数据源插件化 | `crawler/sources.py` 每个来源一个函数 + `normalize` 统一 schema，新增平台不动主流程 |
| Agent 护栏 | `MAX_STEPS=6` 步数上限；超限走 `summarize` 节点强制收尾；工具失败降级为文本反馈，模型可见可恢复 |
| 评测与产品同检索策略 | 评测脚本与 CLI/API 用同一套“按岗位去重取 top-k”，指标数字才对产品有解释力 |

## 声明

- 数据来自腾讯招聘、Hacker News 的公开接口，仅用于个人学习与技术研究，请勿用于商业用途或高频抓取；
- 遵守目标站点服务条款，采集时设置了重试退避与礼貌 User-Agent，`--max-pages` 控制请求量。

## 技术栈

Python 3.10+ · requests · SQLite · numpy · BGE-M3（嵌入） · DeepSeek-V3（对话，硅基流动 API） · LangGraph · FastAPI · SSE · pytest
