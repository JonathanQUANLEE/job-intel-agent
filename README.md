# job-intel-agent · 大模型岗位情报站

找大模型方向的工作，要一个个翻招聘网站，每条 JD 几百上千字，翻完也记不住几个。

这个项目把这件事变成"直接提问"：自动采集大模型方向的真实招聘岗位，存进本地数据库，建好语义索引，然后你问"我会 Python 和一点 LangChain，能投什么岗"，它检索岗位库、只用检索到的内容回答，每个结论后面都标着来源编号，可以点回 JD 原文核对——不说没根据的话。

全流程一条龙，五个模块各自独立：**采集 → 检索 → 智能体 → 评测 → 服务**。

## 它是怎么工作的

```
腾讯招聘 / Hacker News 公开接口
      │
      │  crawler  采集：requests 拉接口，清洗成统一格式，
      │            以 job_id 为主键写入 SQLite，重复运行自动去重
      ▼
SQLite · jobs 表（当前 88 条真实岗位，其中 83 条带完整 JD）
      │
      │  rag      建索引：JD 按句子边界切成约 900 字的片段，
      │           BGE-M3 转成 1024 维向量，存进同一库的 jd_chunks 表
      ▼
提问时：问题也转成向量，numpy 算余弦相似度，取最相关的片段
      │
      │  agent    LangGraph 把检索和技能比对做成两个工具，
      │           模型自己决定调哪个、查几轮
      ▼
回答：LLM 只依据检索到的片段作答，标注来源编号，可回查原文
```

为什么用语义检索而不是关键词搜索？"熟悉 PyTorch"和"掌握深度学习框架"关键词对不上，但意思相近，向量能匹配上。这正是 JD 匹配场景里最需要的能力。

## 三种用法

| 命令 | 是什么 | 什么时候用 |
|---|---|---|
| `python -m rag --ask "问题"` | 一次检索、一次回答，带来源列表 | 最快看到效果 |
| `python -m agent --ask "问题"` | 智能体模式，模型自主决定查什么、查几轮 | 想看 Agent 的决策过程 |
| `python -m api` | FastAPI 流式问答服务 | 想做成产品或给别人调用 |

## 快速开始

**准备**：Python 3.10+，一个硅基流动的 API Key。到 https://console.siliconflow.cn 注册即送额度，把全流程跑一遍只花几分钱。模型用的是 DeepSeek-V3 做对话、BGE-M3 做嵌入，走 OpenAI 兼容接口。

**1. 安装**

```bash
git clone https://github.com/JonathanQUANLEE/job-intel-agent.git
cd job-intel-agent
pip install -r requirements.txt
```

**2. 配置 Key**

项目根目录建一个 `.env` 文件，写入一行：

```
SILICONFLOW_API_KEY=sk-你的Key
```

代码自己会读这个文件，不需要装 python-dotenv，也不会把 Key 提交进仓库。

**3. 采集岗位**

```bash
python -m crawler --source tencent --keyword "大模型" --max-pages 2
python -m crawler --source tencent --keyword "Agent" --max-pages 2
```

运行完会打印类似这样的结果：

```
[tencent] 原始 20 条 → 有效 20 条 → 新增 18 条（重复/跳过 2 条）
```

换关键词多跑几遍就多攒几批岗位。重复运行不会产生重复数据：job_id 是主键，插不进去就跳过，所以这个脚本跑多少次库都是干净的。看看攒了多少：

```bash
python -m crawler --stats     # 各来源入库条数
python -m crawler --sample 5  # 抽 5 条看看标题和地点
```

Hacker News 是另一个数据源（`--source hackernews`），拉的是顶帖，无需任何 Key，主要用来练手和验证"新增一个数据源只要加一个函数"的插件化设计。真正的岗位数据以腾讯招聘为主。

仓库里的 `data/jobs_export.csv` 是一份采集结果快照，88 条岗位，GitHub 网页上可以直接点开看；要问答还需要自己跑上面的采集命令。

**4. 建索引**

```bash
python -m rag --index
```

把库里每条完整 JD 切块、向量化、入库，逐条打印进度：

```
[1/83] tencent-10001632    3 块
[2/83] tencent-10001645    2 块
...
完成：83 个岗位 / 246 个片段。下一步：python -m rag --ask "..."
```

增量执行：已经建过索引的岗位自动跳过，重复跑不烧嵌入额度。想全量重建就加 `--force`。

**5. 开始提问**

```bash
python -m rag --ask "深圳有哪些大模型岗位？"
```

先打印回答，末尾列出本次检索命中的岗位和链接：

```
========================================
=== 检索到的来源（RAG 可追溯性）===
[1] Senior Researcher, Natural Language Processing（新加坡）https://...
[2] ...
```

想看智能体怎么工作，换 agent 模式：

```bash
python -m agent --ask "我会 Python 和一点 LangChain，适合投什么岗位？"
```

终端会完整打印模型的每一步决策——调了哪个工具、工具返回了什么、然后继续思考，最后才给出回答，并告诉你这一轮一共调用了几次工具。rag 模式是"搜一次答一次"，agent 模式是模型自己拿主意，两种都值得跑一遍感受差别。

**6. 起一个流式问答服务**

```bash
python -m api
```

打开 http://127.0.0.1:8000/docs 有现成的调试页面，或者直接 curl：

```bash
curl -N -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "深圳有哪些大模型岗位？"}'
```

返回是 SSE 事件流，一共四类事件，前端按类型分流即可：

| 事件 | 内容 |
|---|---|
| `sources` | 检索到的岗位列表，回答开始前先推，让用户第一时间看到依据 |
| `token` | 回答正文，逐段推送，打字机效果 |
| `done` | 结束标记 |
| `error` | 出错信息 |

为什么做流式？生成完整回答要好几秒，流式把等待感从"全部生成完"降到"第一个字出来"，这是所有 AI 产品的标配体验。

**7. 评测**

```bash
python -m evals --skip-faithful   # 只跑召回，不耗对话额度
python -m evals                   # 召回 + 忠实度全量
```

**8. 单元测试**

```bash
python -m pytest tests/ -v
```

离线可跑，不需要 API Key，覆盖爬虫清洗、Agent 图结构、护栏、工具降级和评测集自检。

## 评测：怎么知道它答得准不准

自建 30 条评测集，在 `evals/dataset.json`，从两个维度打分：

- **召回**：该找到的岗位有没有找到、排在第几名。对应两个指标——hit@5 衡量"前 5 名里有没有正确答案"，MRR 衡量"正确答案排得有多靠前"。改了切块大小、检索条数这些参数，跑一遍脚本就能对比前后数字，相当于给检索质量做回归测试。
- **忠实度**：回答里提到的岗位、城市、技能要求，是不是都能在检索片段里找到依据。这一步由 LLM 当裁判逐条判定，判不了就按不忠实计，从从严。

评测脚本和命令行、API 用的是同一套检索策略，所以评测数字对产品行为有解释力，不是另起炉灶跑个好看分数。

## 项目结构

```
crawler/     采集：http_client 请求重试退避 / sources 数据源插件 / normalize 清洗 / storage 入库
rag/         检索：chunker 句子边界切块 / embedder BGE-M3 嵌入 / store 向量存取与检索 / llm 带引用回答
agent/       智能体：tools 两个工具 / graph LangGraph 编排 / llm_client
evals/       评测：dataset.json 30 条 / metrics 纯函数指标 / run_eval 评测脚本
api/         服务：FastAPI + SSE
tests/       离线单元测试
data/        采集数据快照 CSV
jobs.db      SQLite，运行时生成，不入库
```

## 几个设计上的取舍

| 决策 | 为什么 |
|---|---|
| SQLite 单库存岗位和向量 | 几百条规模，少一个组件就少一份部署维护成本；结构化字段和向量同库，事务一致 |
| numpy 暴力余弦，不上向量数据库 | 实测 300 条向量完整检索平均 3.4 毫秒，够用；检索收在 `rag/store.py` 一个函数里，数据量上来换 FAISS，调用方不用改 |
| 数据源插件化 | 一个来源一个函数，normalize 统一格式，新增平台不动主流程 |
| Agent 设护栏 | 默认 6 步上限，超限强制收尾；工具失败降级为文本反馈，模型看得见、能接着恢复 |
| 评测与产品同检索策略 | 指标数字对线上行为有解释力 |

## 数据来源与声明

- 岗位数据来自腾讯招聘的公开查询接口和 Hacker News 的公开 API，仅用于个人学习和技术研究，请勿商用或高频抓取；
- 采集端做了限速和重试退避，`--max-pages`、`--max-items` 控制请求量，遵守目标站点的服务条款。

## 技术栈

Python 3.10+ · requests · SQLite · numpy · BGE-M3 嵌入 · DeepSeek-V3 对话 · 硅基流动 API · LangGraph · FastAPI · SSE · pytest
