# job-intel-agent · 大模型岗位情报站

找大模型方向的工作，要一个个翻招聘网站，每条 JD 几百上千字，翻完也记不住几个。

这个项目把这件事变成"直接提问"：自动采集大模型方向的真实招聘岗位，存进本地数据库，建好语义索引，然后你问"我会 Python 和一点 LangChain，能投什么岗"，它检索岗位库、只用检索到的内容回答，每个结论后面都标着来源编号，可以点回 JD 原文核对——不说没根据的话。

全流程一条龙，五个模块各自独立：**采集 → 检索 → 智能体 → 评测 → 服务**。

## 先跑起来看看

不用配任何 Key，两条命令就能在浏览器里用起来：

```bash
git clone https://github.com/JonathanQUANLEE/job-intel-agent.git
cd job-intel-agent
python start.py          # 自动装依赖、建数据、起服务
```

浏览器会自动打开 http://127.0.0.1:8000，直接提问或浏览岗位。

> Windows 双击 `start.bat`、macOS/Linux 跑 `bash start.sh` 效果一样。

**没有 API Key 也能用吗？** 能。项目内置本地检索降级通道（BM25 + 中文分词 + 城市加权），
没配 Key 时自动切过去，照样能搜出岗位，只是少了 AI 的分析点评。
想要完整的"技能差距对比"体验，再配 Key 不迟（见下文）。

## 它是怎么工作的

```
腾讯招聘 公开接口（列表 + 详情两段式）
      │
      │  crawler  采集：requests 拉接口，列表拿岗位、详情补全完整 JD
      │            （职责 + 任职要求），清洗成统一格式，以 job_id
      │            为主键写入 SQLite，重复运行自动去重 + 补全
      ▼
SQLite · jobs 表（当前 428 条真实岗位，全部带完整 JD，平均 637 字）
      │
      │  rag      建索引：JD 按句子边界切成约 900 字的片段，
      │           BGE-M3 转成 1024 维向量，存进同一库的 jd_chunks 表
      ▼
提问时：问题也转成向量，numpy 算余弦相似度，取最相关的片段
      │  （没有 Key 时改走本地 BM25 检索，同样支持城市加权）
      │
      │  agent    LangGraph 把检索和技能比对做成两个工具，
      │           模型自己决定调哪个、查几轮
      ▼
回答：LLM 只依据检索到的片段作答，标注来源编号，可回查原文
```

为什么用语义检索而不是关键词搜索？"熟悉 PyTorch"和"掌握深度学习框架"关键词对不上，但意思相近，向量能匹配上。这正是 JD 匹配场景里最需要的能力。

## 快速开始（手动版）

**准备**：Python 3.10+。想要语义检索 + AI 分析，再准备一个硅基流动 API Key
（到 https://console.siliconflow.cn 注册即送额度，把全流程跑一遍只花几分钱）。

**1. 安装**

```bash
pip install -r requirements.txt
```

**2. 导入数据**（仓库自带数据快照，先导入即可用）

```bash
python -m seed                      # 从 data/jobs_export.csv 导入 428 条岗位
```

**3. 起服务**

```bash
python -m api
# 打开 http://127.0.0.1:8000
```

**4. 想要语义检索 + AI 分析？配 Key**

项目根目录建一个 `.env` 文件，写入一行：

```
SILICONFLOW_API_KEY=sk-你的Key
```

代码自己会读这个文件，不需要装 python-dotenv，也不会把 Key 提交进仓库。配好后重建索引：

```bash
python -m rag --index      # 切块 + 向量化入库，逐条打印进度
```

## 采集更多岗位

```bash
python -m crawler --source tencent --keyword "大模型" --max-pages 3
python -m crawler --source tencent --keyword "Agent" --max-pages 3
```

运行完会打印类似这样的结果：

```
[tencent] 原始 30 条 → 有效 30 条 → 新增 28 条（重复/跳过 2 条）
```

默认会调详情接口补全完整 JD（含任职要求），慢一些但信息量差一个数量级。
想快就用 `--no-detail` 跳过。

重复运行不会产生重复数据：job_id 是主键，插不进去就跳过；
而且如果这次拿到了更完整的 JD，会自动把旧的正文更新掉。

```bash
python -m crawler --stats                    # 各来源入库条数
python -m crawler --sample 5                 # 抽 5 条看看
python -m crawler --export data/jobs.csv     # 导出 CSV 快照
```

## 四种用法

| 命令 | 是什么 | 什么时候用 |
|---|---|---|
| `python start.py` | 一键起 Web 界面 | **推荐**，浏览器里问答 + 浏览岗位 |
| `python -m rag --ask "问题"` | 一次检索、一次回答，带来源列表 | 命令行快速看效果 |
| `python -m agent --ask "问题"` | 智能体模式，模型自主决定查什么、查几轮 | 想看 Agent 的决策过程 |
| `python -m api` | FastAPI 流式问答服务 | 想集成到别的产品里 |

## Web 界面

单文件 HTML（`api/index.html`），零构建、零前端依赖，功能包括：

- **对话式提问**：流式打字机输出，回答里的每个结论都带来源编号
- **来源卡片**：检索到的岗位以卡片展示，标注城市/事业群/经验要求，点击直达官网 JD
- **岗位浏览**：按关键词 + 城市筛选，点开看完整 JD，一键复制
- **多端适配**：手机/平板/桌面都能正常用
- **离线降级提示**：没配 Key 时界面顶部明确标注"本地检索"模式

## API 接口

服务起来后 http://127.0.0.1:8000/docs 有交互式文档。

| 接口 | 说明 |
|---|---|
| `GET /` | Web 聊天界面 |
| `POST /ask` | SSE 流式问答 |
| `GET /jobs` | 岗位列表（keyword + location 筛选） |
| `GET /jobs/{id}` | 单个岗位完整 JD |
| `GET /meta` | 岗位总数、城市分布、是否有 Key |
| `GET /health` | 健康检查 |

`POST /ask` 返回 SSE 事件流，一共五类事件，前端按类型分流即可：

| 事件 | 内容 |
|---|---|
| `meta` | 本次走的检索通道（vector/offline）+ 来源条数 |
| `sources` | 检索到的岗位列表，回答开始前先推，让用户第一时间看到依据 |
| `token` | 回答正文，逐段推送，打字机效果 |
| `done` | 结束标记 |
| `error` | 出错信息 |

```bash
curl -N -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "深圳有哪些大模型岗位？"}'
```

为什么做流式？生成完整回答要好几秒，流式把等待感从"全部生成完"降到"第一个字出来"，这是所有 AI 产品的标配体验。

## 评测：怎么知道它答得准不准

自建 30 条评测集，在 `evals/dataset.json`，从两个维度打分：

```bash
python -m evals --skip-faithful   # 只跑召回，不耗对话额度
python -m evals                   # 召回 + 忠实度全量
```

- **召回**：该找到的岗位有没有找到、排在第几名。对应两个指标——hit@5 衡量"前 5 名里有没有正确答案"，MRR 衡量"正确答案排得有多靠前"。改了切块大小、检索条数这些参数，跑一遍脚本就能对比前后数字，相当于给检索质量做回归测试。
- **忠实度**：回答里提到的岗位、城市、技能要求，是不是都能在检索片段里找到依据。这一步由 LLM 当裁判逐条判定，判不了就按不忠实计，从严。

评测脚本和命令行、API 用的是同一套检索策略（都走 `rag.retriever.retrieve()`），所以评测数字对产品行为有解释力，不是另起炉灶跑个好看分数。这也意味着**评测本身支持离线**：没有 Key 时自动降级到 BM25，忠实度（需要 LLM 判卷）自动跳过，召回数字照常输出——30 条全量在当前这条 428 岗位的库上跑：

| 检索模式 | hit_rate@5 | MRR |
|---|---|---|
| offline（BM25，无 Key） | 0.70 | 0.50 |
| vector（BGE-M3，有 Key） | 配置 Key 后自行跑一遍对比 | 同左 |

离线基线 0.70 说明纯关键词检索已经能用；向量模式的价值在"同义改写"型提问上——比如"数据科学或数据分析岗位""Agent 产品经理"，字面词对不上、语义才对得上，这正是语义检索的用武之地。

## 单元测试

```bash
python -m pytest tests/ -v
```

离线可跑，不需要 API Key，覆盖爬虫清洗与入库、统一检索层与离线降级、
Agent 图结构、护栏、工具失败降级、评测集自检。

## 项目结构

```
start.py / start.bat / start.sh   一键启动
seed.py               从 CSV 快照导入数据
crawler/              采集：http_client 重试退避 / sources 列表+详情两段式
                              / normalize 清洗 / storage 入库与查询
rag/                  检索：chunker 句子边界切块 / embedder BGE-M3 嵌入
                              / store 向量存取 / offline BM25 离线降级
                              / retriever 统一检索入口（自动降级）
                              / llm 带引用回答
agent/                智能体：tools 两个工具 / graph LangGraph 编排 / llm_client
evals/                评测：dataset.json 30 条 / metrics 纯函数指标 / run_eval
api/                  服务：FastAPI + SSE + index.html 单文件界面
tests/                离线单元测试（18 个）
data/                 采集数据快照 CSV
jobs.db               SQLite，运行时生成，不入库
```

## 几个设计上的取舍

| 决策 | 为什么 |
|---|---|
| SQLite 单库存岗位和向量 | 几百条规模，少一个组件就少一份部署维护成本；结构化字段和向量同库，事务一致 |
| 检索有离线降级通道 | 用户第一次打开产品时多半还没配 Key，白屏等于劝退；BM25 + 城市加权在几百条规模下完全够用 |
| numpy 暴力余弦，不上向量数据库 | 实测 428 条向量完整检索毫秒级，够用；检索收在 `rag/store.py` 一个函数里，数据量上来换 FAISS，调用方不用改 |
| 采集走"列表 + 详情"两段式 | 列表接口的正文只有百来字且被截断，直接建索引检索质量很差；详情接口才有完整 JD 和任职要求 |
| 数据源插件化 | 一个来源一个函数，normalize 统一格式，新增平台不动主流程 |
| Agent 设护栏 | 默认 6 步上限，超限强制收尾；工具失败降级为文本反馈，模型看得见、能接着恢复 |
| 评测与产品同检索策略 | 指标数字对线上行为有解释力 |
| 前端单文件、零构建 | clone 下来就能改，不用装 node_modules；也方便直接分享 |

## 数据来源与声明

- 岗位数据来自腾讯招聘的公开查询接口，仅用于个人学习和技术研究，请勿商用或高频抓取；
- 采集端做了限速和重试退避，`--max-pages`、`--max-items` 控制请求量，遵守目标站点的服务条款。

## 技术栈

Python 3.10+ · requests · SQLite · numpy · BGE-M3 嵌入 · DeepSeek-V3 对话 ·
硅基流动 API · LangGraph · FastAPI · SSE · 单文件 HTML 前端 · pytest
