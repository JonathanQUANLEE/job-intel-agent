"""离线检索：没有 API Key 时的本地兜底方案（零依赖、纯 Python）。

为什么需要它？
RAG 的主路径依赖嵌入向量，而向量要调云端接口 —— 没 Key 或断网就彻底趴窝。
用户第一次打开产品时最可能就是这个状态："我看一眼，但我还没配 Key"。
这时候如果直接报错说"未配置 Key 无法使用"，产品等于劝退了 90% 的人。

本模块用经典的 BM25（在 jieba 不可用时退化为字符 n-gram 分词）做本地检索：
  - 中文无空格，用「单字 + 双字组合」切分，召回够用且零依赖；
  - BM25 比裸 TF 多考虑了"词有多罕见"（IDF）和"文档长度归一"，
    比朴素关键词命中更准，仍然是几行数学。

它的定位是**降级通道**，不追求超过向量检索 ——
只是在"没法用向量"时，别让产品变成一块白板。
"""

import math
import re
import sqlite3
from collections import Counter
from contextlib import closing

# 中文停用词（高频但无区分度，参与打分只会加噪声）
_STOPWORDS = {
    "的", "了", "和", "是", "在", "有", "与", "及", "或", "等", "对", "为", "我", "你",
    "什么", "哪些", "怎么", "如何", "可以", "能", "会", "要", "想", "找", "请", "一个",
    "the", "a", "an", "of", "and", "or", "to", "is", "are", "for", "in", "on", "with",
}


# 常见城市名，用于从查询里识别地点意图（离线检索没法学语义，只能靠词表）
_CITIES = [
    "北京", "上海", "深圳", "广州", "杭州", "成都", "重庆", "武汉", "南京", "西安",
    "苏州", "天津", "长沙", "合肥", "郑州", "厦门", "青岛", "大连", "福州", "济南",
    "新加坡", "伦敦", "首尔", "东京", "巴黎", "纽约", "西雅图", "中国香港", "中国台湾",
]

# 表达"地点"的常见疑问词，用来定位城市名在句子里的位置（"深圳有哪些..."）
_CITY_HINT_WORDS = ("哪些", "有什么", "有没", "岗位", "工作", "机会", "职位")


def extract_city(query: str) -> str:
    """从问题里抽出城市名。抽不到返回空串。

    离线检索靠词表匹配，比不了向量语义，但"深圳有哪些岗位"这类问题
    只要认出"深圳"，就能把地点约束加进检索 —— 这是性价比最高的一步。
    """
    for city in _CITIES:
        if city in query:
            return city
    return ""


def tokenize(text: str) -> list[str]:
    """中文友好的分词：抽英文/数字整词 + 中文单字与双字组合。

    为什么不全上 jieba？—— 它是额外依赖，而 BM25 只是降级通道。
    双字组合（bigram）已经能抓住"大模型""算法""推理"这类关键概念，
    加上英文单词整取（Python / PyTorch / RAG），召回质量对本项目够用。
    """
    text = text.lower()
    tokens: list[str] = []

    # 英文与数字：连续一段算一个词（保留 PyTorch / pytorch 大小写归一后的词）
    for word in re.findall(r"[a-z0-9_+#\.]+", text):
        if word not in _STOPWORDS and len(word) > 1:
            tokens.append(word)

    # 中文：连续汉字串，生成单字 + 相邻双字
    for seq in re.findall(r"[\u4e00-\u9fff]+", text):
        for ch in seq:
            if ch not in _STOPWORDS:
                tokens.append(ch)
        for i in range(len(seq) - 1):
            bigram = seq[i : i + 2]
            if bigram not in _STOPWORDS:
                tokens.append(bigram)

    return tokens


def _load_docs(db_path: str) -> list[tuple[str, str, str]]:
    """取所有岗位的 (job_id, location, 全文)。全文 = 标题 + 地点 + 正文。

    标题和地点拼进全文很重要：用户问"深圳"，只在正文里找可能漏掉——
    地点字段本身也是强信号。location 单独返回，用于城市加权。
    """
    with closing(sqlite3.connect(db_path)) as conn:
        rows = conn.execute(
            "SELECT job_id, COALESCE(location,''), "
            "COALESCE(title,'') || ' ' || COALESCE(location,'') "
            "|| ' ' || COALESCE(company,'') || ' ' || COALESCE(raw_text,'') FROM jobs "
            "WHERE raw_text != ''"
        ).fetchall()
    return rows


class BM25Index:
    """内存 BM25 索引。每次查询现建（几百条文档毫秒级，不值得做持久化缓存）。"""

    K1 = 1.5   # 词频饱和系数：一个词出现 10 次不该比出现 3 次有用 3 倍
    B = 0.75   # 长度归一强度：长文档天然多命中，要压一压

    # 城市加权系数：问题里点名了城市，该城市岗位分数乘这个倍数。
    # 取 2.5 是经验值——足够把"北京的同名岗位"压到"深圳的目标岗位"之后，
    # 又不至于让"深圳的无关岗位"盖过"深圳的高相关岗位"。
    CITY_BOOST = 2.5

    def __init__(self, docs: list[tuple[str, str, str]]):
        self.job_ids = [d[0] for d in docs]
        self.locations = [d[1] for d in docs]
        self.term_freqs: list[Counter] = []
        doc_freq: Counter = Counter()
        self.lengths: list[int] = []

        for _, _, text in docs:
            tf = Counter(tokenize(text))
            self.term_freqs.append(tf)
            self.lengths.append(sum(tf.values()))
            doc_freq.update(tf.keys())

        n = len(docs) or 1
        self.avg_len = (sum(self.lengths) / n) or 1
        # IDF：出现得越普遍的词权重越低（"的"这种几乎无区分度）
        self.idf = {
            term: math.log(1 + (n - freq + 0.5) / (freq + 0.5))
            for term, freq in doc_freq.items()
        }

    def search(self, query: str, top_k: int = 5) -> list[tuple[float, str]]:
        """返回 [(归一化分数, job_id), ...] 降序。分数归一化到 0~1，便于和向量分统一展示。"""
        q_terms = tokenize(query)
        if not q_terms:
            return []

        want_city = extract_city(query)   # 问题里点名的城市（可能为空）

        scores: list[tuple[float, str]] = []
        for idx, tf in enumerate(self.term_freqs):
            score = 0.0
            dl = self.lengths[idx] or 1
            for term in q_terms:
                f = tf.get(term, 0)
                if not f:
                    continue
                idf = self.idf.get(term, 0.0)
                score += idf * (f * (self.K1 + 1)) / (
                    f + self.K1 * (1 - self.B + self.B * dl / self.avg_len)
                )
            if score <= 0:
                continue
            # 城市加权：问题点名了城市，该城市岗位提权
            if want_city and want_city in self.locations[idx]:
                score *= self.CITY_BOOST
            scores.append((score, self.job_ids[idx]))

        scores.sort(reverse=True)
        if not scores:
            return []
        top = scores[0][0] or 1.0
        return [(round(s / top, 4), jid) for s, jid in scores[:top_k]]


def search_offline(query: str, db_path: str = "jobs.db", top_k: int = 5) -> list[tuple[float, str, str]]:
    """离线检索入口。返回 [(分数, job_id, 命中片段), ...]，与 rag.store.search 同形。

    片段取正文前 600 字（够模型/前端展示要点，也不必把整篇 JD 塞回去）。
    接口形状和向量检索保持一致，上层切换通道时无需改代码。
    """
    docs = _load_docs(db_path)
    if not docs:
        return []
    index = BM25Index(docs)
    hits = index.search(query, top_k=top_k)

    texts = {d[0]: d[2] for d in docs}
    return [(score, job_id, texts[job_id][:600]) for score, job_id in hits]
