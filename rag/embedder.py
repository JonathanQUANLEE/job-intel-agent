"""嵌入客户端：调硅基流动 /embeddings 接口，把文本变成向量。

向量 = 一串浮点数（BGE-M3 输出 1024 维）。
核心性质："语义相似的两段话 → 向量夹角小"。
正是这个性质，让后面的余弦检索能"按意思"而不是"按字面"找内容。

接口是 OpenAI 兼容格式（文档：docs.siliconflow.com → Create embeddings）。
"""

import requests

from .config import API_KEY, BASE_URL, EMBED_MODEL

_BATCH = 16  # 每请求最多嵌入条数（接口有 token 上限，超出会被拒）


def embed_texts(texts: list[str]) -> list[list[float]]:
    """批量嵌入，返回与 texts 同顺序的向量列表。

    注意按 index 排回原序——接口文档明说不保证返回顺序。
    """
    if not API_KEY:
        raise RuntimeError(
            "没有找到 SILICONFLOW_API_KEY。请到 console.siliconflow.cn 注册拿 Key，"
            "然后写进项目根目录的 .env 文件（格式：SILICONFLOW_API_KEY=sk-xxx）"
        )
    if not texts:
        return []

    vectors: list[list[float]] = []
    for start in range(0, len(texts), _BATCH):
        batch = texts[start : start + _BATCH]
        resp = requests.post(
            f"{BASE_URL}/embeddings",
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
            },
            json={"model": EMBED_MODEL, "input": batch},
            timeout=(5, 60),  # 批量嵌入可能较慢，读取超时给足
        )
        resp.raise_for_status()
        data = resp.json()["data"]
        data.sort(key=lambda d: d["index"])  # 按 index 排回原序
        vectors.extend(d["embedding"] for d in data)
    return vectors
