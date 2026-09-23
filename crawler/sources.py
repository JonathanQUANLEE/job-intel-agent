"""数据源模块：每个函数负责"从一个来源抓回原始记录列表"。

约定：返回 list[dict]，dict 内部字段保持来源原样（raw），
     统一 schema 交给 normalize.py 做 —— 采集和清洗分离，各改各的互不影响。
"""

import time

from .http_client import get_json

TENCENT_QUERY_URL = "https://careers.tencent.com/tencentcareer/api/post/Query"
HN_API = "https://hacker-news.firebaseio.com/v0"


def fetch_tencent_jobs(session, keyword: str, max_pages: int = 2) -> list[dict]:
    """腾讯招聘公开接口（无需登录）。

    分页三要素：pageIndex 从 1 开始、pageSize 控制每页数量、
    空页或不足一页说明到底了，提前 break。
    """
    results: list[dict] = []
    for page in range(1, max_pages + 1):
        payload = get_json(session, TENCENT_QUERY_URL, params={
            "timestamp": 0,
            "keyword": keyword,
            "pageIndex": page,
            "pageSize": 10,
            "language": "zh-cn",
        })
        data = (payload or {}).get("Data") or {}
        posts = data.get("Posts") or []
        if not posts:
            break  # 没有更多岗位了
        results.extend(posts)
        time.sleep(1.5)  # 限速：对服务器友好，也是反爬最基本的自我修养
    return results


def fetch_hackernews_items(session, max_items: int = 20) -> list[dict]:
    """Hacker News 热帖（免费、无 Key、纯 JSON）—— 用来练手的沙盒数据源。

    两段式请求：先拿 id 列表，再逐个拿详情。很多真实 API 都是这个模式。
    """
    top_ids = get_json(session, f"{HN_API}/topstories.json")[:max_items]
    items = []
    for item_id in top_ids:
        items.append(get_json(session, f"{HN_API}/item/{item_id}.json"))
        time.sleep(0.3)  # 逐条请求，节奏放慢
    return items
