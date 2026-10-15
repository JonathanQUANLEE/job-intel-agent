"""数据源模块：每个函数负责"从一个来源抓回原始记录列表"。

约定：返回 list[dict]，dict 内部字段保持来源原样（raw），
     统一 schema 交给 normalize.py 做 —— 采集和清洗分离，各改各的互不影响。

腾讯招聘使用「列表接口 + 详情接口」两段式：
  1. Query  列表接口：一次给一页岗位，字段少、正文被截断；
  2. Detail 详情接口：按 PostId 取完整 JD（职责 + 要求），信息量差一个数量级。
列表里带的正文只有百来字，直接拿去建索引检索质量很差，所以这里主动补全详情。
"""

import time

from .http_client import get_json

TENCENT_QUERY_URL = "https://careers.tencent.com/tencentcareer/api/post/Query"
TENCENT_DETAIL_URL = "https://careers.tencent.com/tencentcareer/api/post/ByPostId"
HN_API = "https://hacker-news.firebaseio.com/v0"


def fetch_tencent_detail(session, post_id: str, retries: int = 2) -> dict:
    """按 PostId 取单个岗位详情（含完整 Responsibility + Requirement）。

    详情接口偶尔会抽风返回空 Data，重试几次；仍失败则返回 {}，
    由调用方回退到列表接口的截断正文 —— 宁可要短正文，也不要丢岗位。
    """
    for attempt in range(retries + 1):
        try:
            payload = get_json(session, TENCENT_DETAIL_URL, params={
                "postId": post_id,
                "language": "zh-cn",
            })
            data = (payload or {}).get("Data") or {}
            if data:
                return data
        except Exception:  # noqa: BLE001 —— 详情补全过程不该把整轮采集搞崩
            if attempt >= retries:
                return {}
        time.sleep(0.6 * (attempt + 1))
    return {}


def fetch_tencent_jobs(
    session,
    keyword: str,
    max_pages: int = 2,
    with_detail: bool = True,
) -> list[dict]:
    """腾讯招聘公开接口（无需登录）。

    分页三要素：pageIndex 从 1 开始、pageSize 控制每页数量、
    空页或不足一页说明到底了，提前 break。
    with_detail=True 时逐个补全详情正文（慢一些，但 JD 完整得多）。
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

    if with_detail:
        results = _enrich_with_detail(session, results)
    return results


def _enrich_with_detail(session, posts: list[dict]) -> list[dict]:
    """把详情接口返回的完整正文合并回列表记录。

    合并策略：详情有值就覆盖列表的截断值（尤其是 Responsibility），
    并把 Requirement（任职要求）拼进正文尾部 —— 要求是匹配技能的关键依据，
    只留职责会让"我会 PyTorch 能投什么"这类问题检索不到岗位。
    """
    for post in posts:
        post_id = post.get("RecruitPostId")
        if not post_id:
            continue
        detail = fetch_tencent_detail(session, str(post_id))
        if not detail:
            continue

        resp = (detail.get("Responsibility") or "").strip()
        req = (detail.get("Requirement") or "").strip()
        # 详情正文通常更完整；只有明显更长才覆盖，避免被劣质数据盖掉好数据
        if len(resp) > len((post.get("Responsibility") or "").strip()):
            post["Responsibility"] = resp
        if req and req not in (post.get("Responsibility") or ""):
            post["Responsibility"] = (post.get("Responsibility") or "").strip() + "\n\n【任职要求】\n" + req
        time.sleep(0.5)  # 详情接口也限速，别把人家打疼了
    return posts


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
