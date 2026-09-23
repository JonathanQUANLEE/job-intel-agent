"""公共 HTTP 客户端：Session 复用连接 + 自动重试退避。

为什么单独一个文件？
所有数据源共用同一套"礼貌策略"（UA、超时、重试、退避），
改一处 = 全部数据源生效，这就是"模块化"的最小体现。
"""

import requests
from requests.adapters import HTTPAdapter, Retry

USER_AGENT = "job-intel-agent/0.1 (personal learning project; contact: you@example.com)"

# 429=请求太频繁, 5xx=服务器临时故障 —— 这类错误值得重试，404 之类的没必要
_RETRYABLE_STATUS = [429, 500, 502, 503, 504]


def build_session() -> requests.Session:
    """创建一个带重试策略的 Session。

    Session 的好处：
    1. 复用 TCP 连接（同一站点连续请求快很多）；
    2. 统一挂载 headers / 重试策略，不用每次 get() 都传。
    """
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})

    retry = Retry(
        total=3,                                  # 连接错误+状态码错误，合计最多重试 3 次
        backoff_factor=1,                         # 退避间隔：1s → 2s → 4s（指数增长）
        status_forcelist=_RETRYABLE_STATUS,       # 只有这些状态码触发重试
        allowed_methods=["GET"],                  # 只对 GET 重试，安全
        respect_retry_after_header=True,          # 服务器说"等 N 秒再试"就听话
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def get_json(session: requests.Session, url: str, params: dict | None = None) -> dict | list:
    """GET 一个 JSON 接口并解析。所有数据源都走这里，行为保持一致。

    timeout=(连接超时, 读取超时)：不设 timeout 的 requests 请求可能永远卡死。
    raise_for_status()：4xx/5xx 直接抛异常，把问题暴露在抓取阶段而不是解析阶段。
    """
    resp = session.get(url, params=params, timeout=(5, 20))
    resp.raise_for_status()
    return resp.json()
