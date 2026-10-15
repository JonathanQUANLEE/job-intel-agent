"""RAG 配置：API Key 从环境变量读，绝不写进代码。

读取顺序：先看项目根目录的 .env 文件，再看系统环境变量。
三个可调项（都可用环境变量覆盖）：
  SF_LLM_MODEL     对话模型，默认 deepseek-ai/DeepSeek-V3（硅基流动模型名）
  SF_EMBED_MODEL   嵌入模型，默认 BAAI/bge-m3（1024 维，中英通吃）
  SF_API_BASE      接口地址，默认硅基流动；换成任意 OpenAI 兼容网关都能跑
"""

import os
from pathlib import Path

# 硅基流动 OpenAI 兼容接口的公共前缀（文档：docs.siliconflow.com）
BASE_URL = os.environ.get("SF_API_BASE", "https://api.siliconflow.com/v1")

EMBED_MODEL = os.environ.get("SF_EMBED_MODEL", "BAAI/bge-m3")
LLM_MODEL = os.environ.get("SF_LLM_MODEL", "deepseek-ai/DeepSeek-V3")


def _load_env(path: str = ".env") -> None:
    """极简 .env 加载器：每行 KEY=VALUE，# 开头是注释。

    为什么不引 python-dotenv？——十几行能解决的，不引依赖。
    """
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        # setdefault：系统环境变量优先，.env 只是兜底
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


_load_env()

API_KEY = os.environ.get("SILICONFLOW_API_KEY", "")


def has_key() -> bool:
    """有没有配 Key。检索/问答的降级判断都看这里，避免各处重复判断空字符串。"""
    return bool(API_KEY.strip())


def reload_key() -> None:
    """重新读一次 .env（前端让用户当场填 Key 时用，不重启服务就能生效）。

    注意：只覆盖进程环境里的 SILICONFLOW_API_KEY —— 用户现场填的 Key
    会落盘到 .env（本地文件，已在 .gitignore 里），下次启动仍然有效。
    """
    global API_KEY
    p = Path(".env")
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("SILICONFLOW_API_KEY") and "=" in line:
                _, _, value = line.partition("=")
                os.environ["SILICONFLOW_API_KEY"] = value.strip().strip("\"'")
                break
    API_KEY = os.environ.get("SILICONFLOW_API_KEY", "")
