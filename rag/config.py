"""RAG 配置：API Key 从环境变量读，绝不写进代码。

读取顺序：先看项目根目录的 .env 文件，再看系统环境变量。
两个可调项（都可用环境变量覆盖）：
  SF_LLM_MODEL    对话模型，默认 deepseek-ai/DeepSeek-V3（硅基流动模型名）
  SF_EMBED_MODEL  嵌入模型，默认 BAAI/bge-m3（1024 维，中英通吃）
"""

import os
from pathlib import Path

# 硅基流动 OpenAI 兼容接口的公共前缀（文档：docs.siliconflow.com）
BASE_URL = "https://api.siliconflow.com/v1"

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
