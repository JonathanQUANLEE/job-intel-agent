#!/usr/bin/env bash
# ============================================================
#  Agent 岗位情报站 · 一键启动（macOS / Linux）
#  用法：bash start.sh
# ============================================================
set -e
cd "$(dirname "$0")"

echo ""
echo "  ============================================"
echo "   Agent 岗位情报站 · 启动中"
echo "  ============================================"
echo ""

# ---- 1. 找 Python ----
if command -v python3 >/dev/null 2>&1; then
  PY=python3
elif command -v python >/dev/null 2>&1; then
  PY=python
else
  echo "  [x] 没找到 Python，请先安装 Python 3.10+"
  exit 1
fi
echo "  [1/4] $($PY --version) 已就绪"

# ---- 2. 虚拟环境 ----
if [ ! -x ".venv/bin/python" ]; then
  echo "  [2/4] 首次运行，创建虚拟环境..."
  $PY -m venv .venv
else
  echo "  [2/4] 虚拟环境已存在"
fi

# ---- 3. 依赖 ----
echo "  [3/4] 检查依赖..."
./.venv/bin/python -m pip install -q --disable-pip-version-check -r requirements.txt

# ---- 4. 起服务 ----
echo "  [4/4] 启动服务..."
echo ""
echo "  请打开浏览器访问 http://127.0.0.1:8000"
echo "  按 Ctrl+C 停止服务。"
echo ""

# macOS 自动开浏览器（Linux 上大多数环境没有 open，静默忽略）
if command -v open >/dev/null 2>&1; then open http://127.0.0.1:8000 || true; fi
if command -v xdg-open >/dev/null 2>&1; then xdg-open http://127.0.0.1:8000 || true; fi

exec ./.venv/bin/python -m api
