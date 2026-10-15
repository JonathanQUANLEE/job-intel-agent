"""一键启动：python start.py

把"装依赖 → 准备数据 → 起服务 → 开浏览器"四步收成一条命令。

为什么用 Python 而不是 .bat / .sh？
- 一份代码跨平台，Windows / macOS / Linux 行为一致；
- 能处理 bat 做不到的事：自动找可用的 Python 解释器、检查版本、
  判断数据是否已导入、优雅处理端口占用。

不做的事：不自动配 API Key（那是用户的选择），没 Key 时服务会走本地降级模式。
"""

import os
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REQUIRED = ["requests", "numpy", "fastapi", "uvicorn", "pydantic", "langgraph"]
HOST, PORT = "127.0.0.1", 8000


def step(n: int, total: int, msg: str) -> None:
    print(f"  [{n}/{total}] {msg}")


def check_python() -> bool:
    if sys.version_info < (3, 10):
        print(f"  [x] 需要 Python 3.10+，当前是 {sys.version.split()[0]}")
        print("      下载：https://www.python.org/downloads/")
        return False
    print(f"  [1/4] Python {sys.version.split()[0]} 已就绪")
    return True


def ensure_deps() -> bool:
    """缺哪个装哪个。已经装好就跳过，不浪费用户时间。"""
    missing = []
    for mod in REQUIRED:
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if not missing:
        print("  [2/4] 依赖已就绪")
        return True

    print(f"  [2/4] 安装缺失依赖：{', '.join(missing)} ...")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "--disable-pip-version-check",
             "-r", str(ROOT / "requirements.txt")],
            cwd=str(ROOT),
        )
    except subprocess.CalledProcessError:
        print("  [x] 依赖安装失败，请检查网络或手动运行：")
        print(f"      pip install -r requirements.txt")
        return False
    print("  [2/4] 依赖安装完成")
    return True


def ensure_data() -> bool:
    """库里有岗位就直接用；空了就尝试从 CSV 快照导入。"""
    db = ROOT / "jobs.db"
    try:
        sys.path.insert(0, str(ROOT))
        from crawler.storage import count_jobs

        n = count_jobs(db) if db.exists() else 0
        if n > 0:
            print(f"  [3/4] 岗位库已有 {n} 条数据")
            return True
    except Exception:  # noqa: BLE001 —— 读库失败就当作空库，走导入流程
        pass

    csv = ROOT / "data" / "jobs_export.csv"
    if csv.exists():
        print("  [3/4] 岗位库为空，从数据快照导入...")
        try:
            from seed import load_csv
            from crawler.storage import save_jobs

            rows = load_csv(str(csv))
            inserted, _ = save_jobs(rows, db)
            print(f"  [3/4] 已导入 {inserted} 条岗位")
            return True
        except Exception as e:  # noqa: BLE001
            print(f"  [!] 导入失败：{e}")
    else:
        print("  [!] 没找到数据快照，岗位库为空")

    print("      稍后可在界面上直接浏览；采集更多数据运行：")
    print('      python -m crawler --source tencent --keyword "大模型" --max-pages 3')
    return True


def port_in_use(host: str, port: int) -> bool:
    """检查端口是否已被占用（通常是自己上次没关的服务）。"""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def serve() -> None:
    """起服务并自动开浏览器。用 uvicorn 直接跑，避免子进程多一层。"""
    global PORT

    if port_in_use(HOST, PORT):
        print(f"  [!] 端口 {PORT} 已被占用（可能上次的服务还在跑）")
        for candidate in range(PORT + 1, PORT + 11):
            if not port_in_use(HOST, candidate):
                print(f"      自动改用 {candidate} 端口")
                PORT = candidate
                break
        else:
            print("  [x] 8000-8010 都被占用了，请先关掉其它服务")
            return

    print(f"  [4/4] 启动服务...\n")
    print(f"  ┌──────────────────────────────────────────┐")
    print(f"  │  访问地址： http://{HOST}:{PORT}")
    print(f"  │  按 Ctrl+C 停止服务")
    print(f"  └──────────────────────────────────────────┘")
    print()

    def open_browser():
        time.sleep(2.0)  # 等服务起来再开，免得打开是"无法连接"
        webbrowser.open(f"http://{HOST}:{PORT}")

    threading.Thread(target=open_browser, daemon=True).start()

    sys.path.insert(0, str(ROOT))
    os.chdir(ROOT)
    import uvicorn

    uvicorn.run("api.server:app", host=HOST, port=PORT, reload=False)


def main() -> None:
    print()
    print("  ============================================")
    print("   Agent 岗位情报站 · 启动中")
    print("  ============================================")
    print()

    if not check_python() or not ensure_deps() or not ensure_data():
        sys.exit(1)

    try:
        serve()
    except KeyboardInterrupt:
        print("\n  服务已停止。")


if __name__ == "__main__":
    main()
