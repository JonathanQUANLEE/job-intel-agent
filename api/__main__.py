"""启动入口：python -m api  →  http://127.0.0.1:8000 打开聊天界面。

默认只监听 127.0.0.1（本机访问）。要让局域网内其它设备访问，
用 --host 0.0.0.0 显式开启——默认不开，避免无意中把服务暴露到网络上。
"""

import argparse

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(prog="api", description="Agent 岗位情报站 · Web 服务")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址（默认仅本机）")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="改代码自动重启（开发用）")
    args = parser.parse_args()

    print(f"访问 http://{args.host if args.host != '0.0.0.0' else '127.0.0.1'}:{args.port}")
    uvicorn.run("api.server:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
