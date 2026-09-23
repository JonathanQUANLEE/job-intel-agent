"""启动入口：python -m api  →  http://127.0.0.1:8000/docs 有交互式文档。"""

import uvicorn

if __name__ == "__main__":
    uvicorn.run("api.server:app", host="127.0.0.1", port=8000, reload=False)
