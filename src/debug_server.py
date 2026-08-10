"""诊断服务: 记录并回显前端 (Cherry Studio) 传入的请求体.

用途: 验证前端实际发送的 model 字段、System Prompt (role=system 消息)
与消息结构, 为多库路由和系统提示词透传提供观测点. 与主服务 (8000)
完全隔离, 诊断失败不波及生产路由.

- 独立监听 8001 端口, 不影响主服务 (8000); 常驻作为长期诊断工具.
- Cherry Studio 添加第二个服务商指向 http://localhost:8001/v1, 模型 ID 填 debug.
- 每次请求: 完整记录到 logs/debug_requests.log + 以 OpenAI 格式回显到对话窗.
- 不调用 LLM/Embedding, 零 token 成本.

运行: python -m src.debug_server
"""
from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# 日志写到项目内持久目录 (避免 /tmp 被系统清理); 路径见 .gitignore /logs/
_BASE_DIR = Path(__file__).resolve().parent.parent
_DEBUG_LOG_DIR = _BASE_DIR / "logs"
DEBUG_LOG = _DEBUG_LOG_DIR / "debug_requests.log"


def _log_request(model: str, body: dict) -> None:
    """追加记录一次请求到日志文件 (含完整 messages)."""
    _DEBUG_LOG_DIR.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "body": body,
    }
    with open(DEBUG_LOG, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False, indent=2) + "\n")


def _format_echo(body: dict) -> str:
    """把请求体格式化为可读的对话窗回显文本."""
    lines = [
        "=== Debug Echo: 前端实际发送的请求 ===",
        f"model      : {body.get('model', '')}",
        f"stream     : {body.get('stream', False)}",
        "",
        "messages   :",
    ]
    for i, m in enumerate(body.get("messages", []), start=1):
        role = m.get("role", "?")
        content = m.get("content", "")
        marker = "  << SYSTEM (透传目标)" if role == "system" else ""
        if isinstance(content, list):
            content = json.dumps(content, ensure_ascii=False)
        lines.append(f"  [{i}] role={role}{marker}")
        lines.append(f"      content: {content}")
    lines.append("")
    lines.append("判定提示:")
    lines.append("  - model 字段决定后端路由到 storage/{model}; 多库 = 多个模型名.")
    lines.append("  - system 消息会被后端 _extract_system_prompt 解析并透传.")
    return "\n".join(lines)


class ChatRequest(BaseModel):
    model: str = Field(..., min_length=1)
    messages: list[dict] = Field(..., min_length=1)
    stream: bool = Field(default=False)
    # 其余 OpenAI 参数忽略, 仅转发回显


app = FastAPI(title="My Novel RAG Debug Echo")


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/models")
async def list_models() -> dict:
    return {"object": "list", "data": [{"id": "debug", "object": "model"}]}


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest) -> object:
    body = req.model_dump()
    _log_request(req.model, body)

    content = _format_echo(body)
    if req.stream:
        message_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())

        def _chunks():
            yield (
                'data: {"id": "%s", "object": "chat.completion.chunk", "created": %d, '
                '"model": "%s", "choices": [{"index": 0, "delta": {"role": "assistant"}, '
                '"finish_reason": null}]}\n\n' % (message_id, created, req.model)
            )
            for token in content.split("\n"):
                payload = json.dumps(
                    {
                        "id": message_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": req.model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": token + "\n"},
                                "finish_reason": None,
                            }
                        ],
                    },
                    ensure_ascii=False,
                )
                yield f"data: {payload}\n\n"
            yield (
                'data: {"id": "%s", "object": "chat.completion.chunk", "created": %d, '
                '"model": "%s", "choices": [{"index": 0, "delta": {}, '
                '"finish_reason": "stop"}]}\n\n' % (message_id, created, req.model)
            )
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            _chunks(), media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.debug_server:app", host="0.0.0.0", port=8001, reload=False)