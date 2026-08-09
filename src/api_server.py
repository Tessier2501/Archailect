"""FastAPI 服务: 暴露 OpenAI 兼容的 /v1/chat/completions。

三大核心逻辑:
  1. 多书路由: 按请求 model 字段懒加载 storage/{model} 的 LightRAG 实例
  2. System Prompt 透传: 见 build_rag_system_prompt() 注释
  3. 响应格式: 严格 OpenAI 规范 (非流式 JSON / 流式 SSE)
"""
from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache
from pydantic import BaseModel, Field

from src.config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    STORAGE_DIR,
    build_embedding_func,
)

# ============================================================
# 一、多书路由的核心数据结构 (LRU + 并发锁 + 在途请求保护)
# ============================================================
# 每个 model 对应一个独立 LightRAG 实例, working_dir 指向 storage/{model},
# 从物理目录层面保证知识库完全隔离。

# 实例缓存: OrderedDict 支持 LRU 顺序 (move_to_end 更新热度)
_rag_instances: OrderedDict[str, LightRAG] = OrderedDict()
# 每实例在途请求计数: 淘汰前必须为 0, 防止中断正在进行的查询
_rag_refcounts: dict[str, int] = {}
# per-model 异步锁: 保证同一本书的首次加载不会并发重复实例化
_rag_locks: dict[str, asyncio.Lock] = {}
# 缓存容量上限 (实例数), 超出后惰性淘汰最久未用且无在途请求的实例
_RAG_CACHE_MAX = int(os.getenv("RAG_CACHE_MAX", "8"))


async def _finalize_and_remove(model: str) -> None:
    """将实例移出缓存并释放资源 (必须在持有该 model 锁的上下文中调用)。"""
    rag = _rag_instances.pop(model, None)
    _rag_refcounts.pop(model, None)
    _rag_locks.pop(model, None)
    if rag is not None:
        try:
            await rag.finalize_storages()
        except Exception:
            pass


async def _evict_if_needed() -> None:
    """容量超限时, 从最久未用开始淘汰「无在途请求」的实例。

    - 在 _get_rag_instance 的锁内调用, 天然无并发风险。
    - 只有 refcount == 0 的实例才可淘汰; 若全部在途则暂不淘汰,
      待请求结束后由下一次访问自然触发。
    """
    while len(_rag_instances) > _RAG_CACHE_MAX:
        victim = next(
            (m for m in _rag_instances if _rag_refcounts.get(m, 0) == 0),
            None,
        )
        if victim is None:
            break
        await _finalize_and_remove(victim)


async def _get_rag_instance(model: str) -> LightRAG:
    """按 model 懒加载/缓存 LightRAG 实例 (多书路由)。

    - 并发安全: 每 model 一把 asyncio.Lock, 同一本书的首次加载仅一个协程执行
      (双重检查, 等待锁期间其他协程可能已创建)。
    - LRU 淘汰: 容量超限时在锁内惰性淘汰无在途请求的最久未用实例。
    - 目录不存在时抛 404 语义错误。
    """
    # 快路径: 已缓存则直接返回 (不持有锁, 避免无谓竞争)
    if model in _rag_instances:
        _rag_instances.move_to_end(model)
        return _rag_instances[model]

    # 慢路径: 加锁, 双重检查后创建
    if model not in _rag_locks:
        _rag_locks[model] = asyncio.Lock()
    async with _rag_locks[model]:
        if model in _rag_instances:  # 双重检查
            _rag_instances.move_to_end(model)
            return _rag_instances[model]

        storage_dir = STORAGE_DIR / model
        if not storage_dir.is_dir():
            raise HTTPException(
                status_code=404,
                detail=f"知识库 '{model}' 不存在。请先运行: python -m src.builder --txt xxx.txt --book {model}",
            )

        rag = LightRAG(
            working_dir=str(storage_dir),
            llm_model_func=openai_complete_if_cache,
            llm_model_name=DEEPSEEK_MODEL,
            llm_model_kwargs={"base_url": DEEPSEEK_BASE_URL, "api_key": DEEPSEEK_API_KEY},
            embedding_func=build_embedding_func(),
        )
        _rag_instances[model] = rag
        _rag_refcounts[model] = 0
        await _evict_if_needed()
        return rag


# ============================================================
# 二、System Prompt 透传 (核心难点)
# ============================================================
def build_rag_system_prompt(user_system_prompt: str) -> str:
    """包装用户 System Prompt, 使其在不丢失的前提下注入检索上下文。

    背景: LightRAG 的 kg_query 会对传入的 system_prompt 强制调用
          `str.format(response_type=..., user_prompt=..., context_data=...)`
          (见 lightrag/operate.py:4288-4293)。
    因此必须:
      1. 把用户原文的 { } 转义为 {{ }}, 否则会被 format 当作占位符,
         要么吞掉检索上下文, 要么抛 KeyError;
      2. 提供 {context_data} 占位符, 让 LightRAG 把检索到的书籍背景注入;
      3. 提供 {user_prompt} 占位符, 兼容 QueryParam.user_prompt 附加指令。
    这样「用户系统提示词」与「检索上下文」两条信息流都无损地进最终 prompt。
    """
    escaped = user_system_prompt.replace("{", "{{").replace("}", "}}")
    return (
        "---User-defined Role---\n"
        f"{escaped}\n\n"
        "You MUST obey the role constraints defined above.\n"
        "Then answer the user query, grounding on the knowledge base context "
        "provided below when it is relevant.\n\n"
        "Additional instructions: {user_prompt}\n\n"
        "---Knowledge Base Context---\n"
        "{context_data}"
    )


def _extract_system_prompt(messages: list[dict[str, Any]]) -> str:
    """提取所有 role=system 的消息内容, 用换行合并 (透传不丢弃)。"""
    parts = [m.get("content", "") for m in messages if m.get("role") == "system"]
    return "\n".join(p for p in parts if isinstance(p, str) and p.strip())


def _extract_last_user_query(messages: list[dict[str, Any]]) -> str:
    """提取最后一条 user 消息作为交给 LightRAG 的提问。"""
    for m in reversed(messages):
        if m.get("role") == "user":
            content = m.get("content", "")
            if isinstance(content, str) and content.strip():
                return content
            # 处理 OpenAI 多段 content 数组 (string | list[part])
            if isinstance(content, list):
                texts = [
                    part.get("text", "")
                    for part in content
                    if isinstance(part, dict) and part.get("type") == "text"
                ]
                joined = "".join(texts).strip()
                if joined:
                    return joined
    raise HTTPException(status_code=400, detail="请求缺少有效的 user 消息")


# ============================================================
# 三、OpenAI 兼容响应组装
# ============================================================
def _openai_error(body: dict[str, Any], status: int) -> HTTPException:
    return HTTPException(
        status_code=status,
        detail={
            "error": {
                "message": body.get("message", "Unknown error"),
                "type": body.get("type", "invalid_request_error"),
                "code": body.get("code"),
            }
        },
    )


def _non_stream_response(content: str, model: str) -> dict[str, Any]:
    """非流式响应的严格 OpenAI 结构。"""
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def _sse_event(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


async def _stream_response(
    token_iter: AsyncIterator[str], model: str
) -> AsyncIterator[str]:
    """把 LightRAG 流式输出包装为 OpenAI SSE chunk 序列。

    客户端中断处理: 前端停止生成/关闭连接时, 生成器收到
    asyncio.CancelledError。它继承自 BaseException, 普通 except Exception
    捕获不到, 必须显式捕获。取消中的 task 内无法直接 await (会立即再抛
    CancelledError), 因此清理动作用 asyncio.shield() 隔离后再 raise。
    """
    message_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    try:
        # 首帧: 声明 assistant 角色
        yield _sse_event(
            {
                "id": message_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
                ],
            }
        )
        async for token in token_iter:
            yield _sse_event(
                {
                    "id": message_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [
                        {"index": 0, "delta": {"content": token}, "finish_reason": None}
                    ],
                }
            )
        # 结束帧
        yield _sse_event(
            {
                "id": message_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
        )
        yield "data: [DONE]\n\n"
    except asyncio.CancelledError:
        # 客户端断开: 优雅关闭底层 LLM 异步生成器, 不向上溢出
        if hasattr(token_iter, "aclose"):
            await asyncio.shield(token_iter.aclose())  # type: ignore[union-attr]
        raise


# ============================================================
# 四、FastAPI 应用
# ============================================================
class ChatRequest(BaseModel):
    model: str = Field(..., min_length=1)
    messages: list[dict[str, Any]] = Field(..., min_length=1)
    stream: bool = Field(default=False)
    temperature: float | None = None
    max_tokens: int | None = None
    # 其余 OpenAI 参数忽略即可, 不影响功能


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    # 关闭时持久化所有实例的缓存数据
    for model, rag in _rag_instances.items():
        try:
            await rag.finalize_storages()
        except Exception:
            pass


app = FastAPI(title="My Novel RAG", lifespan=lifespan)

# Cherry Studio / 其他前端跨域访问必需
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/models")
async def list_models() -> dict[str, Any]:
    """方便 Cherry Studio 发现可用模型。"""
    models = [
        {"id": d.name, "object": "model", "owned_by": "my-novel-rag"}
        for d in sorted(STORAGE_DIR.iterdir())
        if d.is_dir()
    ]
    return {"object": "list", "data": models}


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest) -> Any:
    # ---- 1. 多书路由: 解析 model 并获取对应实例 (并发安全 + LRU) ----
    # 先登记在途引用, 防止 LRU 淘汰正在被查询的实例
    _rag_refcounts[req.model] = _rag_refcounts.get(req.model, 0) + 1
    try:
        try:
            rag = await _get_rag_instance(req.model)
        except HTTPException as exc:
            raise _openai_error(
                {
                    "message": exc.detail,
                    "type": "invalid_request_error",
                    "code": "model_not_found",
                },
                exc.status_code,
            )

        # ---- 2. System Prompt 透传 + 提取用户提问 ----
        user_system_prompt = _extract_system_prompt(req.messages)
        query = _extract_last_user_query(req.messages)
        rag_system_prompt = build_rag_system_prompt(user_system_prompt)

        # ---- 3. LightRAG 查询 (hybrid 模式; 不依赖 1.5.6 的默认 mix) ----
        param = QueryParam(mode="hybrid", stream=req.stream)
        result = await rag.aquery(query, param=param, system_prompt=rag_system_prompt)

        # ---- 4. 响应格式: 流式 SSE / 非流式 JSON ----
        if req.stream:
            # aquery stream=True 返回 AsyncIterator[str]
            return StreamingResponse(
                _stream_response(result, req.model),  # type: ignore[arg-type]
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        return _non_stream_response(str(result), req.model)
    finally:
        # 请求结束释放引用; 若容量超限, 下一次 _get_rag_instance 的锁内会触发淘汰
        _rag_refcounts[req.model] = max(0, _rag_refcounts.get(req.model, 0) - 1)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.api_server:app", host="0.0.0.0", port=8000, reload=True)