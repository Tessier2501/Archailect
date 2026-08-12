"""FastAPI 服务: 暴露 OpenAI 兼容的 /v1/chat/completions.

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
from pydantic import BaseModel, Field

from src.config import (
    RAG_CACHE_MAX,
    STORAGE_DIR,
    build_embedding_func,
    build_llm_func,
    build_rerank_func,
    build_role_llm_configs,
)

# ============================================================
# 一, 多书路由的核心数据结构 (LRU + 并发锁 + 在途请求保护)
# ============================================================
# 每个 model 对应一个独立 LightRAG 实例, working_dir 指向 storage/{model},
# 从物理目录层面保证知识库完全隔离.

# 实例缓存: OrderedDict 支持 LRU 顺序 (move_to_end 更新热度)
_rag_instances: OrderedDict[str, LightRAG] = OrderedDict()
# 每实例在途请求计数: 淘汰前必须为 0, 防止中断正在进行的查询
_rag_refcounts: dict[str, int] = {}
# per-model 异步锁: 保证同一本书的首次加载不会并发重复实例化
_rag_locks: dict[str, asyncio.Lock] = {}
# 缓存容量上限 (实例数, 来自 src/config.py RAG_CACHE_MAX)
# 超出后惰性淘汰最久未用且无在途请求的实例
_RAG_CACHE_MAX = RAG_CACHE_MAX


async def _finalize_and_remove(model: str) -> None:
    """将实例移出缓存并释放资源 (必须在持有该 model 锁的上下文中调用)."""
    rag = _rag_instances.pop(model, None)
    _rag_refcounts.pop(model, None)
    _rag_locks.pop(model, None)
    if rag is not None:
        try:
            await rag.finalize_storages()
        except Exception:
            pass


async def _evict_if_needed() -> None:
    """容量超限时, 从最久未用开始淘汰无在途请求的实例.

    - 在 _get_rag_instance 的锁内调用, 天然无并发风险.
    - 只有 refcount == 0 的实例才可淘汰; 若全部在途则暂不淘汰,
      待请求结束后由下一次访问自然触发.
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
    """按 model 懒加载/缓存 LightRAG 实例 (多书路由).

    - 并发安全: 每 model 一把 asyncio.Lock, 同一本书的首次加载仅一个协程执行
      (双重检查, 等待锁期间其他协程可能已创建).
    - LRU 淘汰: 容量超限时在锁内惰性淘汰无在途请求的最久未用实例.
    - 目录不存在时抛 404 语义错误.
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
                detail=f"知识库 '{model}' 不存在. 请先运行: python -m src.builder --txt xxx.txt --book {model}",
            )

        rag_kwargs: dict = {
            "working_dir": str(storage_dir),
            "workspace": model,
            "llm_model_func": build_llm_func(),
            "embedding_func": build_embedding_func(),
        }
        # 可选 rerank 精排: 未配置 (env 三键任一为空) 时仅警告无害空转;
        # 与 enable_rerank 联动, 见 chat_completions 的 QueryParam.
        rerank_func = build_rerank_func()
        if rerank_func is not None:
            rag_kwargs["rerank_model_func"] = rerank_func
        role_configs = build_role_llm_configs()
        if role_configs:
            rag_kwargs["role_llm_configs"] = role_configs
        rag = LightRAG(**rag_kwargs)
        # 与 builder.py 一致: 显式初始化 pipeline_status 与各 storage.
        # 缺失会导致查询路径 async with None (pipeline_status_lock 未初始化),
        # 已实测: 建图成功但查询报 "NoneType does not support async context manager".
        # workspace=model: 与 builder (workspace=book) 对齐, 共享内存缓存按
        # (namespace, workspace) 隔离, 否则多库实例共用 "" 命名空间互相覆盖 (已实测串台).
        await rag.initialize_storages()
        _rag_instances[model] = rag
        _rag_refcounts[model] = 0
        await _evict_if_needed()
        return rag


# ============================================================
# 二, System Prompt 透传 (核心难点)
# ============================================================
def build_rag_system_prompt(user_system_prompt: str) -> str:
    """包装用户 System Prompt, 使其在不丢失的前提下注入检索上下文.

    注意 (2026-08-12 提案 13 之后): 主链路已改为手动组装 merged_sys_prompt 并直调
    LLM 封装器 (openai_complete_if_cache), 不再经 LightRAG kg_query 的 .format().
    本函数保留为兼容工具/文档参考 (若未来回归 aquery 路径, 仍须用它做 { } 转义).

    原理 (历史路径, 供 aquery 回归参考): LightRAG 的 kg_query 会对传入的
    system_prompt 强制调用 str.format(response_type=..., user_prompt=...,
    context_data=...). 那时必须把用户原文的 { } 转义为 {{ }}, 并提供
    {context_data}/{user_prompt} 占位符, 否则检索上下文丢失或抛 KeyError.
    本函数即为此形态的模板生成器.
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
    """提取所有 role=system 的消息内容, 用换行合并 (透传不丢弃)."""
    parts = [m.get("content", "") for m in messages if m.get("role") == "system"]
    return "\n".join(p for p in parts if isinstance(p, str) and p.strip())


_MSG_CONTENT_MAX = 6


def _msg_to_text(content: Any) -> str:
    """将 OpenAI 消息 content (string | list[part]) 规整为纯文本."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return ""


def _extract_conversation_history(
    messages: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """取最后一条 user 之前的最近 N 条消息作为对话历史.

    仅作 LLM 生成上下文 (1.5.6 验证: 不参与检索), 限 _MSG_CONTENT_MAX 条控 token.
    system 消息已在 _extract_system_prompt 单独透传, 这里排除避免重复.
    """
    history: list[dict[str, str]] = []
    # 从后往前, 跳过最后一条 user (它才是本次检索/提问串)
    for m in reversed(messages[:-1]):
        role = m.get("role", "")
        if role not in ("user", "assistant"):
            continue
        text = _msg_to_text(m.get("content", "")).strip()
        if not text:
            continue
        history.append({"role": role, "content": text})
        if len(history) >= _MSG_CONTENT_MAX:
            break
    history.reverse()
    return history


def _extract_last_user_query(messages: list[dict[str, Any]]) -> str:
    """提取最后一条 user 消息作为交给 LightRAG 的提问."""
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
# 二.5, 聚焦查询改写 + 双路检索并集 (提案 13, 2026-08-12)
# ============================================================
# 根因 (七轮诊断收口): 用户原样反事实提问 (如 "为什么没做 X") 与答案段
# (陈述句 "决定做 Y 因为...") 语义错配, 任何 embedding 下都召不回答案段.
# 而对"聚焦的陈述化检索串" (实体名 + 具体属性 + 动作结果, 少而准) mix 链路
# 能稳定命中 (chunk-104 = #3). 因此在本服务检索前用 QUERY 模型做聚焦改写,
# 改写串负责检索, 生成仍对准用户原始问题.
#
# 关键约束 (第七轮复核实证 + 三测试回传修正):
# - 保留原问题全部关键实体/事实;
# - 仅做问题指向的"最小因果补全" (补全未做的替代决策), 禁止答案外想象
#   (bioluminescence/warhead 等完整版 HyDE 失败形态);
# - 输出只给检索串, 不解释.
REWRITE_QUERY_SYS = (
    "You rewrite a user's question into a focused English retrieval string "
    "for a book knowledge base. Rules:\n"
    "1. Keep ALL key entities and facts already present in the question.\n"
    "2. Do NOT add imagined environment, attributes, or plot details not "
    "mentioned in the question (e.g. biology, weapons, geology).\n"
    "3. Adapt the retrieval form to the question yourself: for counterfactual "
    "'why X did not do Y' questions, perform the minimal causal completion -- "
    "include X's constraint AND the alternative decision/action the "
    "counterfactual implies, using decision words such as "
    "chose/decided/copy/move/relocate/disrupt/destroy (e.g. 'was programmed to "
    "stop B but could not destroy it; chose to move a copy of B elsewhere "
    "instead'); for broad descriptive questions, emit a compact list of the "
    "core entities/concepts and their relationships instead; for simple lookups, "
    "keep it short and entity-focused. You decide which form retrieves best.\n"
    "4. Remove the question's rhetorical shell; keep the semantic core "
    "(who, what was not done, why).\n"
    "5. Output ONLY the rewritten string, no explanation."
)


async def _rewrite_query(query: str) -> str:
    """把用户问题改写为聚焦检索串 (提案 13). 失败回退原 query (fail-safe)."""
    try:
        llm = build_llm_func()
        rewritten = await llm(query, system_prompt=REWRITE_QUERY_SYS)
        rewritten = str(rewritten).strip()
        return rewritten if rewritten else query
    except Exception:
        # fail-safe: 改写失败不阻断正常查询, 回退原始 query
        return query


async def _retrieve_union(
    rag: LightRAG,
    queries: list[str],
    rerank_on: bool,
) -> tuple[list[dict[str, Any]], list[list[dict[str, Any]]]]:
    """多检索串并行 (aquery_data) 合并候选池.

    返回 (merged, per_query_chunks):
      - merged: 按 chunk_id 去重的并集 (保持首次出现顺序) —— 供生成.
      - per_query_chunks: 每路检索各自去重后的有序候选 —— 供 P-3' 源簇保底.
    双路并集 (提案 13): 改写串召回决策段, 原问题串召回泛化情节.
    检索阶段 conversation_history 不参与 (1.5.6 仅生成上下文); 关键词提取受缓存保护.
    """
    param = QueryParam(
        mode="mix",
        top_k=20,
        chunk_top_k=12,
        enable_rerank=rerank_on,
        conversation_history=[],
    )
    results = await asyncio.gather(
        *[rag.aquery_data(q, param=param) for q in queries],
        return_exceptions=True,
    )
    per_query_chunks: list[list[dict[str, Any]]] = []
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for result in results:
        if isinstance(result, BaseException):
            continue
        if not result.get("status") == "success":
            continue
        chunks = result.get("data", {}).get("chunks", [])
        if not isinstance(chunks, list):
            continue
        q_seen: set[str] = set()
        q_chunks: list[dict[str, Any]] = []
        for chunk in chunks:
            if not isinstance(chunk, dict):
                continue
            key = chunk.get("chunk_id") or chunk.get("content", "")
            if key in q_seen:
                continue
            q_seen.add(key)
            q_chunks.append(chunk)
            if key not in seen:
                seen.add(key)
                merged.append(chunk)
        per_query_chunks.append(q_chunks)
    return merged, per_query_chunks


# P-3' 源簇保底: 每路检索保底 K 个核心候选 (防单路高权重簇淹没他路), 其余按并集顺序补齐.
# 仅决定"有资格参与后续相关性重排", 最终取舍仍由 P-1 与用户原始问题相关度决定.
# P-1: 保底/补齐后的候选池用 QUERY 模型按"用户原始问题"相关度重排, 取前 _RERANK_TOP_N 进生成 (相关优先, 去低相关噪声).
_NEIGHBORHOOD_K = 3
_RERANK_POOL_MAX = 24
_RERANK_TOP_N = 12


def _floor_per_source(
    per_query_chunks: list[list[dict[str, Any]]],
    merged: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """P-3' 源簇保底: 返回最多 _RERANK_POOL_MAX 个候选 (供 P-1 重排)."""
    key_of = lambda c: c.get("chunk_id") or c.get("content", "")
    floored: list[dict[str, Any]] = []
    seen: set[str] = set()
    for q_chunks in per_query_chunks:
        for chunk in q_chunks[:_NEIGHBORHOOD_K]:
            key = key_of(chunk)
            if key not in seen:
                seen.add(key)
                floored.append(chunk)
    # 补齐: 按并集顺序, 填入未保底的候选, 直至池上限.
    for chunk in merged:
        if len(floored) >= _RERANK_POOL_MAX:
            break
        key = key_of(chunk)
        if key not in seen:
            seen.add(key)
            floored.append(chunk)
    return floored


RELEVANCE_RERANK_SYS = (
    "You are a relevance ranker. Given the user's ORIGINAL question and a "
    "numbered list of candidate passages, rank the passages by how directly "
    "they help answer that exact question.\n"
    "Output ONLY a JSON object with key \"ranking\": an array of the candidate "
    "indices (0-based, as listed) ordered from most to least relevant. Use every "
    "index exactly once. No explanation."
)


async def _relevance_rerank(
    chunks: list[dict[str, Any]],
    user_query: str,
) -> list[dict[str, Any]]:
    """P-1 两级精排: 用 QUERY 模型按与用户原始问题的相关性重排候选, 取前 12 进生成.

    与改写串/检索串无关的原始问题做打分基准. 异常/非法输出回退原顺序 (fail-safe).
    """
    if not chunks:
        return chunks
    llm = build_llm_func()
    numbered = "\n\n".join(
        f"[{i}] {c.get('content', '')}" for i, c in enumerate(chunks)
    )
    prompt = (
        f"User's original question:\n{user_query}\n\n"
        f"Candidate passages:\n{numbered}"
    )
    try:
        raw = await llm(prompt, system_prompt=RELEVANCE_RERANK_SYS)
        raw = str(raw).strip()
        # 容错: 取首个外层 JSON 对象
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1:
            raise ValueError("no JSON object in rerank output")
        data = json.loads(raw[start : end + 1])
        order = data.get("ranking")
        if not isinstance(order, list) or len(order) != len(chunks):
            raise ValueError("invalid ranking length")
        idx = [int(v) for v in order]
        if sorted(idx) != list(range(len(chunks))):
            raise ValueError("ranking indices not a permutation")
        ranked = [chunks[i] for i in idx]
    except Exception:
        # fail-safe: 重排失败回退原顺序
        return chunks
    return ranked[:_RERANK_TOP_N]


def _chunks_to_context(chunks: list[dict[str, Any]]) -> str:
    """并集候选转 Knowledge Base Context 文本 (供生成提示词)."""
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        parts.append(f"[Chunk {i}] {chunk.get('content', '')}")
    return "\n\n".join(parts)


# ============================================================
# 三, OpenAI 兼容响应组装
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
    """非流式响应的严格 OpenAI 结构."""
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
    """把 LightRAG 流式输出包装为 OpenAI SSE chunk 序列.

    客户端中断处理: 前端停止生成/关闭连接时, 生成器收到
    asyncio.CancelledError. 它继承自 BaseException, 普通 except Exception
    捕获不到, 必须显式捕获. 取消中的 task 内无法直接 await (会立即再抛
    CancelledError), 因此清理动作用 asyncio.shield() 隔离后再 raise.
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
# 四, FastAPI 应用
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
    """方便 Cherry Studio 发现可用模型."""
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

        # ---- 2. System Prompt 透传 + 提取用户提问 + 多轮上下文 ----
        user_system_prompt = _extract_system_prompt(req.messages)
        query = _extract_last_user_query(req.messages)
        history = _extract_conversation_history(req.messages)
        rag_system_prompt = build_rag_system_prompt(user_system_prompt)

        # ---- 3. 聚焦查询改写 + 双路检索并集 (提案 13) + 源簇保底 (P-3') + 相关性重排 (P-1) ----
        # 检索前把用户问题改写为聚焦检索串 (P-2' 自适应), 改写失败回退原 query (fail-safe).
        # 双路并集: 改写串召回核心段, 原问题串召回泛化情节, 按 chunk_id 去重.
        # 源簇保底 (P-3'): 每路保底 K 个核心候选, 防单路高权重簇淹没他路.
        # 相关性重排 (P-1): 用 QUERY 模型按"用户原始问题"重排候选, 取前 12 进生成 (相关优先, 去低相关噪声).
        # 生成仍对准用户原始问题 (各检索串只负责召回).
        rerank_available = build_rerank_func() is not None
        rewritten = await _rewrite_query(query)
        try:
            merged, per_query_chunks = await _retrieve_union(
                rag,
                [query, rewritten] if rewritten != query else [query],
                rerank_available,
            )
            pool = _floor_per_source(per_query_chunks, merged)
            final_chunks = await _relevance_rerank(pool, query)
            context_data = _chunks_to_context(final_chunks)
            if not context_data.strip():
                context_data = "No relevant context found."
        except HTTPException:
            raise
        except Exception:
            # fail-safe: 检索并集异常退化为原 aquery 路径 (单路, 保持服务可用)
            param = QueryParam(
                mode="mix",
                stream=req.stream,
                top_k=20,
                chunk_top_k=12,
                enable_rerank=rerank_available,
                conversation_history=history,
            )
            result = await rag.aquery(query, param=param, system_prompt=rag_system_prompt)

            if req.stream:
                return StreamingResponse(
                    _stream_response(result, req.model),  # type: ignore[arg-type]
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "X-Accel-Buffering": "no",
                    },
                )
            return _non_stream_response(str(result), req.model)

        # 组装检索上下文到 system_prompt. 注意: 本路径绕过 LightRAG 的 kg_query
        # (不再经 operate.py 的 str.format), 改由 LLM 封装器直接调
        # openai_complete_if_cache — 故用户原文的 { } 无需转义 (原转义仅针对
        # LightRAG .format() 路径, 见 build_rag_system_prompt 注释). 原始问题
        # 作为生成输入 (query), 检索上下文经 system_prompt 注入.
        merged_sys_prompt = (
            "---User-defined Role---\n"
            f"{user_system_prompt}\n\n"
            "You MUST obey the role constraints defined above.\n"
            "Then answer the user query, grounding on the knowledge base context "
            "provided below when it is relevant.\n\n"
            "---Knowledge Base Context---\n"
            f"{context_data}"
        )

        llm = build_llm_func()
        result = await llm(
            query,
            system_prompt=merged_sys_prompt,
            history_messages=history or None,
        )

        # ---- 4. 响应格式: 流式 SSE / 非流式 JSON ----
        if req.stream:
            # 生成已完成 (非流式调用), 流式响应按单块扇出模拟
            async def _single_chunk_stream() -> AsyncIterator[str]:
                yield _sse_event(
                    {
                        "id": f"chatcmpl-{uuid.uuid4().hex}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": req.model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"role": "assistant"},
                                "finish_reason": None,
                            }
                        ],
                    }
                )
                yield _sse_event(
                    {
                        "id": f"chatcmpl-{uuid.uuid4().hex}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": req.model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": str(result)},
                                "finish_reason": None,
                            }
                        ],
                    }
                )
                yield _sse_event(
                    {
                        "id": f"chatcmpl-{uuid.uuid4().hex}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": req.model,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    }
                )
                yield "data: [DONE]\n\n"

            return StreamingResponse(
                _single_chunk_stream(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        return _non_stream_response(str(result), req.model)
    finally:
        # 请求结束释放引用; 若容量超限, 下一次 _get_rag_instance 的锁内会触发淘汰
        _rag_refcounts[req.model] = max(0, _rag_refcounts.get(req.model, 0) - 1)


if __name__ == "__main__":
    import uvicorn

    # reload=False: 正式运行避免 reloader+worker 双进程与 VSCode 端口转发噪声.
    uvicorn.run("src.api_server:app", host="0.0.0.0", port=8000, reload=False)
