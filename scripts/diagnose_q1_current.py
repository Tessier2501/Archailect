"""一次性复测 (2026-08-11): 用当前生产 QueryParam 配置复测 Q1 反事实问题. 只读, 不改生产代码/索引.

生产配置 (api_server.py chat_completions):
  mode=mix, top_k=20, chunk_top_k=12, enable_rerank=build_rerank_func() is not None, conversation_history=history
依据用户实测 (合并澄清句+反事实问题的单条消息), 这里 conversation_history=[] (单轮).
"""
from __future__ import annotations

import asyncio
import json

from lightrag import LightRAG, QueryParam

from src.config import (
    STORAGE_DIR,
    build_embedding_func,
    build_llm_func,
    build_rerank_func,
    build_role_llm_configs,
)

Q1_FULL = (
    "在提及凝胶时，我特指“Node 1211/BCC”，或“1211”。"
    "为什么智能凝胶即使被编程为阻止ßehemoth，仍然未能把Lenie Clarke消灭在beebe站？"
    "在她调查海底核弹的时候，将核弹引爆，不就能有效地遏制ßehemoth吗？"
)

# 答案锚点: 均来自 Starfish 原文 17269-17357 区域
ANCHORS = {
    "identity_17269": "chess and checkers",          # Node 1211 身份/编程
    "choice_17337": "biosphere or \u00dfehemoth",     # 二选一决策
    "answer_17345": "self-sustaining copy",          # 案例 1 核心答案
}


async def main() -> None:
    kwargs: dict = {
        "working_dir": str(STORAGE_DIR / "rifters"),
        "workspace": "rifters",
        "llm_model_func": build_llm_func(),
        "embedding_func": build_embedding_func(),
    }
    rerank_func = build_rerank_func()
    if rerank_func is not None:
        kwargs["rerank_model_func"] = rerank_func
    role_configs = build_role_llm_configs()
    if role_configs:
        kwargs["role_llm_configs"] = role_configs

    rag = LightRAG(**kwargs)
    await rag.initialize_storages()
    try:
        param = QueryParam(
            mode="mix",
            top_k=20,
            chunk_top_k=12,
            enable_rerank=rerank_func is not None,
            conversation_history=[],
        )
        result = await rag.aquery_data(Q1_FULL, param=param)
        data = result.get("data", {}) if result.get("status") == "success" else {}
        chunks = data.get("chunks", [])
        entities = data.get("entities", [])

        print("status:", result.get("status"))
        print("n_entities:", len(entities))
        print("n_chunks (final):", len(chunks))
        print("entities (前10):")
        for e in entities[:10]:
            print("  ", e.get("entity_name"), "|", e.get("entity_type"))
        print()
        print("--- 锚点命中检查 ---")
        found: dict[str, bool] = {k: False for k in ANCHORS}
        for c in chunks:
            content = c.get("content", "")
            for k, anchor in ANCHORS.items():
                if anchor in content:
                    found[k] = True
        for k, hit in found.items():
            print(f"  {k}: {'HIT' if hit else 'MISS'}")
        print()
        print("--- 前 5 条 final chunk 预览 ---")
        for i, c in enumerate(chunks[:5]):
            print(f"[{i}] {c.get('content', '')[:150]!r}")
    finally:
        await rag.finalize_storages()


if __name__ == "__main__":
    asyncio.run(main())