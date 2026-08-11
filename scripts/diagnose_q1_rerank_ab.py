"""对照诊断 (2026-08-11): rerank on/off × 3 次, 判定 Q1 答案锚点 MISS 的机制.

- rerank on:  走精排 (与生产一致), 应得 final chunks 约 12 条
- rerank off: 无精排, 候选按原始顺序截断到 chunk_top_k=12
对比两者锚点命中: 若 off 命中而 on MISS → rerank 挤出锚点; 若两者都 MISS → 候选池本身无锚点 (召回失败).
"""
from __future__ import annotations

import asyncio

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

ANCHORS = {
    "identity_17269": "chess and checkers",
    "choice_17337": "biosphere or \u00dfehemoth",
    "answer_17345": "self-sustaining copy",
}

N_RUNS = 3


async def run_once(rag: LightRAG, tag: str, enable_rerank: bool) -> dict:
    param = QueryParam(
        mode="mix",
        top_k=20,
        chunk_top_k=12,
        enable_rerank=enable_rerank,
        conversation_history=[],
    )
    result = await rag.aquery_data(Q1_FULL, param=param)
    chunks = result.get("data", {}).get("chunks", []) if result.get("status") == "success" else []
    hits = {}
    for c in chunks:
        content = c.get("content", "")
        for k, anchor in ANCHORS.items():
            if anchor in content:
                hits.setdefault(k, 0)
                hits[k] += 1
    print(f"  [{tag}] n_chunks={len(chunks)} anchors={hits or 'NONE'}")
    return {k: bool(v) for k, v in hits.items()}


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
        print("=== rerank OFF (无精排, 原始截断) ===")
        for i in range(N_RUNS):
            await run_once(rag, f"off-{i}", enable_rerank=False)
        print("=== rerank ON (精排 12) ===")
        for i in range(N_RUNS):
            await run_once(rag, f"on-{i}", enable_rerank=True)
    finally:
        await rag.finalize_storages()


if __name__ == "__main__":
    asyncio.run(main())