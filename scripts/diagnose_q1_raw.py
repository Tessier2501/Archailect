"""诊断 (2026-08-11): 验证 Q1 反事实问题的答案锚点是否在【原始检索候选池】中.

目的: 判定 "最终 chunks MISS" 的根因是
  (a) 原始检索 (top_k=20) 就没召回锚点 → 召回层失败, 后续 rerank/截断无从谈起
  (b) 原始检索召回了锚点, 但被后续处理 (unified chunk 合并/截断) 挤掉 → 需调整 chunk_top_k 等
依据: rerank on/off ×3 最终 chunks 全 MISS, 已排除 rerank 挤出.

用 aquery_data 只拿结构化结果不做 LLM 生成; hybrid 模式走关键词提取 (受 llm cache 保护).
"""
from __future__ import annotations

import asyncio

from lightrag import LightRAG, QueryParam

from src.config import (
    STORAGE_DIR,
    build_embedding_func,
    build_llm_func,
    build_role_llm_configs,
)

Q1_SIMPLE = (
    "为什么智能凝胶 Node 1211/BCC 即使被编程为阻止 Behemoth，"
    "仍然未能把 Lenie Clarke 消灭在 Beebe 站？"
)

ANCHORS = {
    "identity_17269": "chess and checkers",
    "choice_17337": "biosphere or \u00dfehemoth",
    "answer_17345": "self-sustaining copy",
}


async def main() -> None:
    kwargs: dict = {
        "working_dir": str(STORAGE_DIR / "rifters"),
        "workspace": "rifters",
        "llm_model_func": build_llm_func(),
        "embedding_func": build_embedding_func(),
    }
    role_configs = build_role_llm_configs()
    if role_configs:
        kwargs["role_llm_configs"] = role_configs

    rag = LightRAG(**kwargs)
    await rag.initialize_storages()
    try:
        # 用较大 top_k 观察原始候选池规模; enable_rerank=False 排除精排影响
        param = QueryParam(
            mode="hybrid",
            top_k=50,
            chunk_top_k=50,
            enable_rerank=False,
            conversation_history=[],
        )
        result = await rag.aquery_data(Q1_SIMPLE, param=param)
        data = result.get("data", {}) if result.get("status") == "success" else {}
        chunks = data.get("chunks", [])
        entities = data.get("entities", [])

        print("status:", result.get("status"))
        print("n_entities:", len(entities))
        print("n_chunks (top_k=50):", len(chunks))
        print()
        print("--- 锚点命中检查 ---")
        found: dict[str, bool] = {k: False for k in ANCHORS}
        for c in chunks:
            content = c.get("content", "")
            for k, anchor in ANCHORS.items():
                if anchor in content:
                    found[k] = True
                    print(f"  {k}: HIT (chunk 含 '{anchor}')")
        for k, hit in found.items():
            if not hit:
                print(f"  {k}: MISS")
        print()
        print("--- 实体名 (前 15) ---")
        for e in entities[:15]:
            print("  ", e.get("entity_name"), "|", e.get("entity_type"))
        print()
        print("--- chunk 是否含与 1211/Behemoth 相关的线索 ---")
        clue_names = ["1211", "behemoth", "\u00dfehemoth", "scanlon", "checkers", "gel"]
        for i, c in enumerate(chunks):
            content = c.get("content", "")
            hits = [n for n in clue_names if n in content.lower()]
            if hits:
                print(f"  [chunk {i}] 含线索 {hits}: {content[:100]!r}")
    finally:
        await rag.finalize_storages()


if __name__ == "__main__":
    asyncio.run(main())