"""实验 (只读, 不改生产代码/索引): 检索串扩展对 Q1 反事实问题的召回改善验证.

对应 IMPROVEMENTS.md 提案 1 (经 IDE 评审修正, 2026-08-11):

疑点 A 修正 (自答式泄露):
- 静态扩展臂只允许含"问题中已出现的词元或其直译" (实体名/内容名词, 如 凝胶→gel).
  严禁答案侧词汇 (copy/move/decide/biosphere/self-sustaining 等) — 否则实验自证有效,
  且生产端无答案可抄, 不可泛化.
- "自答式"检索的通用化形态 = 运行时由 LLM 从问题现场生成 (查询改写 / HyDE 假想文档):
  生成时 LLM 只见问题不见语料, 即使偶合答案词也是生产可复现行为, 不构成泄露.
  故增设 LLM 改写 / HyDE 两条臂, 验证该机制本身; 生成串全文打印供人工泄露审查.

疑点 B 修正 (基线口径):
- 基线 = 本脚本内 Q_ORIG 臂, 与所有扩展臂同一进程、同一生产配置
  (mix/top_k=20/chunk_top_k=12/rerank 随 .env), 同口径对照.
- scripts/diagnose_q1_raw.py (hybrid/top_k=50) 不再作基线, 仅作候选池上限核查.

运行: python -m scripts.probe_query_expansion
成本: mix 模式关键词提取走 LLM (受 llm_response_cache 保护);
      LLM 改写/HyDE 各 1 次短调用 (直接调 API, 不入 LightRAG 缓存, 重跑会再计费, 量级数百 token).
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

# ---- 基线: 用户原始问题 (澄清句+反事实合并, 与 PLAN §5.1 用户实测一致) ----
Q_ORIG = (
    "在提及凝胶时，我特指“Node 1211/BCC”，或“1211”。"
    "为什么智能凝胶即使被编程为阻止ßehemoth，仍然未能把Lenie Clarke消灭在beebe站？"
    "在她调查海底核弹的时候，将核弹引爆，不就能有效地遏制ßehemoth吗？"
)

# ---- 静态扩展臂: 仅问题派生词元 (字面或直译), 无答案侧词汇 ----
# 词元来源核对: Node 1211/BCC/1211/ßehemoth/Lenie Clarke/beebe 站 = 问题字面;
# gel←凝胶, nuke←核弹, Behemoth←ßehemoth 变体, prevent/stop←阻止. 均非答案词.
Q_EXT1 = (
    "Node 1211/BCC 1211 gel ßehemoth Behemoth Lenie Clarke Beebe station "
    "nuke prevent stop"
)

# ---- LLM 运行时生成臂 (改写 / HyDE), 生成时只见问题不见语料 ----
REWRITE_SYS = (
    "You rewrite a user's question about a novel into 2-3 short declarative English "
    "statements, phrased as they might literally appear in the novel's text, to improve "
    "passage retrieval. Do NOT answer the question. Do NOT explain. Output only the statements."
)
HYDE_SYS = (
    "Write one short passage (3-5 sentences, English, in the style of the source novel) "
    "that would plausibly contain the answer to the user's question. Invent specifics as "
    "needed. Do NOT explain. Output only the passage."
)

# 答案锚点 (Starfish 17269-17357), 与 scripts/diagnose_q1_raw.py 一致
ANCHORS = {
    "identity_17269": "chess and checkers",
    "choice_17337": "biosphere or ßehemoth",
    "answer_17345": "self-sustaining copy",
}


async def run_query(rag: LightRAG, tag: str, query: str, rerank_on: bool) -> dict:
    """以生产配置检索一条查询串, 报告锚点命中. 各臂同口径."""
    param = QueryParam(
        mode="mix",
        top_k=20,
        chunk_top_k=12,
        enable_rerank=rerank_on,
        conversation_history=[],
    )
    result = await rag.aquery_data(query, param=param)
    chunks = (
        result.get("data", {}).get("chunks", [])
        if result.get("status") == "success"
        else []
    )
    hits: dict[str, bool] = {k: False for k in ANCHORS}
    for c in chunks:
        content = c.get("content", "")
        for k, anchor in ANCHORS.items():
            if anchor in content:
                hits[k] = True
    n_hit = sum(hits.values())
    print(
        f"  [{tag:<22}] n_chunks={len(chunks):<3} 锚点命中={n_hit}/3 "
        f"({', '.join(k for k, v in hits.items() if v) or 'NONE'})"
    )
    return {"tag": tag, "n_chunks": len(chunks), "hits": hits, "n_hit": n_hit}


async def main() -> None:
    # ---- 1. LLM 运行时生成检索串 (改写 / HyDE), 先于检索以便全文打印审查 ----
    llm = build_llm_func()
    print("=== LLM 生成检索串 (人工泄露审查: 静态臂禁答案词; LLM 臂为运行时生成, 不算泄露) ===")
    rewrite = await llm(Q_ORIG, system_prompt=REWRITE_SYS)
    print(f"\n[LLM 改写串]\n{rewrite}\n")
    hyde = await llm(Q_ORIG, system_prompt=HYDE_SYS)
    print(f"[HyDE 假想段落]\n{hyde}\n")

    # ---- 2. 实例化 LightRAG (与生产同参数) ----
    kwargs: dict = {
        "working_dir": str(STORAGE_DIR / "rifters"),
        "workspace": "rifters",
        "llm_model_func": llm,
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
    rerank_on = rerank_func is not None
    try:
        print(f"=== 检索对照 (生产配置: mix/top_k=20/chunk_top_k=12/rerank={'on' if rerank_on else 'off'}) ===")
        print(f"锚点: {list(ANCHORS)}\n")
        arms = [
            ("Q_ORIG(基线)", Q_ORIG),
            ("Q_EXT1_静态问题派生", Q_EXT1),
            ("Q_LLM_改写", rewrite),
            ("Q_LLM_HYDE", hyde),
        ]
        results = []
        for tag, q in arms:
            results.append(await run_query(rag, tag, q, rerank_on))

        # ---- 3. 汇总判定 ----
        base = results[0]["n_hit"]
        static = results[1]["n_hit"]
        llm_best = max(results[2]["n_hit"], results[3]["n_hit"])
        print("\n=== 汇总 ===")
        print(f"基线 Q_ORIG: {base}/3 | 静态扩展: {static}/3 | LLM 臂最优: {llm_best}/3")
        if base >= 2:
            print("结论: 原串已召回 -> 当前生产或已足够, 提案 1 边际收益低.")
        elif static > base or llm_best > base:
            best = []
            if static > base:
                best.append(f"静态臂({static}/3, 零成本)")
            if llm_best > base:
                best.append(f"LLM 臂({llm_best}/3, 每查询+1 次短调用)")
            print(f"结论: 扩展补齐召回 -> 有效通道: {', '.join(best)}. 建议接入生产 (静态优先).")
        else:
            print("结论: 全臂 MISS -> 提案 1 整体无效, 转向语料/embedding 层方案 (需重议).")
    finally:
        await rag.finalize_storages()


if __name__ == "__main__":
    asyncio.run(main())
