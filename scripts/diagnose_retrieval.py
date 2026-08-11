"""一键诊断: 验证 LightRAG 检索失败根因 (只读, 不改任何生产代码/索引).

运行: python -m scripts.diagnose_retrieval
输出: scripts/diagnose_report.json + 终端可读汇总表.

诊断矩阵:
  - 4 组问题 (Q1 反事实 / Q2 孤立澄清 / Q3 简单控制 / Q4 多轮指代)
  - 3 种 mode (hybrid / naive / mix)
  - 2 种关键词变体 (V1 走 LLM 提取 = 现状基线 / V2 显式注入别名 = 实体归一化候选)
  - Q4 额外拆 4 变体: conversation_history 有无 × 别名注入有无
总计 22 次 aquery_data 调用. aquery_data 不做 LLM 回答生成;
V2 注入关键词会跳过 LLM 提取 (lightrag/operate.py:4437), 故额外 LLM 成本 = V1 各问题的关键词提取 (受 20MB llm_response_cache 保护).
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Literal

from lightrag import LightRAG, QueryParam

from src.config import (
    STORAGE_DIR,
    build_embedding_func,
    build_llm_func,
    build_role_llm_configs,
)

# ---------------- 配置 ----------------
MODEL = "rifters"
REPORT_PATH = Path(__file__).resolve().parent / "diagnose_report.json"

# 答案锚点: 每个特征串均在原文唯一, 用于判定召回 chunk 是否命中关键段落
ANSWER_ANCHORS = {
    "node1211_identity": "chess and checkers",          # 17269: Node 1211 身份
    "node1211_darkness": "since the Darkness",          # 17309: 1211 的转变背景
    "behemoth_choice": "biosphere or \u00dfehemoth, but not both",  # 17337: 二选一
    "behemoth_answer": "move a self-sustaining copy of \u00dfehemoth",  # 17345: 案例1答案
    "behemoth_logic": "the correct choice of output signals",  # 17333
}

# 同一实体的拼写变体族 (图谱归一化失败的证据)
VARIANT_FAMILIES = {
    "behemoth": ["behemoth", "\u00dfehemoth", "ehemoth", "\u00dfebehemoth", "\u00dfhehemoth"],
    "node1211": ["1211", "node 1211"],
}

# 别名注入 (V2): 显式传给 hl_keywords, 会绕过 LLM 提取 (lightrag/operate.py 4437)
ALIAS_V2 = ["Node 1211/BCC", "1211", "Behemoth", "\u00dfehemoth", "Lenie Clarke", "Beebe station"]

# 诊断矩阵使用的检索模式 (带 Literal 标注, 满足 QueryParam.mode 类型约束)
_MODES: tuple[Literal["hybrid", "naive", "mix"], ...] = ("hybrid", "naive", "mix")

# 问题组
Q1 = "为什么智能凝胶 Node 1211/BCC 即使被编程为阻止 Behemoth，仍然未能把 Lenie Clarke 消灭在 Beebe 站？在她调查海底核弹的时候，将核弹引爆，不就能有效地遏制 Behemoth 吗？"
Q2 = "在提及凝胶时，我特指 Node 1211/BCC 或 1211。"
Q3 = "What is Behemoth?"
Q4A = "Node 1211 是什么？"
Q4B = "它为什么没消灭 Lenie Clarke？"
Q4_HISTORY = [
    {"role": "user", "content": Q4A},
]


def _match_variant(entity_name: str) -> str | None:
    """返回实体名所属的变体族名, 不在族内返回 None."""
    lower = entity_name.lower()
    for family, variants in VARIANT_FAMILIES.items():
        if any(v in lower for v in variants):
            return family
    return None


def _count_variants_in(entity_names: list[str]) -> dict[str, dict[str, int]]:
    """统计召回到的实体在各变体族的命中次数 (用于评估归一化缺失的影响)."""
    result: dict[str, dict[str, int]] = {}
    for name in entity_names:
        fam = _match_variant(name)
        if fam:
            result.setdefault(fam, {}).setdefault(name, 0)
            result[fam][name] += 1
    return result


def _analyze_hit(chunks: list[dict]) -> dict:
    """按答案锚点判定 HIT / PARTIAL / MISS, 并统计跨书噪声."""
    hit_anchors = {"HIT": [], "PARTIAL": []}
    starfish_files = {"starfish"}
    total = len(chunks)
    cross_book = 0
    for chunk in chunks:
        content = chunk.get("content", "")
        fp = chunk.get("file_path", "")
        if fp and "starfish" not in fp.lower():
            cross_book += 1
        matched = [k for k, anchor in ANSWER_ANCHORS.items() if anchor in content]
        if matched:
            hit_anchors["HIT"].extend(matched)
        # 含 ßehemoth 或 Behemoth 主族即视为部分相关
        if not matched and any(v in content for v in VARIANT_FAMILIES["behemoth"]):
            hit_anchors["PARTIAL"].append("behemoth_mention")
    status = "HIT" if hit_anchors["HIT"] else ("PARTIAL" if hit_anchors["PARTIAL"] else "MISS")
    return {
        "status": status,
        "hit_anchors": sorted(set(hit_anchors["HIT"])),
        "partial_anchors": sorted(set(hit_anchors["PARTIAL"])),
        "chunks_total": total,
        "cross_book_chunks": cross_book,
        "cross_book_ratio": round(cross_book / total, 3) if total else 0,
    }


async def run_case(
    rag: LightRAG,
    tag: str,
    query: str,
    mode: Literal["local", "global", "hybrid", "naive", "mix", "bypass"],
    inject_aliases: bool,
    history: list[dict[str, str]] | None = None,
) -> dict:
    param = QueryParam(
        mode=mode,
        top_k=12,
        chunk_top_k=8,
        conversation_history=history or [],
        hl_keywords=list(ALIAS_V2) if inject_aliases else [],
        ll_keywords=[] if inject_aliases else [],
    )
    result = await rag.aquery_data(query, param=param)
    data = result.get("data", {}) if result.get("status") == "success" else {}
    metadata = result.get("metadata", {})

    entities = data.get("entities", [])
    relationships = data.get("relationships", [])
    chunks = data.get("chunks", [])
    entity_names = [e.get("entity_name", "") for e in entities]
    rel_names = []
    for r in relationships:
        rel_names.extend([r.get("src_id", ""), r.get("tgt_id", "")])

    return {
        "tag": tag,
        "mode": mode,
        "inject_aliases": inject_aliases,
        "history": bool(history),
        "query": query[:80],
        "status": result.get("status", "failure"),
        "message": result.get("message", ""),
        "llm_keywords": metadata.get("keywords", {}),
        "entity_variant_hits": _count_variants_in(entity_names),
        "relationship_variant_hits": _count_variants_in(rel_names),
        "n_entities": len(entities),
        "n_relationships": len(relationships),
        "hit": _analyze_hit(chunks),
        "sample_chunk_previews": [c.get("content", "")[:120].replace("\n", " ") for c in chunks[:3]],
    }


async def main() -> None:
    # 与 api_server/builder 完全一致: workspace=model, 共享内存缓存按 (namespace, workspace) 隔离
    kwargs: dict = {
        "working_dir": str(STORAGE_DIR / MODEL),
        "workspace": MODEL,
        "llm_model_func": build_llm_func(),
        "embedding_func": build_embedding_func(),
    }
    role_configs = build_role_llm_configs()
    if role_configs:
        kwargs["role_llm_configs"] = role_configs
    rag = LightRAG(**kwargs)
    await rag.initialize_storages()

    cases: list[dict] = []
    try:
        # Q1/Q2/Q3: 3 mode × 2 变体 (V1 走 LLM / V2 注入别名)
        for q_tag, query in [("Q1", Q1), ("Q2", Q2), ("Q3", Q3)]:
            for mode in _MODES:
                cases.append(await run_case(rag, f"{q_tag}-{mode}-V1", query, mode, inject_aliases=False))
                cases.append(await run_case(rag, f"{q_tag}-{mode}-V2", query, mode, inject_aliases=True))

        # Q4 多轮: 4 变体 (历史有无 × 别名有无), 只测 hybrid
        cases.append(await run_case(rag, "Q4b-hybrid-V1a", Q4B, "hybrid", inject_aliases=False, history=Q4_HISTORY))
        cases.append(await run_case(rag, "Q4b-hybrid-V1b", Q4B, "hybrid", inject_aliases=False, history=None))
        cases.append(await run_case(rag, "Q4b-hybrid-V2a", Q4B, "hybrid", inject_aliases=True, history=Q4_HISTORY))
        cases.append(await run_case(rag, "Q4b-hybrid-V2b", Q4B, "hybrid", inject_aliases=True, history=None))
    finally:
        await rag.finalize_storages()

    report = {"model": MODEL, "n_cases": len(cases), "cases": cases}
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------- 终端汇总 ----------
    print(f"\n{'='*100}\n诊断报告: {MODEL} ({len(cases)} 次检索)\n{'='*100}")
    print(f"{'case':<24} {'mode':<7} {'V1/V2':<6} {'hist':<5} {'stat':<8} {'anchors':<50} {'xbook'}")
    print("-" * 100)
    for c in cases:
        hit = c["hit"]
        anchors = ",".join(hit["hit_anchors"] + hit["partial_anchors"]) or "-"
        print(
            f"{c['tag']:<24} {c['mode']:<7} {'V2' if c['inject_aliases'] else 'V1':<6} "
            f"{'Y' if c['history'] else 'N':<5} {hit['status']:<8} {anchors[:48]:<50} "
            f"{hit['cross_book_ratio']}"
        )
    print("-" * 100)

    # ---------- 汇总统计 ----------
    for q_prefix in ["Q1", "Q2", "Q3"]:
        sub = [c for c in cases if c["tag"].startswith(q_prefix)]
        hit_v1 = sum(1 for c in sub if not c["inject_aliases"] and c["hit"]["status"] == "HIT")
        hit_v2 = sum(1 for c in sub if c["inject_aliases"] and c["hit"]["status"] == "HIT")
        miss_v1 = sum(1 for c in sub if not c["inject_aliases"] and c["hit"]["status"] == "MISS")
        miss_v2 = sum(1 for c in sub if c["inject_aliases"] and c["hit"]["status"] == "MISS")
        print(
            f"\n{q_prefix}: V1 HIT={hit_v1}/3 MISS={miss_v1}/3 | "
            f"V2 HIT={hit_v2}/3 MISS={miss_v2}/3"
        )

    by_mode = {}
    for mode in _MODES:
        sub = [c for c in cases if c["mode"] == mode]
        hit = sum(1 for c in sub if c["hit"]["status"] == "HIT")
        miss = sum(1 for c in sub if c["hit"]["status"] == "MISS")
        by_mode[mode] = f"HIT={hit} MISS={miss}"
    print(f"\n按 mode: {by_mode}")

    q4 = [c for c in cases if c["tag"].startswith("Q4")]
    print("\nQ4 多轮对比 (hybrid):")
    for c in q4:
        print(f"  {c['tag']}: hist={c['history']} alias={'V2' if c['inject_aliases'] else 'V1'} -> {c['hit']['status']}")

    print(f"\n完整 JSON 报告: {REPORT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())