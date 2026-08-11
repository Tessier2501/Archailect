"""决定性前置实验 (只读): 直接测嵌入通道对锚点 chunk 的余弦排名.

对应 IMPROVEMENTS.md 提案 8. 判定 chunk 向量召回通道死活:
- 锚点 chunk 余弦排名靠前但 LightRAG 不返回 -> 候选合并/截断层问题 (提案 9, 0 重建)
- 锚点 chunk 余弦排名靠后 -> embedding 对该叙事文本语义弱 (提案 10, 需重建)

只读: 加载 vdb_chunks.json 做余弦, 不改生产代码/索引.
运行: python -m scripts.probe_embedding_channel
成本: 4 次短 embed 调用 (受免费/付费 fallback 保护).
"""
from __future__ import annotations

import asyncio
import json
import math

from src.config import STORAGE_DIR, build_embedding_func

VDB = STORAGE_DIR / "rifters" / "rifters" / "vdb_chunks.json"

# 锚点 chunk 内容指纹 (Starfish 17269/17337/17345 区域)
ANCHOR_FP = ["chess and checkers", "biosphere or ßehemoth", "self-sustaining copy"]

QUERIES = {
    "Q_ORIG(反事实原问题)": (
        "为什么智能凝胶即使被编程为阻止ßehemoth，仍然未能把Lenie Clarke消灭在beebe站？"
    ),
    "静态问题派生": "Node 1211/BCC 1211 gel ßehemoth Behemoth Lenie Clarke Beebe station nuke prevent stop",
    "HyDE假想段(上轮生成)": (
        "Node 1211/BCC's targeting protocol was hyper-specific... "
        "it could not distinguish between a weapon and a trigger."
    ),
    "Q2对照(naive曾命中)": "在提及凝胶时，我特指 Node 1211/BCC 或 1211。",
}


def _cos(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


async def main() -> None:
    # 加载向量库 (nano-vectordb: data 含 __id__, matrix 为向量行)
    vdb = json.loads(VDB.read_text(encoding="utf-8"))
    data = vdb["data"]
    matrix = vdb["matrix"]
    dim = vdb.get("embedding_dim")
    print(f"vdb_chunks: {len(data)} 条, embedding_dim={dim}")

    # 需 chunk 文本判锚点; vdb 不一定存全文, 故另加载 kv_store_text_chunks 建 id->content
    kv = json.loads(
        (STORAGE_DIR / "rifters" / "rifters" / "kv_store_text_chunks.json")
        .read_text(encoding="utf-8")
    )
    id2content = {cid: v.get("content", "") for cid, v in kv.items()}

    embed = build_embedding_func()
    # 先嵌入全部查询
    q_vecs: dict[str, list[float]] = {}
    for tag, q in QUERIES.items():
        vecs = await embed.func([q])
        q_vecs[tag] = vecs[0]

    # 对每条查询算全部 chunk 余弦并排序
    for tag, qv in q_vecs.items():
        scored = []
        for row, vec in zip(data, matrix):
            cid = row.get("__id__", "")
            scored.append((_cos(qv, vec), cid))
        scored.sort(reverse=True)
        rank_of = {cid: i for i, (_, cid) in enumerate(scored)}

        # 找锚点 chunk (按内容指纹) 及其排名
        anchor_rows = []
        for cid, content in id2content.items():
            if any(fp in content for fp in ANCHOR_FP):
                anchor_rows.append(cid)
        top12 = {cid for _, cid in scored[:12]}
        hit_in_top12 = [c for c in anchor_rows if c in top12]

        print(f"\n[{tag}]")
        if not anchor_rows:
            print("  未在 kv_store_text_chunks 找到锚点 chunk (指纹失配, 需人工核对)")
        for cid in anchor_rows:
            r = rank_of.get(cid, -1)
            s = scored[r][0] if r >= 0 else float("nan")
            print(f"  锚点 {cid}: 余弦排名 {r + 1}/{len(scored)}, 分值 {s:.4f}")
        print(f"  naive top-12 是否含锚点: {'HIT ' + str(hit_in_top12) if hit_in_top12 else 'MISS'}")


if __name__ == "__main__":
    asyncio.run(main())