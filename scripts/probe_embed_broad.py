"""尽职调查实验 (只读): 判定"换 embedding 是否值得重建"的量化依据.

对应 IMPROVEMENTS.md 提案 12. 用户已确认: 4b/8b 维度显著更高 (必重建, 但
llm_response_cache 已备份, 重建仅耗 embedding token). 故聚焦剩余两问:
  1. 广度: 0.6b 是对全库普遍弱, 还是只对 Q1 决策段弱? (若仅局部弱, 换模型收益存疑)
  2. 对比: 4b/8b 对 Q1 锚点 chunk 的全库 top-12 命中是否显著优于 0.6b?
     方法: 现场重嵌全部 433 个 chunk 文本到目标模型向量空间, 同口径做全库 top-12 检索.
     这精确模拟"重建后该模型的真实召回", 是判定收益的决定性证据.

只读: 0.6b 用现 vdb 矩阵 (nano-vectordb 官方语义); 4b/8b 仅现场重嵌 chunk 文本,
      不重建索引, 不改生产代码/索引.
运行: python -m scripts.probe_embed_broad
成本: 0.6b 全库基线 12 次 embed; 对比试算每模型 433 chunk + 4 查询, 分批 embedding 调用.
      4b/8b 单 token 成本高于 0.6b, 但 433 chunk (~2.1MB 文本) 总量仍为小额一次性投入.
"""
from __future__ import annotations

import asyncio
import json
import os

import numpy as np
from nano_vectordb.dbs import buffer_string_to_array

from src.config import STORAGE_DIR, build_embedding_func

VDB_PATH = STORAGE_DIR / "rifters" / "rifters" / "vdb_chunks.json"
KV_PATH = STORAGE_DIR / "rifters" / "rifters" / "kv_store_text_chunks.json"

# 锚点 chunk 内容指纹 (Starfish 17269/17337/17345 区域)
ANCHOR_FP = ["chess and checkers", "biosphere or ßehemoth", "self-sustaining copy"]

# ---- 广度基线: 覆盖三卷多类问题的探针 (query, 期望命中的内容指纹) ----
PROBE_QUERIES = [
    ("人物-Lenie", "Who is Lenie Clarke and what are her implants?", ["Lenie Clarke", "implant"]),
    ("地点-Beebe", "What is Beebe station?", ["Beebe"]),
    ("事件-Darkness", "What happened during the Darkness?", ["Darkness"]),
    ("因果-1211决策", "Why did 1211 choose between biosphere and ßehemoth?", ["biosphere or ßehemoth"]),
    ("反事实-Q1", "为什么智能凝胶即使被编程为阻止ßehemoth，仍然未能把Lenie Clarke消灭在beebe站？", ["chess and checkers", "self-sustaining copy"]),
    ("实体-Behemoth", "What is Behemoth?", ["ßehemoth", "Behemoth"]),
    ("卷二-Behemoth书", "What happens in Behemoth (book 2)?", ["Behemoth"]),
    ("卷三-Maelstrom书", "What happens in Maelstrom (book 3)?", ["Maelstrom"]),
    ("技术-gel", "How does the smart gel work?", ["gel"]),
    ("角色-Scanlon", "Who is Scanlon?", ["Scanlon"]),
    ("设定-空间站", "What is the deep-sea station setting?", ["station"]),
    ("跨卷-三卷关系", "How are Starfish, Behemoth, and Maelstrom connected?", ["Starfish", "Behemoth", "Maelstrom"]),
]

# ---- Q1 对比试算查询 (与提案 8 同 4 条) ----
Q1_QUERIES = {
    "Q_ORIG": "为什么智能凝胶即使被编程为阻止ßehemoth，仍然未能把Lenie Clarke消灭在beebe站？",
    "静态": "Node 1211/BCC 1211 gel ßehemoth Behemoth Lenie Clarke Beebe station nuke prevent stop",
    "HyDE": "Node 1211/BCC's targeting protocol was hyper-specific, calibrated solely to the unique electromagnetic and metabolic signatures of Behemoth. it could not distinguish between a weapon and a trigger.",
    "Q2对照": "在提及凝胶时，我特指 Node 1211/BCC 或 1211。",
}

# 对比试算模型 (用户确认维度>1024; 调用失败则记录并跳过)
COMPARE_MODELS = ["qwen/qwen3-embedding-4b", "qwen/qwen3-embedding-8b"]
EMBED_BATCH = 32  # 与 .env EMBEDDING_BATCH_NUM 对齐

EMBED_URL = os.environ["EMBEDDING_BASE_URL"].rstrip("/") + "/embeddings"
EMBED_KEY = os.environ["EMBEDDING_API_KEY"]


def load_vdb():
    """官方语义加载 vdb_chunks.json -> (chunk_ids, L2 归一化矩阵)."""
    raw = json.loads(VDB_PATH.read_text(encoding="utf-8"))
    dim = raw["embedding_dim"]
    ids = [row["__id__"] for row in raw["data"]]
    matrix = buffer_string_to_array(raw["matrix"]).reshape(-1, dim)
    norm = np.linalg.norm(matrix, axis=1, keepdims=True)
    norm[norm == 0] = 1
    return ids, (matrix / norm).astype("float32")


def load_kv():
    kv = json.loads(KV_PATH.read_text(encoding="utf-8"))
    return {cid: v.get("content", "") for cid, v in kv.items()}


async def embed_direct(model: str, texts: list[str]) -> list[list[float]]:
    """直接调 embeddings 端点 (指定任意模型, 绕过 .env 固定模型), 分批."""
    import httpx
    out: list[list[float]] = []
    async with httpx.AsyncClient(timeout=180) as client:
        for i in range(0, len(texts), EMBED_BATCH):
            batch = texts[i : i + EMBED_BATCH]
            r = await client.post(
                EMBED_URL,
                headers={"Authorization": f"Bearer {EMBED_KEY}"},
                json={"model": model, "input": batch},
            )
            r.raise_for_status()
            data = r.json()["data"]
            data.sort(key=lambda d: d["index"])
            out.extend(d["embedding"] for d in data)
    return out


def _norm_rows(m):
    n = np.linalg.norm(m, axis=1, keepdims=True)
    n[n == 0] = 1
    return (m / n).astype("float32")


def top12_hit_anchor(matrix_norm, ids, id2content, anchor_ids, qvec):
    """全库 top-12 检索, 返回 (是否命中锚点, 各锚点排名)."""
    q = np.array(qvec, dtype="float32")
    q = q / (np.linalg.norm(q) or 1)
    sims = matrix_norm @ q
    order = sims.argsort()[::-1]
    top12 = {ids[i] for i in order[:12]}
    rank_of = {ids[i]: r + 1 for r, i in enumerate(order)}
    hit = any(a in top12 for a in anchor_ids)
    return hit, {a: rank_of.get(a, "-") for a in anchor_ids}


async def main() -> None:
    ids, matrix = load_vdb()
    id2content = load_kv()
    anchor_ids = [cid for cid, c in id2content.items() if any(fp in c for fp in ANCHOR_FP)]
    print(f"vdb: {len(ids)} chunks, dim={matrix.shape[1]}; 锚点 chunk: {anchor_ids}\n")

    embed_06b = build_embedding_func()

    # ============ 1. 广度基线: 0.6b 全库命中率 ============
    print("=== 1. 0.6b 全库广度基线 (12 探针) ===")
    hits = 0
    for tag, query, fps in PROBE_QUERIES:
        qv = (await embed_06b.func([query]))[0]
        q = np.array(qv, dtype="float32"); q = q / (np.linalg.norm(q) or 1)
        sims = matrix @ q
        top12_ids = [ids[i] for i in sims.argsort()[::-1][:12]]
        hit = any(any(fp in id2content.get(cid, "") for fp in fps) for cid in top12_ids)
        hits += hit
        print(f"  [{'HIT' if hit else 'MISS'}] {tag}: {query[:50]!r}")
    rate = hits / len(PROBE_QUERIES)
    print(f"  -> 0.6b 全库命中率: {hits}/{len(PROBE_QUERIES)} = {rate:.0%}\n")

    # ============ 2. Q1 锚点全库 top-12 对比: 0.6b (现库) ============
    print("=== 2. Q1 锚点全库 top-12 对比 ===")
    print("\n  [0.6b 现库]")
    for qtag, query in Q1_QUERIES.items():
        qv = (await embed_06b.func([query]))[0]
        hit, ranks = top12_hit_anchor(matrix, ids, id2content, anchor_ids, qv)
        rk = ", ".join(f"{a.split('-')[-1]}=#{r}" for a, r in ranks.items())
        print(f"    [{'HIT' if hit else 'MISS'}] {qtag}: {rk}")

    # 广度基线数据 (供 4b/8b 同口径对比, 若需)
    probe_texts = [q for _, q, _ in PROBE_QUERIES]

    # ============ 2b. 4b/8b: 现场重嵌全部 chunk 文本后同口径对比 ============
    ordered_ids = list(id2content.keys())
    chunk_texts = [id2content[cid] for cid in ordered_ids]
    for m in COMPARE_MODELS:
        print(f"\n  [{m}] 现场重嵌 {len(chunk_texts)} chunks ...")
        try:
            chunk_vecs = await embed_direct(m, chunk_texts)
            mmat = _norm_rows(np.array(chunk_vecs, dtype="float32"))
            # 重排顺序与 ordered_ids 对齐; 锚点 id 在同空间
            for qtag, query in Q1_QUERIES.items():
                qv = (await embed_direct(m, [query]))[0]
                hit, ranks = top12_hit_anchor(mmat, ordered_ids, id2content, anchor_ids, qv)
                rk = ", ".join(f"{a.split('-')[-1]}=#{r}" for a, r in ranks.items())
                print(f"    [{'HIT' if hit else 'MISS'}] {qtag}: {rk}")
        except Exception as e:
            print(f"    调用失败 {type(e).__name__}: {e}")

    print("\n=== 判定提示 ===")
    print(f"广度命中率 {rate:.0%}: <50% -> 0.6b 普遍弱, 换模型收益大; 高 -> 仅局部弱, 收益存疑.")
    print("4b/8b 的 Q_ORIG/HyDE 若 top-12 HIT 而 0.6b MISS -> 换模型决定性有效, 值得重建.")
    print("若 4b/8b 也 MISS -> 换 embedding 收益有限, 转 BM25 稀疏混合召回 (提案 10c).")


if __name__ == "__main__":
    asyncio.run(main())
