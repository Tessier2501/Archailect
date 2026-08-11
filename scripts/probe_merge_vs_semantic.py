"""决定性对照 (只读): 裁决"合并层淹没"还是"embedding 语义弱".

背景 (2026-08-11 第六轮 IDE 发现的矛盾): 同一 0.6b 模型, HyDE 陈述串在
- 第四轮 probe_query_expansion (LightRAG mix 完整链路): 0 锚点
- 第六轮 probe_embed_broad (纯向量余弦): chunk-104 = #10 (HIT)
但两次 HyDE 串文本不同 (第四轮为运行时生成完整段, 第六轮为拼接版), 无法直接归因.
本实验固定同一 HyDE 串, 同进程三通道对照, 定位 chunk-104 丢失环节.

对照臂 (同一 HyDE 串, 同 0.6b):
  A. 纯向量余弦 (绕过 LightRAG, 直接对 vdb 矩阵算余弦 top-12)
  B. aquery_data naive (LightRAG 纯向量检索, 无实体/关系合并)
  C. aquery_data mix   (LightRAG 生产链路, 实体+关系+向量合并)

判定:
  A HIT 且 B HIT 且 C MISS -> chunk-104 在 mix 合并/截断丢失 = 合并层淹没 (提案 9 方向,
                              对症 = 查询改写进生产 + 混合召回/合并层调权, 0 重建)
  A HIT 且 B MISS          -> 丢失发生在 LightRAG naive 检索内部 (索引/参数), 非合并层
  A MISS                   -> 纯向量也找不到 = embedding 语义弱 (维持提案 10 换模型方向)

只读: A 读 vdb 矩阵; B/C 走 aquery_data (不做 LLM 生成, 仅检索), 不改生产代码/索引.
运行: python -m scripts.probe_merge_vs_semantic
成本: B/C 各若干次检索 (mix 关键词提取受 llm_response_cache 保护); A 无 LLM 调用.
"""
from __future__ import annotations

import asyncio
import json

import numpy as np
from nano_vectordb.dbs import buffer_string_to_array

from lightrag import LightRAG, QueryParam

from src.config import (
    STORAGE_DIR,
    build_embedding_func,
    build_llm_func,
    build_rerank_func,
    build_role_llm_configs,
)

VDB_PATH = STORAGE_DIR / "rifters" / "rifters" / "vdb_chunks.json"
KV_PATH = STORAGE_DIR / "rifters" / "rifters" / "kv_store_text_chunks.json"
ANCHOR_FP = ["chess and checkers", "biosphere or ßehemoth", "self-sustaining copy"]

# 固定同一 HyDE 串 (取第六轮 probe_embed_broad 的拼接版, 保证与纯向量 HIT 结果同口径)
HYDE = (
    "Node 1211/BCC's targeting protocol was hyper-specific, calibrated solely to the "
    "unique electromagnetic and metabolic signatures of Behemoth. it could not distinguish "
    "between a weapon and a trigger."
)


def load_vdb():
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


def anchor_ranks_in_list(chunk_ids_in_order, anchor_ids):
    """给定有序 chunk id 列表, 返回锚点名次与是否进 top-12."""
    rank_of = {cid: i + 1 for i, cid in enumerate(chunk_ids_in_order)}
    ranks = {a: rank_of.get(a, "-") for a in anchor_ids}
    top12 = set(chunk_ids_in_order[:12])
    hit = any(a in top12 for a in anchor_ids)
    return hit, ranks


async def main() -> None:
    ids, matrix = load_vdb()
    id2content = load_kv()
    anchor_ids = [cid for cid, c in id2content.items() if any(fp in c for fp in ANCHOR_FP)]
    print(f"vdb: {len(ids)} chunks, dim={matrix.shape[1]}; 锚点: {anchor_ids}")
    print(f"固定 HyDE 串: {HYDE!r}\n")

    embed = build_embedding_func()

    # ---- A. 纯向量余弦 (绕过 LightRAG) ----
    qv = (await embed.func([HYDE]))[0]
    q = np.array(qv, dtype="float32"); q = q / (np.linalg.norm(q) or 1)
    sims = matrix @ q
    order = sims.argsort()[::-1]
    a_ids_ordered = [ids[i] for i in order]
    a_hit, a_ranks = anchor_ranks_in_list(a_ids_ordered, anchor_ids)
    print(f"[A 纯向量余弦]   {'HIT' if a_hit else 'MISS'} 锚点名次: {a_ranks}")

    # ---- 实例化 LightRAG (B/C 共用) ----
    kwargs: dict = {
        "working_dir": str(STORAGE_DIR / "rifters"),
        "workspace": "rifters",
        "llm_model_func": build_llm_func(),
        "embedding_func": embed,
    }
    rerank_func = build_rerank_func()
    if rerank_func is not None:
        kwargs["rerank_model_func"] = rerank_func
    role_configs = build_role_llm_configs()
    if role_configs:
        kwargs["role_llm_configs"] = role_configs
    rag = LightRAG(**kwargs)
    await rag.initialize_storages()

    async def via_aquery(mode: str, enable_rerank: bool):
        param = QueryParam(
            mode=mode, top_k=20, chunk_top_k=12,
            enable_rerank=enable_rerank, conversation_history=[],
        )
        result = await rag.aquery_data(HYDE, param=param)
        chunks = result.get("data", {}).get("chunks", []) if result.get("status") == "success" else []
        # aquery_data 返回 chunk 列表; 用内容反查 chunk_id
        contents = [c.get("content", "") for c in chunks]
        content2id = {}
        for cid, c in id2content.items():
            content2id.setdefault(c, cid)
        ordered = [content2id.get(ct, "?") for ct in contents]
        hit = any(any(fp in ct for fp in ANCHOR_FP) for ct in contents)
        return hit, ordered, len(chunks)

    try:
        # ---- B. naive (LightRAG 纯向量, 无合并) ----
        b_hit, b_ordered, b_n = await via_aquery("naive", enable_rerank=False)
        b_hit_ids, b_ranks = anchor_ranks_in_list(b_ordered, anchor_ids)
        print(f"[B naive 检索]    {'HIT' if b_hit else 'MISS'} (n={b_n}) 锚点名次: {b_ranks}")

        # ---- C. mix 生产链路 (rerank 与生产一致) ----
        c_hit, c_ordered, c_n = await via_aquery("mix", enable_rerank=rerank_func is not None)
        c_hit_ids, c_ranks = anchor_ranks_in_list(c_ordered, anchor_ids)
        print(f"[C mix 生产链路]  {'HIT' if c_hit else 'MISS'} (n={c_n}) 锚点名次: {c_ranks}")
    finally:
        await rag.finalize_storages()

    # ---- 判定 ----
    print("\n=== 判定 ===")
    print(f"A 纯向量={'HIT' if a_hit else 'MISS'}  B naive={'HIT' if b_hit else 'MISS'}  C mix={'HIT' if c_hit else 'MISS'}")
    if a_hit and b_hit and not c_hit:
        print("-> chunk-104 在 mix 合并/截断丢失 = 合并层淹没 (提案 9 方向).")
        print("   对症: 查询改写进生产 (HyDE/陈述化) + 混合召回/合并层调权, 0 重建.")
    elif a_hit and not b_hit:
        print("-> 纯向量能找但 naive 检索丢 = LightRAG 检索内部 (索引/参数), 非合并层. 需查 naive 通道.")
    elif not a_hit:
        print("-> 纯向量也找不到 = embedding 语义弱, 维持换 embedding 方向 (提案 10).")
    elif a_hit and b_hit and c_hit:
        print("-> 三通道均 HIT: 此前 mix MISS 或由 HyDE 串差异所致. 生产端补查询改写即可.")
    else:
        print("-> 组合异常, 把完整输出发回裁决.")


if __name__ == "__main__":
    asyncio.run(main())
