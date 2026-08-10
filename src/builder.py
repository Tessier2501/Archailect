"""离线建图: python -m src.builder --txt data/xxx.txt [--txt data/yyy.txt ...] --book book-name"""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from lightrag import LightRAG

from src.config import (
    STORAGE_DIR,
    build_embedding_func,
    build_llm_func,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="用 txt 书籍构建 LightRAG 知识库; 可传入多个 --txt 依次插入同一 book (如同一系列的连续多卷)"
    )
    parser.add_argument(
        "--txt",
        required=True,
        action="append",
        type=Path,
        help="txt 书籍文件路径; 可多次传入 (如一个系列的每一卷各一次)",
    )
    parser.add_argument("--book", required=True, help="书籍标识名, 如 rifters")
    parser.add_argument(
        "--max-file-size-mb",
        type=int,
        default=200,
        help="单文件大小上限 (MB), 默认 200. 超大文本须先切片再分批处理 (见 plan.md 风险 #13).",
    )
    return parser.parse_args()


def build_rag(book: str) -> LightRAG:
    """为单本书构造独立 LightRAG 实例. working_dir 指向 storage/{book}.

    workspace=book: 多库共享内存缓存按 (namespace, workspace) 寻址,
    不传时所有实例 workspace 均为 "", 内存缓存互相覆盖导致串台 (已实测).
    注意: workspace 非空会使存储文件路径变为 storage/{book}/{book}/.
    建图 (builder) 与查询 (api_server) 必须传相同 workspace, 保持一致.
    """
    return LightRAG(
        working_dir=str(STORAGE_DIR / book),
        workspace=book,
        llm_model_func=build_llm_func(),
        embedding_func=build_embedding_func(),
    )


async def main() -> None:
    args = parse_args()

    # 大文件防护: 一次性 read_text 载入整个文件, 超大文本引发内存峰值.
    # 超限即明确报错 (防呆), 避免不知情下 OOM.
    for txt in args.txt:
        if not txt.is_file():
            raise FileNotFoundError(f"书籍文件不存在: {txt}")
        file_size_mb = txt.stat().st_size / (1024 * 1024)
        if file_size_mb > args.max_file_size_mb:
            raise RuntimeError(
                f"书籍文件过大: {txt.name} {file_size_mb:.1f} MB > 上限 {args.max_file_size_mb} MB. "
                "超大文本须先切片, 再分批插入, 见 plan.md 风险 #13."
            )

    rag = build_rag(args.book)
    await rag.initialize_storages()
    try:
        for txt in args.txt:
            # 显式 utf-8 (双保险)
            content = txt.read_text(encoding="utf-8")
            if not content.strip():
                raise ValueError(f"书籍文件为空: {txt}")
            # 1.5.6 签名: split_by_character 可控制分块; 默认即可
            # 同一 rag/storage 依次插入, 多卷合并为一个知识库
            await rag.ainsert(content)
            print(f"[OK] 已插入: {txt.name}")
    finally:
        await rag.finalize_storages()

    print(f"[OK] 完成: {args.book} -> {STORAGE_DIR / args.book}")


if __name__ == "__main__":
    asyncio.run(main())