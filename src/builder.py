"""离线建图: python -m src.builder --txt data/xxx.txt --book novel-three-body"""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from lightrag import LightRAG
from lightrag.llm.openai import openai_complete_if_cache

from src.config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    STORAGE_DIR,
    build_embedding_func,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="用 txt 书籍构建 LightRAG 知识库")
    parser.add_argument("--txt", required=True, type=Path, help="txt 书籍文件路径")
    parser.add_argument("--book", required=True, help="书籍标识名, 如 novel-three-body")
    parser.add_argument(
        "--max-file-size-mb",
        type=int,
        default=200,
        help="单文件大小上限 (MB), 默认 200。超大文本须先切片再分批 ainsert (见 plan.md §8 风险 #13)。",
    )
    return parser.parse_args()


def build_rag(book: str) -> LightRAG:
    """为单本书构造独立 LightRAG 实例。working_dir 指向 storage/{book}。"""
    return LightRAG(
        working_dir=str(STORAGE_DIR / book),
        llm_model_func=openai_complete_if_cache,
        llm_model_name=DEEPSEEK_MODEL,
        llm_model_kwargs={
            "base_url": DEEPSEEK_BASE_URL,
            "api_key": DEEPSEEK_API_KEY,
        },
        embedding_func=build_embedding_func(),
    )


async def main() -> None:
    args = parse_args()
    if not args.txt.is_file():
        raise FileNotFoundError(f"书籍文件不存在: {args.txt}")

    # 大文件防护: 一次性 read_text() 会把整个文件载入内存,
    # 超大文本会引发内存峰值, 超限即明确报错 (防呆),
    # 避免在不知情的情况下触发 OOM。
    file_size_mb = args.txt.stat().st_size / (1024 * 1024)
    if file_size_mb > args.max_file_size_mb:
        raise RuntimeError(
            f"书籍文件过大: {file_size_mb:.1f} MB > 上限 {args.max_file_size_mb} MB。"
            "超大文本须先切片, 再对每个切片分别调用 ainsert (分批构建), 见 plan.md §8 风险 #13。"
        )

    # 书籍内容读取: 显式指定 utf-8 (双保险)
    content = args.txt.read_text(encoding="utf-8")
    if not content.strip():
        raise ValueError(f"书籍文件为空: {args.txt}")

    rag = build_rag(args.book)
    await rag.initialize_storages()
    try:
        # 1.5.6 新签名: split_by_character 可控制分块; 默认即可
        await rag.ainsert(content)
    finally:
        await rag.finalize_storages()

    print(f"[OK] 完成: {args.book} -> {STORAGE_DIR / args.book}")


if __name__ == "__main__":
    asyncio.run(main())