# My Novel RAG — 实施计划书（WSL2 修订版）

> **交接对象**：任一后续 agent。请先完整阅读本文档，再按「执行清单」逐步实施。
> **状态**：实施环境已确认为 **WSL2 Ubuntu（原生 Linux）**，目标 Python 环境为 `myenv`（当前 Python 3.14.6）。
> **关键约束**：所有方案已对照 `lightrag-hku 1.5.6` 官方源码（wheel 解压于 `/tmp/lr_1_5_6/src/`）逐行验证。**不要凭旧教程/旧 API 印象修改**。
> **版本说明**：本版由旧 plan.md 修订而来，核心变更为锁定 **lightrag-hku==1.5.6**（旧版锁定 1.5.5），并修正依赖清单与环境适配。

---

## 1. 执行摘要

构建一个「多本书独立隔离」的书籍知识库问答后端：

- 前端：Cherry Studio（或其他 LLM 平台），使用标准 OpenAI 接口格式。
- 后端：FastAPI + Uvicorn，暴露 `POST /v1/chat/completions`。
- RAG 引擎：`lightrag-hku`（**锁定 1.5.6**，1.5.6 为 PyPI 当前最新版，本地 wheel 已备于 `/tmp/lr_1_5_6/lightrag_hku-1.5.6-py3-none-any.whl`）。
- LLM：DeepSeek（OpenAI 兼容 API）。
- Embedding：第三方 OpenAI 兼容 API（**框架阶段 `.env` 填入占位值以构造 `EmbeddingFunc`；1.5.6 在实例化即强制校验 embedding_func，不允许 `None`；真实建图前必须替换为有效 embedding API，更换模型须重建索引**）。

三大核心功能：

1. **多书路由**：按请求 `model` 字段（如 `novel-three-body`）懒加载/缓存指向 `storage/{model}` 的独立 LightRAG 实例；目录不存在时返回 OpenAI 规范错误结构（404）。
2. **System Prompt 透传**：提取 `messages` 中所有 `system` 消息并原样保留，经转义 + 模板包装后与 LightRAG 检索上下文合并，一并交给 DeepSeek。
3. **OpenAI 兼容响应**：非流式 JSON 与流式 SSE 两种格式，Cherry Studio 可直接渲染。

---

## 2. 技术栈与版本锁定

| 组件 | 版本 | 说明 |
|---|---|---|
| Python | >= 3.10（当前环境 3.14.6） | `lightrag-hku` 要求 >=3.10；见 §8 风险 #2 |
| lightrag-hku | **==1.5.6** | 硬锁定。该库 API 迭代极快，升级需重新对照源码验证 |
| openai | >=2.0,<3.0 | **必须显式安装**：`lightrag.llm.openai` 顶层 `from openai import ...`（openai.py:15）硬依赖；1.5.6 api extra 约束 `>=2.0.0,<3.0.0` |
| fastapi | >=0.115 | |
| uvicorn[standard] | >=0.30 | |
| python-dotenv | >=1.0 | 读取 .env |
| DeepSeek | deepseek-chat | 兼容 OpenAI `/v1/chat/completions` |
| Embedding | TBD | OpenAI 兼容第三方 API，代码预留 |

### requirements.txt（最终交付）

```txt
fastapi>=0.115
uvicorn[standard]>=0.30
lightrag-hku==1.5.6
openai>=2.0,<3.0
python-dotenv>=1.0
```

> 依赖说明：1.5.6 的 `openai` 是硬依赖而非可选（`lightrag/llm/openai.py` 顶层导入），旧版 requirements 遗漏此项，本版已修正。

### .gitignore（最终交付，必须包含）

```gitignore
# 索引数据，严禁入库
/storage/

# 环境变量
.env

# Python
__pycache__/
*.pyc
*.pyo
.venv/
venv/
```

---

## 3. 源码验证结论（lightrag-hku 1.5.6，勿踩坑）

以下结论来自对 1.5.6 wheel 源码（`/tmp/lr_1_5_6/src/`）的逐行核对。网上大多数教程基于 1.3.x/1.4.x，**已失效**。旧 plan.md 基于 1.5.5，其核心结论经核对在 1.5.6 中依然成立，差异点见 §3.5。

### 3.1 LightRAG 是 @dataclass，不是传统 __init__

- `@dataclass class LightRAG(_RoleLLMMixin, _StorageMigrationMixin, _PipelineMixin)`（lightrag.py:385）。
- 构造参数全部为类字段（keyword-only 方式传入）。
- 关键字段：`working_dir`、`llm_model_func`、`llm_model_name`、`llm_model_kwargs`、`embedding_func`（lightrag.py:617-722）。
- `initialize_storages()`（lightrag.py:1556）/ `finalize_storages()`（lightrag.py:1644）仍存在；`ainsert`/`aquery` 内部有自动初始化逻辑，但为稳妥在代码中显式调用（见 §5.4 / §5.5）。

### 3.2 ainsert 新签名

```python
async def ainsert(
    self,
    input: str | list[str],
    split_by_character: str | None = None,
    split_by_character_only: bool = False,
    ids: str | list[str] | None = None,
    file_paths: str | list[str] | None = None,
    track_id: str | None = None,
) -> str
```
（lightrag.py:1762-1770）

- 旧的 `chip_size` / `chunk_overlap_size` 参数已移除 → 分块参数改由 `addon_params`（InitVar）控制；`ainsert` 固定使用 fixed-token 分块策略，默认即可，**不要传旧参数**。
- 返回值为 tracking ID（新行为，1.5.5 即为如此）。
- 如需递归字符 (R) / 语义向量 (V) / 段落语义 (P) 分块策略，须改用 `apipeline_enqueue_documents` + `apipeline_process_enqueue_documents`（SDK 的 `ainsert` 无法选择这些策略，见 lightrag.py:1771-1789 文档注）。本项目默认使用 `ainsert` 即可。

### 3.3 aquery / aquery_llm 签名（System Prompt 官方注入点）

```python
async def aquery(
    self,
    query: str,
    param: QueryParam = QueryParam(),
    system_prompt: str | None = None,
) -> str | AsyncIterator[str]
```
（lightrag.py:3643-3648）

- 第三个参数 `system_prompt` 即官方 System Prompt 注入点。`aquery` 内部包装 `aquery_llm`（lightrag.py:3666），按 `param.stream` 返回响应内容或异步迭代器。
- **stream=True 时返回 `AsyncIterator[str]`，非流式返回 `str`**（lightrag.py:3671-3674）。

### 3.4 ★ 核心陷阱：system_prompt 会被强制 .format()

`lightrag/operate.py` 的 `kg_query`（**1.5.6 行号 4288-4293**；旧 plan.md 记载 4198 为 1.5.5 行号）：

```python
sys_prompt_temp = system_prompt if system_prompt else PROMPTS["rag_response"]
sys_prompt = sys_prompt_temp.format(
    response_type=response_type,
    user_prompt=user_prompt,
    context_data=context_result.context,
)
```

推论（直接影响 System Prompt 透传实现）：
1. 直接传 Cherry Studio 的 System Prompt **原文**，且原文不含 `{context_data}` 占位符 → **检索上下文丢失**。
2. 原文若含任意 `{...}`（如 JSON 示例、正则）→ `.format()` 抛 `KeyError`/`IndexError`。
3. **正确做法**：服务端先把用户原文的 `{`→`{{`、`}`→`}}` 转义，再包装成含 `{response_type}`/`{user_prompt}`/`{context_data}` 占位符的模板传入 `aquery(..., system_prompt=模板)`。

> 注意：`.format()` 强制发生在 `kg_query` 的 local/global/hybrid/mix 路径（operate.py:4288）。naive/bypass 路径的模板行为见 operate.py 5300/6234 区域，本项目使用 hybrid 模式，走 4288 路径。

### 3.5 QueryParam 关键字段 ★含 1.5.6 差异

```python
QueryParam(
    mode="hybrid",          # local/global/hybrid/naive/mix/bypass
    stream=False,           # True 时 aquery 返回 AsyncIterator[str]
    top_k=..., chunk_top_k=...,
    user_prompt=None,       # 附加指令，会注入到 prompt 模板的 {user_prompt}
    conversation_history=[],# [{"role","content"},...] 仅作上下文，不参与检索
)
```
（base.py:90-160）

- **★ 1.5.6 差异**：默认 `mode` 从 1.5.5 的 `"hybrid"` 变为 **`"mix"`**（base.py:93）。本项目代码显式传 `mode="hybrid"` 故不受影响；但**不要依赖默认值**，查询时务必显式指定。
- 1.5.6 新增字段（与此项目无关但需知晓）：`enable_rerank`（默认由 `RERANK_BY_DEFAULT` 环境变量控制，默认 true）、`include_references` 等。未配置 rerank 模型时 `enable_rerank=True` 仅发警告不报错。

### 3.6 LLM 封装

`lightrag.llm.openai.openai_complete_if_cache` 签名（openai.py:244，DeepSeek 直接用）：

```python
async def openai_complete_if_cache(
    model: str,
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list[dict[str, Any]] | None = None,
    enable_cot: bool = False,
    base_url: str | None = None,
    api_key: str | None = None,
    token_tracker: Any | None = None,
    stream: bool | None = None,
    timeout: int | None = None,
    keyword_extraction: bool = False,
    use_azure: bool = False,
    azure_deployment: str | None = None,
    api_version: str | None = None,
    image_inputs: list[Any] | None = None,
    **kwargs: Any,
) -> str
```

- 依赖 `openai>=2.0,<3.0`（顶层 `from openai import ...`，openai.py:15）。
- 支持 DeepSeek 风格 `reasoning_content`（COT），非流式时以 `<think>` 标签前置；本项目不启用 `enable_cot`。

### 3.7 Embedding 接入（第三方程式，预留）

```python
from lightrag.llm.openai import openai_embed
from lightrag.utils import EmbeddingFunc, wrap_embedding_func_with_attrs

async def openai_embed(
    texts: list[str],
    model: str = "text-embedding-3-small",
    base_url: str | None = None,
    api_key: str | None = None,
    embedding_dim: int | None = None,   # 由装饰器自动注入，勿手动传
    max_token_size: int | None = None,  # 由装饰器自动注入
    ...
) -> np.ndarray
```
（openai.py:955；`EmbeddingFunc` 定义 utils.py:540；`wrap_embedding_func_with_attrs` utils.py:2363）

```python
# 官方封装方式：
embedding_func = EmbeddingFunc(
    embedding_dim=1024,
    max_token_size=8192,
    func=wrap_embedding_func_with_attrs(
        embedding_dim=1024,
        max_token_size=8192,
    )(openai_embed),
)
```

- 导入路径：`EmbeddingFunc` 与 `wrap_embedding_func_with_attrs` 位于 `lightrag.utils`。
- **★ 1.5.6 强制 embedding（重大差异，推翻旧版假设）**：`LightRAG.__post_init__` 无条件实例化三个向量存储（lightrag.py:1447-1463），`_validate_embedding_func()`（base.py:241-245）在 `embedding_func=None` 时抛 `ValueError`。**`embedding_func` 在 1.5.6 下必填，不存在「无 embedding 跑纯净关键词」模式**（旧 plan.md 基于 1.5.5 的假设已失效，本章节旧文字作废）。
- **框架阶段策略**：`.env` 的 `EMBEDDING_*` 填占位值，`build_embedding_func()` 始终构造 `EmbeddingFunc`（`openai_embed` 包装），使 `LightRAG` 正常实例化、服务框架可启动。占位配置下若误触发建图会得到 OpenAI API 明确报错（Fail-Fast），不会静默产生假数据。真实建图前必须替换为有效 embedding API；更换 embedding 模型须重建索引，见 §8 风险 #3。

### 3.8 其它

- `aquery` 已解包 `aquery_llm` 的返回 dict（lightrag.py:3666-3674），**业务侧直接用 `aquery`**，非流式取 `str`、流式取 `AsyncIterator[str]`。
- `QueryResult` / `aquery_data`：如需结构化检索结果（不带 LLM 生成），用 `aquery_data(query, param)`（lightrag.py:3701），本项目暂不启用。

---

## 4. 目录结构规范（最终交付）

```
Archailect/                   # 本项目根目录（即 plan 所在目录）
├── plan.md                   # 本文档
├── .gitignore                # 必须含 /storage
├── .env.example              # 环境变量模板
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── config.py             # 环境变量 + LLM/Embedding 工厂
│   ├── builder.py            # 离线建图
│   └── api_server.py         # FastAPI 主服务
├── data/                     # 原始 txt 书籍
│   └── .gitkeep
└── storage/                  # LightRAG 索引 (git 忽略) storage/{book}/
```

> 项目根目录即 `/home/tessier/Archailect`（WSL2 Linux 原生路径，避免 `/mnt/c` 性能问题，已满足）。

---

## 5. 完整参考代码

### 5.1 src/__init__.py

```python
"""My Novel RAG 后端服务。"""
```

### 5.2 src/config.py

```python
"""集中管理环境变量, 并提供 LLM / Embedding 工厂函数。"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc, wrap_embedding_func_with_attrs

load_dotenv()

# ---- 路径 ----
BASE_DIR = Path(__file__).resolve().parent.parent          # Archailect/
DATA_DIR = BASE_DIR / "data"
STORAGE_DIR = BASE_DIR / "storage"


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"缺少环境变量: {name}, 请检查 .env")
    return value


# ---- LLM (DeepSeek) ----
DEEPSEEK_API_KEY = _require_env("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

# ---- Embedding (预留, 第三方 OpenAI 兼容 API) ----
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", "")
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))
EMBEDDING_MAX_TOKEN_SIZE = int(os.getenv("EMBEDDING_MAX_TOKEN_SIZE", "8192"))


def build_embedding_func() -> EmbeddingFunc:
    """构造 OpenAI 兼容的 embedding 函数。

    设计决策 (用户确认): 1.5.6 在实例化即强制校验 embedding_func (不允许 None),
    因此本函数始终返回 EmbeddingFunc。框架阶段 .env 的 EMBEDDING_* 为占位值,
    使 LightRAG 可正常实例化、服务可启动; 占位配置下误触发建图会得到 OpenAI API
    的明确报错 (Fail-Fast), 不会静默产生假数据。真实建图前必须将 EMBEDDING_*
    替换为有效配置; 更换 embedding 模型后须重建全部索引 (见 plan.md §8 风险表)。
    """
    return EmbeddingFunc(
        embedding_dim=EMBEDDING_DIM,
        max_token_size=EMBEDDING_MAX_TOKEN_SIZE,
        func=wrap_embedding_func_with_attrs(
            embedding_dim=EMBEDDING_DIM,
            max_token_size=EMBEDDING_MAX_TOKEN_SIZE,
        )(openai_embed),
    )
```

### 5.3 .env.example

```dotenv
# DeepSeek (必填)
DEEPSEEK_API_KEY=sk-xxxxxxxx
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
DEEPSEEK_MODEL=deepseek-chat

# Embedding 第三方 OpenAI 兼容 API
# 框架阶段填占位值即可 (使 LightRAG 可实例化); 真实建图前必须替换为有效配置
EMBEDDING_BASE_URL=https://api.openai.com/v1
EMBEDDING_API_KEY=sk-placeholder
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIM=1024
EMBEDDING_MAX_TOKEN_SIZE=8192

# 多书实例 LRU 缓存上限 (可选, 默认 8)
RAG_CACHE_MAX=8
```

### 5.4 src/builder.py

```python
"""离线建图: python -m src.builder --txt data/xxx.txt --book novel-three-body"""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from lightrag import LightRAG
from lightrag.llm.openai import openai_complete_if_cache

from src.config import (
    DATA_DIR,
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
        help="单文件大小上限 (MB), 默认 200。超大文本须先切片再分批 ainsert (见 §8 风险 #13)。",
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
            "超大文本须先切片, 再对每个切片分别调用 ainsert (分批构建), 见 §8 风险 #13。"
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
```

### 5.5 src/api_server.py

```python
"""FastAPI 服务: 暴露 OpenAI 兼容的 /v1/chat/completions。

三大核心逻辑:
  1. 多书路由: 按请求 model 字段懒加载 storage/{model} 的 LightRAG 实例
  2. System Prompt 透传: 见 build_rag_system_prompt() 注释
  3. 响应格式: 严格 OpenAI 规范 (非流式 JSON / 流式 SSE)
"""
from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache
from pydantic import BaseModel, Field

from src.config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    STORAGE_DIR,
    build_embedding_func,
)

# ============================================================
# 一、多书路由的核心数据结构 (LRU + 并发锁 + 在途请求保护)
# ============================================================
# 每个 model 对应一个独立 LightRAG 实例, working_dir 指向 storage/{model},
# 从物理目录层面保证知识库完全隔离。

# 实例缓存: OrderedDict 支持 LRU 顺序 (move_to_end 更新热度)
_rag_instances: OrderedDict[str, LightRAG] = OrderedDict()
# 每实例在途请求计数: 淘汰前必须为 0, 防止中断正在进行的查询
_rag_refcounts: dict[str, int] = {}
# per-model 异步锁: 保证同一本书的首次加载不会并发重复实例化
_rag_locks: dict[str, asyncio.Lock] = {}
# 缓存容量上限 (实例数), 超出后惰性淘汰最久未用且无在途请求的实例
_RAG_CACHE_MAX = int(os.getenv("RAG_CACHE_MAX", "8"))


async def _finalize_and_remove(model: str) -> None:
    """将实例移出缓存并释放资源 (必须在持有该 model 锁的上下文中调用)。"""
    rag = _rag_instances.pop(model, None)
    _rag_refcounts.pop(model, None)
    _rag_locks.pop(model, None)
    if rag is not None:
        try:
            await rag.finalize_storages()
        except Exception:
            pass


async def _evict_if_needed() -> None:
    """容量超限时, 从最久未用开始淘汰「无在途请求」的实例。

    - 在 _get_rag_instance 的锁内调用, 天然无并发风险。
    - 只有 refcount == 0 的实例才可淘汰; 若全部在途则暂不淘汰,
      待请求结束后由下一次访问自然触发。
    """
    while len(_rag_instances) > _RAG_CACHE_MAX:
        victim = next(
            (m for m in _rag_instances if _rag_refcounts.get(m, 0) == 0),
            None,
        )
        if victim is None:
            break
        await _finalize_and_remove(victim)


async def _get_rag_instance(model: str) -> LightRAG:
    """按 model 懒加载/缓存 LightRAG 实例 (多书路由)。

    - 并发安全: 每 model 一把 asyncio.Lock, 同一本书的首次加载仅一个协程执行
      (双重检查, 等待锁期间其他协程可能已创建)。
    - LRU 淘汰: 容量超限时在锁内惰性淘汰无在途请求的最久未用实例。
    - 目录不存在时抛 404 语义错误。
    """
    # 快路径: 已缓存则直接返回 (不持有锁, 避免无谓竞争)
    if model in _rag_instances:
        _rag_instances.move_to_end(model)
        return _rag_instances[model]

    # 慢路径: 加锁, 双重检查后创建
    if model not in _rag_locks:
        _rag_locks[model] = asyncio.Lock()
    async with _rag_locks[model]:
        if model in _rag_instances:  # 双重检查
            _rag_instances.move_to_end(model)
            return _rag_instances[model]

        storage_dir = STORAGE_DIR / model
        if not storage_dir.is_dir():
            raise HTTPException(
                status_code=404,
                detail=f"知识库 '{model}' 不存在。请先运行: python -m src.builder --txt xxx.txt --book {model}",
            )

        rag = LightRAG(
            working_dir=str(storage_dir),
            llm_model_func=openai_complete_if_cache,
            llm_model_name=DEEPSEEK_MODEL,
            llm_model_kwargs={"base_url": DEEPSEEK_BASE_URL, "api_key": DEEPSEEK_API_KEY},
            embedding_func=build_embedding_func(),
        )
        _rag_instances[model] = rag
        _rag_refcounts[model] = 0
        await _evict_if_needed()
        return rag


# ============================================================
# 二、System Prompt 透传 (核心难点)
# ============================================================
def build_rag_system_prompt(user_system_prompt: str) -> str:
    """包装用户 System Prompt, 使其在不丢失的前提下注入检索上下文。

    背景: LightRAG 的 kg_query 会对传入的 system_prompt 强制调用
          `str.format(response_type=..., user_prompt=..., context_data=...)`
          (见 lightrag/operate.py:4288-4293)。
    因此必须:
      1. 把用户原文的 { } 转义为 {{ }}, 否则会被 format 当作占位符,
         要么吞掉检索上下文, 要么抛 KeyError;
      2. 提供 {context_data} 占位符, 让 LightRAG 把检索到的书籍背景注入;
      3. 提供 {user_prompt} 占位符, 兼容 QueryParam.user_prompt 附加指令。
    这样「用户系统提示词」与「检索上下文」两条信息流都无损地进最终 prompt。
    """
    escaped = user_system_prompt.replace("{", "{{").replace("}", "}}")
    return (
        "---User-defined Role---\n"
        f"{escaped}\n\n"
        "You MUST obey the role constraints defined above.\n"
        "Then answer the user query, grounding on the knowledge base context "
        "provided below when it is relevant.\n\n"
        "Additional instructions: {user_prompt}\n\n"
        "---Knowledge Base Context---\n"
        "{context_data}"
    )


def _extract_system_prompt(messages: list[dict[str, Any]]) -> str:
    """提取所有 role=system 的消息内容, 用换行合并 (透传不丢弃)。"""
    parts = [m.get("content", "") for m in messages if m.get("role") == "system"]
    return "\n".join(p for p in parts if isinstance(p, str) and p.strip())


def _extract_last_user_query(messages: list[dict[str, Any]]) -> str:
    """提取最后一条 user 消息作为交给 LightRAG 的提问。"""
    for m in reversed(messages):
        if m.get("role") == "user":
            content = m.get("content", "")
            if isinstance(content, str) and content.strip():
                return content
            # 处理 OpenAI 多段 content 数组 (string | list[part])
            if isinstance(content, list):
                texts = [
                    part.get("text", "")
                    for part in content
                    if isinstance(part, dict) and part.get("type") == "text"
                ]
                joined = "".join(texts).strip()
                if joined:
                    return joined
    raise HTTPException(status_code=400, detail="请求缺少有效的 user 消息")


# ============================================================
# 三、OpenAI 兼容响应组装
# ============================================================
def _openai_error(body: dict[str, Any], status: int) -> HTTPException:
    return HTTPException(
        status_code=status,
        detail={
            "error": {
                "message": body.get("message", "Unknown error"),
                "type": body.get("type", "invalid_request_error"),
                "code": body.get("code"),
            }
        },
    )


def _non_stream_response(content: str, model: str) -> dict[str, Any]:
    """非流式响应的严格 OpenAI 结构。"""
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def _sse_event(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


async def _stream_response(
    token_iter: AsyncIterator[str], model: str
) -> AsyncIterator[str]:
    """把 LightRAG 流式输出包装为 OpenAI SSE chunk 序列。

    客户端中断处理: 前端停止生成/关闭连接时, 生成器收到
    asyncio.CancelledError。它继承自 BaseException, 普通 except Exception
    捕获不到, 必须显式捕获。取消中的 task 内无法直接 await (会立即再抛
    CancelledError), 因此清理动作用 asyncio.shield() 隔离后再 raise。
    """
    message_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    try:
        # 首帧: 声明 assistant 角色
        yield _sse_event(
            {
                "id": message_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
                ],
            }
        )
        async for token in token_iter:
            yield _sse_event(
                {
                    "id": message_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [
                        {"index": 0, "delta": {"content": token}, "finish_reason": None}
                    ],
                }
            )
        # 结束帧
        yield _sse_event(
            {
                "id": message_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
        )
        yield "data: [DONE]\n\n"
    except asyncio.CancelledError:
        # 客户端断开: 优雅关闭底层 LLM 异步生成器, 不向上溢出
        if hasattr(token_iter, "aclose"):
            await asyncio.shield(token_iter.aclose())  # type: ignore[union-attr]
        raise


# ============================================================
# 四、FastAPI 应用
# ============================================================
class ChatRequest(BaseModel):
    model: str = Field(..., min_length=1)
    messages: list[dict[str, Any]] = Field(..., min_length=1)
    stream: bool = Field(default=False)
    temperature: float | None = None
    max_tokens: int | None = None
    # 其余 OpenAI 参数忽略即可, 不影响功能


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    # 关闭时持久化所有实例的缓存数据
    for model, rag in _rag_instances.items():
        try:
            await rag.finalize_storages()
        except Exception:
            pass


app = FastAPI(title="My Novel RAG", lifespan=lifespan)

# Cherry Studio / 其他前端跨域访问必需
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/models")
async def list_models() -> dict[str, Any]:
    """方便 Cherry Studio 发现可用模型。"""
    models = [
        {"id": d.name, "object": "model", "owned_by": "my-novel-rag"}
        for d in sorted(STORAGE_DIR.iterdir())
        if d.is_dir()
    ]
    return {"object": "list", "data": models}


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest) -> Any:
    # ---- 1. 多书路由: 解析 model 并获取对应实例 (并发安全 + LRU) ----
    # 先登记在途引用, 防止 LRU 淘汰正在被查询的实例
    _rag_refcounts[req.model] = _rag_refcounts.get(req.model, 0) + 1
    try:
        try:
            rag = await _get_rag_instance(req.model)
        except HTTPException as exc:
            raise _openai_error(
                {
                    "message": exc.detail,
                    "type": "invalid_request_error",
                    "code": "model_not_found",
                },
                exc.status_code,
            )

        # ---- 2. System Prompt 透传 + 提取用户提问 ----
        user_system_prompt = _extract_system_prompt(req.messages)
        query = _extract_last_user_query(req.messages)
        rag_system_prompt = build_rag_system_prompt(user_system_prompt)

        # ---- 3. LightRAG 查询 (hybrid 模式; 不依赖 1.5.6 的默认 mix) ----
        param = QueryParam(mode="hybrid", stream=req.stream)
        result = await rag.aquery(query, param=param, system_prompt=rag_system_prompt)

        # ---- 4. 响应格式: 流式 SSE / 非流式 JSON ----
        if req.stream:
            # aquery stream=True 返回 AsyncIterator[str]
            return StreamingResponse(
                _stream_response(result, req.model),  # type: ignore[arg-type]
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        return _non_stream_response(str(result), req.model)
    finally:
        # 请求结束释放引用; 若容量超限, 下一次 _get_rag_instance 的锁内会触发淘汰
        _rag_refcounts[req.model] = max(0, _rag_refcounts.get(req.model, 0) - 1)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.api_server:app", host="0.0.0.0", port=8000, reload=True)
```

---

## 6. 执行清单（WSL2 Ubuntu，myenv 环境）

> 前置确认：当前已在 WSL2 Ubuntu。conda 环境 `myenv` 现为 Python 3.14.6（2026-08 实测）。项目根 `/home/tessier/Archailect` 位于 Linux 原生路径，性能无忧。

```bash
# 0. 激活目标 conda 环境 (myenv)
conda activate myenv

# 1. 确认 Python 版本 (>=3.10 即可; 3.14 见 §8 风险 #2)
python --version

# 2. 项目文件 (本 plan 落盘后应已就绪)
#    需创建: requirements.txt / .gitignore / .env.example / src/__init__.py
#            src/config.py / src/builder.py / src/api_server.py / data/.gitkeep
#    若以上文件尚不存在, 按 §5 参考代码逐一落盘

# 3. 安装依赖
#    优先使用本地 wheel 安装 lightrag-hku 1.5.6 (已备于 /tmp/lr_1_5_6/)
pip install /tmp/lr_1_5_6/lightrag_hku-1.5.6-py3-none-any.whl
pip install -r requirements.txt
#    若本地 wheel 缺失或安装失败, 回退到 PyPI:
#    pip install lightrag-hku==1.5.6 openai>=2.0,<3.0

# 4. 配置环境变量
cp .env.example .env
# 编辑 .env:
#   - DEEPSEEK_API_KEY: 必填
#   - EMBEDDING_*: 框架阶段为占位值即可; 真实建图前必须替换为有效 embedding API

# 5. 放置书籍 txt 到 data/ 目录

# 6. 建图 (每本书一次)
python -m src.builder --txt data/三体.txt --book novel-three-body
python -m src.builder --txt data/金庸.txt  --book novel-jin-yong

# 7. 启动服务
python -m src.api_server
# 默认 0.0.0.0:8000
```

### Cherry Studio 接入

1. 设置 → 接入提供方 → 添加自定义 OpenAI 兼容服务商。
2. API 地址：`http://localhost:8000/v1`（远程则填服务器 IP/域名）。
3. API Key：任意占位字符串（此服务不做鉴权）。
4. 模型名：填 `novel-three-body`（或已建库的 model 名）。
5. 在该模型的「系统提示词（System Prompt）」中填写任意系统提示词，验证透传生效。

### 冒烟测试（curl）

```bash
# 非流式
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"novel-three-body","messages":[{"role":"system","content":"你是一个严谨的书评人"},{"role":"user","content":"介绍一下主要角色"}]}'

# 流式
curl -sN http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"novel-three-body","stream":true,"messages":[{"role":"user","content":"这本书讲了什么"}]}'
```

---

## 7. 测试要点（实施后必须验证）

| # | 验证项 | 预期 |
|---|---|---|
| 1 | `POST /v1/chat/completions`（非流式） | 返回 OpenAI 规范 JSON，`choices[0].message.content` 为回答 |
| 2 | 同上 + `stream:true` | SSE 分块输出，结尾 `data: [DONE]` |
| 3 | System Prompt 透传 | 问："根据我给你的设定，你是谁？" 模型回答与设定一致 |
| 4 | 多书隔离 | `model=novel-jin-yong` 提问金庸内容，`model=novel-three-body` 提问三体内容，互不串台 |
| 5 | 不存在的 model | 返回 404 语义错误且格式为 OpenAI `error` 结构 |
| 6 | `GET /healthz` | `{"status":"ok"}` |
| 7 | `GET /v1/models` | 列出 storage 下已建库目录名 |
| 8 | 同一 model 并发首访 | 多个并发请求同时首次访问同一 model，仅实例化一次，无报错、无重复初始化（per-model 锁） |
| 9 | LRU 淘汰不中断在途查询 | 并发请求 A 查询中，其余请求触发淘汰，A 的实例不因淘汰而 `finalize`，回答正常返回 |
| 10 | 流式中断 (客户端断开) | 中断后服务不抛未处理异常，其余请求不受影响，日志无 `CancelledError` 泄漏 |
| 11 | Prompt 含 `{}` 字符 | 系统提示词含 JSON 示例/正则 `{...}` 时不抛 `KeyError`，转义 + 模板包装生效（§3.4） |
| 12 | QueryParam 显式 mode | 请求 `mode="hybrid"` 生效；不因 1.5.6 默认 `mix` 造成行为漂移 | 

---

## 8. 风险登记表

| # | 风险 | 影响 | 对策 |
|---|---|---|---|
| 1 | `lightrag-hku` 升级 API 变动 | 代码失效 | **锁死 1.5.6**；升级前先对照源码重新验证 `LightRAG` 字段、`ainsert`、`aquery`、`QueryParam`、`openai_*` 签名 |
| 2 | **Python 3.14 兼容性（本项目最大未知数）** | myenv 为 Python 3.14.6，1.5.6 官方要求 >=3.10 但依赖链未实车验证 3.14 | 安装时若 numpy/pandas/pydantic 等解析失败或运行时异常：**回退方案** `conda create -n novel-rag python=3.11` 新环境重装；不强行在 3.14 上打补丁 |
| 3 | 占位 embedding 配置 + 误触发建图 | 对无效 API 发起真实请求，报 401/连接错误（Fail-Fast，不产生假数据） | 框架阶段只做实例化/启动，不执行 `ainsert`；建图前必须将 `EMBEDDING_*` 替换为有效配置。更换 embedding 模型后**必须重新建图**（`rm -rf storage/{book}` 后重跑 builder） |
| 4 | LightRAG 实例跨事件循环复用 | asyncio 报错 | 实例在 FastAPI 事件循环内懒加载缓存；若未来改多 worker，需进程级隔离或每请求新建 |
| 5 | System Prompt 含 `{}` | `.format()` KeyError | 已实现转义 + 模板包装（§5.5 `build_rag_system_prompt`），不可移除（operate.py:4288 强制 format） |
| 6 | `/mnt/c` 上建图性能差 | 慢 | 项目已放 Linux 原生路径 `~/Archailect`（WSL2 内），已满足 |
| 7 | 文件编码 | 中文乱码 | 所有 `open()` 显式 `encoding='utf-8'`（已写死在代码） |
| 8 | `storage/` 撑爆 git | 仓库膨胀 | `.gitignore` 强制忽略 `/storage/` |
| 9 | DeepSeek 计费 | 建图耗时耗 token | 建图前用短文本试跑一次 `--book test` 验证链路后再全量 |
| 10 | 实例缓存内存膨胀 (多书 OOM) | 内存耗尽 | LRU 上限 `RAG_CACHE_MAX=8`（可配），只淘汰引用计数为 0 的实例（§5.5 `_evict_if_needed`），防止中断在途查询 |
| 11 | 并发首访重复实例化/文件锁冲突 | 重复初始化、脏数据 | per-model `asyncio.Lock()` + 双重检查（§5.5 `_get_rag_instance`） |
| 12 | 客户端断开流式中断 | 底层 LLM 连接未清理 | `_stream_response` 显式捕获 `CancelledError` + `asyncio.shield()` 关闭生成器（§5.5） |
| 13 | 超大体积单体文本建图 OOM | 内存峰值、进程被杀 | builder 默认 200MB 上限 + 明确报错（防呆）；超大文本须预切片，对每个切片分别 `ainsert`（可传 `track_id` 合并文档状态），分批完成建图 |
| 14 | **1.5.5→1.5.6 默认 behavior 变化** | 依赖默认值会得到非预期检索模式 | 1.5.6 `QueryParam.mode` 默认 `"mix"`（1.5.5 为 `"hybrid"`）；查询代码必须显式传 `mode="hybrid"`（已落实 §5.5） |
| 15 | **openai 库版本误配** | import 失败 | requirements 显式 `openai>=2.0,<3.0`（1.5.6 api extra 约束；`lightrag.llm.openai` 顶层导入） |
| 16 | **1.5.6 强制 embedding（相对旧版最大行为差异）** | `embedding_func=None` 时实例化即抛 `ValueError`；旧 plan「无 embedding 可跑纯关键词」假设失效 | `build_embedding_func()` 始终构造 `EmbeddingFunc`；框架阶段 `.env` 用占位值，真实建图前替换为有效 embedding API（本表风险 #3） |

---

## 9. 交付物清单（最终完成标准）

**阶段一：框架就绪（当前目标）**
- [ ] `Archailect/.gitignore`（含 `/storage/`）
- [ ] `Archailect/requirements.txt`（`lightrag-hku==1.5.6` + `openai>=2.0,<3.0`）
- [ ] `Archailect/.env.example`（`EMBEDDING_*` 为占位值）
- [ ] `Archailect/src/__init__.py`
- [ ] `Archailect/src/config.py`
- [ ] `Archailect/src/builder.py`
- [ ] `Archailect/src/api_server.py`
- [ ] `Archailect/data/` 放置书籍 txt 的 `.gitkeep`
- [ ] `myenv` 环境安装成功（lightrag-hku 1.5.6 / fastapi / uvicorn / openai / python-dotenv）
- [ ] 框架冒烟：`LightRAG` 实例化 + `QueryParam` 正常（不触发 embedding API 调用）

**阶段二：真实建图与验收（需真实 embedding API，另行执行）**
- [ ] `.env` 的 `EMBEDDING_*` 替换为有效配置
- [ ] 至少一本书完成建图（`storage/{book}/` 存在）
- [ ] 服务启动成功，`/healthz` 返回 ok
- [ ] §7 全部测试要点通过
- [ ] Cherry Studio 接入并验证 System Prompt 透传
