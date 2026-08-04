# My Novel RAG — 实施计划书（交接用）

> **交接对象**：任一后续 agent。请先完整阅读本文档，再按「执行清单」逐步实施。
> **状态**：待 WSL2 环境就绪后执行。目标平台为 Linux (WSL2 Ubuntu)。
> **关键约束**：所有方案已对照 `lightrag-hku 1.5.5` 官方源码逐行验证，**不要凭旧教程/旧 API 印象修改**。

---

## 1. 执行摘要

构建一个「多本书独立隔离」的书籍知识库问答后端：

- 前端：Cherry Studio（或其他 LLM 平台），使用标准 OpenAI 接口格式。
- 后端：FastAPI + Uvicorn，暴露 `POST /v1/chat/completions`。
- RAG 引擎：`lightrag-hku`（**锁定 1.5.5**）。
- LLM：DeepSeek（OpenAI 兼容 API）。
- Embedding：第三方 OpenAI 兼容 API（**暂未定案，代码预留接入点，未配置时返回 `None`；定案配置后才能进入正式建图/查询链路，配置后须重建全部索引**）。

三大核心功能：

1. **多书路由**：按请求 `model` 字段（如 `novel-three-body`）懒加载/缓存指向 `storage/{model}` 的独立 LightRAG 实例；目录不存在时返回友好错误。
2. **System Prompt 透传**：提取 `messages` 中所有 `system` 消息并原样保留，与 LightRAG 检索上下文合并后一并交给 DeepSeek。
3. **OpenAI 兼容响应**：非流式 JSON 与流式 SSE 两种格式，Cherry Studio 可直接渲染。

---

## 2. 技术栈与版本锁定

| 组件 | 版本 | 说明 |
|---|---|---|
| Python | >= 3.10（建议 3.11） | `lightrag-hku` 要求 >=3.10 |
| lightrag-hku | **==1.5.5** | 硬锁定。该库 API 迭代极快，升级需重新对照源码验证 |
| fastapi | >=0.115 | |
| uvicorn[standard] | >=0.30 | |
| python-dotenv | >=1.0 | 读取 .env |
| DeepSeek | deepseek-chat | 兼容 OpenAI `/v1/chat/completions` |
| Embedding | TBD | OpenAI 兼容第三方 API，代码预留 |

### requirements.txt（最终交付）

```txt
fastapi>=0.115
uvicorn[standard]>=0.30
lightrag-hku==1.5.5
python-dotenv>=1.0
```

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

## 3. 源码验证结论（lightrag-hku 1.5.5，勿踩坑）

以下结论来自对 1.5.5 wheel 源码的逐行核对。网上大多数教程基于 1.3.x/1.4.x，**已失效**。

### 3.1 LightRAG 是 @dataclass，不是传统 __init__

- 构造参数全部为类字段（keyword-only 方式传入）。
- 关键字段：`working_dir`、`llm_model_func`、`llm_model_name`、`llm_model_kwargs`、`embedding_func`。
- 无需手动调用 `initialize_storages()`/`finalize_storages()` 之外的初始化——`ainsert`/`aquery` 内部会自动管理（`auto_manage_storages_states` 默认 False，但建图/查询路径内有自动初始化逻辑；为稳妥可显式调用，见 §5 代码）。

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
)
```

- 旧的 `chip_size` / `chunk_overlap_size` 参数已移除 → 分块参数改由 `addon_params`（InitVar）控制，默认即可，**不要传旧参数**。

### 3.3 aquery / aquery_llm 签名（System Prompt 官方注入点）

```python
async def aquery(
    self,
    query: str,
    param: QueryParam = QueryParam(),
    system_prompt: str | None = None,
) -> str | AsyncIterator[str]
```

- 第三个参数 `system_prompt` 即官方 System Prompt 注入点。`aquery` 内部包装 `aquery_llm`，按 `param.stream` 返回响应内容或异步迭代器。
- **stream=True 时返回 `AsyncIterator[str]`，非流式返回 `str`**。

### 3.4 ★ 核心陷阱：system_prompt 会被强制 .format()

`lightrag/operate.py` 的 `kg_query`（约 4198-4203 行）：

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

### 3.5 QueryParam 关键字段

```python
QueryParam(
    mode="hybrid",          # local/global/hybrid/naive/mix/bypass
    stream=False,           # True 时 aquery 返回 AsyncIterator[str]
    top_k=..., chunk_top_k=...,
    user_prompt=None,       # 附加指令，会注入到 prompt 模板的 {user_prompt}
    conversation_history=[],# [{"role","content"},...] 仅作上下文，不参与检索
)
```

### 3.6 LLM 封装

`lightrag.llm.openai.openai_complete_if_cache` 签名（DeepSeek 直接用）：

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
    ...
) -> str
```

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
- **未配置时 `embedding_func=None`**：不影响代码运行与建图（hybrid 模式会退化为基于关键词/图的方式，效果打折）；但 Embedding API 定案后**必须重新建图**，见 §8 风险表。

### 3.8 其它

- `QueryResult`：`content`（非流式文本）/ `response_iterator`（流式迭代器）/ `raw_data` / `is_streaming`。
- `aquery_llm` 返回 dict，包含 `llm_response.content` 或 `llm_response.response_iterator`；`aquery` 已帮你解包，**业务侧直接用 `aquery`**。

---

## 4. 目录结构规范（最终交付）

```
my-novel-rag/
├── .gitignore              # 必须含 /storage
├── .env.example            # 环境变量模板
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── config.py           # 环境变量 + LLM/Embedding 工厂
│   ├── builder.py          # 离线建图
│   └── api_server.py       # FastAPI 主服务
├── data/                   # 原始 txt 书籍
│   └── .gitkeep
└── storage/                # LightRAG 索引 (git 忽略) storage/{book}/
```

> 项目根目录建议：`~/my-novel-rag`（WSL2 Linux 原生路径，避免 `/mnt/c` 性能问题）。

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
BASE_DIR = Path(__file__).resolve().parent.parent          # my-novel-rag/
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


def build_embedding_func() -> EmbeddingFunc | None:
    """构造 OpenAI 兼容的 embedding 函数。

    设计决策 (用户确认): 不做 Fail-Fast, 未配置 (.env 未填 EMBEDDING_* 三件套)
    时返回 None。使用前提: embedding 定案并配好 API 后才进入正式建图/查询链路;
    若用 None 建过图, 配置 embedding 后必须重建全部索引 (见 plan.md §8 风险表)。
    """
    if not (EMBEDDING_BASE_URL and EMBEDDING_API_KEY and EMBEDDING_MODEL):
        return None
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

# Embedding 第三方 OpenAI 兼容 API (预留, 定案后填写)
EMBEDDING_BASE_URL=
EMBEDDING_API_KEY=
EMBEDDING_MODEL=
EMBEDDING_DIM=1024
EMBEDDING_MAX_TOKEN_SIZE=8192
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
        help="单文件大小上限 (MB), 默认 200。超大文本须先切片再分批 ainsert (见 §8)。",
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
            "超大文本须先切片, 再对每个切片分别调用 ainsert (分批构建), 见 §8 风险表。"
        )

    # 书籍内容读取: 显式指定 utf-8 (Linux 默认亦为 utf-8, 双保险)
    content = args.txt.read_text(encoding="utf-8")
    if not content.strip():
        raise ValueError(f"书籍文件为空: {args.txt}")

    rag = build_rag(args.book)
    await rag.initialize_storages()
    try:
        # 1.5.5 新签名: split_by_character 可控制分块; 默认即可
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
          (见 lightrag/operate.py)。
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

        # ---- 3. LightRAG 查询 (hybrid 模式) ----
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

## 6. WSL2 专项执行清单（Linux）

```bash
# 0. 前置: 安装 WSL2 + Ubuntu, 启动后进入
#    建议项目放在 Linux 原生路径, 避免 /mnt/c 性能问题

# 1. 安装 Python (>=3.10) 与 conda (可选)
#    若用 conda:
conda create -n novel-rag python=3.11 -y
conda activate novel-rag

# 2. 创建项目
mkdir -p ~/my-novel-rag/src ~/my-novel-rag/data ~/my-novel-rag/storage
# 将 plan.md 中的代码落盘为对应文件:
#   requirements.txt / .gitignore / .env.example / src/__init__.py
#   src/config.py / src/builder.py / src/api_server.py

# 3. 安装依赖
cd ~/my-novel-rag
pip install -r requirements.txt

# 4. 配置环境变量
cp .env.example .env
# 编辑 .env, 填入 DEEPSEEK_API_KEY (必填)

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

---

## 8. 风险登记表

| # | 风险 | 影响 | 对策 |
|---|---|---|---|
| 1 | `lightrag-hku` 升级 API 变动 | 代码失效 | **锁死 1.5.5**；升级前先对照源码重新验证 `LightRAG` 字段、`ainsert`、`aquery`、`QueryParam`、`openai_*` 签名 |
| 2 | Embedding 定案后 | 旧图无向量，hybrid 效果打折 | **必须重新建图**（`rm -rf storage/{book}` 后重跑 builder）；因此建议**先定 embedding 再正式建图** |
| 3 | LightRAG 实例跨事件循环复用 | asyncio 报错 | 实例在 FastAPI 事件循环内懒加载缓存；若未来改多 worker，需进程级隔离或每请求新建 |
| 4 | System Prompt 含 `{}` | `.format()` KeyError | 已实现转义 + 模板包装（§5.5 `build_rag_system_prompt`），不可移除 |
| 5 | `/mnt/c` 上建图性能差 | 慢 | 项目放 Linux 原生路径 `~/my-novel-rag` |
| 6 | Windows 编码 | 中文乱码 | 所有 `open()` 显式 `encoding='utf-8'`（已写死在代码） |
| 7 | `storage/` 撑爆 git | 仓库膨胀 | `.gitignore` 强制忽略 `/storage/` |
| 8 | DeepSeek 计费 | 建图耗时耗 token | 建图前用短文本试跑一次 `--book test` 验证链路后再全量 |
| 9 | 实例缓存内存膨胀 (多书 OOM) | 内存耗尽 | LRU 上限 `RAG_CACHE_MAX=8`（可配），**只淘汰引用计数为 0 的实例**（§5.5 `_evict_if_needed`），防止中断在途查询 |
| 10 | 并发首访重复实例化/文件锁冲突 | 重复初始化、脏数据 | per-model `asyncio.Lock()` + 双重检查（§5.5 `_get_rag_instance`） |
| 11 | 客户端断开流式中断 | 底层 LLM 连接未清理 | `_stream_response` 显式捕获 `CancelledError` + `asyncio.shield()` 关闭生成器（§5.5） |
| 12 | 超大体积单体文本建图 OOM | 内存峰值、进程被杀 | builder 默认 200MB 上限 + 明确报错（防呆）；**正解**：超大文本须预切片，对每个切片分别 `ainsert`（可传 `track_id` 合并文档状态），分批完成建图 |

---

## 9. 交付物清单（最终完成标准）

- [ ] `my-novel-rag/.gitignore`（含 `/storage/`）
- [ ] `my-novel-rag/requirements.txt`（`lightrag-hku==1.5.5`）
- [ ] `my-novel-rag/.env.example`
- [ ] `my-novel-rag/src/__init__.py`
- [ ] `my-novel-rag/src/config.py`
- [ ] `my-novel-rag/src/builder.py`
- [ ] `my-novel-rag/src/api_server.py`
- [ ] `my-novel-rag/data/` 内放置书籍 txt
- [ ] 至少一本书完成建图（`storage/{book}/` 存在）
- [ ] 服务启动成功，`/healthz` 返回 ok
- [ ] §7 全部测试要点通过
- [ ] Cherry Studio 接入并验证 System Prompt 透传