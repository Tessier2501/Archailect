"""集中管理环境变量, 并提供 LLM / Embedding 工厂函数.

架构 (2026-08-11):
- LLM 分角色: QUERY (主, 必填) / KEYWORD / EXTRACT (可选, 空=回退 QUERY).
  每个已配置角色均有 free/paid 双档 + 独立 fallback 电路:
  免费优先 → 429/限流后"仅当次"退付费 (见 _FallbackCircuit), 下次自动回免费.
- KEYWORD/EXTRACT 未配置时返回 None → LightRAG 回退到 base llm_model_func
  (=QUERY wrapper, QUERY 代劳).
- Provider: QUERY_BASE_URL/QUERY_API_KEY 与 EMBEDDING_BASE_URL/EMBEDDING_API_KEY
  必填; KEYWORD/EXTRACT 可选, 各自可配独立 provider (缺省回退 QUERY).
- Embedding 免费优先+付费兜底, 同模型同维度可混用.
"""
from __future__ import annotations

import os
import threading
import time
from functools import partial
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc, wrap_embedding_func_with_attrs

# override=True: .env 是权威配置源. 默认不覆盖会导致 shell 残留同名
# 环境变量 (如 source .env 后的旧值) 遮蔽 .env 的新配置, 已实测踩坑.
load_dotenv(override=True)

# ---- 路径 ----
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
STORAGE_DIR = BASE_DIR / "storage"

# ---- 缓存 ----
RAG_CACHE_MAX = int(os.getenv("RAG_CACHE_MAX", "8"))


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"缺少环境变量: {name}, 请检查 .env")
    return value


def _resolve_env(name: str) -> str:
    """读取环境变量并拒绝未填写的占位符 (<...>)."""
    value = _require_env(name)
    if "<" in value or ">" in value or value.startswith("$"):
        raise RuntimeError(
            f"环境变量 {name} 仍是占位符, 请先在 .env 填入真实值 "
            f"(当前值: {value!r})"
        )
    return value


def _opt_env(name: str) -> str:
    """读取可选环境变量: 空串或缺失返回空 (用于可选角色/免费模型)."""
    return os.getenv(name, "").strip()


# ============================================================
# Provider 配置: QUERY (主 LLM) 与 Embedding 独立必填
# ============================================================
QUERY_BASE_URL = _resolve_env("QUERY_BASE_URL")
QUERY_API_KEY = _resolve_env("QUERY_API_KEY")
EMBEDDING_BASE_URL = _resolve_env("EMBEDDING_BASE_URL")
EMBEDDING_API_KEY = _resolve_env("EMBEDDING_API_KEY")

# ============================================================
# LLM 角色 provider: KEYWORD/EXTRACT 可选, 缺省回退 QUERY
# ============================================================
KEYWORD_BASE_URL = _opt_env("KEYWORD_BASE_URL") or QUERY_BASE_URL
KEYWORD_API_KEY = _opt_env("KEYWORD_API_KEY") or QUERY_API_KEY
EXTRACT_BASE_URL = _opt_env("EXTRACT_BASE_URL") or QUERY_BASE_URL
EXTRACT_API_KEY = _opt_env("EXTRACT_API_KEY") or QUERY_API_KEY

# ============================================================
# LLM 角色模型配置
# QUERY 必填; KEYWORD/EXTRACT 可选 (空 = 回退 QUERY)
# ============================================================
QUERY_FREE_MODEL = _opt_env("QUERY_FREE_MODEL")
QUERY_PAID_MODEL = _resolve_env("QUERY_PAID_MODEL")

KEYWORD_FREE_MODEL = _opt_env("KEYWORD_FREE_MODEL")
KEYWORD_PAID_MODEL = _opt_env("KEYWORD_PAID_MODEL")
EXTRACT_FREE_MODEL = _opt_env("EXTRACT_FREE_MODEL")
EXTRACT_PAID_MODEL = _opt_env("EXTRACT_PAID_MODEL")

EMBEDDING_MODEL_FREE = _opt_env("EMBEDDING_MODEL_FREE")
EMBEDDING_MODEL_PAID = _resolve_env("EMBEDDING_MODEL_PAID")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))
EMBEDDING_MAX_TOKEN_SIZE = int(os.getenv("EMBEDDING_MAX_TOKEN_SIZE", "32768"))
FALLBACK_COOLDOWN = float(os.getenv("FALLBACK_COOLDOWN", "60"))


# ============================================================
# 免费/付费双档状态机 (线程安全)
# ============================================================
class _FallbackCircuit:
    """免费优先; has_free=False 时恒用付费 (免费模型空置场景).

    严格"仅当次回退"语义 (2026-08-11 用户确认):
    - 默认每调用都先试免费 (无冷却窗口).
    - 免费 429 失败 → trip() 置一次性标志 → 仅下次调用强制走付费, 再下一次自动恢复免费 (不依赖时间冷却).
    - has_free=False 时恒用付费.

    注: cooldown 参数保留兼容 (若 >0 则退化为旧时间冷却语义, 暂未用).
    """

    def __init__(self, cooldown: float, has_free: bool) -> None:
        self._cooldown = cooldown
        self._has_free = has_free
        self._lock = threading.Lock()
        self._paid_once: bool = False

    def should_use_paid(self) -> bool:
        with self._lock:
            if not self._has_free:
                return True
            # 仅消费一次性付费标志: 本次判定后即清除, 下次回到免费.
            flag = self._paid_once
            self._paid_once = False
            return flag

    def trip(self) -> None:
        with self._lock:
            self._paid_once = True


def _make_llm_wrapper(
    base_url: str,
    api_key: str,
    free_model: str | None,
    paid_model: str,
    circuit: _FallbackCircuit,
) -> Callable[..., Any]:
    """构造一个带 free/paid fallback 的 LLM 异步包装函数.

    供 LightRAG 的 llm_model_func 或 role_llm_configs 使用 (角色角色).
    free_model=None 时直接用 paid.
    """

    async def _wrapper(
        prompt: str,
        system_prompt: str | None = None,
        history_messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> str:
        use_paid = circuit.should_use_paid()
        model = paid_model if use_paid else free_model
        call_kwargs = {
            "prompt": prompt,
            "model": model,
            "base_url": base_url,
            "api_key": api_key,
            "system_prompt": system_prompt,
            "history_messages": history_messages,
            **kwargs,
        }
        try:
            return await openai_complete_if_cache(**call_kwargs)
        except Exception:
            # openai SDK 已对 429 内建重试; 这里捕获重试耗尽后的限流, 付费兜底.
            circuit.trip()
            call_kwargs["model"] = paid_model
            return await openai_complete_if_cache(**call_kwargs)

    return _wrapper


# ---- 各角色电路 ----
_QUERY_CIRCUIT = _FallbackCircuit(FALLBACK_COOLDOWN, has_free=bool(QUERY_FREE_MODEL))
_KEYWORD_CIRCUIT = _FallbackCircuit(
    FALLBACK_COOLDOWN, has_free=bool(KEYWORD_FREE_MODEL)
)
_EXTRACT_CIRCUIT = _FallbackCircuit(
    FALLBACK_COOLDOWN, has_free=bool(EXTRACT_FREE_MODEL)
)
_EMBED_FALLBACK = _FallbackCircuit(FALLBACK_COOLDOWN, has_free=bool(EMBEDDING_MODEL_FREE))


def build_llm_func():
    """返回 QUERY 主模型的 LightRAG llm_model_func (用 QUERY 独立 provider)."""
    return _make_llm_wrapper(
        base_url=QUERY_BASE_URL,
        api_key=QUERY_API_KEY,
        free_model=QUERY_FREE_MODEL if QUERY_FREE_MODEL else None,
        paid_model=QUERY_PAID_MODEL,
        circuit=_QUERY_CIRCUIT,
    )


def build_role_llm_configs() -> dict[str, Any] | None:
    """构造 LightRAG 的 role_llm_configs (仅含已配置的 KEYWORD/EXTRACT 角色).

    未配置的角色返回 None → LightRAG 回退到 base llm_model_func (QUERY).
    """
    configs: dict[str, Any] = {}
    if KEYWORD_FREE_MODEL or KEYWORD_PAID_MODEL:
        configs["keyword"] = {
            "func": _make_llm_wrapper(
                base_url=KEYWORD_BASE_URL,
                api_key=KEYWORD_API_KEY,
                free_model=KEYWORD_FREE_MODEL if KEYWORD_FREE_MODEL else None,
                paid_model=KEYWORD_PAID_MODEL or QUERY_PAID_MODEL,
                circuit=_KEYWORD_CIRCUIT,
            ),
            "kwargs": {
                "base_url": KEYWORD_BASE_URL,
                "api_key": KEYWORD_API_KEY,
            },
        }
    if EXTRACT_FREE_MODEL or EXTRACT_PAID_MODEL:
        configs["extract"] = {
            "func": _make_llm_wrapper(
                base_url=EXTRACT_BASE_URL,
                api_key=EXTRACT_API_KEY,
                free_model=EXTRACT_FREE_MODEL if EXTRACT_FREE_MODEL else None,
                paid_model=EXTRACT_PAID_MODEL or QUERY_PAID_MODEL,
                circuit=_EXTRACT_CIRCUIT,
            ),
            "kwargs": {
                "base_url": EXTRACT_BASE_URL,
                "api_key": EXTRACT_API_KEY,
            },
        }
    return configs if configs else None


# ============================================================
# Embedding (独立 provider, 免费优先 + 付费兜底)
# ============================================================
async def _embed_with_fallback(texts: list[str]) -> list[list[float]]:
    use_paid = _EMBED_FALLBACK.should_use_paid()
    model = EMBEDDING_MODEL_PAID if use_paid else EMBEDDING_MODEL_FREE
    try:
        return await openai_embed.func(
            model=model,
            base_url=EMBEDDING_BASE_URL,
            api_key=EMBEDDING_API_KEY,
            texts=texts,
        )
    except Exception:
        _EMBED_FALLBACK.trip()
        return await openai_embed.func(
            model=EMBEDDING_MODEL_PAID,
            base_url=EMBEDDING_BASE_URL,
            api_key=EMBEDDING_API_KEY,
            texts=texts,
        )


def build_embedding_func() -> EmbeddingFunc:
    return EmbeddingFunc(
        embedding_dim=EMBEDDING_DIM,
        max_token_size=EMBEDDING_MAX_TOKEN_SIZE,
        func=wrap_embedding_func_with_attrs(
            embedding_dim=EMBEDDING_DIM,
            max_token_size=EMBEDDING_MAX_TOKEN_SIZE,
        )(_embed_with_fallback),
    )