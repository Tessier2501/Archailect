"""双库隔离受控实验: 交替查询 test / test2, 记录答案与耗时.

用途: 区分 test 首次查询异常返回 test2 内容的现象是
  A) 运行时缓存/命名空间串扰 (查过 test2 后 test 命中其缓存), 还是
  B) LLM 偶发幻觉 (无检索上下文时生造).
判定: 若 test 答案内容与 test2 内容逐字一致且耗时极短 (<1s), 高度疑似缓存串扰;

用法: python -m src.dual_probe [--rounds 3]
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request
from datetime import datetime, timezone

API = "http://localhost:8000/v1/chat/completions"
QUESTIONS = [
    "这本书的主角是谁? 一句话回答",
    "这本书的故事设定在哪里? 一句话回答",
]
OUT = "/tmp/dual_probe.log"


def post(model: str, user: str) -> tuple[str, float]:
    body = json.dumps(
        {"model": model, "messages": [{"role": "user", "content": user}], "stream": False}
    ).encode("utf-8")
    req = urllib.request.Request(
        API, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    start = time.monotonic()
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    elapsed = time.monotonic() - start
    content = data["choices"][0]["message"]["content"]
    return content, elapsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()

    lines: list[str] = []
    # 交替: test -> test2 -> test -> test2 ... (每模型 rounds 次)
    for i in range(args.rounds):
        for model in ("test", "test2"):
            for q in QUESTIONS:
                content, elapsed = post(model, q)
                line = (
                    f"{datetime.now(timezone.utc).isoformat()} "
                    f"round={i + 1} model={model} elapsed={elapsed:.2f}s "
                    f"q={q[:12]!r} ans={content!r}"
                )
                lines.append(line)
                print(line, flush=True)

    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\n[OK] 已写入 {OUT}")


if __name__ == "__main__":
    main()