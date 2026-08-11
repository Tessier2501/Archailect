# My Novel RAG 优化提案 (Improvements Proposal)

> 只读规划文档, 不内嵌代码 (权威代码以 src/ 为准). 每项均附: 问题/证据 → 方案 → 验证方法 → 成本.
> 创建: 2026-08-11. 状态: 待用户审批, 未实施.
> 依据: 全量代码审查 + PLAN.md 三轮诊断 (22 次检索实测 + rerank AB + top_k=50 原始候选池验证).
> 修订: 2026-08-11 采纳 IDE agent 评审疑点 A (静态扩展自答式泄露) / 疑点 B (基线口径不一致).

---

## 总览 (优先级排序)

| # | 名称 | 直击问题 | 证据强度 | 成本 | 状态/建议 |
|---|------|---------|---------|------|------|
| 13 | **生产端查询改写 (聚焦陈述化)** | 反事实召回失败 (真实痛点) | ★★★★★ (七轮收口) | 低 (0 重建, +1 短调用) | **对症方案, 待批实施** |
| 1 | 检索串扩展 (静态/原样 HyDE) | — | — | — | **证伪 (原样反事实全 MISS)** |
| 8 | Embedding 语义强度直测 | — | — | — | 已执行, 结论被第七轮修正 |
| 9 | ~~chunk 合并调权~~ | — | — | — | **排除 (mix #3, 合并层无害)** |
| 10 | ~~换 embedding/4b/8b~~ | — | — | — | **论据不足 (广度92%+Q_ORIG三模型全MISS)** |
| 12 | Embedding 尽职调查 | — | — | — | 已执行 (引出第七轮裁决) |
| 2 | System Prompt 指称权威+因果引导 (F2+F3) | 召回到但不会用 | ★★★ | 低 | 与 13 同批 (可选增强) |
| 3 | 诊断工具修复 (file_path 噪声度量) | 诊断数据失真 | ★★★ | 低 | 可选 |
| 4 | ~~缓存加载竞态加固~~ | — | — | — | 撤回 (非缺陷) |
| 5 | 流式错误帧 | 客户端断开后无错误信号 | ★★ | 低 | 可选 |
| 6 | 书源标记 file_paths (E) | 无法按书过滤 | ★★★ | 高 (重建) | 暂缓 |
| 7 | 实体归一化 (C) | ß 变体分裂 | ★★ | 高 (重建) | 暂缓 |

**根因收口 (七轮诊断)**: 用户原样反事实提问 (Q_ORIG) 在任何 embedding 下都检索不到答案段 (语义错配); 但**聚焦的陈述化检索串能稳定命中** (mix #3). 对症 = 生产端查询改写 (提案 13), 0 重建.

---

## 提案 1: 检索串扩展 (Retrieval Query Expansion) — ❌ 已证伪 (2026-08-11 第四轮)

**证伪结论**: `scripts/probe_query_expansion.py` 四臂同口径对照全 MISS (基线/静态/LLM改写/HyDE 均 0/3). 连语义上几乎等同答案转述的 HyDE 段也未能经向量通道带回锚点 chunk → **失败不在检索串措辞层, 而在 chunk 向量召回/候选合并层**. 本提案不接入生产. 详见 PLAN.md §5.2.

### (原方案存档备查, 已不采用)

### 问题 (F4: 反事实召回层覆盖不足)
PLAN §5.1 第三轮已确证: Q1 反事实问题的答案段 **根本未进入候选池** (top_k=50/hybrid 无 rerank 仍 MISS). 根因是问句与答案段的语义结构错配 — 问句 "为什么没做 X" 与答案段 "决定做 Y 因为..." 向量相似度低, 实体图检索召回 Behemoth 族但 chunk 通道被字面不匹配拖死.

### 方案 (双臂, 同一实验同测)

**1a. 静态多通道扩展 (零 LLM 成本, 推荐先做)**
对检索串做确定性规则扩展, 用 `aquery_data` 跑 **多组扩展串**, 合并候选池后再送 LLM:
- 原串 (保留): 完整用户问题.
- 扩展 (实体锚定): 剥离疑问/反事实外壳, 仅保留问题中已出现的词元或其直译. 例: Q1 → `Node 1211/BCC 1211 gel ßehemoth Lenie Clarke Beebe nuke`.
  **泄露约束 (IDE 评审疑点 A 修正)**: 静态扩展只允许问题派生词元 (实体名/内容名词及直译); 严禁答案侧词汇 (copy/move/decide/biosphere/self-sustaining 等) — 否则实验自证有效, 且生产端无答案可抄, 不可泛化.

实现要点: 复用 LightRAG 已有的 `hl_keywords`/`ll_keywords` 机制 **但不做别名注入** (PLAN §2.3 证伪 V2), 而是 **多查询并行 + 候选并集**. 每查询独立走完整 aquery_data, 合并去重 chunks (按 chunk_id), 再交由现有 LLM 生成. 不绕开 LLM 关键词提取.

**1b. LLM 查询改写 / HyDE (每查询 +1 次短调用)**
运行时由 LLM 把反事实问句改写为陈述式检索串, 或生成假想答案段 (HyDE) 再检索.
**通用性说明 (回应疑点 A)**: 生成时 LLM 只见问题、不见语料, 即使生成串偶合答案词也是生产可复现行为, 不构成实验泄露 — 这正是"自答式检索"的合法泛化形态, 顺带获得中文问→英文语料的跨语言桥接. 生成串全文打印供人工泄露审查.
直击 F4, 每查询多一次短 LLM 往返 (+延迟 +token, 量级数百 token/次).

### 验证方法 (同口径基线, IDE 评审疑点 B 修正)
- `scripts/probe_query_expansion.py` 四臂 **同一进程同一生产配置** (mix/top_k=20/chunk_top_k=12/rerank 随 .env):
  ① Q_ORIG 基线 ② 静态问题派生扩展 ③ LLM 改写 ④ LLM HyDE. 报告三锚点 (identity_17269 / choice_17337 / answer_17345) 命中.
- 判定标准: 扩展臂命中 ≥2 且基线臂 MISS → 扩展有效; 静态臂有效则零成本, LLM 臂有效则每查询 +1 短调用. **静态臂禁答案词, LLM 臂生成串全文打印供泄露审查.**
- 基线说明: `diagnose_q1_raw.py` (hybrid/top_k=50) **不再作基线** (口径与生产不一致), 仅作候选池上限核查.
- 回归: README §9 验收 1-7 + PLAN Q2 (澄清句须仍 HIT, 防扩展干扰简单问题).

### 成本
- 中: 0 索引重建; 1a 每查询多 1-2 次 aquery_data 检索 (无 LLM 生成); 1b 每查询多 1 次短 LLM 调用 (数百 token). 实验阶段用脚本验证, 生产接入时再评估延迟.

---

## 提案 2: System Prompt 模板增加 F2 + F3 指令

### 问题
PLAN §5.1 用户实测: LLM 召回到 1211 决策段描述后, 纠结 "Node 1211/BCC 是 network node 不是 gel", 质疑用户指称, 未完成反事实推理. 模板缺两条约束.

### 方案
改 `build_rag_system_prompt()` 的包装模板 (不改用户原文转义逻辑), 追加两条固定指令:
- **F2 指称权威**: "The user is the authority on what they refer to. Never challenge or re-classify a user's term (e.g. calling Node 1211/BCC a 'gel') based on its KB entity type; if the user names it, treat their designation as correct and answer the intent."
- **F3 因果/决策段直接引用**: "When the context contains a decision, choice, or causal account relevant to the query, quote and reason from it directly; prefer citing the entity's stated rationale over speculation."

### 验证方法
- 复跑 PLAN §5.1 的用户实测路径 (澄清句+反事实合为单条消息), 检查回答是否 (a) 不再质疑 "凝胶" 指称, (b) 引用 1211 决策逻辑.
- 回归 README §9 验收 3 (System Prompt 透传 "深渊向导" 仍生效) + 验收 11 (prompt 含 `{}` 不抛 KeyError).

### 成本
低 (改一个函数 + 回归测试).

---

## 提案 3: 修复诊断脚本的 file_path 噪声度量

### 问题
`diagnose_retrieval.py::_analyze_hit` 用 `file_path` 判跨书噪声, 但 PLAN §2.3 已证全部 chunk `file_path='unknown_source'` → `cross_book_ratio` 恒 1.0, 该列数据无意义, 误导诊断结论.

### 方案
- `_analyze_hit` 改为按内容指纹 (如 chunk 含 "Lenie Clarke"/"Beebe" 等 Starfish 专有名 vs "Maelstrom" 等书名关键词) 近似判书源, 或直接删除 cross_book 列并在报告标注 "file_path 缺失, 书源不可考 (待提案 6 重建后恢复)".
- 同步在 diagnose_report.json 重跑时刷新该字段.

### 验证方法
复跑 `python -m scripts.diagnose_retrieval`, 确认报告不再输出恒 1.0 的伪指标.

### 成本
低 (改诊断脚本, 不碰生产).

---

## 提案 4: 缓存加载竞态加固 (对应验收 8-10) — 修正: 非缺陷

### 修正说明 (2026-08-11, 自查复核撤回)
初审曾判 `_get_rag_instance` 慢路径 `_rag_locks[model] = asyncio.Lock()` 检查与赋值非原子 → 并发首访同一新 model 会建多把锁 → 重复实例化 (验收 8 风险). **复核后撤回**: asyncio 单线程事件循环下, 同步代码块 `if model not in _rag_locks: _rag_locks[model] = asyncio.Lock()` 不含 `await`, 两个协程之间不存在交错点, 字典写入本身原子, 该竞态实际不成立.
同理 `_evict_if_needed` 遍历 `_rag_instances` 与快路径 `move_to_end` 的"迭代中修改"风险, 也因迭代与修改之间无 `await` 交错而不成立.

**本提案撤回, 不改代码.** 验收 8/9/10 仍为**待实测**项 (README §9), 但其验证目的是"实测确认无并发缺陷"而非"修复已知缺陷"; probe 脚本仍可用于确认性测试 (验收 8: 并发首访仅实例化一次; 9: LRU 淘汰不中断在途; 10: 流式中断无泄漏), 属可选的回归确认, 非修复.

---

## 提案 5: 流式错误帧

### 问题
`_stream_response` 仅在 `CancelledError` 时清理, 若 LLM 流中途因限流/上游错误抛普通异常, 客户端只见连接断开, 无错误语义.

### 方案
在 `_stream_response` 增加 `except Exception` 分支: 发一帧 `{"error": {...}}` 的 SSE 再 `data: [DONE]`, 与 OpenAI 流式错误约定对齐.

### 验证
probe 脚本模拟上游抛错 (mock token_iter), 断言客户端收到 error 帧 + [DONE].

### 成本
低.

---

## 提案 6: 书源标记 file_paths (E) — 暂缓

### 内容
builder `ainsert` 传 `file_paths=[str(txt)]`, 索引记录每段书源. 恢复按书过滤/噪声量化能力.

### 为何暂缓
- 需 **删除并重建 `storage/rifters` 索引** (重跑 LLM 提取 + token 计费, 且须先移回 20MB llm_response_cache 存档).
- 当前瓶颈在召回层 (提案 1), 书源标记不直接改善回答. PLAN §5 已将 E 与 C 合并决策.
- 建议: 待提案 1 验证召回改善后, 若仍需按书过滤/评估, 再与提案 7 一次性重建.

---

## 提案 7: 实体归一化 (C) — 暂缓

### 依据
PLAN §3 根因排序第 3 (★★★): Q3 直接提问实体召回已成功 (ßehemoth 109 次命中), 分裂对直接提问影响有限; 重建成本高. 维持 PLAN "暂缓/可选" 结论, 与提案 6 同批决策.

---

## 提案 8: Embedding 语义强度直接验证 — 决定性前置实验（已执行, 2026-08-11 第五轮）

**动机**：第四轮实验把根因从"检索串措辞"排除，钉死在 chunk 向量召回/候选合并层。需区分是合并层淹没（提案 9）还是 embedding 语义弱（提案 10）。

**执行结果**（IDE agent 用 nano-vectordb 官方加载语义复算；`data[i]["vector"]` 为压缩快照不可直读，须 `buffer_string_to_array(matrix).reshape(-1, dim)`):

| 查询 | chunk-102 (identity) | chunk-104 (choice+answer) | top-12 |
|---|---|---|---|
| Q_ORIG 反事实 | 356/433 (0.4253) | 335/433 (0.4398) | MISS |
| 静态问题派生 | 409/433 (0.2613) | 160/433 (0.4104) | MISS |
| HyDE 假想段 | 422/433 (0.2979) | **37/433** (0.4542) | MISS |
| Q2 对照 | 160/433 (0.3087) | 49/433 (0.3478) | MISS |

**判定：指向提案 10 (embedding 语义弱, 需重建/换模型), 排除提案 9。**
- 连语义几乎等同答案的 HyDE 段，chunk-104 也仅 37/433（前 8.5%),chunk-102 垫底 422/433 → 锚点进不了 naive top-12,**合并/截断层没有机会挤出它**。
- top 分值仅 0.26-0.45,433 条 chunk 余弦区分度整体弱 → qwen3-embedding-0.6b 对长句科幻叙事+ß 变体表征不足。

**Q2 矛盾澄清**:PLAN §2.2 "Q2 naive 命中 17269" 实为命中 `node1211_darkness` 锚点（"since the Darkness",chunk-103 区域），非 chunk-102/104 指纹；故 IDE 复算 Q2 对照对 102/104 MISS 不构成矛盾。但 chunk-104 在 Q2 下仍 49/433 靠后，不推翻"embedding 对 Q1 决策段弱"的结论。

**成本**：低（只读 + 4 次短 embed 调用）。

---

## 提案 12: Embedding 尽职调查（广度 + 维度）— 重建前最后把关

**动机**：提案 8 只证了"0.6b 对 Q1 决策段弱"。重建前必须确认两件事，否则可能白重建：
1. **广度**:0.6b 是对全库普遍弱，还是只对这一段弱？（若仅局部弱，换模型收益存疑）
2. **维度**：换同系列的 4b/8b 是否必须改 `EMBEDDING_DIM`？（若保持 1024 则索引向量可复用，免重建；若维度变了则必重建）

**做法**(`scripts/probe_embed_broad.py`，只读 + 若干短 embed 调用）:
- **广度抽样**：对全库 433 个 chunk，用一批"已知答案位置"的探针查询（覆盖三卷、多类问题：人物/地点/事件/因果/反事实）测 0.6b 的命中率分布，而非只盯 Q1 一段。产出"0.6b 全库命中率基线"。
- **维度核查**：确认 qwen3-embedding-4b/8b 的 embedding 维度是否可设为 1024（与现索引一致）或为 2560/4096。查 provider 文档/实测一次短 embed 调用看返回维度。
- **对比试算**：用 4b 或 8b 对 Q1 的 4 条查询 + 锚点 chunk 重新嵌入，看锚点排名是否从 37/422 提升到 top-12。若提升显著 → 换模型值得重建；若提升有限 → 转 BM25 混合召回或接受现状。

**成本**：低-中（只读索引 + 若干 embed 调用；若需 4b/8b 试算，单次 embed 成本高于 0.6b 但仍属小额）。

**判定产出**：给"换 embedding 是否值得重建"一个量化依据，避免盲目重建。

---

## 提案 9: chunk 通道加权 / 检索合并调优 — ❌ 已排除 (2026-08-11 第七轮确证)

**排除依据 (七轮收口)**: 提案 8 曾以"锚点进不了 naive top-12"判排除; 第七轮固定 HyDE 串三通道对照显示 **mix 生产链路 chunk-104 = #3**, 实体+关系合并未挤掉向量召回反而抬升 → 合并层无害. 无论哪种口径, 调合并权重均无必要. 本提案不实施.

---

## 提案 10: 换 embedding 模型 / 加 BM25 混合召回 — ❌ 论据不足, 不重建 (2026-08-11 第七轮确证)

**不实施依据 (六+七轮收口)**:
- 0.6b 全库广度基线 92%, 仅对反事实/决策语义错配类局部弱, 非全库性问题.
- Q_ORIG (用户原样反事实) 在 0.6b/4b/8b 下全部 MISS → 换 embedding 解决不了真实痛点.
- 4b/8b 增益顶点在 Q2对照/静态臂 (非生产形态), 对真实痛点无直接增益; HyDE 下 0.6b #10 → 4b/8b #1 仅边际提升.
- 第七轮: 拼接版 HyDE 串在 mix 链路 #3 HIT, 证明 0.6b+现有链路已够用, 关键在检索串陈述化聚焦, 非 embedding 模型.

本提案 (含 4b/8b 换模型、BM25 混合召回) 均不实施. 若未来发现 0.6b 在其它查询类型上系统性失效, 再重启评估.

---

## 提案 11: System Prompt 注入实体指称别名映射 — 补充观察

**依据**：第四轮日志显示图谱中 `Node 1211/BCC` 已正确拆出 `Node 1211` 与 `BCC`，且各臂实体召回正常。残缺的指称映射主要在 LLM 生成端（把用户的"凝胶/1211/Node 1211/BCC/ß ehemoth"等措辞对齐到同一实体）。可在 system prompt 注入一层轻量"实体别名表"（从图谱高频实体/变体自动抽取），供 LLM 对齐指称。**此项不解决召回缺失**，仅改善"召回到实体后 LLM 的指称对齐"，优先级低于 8/9/10，可与提案 2 合并实施。

**成本**：低-中。

---

## 提案 13: 生产端查询改写 (聚焦陈述化) — ✅ 对症方案, 待批实施

**直击**: 用户原样反事实提问 (Q_ORIG) 检索不到答案段 (七轮收口根因: 语义错配). 聚焦的陈述化检索串已实证稳定命中 (mix #3).

**方案**: 在 `api_server.py` 的 LightRAG 查询前, 加一道**服务端 LLM 查询改写**:
1. 用 QUERY 模型把用户问题改写为**聚焦的陈述式检索串**;
2. 用改写串跑 `aquery` 检索 (配合现有 mix 链路), 实体+关系+向量合并已实证会把答案段抬到前列;
3. **生成回答时仍用用户原始问题** (改写串只用于检索, 不替换用户提问), 保证回答对准用户原意.

**改写指令的关键约束 (第七轮复核实证, 决定成败)**:
- **禁止想象延伸 / 补充背景 / 推演** — 第四轮完整版 HyDE 因含 bioluminescence/nuclear warhead/methane hydrate 等想象词, 把检索引偏 (990 relations 噪声淹没答案段).
- **仅基于问题中已出现的实体与事实**, 聚焦"实体名 + 具体属性 + 动作结果"的少而准形态 (拼接版 #1 HIT 的形态: hyper-specific/calibrated/electromagnetic/metabolic/weapon/trigger).
- 保留反事实/因果的**语义内核** (谁、没做成什么、为何), 只去疑问外壳, 不添加问题外信息.

**伪代码逻辑 (实现细节待批后定)**:
```
user_query = 提取的用户原始问题
rewritten = await llm(user_query, system_prompt=REWRITE_FOCUSED_SYS)   # 聚焦改写
param = QueryParam(mode="mix", top_k=20, chunk_top_k=12, ...)
result = await rag.aquery(rewritten, param=param, system_prompt=rag_system_prompt_with_original_query)
# 回答生成上下文带原始问题, 检索串用 rewritten
```
注意: LightRAG `aquery(query)` 的 query 同时用于检索与生成上下文. 需确认 1.5.6 是否支持检索串与生成问题分离 (如 `QueryParam.user_prompt` 或分两步); 若不支持, 则在 system_prompt 中明确"用户真实问题是 X, 检索串 Y 仅为检索辅助", 或评估改写串直接作为 query 的回答质量. **此为实现前需 IDE 验证的唯一技术点.**

**成本**: 低 (0 重建; 每查询 +1 次短 LLM 改写调用, 数百 token; 受免费/付费 fallback 保护).

**验证方法 (实施后回归)**:
- PLAN Q1 反事实案例: 改写后检索应命中 chunk-104 区域, 回答引用 1211 决策逻辑.
- PLAN Q2 澄清句: 仍须正常 (改写不破坏简单问题).
- README §9 验收 1-7 全过.
- 与提案 2 (F2/F3 指称权威+因果引导) 同批实施, 共同改善"召回到且会用".

---

## 决定性前置实验脚本（可直接落盘为 `scripts/probe_embedding_channel.py`)

```python
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
```

## 建议实施顺序与复测计划

**诊断阶段已全部完成 (七轮收口)**: 根因 = 用户原样反事实提问的检索串语义错配; 对症 = 生产端聚焦查询改写 (提案 13), 0 重建. 换 embedding (10)、合并调权 (9)、检索扩展原样 (1) 均已证伪/排除.

**阶段 1 (待批实施)** — 提案 13 (+ 可选 2/3/5 同批)
1. 先解决唯一技术点: 确认 LightRAG 1.5.6 `aquery` 检索串与生成问题能否分离 (`QueryParam.user_prompt` / system_prompt 标注 / 分两步). IDE 只读探查 `lightrag/operate.py` 签名即可, 不动生产.
2. 实施提案 13 查询改写进 `api_server.py` (聚焦改写指令, 禁止想象延伸).
3. (可选同批) 提案 2 (F2/F3 指称权威+因果引导) 改善"召回到且会用"; 提案 3 (诊断修复); 提案 5 (流式错误帧).
4. 回归: PLAN Q1 反事实 (应命中 chunk-104 并引用 1211 决策) + Q2 澄清句不退化 + README §9 验收 1-7.

**阶段 2 (暂缓)** — 提案 6+7 (书源标记/实体归一化), 需重建索引, 与提案 13 无依赖, 视后续需求单独决策.

**不做项 (已证伪/排除)**
- 检索串扩展/别名注入 (提案 1, PLAN §2.3 V2 + 第四轮证伪).
- 换 embedding 4b/8b、BM25 混合召回 (提案 10, 第六/七轮论据不足).
- 合并层调权 (提案 9, 第七轮 mix #3 无害).
- 缓存竞态加固 (提案 4, asyncio 单线程下非缺陷).
- 升级 lightrag-hku (§10.2 #1 锁死 1.5.6).
- 接入新 rerank provider (rerank 只排序不扩召回, 非瓶颈).

---

## 给 IDE Agent 的行动指示

```bash
cd ~/Archailect && ~/anaconda3/envs/myenv/bin/python -m scripts.probe_merge_vs_semantic
```

只读：A 读 vdb 矩阵（无 LLM 调用）;B/C 走 `aquery_data`（只检索不生成，mix 关键词提取受缓存保护）。回传**完整终端输出**，重点是 A/B/C 三行各自的 HIT/MISS 和锚点名次，以及末尾判定行。

异常：若 `aquery_data` 返回结构里 chunk 无独立 id，脚本已用内容反查 chunk_id 处理；若抛异常发完整 traceback，勿自行改脚本。