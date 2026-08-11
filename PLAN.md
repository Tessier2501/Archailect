# My Novel RAG 问题诊断与改进方案

> 本文档只记录已发现的问题、诊断依据与改进方案, 不内嵌参考代码 (权威代码以 src/ 为准).
> 创建日期: 2026-08-11. 状态: 诊断完成, 改进未实施 (待决策).

---

## 1. 背景与问题现象

实际使用中发现 RAG 经常无法正确回答书籍问题, 用户提供两个真实失败案例:

- **案例 1 (反事实问题)**: 问 "为什么智能凝胶 Node 1211/BCC 即使被编程为阻止 Behemoth, 仍然未能把 Lenie Clarke 消灭在 Beebe 站? 在她调查海底核弹的时候, 将核弹引爆, 不就能有效地遏制 Behemoth 吗?"
  - RAG 回答 "Node 1211/BCC 在知识库中不存在", 实际该实体与答案均在原文 (Starfish 第 17269/17345 行).
- **案例 2 (多轮澄清)**: 问 "在提及凝胶时, 我特指 Node 1211/BCC 或 1211."
  - RAG 回答确认了实体, 但**完全没有上下文能力**: 忽略了上一轮曾问过 Node 1211/BCC.

用户补充: ßehemoth 与 Behemoth 在文中均出现, 指同一实体, RAG 理应识别.

---

## 2. 诊断方法与证据链

诊断对象: `storage/rifters/` (Rifters 三卷合库, 约 2.1MB 纯文本, 索引含 66MB 关系向量).

### 2.1 索引级证据 (只读检查 kv_store_*/graphml)

1. **实体归一化失败**: 图谱中同一实体分裂为多个拼写变体 — `ßehemoth` / `Behemoth` / `ehemoth` / `ßebehemoth` / `ßhehemoth`; `Node 1211/BCC` 与 `1211` 是两个独立实体.
2. **答案存在性**: 案例 1 的答案原文存在 (17269 行 "Node 1211/BCC had been solving its whole life..."; 17345 行 "1211 被编程搅乱 ßehemoth 环境, 但无法根除条件, 转而复制搬走 ßehemoth") — 确证为检索失败而非知识缺失.

### 2.2 检索级证据 (aquery_data 实测, 22 次)

对 4 组问题 × 3 种 mode (hybrid/naive/mix) × 2 种关键词变体 (V1 走 LLM 提取 / V2 显式注入别名) 实测:

| 组 | 问题 | 结果 |
|---|---|---|
| Q1 | 案例 1 反事实原问题 | 全部 6 次 PARTIAL/MISS, 无一次命中答案行; 实体层虽召回到 Behemoth 族, 但 final chunks 全为无关段落 |
| Q2 | 案例 2 澄清句 (单轮) | **检索成功**: hybrid V1 实体命中 Node 1211/BCC (关系 14 次), chunk 命中 Starfish 锚点 |
| Q3 | "What is Behemoth?" (控制组) | 实体层成功 (ßehemoth 关系命中 109 次), 但 final chunks 无 17269 锚点 — 实体召回与 chunk 召回脱节 |
| Q4 | 多轮指代 "它为什么没消灭 Lenie Clarke" | 4 种组合全 PARTIAL; conversation_history 有无对检索结果无影响 (证: 1.5.6 的 history 仅作 LLM 生成上下文, 不参与检索) |

**关键对照发现**: Q2 的 `naive` (纯向量) 模式能命中 17269 锚点, 而 Q1 的字面不匹配问题在全部 mode 下均无法命中答案 — 证明向量通道可找回 Q2 类段落, 但对 Q1 反事实字面错配无能为力.

### 2.3 结构性证据

- **所有 chunk 的 file_path = 'unknown_source'**: `builder.py` 的 `ainsert(content)` 未传 `file_paths`, 索引无书源标记. 后果: (1) 无法按书过滤; (2) 诊断中 cross_book_ratio 全部失真为 1.0, 无法量化跨书噪声.
- **别名注入反效果 (V2)**: 显式注入 6 个别名到 `hl_keywords` 后, Q2 实体命中 25→13、39→12、HIT→PARTIAL — 注入绕过了 LLM 关键词提取, 干扰图谱检索权重分配, 适得其反.

---

## 3. 根因排序

| # | 根因 | 针对案例 | 证据强度 |
|---|---|---|---|
| 1 | **多轮上下文丢失** — api_server 只取最后一条 user 消息, 未传 conversation_history | 案例 2 | ★★★★★ Q2 检索成功但生产失败, 差距在上下文传递环节 |
| 2 | **反事实/矛盾式提问的检索语义错配** — 检索串与答案段字面不匹配, 实体图检索无法召回 | 案例 1 | ★★★★ Q1 全部 mode 失败 |
| 3 | **实体归一化分裂** — ß 变体 5+ 形式各自独立 | 渗透两案例 | ★★★ 对直接提问影响有限 (Q3 实体召回成功), 对长问题有干扰 |
| 4 | 三本合并建库放大检索噪声面 | 次要 | ★★ 66MB 关系图增大噪声, 非主因 |
| 5 | file_path 缺书源标记 (结构限制) | 非直接 | ★★★ 是无法按书过滤/量化噪声的根 |

文本量 (2.1MB) 本身不是问题; "三本合并"是次要噪声源, 非根因.

---

## 4. 改进方案 (四项)

### A. 改进检索策略 (针对反事实语义错配)

- **内容**: 默认 `mode` 从 `hybrid` 切换 `mix` (kg+向量双通道, 覆盖面更广); 反事实类长问题动态调大 `top_k/chunk_top_k` (如 12→20); `enable_rerank=False` 消除无模型空转.
- **预期依据**: 诊断中 Q2 的 naive (纯向量) 模式命中 17269 锚点, 说明向量通道可找回这类段落, mix 让向量通道稳定参与; 低成本 (改 QueryParam), 需回归验证.

### A.1 Rerank 模型机制调查 (方案 A 可选扩展)

Rerank (精排) 是两段式检索的第二段: 召回阶段取回候选池后, 对候选做二次相关性打分并重排, 只把最相关的 top_n 送入 LLM 上下文. **Rerank 不扩大召回** — 召回阶段未找回的段落, rerank 永远不会见到.

- **触发链**: 构造参数 `rerank_model_func` (lightrag.py:741, 默认 None) → 查询时 `apply_rerank_if_enabled` (utils.py:5470) 按 `enable_rerank` 调用; 未配置 rerank 函数时仅打警告并原样返回候选 (无害空转, 即诊断日志中反复出现的 WARNING 来源).
- **签名契约**: LightRAG 以关键字调用 `rerank_func(query=..., documents=[doc文本], top_n=...)`, 要求返回 `[{"index": int, "relevance_score": float}, ...]` (utils.py:5515).
- **内建 Provider**: `jina_rerank` (multilingual, 适配中英混合) / `cohere_rerank` / `ali_rerank` (rerank.py), 均 async 且已按新格式返回, 可直接作为 rerank_model_func.
- **对本项目的意义**:
  - 能改善: 候选池内无关 chunk 稀释答案 (Q1/Q3 "实体召回到 Behemoth 族但 final chunks 混入无关段落") — 精排把相关段提前.
  - 不能改善: Q1 反事实的根因是召回失败 (候选池无 17269 答案段), rerank 无从重排; Q2 检索已成功, 缺多轮上下文. 故 rerank 优先级低于 B/A, 属锦上添花.
- **接入成本**: 需上游提供 rerank API (或另配 Jina/Cohere/Aliyun key); config.py 增 rerank 工厂, LightRAG 构造传 rerank_model_func; 零索引重建, 每查询多一次短请求.
- **决策**: 当前状态不做; 待 B/A 实施后再评估是否接入.

### B. 多轮上下文透传 (针对指代丢失) — 确定性最高

- **内容**: api_server 提取最近 N 条消息传 `QueryParam.conversation_history`, 让 LLM 生成时拥有前文语境; 将前一轮并入检索串缓解指代.
- **预期依据**: Q2 检索链路成功 (实体/关系/chunk 均命中), 生产失败 100% 在上下文传递环节; 不重建索引, 立即生效.

### C. 实体归一化预处理 (针对 ß 变体分裂) — 降级可选

- **内容**: 建图前对原文归一化 (ßehemoth→Behemoth), 或查询端做 Unicode/大小写鲁棒化.
- **预期依据 (降级)**: Q3 实体召回已成功 (ßehemoth 109 次命中), 分裂对直接提问影响有限; 重建索引成本极高 (重跑 LLM 提取 + token 计费), 除非发现系统性失败否则不建议.

### E. 书源标记 (结构性改进)

- **内容**: builder 的 `ainsert` 增加 `file_paths=[str(txt)]`, 索引记录每段书源.
- **预期依据**: 未来按书过滤/分卷评估的前提; 当前不直接改善回答, 与 C 共享重建成本, 可合并决策.

> 注: 曾考虑的"D 别名注入"方案 (显式注入实体别名到 hl_keywords) 已被诊断证伪 — 注入后 Q2 实体命中 25→13、HIT→PARTIAL, 因绕过了 LLM 关键词提取并干扰图谱检索权重分配. 证据见 §2.3, 不纳入改进方案.

---

## 5. 优先级与待办

| 方案 | 成本 | 确定性 | 建议 |
|---|---|---|---|
| B 多轮上下文 | 低 | ★★★★★ | 立即实施 |
| A mix+调参 | 低 | ★★★☆ | 立即实施, 需回归 |
| E 书源标记 | 中 (重建) | ★★★ | 与 C 合并决策 |
| C 实体归一化 | 高 (重建) | ★★☆ | 暂缓/可选 |

**实施前须回归验证**: README §9 验收表 1-7 + 本文档 Q1/Q2 案例. 诊断脚本 (scripts/diagnose_retrieval.py) 保留用于改进前后对比复跑.

---

## 5.1 复测记录: 方案 A/A.1/B 实施后 (2026-08-11 第二次诊断)

### 背景
A (mix+调参), A.1 (rerank 12), B (conversation_history) 实施并重启后, 用户实测反馈失败模式变化:

- 用户将澄清句与反事实问题合为**单条消息**重测 (变相绕过 B 的多轮需求).
- RAG 思考文本显示: **答案段 (1211 决策段, 即 17337-17357 区域) 已被召回** ("Document Chunks 中有一段 1211 的详细叙述…决定选择 Behemoth 还是 biosphere"), 但 LLM 纠结于 "Node 1211/BCC 是 network node 不是 gel", 质疑用户指称, 未完成反事实推理.

### 复测数据 (scripts/diagnose_q1_current.py, 当前生产配置单轮)

- 配置: mode=mix, top_k=20, chunk_top_k=12, enable_rerank=True (rerank 12), conversation_history=[]
- 结果: **答案锚点全 MISS** — identity_17269 / choice_17337 / answer_17345 均未命中; final chunks 全为无关段落 (Joel/Fischer/Lubin 等)

### 结论: 检索已改善但不稳定, 召回层覆盖仍是根因 (2026-08-11 第三轮诊断修正)

**第三轮诊断证据 (scripts/diagnose_q1_rerank_ab.py + diagnose_q1_raw.py):**

1. **rerank off/on × 3 对照** (当前配置 mix/top_k=20/chunk_top_k=12): 6 次全部 MISS — **排除 rerank 挤出答案段**的假设.
2. **原始候选池验证** (hybrid/top_k=50, chunk_top_k=50, enable_rerank=False): 仍无任何锚点 — 答案段 **根本未进入候选池**, 非 rerank/截断/后续处理所致.

**结论修正: 根因是"反事实问题在检索召回层的覆盖不足", 而非此前的"推理层不会用".**
- 用户实测那次 LLM 思考中见到 "1211 决策段描述", 应来自 **KG 实体描述** (entity description 而非 document chunk) + 相邻段落 (chunk 4/5 含 ßehemoth/gel 线索); LLM 据此推理, 但缺少 17269-17357 的原始文本锚点.
- 向量检索对 "为什么没做X / 反事实" 类提问的语义匹配弱: 答案段以陈述句描述 1211 的决策 (select/move), 与问句形成 "矛盾-解释" 结构, 向量相似度不足.
- rerank 12 已生效 (日志证实), 但 rerank 只改善排序、不扩大召回 — 候选池无锚点时无从发力.

| 观察 | 含义 |
|---|---|
| rerank off/on 全 MISS + top_k=50 无锚点 | **答案段未进入候选池** = 召回层失败, 与 rerank/截断/推理层无关 |
| 用户实测: 思考中见 "1211 决策段" | 应来自 KG 实体描述 + 相邻 ßehemoth/gel 线索段, 非原文锚点 |
| mix/top_k=20/rerank 12 已显著优于 hybrid/top_k=12 基线 | Q3 冒烟曾直接命中 ßehemoth 语义, 检索层改善确认 |

### 新卡点 (待决策, 未实施)

| # | 卡点 | 机制 | 建议成本 |
|---|---|---|---|
| F1 | KG 实体描述不完整 | 1211 建图分类 artifact/network node, description 无 gel 角色 → LLM 被误导质疑用户 | 高 (重建索引) |
| F2 | System Prompt 缺"指称权威"约束 | 模板无"用户指称优先, 不得因类型/措辞差异质疑"指令 | 低 (改 build_rag_system_prompt) |
| F3 | 反事实多步推理无引导 | 模板无"若有因果/决策段则直接引用其逻辑"指令, LLM 不串联 1211 选择搬运→不打 Lenie | 低 (改 build_rag_system_prompt) |
| F4 | 反事实/矛盾式提问的召回覆盖不足 | 问句与答案段语义距离远, top_k=50 仍无锚点 → 系统性召回不到, 非漂移 | 中 (需改写检索串或查询改写) |

### 建议下一步 (未实施, 按第三轮证据更新优先级)
1. 中风险: **查询改写 (question rewrite)** — 服务端用 LLM 把反事实问句改写为陈述式检索串 ("Why didn't X do Y?" → "X decided to not do Y because..."), 或扩检索串加 "not/couldn't/refused" 反义表达. 这是直击 F4 的方向, 成本中.
2. 低风险: build_rag_system_prompt 模板增加 F2 (指称权威) + F3 (因果/决策段直接引用) 指令 — 治理"召回到但不会用", 仍值得做, 但非本轮根因.
3. 高成本暂缓: 重建索引补实体角色描述 (F1) 与实体归一化 (C) — 证据显示召回层瓶颈在查询语义而非索引质量, 重建的边际收益存疑.

---

## 5.2 检索扩展实验: 提案 1 证伪 (2026-08-11 第四轮)

### 实验设计 (scripts/probe_query_expansion.py, 经 IDE 评审修正)
- 疑点 A 修正: 静态扩展臂只允许问题中已出现的词元或直译 (凝胶→gel, 核弹→nuke), 严禁答案侧词汇 (copy/move/decide 等), 否则实验自答式泄露且不可泛化; "自答式检索"合法形态 = 运行时 LLM 生成 (改写/HyDE), LLM 只见问题不见语料, 不构成泄露.
- 疑点 B 修正: 基线 = 同脚本同进程同生产配置 (mix/top_k=20/chunk_top_k=12/rerank on) 的 Q_ORIG 臂; diagnose_q1_raw.py (hybrid/top_k=50) 不再作基线.

### 结果 (四臂同口径对照)
| 臂 | 检索串 | 锚点命中 |
|---|---|---|
| Q_ORIG 基线 | 用户原始反事实问题 | 0/3 |
| 静态问题派生 | 仅问题词元+直译 | 0/3 |
| LLM 改写 | 运行时陈述式改写 | 0/3 |
| HyDE 假想段 | LLM 现场生成的假想答案段 | 0/3 |

### 结论
连语义上几乎等同答案转述的 HyDE 段也未能经向量通道带回锚点 chunk → **失败不在检索串措辞层, 而在 chunk 向量召回/候选合并层**. 实体/关系层四臂均正常召回 (55-57 entities, ~1000 relations), 最终 chunks 全为无关段落 → §2.2 "实体召回与 chunk 召回脱节"进一步坐实. 检索扩展 (提案 1) 证伪, 不接入生产. 下一步: 直接量化嵌入通道死活 (scripts/probe_embedding_channel.py, 见 IMPROVEMENTS.md 提案 8).

---

## 5.3 嵌入通道直测: embedding 语义弱 (2026-08-11 第五轮)

### 方法与修正
IDE agent 源码级定位: `vdb_chunks.json` 的 `data[i]["vector"]` 是 zlib 压缩 base64 快照, 非明文向量; 真向量在 `matrix` (base64, `buffer_string_to_array + reshape(-1, dim)`), 与 nano-vectordb `_cosine_query` 的 L2 归一化语义一致. 据此官方语义复算.

### 结果 (锚点 chunk: 102=identity_17269, 104=choice_17337+answer_17345)
| 查询 | chunk-102 | chunk-104 | top-12 |
|---|---|---|---|
| Q_ORIG 反事实 | 356/433 (0.4253) | 335/433 (0.4398) | MISS |
| 静态问题派生 | 409/433 (0.2613) | 160/433 (0.4104) | MISS |
| HyDE 假想段 | 422/433 (0.2979) | **37/433** (0.4542) | MISS |
| Q2 对照 | 160/433 (0.3087) | 49/433 (0.3478) | MISS |

### 结论
连语义几乎等同答案的 HyDE 段, 锚点也进不了 naive top-12 → **合并/截断层没有机会挤出锚点, 排除"合并层淹没" (提案 9)**; 433 条 chunk 余弦 top 分值仅 0.26-0.45, 区分度整体弱 → **根因 = qwen3-embedding-0.6b 对长句科幻叙事+ß 变体表征不足 (embedding 语义弱)**, 指向换模型/混合召回 (提案 10).

**Q2 矛盾澄清**: §2.2 "Q2 naive 命中 17269" 实为命中 `node1211_darkness` 锚点 ("since the Darkness", chunk-103 区域), 非 chunk-102/104 指纹; 故本轮 Q2 对照对 102/104 MISS 不构成矛盾, 且 chunk-104 在 Q2 下仍靠后不推翻结论.

**下一步**: 重建前做尽职调查 (提案 12, scripts/probe_embed_broad.py) — 广度 (0.6b 是否全库普遍弱) + 维度 (4b/8b 是否 1024 免重建) + 对比试算 (4b/8b 锚点排名是否显著提升), 用数据拍板"换模型是否值得重建/换哪个/维度是否变".

---

## 5.4 尽职调查: 换 embedding 论据不足 (2026-08-11 第六轮)

用户确认: 4b/8b 维度显著更高 (必重建, 但 llm_response_cache 已备份, 重建仅耗 embedding token). 实验聚焦广度+对比 (scripts/probe_embed_broad.py, 只读, 4b/8b 现场重嵌 433 chunk).

### 结果
1. **0.6b 全库广度基线 11/12 = 92%** — 唯一 MISS 恰是"反事实-Q1"探针. **0.6b 并非普遍弱, 仅对反事实/决策语义错配类问题弱** (符合用户痛点但非全库性问题).
2. **Q1 锚点 top-12 对比** (chunk-102/104):

| 查询 | 0.6b 现库 | 4b 重嵌 | 8b 重嵌 |
|---|---|---|---|
| Q_ORIG | MISS (356/335) | MISS (223/270) | MISS (221/259) |
| 静态 | MISS | **HIT** (104=#9) | MISS (104=#42) |
| HyDE | **HIT** (104=#10) | **HIT** (104=#1) | **HIT** (104=#1) |
| Q2对照 | MISS (160/49) | **HIT** (102=#4, 104=#10) | **HIT** (102=#5, 104=#15) |

### IDE 交叉发现 (两处矛盾, 推翻第五轮初步判定)
- **① HyDE 0.6b 纯向量已 HIT (#10), 但第四轮 LightRAG mix 链路 HyDE 臂 0/3 MISS** → 真向量通道对陈述化 HyDE 串**能**召回 chunk-104, 疑似 **mix 链路在实体+关系合并/截断时丢掉了向量召回** (指向提案 9 合并层, 非 embedding 弱). 但两处 HyDE 串文本不同 (第四轮运行时完整段 / 第六轮拼接版), 口径未锁, 需同串对照复核.
- **② Q_ORIG 在任何模型下都 MISS** (102/104 均在 #220 后) → **换 embedding 解决不了用户原样反事实提问**; 4b/8b 增益顶点在 Q2对照/静态臂 (非真实痛点形态). 真正对症的是 **HyDE/陈述化改写** (0.6b 已够, 4b/8b 仅 #10→#1 边际提升).

### 结论
换 embedding 整体论据不足: 广度 92% 全库健康 + Q_ORIG 三模型全 MISS + HyDE 0.6b 已 HIT. **新关键问题**: 既然纯向量 0.6b+HyDE 能 HIT (#10), 为何 LightRAG 完整链路丢锚点? 下一步**裁决 mix 链路向量召回在合并/截断的去向** (提案 9 复核), 而非直接重建.

---

## 5.5 合并层 vs 语义裁决 (2026-08-11 第七轮, 进行中)

为锁定 §5.4 ①的口径矛盾 (三次 HyDE 串互不相同: 第四轮运行时完整段 / 第五轮硬编码截断版 / 第六轮拼接版), 用 scripts/probe_merge_vs_semantic.py **固定同一 HyDE 串**, 同进程三通道对照:
- A. 纯向量余弦 (绕过 LightRAG, 直接对 vdb 矩阵 top-12)
- B. aquery_data naive (LightRAG 纯向量检索, 无合并)
- C. aquery_data mix (生产链路, 实体+关系+向量合并)

判定: A HIT+B HIT+C MISS → 合并层淹没 (提案 9, 对症=查询改写+混合召回, 0 重建); A HIT+B MISS → naive 检索内部丢 (查索引/参数); A MISS → embedding 语义弱 (维持换模型).

### 结果 (固定同一拼接版 HyDE 串, 三通道同进程)
- A 纯向量余弦: **HIT** (chunk-104=#10)
- B naive 检索: **HIT** (chunk-104=#10)
- C mix 生产链路: **HIT** (chunk-104=**#3**) — 实体+关系合并未挤掉向量召回, 反而抬到第 3
→ **推翻提案 8 "锚点进不了 top-12" 与提案 9 "已排除" 在 HyDE 陈述形态下的前提; "embedding 语义弱"在此形态不成立; 换 4b/8b 仅 #10→#1 边际收益, 无需重建.**

### 复核 (用户指示, 排除检索随机性): 完整版 vs 拼接版同进程 mix 对照
- 第四轮运行时**完整版** HyDE 段 (含 bioluminescence/nuclear warhead/methane hydrate 等想象延伸词): 3 次重跑 **全 MISS**; query nodes 含大量延伸词, Local query 990 relations, 噪声实体/关系把检索引向 Behemoth 生态邻域, 双重淹没答案段.
- 第六轮**拼接版** (仅前两句, hyper-specific/calibrated/electromagnetic/metabolic/weapon/trigger): 同进程 **#1 HIT**; query nodes 精确聚焦, 481 relations.
- 结论: **文本差异归因成立, 检索随机性排除**. 第四轮 "HyDE 0/3 MISS" 非机制失效, 而是该轮 LLM 生成的完整版含过多想象延伸内容把检索引偏. **机制 (HyDE/陈述化改写) 有效, 关键在生成内容的精度/聚焦度.**

### 最终结论 (七轮诊断收口)
- 根因: 用户原样反事实提问 (Q_ORIG) 在任何 embedding 下都检索不到答案段 (语义错配); 但**陈述化/聚焦的检索串能稳定命中** (mix 链路 #3, 纯向量 #10).
- 对症方案 = **生产端查询改写**: 检索前把反事实问句改写为"仅基于问题实体与事实、禁止想象延伸"的聚焦陈述检索串, 配合现有 mix 链路即可命中. **0 重建, 每查询 +1 次短 LLM 改写调用.**
- 证伪/排除: 换 embedding (提案 10, 论据不足)、合并层调权 (提案 9, mix #3 显示无害) 均无必要.
- 生产改写指令关键约束: **禁止想象延伸/补充背景/推演**, 仅聚焦"实体名 + 具体属性 + 动作结果" (拼接版形态), 避免重建第四轮完整版的高噪声误例.

---

## 6. 诊断产物

| 文件 | 状态 | 说明 |
|---|---|---|
| scripts/diagnose_report.json | 保留 | 22 次检索原始数据 (971 行) |
| scripts/diagnose_retrieval.py | 保留 | 诊断矩阵脚本, 供改进前后复跑对比 |
| scripts/diagnose_q1_current.py | 保留 | §5.1 复测脚本: 当前生产配置复测 Q1 答案锚点 |
| scripts/diagnose_q1_rerank_ab.py | 保留 | §5.1 第三轮: rerank off/on ×3 对照, 排除 rerank 挤出锚点 |
| scripts/diagnose_q1_raw.py | 保留 | §5.1 第三轮: 原始候选池 (top_k=50) 锚点验证, 判定召回层失败 |
| scripts/probe_query_expansion.py | 保留 | §5.2 第四轮: 检索扩展四臂对照, 提案 1 证伪 |
| scripts/probe_embedding_channel.py | 保留 | §5.3 第五轮: 嵌入通道余弦排名直测, 判 embedding 语义弱 |
| scripts/probe_embed_broad.py | 保留 | §5.4 第六轮: 广度基线 11/12 + 4b/8b 重嵌对比, 换模型论据不足 |
| scripts/probe_merge_vs_semantic.py | 保留 | §5.5 第七轮: 固定 HyDE 串三通道对照, 裁决合并层 vs 语义 |
| scripts/check_filepath.py | 已删除 | 一次性 file_path 验证脚本 |
