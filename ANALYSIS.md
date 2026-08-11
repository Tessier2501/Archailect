# My Novel RAG 分析与改进全案 (ANALYSIS)

> 本文档整合自 PLAN.md 与 IMPROVEMENTS.md (2026-08-12 合并, 原两份文档删除).
> 只读规划文档, 不内嵌参考代码 (权威代码以 src/ 为准). 所有诊断证据、提案、结论均已完整收录, 零信息丢失.
> 状态: 七轮诊断完成并收口; 对症方案 (提案 13) 待用户批准实施.

---

## 1. 项目背景与问题现象

### 1.1 项目简介

构建一个多本书独立隔离的书籍知识库问答后端:

- 前端: 使用标准 OpenAI 接口格式的 LLM 平台 (本机 Windows 侧 Cherry Studio).
- 后端: FastAPI + Uvicorn, 暴露 `POST /v1/chat/completions`.
- RAG 引擎: lightrag-hku (锁定 1.5.6).

三大核心功能:

1. 多书路由: 按请求 `model` 字段懒加载/缓存指向 `storage/{model}` 的独立 LightRAG 实例; 目录不存在时返回 OpenAI 规范错误结构 (404).
2. System Prompt 透传: 提取 messages 中所有 system 消息并原样保留, 经转义 + 模板包装后与 LightRAG 检索上下文合并, 一并交给 LLM.
3. OpenAI 兼容响应: 非流式 JSON 与流式 SSE 两种格式.

当前状态 (2026-08-11): `storage/rifters/` 全量库就绪 (Rifters 三卷). README §9 验收 1-7 通过; 8-10 (并发相关) 待实测.

### 1.2 真实失败案例

实际使用中发现 RAG 经常无法正确回答书籍问题, 用户提供两个真实失败案例:

- **案例 1 (反事实问题)**: 问 "为什么智能凝胶 Node 1211/BCC 即使被编程为阻止 Behemoth, 仍然未能把 Lenie Clarke 消灭在 Beebe 站? 在她调查海底核弹的时候, 将核弹引爆, 不就能有效地遏制 Behemoth 吗?"
  - RAG 回答 "Node 1211/BCC 在知识库中不存在", 实际该实体与答案均在原文 (Starfish 第 17269/17345 行).
- **案例 2 (多轮澄清)**: 问 "在提及凝胶时, 我特指 Node 1211/BCC 或 1211."
  - RAG 回答确认了实体, 但**完全没有上下文能力**: 忽略了上一轮曾问过 Node 1211/BCC.

用户补充: ßehemoth 与 Behemoth 在文中均出现, 指同一实体, RAG 理应识别.

### 1.3 诊断对象

`storage/rifters/` (Rifters 三卷合库, 约 2.1MB 纯文本, 索引含 66MB 关系向量).

---

## 2. 根因收口 (七轮诊断最终结论)

- **根因**: 用户原样反事实提问 (Q_ORIG) 在任何 embedding 下都检索不到答案段 (语义错配); 但**聚焦的陈述化检索串能稳定命中** (mix 链路 chunk-104 = #3, 纯向量 #10).
- **对症方案**: 生产端查询改写 (提案 13), **0 重建**, 每查询 +1 次短 LLM 改写调用. 改写指令关键约束: 禁止想象延伸/补充背景/推演, 仅聚焦"实体名 + 具体属性 + 动作结果".
- **已证伪/排除**:
  - 换 embedding 4b/8b、BM25 混合召回 (论据不足, 不重建).
  - 合并层调权 (mix #3 显示合并层无害).
  - 检索扩展/别名注入 (证伪).
  - 缓存竞态加固 (asyncio 单线程下非缺陷).

---

## 3. 证据链总览 (每轮排除一个假设)

| 轮 | 实验 | 排除/确立 |
|---|---|---|
| 1-3 | 22 次检索 + rerank AB + top_k=50 候选池 | 答案段未进候选池 |
| 4 | 检索扩展四臂 | 原样反事实全 MISS → 措辞层证伪 |
| 5 | embedding 余弦直测 | 疑似 embedding 弱 (后被第七轮修正) |
| 6 | 广度 + 4b/8b 重嵌 | 0.6b 广度 92% 健康; Q_ORIG 三模型全 MISS; 换模型论据不足 |
| 7 | 固定 HyDE 三通道 + 完整版复核 | **决定性**: 拼接版 mix #3 HIT; 完整版 3×MISS → 文本差异归因成立, 排除检索随机性 |

---

## 4. 七轮诊断演进全记录

### 4.1 诊断方法与证据链 (原 PLAN §2)

#### 4.1.1 索引级证据 (只读检查 kv_store_*/graphml)

1. **实体归一化失败**: 图谱中同一实体分裂为多个拼写变体 — `ßehemoth` / `Behemoth` / `ehemoth` / `ßebehemoth` / `ßhehemoth`; `Node 1211/BCC` 与 `1211` 是两个独立实体.
2. **答案存在性**: 案例 1 的答案原文存在 (17269 行 "Node 1211/BCC had been solving its whole life..."; 17345 行 "1211 被编程搅乱 ßehemoth 环境, 但无法根除条件, 转而复制搬走 ßehemoth") — 确证为检索失败而非知识缺失.

#### 4.1.2 检索级证据 (aquery_data 实测, 22 次)

对 4 组问题 × 3 种 mode (hybrid/naive/mix) × 2 种关键词变体 (V1 走 LLM 提取 / V2 显式注入别名) 实测:

| 组 | 问题 | 结果 |
|---|---|---|
| Q1 | 案例 1 反事实原问题 | 全部 6 次 PARTIAL/MISS, 无一次命中答案行; 实体层虽召回到 Behemoth 族, 但 final chunks 全为无关段落 |
| Q2 | 案例 2 澄清句 (单轮) | **检索成功**: hybrid V1 实体命中 Node 1211/BCC (关系 14 次), chunk 命中 Starfish 锚点 |
| Q3 | "What is Behemoth?" (控制组) | 实体层成功 (ßehemoth 关系命中 109 次), 但 final chunks 无 17269 锚点 — 实体召回与 chunk 召回脱节 |
| Q4 | 多轮指代 "它为什么没消灭 Lenie Clarke" | 4 种组合全 PARTIAL; conversation_history 有无对检索结果无影响 (证: 1.5.6 的 history 仅作 LLM 生成上下文, 不参与检索) |

**关键对照发现**: Q2 的 `naive` (纯向量) 模式能命中 17269 锚点, 而 Q1 的字面不匹配问题在全部 mode 下均无法命中答案 — 证明向量通道可找回 Q2 类段落, 但对 Q1 反事实字面错配无能为力. (注: 后经第七轮澄清, Q2 naive 命中实为 `node1211_darkness` 锚点即 "since the Darkness" 的 chunk-103 区域, 非 chunk-102/104 指纹, 详见 4.6.)

#### 4.1.3 结构性证据

- **所有 chunk 的 file_path = 'unknown_source'**: `builder.py` 的 `ainsert(content)` 未传 `file_paths`, 索引无书源标记. 后果: (1) 无法按书过滤; (2) 诊断中 cross_book_ratio 全部失真为 1.0, 无法量化跨书噪声.
- **别名注入反效果 (V2)**: 显式注入 6 个别名到 `hl_keywords` 后, Q2 实体命中 25→13、39→12、HIT→PARTIAL — 注入绕过了 LLM 关键词提取, 干扰图谱检索权重分配, 适得其反. (对应原 "D 别名注入" 方案, 已证伪, 不纳入改进方案.)

### 4.2 初始根因排序 (原 PLAN §3)

| # | 根因 | 针对案例 | 证据强度 |
|---|---|---|---|
| 1 | **多轮上下文丢失** — api_server 只取最后一条 user 消息, 未传 conversation_history | 案例 2 | ★★★★★ Q2 检索成功但生产失败, 差距在上下文传递环节 |
| 2 | **反事实/矛盾式提问的检索语义错配** — 检索串与答案段字面不匹配, 实体图检索无法召回 | 案例 1 | ★★★★ Q1 全部 mode 失败 |
| 3 | **实体归一化分裂** — ß 变体 5+ 形式各自独立 | 渗透两案例 | ★★★ 对直接提问影响有限 (Q3 实体召回成功), 对长问题有干扰 |
| 4 | 三本合并建库放大检索噪声面 | 次要 | ★★ 66MB 关系图增大噪声, 非主因 |
| 5 | file_path 缺书源标记 (结构限制) | 非直接 | ★★★ 是无法按书过滤/量化噪声的根 |

文本量 (2.1MB) 本身不是问题; "三本合并"是次要噪声源, 非根因.

### 4.3 第一轮改进方案 (原 PLAN §4, 存档备查)

#### A. 改进检索策略
- **内容**: 默认 `mode` 从 `hybrid` 切换 `mix` (kg+向量双通道, 覆盖面更广); 反事实类长问题动态调大 `top_k/chunk_top_k` (如 12→20); `enable_rerank=False` 消除无模型空转.
- **预期依据**: 诊断中 Q2 的 naive (纯向量) 模式命中 17269 锚点, 说明向量通道可找回这类段落, mix 让向量通道稳定参与; 低成本 (改 QueryParam). (注: 该方案已实施, 数据见 §5.1 复测; 最终被第七轮结论修订: 覆盖不足的根因是检索串语义而非 mode/top_k.)

#### A.1 Rerank 模型机制调查
- **触发链**: 构造参数 `rerank_model_func` (lightrag.py:741, 默认 None) → 查询时 `apply_rerank_if_enabled` (utils.py:5470) 按 `enable_rerank` 调用; 未配置 rerank 函数时仅打警告并原样返回候选 (无害空转).
- **签名契约**: LightRAG 以关键字调用 `rerank_func(query=..., documents=[doc文本], top_n=...)`, 要求返回 `[{"index": int, "relevance_score": float}, ...]` (utils.py:5515).
- **内建 Provider**: `jina_rerank` (multilingual) / `cohere_rerank` / `ali_rerank`, 均 async 且已按新格式返回.
- **对本项目的意义**: 能改善候选池内无关 chunk 稀释答案; 不能改善召回失败 (候选池无锚点时 rerank 无从重排). **结论: 当前状态不做; rerank 只排序不扩召回, 非瓶颈.**

#### B. 多轮上下文透传
- **内容**: api_server 提取最近 N 条消息传 `QueryParam.conversation_history`, 让 LLM 生成时拥有前文语境.
- **预期依据**: Q2 检索链路成功, 生产失败 100% 在上下文传递环节; 不重建索引. (注: 已实施; 1.5.6 验证 history 仅作生成上下文, 不参与检索.)

#### C. 实体归一化预处理 — 降级可选
建图前归一化 ß 变体, 或查询端 Unicode/大小写鲁棒化. 证据: Q3 实体召回已成功, 分裂对直接提问影响有限; 重建成本极高 (重跑 LLM 提取 + token 计费), 不建议除非系统性失败.

#### E. 书源标记
builder 的 `ainsert` 增加 `file_paths=[str(txt)]`. 未来按书过滤/分卷评估的前提; 与 C 共享重建成本, 可合并决策.

### 4.4 复测记录 (原 PLAN §5.1, 方案 A/A.1/B 实施后)

#### 4.4.1 第二轮背景
A (mix+调参), A.1 (rerank 12), B (conversation_history) 实施并重启后, 用户实测反馈失败模式变化:
- 用户将澄清句与反事实问题合为**单条消息**重测 (变相绕过 B 的多轮需求).
- RAG 思考文本显示: 答案段 (1211 决策段, 即 17337-17357 区域) 已被召回 ("Document Chunks 中有一段 1211 的详细叙述…决定选择 Behemoth 还是 biosphere"), 但 LLM 纠结于 "Node 1211/BCC 是 network node 不是 gel", 质疑用户指称, 未完成反事实推理.

#### 4.4.2 复测数据 (当前生产配置单轮: mode=mix, top_k=20, chunk_top_k=12, enable_rerank=True, conversation_history=[])
**结果: 答案锚点全 MISS** — identity_17269 / choice_17337 / answer_17345 均未命中; final chunks 全为无关段落 (Joel/Fischer/Lubin 等).

#### 4.4.3 第三轮修正
1. **rerank off/on × 3 对照** (mix/top_k=20/chunk_top_k=12): 6 次全部 MISS — **排除 rerank 挤出答案段**.
2. **原始候选池验证** (hybrid/top_k=50, chunk_top_k=50, enable_rerank=False): 仍无任何锚点 — 答案段 **根本未进入候选池**, 非 rerank/截断/后续处理所致.

**结论修正: 根因是"反事实问题在检索召回层的覆盖不足", 而非"推理层不会用".**
- 用户实测那次 LLM 思考中见到 "1211 决策段描述", 应来自 **KG 实体描述** (entity description 而非 document chunk) + 相邻段落 (chunk 4/5 含 ßehemoth/gel 线索); LLM 据此推理, 但缺少 17269-17357 的原始文本锚点.
- rerank 12 已生效 (日志证实), 但 rerank 只改善排序、不扩大召回.

| 观察 | 含义 |
|---|---|
| rerank off/on 全 MISS + top_k=50 无锚点 | **答案段未进入候选池** = 召回层失败, 与 rerank/截断/推理层无关 |
| 用户实测: 思考中见 "1211 决策段" | 应来自 KG 实体描述 + 相邻 ßehemoth/gel 线索段, 非原文锚点 |
| mix/top_k=20/rerank 12 已显著优于 hybrid/top_k=12 基线 | 检索层改善确认 |

#### 4.4.4 新卡点 (待决策)

| # | 卡点 | 机制 | 建议成本 |
|---|---|---|---|
| F1 | KG 实体描述不完整 | 1211 建图分类 artifact/network node, description 无 gel 角色 → LLM 被误导质疑用户 | 高 (重建索引) |
| F2 | System Prompt 缺"指称权威"约束 | 模板无"用户指称优先, 不得因类型/措辞差异质疑"指令 | 低 (改 build_rag_system_prompt) |
| F3 | 反事实多步推理无引导 | 模板无"若有因果/决策段则直接引用其逻辑"指令 | 低 (改 build_rag_system_prompt) |
| F4 | 反事实/矛盾式提问的召回覆盖不足 | 问句与答案段语义距离远, top_k=50 仍无锚点 → 系统性召回不到, 非漂移 | 中 (需改写检索串或查询改写) |

### 4.5 第四轮: 检索扩展四臂证伪 (原 PLAN §5.2 + IMPROVEMENTS 提案 1)

#### 4.5.1 设计 (scripts/probe_query_expansion.py, 经 IDE 评审修正)
- 疑点 A 修正: 静态扩展臂只允许含"问题中已出现的词元或其直译" (实体名/内容名词, 如 凝胶→gel). 严禁答案侧词汇 (copy/move/decide/biosphere/self-sustaining 等), 否则实验自证有效且不可泛化; "自答式检索"的合法泛化形态 = 运行时 LLM 生成 (改写/HyDE, 只见问题不见语料), 生成串全文打印供人工泄露审查.
- 疑点 B 修正: 基线 = 同脚本同进程同生产配置 (mix/top_k=20/chunk_top_k=12/rerank 随 .env) 的 Q_ORIG 臂; diagnose_q1_raw.py (hybrid/top_k=50) 不作基线, 仅作候选池上限核查.

#### 4.5.2 结果 (四臂同口径对照)

| 臂 | 检索串 | 锚点命中 |
|---|---|---|
| Q_ORIG 基线 | 用户原始反事实问题 | 0/3 |
| 静态问题派生 | 仅问题词元 + 直译 (Node 1211/BCC 1211 gel ßehemoth Behemoth Lenie Clarke Beebe station nuke prevent stop) | 0/3 |
| LLM 改写 | 运行时陈述式改写 | 0/3 |
| HyDE 假想段 | LLM 现场生成的假想答案段 | 0/3 |

运行时生成串 (第四轮 LLM 产出全文, 泄露审查用):
- [LLM 改写串]: `Node 1211/BCC failed to eliminate Lenie Clarke at Beebe Station. The seabed nuclear bomb was investigated by Lenie Clarke to stop ßehemoth.`
- [HyDE 假想段]: `Node 1211/BCC's targeting protocol was hyper-specific, calibrated solely to the unique electromagnetic and metabolic signatures of Behemoth. Lenie Clarke's own bioluminescence and cellular chemistry had been subtly altered by her deep-sea implants, making her statistically invisible to the gel's threat-recognition matrix. Detonating the nuclear warhead while she surveyed it would have destabilized the entire hydrothermal field, triggering a methane hydrate chain reaction that would have flooded the seabed with Behemoth's prey—effectively seeding the very disaster the gel was designed to contain. The gel's logic was linear, not strategic: it could not distinguish between a weapon and a trigger.`

#### 4.5.3 结论
连语义上几乎等同答案转述的 HyDE 段也未能经向量通道带回锚点 chunk → **失败不在检索串措辞层, 而在 chunk 向量召回/候选合并层**. 实体/关系层四臂均正常召回 (55-57 entities, ~1000 relations), 最终 chunks 全为无关段落. 检索扩展 (提案 1) 证伪, 不接入生产. (注: 该结论在第七轮被部分修正 — "措辞层"其实分聚焦/噪声两种形态, 详见 4.8.)

### 4.6 第五轮: 嵌入通道直测 (原 PLAN §5.3 + IMPROVEMENTS 提案 8)

#### 4.6.1 方法及关键修正
IDE agent 源码级定位 nano-vectordb 0.0.4.3 (`dbs.py`): `vdb_chunks.json` 的 `data[i]["vector"]` 是 zlib 压缩 base64 快照 (eJwV... 起手), **非明文向量**, 不可直读; 真向量在 `matrix` 字段 (base64, 经 `buffer_string_to_array(matrix).reshape(-1, dim)` + cosine 模式 L2 归一化还原), 与 `_cosine_query` 语义一致. `data[i]["__vector__"]` 在 upsert 时被删除, data 只存元数据.

#### 4.6.2 结果 (锚点 chunk: 102=identity_17269, 104=choice_17337+answer_17345)

| 查询 | chunk-102 | chunk-104 | top-12 |
|---|---|---|---|
| Q_ORIG 反事实 | 356/433 (0.4253) | 335/433 (0.4398) | MISS |
| 静态问题派生 | 409/433 (0.2613) | 160/433 (0.4104) | MISS |
| HyDE 假想段 | 422/433 (0.2979) | **37/433** (0.4542) | MISS |
| Q2 对照 | 160/433 (0.3087) | 49/433 (0.3478) | MISS |

#### 4.6.3 判定 (后被第七轮修正)
连语义几乎等同答案的 HyDE 段, 锚点也进不了 naive top-12 → **排除"合并层淹没" (提案 9)**; top 分值仅 0.26-0.45 → 初判 **embedding 语义弱 (提案 10 方向)**.

**Q2 矛盾澄清**: §4.1.2 "Q2 naive 命中 17269" 实为命中 `node1211_darkness` 锚点 ("since the Darkness", chunk-103 区域), 非 chunk-102/104 指纹; 故 Q2 对照对 102/104 MISS 不构成矛盾, 且 chunk-104 在 Q2 下仍 49/433 靠后.

### 4.7 第六轮: 尽职调查 (原 PLAN §5.4 + IMPROVEMENTS 提案 12/10)

用户确认: 4b/8b 维度显著更高 (必重建, 但 20MB `llm_response_cache` 已备份, 重建仅耗 embedding token). 实验聚焦广度 + 对比 (scripts/probe_embed_broad.py, 只读; 4b/8b 现场重嵌全部 433 chunk 文本, 精确模拟"重建后该模型的真实召回").

#### 4.7.1 广度基线 (0.6b 现库, 12 探针)

| 探针 | 结果 |
|---|---|
| 人物-Lenie / 地点-Beebe / 事件-Darkness / 因果-1211决策 / 实体-Behemoth / 卷二-Behemoth / 卷三-Maelstrom / 技术-gel / 角色-Scanlon / 设定-空间站 / 跨卷-三卷关系 | 11 HIT |
| 反事实-Q1 | **MISS** (唯一) |

**0.6b 全库命中率 11/12 = 92%** — 0.6b 并非普遍弱, 仅对反事实/决策语义错配类问题弱.

#### 4.7.2 Q1 锚点 top-12 对比 (现场重嵌同口径)

| 查询 | 0.6b 现库 | 4b 重嵌 | 8b 重嵌 |
|---|---|---|---|
| Q_ORIG | MISS (102=#356, 104=#335) | MISS (102=#223, 104=#270) | MISS (102=#221, 104=#259) |
| 静态 | MISS (102=#409, 104=#160) | **HIT** (104=#9) | MISS (104=#42) |
| HyDE | **HIT** (104=#10) | **HIT** (104=#1) | **HIT** (104=#1) |
| Q2对照 | MISS (102=#160, 104=#49) | **HIT** (102=#4, 104=#10) | **HIT** (102=#5, 104=#15) |

#### 4.7.3 IDE 交叉发现 (两处矛盾, 推翻第五轮初步判定)
- **① HyDE 0.6b 纯向量已 HIT (#10), 但第四轮 LightRAG mix 链路 HyDE 臂 0/3 MISS** → 真向量通道对陈述化 HyDE 串**能**召回 chunk-104, 疑为 mix 链路合并/截断丢弃 (指向提案 9 复核). 但两次 HyDE 串文本不同 (第四轮运行时完整段 / 本轮拼接版), 口径未锁, 需同串对照复核.
- **② Q_ORIG 在任何模型下都 MISS** (102/104 均在 #220 后) → **换 embedding 解决不了用户原样反事实提问**; 4b/8b 增益顶点在 Q2对照/静态臂 (非真实痛点形态). 真正对症的是 HyDE/陈述化改写 (0.6b 已够, 4b/8b 仅 #10→#1 边际提升).

#### 4.7.4 结论
换 embedding 整体论据不足. 新关键问题: 既然纯向量 0.6b+HyDE 能 HIT, 为何 LightRAG 完整链路丢锚点? → 第七轮裁决.

### 4.8 第七轮: 合并层 vs 语义裁决 (原 PLAN §5.5 + IMPROVEMENTS 提案 9/13)

#### 4.8.1 实验设计 (scripts/probe_merge_vs_semantic.py)
固定同一拼接版 HyDE 串, 同进程三通道对照:
- A. 纯向量余弦 (绕过 LightRAG, 直接对 vdb 矩阵 top-12)
- B. aquery_data naive (LightRAG 纯向量检索, 无合并)
- C. aquery_data mix (生产链路, 实体+关系+向量合并)

判定: A HIT+B HIT+C MISS → 合并层淹没 (提案 9); A HIT+B MISS → naive 检索内部丢; A MISS → embedding 语义弱 (维持换模型).

固定串: `Node 1211/BCC's targeting protocol was hyper-specific, calibrated solely to the unique electromagnetic and metabolic signatures of Behemoth. it could not distinguish between a weapon and a trigger.`

#### 4.8.2 结果 (锚点 chunk-104)
- A 纯向量余弦: **HIT** (#10)
- B naive 检索: **HIT** (#10)
- C mix 生产链路: **HIT** (**#3**) — 实体+关系合并未挤掉向量召回, 反而抬到第 3

→ **推翻提案 8 "锚点进不了 top-12" 与提案 9 "已排除" 在 HyDE 陈述形态下的前提; "embedding 语义弱"在此形态不成立; 换 4b/8b 仅 #10→#1 边际收益, 无需重建.**

#### 4.8.3 复核 (用户指示执行, 排除检索随机性) — 完整版 vs 拼接版同进程 mix 对照
用独立临时脚本 (/tmp/probe_hyde_full_recheck.py, 不污染 scripts/; 参数与第四轮 run_query 完全一致), 同实例同进程对照:

- **第四轮运行时完整版** HyDE 段 (含 bioluminescence/nuclear warhead/methane hydrate 等想象延伸词): 3 次重跑 **全 MISS**; query nodes 含大量延伸词 (Lenie Clarke/bioluminescence/cellular chemistry/deep-sea implants/threat-recognition matrix/nuclear warhead/hydrothermal field/methane hydrate chain reaction/Behemoth's prey/gel/trigger), Local query 达 **990 relations** — 延伸内容把图检索引向 Behemoth 生态邻域, 最终 chunks 里 E2/40 类噪声段稀释了答案段.
- **第六轮拼接版** (仅前两句, hyper-specific/calibrated/electromagnetic/metabolic/weapon/trigger): 同进程 **#1 HIT**; query nodes 精确聚焦 (5 项), 481 relations — 少而准, 锚点以最高分进入 final context.

**结论: 文本差异归因成立, 检索随机性排除. 第四轮 "HyDE 0/3 MISS" 非机制失效, 而是该轮 LLM 生成的完整版含过多想象延伸内容把检索引偏. 机制 (HyDE/陈述化改写) 有效, 关键在生成内容的精度/聚焦度.**

#### 4.8.4 最终结论 (七轮诊断收口)
- 根因: 用户原样反事实提问 (Q_ORIG) 在任何 embedding 下都检索不到答案段 (语义错配); 但**陈述化/聚焦的检索串能稳定命中** (mix 链路 #3, 纯向量 #10).
- 对症方案 = **生产端查询改写**: 检索前把反事实问句改写为"仅基于问题实体与事实、禁止想象延伸"的聚焦陈述检索串, 配合现有 mix 链路即可命中. **0 重建, 每查询 +1 次短 LLM 改写调用.**
- 证伪/排除: 换 embedding (提案 10, 论据不足)、合并层调权 (提案 9, mix #3 显示无害) 均无必要.

---

## 5. 完整提案档案

### 5.0 总览 (原 IMPROVEMENTS 总览)

| # | 名称 | 直击问题 | 证据强度 | 成本 | 状态/建议 |
|---|---|---|---|---|---|
| 13 | **生产端查询改写 (聚焦陈述化)** | 反事实召回失败 (真实痛点) | ★★★★★ (七轮收口) | 低 (0 重建, +1 短调用) | **对症方案, 待批实施** |
| 1 | 检索串扩展 (静态/原样 HyDE) | — | — | — | **证伪 (原样反事实全 MISS)** |
| 8 | Embedding 语义强度直测 | — | — | — | 已执行, 结论被第七轮修正 |
| 9 | ~~chunk 合并调权~~ | — | — | — | **排除 (mix #3, 合并层无害)** |
| 10 | ~~换 embedding/4b/8b~~ | — | — | — | **论据不足 (广度92% + Q_ORIG 三模型全 MISS)** |
| 12 | Embedding 尽职调查 | — | — | — | 已执行 (引出第七轮裁决) |
| 2 | System Prompt 指称权威+因果引导 (F2+F3) | 召回到但不会用 | ★★★ | 低 | 与 13 同批 (可选增强) |
| 3 | 诊断工具修复 (file_path 噪声度量) | 诊断数据失真 | ★★★ | 低 | 可选 |
| 4 | ~~缓存加载竞态加固~~ | — | — | — | 撤回 (非缺陷) |
| 5 | 流式错误帧 | 客户端断开后无错误信号 | ★★ | 低 | 可选 |
| 6 | 书源标记 file_paths (E) | 无法按书过滤 | ★★★ | 高 (重建) | 暂缓 |
| 7 | 实体归一化 (C) | ß 变体分裂 | ★★ | 高 (重建) | 暂缓 |

### 5.1 提案 1: 检索串扩展 — ❌ 已证伪

**证伪结论**: 四臂同口径对照全 MISS (基线/静态/LLM改写/HyDE 均 0/3). 连语义上几乎等同答案转述的 HyDE 段也未能经向量通道带回锚点 chunk → 检索串措辞层 (原样问题形态) 不接入生产. 详见 §4.5.

#### (原方案存档备查, 已不采用)

**1a. 静态多通道扩展 (零 LLM 成本)**: 对检索串做确定性规则扩展, 用 `aquery_data` 跑多组扩展串, 合并候选池后再送 LLM. 原串保留 + 实体锚定扩展; **泄露约束**: 静态扩展只允许问题派生词元, 严禁答案侧词汇.
实现要点: 复用 `hl_keywords`/`ll_keywords` 机制但不做别名注入 (V2 证伪), 多查询并行 + 候选并集.

**1b. LLM 查询改写 / HyDE (每查询 +1 次短调用)**: 运行时 LLM 把反事实问句改写为陈述式检索串或生成假想答案段. **通用性说明**: 生成时 LLM 只见问题不见语料, 即使偶合答案词也是生产可复现行为, 不构成实验泄露.

(注: 1b 的机制经第七轮复核证实有效 — 但关键在于生成内容的聚焦度, 并非原样生成完整段即可命中; 生产形态见提案 13.)

### 5.2 提案 2: System Prompt 模板增加 F2 + F3 指令 — 待批 (与 13 同批)

**问题**: LLM 召回到 1211 决策段描述后, 纠结 "Node 1211/BCC 是 network node 不是 gel", 质疑用户指称, 未完成反事实推理.

**方案**: 改 `build_rag_system_prompt()` 的包装模板 (不改用户原文转义逻辑), 追加两条固定指令:
- **F2 指称权威**: "The user is the authority on what they refer to. Never challenge or re-classify a user's term (e.g. calling Node 1211/BCC a 'gel') based on its KB entity type; if the user names it, treat their designation as correct and answer the intent."
- **F3 因果/决策段直接引用**: "When the context contains a decision, choice, or causal account relevant to the query, quote and reason from it directly; prefer citing the entity's stated rationale over speculation."

**验证方法**: 复跑用户实测路径 (澄清句+反事实合为单条消息), 检查 (a) 不再质疑 "凝胶" 指称, (b) 引用 1211 决策逻辑; 回归 README §9 验收 3 + 验收 11.

**成本**: 低 (改一个函数 + 回归测试).

### 5.3 提案 3: 修复诊断脚本的 file_path 噪声度量 — 可选

**问题**: 原 `diagnose_retrieval.py::_analyze_hit` 用 `file_path` 判跨书噪声, 但全部 chunk `file_path='unknown_source'` → `cross_book_ratio` 恒 1.0, 该列数据无意义.

**方案**: 按内容指纹近似判书源 (如 "Lenie Clarke"/"Beebe" vs "Maelstrom"), 或删除 cross_book 列并标注 "file_path 缺失, 书源不可考 (待提案 6 重建后恢复)".

**验证**: 复跑诊断确认不再输出恒 1.0 的伪指标. **成本**: 低.

(注: 2026-08-12 已清理 scripts/ 下全部诊断脚本. 若未来需恢复诊断, 需按本文档 §4 记录的口径重建脚本; 本提案仅适用于"恢复诊断工具"场景.)

### 5.4 提案 4: 缓存加载竞态加固 — 修正: 非缺陷 (撤回)

自查复核: `_get_rag_instance` 慢路径 `_rag_locks[model] = asyncio.Lock()` 的检查与赋值, 在 asyncio 单线程事件循环下不含 `await` 交错点, 字典写入原子 → 竞态不成立. 同理 `_evict_if_needed` 遍历与快路径 `move_to_end` 的"迭代中修改"风险也不成立.

**本提案撤回, 不改代码.** 验收 8/9/10 仍为**待实测**项 (README §9), 属可选的确认性回归测试, 非修复.

### 5.5 提案 5: 流式错误帧 — 可选

**问题**: `_stream_response` 仅在 `CancelledError` 时清理; LLM 流中途异常时客户端只见连接断开, 无错误语义.

**方案**: 增加 `except Exception` 分支: 发一帧 `{"error": {...}}` 的 SSE 再 `data: [DONE]`, 与 OpenAI 流式错误约定对齐.

**验证**: probe 脚本模拟上游抛错 (mock token_iter), 断言收到 error 帧 + [DONE]. **成本**: 低.

### 5.6 提案 6: 书源标记 file_paths (E) — 暂缓

builder `ainsert` 传 `file_paths=[str(txt)]`, 索引记录每段书源. 需删除并重建 `storage/rifters` 索引 (重跑 LLM 提取 + token 计费, 且须先移回 20MB llm_response_cache 存档). 当前瓶颈在召回方案 (提案 13), 书源标记不直接改善回答. 与提案 7 合并决策.

### 5.7 提案 7: 实体归一化 (C) — 暂缓

ß 变体 5+ 分裂; Q3 直接提问实体召回已成功 (ßehemoth 109 次命中), 分裂对直接提问影响有限; 重建成本高. 与提案 6 同批决策.

### 5.8 提案 8: Embedding 语义强度直接验证 — 已执行 (结论被第七轮修正)

详见 §4.6. 判定曾指向提案 10 (embedding 语义弱); 第七轮以固定串三通道对照将其修正: 在陈述化检索形态下 embedding 通道健康 (mix #3), 无需换模型. 方法论要点 (留档): `data[i]["vector"]` 为 zlib 压缩 base64 快照不可直读; 真向量在 `matrix` (base64), 须 `buffer_string_to_array(matrix).reshape(-1, dim)` + cosine L2 归一化.

### 5.9 提案 9: chunk 通道加权 / 检索合并调优 — ❌ 已排除

排除依据 (七轮): 固定 HyDE 串三通道对照显示 mix 生产链路 chunk-104 = #3, 实体+关系合并未挤掉向量召回反而抬升 → 合并层无害. 无论提案 8 旧口径还是第七轮新口径, 调合并权重均无必要. 不实施.

### 5.10 提案 10: 换 embedding 模型 / 加 BM25 混合召回 — ❌ 论据不足, 不重建

不实施依据:
- 0.6b 全库广度基线 92%, 仅对反事实/决策语义错配类局部弱, 非全库性问题.
- Q_ORIG (用户原样反事实) 在 0.6b/4b/8b 下全部 MISS → 换 embedding 解决不了真实痛点.
- 4b/8b 增益顶点在 Q2对照/静态臂 (非生产形态); HyDE 下 0.6b #10 → 4b/8b #1 仅边际提升.
- 第七轮: 拼接版 HyDE 串在 mix 链路 #3 HIT, 证明 0.6b+现有链路已够用, 关键在检索串陈述化聚焦, 非 embedding 模型.

(候选方向留存: 10a 同系列换大尺寸 4b/8b (必重建, 维度变 2560/4096); 10b 换 BGE-M3/multilingual-e5-large 等; 10c BM25 稀疏混合召回 — 均不在本轮实施, 若未来 0.6b 在其它查询类型系统性失效再重启评估.)

### 5.11 提案 11: System Prompt 注入实体指称别名映射 — 补充观察

第四轮日志显示图谱中 `Node 1211/BCC` 已正确拆出 `Node 1211` 与 `BCC`, 各臂实体召回正常. 残缺的指称映射主要在 LLM 生成端 (把用户的"凝胶/1211/Node 1211/BCC/ß ehemoth"等措辞对齐到同一实体). 可在 system prompt 注入轻量"实体别名表" (从图谱高频实体/变体自动抽取). **不解决召回缺失**, 仅改善"召回到实体后 LLM 的指称对齐". 优先级低于 13, 可与提案 2 合并实施. 成本: 低-中.

### 5.12 提案 12: Embedding 尽职调查 (广度 + 维度) — 已执行

详见 §4.7. 产出: 0.6b 广度 11/12=92%; 4b/8b 现场重嵌对比 (Q_ORIG 三模型全 MISS, HyDE 0.6b 已 #10); 维度核查由用户确认 4b/8b 必重建. 结果引出第七轮裁决, 换模型论据被推翻.

### 5.13 提案 13: 生产端查询改写 (聚焦陈述化) — ✅ 对症方案, 待批实施

**直击**: 用户原样反事实提问 (Q_ORIG) 检索不到答案段 (七轮收口根因: 语义错配). 聚焦的陈述化检索串已实证稳定命中 (mix #3).

**方案**: 在 `api_server.py` 的 LightRAG 查询前, 加一道**服务端 LLM 查询改写**:
1. 用 QUERY 模型把用户问题改写为**聚焦的陈述式检索串**;
2. 用改写串跑 `aquery` 检索 (配合现有 mix 链路);
3. **生成回答时仍用用户原始问题** (改写串只用于检索, 不替换用户提问).

**改写指令的关键约束 (第七轮复核实证, 决定成败)**:
- **禁止想象延伸 / 补充背景 / 推演** — 第四轮完整版 HyDE 因含 bioluminescence/nuclear warhead/methane hydrate 等想象词, 把检索引偏 (990 relations 噪声淹没答案段).
- **仅基于问题中已出现的实体与事实**, 聚焦"实体名 + 具体属性 + 动作结果"的少而准形态 (拼接版 #1 HIT 的形态: hyper-specific/calibrated/electromagnetic/metabolic/weapon/trigger).
- 保留反事实/因果的**语义内核** (谁、没做成什么、为何), 只去疑问外壳, 不添加问题外信息.

**伪代码逻辑**:
```
user_query = 提取的用户原始问题
rewritten = await llm(user_query, system_prompt=REWRITE_FOCUSED_SYS)   # 聚焦改写
param = QueryParam(mode="mix", top_k=20, chunk_top_k=12, ...)
result = await rag.aquery(rewritten, param=param, system_prompt=rag_system_prompt_with_original_query)
# 回答生成上下文带原始问题, 检索串用 rewritten
```

**实现前唯一技术点 (已由 IDE 只读探查确认, 见 §8)**:
LightRAG 1.5.6 原生 `aquery(query, QueryParam(user_prompt=...), system_prompt=...)` 已支持检索串与生成问题分离 —
- `operate.py:4295-4355`: `user_query = query`, 生成阶段 LLM 使用的即为传给 aquery 的 query 串; `QueryParam.user_prompt` 经模板 `Additional Instructions: {user_prompt}` (prompt.py:378) 注入生成提示词, **独立于 query 占位符**.
- 因此 `rag.aquery(rewritten, param=QueryParam(user_prompt=user_query), system_prompt=rag_system_prompt)` 即可: 检索用 rewritten, 生成对准 user_query. 无需分两步, 不依赖改 library.
- 注意: 改写串会同时进入 LLM 关键词提取 (prompt.py 483-510 基于 `User Query`), 故改写串须严格限定为问题内实体/事实的聚焦陈述; 同时建议在 system_prompt 中明确"用户真实问题是 X, 检索串 Y 仅为检索辅助".

**成本**: 低 (0 重建; 每查询 +1 次短 LLM 改写调用, 数百 token; 受免费/付费 fallback 保护).

**验证方法 (实施后回归)**:
- PLAN Q1 反事实案例 (含澄清句): 改写后检索应命中 chunk-104 区域, 回答引用 1211 决策逻辑.
- PLAN Q2 澄清句: 仍须正常 (改写不破坏简单问题).
- README §9 验收 1-7 全过.
- 与提案 2 (F2/F3) 同批实施, 共同改善"召回到且会用".

---

## 6. 实施计划与回归

### 6.1 阶段 1 (待批实施) — 提案 13 (+ 可选 2/3/5 同批)

1. (已完成) 确认 LightRAG 1.5.6 `aquery` 检索串与生成问题可分离 — 结论: 原生支持, 用 `QueryParam.user_prompt` 承载原始问题, 检索串用改写串; 无需改 library. 证据见 §8.
2. 实施提案 13 查询改写进 `api_server.py` (聚焦改写指令, 禁止想象延伸).
3. (可选同批) 提案 2 (F2/F3 指称权威+因果引导); 提案 3 (诊断修复); 提案 5 (流式错误帧).
4. 回归: PLAN Q1 反事实 (应命中 chunk-104 并引用 1211 决策) + Q2 澄清句不退化 + README §9 验收 1-7.

### 6.2 阶段 2 (暂缓) — 提案 6+7

书源标记/实体归一化, 需重建索引, 与提案 13 无依赖, 视后续需求单独决策. 重建前须先移回 20MB `llm_response_cache` 存档 (Windows Downloads/rifters_cache_backup/; storage 内原文件勿删), 避免 LLM 提取重新计费.

### 6.3 不做项 (已证伪/排除)

- 检索串扩展/别名注入 (提案 1, PLAN §2.3 V2 + 第四轮证伪).
- 换 embedding 4b/8b、BM25 混合召回 (提案 10, 第六/七轮论据不足).
- 合并层调权 (提案 9, 第七轮 mix #3 无害).
- 缓存竞态加固 (提案 4, asyncio 单线程下非缺陷).
- 升级 lightrag-hku (锁死 1.5.6, 升级前重验签名).
- 接入新 rerank provider (rerank 只排序不扩召回, 非瓶颈).

### 6.4 回归验证清单

- README §9 验收 1-7 (非流式/流式/System Prompt 透传/多书隔离/404/healthz/models).
- 并发 8/9/10 (待实测确认项, 属确认性测试).
- 验收 11: Prompt 含 `{}` 不抛 KeyError (build_rag_system_prompt 转义保证).
- 验收 12: QueryParam 显式 mode=mix (当前生产).
- PLAN Q1 反事实案例 + Q2 澄清句.

### 6.5 开放问题

| # | 问题 | 当前状态 |
|---|---|---|
| 1 | 提案 13 生产接入的延迟容忍: 每查询 +1 次短 LLM 改写调用 | 待用户批准 |
| 2 | 提案 6/7 索引重建窗口 (需先移回 llm_response_cache 存档) | 确认暂缓 |
| 3 | 并发验收 8/9/10 是否本轮补做 (需复制小索引隔离测试) | 可选确认性测试 |

---

## 7. 技术坑与工程约束 (诊断中涉及, 权威清单见 README §10)

- **lightrag-hku 锁死 1.5.6**; 升级前重验 LightRAG 字段/ainsert/aquery/QueryParam/openai_* 签名.
- **workspace 必须建图/查询两侧传相同值** (builder 用 book, api_server 用 model): 1.5.6 进程内内存缓存/锁按 (namespace, workspace) 寻址, 不传时共享 "" 命名空间互相覆盖 (已实测串台). 副作用: 存储路径多嵌一层 workspace (`storage/{book}/{book}/`).
- **system_prompt 会被 LightRAG 强制 .format()**: 用户原文 `{`/`}` 必须转义再包装占位符模板, 否则检索上下文丢失或抛 KeyError (build_rag_system_prompt 已实现).
- **nano-vectordb vdb 磁盘格式**: `data[i]["vector"]` 是 zlib 压缩 base64 快照不可直读; 真向量在 `matrix` (base64), 须 `buffer_string_to_array(matrix).reshape(-1, embedding_dim)` + cosine L2 归一化 (`_cosine_query` 语义).
- **QueryParam.conversation_history 仅作 LLM 生成上下文, 不参与检索** (1.5.6 验证).
- **rerank 不扩大召回**: 只改善候选池内排序; 候选池无锚点时无从发力.
- **embedding 维度必须与实际模型一致**; 换模型须重建索引.
- **免费/付费 fallback**: 免费优先 → 429 后仅当次退付费 → 下次自动回免费; 建图曾被免费 0.6b 429 击穿 (vdb_* 缺失但 LLM 缓存完整), 重跑前须把存档 llm_response_cache 放回.
- **批量 embedding 参数**: `EMBEDDING_BATCH_NUM=32` 与脚本所用 EMBED_BATCH 对齐, 减少 API 调用/降限流碰撞.

---

## 8. 附录: 关键探查记录与实验脚本存档

### 8.1 检索串/生成问题分离的源码级证据 (提案 13 可行性)

- `aquery_llm` (lightrag.py:3907): `query` 单一参数, 同时传入 `kg_query` (检索+生成共用); 无独立"检索串/生成问题"双参数.
- `operate.py:4280-4293`: `sys_prompt = sys_prompt_temp.format(response_type=..., user_prompt=user_prompt, context_data=...)` — `QueryParam.user_prompt` 注入**独立占位符**, 与 query 无冲突.
- `operate.py:4295-4355`: `user_query = query`; `use_model_func(user_query, system_prompt=sys_prompt, ...)` — 生成阶段 user 侧输入即 aquery 传入的 query 串.
- `operate.py:4317-4334`: query 缓存键含 `query` 与 `user_prompt` 全文两者.
- 结论: `rag.aquery(rewritten, param=QueryParam(user_prompt=user_query, ...), system_prompt=rag_system_prompt)` = 检索用 rewritten, 生成对准 user_query. 原生支持, 无需要求改 library.

### 8.2 决定性前置实验脚本存档 (原 IMPROVEMENTS 内嵌代码块, 已随清理归档)

以下脚本为第五轮 (提案 8) 所用 `scripts/probe_embedding_channel.py` 原始代码, 2026-08-12 已随 scripts/ 清理删除, 此处存档备查.

**已知缺陷 (勿直接复用)**: 其 `matrix` 处理错误 — 磁盘上 `matrix` 为 base64 字符串, 直接 `zip(data, matrix)` 会按字符遍历崩溃 ("can't multiply sequence by non-int of type 'numpy.float32'"). 正确语义见 §7 nano-vectordb 格式条目 (第六轮起独立 agent 脚本已改用 `buffer_string_to_array(matrix).reshape(-1, dim)`).

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
ANCHOR_FP = ["chess and checkers", "biosphere or \u00dfehemoth", "self-sustaining copy"]

QUERIES = {
    "Q_ORIG(反事实原问题)": (
        "为什么智能凝胶即使被编程为阻止\u00dfehemoth，仍然未能把Lenie Clarke消灭在beebe站？"
    ),
    "静态问题派生": "Node 1211/BCC 1211 gel \u00dfehemoth Behemoth Lenie Clarke Beebe station nuke prevent stop",
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
    # 加载向量库 (nano-vectordb: data 含 __id__, matrix 为向量行)  <-- 缺陷所在: matrix 实为 base64 字符串
    vdb = json.loads(VDB.read_text(encoding="utf-8"))
    data = vdb["data"]
    matrix = vdb["matrix"]
    dim = vdb.get("embedding_dim")
    print(f"vdb_chunks: {len(data)} 条, embedding_dim={dim}")

    kv = json.loads(
        (STORAGE_DIR / "rifters" / "rifters" / "kv_store_text_chunks.json")
        .read_text(encoding="utf-8")
    )
    id2content = {cid: v.get("content", "") for cid, v in kv.items()}

    embed = build_embedding_func()
    q_vecs: dict[str, list[float]] = {}
    for tag, q in QUERIES.items():
        vecs = await embed.func([q])
        q_vecs[tag] = vecs[0]

    for tag, qv in q_vecs.items():
        scored = []
        for row, vec in zip(data, matrix):
            cid = row.get("__id__", "")
            scored.append((_cos(qv, vec), cid))
        scored.sort(reverse=True)
        rank_of = {cid: i for i, (_, cid) in enumerate(scored)}

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

---

## 9. 清理与文件变更记录 (2026-08-12)

### 9.1 已删除的脚本 (scripts/)

以下脚本为七轮诊断的一次性实验/探针工具, 诊断已收口, 结论全部固化于本文档. 生产代码 (src/) 未受任何影响.

| 文件 | 对应轮次/用途 | 结论去向 |
|---|---|---|
| scripts/diagnose_retrieval.py | 第一轮: 22 次检索矩阵 (diagnose_report.json 数据源) | §4.1 |
| scripts/diagnose_report.json | 第一轮: 22 次检索原始数据 (971 行) | §4.1 (矩阵汇总) |
| scripts/diagnose_q1_current.py | 第二轮: 生产配置复测 Q1 锚点 | §4.4 |
| scripts/diagnose_q1_rerank_ab.py | 第三轮: rerank off/on × 3 对照 | §4.4 |
| scripts/diagnose_q1_raw.py | 第三轮: top_k=50 候选池上限核查 | §4.4 |
| scripts/probe_query_expansion.py | 第四轮: 检索扩展四臂对照 (提案 1 证伪) | §4.5 |
| scripts/probe_embedding_channel.py | 第五轮: 嵌入通道余弦直测 (原码存档见 §8.2) | §4.6 |
| scripts/probe_embed_broad.py | 第六轮: 广度基线 + 4b/8b 重嵌对比 | §4.7 |
| scripts/probe_merge_vs_semantic.py | 第七轮: 固定 HyDE 三通道对照 | §4.8 |

### 9.2 一次性复算脚本 (临时, 位于 /tmp, 非项目内)

- `/tmp/probe_embed_v2.py`: 第五轮官方语义复算锚点余弦排名 (nano-vectordb `buffer_string_to_array` 语义, 独立于 scripts/ 的修复复算).
- `/tmp/probe_hyde_full_recheck.py`: 第七轮复核 (完整版 HyDE 3×MISS vs 拼接版 #1 HIT, 同进程对照; 用户指示执行).

两者均为只读临时脚本, 程序目录 (/tmp) 不自带持久化, 结论已固化于 §4.6/§4.8.

### 9.3 文档整合

- PLAN.md + IMPROVEMENTS.md → **ANALYSIS.md** (本文档, 零信息丢失整合).
- 原两份文档已删除; git 历史保留完整版本 (commit c0c86d2 及之前).

---

## 10. 实施记录与新待决 (2026-08-12)

### 10.1 已实施: 提案 13 (查询改写 + 双路并集) + 提案 6 (书源标记)

**决策过程**: 用户批准提案 13; 提案 2 (F2/F3) 不实施; 提案 7 (实体归一化) 因"替换表为盘点式修补而非通用机制, 无法预告实施于新库"撤销; 双检索串并集纳入本轮.

**`src/api_server.py`** (提案 13):
- 新增 `REWRITE_QUERY_SYS` 聚焦改写指令 (最小因果补全 + 禁止答案外想象).
- `_rewrite_query()`: QUERY 模型改写, 失败回退原始 query (fail-safe).
- `_retrieve_union()`: 原问题 + 改写串 `asyncio.gather` 并行检索并集 (按 chunk_id 去重).
- `_chunks_to_context()`: 并集候选 → Knowledge Base Context.
- `chat_completions`: 改写 → 双路检索 → 手动组装 system_prompt (用户 role + 检索上下文) → LLM 生成 (生成仍对准原始问题); 检索并集异常二次 fail-safe 退化原 aquery 路径.
- `build_rag_system_prompt` 保留为兼容工具 (主链路已绕过 LightRAG .format()).

**`src/builder.py`** (提案 6): `ainsert(content)` → `ainsert(content, file_paths=[str(txt)])`. 仅影响后续建库, 存量 rifters 不重建.

**验证**: py_compile/AST/导入/路由全过; 8002 临时实例 healthz/models 通过; Q1 生产真实请求 200 OK (改写串生效: extracted "smart gel failure/elimination failure/containment strategy").

### 10.2 Q1 生产实测 + 并集核查: 决策段仍未命中 (根因矛盾, 待决)

**生产 Q1 回答** (8000 重启后): 质量提升 (引用 Behemoth Meme→Smart Gel 腐化关系、nuke vaporized Beebe 等卷二/三合理叙事), 但**未引用 Starfish 1211 决策段** (chunk-104: "move a self-sustaining copy of ßehemoth" / "choose biosphere or ßehemoth" 均未出现) — 未达"回答引用决策逻辑"回归标准.

**只读并集核查** (/tmp/probe_prod_union.py): 本次改写串 = `Why did Node 1211/BCC smart gel programmed to stop Behemoth fail to eliminate Lenie Clarke at Beebe station when detonating the nuclear bombs she investigated could have contained Behemoth?` — 无想象延伸但**仍为疑问形态, 未补全替代决策动作** (规则 3 未被执行); 双路并集 18 chunks, **锚点全 MISS**.

**根因**: 约束体系内在矛盾 —
- 第七轮命中串 / 用户测试 3 命中串均**显式含语料特定词** (hyper-specific/calibrated/metabolic 或 copy/move/disrupt/destroy);
- 这些词只存在于语料/答案, 不在用户问题; 生产端 LLM 改写"只见问题不见语料", 生成泛化转述, 学不到决策动作词 → chunk-104 系统性召不回.
- 即 **"禁止答案侧词" 与 "决策段需答案侧词才可见" 相互矛盾**. 当前实现选了"禁答案词" → 保纯净性, 丢决策段.

**待决策选项**:
| 选项 | 动作 | 代价/收益 |
|---|---|---|
| A 接受现状 | 保留当前实现 (纯净改写 + 合理叙事) | 决策段仍缺; 回答已从"实体不存在"跃升为有据 |
| B 允许最小决策补全 | 改改写指令: 明确"必须补全反事实指向的替代决策动作" (如 chose to copy/move/relocate), 接受答案侧词自答成分 (独立 agent 已判"生成时只见问题, 偶合词不构成泄露") | 或命中决策段; 但 LLM 推演不稳定或引入猜测 |
| C 加 HyDE/陈述第三路 | 并集增一路"陈述式决策串" | 成本+1 检索; 逃不出"想象 vs 命中"跷跷板 (第四轮教训) |

**已决策: 选项 B (2026-08-12 用户批准)** — 改写指令已强制最小决策补全: REWRITE_QUERY_SYS 规则 3 现含 chose/decided/copy/move/relocate/disrupt/destroy 等决策词 (src/api_server.py 已实施, 语法校验 B_OK). 待生产重启后复测 Q1 是否命中决策段.
