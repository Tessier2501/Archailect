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

## 6. 诊断产物

| 文件 | 状态 | 说明 |
|---|---|---|
| scripts/diagnose_report.json | 保留 | 22 次检索原始数据 (971 行) |
| scripts/diagnose_retrieval.py | 保留 | 诊断矩阵脚本, 供改进前后复跑对比 |
| scripts/diagnose_q1_current.py | 保留 | §5.1 复测脚本: 当前生产配置复测 Q1 答案锚点 |
| scripts/diagnose_q1_rerank_ab.py | 保留 | §5.1 第三轮: rerank off/on ×3 对照, 排除 rerank 挤出锚点 |
| scripts/diagnose_q1_raw.py | 保留 | §5.1 第三轮: 原始候选池 (top_k=50) 锚点验证, 判定召回层失败 |
| scripts/check_filepath.py | 已删除 | 一次性 file_path 验证脚本 |
