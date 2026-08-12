# My Novel RAG 分析与改进全案 (ANALYSIS)

> 本文档整合自 PLAN.md 与 IMPROVEMENTS.md (2026-08-12 合并, 原两份文档删除).
> 只读规划文档, 不内嵌参考代码 (权威代码以 src/ 为准).
> 状态: 诊断收口; 对症方案 (查询改写) 已实施; 后续优化方案 (P-1~P-5) 存档待评估, 未实施.
> 2026-08-12 精简: 移除过度特化内容 (具体角色/事件/实体/行号/数据表), 聚焦问题根本.

---

## 1. 项目与问题形态

### 1.1 项目

多本书独立隔离的书籍知识库问答后端: FastAPI 暴露 OpenAI 兼容 `POST /v1/chat/completions`,
按请求 `model` 字段路由到 `storage/{model}` 的独立 LightRAG 实例 (lightrag-hku 1.5.6 锁版);
LLM/Embedding/Rerank 均带免费优先 + 付费兜底 fallback.

### 1.2 问题形态 (抽象自真实失败案例)

RAG 在以下三类问题上表现缺陷, 实际使用中暴露:

| 形态 | 现象 | 本质 |
|---|---|---|
| 反事实/矛盾式提问 | 问"为什么某物应当 X 却未 Y"时答"该物不存在"或答无依据 | 问句与答案段语义结构错配, 检索串与答案字面/语义双不匹配 |
| 多轮指代 | 追问"它为什么没做..."时丢失前文所指 | 对话历史不参与检索 (框架限制), 指代需在改写层合并 |
| 宽泛问题 | 回答引入无关信息或错排情节 | 宽泛问题两路检索均召回大量泛化段落, 核心段被稀释 |

### 1.3 诊断对象

三卷合并库 (约 2.1MB 纯文本). 诊断目标是问题本身, 非具体书名/角色.

---

## 2. 根因收口

- **根因**: 用户原样反事实提问在任何 embedding 下都召不回答案段 (语义错配); 但**聚焦的陈述化检索串** (实体 + 具体属性 + 动作结果, 少而准) 在三个通道 (纯向量 / 纯向量检索 / 生产混合链路) 均稳定命中.
- **关键教训**: 检索串含**想象延伸** (问题外环境/属性/情节) 会把检索引偏; **聚焦陈述**才能命中. 生产端 LLM 改写"只见问题不见语料", 只能做基于问题逻辑的补全, 无法预知库内特定词.
- **对症方案 (已实施)**: 生产端查询改写, 0 重建, 每查询 +1 次短 LLM 改写调用.

---

## 3. 诊断结论摘要 (七轮)

每轮排除一个假设, 最终锁定根因:

| 轮 | 假设 | 结论 |
|---|---|---|
| 1-3 | 候选池/rerank/截断 | 答案段未进入候选池 (召回层失败), 排除后续处理 |
| 4 | 检索串措辞 (原样扩展/改写/HyDE) | 原样反事实与泛化改写全 MISS -> 措辞层对原问题形态证伪 |
| 5 | embedding 语义弱 | 疑似弱 (后被第 7 轮修正) |
| 6 | 换更大 embedding / 混合召回 | 0.6b 全库广度健康; 换模型不解决原样提问 -> 论据不足 |
| 7 | 合并层 vs 语义 (固定陈述串三通道) | 聚焦陈述串三通道 HIT -> 合并层无害、语义通道健康; 差异在检索串聚焦度 |

**排除项** (不实施): 换 embedding / BM25 混合召回; 合并层调权; 检索串扩展/别名注入; 缓存竞态加固 (asyncio 单线程下非缺陷); 实体归一化 (见 §5.7).

---

## 4. 已实施

### 4.1 查询改写 + 双路检索并集 (api_server.py)

- **改写**: 检索前用 LLM 把用户问题改写为聚焦检索串. 失败回退原问题 (fail-safe).
- **改写指令约束**:
  - 保留问题中全部关键实体/事实;
  - 禁止注入问题外的想象 (环境/属性/情节);
  - **允许**反事实问题指向的"最小决策补全" (补全替代动作/决策, 用 chose/copy/move 等通用决策词), 以定位决策段; 决策词由 LLM 从问题逻辑推得, 非从语料抄录;
  - 输出仅检索串.
- **双路并集**: 原问题 + 改写串并行检索, 按 chunk 去重合并, 两路前景互补. 检索并集异常时退化为原单路 aquery (二次 fail-safe).
- **生成**: 仍对准用户原始问题 (检索串只负责召回).

### 4.2 书源标记 (builder.py)

按库建图时对每个文件传源路径; 索引记录每段书源. 仅影响此后新建的库, 存量库不重建.

### 4.3 验证状态

- 语法/AST/导入/路由全量通过; 生产真实请求 200 OK.
- 具体反事实问题的回答已从"实体不存在"跃升为有据 (引用库内合理叙事).
- **待复核**: 决策段 (核心答案段) 在改写后的召回是否稳定达标 (答案词自答成分与召回命中率之间存在权衡, 见 §7 P-2'/P-1).

---

## 5. 提案档案 (含已证伪/不实施)

### 5.1 查询改写 (已实施, 见 §4.1)

### 5.2 书源标记 (已实施, 见 §4.2)

### 5.3 检索串扩展/别名注入 — 证伪
原样反事实问题多形态扩展全 MISS; 别名注入干扰关键词提取权重分配, 反效果. 不实施.

### 5.4 合并层调权 — 排除
聚焦陈述串在生产混合链路已 HIT 且排名靠前, 合并层无害. 调权无意义.

### 5.5 换 embedding / BM25 混合召回 — 排除
现 embedding 全库广度健康; 原样反事实问题在多个规模模型下均 MISS, 换模型不解决; 聚焦陈述串下现模型已够. 不重建.

### 5.6 缓存竞态加固 — 撤回
asyncio 单线程下检查-赋值无交错点, 字典写入原子, 竞态不成立. 不改代码.

### 5.7 实体归一化 — 撤销
替换表是"盘点式修补": 只能处理已诊断出的变体, 未来新库的新分裂形态无法在建库时预告修正, 需事后发现再补表. 不符合"零库特制、自适应、可预告"原则. 不实施. (注: 同一评审标准应用于 §7 方案选型.)

### 5.8 System Prompt 注入指称权威/因果引导 (F2/F3) — 不实施 (与 13 同批曾评估)
用户决策不实施; 命中后生成质量已由实际回答验证基本达标.

---

## 6. 技术坑与工程约束

- lightrag-hku 锁死 1.5.6; 升级前重验各签名.
- workspace 建图/查询必须同名 (builder=book, api_server=model), 否则内存缓存串台; 副作用存储路径多嵌一层.
- system_prompt 会被框架强制 `.format()`: 用户原文 `{`/`}` 需转义 (仅在走框架 aquery 路径时; 直调 LLM 路径无需).
- vector 库磁盘格式: 数据条目的向量字段为压缩快照不可直读; 真向量在矩阵字段 (base64, 需解码 + reshape + 归一化).
- 对话历史仅作生成上下文, 不参与检索; 指代须在改写层合并.
- rerank 只排序不扩大召回.
- embedding 维度须与实际模型一致; 换模型须重建索引.
- 免费档限流可击穿建图; 重跑前须放回缓存放档.
- (2026-08-12) 网关不支持请求级关闭思考 (禁用参数被静默丢弃); 但 `reasoning_effort=low` 经角色 wrapper 注入可行, KEYWORD/EXTRACT 已配 low (压缩思考省 token/提速, 输出等价), QUERY 生成通道不配 (回答质量优先). env 留空 = 不注入 (默认思考).

---

## 7. 优化方案存档 (P-1~P-5, 用户评审, 未实施)

> 针对: 宽泛问题抓不住重点、引入无关信息、情节错排. 全部方案遵循**零库特制、自适应、可预告**原则 (同 §5.7 标准), 不依赖库大小/类型/具体实体.

### P-1 两级精排: 检索后 LLM 相关性重排
并集候选 top-24 -> 用 QUERY 模型按"与**用户原始问题**的相关性"打分重排 -> 取 top-8~12 进生成.
相关性由模型对即时查询/即时候选判定, 无需预知问题形态或库结构. 成本: +1 短 LLM 调用.

### P-2' 改写策略自生成 (替代硬编码问题类型路由)
不分类型. 改写指令改为"自行判断如何将本问题改写为利于检索的串 (可陈述化/实体清单/多句), 只要不注入问题外想象".
把"类型->策略"编码交给 LLM 一次自适应决策, 新问题形态自动覆盖, 无需事后补规则. 成本: 0 (改写调用本身).

### P-3' 检索源多样性保底 (替代按分卷分组配额)
不按"书", 按并集来源簇 (每路检索各一簇) 每簇保底保留 K 个核心候选, 防单路高权重簇淹没他路有效召回.
任何库 (无论是否分卷) 均成立, 无库特制. 成本: 0 (纯分组逻辑).

### P-4 校验回路 (后台可选)
生成后用并集上下文核对回答引用与原文一致性, 错时重生成或标注. 成本: +1~2 LLM 调用 (仅长回答/宽泛问题条件触发).
自核可靠性有限, 不建议默认开.

### P-5 历史主题注入改写
改写调用并入最近轮次的主题实体 (对话历史的指代在改写层合并, 检索层不参与), 宽泛追问可链接前文所指. 成本: 0 (改写调用并入历史文本).

**评估**: P-1/P-2'/P-3'/P-5 均普适可预告, 建议作为下一波候选 (合计每查询 +2~3 短 LLM 调用); P-4 为后台可选项. 用户 2026-08-12 评审: 暂不实施, 存档待评估.

---

## 8. 当前状态与复测

- 生产服务运行中; 查询改写 + 双路并集已生效.
- 复测标准 (抽象):
  - 反事实提问: 回答应引用核心决策段逻辑, 不再答"不存在";
  - 多轮澄清: 指代应生效且不破坏简单问题;
  - 宽泛问题: 应抓住重点, 少无关信息, 情节排列合理.
- 清理: scripts/ 诊断脚本已全部删除 (git 历史保留); /tmp 本项目建设临时文件已清除; ANALYSIS.md 为本全案唯一文档, README.md 为运维权威.

---

## 9. 运维实战档案 (2026-08-12 晚: Pushing Ice 建库)

### 9.1 双库就绪

- `storage/rifters/` (Rifters 三卷) 与 `storage/pushing-ice/` (Pushing Ice 单卷) 均已建成; 主 LLM 已切 DS 官方 API (`api.deepseek.com`, `deepseek-v4-flash`, 免费档留空=仅付费), embedding/rerank 仍用 cherryin。

### 9.2 建库两次失败记录

| 轮 | 现象 | 根因 |
|---|---|---|
| 1 | cherryin 全通道 429 (all candidate channels are rate limited) → chunk-128 实体提取被 SDK 无限退避卡住 → 超 lightrag 1800s worker 上限 → 整篇 `failed` (`kg_write_state=pre_graph`), graphml 从未生成 | 上游限流风暴; lightrag worker 硬超时 (非 env `LLM_TIMEOUT` 可调, 日志 `Worker execution timeout` 来自 utils.py WorkerTimeoutError) |
| 2 | 重跑日志 `WARNING: No new unique documents were found`, 进程立即退出 | lightrag `ainsert` 去重**只按 doc_id 是否已在 doc_status 中** (`json_doc_status_impl.filter_keys = set(keys) - set(data.keys())`), 不看状态 → `failed` 文档永不自动重试 |

### 9.3 B-1 续跑方案 (已实测, 最小计费)

**原理**: 去重只看 doc_status 中 doc_id 存在性 → 删除该 doc 的 doc_status 记录 (含 `dup-*` 残留), 即可让 `ainsert` 重新接纳, 从分块后重新抽取。

**步骤** (已成功执行于 pushing-ice):
1. 备份 `kv_store_llm_response_cache.json` (5.9MB extract 缓存, **核心资产**; 先备份到库外如 Windows Downloads) 与 `kv_store_doc_status.json` (回滚保障)。
2. 删 doc_status 中该 doc_id 记录 + 其 `dup-*` 残留 (filename 重复记录, 由第二轮 "No new unique documents" 生成)。
3. 重跑 builder → 按内容哈希命中旧缓存 → **前 127/230 chunk 实体抽取零计费**, 仅剩余 ~103 chunk 走新 provider 计费; full_docs/vdb/缓存文件均不动。
4. 验证: `doc_status=processed`, graphml + 三路 vdb 落盘, 冒烟回答准确。

**教训**: 若直接 `rm -rf storage/{book}` 会连 llm_response_cache 一起删, 前 127 chunk 全部重新计费 — 先备份缓存再删库是错误路径的修正。

### 9.4 LLM 缓存 identity 分区 (源码取证)

- 缓存 key = `{mode}:{cache_type}:{hash}`, hash 含 `serialize_llm_cache_identity(identity)`; identity 由 role/binding/model/host 组成, **排除 api_key/base_url** (utils.py `get_llm_cache_identity`)。
- 本项目 role config 无 metadata + 未传 `llm_model_name` → identity.model 恒为默认 `gpt-4o-mini` (仅缓存分区标识, **不参与任何 API 调用/质量路径**)。换 provider/模型名不影响旧缓存复用; 换不同架构 LLM 会跨模型误命中缓存 → 届时须 `rm -rf storage/{book}` 重建, 或根治 (构造传 `llm_model_name` / role config 补 metadata.model, 但改后旧缓存 key 全 miss)。
- 细节: extract 缓存 cache_type 实际为 `"analysis"` (pipeline.py), 即使日志显示 `default:extract:*`; 273 条缓存 key 全部一致佐证 identity 恒定。

### 9.5 端口/进程卫生

- 物理 LISTEN 仅本项目 8000 (api_server) + 系统 DNS + VSCode Server; **无僵尸端口/进程** (WSL2 端口随进程退出自动释放)。VSCode 端口转发面板条目为 UI 层配置残留 (不耗资源), 需 UI 手动移除失效条目, CLI 无法干预。
- `/tmp/api_svr_8002.log` 等本项目临时文件已清理; `logs/build_pushing_ice.fail1.log` 已删 (保留成功轮日志)。

### 9.6 QUERY 双 provider 重构 (2026-08-12, src/config.py)

**背景**: 用户重格式化 `.env` — QUERY 从"单 provider + 双 model"升级为"**双 provider × 各自 model**": 免费档 `QUERY_FREE_BASE_URL/API_KEY/MODEL` (cherryin) + 付费档 `QUERY_PAID_BASE_URL/API_KEY/MODEL` (DS 官方)。KEYWORD/EXTRACT 合并为 `KEYWORD/EXTRACT_*` 键 (键名含 `/`)。

**代码调整** (仅 src/config.py, api_server/builder 零改动 — 工厂签名不变):
1. `_make_llm_wrapper` 重构: 入参改为**双三元组** `(base_url, api_key, model)`; 按 `_FallbackCircuit.should_use_paid()` 选择**整组** (URL+Key+Model 一起切), 429 兜底也切到 paid 整组。显式 `kwargs.pop("base_url"/"api_key"/"model")` 防 lightrag role kwargs 覆盖档位选择。
2. QUERY 免费档启用判定: `_HAS_QUERY_FREE = 三键全非空` (防半配置, 任缺 → 恒付费)。
3. 角色回退: `_role_tier_triple` — model/api/url **任一空 → 该档回退 QUERY 对应档** (用户确认); `_role_configured` — 双 model 均空 → 角色不配置 (QUERY 代劳)。
4. Embedding/Rerank 不变 (新 .env 仍单 provider 双 model)。

**验证**: `import src.config` 通过; `_HAS_QUERY_FREE=True`; free=cherryin.net/`deepseek/deepseek-v4-flash(free)`, paid=api.deepseek.com/`deepseek-v4-flash`; 角色 triple 正确回退; role_cfg=None; 三工厂可构建。重启 8000 后 healthz/models/真实问答冒烟全通过 (1m40s 回答准确)。

**意义**: 免费档 429 → 整组切付费档 (DS 官方 独立上游), 根治此前"cherryin 全通道 429 死局" (旧架构付费兜底仍在同一 cherryin 网关)。
