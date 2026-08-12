# My Novel RAG

> 多本书独立隔离的书籍知识库问答后端。前端使用标准 OpenAI 接口格式的 LLM 平台 (本机 Windows 侧 Cherry Studio), 后端基于 FastAPI + LightRAG。
> 当前状态 (2026-08-12): 已建成。 `storage/rifters/` 全量库就绪 (Rifters 三卷); 查询改写 + 双路并集 (检索优化) 已实施; 验收测试 1-7 通过; 并发相关验收项 8-10 待验证 (见 §9)。
> 权威代码: 全部实现以 `src/` 目录下的源码为准, 本文件不内嵌参考代码, 以避免文档与源码漂移。
> 诊断与改进全案: 见 `ANALYSIS.md`。

---

## 1. 项目简介

构建一个多本书独立隔离的书籍知识库问答后端:

- 前端: 使用标准 OpenAI 接口格式的 LLM 平台 (本机 Windows 侧 Cherry Studio)。
- 后端: FastAPI + Uvicorn, 暴露 `POST /v1/chat/completions`。
- RAG 引擎: lightrag-hku (锁定 1.5.6)。

三大核心功能:

1. 多书路由: 按请求 `model` 字段懒加载/缓存指向 `storage/{model}` 的独立 LightRAG 实例; 目录不存在时返回 OpenAI 规范错误结构 (404)。
2. System Prompt 透传: 提取 messages 中所有 system 消息并原样保留, 经转义 + 模板包装后与 LightRAG 检索上下文合并, 一并交给 LLM。
3. OpenAI 兼容响应: 非流式 JSON 与流式 SSE 两种格式, 前端平台可直接渲染。

---

## 2. 核心机制

### 2.1 多书路由 (并发安全 LRU 缓存)

- 每个 `model` 对应一个独立 LightRAG 实例, `working_dir` 指向 `storage/{model}`, 从物理目录层面保证知识库完全隔离。
- 实现于 `src/api_server.py`:
  - LRU 缓存: `OrderedDict` 维护热度 (`move_to_end` 更新), 容量上限 `RAG_CACHE_MAX` (默认 8, 见 `.env`)。
  - per-model 异步锁: 同一本书的首次加载仅一个协程执行实例化 (双重检查), 避免并发重复初始化。
  - 在途请求保护: 每实例维护 refcount, 淘汰前必须为 0, 防止中断正在进行的查询; 关闭时统一 `finalize_storages()` 持久化。
- 目录不存在时返回 OpenAI 规范错误: `type=invalid_request_error`, `code=model_not_found`, HTTP 404。

### 2.2 System Prompt 透传

- 提取所有 `role=system` 消息 (换行合并, 不丢弃), 交给 `aquery(system_prompt=...)`。
- 关键陷阱: LightRAG 会对传入的 system_prompt 强制调用 `str.format(...)`。因此服务端先把用户原文的 `{`/`}` 转义为 `{{`/`}}`, 再包装成含 `{context_data}`/`{user_prompt}` 占位符的模板, 实现用户角色设定与检索上下文双流无损注入。实现见 `build_rag_system_prompt()`。
- 结论: 直接传用户原文会丢失检索上下文或抛 KeyError (原文含 `{...}` 时); 转义 + 模板包装是必须的。

### 2.3 查询参数

- `QueryParam(mode="mix", top_k=20, chunk_top_k=12)`, `mode` 显式传 `"mix"` (kg+向量双通道), 不依赖默认值, 防止版本漂移。
- `top_k`/`chunk_top_k` 显式调大: 提升"因果/叙事细节"所在原文 chunk 的召回, 缓解实体图对因果时序覆盖弱导致的回答片面。
- `enable_rerank`: 随 `build_rerank_func()` 是否配置 (环境变量 RERANK_* 三键任一为空则 False, 消除无害空转警告)。

### 2.4 查询改写 + 双路检索并集 (检索优化)

为缓解"反事实/矛盾式提问"召不回答案段 (问句与答案语义结构错配), 服务端在检索前:
1. 用 QUERY 模型把用户问题改写为**聚焦检索串** (保留全部关键实体/事实; 禁止问题外想象; 允许反事实问题的"最小决策补全"; 输出仅检索串);
2. **双路并集**: 原问题 + 改写串并行检索 (mix), 按 chunk 去重合并两路候选;
3. 生成仍对准**用户原始问题** (检索串只用于召回).

实现于 `src/api_server.py` (`_rewrite_query`/`_retrieve_union`); 改写失败回退原问题, 并集异常退化为原单路 aquery (双重 fail-safe)。代价: 每查询 +1 次短 LLM 改写调用 (数百 token, 受免费/付费 fallback 保护)。

### 2.5 LLM / Embedding 封装 (src/config.py)

- LLM 分角色: QUERY (主, 必填) / KEYWORD / EXTRACT (可选, 空=回退 QUERY)。每个已配置角色均有 免费/付费双档 + 独立 fallback 电路: 免费优先 → 429/限流后"仅当次"退付费 → 下次自动回免费。
- 关键契约: LightRAG 以 `llm_model_func(prompt, system_prompt=..., **kwargs)` 位置调用 LLM, 而 `openai_complete_if_cache` 第一位置参数是 `model`。必须用显式 async wrapper 承接 LightRAG 的位置参数, 再以关键字转调 (直接裸传/`partial` 绑定 model 都会参数错位)。
- Embedding 陷阱: `openai_embed` 本身已是装饰后的 `EmbeddingFunc` 实例, 直接包装它会导致维度被 unwrap 回 1536。必须取 `openai_embed.func` 原始未装饰函数再包装; `EmbeddingFunc` 只按 `func(texts)` 调用, 无 kwargs 传递机制, 因此 provider 配置必须偏函数预绑定, 否则回退读环境变量 `OPENAI_API_KEY` 而 KeyError。

---

## 3. 技术栈与版本锁定

### requirements.txt

```txt
fastapi>=0.115
uvicorn[standard]>=0.30
lightrag-hku==1.5.6
openai>=2.0,<3.0
python-dotenv>=1.0
```

### .gitignore

```gitignore
# 索引数据, 严禁入库
/storage/

# 环境变量
.env

# Python
__pycache__/
*.pyc
*.pyo
.venv/
venv/

# 诊断日志 (debug_server 落盘)
/logs/
```

---

## 4. 目录结构

```
~/Archailect/
├── README.md                 # 本文件
├── .gitignore                # 含 /storage
├── .env                      # 真实配置 (git 忽略, 运行时 load_dotenv 读取)
├── pyrightconfig.json        # Pylance/Pyright 解析路径 (myenv site-packages)
├── requirements.txt
├── .vscode/
│   └── settings.json         # terminal.useEnvFile + defaultInterpreterPath
├── src/
│   ├── __init__.py
│   ├── config.py             # 环境变量 + LLM/Embedding 工厂
│   ├── builder.py            # 离线建图 (支持多 --txt 合并同一 book)
│   ├── api_server.py         # FastAPI 主服务
│   └── debug_server.py       # 长期诊断服务 (model=debug, 独立 8001 端口, 与主服务隔离)
├── data/                     # 原始 txt 书籍
│   ├── 1 - Starfish - Peter Watts.txt
│   ├── 2 - Behemoth - Peter Watts.txt
│   └── 3 - Maelstrom - Peter Watts.txt
├── logs/                     # debug_server 请求日志 (git 忽略, 运行时自动创建)
└── storage/                  # LightRAG 索引 (git 忽略) storage/{book}/
```

存储路径注意: workspace 非空后索引位于 `storage/{book}/{book}/`; `storage/{book}` 外层为工作目录, 内层为 workspace 数据。
当前状态: `storage/rifters/` (全量库, workspace=rifters) 已建成。

---

## 5. 快速开始

### 5.0 环境要求

- Linux 原生路径 (如 `~/Archailect`), 避免 `/mnt/c` 性能差
- Python >= 3.10 (当前开发环境: conda myenv, Python 3.14.6)
- conda 环境 (推荐用 conda 安装依赖, 而非 pip, 除非项目另行指定)

### 5.1 安装依赖

```bash
conda activate myenv

# 确认 Python 版本 (>=3.10 即可)
python --version

# 新环境首次安装: lightrag-hku 1.5.6 用本地 wheel, 其余走 requirements
pip install /tmp/lr_1_5_6/lightrag_hku-1.5.6-py3-none-any.whl
pip install -r requirements.txt
```

### 5.2 配置环境变量 (.env)

键结构如下 (`.env` 已填真实 key/模型名; 启动即校验, 占位符未填会报错防呆):

- `QUERY_BASE_URL` / `QUERY_API_KEY` — QUERY 主 LLM provider, 必填
- `KEYWORD_BASE_URL` / `KEYWORD_API_KEY`, `EXTRACT_BASE_URL` / `EXTRACT_API_KEY` — 可选, 缺省回退 QUERY
- `QUERY_FREE_MODEL` / `QUERY_PAID_MODEL` — QUERY 主模型, 付费模型名必填
- `KEYWORD_FREE_MODEL` / `KEYWORD_PAID_MODEL`, `EXTRACT_FREE_MODEL` / `EXTRACT_PAID_MODEL` — 可选, 空=回退 QUERY
- `EMBEDDING_BASE_URL` / `EMBEDDING_API_KEY` — 必填
- `EMBEDDING_MODEL_FREE` / `EMBEDDING_MODEL_PAID` — 免费模型可留空 = 仅用付费
- `EMBEDDING_DIM=1024` — 必须与 embedding 模型实际维度一致 (qwen3-embedding-0.6b 返回 1024 维; 4b=2560, 8b=4096)
- `FALLBACK_COOLDOWN=10`, `LLM_TIMEOUT=900`, `RAG_CACHE_MAX=8`

注意:

- 免费模型名留空 = 仅用付费 (适配只有付费模型的上游)。
- 免费/付费嵌入必须同模型同向量维度, 向量才可混用。
- 可选角色 (KEYWORD/EXTRACT) 若配置则经 `role_llm_configs` 注入, 留空则全由 QUERY 代劳。

### 5.3 建图 (每系列一次)

```bash
# 放置书籍 txt 到 data/ 目录; 多卷用多个 --txt 合并为同一 book
python -m src.builder --txt "data/1 - Starfish - Peter Watts.txt" \
                      --txt "data/2 - Behemoth - Peter Watts.txt" \
                      --txt "data/3 - Maelstrom - Peter Watts.txt" --book rifters
```

完成后验证: `storage/rifters/` 索引完整 (graphml + kv_store_* + vdb_*.json) 且日志无 "Failed to extract"。

注: 2026-08-12 起建图会自动为每个 chunk 记录书源路径 (`file_paths`), 后续可按书过滤书源; 存量库 (实施前建成) 的书源仍为 unknown_source, 不重建。

### 5.4 启动服务

```bash
python -m src.api_server
# 默认 0.0.0.0:8000, reload=False
```

---

## 6. 前端 LLM 平台接入 (e.g., Cherry Studio, Windows 宿主)

1. 确保后端已启动: `python -m src.api_server` (监听 0.0.0.0:8000)。
2. Cherry Studio: 设置 → 模型服务 → 添加服务商 → 选 OpenAI 兼容 (自定义)。
3. API 地址: `http://localhost:8000/v1` (WSL2 内置 localhost 转发; 若不通改用 VSCode 端口隧道或局域网 IP)。
4. API Key: 任意占位字符串 (此服务不做鉴权, 如 `my-novel-rag`)。
5. 模型 ID: 填 storage 下已建库目录名 (如 `rifters`) — 一个知识库 = 一个模型 ID, 这是多书路由的关键。
6. 在该模型的系统提示词中填写任意系统提示词, 验证透传生效。
7. 验证: 提问书籍相关问题 (如 "介绍一下主要角色"); 流式回复会含 `<think>...</think>` 推理内容 (DS v4-flash 特性), 非流式没有。

冒烟测试 (curl):

```bash
# 非流式
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"rifters","messages":[{"role":"system","content":"你是一个严谨的书评人"},{"role":"user","content":"介绍一下主要角色"}]}'

# 流式
curl -sN http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"rifters","stream":true,"messages":[{"role":"user","content":"这本书讲了什么"}]}'
```

---

## 7. 诊断服务

- 文件: `src/debug_server.py`, 独立监听 8001 端口, 不影响主服务 (8000); 与生产路由完全隔离。
- 用途: 验证前端实际发送的 `model` 字段 / System Prompt (`role=system` 消息) / 消息结构 / `stream` 标志; 排查多库路由与系统提示词透传问题。
- 每次请求: 完整记录到 `logs/debug_requests.log` (git 忽略, `ensure_ascii=False`) + 以 OpenAI 格式将请求体回显到对话窗。
- 零 token 成本: 不调用 LLM/Embedding, 常驻无负担。
- Cherry Studio: 添加第二个服务商指向 `http://localhost:8001/v1`, 模型 ID 填 `debug`。
- 运行: `python -m src.debug_server`。

---

## 8. API 一览

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/healthz` | 健康检查, 返回 `{"status":"ok"}` |
| GET | `/v1/models` | 列出 storage 下已建库目录名 (每个模型一个 ID) |
| POST | `/v1/chat/completions` | 主问答入口; `body.model` 决定路由, `stream` 决定响应格式 |

`POST /v1/chat/completions` 请求体要点:

- `model` (必填): storage 下已建库目录名; 不存在返回 404 OpenAI 规范错误。
- `messages` (必填): 至少一条; 所有 `role=system` 消息被透传, 最后一条有效 `role=user` 消息作为检索提问 (支持 OpenAI 多段 content 数组)。
- `stream` (可选, 默认 false): false 返回 JSON, true 返回 SSE (`data: [DONE]` 收尾)。
- `temperature` / `max_tokens` 接受但不影响功能; 其余 OpenAI 参数忽略。

---

## 9. 验收状态

| # | 验证项 | 预期 | 状态 (2026-08-11) |
|---|---|---|---|
| 1 | POST /v1/chat/completions (非流式) | OpenAI 规范 JSON, `choices[0].message.content` 为回答 | ✅ 通过 (curl + Cherry Studio 真实问答) |
| 2 | 同上 + `stream:true` | SSE 分块输出, 结尾 `data: [DONE]` | ✅ 通过 (curl 798 帧; Cherry Studio 默认流式无错) |
| 3 | System Prompt 透传 | 问: "根据我给你的设定, 你是谁?" 回答与设定一致 | ✅ 通过 (curl "深渊向导"已验; Cherry Studio 端可用 debug 回显观测) |
| 4 | 多书隔离 | 不同 model 提问互不串台 | ✅ 通过 (test=Joel Kita, test2=Mermaid, 12 轮实验零串扰, workspace 隔离修复后) |
| 5 | 不存在的 model | 404 语义错误且为 OpenAI error 结构 | ✅ 实测通过 (`type=invalid_request_error`, `code=model_not_found`) |
| 6 | GET /healthz | `{"status":"ok"}` | ✅ 实测通过 |
| 7 | GET /v1/models | 列出 storage 下已建库目录名 | ✅ 实测通过 (现全量库为 rifters) |
| 8 | 同一 model 并发首访 | 仅实例化一次, 无重复初始化 (per-model 锁) | ⏳ 待并发验收 |
| 9 | LRU 淘汰不中断在途查询 | 在途实例不被 finalize, 回答正常返回 | ⏳ 待并发验收 |
| 10 | 流式中断 (客户端断开) | 无未处理异常, 日志无 CancelledError 泄漏 | ⏳ 待验收 |
| 11 | Prompt 含 `{}` 字符 | 不抛 KeyError, 转义 + 模板包装生效 | ✅ 实现保证 (`build_rag_system_prompt` 转义) |
| 12 | QueryParam 显式 mode | `mode="mix"` 生效; 不因默认漂移 | ✅ 实现保证 (显式传 mix) |

---

## 10. 维护指南

### 10.1 设计关键点与已知陷阱 (为什么代码这么写)

1. LightRAG 实例查询前必须 `await initialize_storages()`: 构造后若未调用, `pipeline_status_lock` 为 None, 查询路径 `async with None` 报 "NoneType object does not support the asynchronous context manager protocol"。builder 与 api_server 均已显式调用。
2. `workspace` 必须建图/查询两侧传相同值: LightRAG 1.5.6 磁盘路径按 `working_dir` 隔离, 但进程内共享内存缓存/锁按 `(namespace, workspace)` 寻址; 不传 `workspace` 时所有实例共用 "" 命名空间, LLM 响应缓存互相覆盖出现双向串扰 (已实测)。builder 用 `workspace=book`, api_server 用 `workspace=model`。副作用: 存储路径多嵌一层 workspace, 变为 `storage/{book}/{book}/`。
3. System Prompt 会被 LightRAG 强制 `.format()`: 用户原文必须转义 `{`/`}` 再包装占位符模板, 否则检索上下文丢失或抛 KeyError。详见 §2.2。
4. LLM 封装必须用显式 async wrapper: LightRAG 位置调用 `llm_model_func(prompt, system_prompt=...)`, 而 `openai_complete_if_cache` 第一位置参数是 `model`, 参数错位会报 "missing 'prompt'" 或 "multiple values for 'model'"。见 `src/config.py` `_make_llm_wrapper`。
5. Embedding 必须取 `openai_embed.func` 原始函数: 直接包装装饰后的实例会把外层维度 unwrap 成 1536; 且 provider 必须 partial 预绑定, 否则回退读环境变量 `OPENAI_API_KEY` 而 KeyError。见 `src/config.py` `build_embedding_func`。
6. `EMBEDDING_DIM` 必须与实际模型维度一致: 误设会导致 nano-vectordb 维度校验失败。更换 embedding 模型必须重建索引 (见 §10.2 #2)。
7. DeepSeek v4-flash 是推理模型: 官方 API 返回 `reasoning_content` (thinking); 非流式 aquery 只拿 content, 但流式 SSE 会原样透传 `<think>...</think>` 推理帧。前端若不需要 thinking, 需在响应层过滤。
8. DeepSeek 官方 API 模型名不带 `deepseek/` 前缀: 只认 `deepseek-v4-flash` / `deepseek-v4-pro` (带前缀实测 HTTP 400)。
9. `load_dotenv(override=True)`: `.env` 是权威配置源, override 防止 shell 残留同名环境变量遮蔽 `.env` 新值 (已实测踩坑)。

### 10.2 操作注意事项 (对未来动作的约束)

| # | 场景 | 约束 |
|---|---|---|
| 1 | 升级 lightrag-hku | 锁死 1.5.6; 升级前重验 LightRAG 字段/ainsert/aquery/QueryParam/openai_* 签名 |
| 2 | 更换 embedding 模型或维度 | 旧索引向量不匹配, 必须重建: `rm -rf storage/{book}` 后重跑 builder |
| 3 | 给超大文本建图 | builder 默认 200MB 单文件上限; 超大文本须预切片分批插入 |
| 4 | 新环境安装 | 项目须放 Linux 原生路径 (如 `~/Archailect`), 避免 `/mnt/c` 性能差 |
| 5 | 多书 OOM 防护 | LRU 上限 `RAG_CACHE_MAX=8`, 只淘汰无在途请求的实例 |
| 6 | 全量建图前 | 先用短文本试跑验证链路, 避免浪费 LLM token |
| 7 | 上游响应慢导致建图超时 | 提取 worker 480s 超时失败 (已实测); `.env` 设 `LLM_TIMEOUT=900` 调大超时上限 (lightrag-hku 读取该环境变量) |
| 8 | LightRAG 实例查询前 | 必须 `await initialize_storages()`, 否则 `async with None` 查询失败 (已实测) |
| 9 | 修改 .env 后不生效 | shell 残留同名环境变量会遮蔽 .env (`load_dotenv` 默认不覆盖); config.py 已用 `override=True`, 但 shell 手工 export 需谨慎 |
| 10 | DS 官方 API 模型名 | 不带 `deepseek/` 前缀 (`deepseek-v4-flash` / `deepseek-v4-pro`); cherryin 前缀名在官方 API 报 400 (已实测) |
| 11 | 推理模型 thinking | DS v4-flash 流式含 `<think>` 推理帧; 非流式 content 不含. 前端若需过滤 thinking, 在响应层处理 |
| 12 | 服务无鉴权 | 仅限本机/VSCode 转发; 远程暴露需前置鉴权或绑定 127.0.0.1 + 反向代理 |
| 13 | .env 占位符未填 | config.py 启动即 `_resolve_env` 报错, 防止静默回退旧配置; 填真实值前不要运行 builder/api_server |
| 14 | 免费档限流击穿 | 全量建图曾在 flush 阶段被 0.6b(free) 429 击穿 (vdb_* 缺失但 LLM 缓存完整). 已改免费优先+付费兜底 (FALLBACK_COOLDOWN); 重跑前务必把存档的 `llm_response_cache` 放回, 否则 DS 重新计费 |
| 15 | 缓存档案 | 20MB LLM 响应缓存存于 Windows Downloads/rifters_cache_backup/; storage 内原文件勿删, 重跑前移回 `storage/rifters/rifters/` |
| 16 | provider 结构 | `QUERY_BASE_URL`/`QUERY_API_KEY` 与 `EMBEDDING_BASE_URL`/`EMBEDDING_API_KEY` 必填 (config.py `_resolve_env` 校验); KEYWORD/EXTRACT 的 `*_BASE_URL`/`*_API_KEY` 可选, 留空回退 QUERY |
| 17 | 免费模型空置 | `QUERY_FREE_MODEL` / `EMBEDDING_MODEL_FREE` 留空 = 仅用付费 (config.py `_FallbackCircuit` has_free=False 恒付费); 付费模型名必填 |

### 10.3 备份与缓存档案

- LLM 响应缓存: 20MB 缓存文件存于 Windows Downloads/rifters_cache_backup/。`storage/rifters/rifters/` 内原文件勿删; 若需重跑建图, 先把存档的 `llm_response_cache` 移回, 否则 DeepSeek 重新计费。
- 索引重建: 更换 embedding 模型/维度、或迁移 workspace 结构后, 旧索引不可复用, 须删除 `storage/{book}` 后重建 (见 §10.2 #2)。

---

## 11. 安全注意

本服务无鉴权, 仅限本机/VSCode 转发使用。如需局域网远程访问, 建议前置鉴权或绑定 127.0.0.1 + 反向代理。