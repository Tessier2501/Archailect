# My Novel RAG 实施计划书

> 交接对象: 任一后续 agent. 请先完整阅读本文档, 再按执行清单逐步实施.
> 实施环境: WSL2 Ubuntu, conda 环境 myenv (Python 3.14.6), 目标路径 ~/Archailect (Linux 原生路径).
> 权威代码: 全部实现以 src/ 目录下的源码为准, 本文档不内嵌参考代码, 以避免文档与源码漂移.
> 更新记录: 2026-08-09 全链路验证通过 (主模型切 DS 官方, 嵌入切 qwen3-embedding-0.6b(free), 测试建图+端到端查询成功); Cherry Studio 接入指南并入 §5.3.
> 更新记录: 2026-08-10 Cherry Studio (Windows 宿主) 全链路验证通过 (连通性 404 判读 + 真实问答双确认); 新增长期诊断服务 src/debug_server.py (model=debug, 独立 8001 端口, 见 §5.1.1), 日志持久化到 logs/.

---

## 1. 执行摘要

构建一个多本书独立隔离的书籍知识库问答后端:

- 前端: 使用标准 OpenAI 接口格式的 LLM 平台 (本机 Windows 侧 Cherry Studio).
- 后端: FastAPI + Uvicorn, 暴露 POST /v1/chat/completions.
- RAG 引擎: lightrag-hku (锁定 1.5.6).
- LLM: DeepSeek v4-flash (DeepSeek 官方 API, 推理模型).
- Embedding: Qwen3-Embedding-0.6B 经 cherryin 网关 (免费档).

三大核心功能:

1. 多书路由: 按请求 model 字段懒加载/缓存指向 storage/{model} 的独立 LightRAG 实例; 目录不存在时返回 OpenAI 规范错误结构 (404).
2. System Prompt 透传: 提取 messages 中所有 system 消息并原样保留, 经转义 + 模板包装后与 LightRAG 检索上下文合并, 一并交给 LLM.
3. OpenAI 兼容响应: 非流式 JSON 与流式 SSE 两种格式, 前端平台可直接渲染.

---

## 2. 技术栈与版本锁定

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
```

---

## 3. 源码验证结论

### 3.1 LightRAG 是 @dataclass

- @dataclass class LightRAG(_RoleLLMMixin, _StorageMigrationMixin, _PipelineMixin) (lightrag.py 第 385 行).
- 构造参数全部为类字段.
- 关键字段: working_dir, llm_model_func, llm_model_name, llm_model_kwargs, embedding_func (lightrag.py 第 617-722 行).
- initialize_storages (第 1556 行) / finalize_storages (第 1644 行) 存在; 代码中显式调用. **查询前必须调用 initialize_storages, 否则查询路径 pipeline_status_lock 为 None, 报 "NoneType does not support async context manager", 见 §3.9.**

### 3.2 ainsert 签名

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
(lightrag.py 第 1762-1770 行)

- 旧参数 chip_size/chunk_overlap_size 已移除; 分块由 addon_params 控制, 不传旧参数.
- ainsert 固定使用 fixed-token 分块策略, 默认即可.
- 如需 R/V/P 分块策略, 须改用 apipeline_enqueue_documents + apipeline_process_enqueue_documents. 本项目使用 ainsert.

### 3.3 aquery 签名 (System Prompt 官方注入点)

```python
async def aquery(
    self,
    query: str,
    param: QueryParam = QueryParam(),
    system_prompt: str | None = None,
) -> str | AsyncIterator[str]
```
(lightrag.py 第 3643-3648 行)

- system_prompt 为官方注入点; aquery 内部包装 aquery_llm (第 3666 行).
- stream=True 返回 AsyncIterator[str], 非流式返回 str.

### 3.4 核心陷阱: system_prompt 会被强制 .format()

lightrag/operate.py kg_query (第 4288-4293 行):

```python
sys_prompt_temp = system_prompt if system_prompt else PROMPTS["rag_response"]
sys_prompt = sys_prompt_temp.format(
    response_type=response_type,
    user_prompt=user_prompt,
    context_data=context_result.context,
)
```

推论:

1. 直接传用户 System Prompt 原文且不含 {context_data} 占位符 -> 检索上下文丢失.
2. 原文若含任意 {...} (如 JSON 示例, 正则) -> .format() 抛 KeyError/IndexError.
3. 正确做法: 服务端先把用户原文的 { 转义为 {{, } 转义为 }}, 再包装成含 {response_type}/{user_prompt}/{context_data} 占位符的模板传入 aquery(system_prompt=模板), 实现见 src/api_server.py build_rag_system_prompt().

### 3.5 QueryParam 关键字段

```python
QueryParam(
    mode="hybrid",          # local/global/hybrid/naive/mix/bypass
    stream=False,           # True 时 aquery 返回 AsyncIterator[str]
    top_k=..., chunk_top_k=...,
    user_prompt=None,       # 附加指令, 注入 prompt 模板 {user_prompt}
    conversation_history=[],# [{"role","content"},...] 仅作上下文, 不参与检索
)
```
(base.py 第 90-160 行)

- 默认 mode 为 "mix" (base.py 第 93 行), 本项目查询显式传 "hybrid", 不依赖默认值.
- enable_rerank 默认由 RERANK_BY_DEFAULT 控制 (默认 true); 未配置 rerank 模型时仅发警告.

### 3.6 LLM 封装

openai_complete_if_cache (openai.py 第 244 行): 第一位置参数是 model, 第二是 prompt.

```python
async def openai_complete_if_cache(
    model: str,
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list[dict[str, Any]] | None = None,
    enable_cot: bool = False,
    base_url: str | None = None,
    api_key: str | None = None,
    ...,
    **kwargs: Any,
) -> str
```

- 关键调用契约: LightRAG 以 llm_model_func(prompt, system_prompt=..., **kwargs) 位置调用 LLM.
- 而 openai_complete_if_cache 第一位置参数是 model: 直接裸传/llm_model_kwargs 传 model/partial 绑定 model 都会导致参数错位 (missing 'prompt' 或 multiple values for 'model').
- 正确做法: 显式 async wrapper 承接 LightRAG 的位置参数, 再以关键字转调 openai_complete_if_cache, 见 src/config.py 的 _llm_wrapper.

### 3.7 Embedding 封装

- openai_embed 本身已是 @wrap_embedding_func_with_attrs 装饰后的 EmbeddingFunc 实例 (默认 dim=1536).
- 若直接包装它, EmbeddingFunc.__post_init__ 会把外层维度 unwrap 成内层 1536.
- 必须用 openai_embed.func 取其原始未装饰函数, 再用 functools.partial 预绑定 model/base_url/api_key 包装, 见 src/config.py 的 build_embedding_func.
- EmbeddingFunc 只按 func(texts) 调用, 无 kwargs 传递机制, 因此 provider 配置必须 partial 预绑定, 否则 openai_embed 回退读环境变量 OPENAI_API_KEY 而 KeyError.

### 3.8 其他

- aquery 已解包 aquery_llm 的返回 dict (第 3666-3674 行), 业务侧直接用 aquery.
- aquery_data 返回结构化检索结果, 本项目暂不启用.

### 3.9 实测陷阱 (2026-08-09 全链路验证中发现)

1. **initialize_storages() 缺失导致查询全面失败**: LightRAG 构造后若未调用 initialize_storages(), pipeline_status/lock 等基础设施未初始化, 查询路径 `async with None` 抛 "NoneType object does not support the asynchronous context manager protocol", 且 aquery_llm 内部吞掉异常返回失败 dict, api_server 把 str(None) 当回答返回 "None". builder 因显式调用过 initialize_storages 而建图正常, 与 api_server 形成反差. **所有实例构造后必须 await initialize_storages().**
2. **load_dotenv() 默认不覆盖已存在环境变量**: shell 中 source .env 残留的旧 DEEPSEEK_MODEL 会遮蔽 .env 新值, 导致修改 .env 不生效 (实测旧 "deepseek/" 前缀导致建图 400). src/config.py 已改为 load_dotenv(override=True).
3. **DeepSeek v4-flash 是推理模型**: 官方 API 返回 reasoning_content (thinking); 非流式 aquery 只拿 content (不含 thinking), 但流式 SSE 会原样透传 `<think>...</think>` 推理帧. 若前端不需 thinking, 需在响应层过滤.
4. **DeepSeek 官方 API 模型名不带 "deepseek/" 前缀**: cherryin 网关用 "deepseek/deepseek-v4-flash", 官方 API 只认 "deepseek-v4-flash"/"deepseek-v4-pro" (带前缀实测 HTTP 400).
5. **EMBEDDING_DIM 必须与实际模型维度一致**: 实测 qwen3-embedding-0.6b 返回 1024 维 (4b=2560, 8b=4096). 误设 1536 (OpenAI ada-002 默认) 会导致 nano-vectordb 维度校验失败. 更换 embedding 模型必须重建索引 (§7 #2).
6. **多库共享内存缓存串扰 (workspace 隔离)**: LightRAG 1.5.6 的存储磁盘路径按 working_dir 隔离, 但进程内共享内存缓存/锁按 `(namespace, workspace)` 寻址; workspace 默认取 WORKSPACE 环境变量, 不传为空字符串 "". 多库实例若不传 workspace 共用 "" 命名空间, LLM 响应缓存互相覆盖, 交替查询出现双向串扰 (已实测: test2 被 test 覆盖成 Joel Kita, 反之亦然). 修复: 建图 (builder workspace=book) 与查询 (api_server workspace=model) 两侧必须传相同 workspace; 副作用是存储路径多嵌一层 workspace, 变为 storage/{book}/{book}/, 现有索引需重建.

---

## 4. 目录结构

```
~/Archailect/
├── plan.md                   # 本文档
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
│   ├── 3 - Maelstrom - Peter Watts.txt
│   ├── test-sample.txt       # 测试用切片 (Starfish 前 250 行; 全量建图完成前保留)
│   └── test2-sample.txt      # 测试用切片 (Maelstrom 前 250 行; 双库对照用)
└── storage/                  # LightRAG 索引 (git 忽略) storage/{book}/
```

存储路径注意: workspace 非空后索引位于 storage/{book}/{book}/ (如 storage/test/test/, storage/test2/test2/); storage/{book} 外层为工作目录, 内层为 workspace 数据.
状态: storage/test/、storage/test2/ (测试库) 与 data/test-sample.txt、data/test2-sample.txt (切片) 当前保留, 供 Cherry Studio 双库对照验证; 全量建库用 rifters 后目录变为 storage/rifters/rifters/.

---

## 5. 建图与运行

### 5.1 当前状态 (2026-08-09 实测)

- 主模型已切 DeepSeek 官方 API: DEEPSEEK_BASE_URL=https://api.deepseek.com, DEEPSEEK_MODEL=deepseek-v4-flash (官方模型名, 不带 "deepseek/" 前缀).
- 嵌入模型已切 qwen3-embedding-0.6b(free) (cherryin 免费档, 1024 维), EMBEDDING_DIM=1024.
- 测试文本建图实验成功: 250 行切片 -> 51 entities / 73 relations, 索引完整 (graphml + kv_store_* + vdb_*.json), 日志无 "Failed to extract".
- 端到端查询验收全通: 非流式/流式 SSE/System Prompt 透传/404 错误结构/模型列表/健康检查, 详见 §6.
- 已修复两个代码缺陷: config.py load_dotenv(override=True); api_server.py 构造后 await initialize_storages().
- Cherry Studio 接入验证: 404 判读为链路通 + 真实问答成功 (model=test 回答准确列举知识库实体/关系/文档片段, 无编造), 详见 §6.
- 双库隔离修复: 2026-08-10 发现 workspace 串扰 (双向), 已通过 builder/api_server 传 workspace 修复, test/test2 重建后 12 轮实验零串扰 (见 §3.9 #6).
- 诊断服务: src/debug_server.py 为长期诊断服务 (model=debug, 独立 8001 端口). 用于排查多库路由与系统提示词透传问题; 曾有受控实验脚本 src/dual_probe.py 已随使命完成删除.
- 待办: storage/rifters 全量建库 (三卷合并, 约 2.1MB 文本), 见 5.2 执行清单第 6 步.

### 5.1.1 调试服务 (长期诊断工具, 独立 8001 端口)

- 文件: src/debug_server.py, 独立监听 8001 端口, 不影响主服务 (8000); 与生产路由完全隔离.
- 用途: 验证前端实际发送的 model 字段 / System Prompt (role=system 消息) / 消息结构 / stream 标志; 排查多库路由与系统提示词透传问题.
- 每次请求: 完整记录到 logs/debug_requests.log (git 忽略, ensure_ascii=False) + 以 OpenAI 格式将请求体回显到对话窗.
- 零 token 成本: 不调用 LLM/Embedding, 常驻无负担.
- Cherry Studio: 添加第二个服务商指向 http://localhost:8001/v1, 模型 ID 填 debug.
- 运行: python -m src.debug_server.

### 5.2 执行清单

```bash
# 0. 激活目标 conda 环境 (myenv)
conda activate myenv

# 1. 确认 Python 版本 (>=3.10 即可)
python --version

# 2. 项目文件均已就绪 (src/requirements/.env/.gitignore 等)

# 3. 安装依赖 (myenv 已装好; 新环境重装时用)
pip install /tmp/lr_1_5_6/lightrag_hku-1.5.6-py3-none-any.whl
pip install -r requirements.txt

# 4. 配置环境变量 (.env 已存在且含真实配置; 新环境按如下键名手动创建):
#   - DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL=https://api.deepseek.com, DEEPSEEK_MODEL=deepseek-v4-flash
#   - EMBEDDING_BASE_URL=https://open.cherryin.ai/v1, EMBEDDING_API_KEY, EMBEDDING_MODEL=qwen/qwen3-embedding-0.6b(free)
#   - EMBEDDING_DIM=1024, EMBEDDING_MAX_TOKEN_SIZE=32768
#   - LLM_TIMEOUT=900 (lightrag-hku 读取, 默认 240; 建图曾因 cherryin 响应慢触发 480s worker 超时失败, 需调大)
#   - RAG_CACHE_MAX=8
#   注意: DEEPSEEK_MODEL 不带 "deepseek/" 前缀 (官方 API 只认 deepseek-v4-flash/deepseek-v4-pro);
#         EMBEDDING_DIM 必须与实际模型维度一致 (0.6b=1024), 否则 nano-vectordb 维度校验失败.

# 5. 放置书籍 txt 到 data/ 目录 (已有 Rifters 三卷)

# 6. 建图 (每系列一次; 多卷用多个 --txt 合并)
python -m src.builder --txt "data/1 - Starfish - Peter Watts.txt" \
                      --txt "data/2 - Behemoth - Peter Watts.txt" \
                      --txt "data/3 - Maelstrom - Peter Watts.txt" --book rifters
# 完成后验证: storage/rifters/ 索引完整 (graphml + kv_store_* + vdb_*.json) 且日志无 "Failed to extract"

# 7. 启动服务 (正式运行建议去掉 reload; 当前 __main__ 已改为 reload=False)
python -m src.api_server
# 默认 0.0.0.0:8000
```

### 5.3 前端 LLM 平台接入 (Cherry Studio, Windows 宿主)

1. 确保后端已启动: `python -m src.api_server` (监听 0.0.0.0:8000).
2. Cherry Studio: 设置 -> 模型服务 -> 添加服务商 -> 选 OpenAI 兼容 (自定义).
3. API 地址: `http://localhost:8000/v1` (WSL2 内置 localhost 转发; 若不通改用 VSCode 端口隧道或局域网 IP).
4. API Key: 任意占位字符串 (此服务不做鉴权, 如 `my-novel-rag`).
5. 模型 ID: 填 storage 下已建库目录名 (如 rifters) — 一个知识库 = 一个模型 ID, 这是多书路由的关键.
6. 在该模型的系统提示词中填写任意系统提示词, 验证透传生效.
7. 验证: 提问书籍相关问题 (如 "介绍一下主要角色"); 流式回复会含 `<think>...</think>` 推理内容 (DS v4-flash 特性), 非流式没有.

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

安全注意: 本服务无鉴权, 仅限本机/VSCode 转发使用. 如需局域网远程访问, 建议前置鉴权或绑定 127.0.0.1 + 反向代理.

---

## 6. 测试要点 (验收依据)

| # | 验证项 | 预期 | 状态 (2026-08-09) |
|---|---|---|---|
| 1 | POST /v1/chat/completions (非流式) | OpenAI 规范 JSON, choices[0].message.content 为回答 | ✅ 通过 (curl + Cherry Studio 真实问答) |
| 2 | 同上 + stream:true | SSE 分块输出, 结尾 data: [DONE] | ✅ 通过 (curl 798 帧; Cherry Studio 默认流式无错) |
| 3 | System Prompt 透传 | 问: "根据我给你的设定, 你是谁?" 回答与设定一致 | ✅ 通过 (curl"深渊向导"已验; Cherry Studio 端可用 debug 回显观测) |
| 4 | 多书隔离 | 不同 model 提问互不串台 | ✅ 通过 (test=Joel Kita, test2=Mermaid, 12 轮实验零串扰, workspace 隔离修复后) |
| 5 | 不存在的 model | 404 语义错误且为 OpenAI error 结构 | ✅ 实测通过 (type=invalid_request_error, code=model_not_found) |
| 6 | GET /healthz | {"status":"ok"} | ✅ 实测通过 |
| 7 | GET /v1/models | 列出 storage 下已建库目录名 | ✅ 实测通过 (列出 test) |
| 8 | 同一 model 并发首访 | 仅实例化一次, 无重复初始化 (per-model 锁) | ⏳ 待并发验收 |
| 9 | LRU 淘汰不中断在途查询 | 在途实例不被 finalize, 回答正常返回 | ⏳ 待并发验收 |
| 10 | 流式中断 (客户端断开) | 无未处理异常, 日志无 CancelledError 泄漏 | ⏳ 待验收 |
| 11 | Prompt 含 {} 字符 | 不抛 KeyError, 转义 + 模板包装生效 | ✅ 实现保证 (build_rag_system_prompt 转义) |
| 12 | QueryParam 显式 mode | mode="hybrid" 生效; 不因默认 mix 漂移 | ✅ 实现保证 (显式传 hybrid) |

---

## 7. 操作注意事项 (对未来动作的约束)

| # | 场景 | 约束 |
|---|---|---|
| 1 | 升级 lightrag-hku | 锁死 1.5.6; 升级前重验 LightRAG 字段/ainsert/aquery/QueryParam/openai_* 签名 (见第 3 节) |
| 2 | 更换 embedding 模型或维度 | 旧索引向量不匹配, 必须重建: rm -rf storage/{book} 后重跑 builder |
| 3 | 给超大文本建图 | builder 默认 200MB 单文件上限; 超大文本须预切片分批插入 |
| 4 | 新环境安装 | 项目须放 Linux 原生路径 (如 ~/Archailect), 避免 /mnt/c 性能差 |
| 5 | 多书 OOM 防护 | LRU 上限 RAG_CACHE_MAX=8, 只淘汰无在途请求的实例 |
| 6 | 全量建图前 | 先用短文本试跑验证链路, 避免浪费 LLM token |
| 7 | cherryin 响应慢导致建图超时 | 提取 worker 480s 超时失败 (已实测) | .env 设 LLM_TIMEOUT=900 调大超时上限 (lightrag-hku 读取该环境变量) |
| 8 | LightRAG 实例查询前 | 必须 await initialize_storages(), 否则 async with None 查询失败 (已实测, 见 §3.9) |
| 9 | 修改 .env 后不生效 | shell 残留同名环境变量会遮蔽 .env (load_dotenv 默认不覆盖); config.py 已用 override=True, 但 shell 手工 export 需谨慎 |
| 10 | DS 官方 API 模型名 | 不带 "deepseek/" 前缀 (deepseek-v4-flash / deepseek-v4-pro); cherryin 前缀名在官方 API 报 400 (已实测) |
| 11 | 推理模型 thinking | DS v4-flash 流式含 <think> 推理帧; 非流式 content 不含. 前端若需过滤 thinking, 在响应层处理 |
| 12 | 服务无鉴权 | 仅限本机/VSCode 转发; 远程暴露需前置鉴权或绑定 127.0.0.1 + 反向代理 |