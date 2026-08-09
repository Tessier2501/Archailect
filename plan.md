# My Novel RAG 实施计划书

> 交接对象: 任一后续 agent. 请先完整阅读本文档, 再按执行清单逐步实施.
> 实施环境: WSL2 Ubuntu, conda 环境 myenv (Python 3.14.6), 目标路径 ~/Archailect (Linux 原生路径).
> 权威代码: 全部实现以 src/ 目录下的源码为准. 本文档不内嵌参考代码, 以避免文档与源码漂移.

---

## 1. 执行摘要

构建一个多本书独立隔离的书籍知识库问答后端:

- 前端: 使用标准 OpenAI 接口格式的 LLM 平台.
- 后端: FastAPI + Uvicorn, 暴露 POST /v1/chat/completions.
- RAG 引擎: lightrag-hku (锁定 1.5.6).
- LLM: 占位符
- Embedding: 占位符

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
- initialize_storages (第 1556 行) / finalize_storages (第 1644 行) 存在; 代码中显式调用.

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
3. 正确做法: 服务端先把用户原文的 { 转义为 {{, } 转义为 }}, 再包装成含 {response_type}/{user_prompt}/{context_data} 占位符的模板传入 aquery(system_prompt=模板).

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
│   └── api_server.py         # FastAPI 主服务
├── data/                     # 原始 txt 书籍
│   ├── 1 - Starfish - Peter Watts.txt
│   ├── 2 - Behemoth - Peter Watts.txt
│   └── 3 - Maelstrom - Peter Watts.txt
└── storage/                  # LightRAG 索引 (git 忽略) storage/{book}/
```

---

## 5. 建图与运行

### 5.1 当前状态

- 建图尝试已失败: cherryin 响应慢触发 LightRAG 提取 worker 480s 超时 (15 次 TimeoutError), 文档提取被放弃, 索引残缺.
- 已在 .env 追加 LLM_TIMEOUT=900 解决超时上限; 失败进程已终止, storage/rifters 已清空待重跑.
- 重跑命令见 5.2 执行清单第 6 步; 完成后需验证: storage/rifters 下索引完整 (含 entities/relations/graphml) 且日志无 "Failed to extract". 准备更换llm和embedding提供商. 当前在文档中使用占位符占位.

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
#   - DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
#   - EMBEDDING_BASE_URL, EMBEDDING_API_KEY, EMBEDDING_MODEL
#   - EMBEDDING_DIM=4096, EMBEDDING_MAX_TOKEN_SIZE=32768
#   - LLM_TIMEOUT=900 (lightrag-hku 读取, 默认 240; cherryin 响应慢时需调大, 否则建图提取 worker 480s 超时失败)
#   - RAG_CACHE_MAX=8
#   - 模型 ID: deepseek/deepseek-v4-flash 与 qwen/qwen3-embedding-8b

# 5. 放置书籍 txt 到 data/ 目录 (已有 Rifters 三卷)

# 6. 建图 (每系列一次; 多卷用多个 --txt 合并)
python -m src.builder --txt "data/1 - Starfish - Peter Watts.txt" \
                      --txt "data/2 - Behemoth - Peter Watts.txt" \
                      --txt "data/3 - Maelstrom - Peter Watts.txt" --book rifters

# 7. 启动服务
python -m src.api_server
# 默认 0.0.0.0:8000
```

### 前端 LLM 平台接入

1. 设置 -> 接入提供方 -> 添加自定义 OpenAI 兼容服务商.
2. API 地址: http://localhost:8000/v1 (远程则填服务器 IP/域名).
3. API Key: 任意占位字符串 (此服务不做鉴权).
4. 模型名: 填 rifters (storage 下已建库目录名).
5. 在该模型的系统提示词中填写任意系统提示词, 验证透传生效.

### 冒烟测试 (curl)

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

## 6. 测试要点 (验收依据)

| # | 验证项 | 预期 |
|---|---|---|
| 1 | POST /v1/chat/completions (非流式) | OpenAI 规范 JSON, choices[0].message.content 为回答 |
| 2 | 同上 + stream:true | SSE 分块输出, 结尾 data: [DONE] |
| 3 | System Prompt 透传 | 问: "根据我给你的设定, 你是谁?" 回答与设定一致 |
| 4 | 多书隔离 | 不同 model 提问互不串台 |
| 5 | 不存在的 model | 404 语义错误且为 OpenAI error 结构 |
| 6 | GET /healthz | {"status":"ok"} |
| 7 | GET /v1/models | 列出 storage 下已建库目录名 |
| 8 | 同一 model 并发首访 | 仅实例化一次, 无重复初始化 (per-model 锁) |
| 9 | LRU 淘汰不中断在途查询 | 在途实例不被 finalize, 回答正常返回 |
| 10 | 流式中断 (客户端断开) | 无未处理异常, 日志无 CancelledError 泄漏 |
| 11 | Prompt 含 {} 字符 | 不抛 KeyError, 转义 + 模板包装生效 |
| 12 | QueryParam 显式 mode | mode="hybrid" 生效; 不因默认 mix 漂移 |

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