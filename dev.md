# AceClaw Python 重构需求与后端设计

> 版本：0.8  
> 状态：进行中  
> 后端：**FastAPI + Uvicorn（Async HTTP + SSE）**  
> Agent 引擎：**LangChain 1.0+**（`create_agent` + 工具循环回退）  
> 语言：**Python 3.11+**

---

## 1. 目标与范围

本阶段目标是用 Python 重构并实现 OpenClaw 基础能力，先完成可运行的后端内核，覆盖：

- 异步 HTTP API（FastAPI）
- SSE 流式推送（在整轮 Agent 完成后按词片增量输出，见 §3.2）
- LangChain 1.0 Agent 运行时（工具调用；`create_agent` 不可用时回退手写多步 `bind_tools` 循环）
- 会话记忆：`*.md` 全文 transcript、`*.json` 结构化消息、周期性 **LLM 长期要点抽取**（`memory_extraction`）、可选 **`{session_id}_longterm.md` 镜像**
- Skill 扫描与管理（`skill_manager` + System Prompt 内联注入）
- 可选 **Milvus 会话向量记忆**（与 RAG 知识库分离）；本地 **LlamaIndex + BM25** 知识检索（`search_knowledge_base`）

非本阶段重点（后续迭代）：

- 完整前端 IDE
- 复杂多 Agent 编排
- 生产级多租户、分布式追踪与配额治理

---

## 2. 总体架构

### 2.1 技术选型

- Web：`FastAPI`
- ASGI Server：`Uvicorn`
- Agent：`LangChain 1.0+`（`backend/app/services/agent_runtime.py`：`create_agent` 优先，异常时回退 `_manual_agent`，单轮最多 24 步工具调用）
- LLM 服务：**DeepSeek 优先**（OpenAI 兼容接入；`temperature` 见 `config.toml` / `[llm]`）
- Embedding：**RAG 知识库** 与 **Milvus 会话记忆** 共用 `build_embedding_model`（Ollama 条件启用或 OpenAI 兼容接口；见 `model_factory`）
- 数据存储：本地文件系统（`backend/storage`，见 §4）；System Prompt 片段在 `backend/workspace/`（与 Shell 沙箱 `storage/workspace` 分离）
- 传输协议：JSON over HTTP + `text/event-stream`（SSE）

### 2.2 分层

- `api`：路由层，负责参数校验与响应协议
- `services`：业务层（`agent_runtime`、`memory_store`、`memory_extraction`、`skill_manager`、`system_prompt`、`vector_memory`、`storage_files`）
- `schemas`：Pydantic 请求/响应模型
- `core`：配置（`config.py`）、模型工厂（`model_factory.py`）、逻辑路径解析（`storage_paths.py`）
- `tools`：核心工具构建（`bootstrap.py` 及各 `*_tool.py`）
- `storage`：运行期业务文件（`skill/`、`memory/`、`workspace/`、`knowledge/`、`index/`）

### 2.3 本地存储根目录（统一约定）

- **唯一数据根目录**：`backend/storage/`。技能、记忆及后续扩展的本地文件型资源，**默认均放在该目录下**，读写路径均相对此根解析。
- **当前子目录**：
  - 技能：`backend/storage/skill/`（API 逻辑前缀 `skill/`）
  - 记忆：`backend/storage/memory/`（API 逻辑前缀 `memory/`）
- 配置常量见 `backend/app/core/config.py` 中的 `STORAGE_ROOT`、`SKILL_DIR`、`MEMORY_DIR`。

---

## 3. 核心能力设计

### 3.1 异步 HTTP API

基础接口（MVP）：

- `GET /health`：健康检查
- `GET /api/skills`：列出可用 skills；带 `?path=skill/xxx.md` 时读取单文件全文
- `POST /api/skills`：按路径保存技能 Markdown
- `POST /api/skills/reload`：重载 skills 缓存
- `GET /api/memories`：记忆文件列表；带 `?path=memory/xxx.md` 或 `memory/xxx.json` 时读取单文件全文
- `POST /api/memories`：按路径保存记忆文件
- `GET /api/sessions`：会话列表（按 `updated_at` 降序，数据来自 `storage/memory/*.json`，见 §3.7）
- `POST /api/chat`：对话；请求体字段 `stream` 为 `true` 时 **SSE 流式**，为 `false` 时 **JSON 同步**（不再单独提供 `/api/chat/stream`）

### 3.2 SSE 事件协议

`POST /api/chat` 且 `stream: true` 时，当前实现固定发送以下事件（`backend/app/api/routes/chat.py`）：

- `start`：携带 `session_id`
- `token`：增量文本片段（**注意**：Agent 与工具整轮执行结束后，再将最终答复按空格分片流式输出，**不是**模型 token 级实时流）
- `memory_saved`：`MemoryStore.append_turn` 落盘完成后触发
- `end`：携带最终 `output` 全文

数据格式：

```text
event: token
data: {"content":"..."}
```

（如需 `error` 事件或真·流式 LLM，可在后续迭代扩展。）

### 3.3 文件记忆与会话向量写入

记忆目录约定（物理路径，对应 API 逻辑路径前缀 `memory/`）：

- `backend/storage/memory/{session_id}.md`：人类可读对话全文（按轮追加 `user` / `assistant` 块）
- `backend/storage/memory/{session_id}.json`：结构化消息列表、`updated_at`，以及长期记忆字段 `long_term.bullets`（字符串数组）、`long_term.last_extracted_at` 等
- `backend/storage/memory/{session_id}_longterm.md`：长期要点列表的 Markdown 镜像（便于人工查看；与 JSON 内 bullets 同步）

策略（`memory_store.MemoryStore.append_turn`，在每次 `POST /api/chat` 得到完整助手回复后调用）：

1. 追加 `*.md`，在 `*.json` 的 `messages` 中追加本轮 user / assistant 消息（带 UTC ISO 时间戳）。
2. **长期要点抽取**（可配置）：当 `memory.long_term_enabled` 为真且 `long_term_every_n_user_turns > 0` 时，每累计 N 个 **user** 轮次，取最近 `extraction_context_messages` 条消息拼成上下文，调用 `memory_extraction.extract_long_term_bullets`（独立一次 LLM 调用，输出 JSON 字符串数组），经 `merge_bullets` 去重、截断至 `max_long_term_bullets`，写回 JSON 并刷新 `_longterm.md`。
3. 若 Milvus 已连接（§3.11）：每轮写入 **short** 类向量（整轮 user+assistant 文本）；每条**新**长期要点额外写入 **long** 类向量。

`session_id` 作为会话主键。API 中文件以逻辑路径表示，例如 `memory/main_session.md`、`memory/main_session.json`。

`GET /api/memories` 无 `path` 时返回的每条记录包含：`session_id`、`path_md`、`path_json`、`path_longterm_md`、`updated_at`（由相关文件 mtime 推导）。

### 3.4 Skill 管理

技能文件约定（物理路径，对应 API 逻辑路径前缀 `skill/`）：

- 每个技能对应 `backend/storage/skill/` 下**顶层**一个 Markdown：`*.md`。
- 列表元数据中的 `name` 为**文件名去掉扩展名**（`foo.md` → `foo`）；`path` 为相对 `storage/` 的逻辑路径（如 `skill/foo.md`）；`location` 为磁盘绝对路径。
- **描述摘要**（`description`）：取文件全文**第一行非空且不以 `#` 开头的行**（因此常见写法是 `# 标题` → `## 小节` → 一段纯文字说明，该纯文字行即成为描述）。

System Prompt 组装时（`system_prompt._auto_skills_block`），在 `SKILLS_SNAPSHOT.md` 的 `{{AUTO_SKILLS}}` 位置注入 Markdown 列表，形态如下（示意）：

```markdown
<available_skills>
- **pokemon_dmg_skill** (`skill/pokemon_dmg_skill.md`): 这个技能可以帮助你在游玩宝可梦冠军时计算属性技能克制信息
- **nyc_subway_platform_schedule_skill** (`skill/nyc_subway_platform_schedule_skill.md`): …
</available_skills>
```

读取规则：

- **服务启动时** `skill_manager.reload()` 扫描 `backend/storage/skill/*.md` 并载入缓存（`main.py` startup）。
- `POST /api/skills` 保存成功后**自动** `reload()`；亦可调用 `POST /api/skills/reload` 仅刷新缓存。
- `GET /api/skills` 返回：`name`、`path`、`location`、`description`。

### 3.5 模型接入与回退策略（新增）

#### 3.5.1 配置来源优先级

1. `backend/config.toml`
2. 环境变量（可覆盖）
3. 代码默认值（DeepSeek）

#### 3.5.2 LLM 默认策略

- 默认供应商：`deepseek`
- 默认聊天实现：配置项 `llm.model_type`（或环境变量 `LLM_MODEL_TYPE`），缺省与 `provider` 对齐；**默认使用 `langchain_deepseek.ChatDeepSeek` 初始化**
- 默认模型：`deepseek-chat`
- 默认 `base_url`：`https://api.deepseek.com/v1`
- **类型映射**（`backend/app/core/model_factory.py`，由 `main` 启动时 `init_agent_llm` 初始化）：
  - `deepseek` → `ChatDeepSeek`
  - `openai`（及别名 `gpt`、`openai_compatible`）→ `langchain_openai.ChatOpenAI`
  - `ollama` → `langchain_ollama.ChatOllama`
- **回退**：`model_type` 无法识别时，直接使用 `ChatDeepSeek`；若按映射构建实例时抛错（依赖、参数、网络校验等），`try/except` 后**回退为 `ChatDeepSeek`**
- **温度**：`llm.temperature`（`config.toml` 或环境变量覆盖合并逻辑见 `get_settings()`）

#### 3.5.3 Embedding 策略（知识库 / RAG）

- 当 `embedding.use_ollama_for_rag=true` 且 `embedding.provider=ollama` 且配置了 `embedding.model` 时：
  - 使用本地 `OllamaEmbeddings`
- 其它情况（未配置或配置不完整）：
  - 自动回退到 DeepSeek 对应配置（OpenAI Embeddings 兼容接口）
  - `embedding.model` 未设置时，默认复用 `llm.model`

#### 3.5.4 配置示例

- 示例文件：`backend/config.toml.example`
- 推荐本地使用方式：复制为 `backend/config.toml` 后填写 `llm.api_key`

#### 3.5.5 记忆子系统（`[memory]`）

`config.toml` 的 `[memory]` 段（及对应环境变量前缀 `MEMORY_*`，见 `config._from_env()`）控制：

| 配置项 | 含义 |
|--------|------|
| `short_term_messages` | 注入 System Prompt 的最近结构化消息条数上限（`user`/`assistant` 各计一条） |
| `long_term_enabled` | 是否启用周期性长期要点抽取 |
| `long_term_every_n_user_turns` | 每 N 个 user 轮触发抽取；`0` 关闭周期触发 |
| `extraction_context_messages` | 传给抽取模型的上下文消息条数 |
| `max_long_term_bullets` | 长期要点列表最大条数（超出丢弃最旧） |
| `vector_recall_long_k` / `vector_recall_short_k` | Milvus 已启用且集合含 `memory_kind` 时，从向量库分别召回 long / short 的最大条数（见 §3.11） |

### 3.6 启动配置与日志（新增）

#### 3.6.1 `.env` 启动加载策略

- 服务初始化时先读取 `.env`。
- 可通过以下方式指定目录：
  - 命令行：`--env-dir <dir>`
  - 环境变量：`ACE_CLAW_ENV_DIR=<dir>`
- 规则：
  - 若**指定了目录**，但该目录下不存在目标 `.env`（默认文件名 `.env`），服务启动直接抛异常并退出。
  - 若未指定目录，则尝试读取默认目录（`backend/`）下 `.env`，不存在时不阻断启动。

#### 3.6.2 请求日志能力

- 每次 HTTP 请求输出一条日志，包含：
  - method
  - path
  - status code
  - client ip
  - duration(ms)

#### 3.6.3 日志路径优先级

1. Uvicorn 启动命令行参数：`--log-path`
2. 环境变量：`ACE_CLAW_LOG_PATH`
3. 默认路径：项目 `backend/` 根目录下 `ace_claw.log`

#### 3.6.4 启动示例

```bash
python backend/app/main.py --reload --env-dir ./backend --log-path ./backend/logs/ace_claw.log
```
### 3.7 会话列表

- **数据源**：`backend/storage/memory/` 下的 `*.json`（排除 `*_longterm` 等后缀）；与 §3.3 会话主文件一致，**不存在**单独的 `storage/session/` 目录。
- **接口**：仅 `GET /api/sessions`，返回 `{ "sessions": [...] }`，每条为 `memory_store.list_memory_files()` 的一行（含 `session_id`、各逻辑路径、`updated_at` 降序）。
- **写入**：会话内容随 `POST /api/chat` 由 `MemoryStore.append_turn` 创建/更新；无单独「创建会话」API。

### 3.8 前端设计

- 框架：Next.js 14+ (App Router)、TypeScript

- UI 组件：Shadcn/UI、Tailwind CSS、Lucide Icons

- 编辑器：Monaco Editor（默认配置 Light Theme）

#### 3.8.1 前端布局

经典问答式风格，左中右三栏式布局：

1.左：功能导航 (Chat/Memory/Skills) + 历史会话列表
2.中：对话流展示 + 可折叠思考链可视化 (Collapsible Thoughts)
3.右：Monaco 编辑器，实时查看/编辑当前使用的 SKILL.md 或 MEMORY.md

### 3.9 服务内置工具

服务除加载用户自定义 Skills 外，必须内置以下 5 个核心基础工具（Core Tools），遵循“优先使用 LangChain 原生工具”原则，工具实现统一集中在 `backend/tools` 并在服务启动时**统一**完成初始化，技术选型与实现规范如下：

#### 3.9.1 命令行操作工具 (Command Line Interface)

- **功能描述**：允许 Agent 在受限的安全环境下执行 Shell 命令

- **实现逻辑**：`langchain_community.tools.ShellTool`，外包一层黑名单校验（`terminal_tool.py`）。

- **配置要求**：`root_dir` 固定为 **`backend/storage/workspace`**（`WORKSPACE_DIR`），防止越权写系统目录；正则黑名单拦截高危片段（如 `rm -rf /`、`shutdown`、`dd if=` 等）。

- **工具名称**：terminal

#### 3.9.2  Python 代码解释器 (Python REPL)

- **功能描述**：赋予 Agent 逻辑计算、数据处理和脚本执行的能力

- **实现逻辑**：底层为 `langchain_experimental.tools.PythonREPLTool`，经 **`StructuredTool`** 包装以兼容新版 Agent 调用签名（`python_repl_tool.py`）。

- **配置要求**：需安装 `langchain-experimental`；REPL **无**独立文件系统沙箱，勿用于执行危险代码路径。

- **工具名称**：python_repl

#### 3.9.3  Fetch 网络信息获取

- **功能描述**：获取 URL 内容或（可选）联网搜索。

- **实现逻辑**（`fetch_url_tool.py`）：对 http(s) 使用 `RequestsGetTool`；若响应像 HTML，则用 **BeautifulSoup + html2text** 转为 Markdown（过长截断约 16k）；**JSON 等非 HTML** 原样返回（过长同样截断）。若配置了 **`TAVILY_API_KEY`** 且输入**不是** URL，则走 **Tavily**；调用写入 `ace_claw` 日志，并由 **`GET /api/usage`** 汇总 token 估算。

- **工具名称**：`fetch_url`

#### 3.9.4  文件读取工具 (File Reader)

- **功能描述**：精准读取本地指定文件内容，是 Agent Skills 机制的核心依赖，用于读取 SKILL.md 详细说明

- **实现逻辑**：`langchain_community.tools.file_management.ReadFileTool`

- **配置要求**：`root_dir` 为 **`backend/`**（`BASE_DIR`）；传入相对路径如 `storage/skill/foo.md`、`storage/memory/x.md`。路径仍受宿主 OS 权限约束，勿依赖其跨仓库隔离。

- **工具名称**：read_file

#### 3.9.5  RAG 检索工具 (Hybrid Retrieval)

- **功能描述**：用户询问 **知识库**（非对话历史、非 `memory/` 文件）时，由 Agent 调用；与 `read_file`（读 `storage/skill`、`storage/memory` 逻辑路径）互补。

- **技术选型**：优先 **LlamaIndex**（`SimpleDirectoryReader` + 向量索引持久化到 `storage/index/knowledge/`）；向量嵌入使用 **OpenAI 兼容** `OpenAIEmbedding`（`settings.llm.api_key` + `embedding.model` 或默认 `text-embedding-3-small` + `embedding.base_url` / `llm.base_url`）。

- **实现逻辑**（`knowledge_tool.py`）：
  - 若 LlamaIndex 依赖齐全且存在可加载文档、嵌入初始化成功：构建 **向量检索 + BM25Retriever** 的 `QueryFusionRetriever`（失败则退化为单向量 `query_engine`）。
  - 否则：对 `storage/knowledge/` 下递归收集的 **`.md` / `.txt`** 做 **BM25Okapi** 关键词检索（不索引 PDF；PDF 需走 LlamaIndex 成功分支时由 `SimpleDirectoryReader` 加载）。
- **工具名称**：`search_knowledge_base`

---

### 3.10 System Prompt

`build_system_prompt(session_id)` 在**每次** Agent 调用时从磁盘重新读取并拼接（`backend/app/services/system_prompt.py`），顺序固定如下；块之间以 `\n\n` 分隔。

```Plain Text

┌────────────────────────────────────────────────────────────┐
│ SKILLS_SNAPSHOT.md（含 {{AUTO_SKILLS}} → 注入 §3.4 列表）   │  ← backend/workspace/
│ SOUL.md / IDENTITY.md / USER.md / AGENTS.md               │  ← 缺失时用占位说明
│ <!-- Long-term Memory (extracted) -->                     │  ← JSON long_term.bullets 或 *_longterm.md
│ <!-- Short-term Memory (recent window) -->                │  ← 最近结构化消息窗口（§3.5.5）
│ <!-- Session archival note -->                            │  ← 提示完整 transcript 在 memory/*.md（不注入正文）
└────────────────────────────────────────────────────────────┘
```

可选：若 Milvus 可用且在 `agent_runtime` 中完成召回，则在上述整体后再追加 `<!-- Vector Memory Matches -->` 区块（见 §3.11）。

### 3.11 向量记忆（可选 Milvus，会话域）

- **配置**：`config.toml` 的 `[vectordb]` 与 `.env`（`VECTOR_DB_ENABLED`、`MILVUS_URI` / `MILVUS_HOST`、`MILVUS_PORT`、`MILVUS_COLLECTION_NAME`、`VECTOR_MEMORY_TOP_K` → `top_k`、`MILVUS_EMBEDDING_DIM`、`MILVUS_METRIC_TYPE` 等）。
- **启用条件**：`vectordb.enabled=true` 且 `provider=milvus`；`main.py` 启动时 `init_vector_memory`。
- **集合 schema**：
  - **新建集合**：包含 `memory_kind`（`short` = 整轮对话摘要向量，`long` = 单条长期要点），检索时可分路召回。
  - **已存在旧集合**（无 `memory_kind` 字段）：启动日志告警，插入/检索走**兼容模式**（不按 long/short 分流；`vector_recall_*` 行为合并为单次 `search`）。
- **失败策略**：`pymilvus` 缺失、连接/建表失败时，`get_vector_memory()` 返回的对象保持 `enabled` 逻辑为假或不可连接，**不阻断**服务；仅文件记忆仍可用。
- **写入**：见 §3.3（`remember_turn` / `remember_long_fact`）。
- **召回**：`AgentRuntime._build_prompt_with_vector_memory` 在用户消息上检索，将命中列表附加在 System Prompt 末尾（支持 long/short 分栏或 legacy 单列）。

## 4. 目录与模块规划（已落地）

```text
backend/
  workspace/          # §3.10 System Prompt：SKILLS_SNAPSHOT / SOUL / IDENTITY / USER / AGENTS（每次请求重读）
  app/
    api/routes/       # health, chat, skills, memories, sessions, usage
    core/             # config, model_factory, storage_paths
    schemas/
    services/         # agent_runtime, system_prompt, memory_store, memory_extraction,
                      # skill_manager, vector_memory, storage_files
    main.py
  tools/              # §3.9 Core Tools：terminal, python_repl, fetch_url, read_file, search_knowledge_base
  storage/
    skill/            # *.md，API: skill/<name>.md
    memory/           # *.md / *.json / *_longterm.md，API: memory/<session_id>.md|json
    workspace/        # ShellTool 终端沙箱（与上列 `backend/workspace/` 不同）
    knowledge/        # RAG 源文档（MD/TXT；PDF 依赖 LlamaIndex 分支）
    index/knowledge/  # LlamaIndex 持久化索引
  requirements.txt
  README.md

frontend/             # Next.js 14 App Router，见 §3.8
  app/                # layout, page, globals.css
  components/         # app-shell, monaco-editor-panel
  lib/                # api 客户端、SSE 解析、cn()
  package.json
  README.md
```

---

## 5. 迭代计划

### Milestone 1

- 后端目录初始化
- FastAPI 应用启动
- 健康检查与基础 API
- SSE 流式输出骨架
- 文件记忆与技能扫描服务

### Milestone 2（部分已完成）

- 接入真实 LangChain ChatModel（DeepSeek / OpenAI 兼容 / Ollama）与温度配置
- Embedding Provider 选择与回退（Ollama 条件启用 ↔ OpenAI 兼容）
- **已完成**：增强 memory（短期窗口注入、周期性长期抽取、可选 Milvus 向量写入/召回）
- 待办：统一事件追踪（JSONL trace）、更细粒度 SSE（工具事件 / 错误帧 / LLM token 流）

### Milestone 3

- 工具执行策略扩展（配额、审计、更多内置工具）
- 前端联调（会话与流式渲染）

---

## 6. 运行方式

在仓库根目录执行：

```bash
pip install -r backend/requirements.txt
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

前端（另开终端）：

```bash
cd frontend
npm install
npm run dev
```

默认前端 `http://localhost:3000`，后端 `http://127.0.0.1:8000`；可通过 `frontend/.env.local` 设置 `NEXT_PUBLIC_API_BASE_URL`。

---

## 7. 后台 API 规范

### 7.1 通用约定

- **物理根目录**：`backend/storage/`。所有通过 API 读写的本地文件，均落盘在该目录下。
- **逻辑路径 → 物理路径**（默认、相对 `backend/storage/`）：
  - `skill/<rest>` → `backend/storage/skill/<rest>`
  - `memory/<rest>` → `backend/storage/memory/<rest>`
- 服务端校验逻辑路径，禁止 `..` 越界出上述子目录。
- `GET` 列表接口**无请求体**；若需读取单个文件，使用 **Query** `path`（与下表一致）。

### 7.2 接口一览


| 方法   | 路径                   | 说明                                                                        |
| ---- | -------------------- | ------------------------------------------------------------------------- |
| GET  | `/health`            | 健康检查，返回 `ok`                                                              |
| GET  | `/api/usage`         | 聚合用量；当前含 `tavily` 字段（未配置 Key 时 `configured=false`）；Tavily 调用仍写 `ace_claw` 日志 |
| POST | `/api/chat`          | 见 §7.3                                                                    |
| GET  | `/api/skills`        | 无 `path`：技能列表；有 `path=skill/xxx.md`：返回该文件 `content`                       |
| POST | `/api/skills`        | 请求体 `{"path": "skill/xxx.md", "content": "..."}` 保存并 `reload()`；响应 `{"path","status":"saved"}` |
| POST | `/api/skills/reload` | 仅重载缓存                                                                     |
| GET  | `/api/memories`      | 无 `path`：返回 `{ "memories": [...] }`（每会话一行，字段见 §3.3）；有 `path`：返回该文件 `content`（`memory/*.md` / `*.json`） |
| POST | `/api/memories`      | 请求体 `{"path": "memory/...", "content": "..."}` 保存；**不**自动重载 skill 缓存                            |
| GET  | `/api/sessions`      | `{ "sessions": [...] }`，数据同源 `GET /api/memories` 列表（§3.7）                                                   |


### 7.3 `POST /api/chat`

请求体（JSON）：

```json
{
  "message": "查询一下北京的天气",
  "session_id": "main_session",
  "stream": true
}
```

- `stream: true`：响应 `Content-Type: text/event-stream`（SSE），事件类型见 §3.2。
- `stream: false`：响应 JSON，形如 `{ "session_id": "...", "output": "..." }`。

### 7.4 验收（与 §7 对齐）

- 服务可启动且 `GET /health` 正常。
- `POST /api/chat` 在 `stream` 为 `true` / `false` 时分别对应 SSE 与 JSON。
- `GET/POST /api/skills`、`GET/POST /api/memories`、`GET /api/sessions` 行为与上表一致；磁盘文件位于 `backend/storage/skill/*`、`backend/storage/memory/*`（含可选 `*_longterm.md`）。

