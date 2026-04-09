# ResumeAgent - 多模态文档问答助手

> 基于 **LangGraph** + **智谱 GLM-4V** + **FAISS** 的智能体问答系统，聚焦简历优化与岗位分析。

## 架构

```
┌─────────────┐    SSE     ┌──────────────────────────────────────────────────────┐
│   前端 UI    │ ◄────────► │  FastAPI + Uvicorn                                   │
│  Vanilla JS  │            │  ┌────────────────────────────────────────────────┐  │
└─────────────┘            │  │  LangGraph Agent 主图（Supervisor + Expert）    │  │
                           │  │                                                │  │
                           │  │  START → supervisor_plan ──┬─ QA Expert       │  │
                           │  │       ↑                    │  (KB/Web/Direct) │  │
                           │  │       │                    ├─ Resume Expert   │  │
                           │  │  supervisor_review         │  (提取+分析)     │  │
                           │  │       │                    └─ JD Expert       │  │
                           │  │       └─ generate_final                        │  │
                           │  └────────────────────────────────────────────────┘  │
                           └──────────────────────────────────────────────────────┘
```

### 多智能体架构

系统采用 **Supervisor + Expert** 架构，由 Supervisor 进行意图识别与任务分发，Expert 节点负责领域执行：

- **Supervisor**：`supervisor_plan`（意图识别 + 路由）→ `expert`（任务执行）→ `supervisor_review`（结果审查 + 循环控制）→ `generate_final`（最终输出）
- **QA Expert**：支持知识库检索 / 联网搜索 / 直接回答三种路径，KB 检索质量不足时自动降级到 Web Search
- **Resume Expert**：简历结构化提取 → JD 匹配检索 → 分析报告生成，支持追问与缓存复用
- **JD Expert**：JD 结构化提取 → 岗位解读报告，支持追问与缓存复用

### 核心能力

| 能力 | 说明 |
|------|------|
| **知识库问答** | 上传 PDF/图片/Markdown，自动切片嵌入 FAISS 向量库，支持语义检索 + RAG 生成 |
| **KB 降级搜索** | 知识库检索质量不足时（相关性低于阈值），自动降级到联网搜索，确保回答质量 |
| **智能路由** | Supervisor 根据用户意图自动决策：知识库检索 / 联网搜索 / 直接回答 |
| **流式输出** | SSE 逐 token 流式返回，前端实时渲染 Markdown |
| **简历分析** | 上传简历（PDF/图片/文本），自动提取结构化信息，检索 JD 并生成分析报告 |
| **JD 分析** | 粘贴或上传岗位描述，自动提取结构化岗位信息并生成岗位解读与简历建议 |
| **多轮对话** | 支持 LangGraph checkpointer 按 thread 持久化会话状态，生产环境可落 PostgreSQL |
| **结果缓存** | Expert 节点带 session 级缓存，相同问题+上下文命中时跳过执行，直接返回 |
| **图片理解** | GLM-4V-Flash 视觉模型，扫描件/截图自动 OCR + 语义理解 |

## 技术栈

- **后端**: Python 3.12 / FastAPI / Uvicorn
- **AI 框架**: LangGraph 1.1 + LangChain 1.2
- **LLM / Embedding / Vision**: 智谱 AI（zai-sdk 0.2.2）— GLM-4V-Flash + Embedding-3
- **向量库**: FAISS (faiss-cpu 1.13)
- **PDF 处理**: PyMuPDF 1.27
- **持久化**: PostgreSQL（会话状态 + 知识库元数据 + Expert 缓存）
- **前端**: 纯 HTML + CSS + Vanilla JS，嵌入 FastAPI static
- **配置管理**: pydantic-settings + .env
- **测试**: pytest + pytest-asyncio（25+ 测试用例）

## 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/SkylarkLiu/ResumeAgent.git
cd ResumeAgent
```

### 本地开发启动

适合本机直接运行 Python 服务进行开发调试。

#### 2. 安装依赖

```bash
pip install -r requirements.txt
```

#### 3. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env`，填入你的智谱 AI API Key：

```env
ZHIPUAI_API_KEY=your_zhipu_api_key_here
```

> 获取 API Key: [智谱 AI 开放平台](https://open.bigmodel.cn/)

#### 4. 启动服务

```bash
python -m uvicorn app.main:app --reload
```

访问 http://localhost:8000 即可使用。

### Docker 部署启动

适合本地容器验证或服务器部署。使用 Docker 时，不需要手动执行 `pip install -r requirements.txt`，依赖会在镜像构建时自动安装。

#### 2. 配置环境变量

```bash
cp .env.example .env
```

最少建议配置：

```env
ZHIPUAI_API_KEY=your_zhipu_api_key_here
CHECKPOINT_DB_URL=postgresql://resumeagent:password@postgres:5432/resumeagent?sslmode=disable
METADATA_DB_URL=postgresql://resumeagent:password@postgres:5432/resumeagent?sslmode=disable
DEBIAN_MIRROR=mirrors.aliyun.com
DEBIAN_MIRROR_SCHEME=http
PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
PIP_FALLBACK_INDEX_URL=https://pypi.org/simple
PIP_DEFAULT_TIMEOUT=180
PIP_RETRIES=10
```

其中 `DEBIAN_MIRROR` 和 `DEBIAN_MIRROR_SCHEME` 用于 Docker 构建阶段的 Debian 软件源加速，`PIP_INDEX_URL` 用于 Python 依赖下载加速，`PIP_FALLBACK_INDEX_URL` 用于主镜像源失败时自动回退官方 PyPI；`PIP_DEFAULT_TIMEOUT` 和 `PIP_RETRIES` 用于降低大包下载超时失败的概率。当前默认建议 `mirrors.aliyun.com + http`，因为部分网络环境下阿里 Debian 源 HTTPS 握手可能失败。

#### 3. 启动容器

```bash
docker compose up -d --build
```

#### 4. 验证服务

```bash
docker compose ps
curl http://localhost:8000/health
```

如果返回中包含：

```json
{"status":"ok","checkpointer_backend":"postgres"}
```

说明 Docker 部署已正常启动，且 PostgreSQL 持久化已生效。

## 项目结构

```
ResumeAgent/
├── app/
│   ├── main.py                    # FastAPI 入口 + lifespan
│   ├── agent/                     # LangGraph Agent 模块
│   │   ├── state.py               #   AgentState / RouteType / TaskType
│   │   ├── graph.py               #   StateGraph 主图（Supervisor + Expert 编排）
│   │   ├── prompts.py             #   路由/QA/JD/简历分析 Prompt 模板
│   │   ├── checkpointer.py        #   PostgreSQL checkpointer 初始化
│   │   ├── session_manager.py     #   会话管理
│   │   ├── agents/                #   多智能体模块
│   │   │   ├── supervisor.py      #     Supervisor（意图识别 + 路由 + 审查）
│   │   │   ├── expert_nodes.py    #     Expert 节点 wrapper（QA/Resume/JD）
│   │   │   ├── expert_cache.py    #     Expert 缓存逻辑
│   │   │   ├── cache_store.py     #     缓存存储后端（State/PostgreSQL）
│   │   │   └── qa_flow.py         #     QA 子图（KB → 评估 → 降级 → 生成）
│   │   ├── subgraphs/             #   分析子图定义
│   │   │   ├── resume_analysis.py
│   │   │   └── jd_analysis.py
│   │   └── nodes/
│   │       ├── kb_search.py       #   KB 检索 + 检索质量评估（降级判断）
│   │       ├── web_search.py      #   Web 搜索（Tavily）
│   │       ├── normalize.py       #   检索结果标准化
│   │       ├── generate.py        #   QA 生成（同步 + 图内流式桥接）
│   │       ├── extract_resume.py  #   简历结构化提取（PDF/图片/文本）
│   │       ├── retrieve_jd.py     #   JD 多查询检索 + 去重
│   │       ├── generate_analysis.py  # 简历分析报告生成
│   │       ├── extract_jd.py      #   JD 结构化提取
│   │       └── analyze_jd.py      #   JD 报告生成
│   ├── api/                       # API 路由层
│   │   ├── agent.py               #   Agent 接口（chat/stream/resume/jd/session）
│   │   ├── chat.py                #   RAG 聊天接口
│   │   ├── ingest.py              #   文档上传/知识库管理
│   │   └── debug.py               #   调试接口（/debug/runtime, /debug/session）
│   ├── core/                      # 核心配置
│   │   ├── config.py              #   pydantic-settings 配置
│   │   ├── logger.py              #   日志配置
│   │   └── observation.py         #   可观测性（统一日志打印）
│   ├── loaders/                   # 文档加载器
│   │   ├── pdf_loader.py          #   PDF 加载（PyMuPDF）
│   │   ├── image_loader.py        #   图片加载（GLM-4V）
│   │   └── text_loader.py         #   文本/Markdown 加载
│   ├── repositories/              # 数据存储层
│   │   ├── vector_store.py        #   FAISS 向量存储
│   │   └── metadata_store.py      #   PostgreSQL 元数据存储
│   ├── schemas/                   # Pydantic Schema
│   │   ├── agent.py
│   │   ├── chat.py
│   │   ├── file.py
│   │   └── ingest.py
│   ├── services/                  # 业务服务层
│   │   ├── llm_service.py         #   智谱 GLM 调用（同步/异步/流式）
│   │   ├── embedding_service.py   #   Embedding 服务
│   │   ├── vision_service.py      #   视觉理解服务（GLM-4V-Flash）
│   │   ├── pdf_service.py         #   PDF 处理服务
│   │   ├── rag_service.py         #   RAG 编排
│   │   ├── retrieval_service.py   #   向量检索
│   │   └── web_search_service.py  #   Tavily 联网搜索
│   └── utils/                     # 工具函数
│       ├── file_router.py         #   文件类型路由
│       └── splitter.py            #   文本分割器
├── static/                        # 前端静态文件
│   ├── index.html
│   ├── css/style.css
│   └── js/app.js
├── tests/                         # 测试
│   ├── conftest.py                #   测试配置 + fixture
│   ├── test_basic_flows.py        #   基础流程测试（9个）
│   ├── test_multi_step.py         #   多轮对话测试（3个）
│   ├── test_cache_hit.py          #   缓存命中测试（4个）
│   ├── test_kb_fallback.py        #   KB 降级测试（14个）
│   └── test_persistence.py        #   持久化测试（9个）
├── knowledge-base/                # 种子知识库文档（Markdown）
├── data/faiss_index/              # FAISS 索引存储（gitignore）
├── API_REFERENCE.md               # API 接口文档
├── pyproject.toml                 # pytest 配置
├── .env.example                   # 环境变量模板
├── requirements.txt               # Python 依赖
└── README.md
```

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/agent/chat/stream` | Agent 流式对话（SSE） |
| POST | `/agent/chat` | Agent 非流式对话 |
| POST | `/agent/resume-analysis` | 简历分析（文本粘贴） |
| POST | `/agent/resume-upload` | 简历分析（文件上传） |
| POST | `/agent/jd-analysis` | JD 分析（文本粘贴） |
| POST | `/agent/jd-upload` | JD 分析（文件上传） |
| GET | `/agent/session/{id}` | 查询会话状态 |
| DELETE | `/agent/session/{id}` | 清空会话 |
| POST | `/ingest/upload` | 上传文档到知识库 |
| GET | `/ingest/sources` | 查询知识库来源 |
| GET | `/ingest/documents` | 查询知识库文档列表 |
| POST | `/ingest/compact` | 压缩 FAISS 索引 |
| GET | `/health` | 健康检查 |

完整 API 文档：[API_REFERENCE.md](API_REFERENCE.md)，或启动后访问 http://localhost:8000/docs

## 知识库

项目内置 10 份种子文档，覆盖两大主题：

- **简历写作**（01-05）：STAR 法则、岗位画像、常见错误、JD 关键词、案例
- **智能体工程师**（06-10）：LangGraph、RAG 优化、Prompt 工程、智谱 API、技能地图

可通过 `/ingest/upload` 接口上传自定义文档扩展知识库。

## 可选配置

在 `.env` 中配置 Tavily API Key 可启用联网搜索能力：

```env
TAVILY_API_KEY=your_tavily_key_here
```

KB 检索降级阈值（默认 0.35，低于此分数自动降级到 Web Search）：

```env
KB_RELEVANCE_THRESHOLD=0.35
```

调试模式（启用后注册 `/debug/*` 接口）：

```env
APP_ENV=development
DEBUG_MODE=true
```

## 测试

```bash
# 在 Docker 容器中运行全部测试
docker exec resumeagent-app pytest tests/ -v

# 运行单个测试文件
docker exec resumeagent-app pytest tests/test_kb_fallback.py -v
```

当前测试覆盖：
- 基础流程（单轮 QA、JD 分析、简历分析、空输入错误）
- 多轮对话（JD→简历→匹配追问、追问缓存复用）
- 缓存命中（Expert 缓存命中/未命中、缓存事件结构）
- KB 降级（空结果降级、低分降级、混合类型过滤、自定义阈值）
- 持久化（后端信息、debug 路由、缓存存储、Checkpointer 初始化）

## 部署到服务器

这一节给出一套从本地验证到服务器上线的完整路径，目标是确保两类数据都能持久化：

- **会话状态**：由 PostgreSQL 保存 LangGraph checkpointer
- **知识库数据**：由 Docker volume 保存 `data/faiss_index` 和 `data/raw`

### 持久化设计

当前项目采用两层持久化：

1. **PostgreSQL**
   用于保存多轮对话状态、session/thread checkpoint、分析链路中的中间状态、Expert 缓存、知识库元数据。

2. **文件卷**
   用于保存：
   - `data/faiss_index`：FAISS 向量索引
   - `data/raw`：知识库原始文件

这样即使 `app` 容器重建，以下数据仍然保留：

- 已上传知识库
- 向量索引
- 用户历史会话
- Expert 缓存结果

### 关键环境变量

至少配置以下变量：

```env
ZHIPUAI_API_KEY=your_zhipu_api_key_here
TAVILY_API_KEY=your_tavily_key_here
CHECKPOINT_DB_URL=postgresql://resumeagent:password@postgres:5432/resumeagent?sslmode=disable
METADATA_DB_URL=postgresql://resumeagent:password@postgres:5432/resumeagent?sslmode=disable
EXPERT_CACHE_BACKEND=state_checkpointer
EXPERT_CACHE_DB_URL=
DEBIAN_MIRROR=mirrors.aliyun.com
DEBIAN_MIRROR_SCHEME=http
PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
PIP_FALLBACK_INDEX_URL=https://pypi.org/simple
PIP_DEFAULT_TIMEOUT=180
PIP_RETRIES=10
```

对应模板可参考 [.env.example](.env.example)。

### 本地容器验证

第一次建议先在本机验证，确认持久化链路正常后再上服务器。

#### 1. 启动服务

项目已提供 [docker-compose.yml](docker-compose.yml)：

```bash
docker compose up -d --build
```

默认会启动两个服务：

- `app`：FastAPI + LangGraph 应用
- `postgres`：会话持久化数据库

#### 2. 检查容器状态

```bash
docker compose ps
```

期望看到：

- `resumeagent-app` 为 `Up`
- `resumeagent-postgres` 为 `Up (healthy)`

#### 3. 健康检查

```bash
curl http://localhost:8000/health
```

返回中应包含：

```json
{
  "status": "ok",
  "checkpointer_backend": "postgres",
  "expert_cache_backend": "state_checkpointer",
  "metadata_store_backend": "postgres"
}
```

如果这里还是 `memory`，说明 `CHECKPOINT_DB_URL` 没生效，或者 PostgreSQL checkpointer 没成功初始化。

如果你想测试独立的 PostgreSQL expert cache，把 `.env` 改成：

```env
EXPERT_CACHE_BACKEND=postgres
EXPERT_CACHE_DB_URL=postgresql://resumeagent:password@postgres:5432/resumeagent?sslmode=disable
```

此时 `/health` 中应看到：

```json
{
  "expert_cache_backend": "postgres"
}
```

说明 expert cache 已切到独立 PostgreSQL 缓存表模式。

知识库 metadata 默认也建议接到 PostgreSQL。若 `METADATA_DB_URL` 留空，会自动回退复用 `CHECKPOINT_DB_URL`。

#### 4. 验证会话持久化

先发起一次对话：

```bash
curl -X POST http://localhost:8000/agent/chat \
  -H "Content-Type: application/json" \
  -d '{"question":"请记住这条测试信息：火龙果-持久化验证。只需简单回复已记住。","session_id":""}'
```

记录返回中的 `session_id`，然后查询会话：

```bash
curl http://localhost:8000/agent/session/<session_id>
```

再重启应用容器：

```bash
docker compose restart app
```

重启后再次查询：

```bash
curl http://localhost:8000/agent/session/<session_id>
```

如果 `message_count` 没掉，说明 PostgreSQL 持久化已生效。

最后再追问一次：

```bash
curl -X POST http://localhost:8000/agent/chat \
  -H "Content-Type: application/json" \
  -d '{"question":"我刚刚让你记住了什么？只回答那条内容本身。","session_id":"<session_id>"}'
```

如果能回答出之前的内容，说明"容器重启后上下文仍可恢复"。

#### 5. 验证知识库持久化

上传文档到知识库后，确认以下目录已产生数据：

- `./data/faiss_index`
- `./data/raw`

然后执行：

```bash
docker compose restart app
```

重启后访问系统并确认知识库仍可使用，即可判定文件卷持久化正常。

### 服务器部署步骤

当本地验证通过后，再部署到服务器：

#### 1. 安装基础环境

服务器需要具备：

- Docker
- Docker Compose
- 可访问智谱 API 的网络环境

#### 2. 拉取代码并配置环境变量

```bash
git clone https://github.com/SkylarkLiu/ResumeAgent.git
cd ResumeAgent
cp .env.example .env
```

编辑 `.env`，至少填写：

- `ZHIPUAI_API_KEY`
- `TAVILY_API_KEY`（可选）
- `CHECKPOINT_DB_URL`
- `DEBIAN_MIRROR`，国内服务器建议设为 `mirrors.aliyun.com`
- `DEBIAN_MIRROR_SCHEME`，建议设为 `http`
- `PIP_INDEX_URL`，国内服务器建议设为 `https://pypi.tuna.tsinghua.edu.cn/simple`
- `PIP_FALLBACK_INDEX_URL`，建议保留 `https://pypi.org/simple` 作为回退源

#### 3. 创建数据目录

建议提前创建本地目录：

```bash
mkdir -p data/faiss_index data/raw
```

#### 4. 启动服务

```bash
docker compose up -d --build
```

#### 5. 验证服务

```bash
curl http://127.0.0.1:8000/health
docker compose ps
docker compose logs --tail=100 app
docker compose logs --tail=100 postgres
```

### 数据卷说明

`docker-compose.yml` 中当前使用了以下持久化卷：

- `./data/faiss_index -> /app/data/faiss_index`
- `./data/raw -> /app/data/raw`
- `postgres_data -> /var/lib/postgresql/data`

含义如下：

- `data/faiss_index`
  保存 FAISS 索引、row_map 和迁移后的轻量 meta 信息
- `data/raw`
  保存原始知识库文件，便于重建索引或回溯来源
- `postgres_data`
  保存 PostgreSQL 数据库文件，决定 session/checkpoint、expert cache 与知识库 metadata 是否可恢复

### 备份建议

最低建议备份这三部分：

1. `data/faiss_index`
2. `data/raw`
3. `postgres_data` 或 PostgreSQL 导出文件

如果做服务器迁移，只要保留这三部分，通常就可以恢复：

- 知识库索引
- 原始文档
- 历史会话状态
- 知识库 documents/chunks metadata

### PostgreSQL Metadata 与迁移

知识库的 chunk 正文和 metadata 现在默认建议存入 PostgreSQL：

- `kb_documents`
- `kb_chunks`

FAISS 目录中保留：

- `index.faiss`
- `row_map.json`
- `meta.json`

如果启动时检测到旧版：

- `index.faiss`
- `metadata.json`

且已配置 `METADATA_DB_URL`（或 `CHECKPOINT_DB_URL`），服务会在启动时尝试自动迁移：

- `metadata.json -> PostgreSQL`
- 重建 `row_map.json`
- 旧文件备份为 `metadata.legacy.json.bak`

### 知识库管理接口

- `GET /ingest/sources`
  - 可选查询参数：`source_type`、`category`
- `GET /ingest/documents`
  - 返回 PostgreSQL metadata 中的逻辑文档列表
- `POST /ingest/compact`
  - 压缩 FAISS 索引，移除已在 PostgreSQL metadata 中删除的失效 row

### 常见问题

#### 1. `/health` 里 `checkpointer_backend` 仍然是 `memory`

通常说明：

- 没配置 `CHECKPOINT_DB_URL`
- PostgreSQL 容器没启动成功
- `langgraph-checkpoint-postgres` 依赖没安装成功

#### 2. 重启后会话丢失

优先检查：

- `postgres_data` 是否真的挂载
- `CHECKPOINT_DB_URL` 是否连到了 compose 里的 `postgres`
- PostgreSQL 容器是否被重建且没保留卷

#### 3. 重启后知识库丢失

优先检查：

- `data/faiss_index` 是否挂载
- `data/raw` 是否挂载
- 是否误删本地目录

## 最近改动

- **多智能体架构**：Supervisor + Expert 架构替换原 Router 单路由，支持循环审查与任务编排
- **Expert 缓存**：QA/Resume/JD Expert 节点带 session 级缓存，相同问题+上下文直接复用
- **KB 降级搜索**：知识库检索质量不足时自动降级到 Web Search，阈值可配置
- **可观测性增强**：统一日志打印 + 前端可观测性标签（缓存命中/降级/复用等）
- **调试接口**：`/debug/runtime` 和 `/debug/session/{id}`，仅在 debug 模式下注册
- **全流式 SSE**：所有分析路径（简历/JD/QA）统一为 SSE 流式输出
- **PostgreSQL 元数据**：知识库 chunk 元数据迁移到 PostgreSQL，支持自动迁移与索引压缩
- **回归测试体系**：25+ 测试用例覆盖基础流程、多轮对话、缓存、KB 降级、持久化

## License

MIT
