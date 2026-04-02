# ResumeAgent - 多模态文档问答助手

> 基于 **LangGraph** + **智谱 GLM-4V** + **FAISS** 的智能体问答系统，聚焦简历优化与岗位分析。

## 架构

```
┌─────────────┐    SSE     ┌──────────────────────────────────────────────┐
│   前端 UI    │ ◄────────► │  FastAPI + Uvicorn                           │
│  Vanilla JS  │            │  ┌────────────────────────────────────────┐  │
└─────────────┘            │  │  LangGraph Agent 主图                  │  │
                           │  │                                        │  │
                           │  │  START → Router ──┬─ QA 路径          │  │
                           │  │                   │  search/normalize  │  │
                           │  │                   │  → generate(stream)│  │
                           │  │                   ├─ 简历分析子图       │  │
                           │  │                   │  extract_resume    │  │
                           │  │                   │  → resolve_jd      │  │
                           │  │                   │  → analysis(stream)│  │
                           │  │                   └─ JD 分析子图        │  │
                           │  │                      extract_jd         │  │
                           │  │                      → analysis(stream) │  │
                           │  └────────────────────────────────────────┘  │
                           └──────────────────────────────────────────────┘
```

### 核心能力

| 能力 | 说明 |
|------|------|
| **知识库问答** | 上传 PDF/图片/Markdown，自动切片嵌入 FAISS 向量库，支持语义检索 + RAG 生成 |
| **智能路由** | LangGraph Router 根据用户意图自动决策：知识库检索 / 联网搜索 / 直接回答 |
| **流式输出** | SSE 逐 token 流式返回，前端实时渲染 Markdown |
| **简历分析** | 上传简历（PDF/图片/文本），自动提取结构化信息，检索 JD 并生成分析报告 |
| **JD 分析** | 粘贴或上传岗位描述，自动提取结构化岗位信息并生成岗位解读与简历建议 |
| **多轮对话** | 支持 LangGraph checkpointer 按 thread 持久化会话状态，生产环境可落 PostgreSQL |
| **图片理解** | GLM-4V-Flash 视觉模型，扫描件/截图自动 OCR + 语义理解 |

## 技术栈

- **后端**: Python 3.13 / FastAPI / Uvicorn
- **AI 框架**: LangGraph 1.1 + LangChain 1.2
- **LLM / Embedding / Vision**: 智谱 AI（zai-sdk 0.2.2）— GLM-4V-Flash + Embedding-3
- **向量库**: FAISS (faiss-cpu 1.13)
- **PDF 处理**: PyMuPDF 1.27
- **前端**: 纯 HTML + CSS + Vanilla JS，嵌入 FastAPI static
- **配置管理**: pydantic-settings + .env

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
DEBIAN_MIRROR=mirrors.aliyun.com
```

其中 `DEBIAN_MIRROR` 用于 Docker 构建阶段的 Debian 软件源加速，国内服务器建议保留默认值；如果你的环境访问官方源稳定，也可以不填。

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
│   ├── main.py                 # FastAPI 入口 + lifespan
│   ├── agent/                  # LangGraph Agent 模块
│   │   ├── state.py            #   AgentState / RouteType / TaskType
│   │   ├── graph.py            #   StateGraph 主图（QA + 两个分析子图）
│   │   ├── prompts.py          #   路由/QA/JD/简历分析 Prompt 模板
│   │   ├── subgraphs/          #   分析子图定义
│   │   │   ├── resume_analysis.py
│   │   │   └── jd_analysis.py
│   │   └── nodes/
│   │       ├── router.py       #   意图路由（结构化输出 + fallback）
│   │       ├── kb_search.py    #   KB 检索（复用 RetrievalService）
│   │       ├── web_search.py   #   Web 搜索（Tavily）
│   │       ├── normalize.py    #   检索结果标准化
│   │       ├── generate.py     #   QA 生成（同步 + 图内流式桥接）
│   │       ├── extract_resume.py  # 简历结构化提取（PDF/图片/文本）
│   │       ├── retrieve_jd.py  #   JD 多查询检索 + 去重
│   │       ├── generate_analysis.py  # 简历分析报告生成
│   │       ├── extract_jd.py   #   JD 结构化提取
│   │       └── analyze_jd.py   #   JD 报告生成
│   ├── api/                    # API 路由层
│   │   ├── agent.py            #   Agent 接口（chat/stream/resume/jd/session）
│   │   ├── chat.py             #   RAG 聊天接口
│   │   └── ingest.py           #   文档上传/知识库管理
│   ├── core/                   # 核心配置
│   │   ├── config.py           #   pydantic-settings 配置
│   │   └── logger.py           #   日志配置
│   ├── loaders/                # 文档加载器（PDF/图片/Markdown/文本）
│   ├── repositories/           # 向量存储（FAISS）
│   ├── schemas/                # Pydantic Schema
│   ├── services/               # 业务服务层
│   │   ├── llm_service.py      #   智谱 GLM 调用（同步/异步/流式）
│   │   ├── embedding_service.py#   Embedding 服务
│   │   ├── vision_service.py   #   视觉理解服务（GLM-4V-Flash）
│   │   ├── rag_service.py      #   RAG 编排
│   │   ├── retrieval_service.py#   向量检索
│   │   └── web_search_service.py #   Tavily 联网搜索
│   └── utils/                  # 工具函数
├── static/                     # 前端静态文件
│   ├── index.html
│   ├── css/style.css
│   └── js/app.js
├── knowledge-base/             # 种子知识库文档（Markdown）
├── data/faiss_index/           # FAISS 索引存储（gitignore）
├── .env.example                # 环境变量模板
├── requirements.txt            # Python 依赖
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
| GET | `/health` | 健康检查 |

完整 API 文档启动后访问：http://localhost:8000/docs

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

## 部署到服务器

这一节给出一套从本地验证到服务器上线的完整路径，目标是确保两类数据都能持久化：

- **会话状态**：由 PostgreSQL 保存 LangGraph checkpointer
- **知识库数据**：由 Docker volume 保存 `data/faiss_index` 和 `data/raw`

### 持久化设计

当前项目采用两层持久化：

1. **PostgreSQL**
   用于保存多轮对话状态、session/thread checkpoint、分析链路中的中间状态。

2. **文件卷**
   用于保存：
   - `data/faiss_index`：FAISS 向量索引
   - `data/raw`：知识库原始文件

这样即使 `app` 容器重建，以下数据仍然保留：

- 已上传知识库
- 向量索引
- 用户历史会话

### 关键环境变量

至少配置以下变量：

```env
ZHIPUAI_API_KEY=your_zhipu_api_key_here
TAVILY_API_KEY=your_tavily_key_here
CHECKPOINT_DB_URL=postgresql://resumeagent:password@postgres:5432/resumeagent?sslmode=disable
```

对应模板可参考 [.env.example](/Users/superskylark/myproject/ResumeAgent/.env.example)。

### 本地容器验证

第一次建议先在本机验证，确认持久化链路正常后再上服务器。

#### 1. 启动服务

项目已提供 [docker-compose.yml](/Users/superskylark/myproject/ResumeAgent/docker-compose.yml)：

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
  "checkpointer_backend": "postgres"
}
```

如果这里还是 `memory`，说明 `CHECKPOINT_DB_URL` 没生效，或者 PostgreSQL checkpointer 没成功初始化。

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

如果能回答出之前的内容，说明“容器重启后上下文仍可恢复”。

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
  保存 FAISS 索引和元数据，决定知识库检索能力是否可恢复
- `data/raw`
  保存原始知识库文件，便于重建索引或回溯来源
- `postgres_data`
  保存 PostgreSQL 数据库文件，决定 session/checkpoint 是否可恢复

### 备份建议

最低建议备份这三部分：

1. `data/faiss_index`
2. `data/raw`
3. `postgres_data` 或 PostgreSQL 导出文件

如果做服务器迁移，只要保留这三部分，通常就可以恢复：

- 知识库索引
- 原始文档
- 历史会话状态

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

- 将简历分析和 JD 分析拆成独立 LangGraph 子图，主图仅保留路由与总调度
- `/agent/resume-analysis`、`/agent/jd-analysis` 改为基于子图流事件驱动 SSE
- `/agent/chat/stream` 现已统一走主图执行，不再在 API 层手动重放 QA 路径
- 路由器已支持 `jd_analysis` 任务类型，同一 session 下可复用 JD 分析结果做后续简历评估
- 新增 PostgreSQL checkpointer 初始化逻辑与 Docker Compose 部署模板，用于生产环境持久化

## License

MIT
