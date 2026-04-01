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
| **多轮对话** | MemorySaver checkpointer 按 thread 持久化会话状态 |
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

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env`，填入你的智谱 AI API Key：

```env
ZHIPUAI_API_KEY=your_zhipu_api_key_here
```

> 获取 API Key: [智谱 AI 开放平台](https://open.bigmodel.cn/)

### 4. 启动服务

```bash
python -m uvicorn app.main:app --reload
```

访问 http://localhost:8000 即可使用。

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

## 最近改动

- 将简历分析和 JD 分析拆成独立 LangGraph 子图，主图仅保留路由与总调度
- `/agent/resume-analysis`、`/agent/jd-analysis` 改为基于子图流事件驱动 SSE
- `/agent/chat/stream` 现已统一走主图执行，不再在 API 层手动重放 QA 路径
- 路由器已支持 `jd_analysis` 任务类型，同一 session 下可复用 JD 分析结果做后续简历评估

## License

MIT
