# ResumeAgent 技术文档

> 面向开发者的手把手复刻指南。本文档覆盖项目架构、技术选型、目录结构、核心链路、关键代码说明、部署方式、扩展方法与调试建议。

---

## 1. 项目概述

ResumeAgent 是一个基于 **FastAPI + LangGraph + LangChain** 构建的智能求职辅助系统，围绕以下核心场景设计：

- 岗位 JD 分析
- 简历分析与优化
- JD / 简历匹配评估
- 模拟面试与面试复盘
- 综合评估（总评分、能力雷达图、改进建议、推荐资源）
- 知识库问答（RAG）

项目采用 **Supervisor + Expert** 的多专家编排模型，并额外提供 **ReAct fallback** 处理非标准请求。知识库侧使用 **FAISS + PostgreSQL Metadata**，会话与状态持久化使用 **LangGraph Checkpointer + PostgreSQL**。

---

## 2. 核心技术选型

### 2.1 Web 与 API 层

- **FastAPI**
  - 提供 REST API 与 SSE 流式接口
  - 承担会话管理、文件上传、知识库导入、Agent 对话等能力

为什么选 FastAPI：
- 轻量
- 易于异步编程
- 原生支持文件上传与 StreamingResponse

### 2.2 智能体编排层

- **LangGraph**
  - 负责 `Supervisor -> Expert -> Review -> Final` 的图编排
  - 天然适合状态机、多分支执行、跨轮次持久化

- **LangChain**
  - 提供消息对象、模型调用抽象、部分工具拼装能力

为什么不是纯 ReAct：
- JD 分析、简历分析、模拟面试等都属于高确定性工作流
- LangGraph 对状态、缓存、SSE 事件更友好

### 2.3 模型服务层

- **智谱 GLM / zai SDK**
  - 普通生成
  - 流式生成
  - tools / tool_call（ReAct fallback 使用）

### 2.4 知识库层

- **FAISS**
  - 负责向量索引与语义召回

- **PostgreSQL**
  - 负责 metadata、documents、chunks、checkpoint、expert cache

为什么不是全部交给 Milvus / pgvector：
- 当前项目已沉淀了稳定的 FAISS 检索实现
- PostgreSQL 更适合业务元数据和会话状态统一持久化
- `FAISS + PostgreSQL metadata` 是当前复杂度与收益比最好的组合

### 2.5 前端层

- 原生 **HTML + CSS + JavaScript**
  - 不引入 React/Vue，便于轻量部署
  - 直接消费 SSE 事件，渲染消息、状态、雷达图、历史会话

---

## 3. 高层架构

```mermaid
graph TD
    A["前端 UI"] --> B["FastAPI"]
    B --> C["Supervisor"]

    C --> D["qa_flow"]
    C --> E["jd_expert"]
    C --> F["resume_expert"]
    C --> G["interview_expert"]
    C --> H["summary_expert"]
    C --> I["react_fallback"]

    D --> J["FAISS + PostgreSQL Metadata"]
    I --> J

    E --> K["Resume Analysis Subgraph"]
    F --> L["JD Analysis Subgraph"]

    C --> M["LangGraph Checkpointer (PostgreSQL)"]
    C --> N["Expert Cache / Tool Cache"]
```

---

## 4. 目录结构

```text
ResumeAgent/
├── app/
│   ├── agent/
│   │   ├── agents/           # Supervisor / experts / react tools / cache
│   │   ├── nodes/            # 具体节点，如 extract_resume / retrieve_jd / analyze_jd
│   │   ├── subgraphs/        # JD 分析、简历分析子图
│   │   ├── graph.py          # 主图编译
│   │   ├── prompts.py        # Prompt 模板
│   │   └── state.py          # AgentState / TaskType / RouteType
│   ├── api/
│   │   ├── agent.py          # /agent/* 路由
│   │   ├── ingest.py         # 知识库导入 / 查询 / compact
│   │   └── debug.py          # 调试接口
│   ├── repositories/
│   │   ├── vector_store.py   # FAISS + row_map
│   │   └── metadata_store.py # PostgreSQL metadata
│   ├── services/
│   │   ├── llm_service.py
│   │   ├── retrieval_service.py
│   │   └── rag_service.py
│   └── main.py               # 应用入口
├── static/
│   ├── index.html
│   ├── js/app.js
│   └── css/style.css
├── data/
│   └── faiss_index/
├── docs/
│   └── TECHNICAL_GUIDE.md
├── docker-compose.yml
├── Dockerfile
└── README.md
```

---

## 5. Agent 主图设计

### 5.1 主图入口

主图定义在：

- [/Users/superskylark/myproject/ResumeAgent/app/agent/graph.py](/Users/superskylark/myproject/ResumeAgent/app/agent/graph.py)

核心节点：

- `supervisor_plan`
- `qa_flow`
- `jd_expert`
- `resume_expert`
- `interview_expert`
- `summary_expert`
- `react_fallback`
- `supervisor_review`
- `generate_final`

### 5.2 流程

```mermaid
graph TD
    START --> supervisor_plan
    supervisor_plan --> qa_flow
    supervisor_plan --> jd_expert
    supervisor_plan --> resume_expert
    supervisor_plan --> interview_expert
    supervisor_plan --> summary_expert
    supervisor_plan --> react_fallback
    supervisor_plan --> generate_final

    qa_flow --> supervisor_review
    jd_expert --> supervisor_review
    resume_expert --> supervisor_review
    interview_expert --> supervisor_review
    summary_expert --> supervisor_review
    react_fallback --> supervisor_review

    supervisor_review --> supervisor_plan
    supervisor_review --> generate_final
    generate_final --> END
```

### 5.3 为什么这样设计

这样做的原因是：

- 核心路径稳定
- Expert 粒度足够明确
- 前端可以通过 SSE 感知每一步状态
- 后续新增 expert 容易

---

## 6. 状态模型设计

状态定义在：

- [/Users/superskylark/myproject/ResumeAgent/app/agent/state.py](/Users/superskylark/myproject/ResumeAgent/app/agent/state.py)

关键字段：

- `messages`
- `route_type`
- `task_type`
- `question_signature`
- `response_mode`
- `execution_plan`
- `agent_outputs`
- `resume_data`
- `jd_data`
- `interview_data`
- `summary_data`
- `context_sources`
- `working_context`
- `expert_cache`
- `tool_cache`

### 6.1 当前任务类型

- `qa`
- `resume_analysis`
- `jd_analysis`
- `summary_assessment`
- `interview_simulation`
- `interview_followup`
- `jd_followup`
- `resume_followup`
- `match_followup`
- `react_fallback`

---

## 7. Supervisor 设计

Supervisor 位于：

- [/Users/superskylark/myproject/ResumeAgent/app/agent/agents/supervisor.py](/Users/superskylark/myproject/ResumeAgent/app/agent/agents/supervisor.py)

它负责三件事：

1. 路由分类
2. 执行计划生成
3. 结果记录与继续/结束决策

### 7.1 路由策略

优先级：

1. 规则命中
2. ReAct fallback 判定
3. LLM 分类
4. 最终 fallback

### 7.2 为什么不用纯 ReAct

因为：

- JD/简历/面试都是固定工作流
- 纯 ReAct 成本更高且不稳定
- 当前项目更适合“固定 Expert 主干 + ReAct 扩展层”

---

## 8. Expert 设计

### 8.1 jd_expert

负责：
- 提取结构化 JD
- 生成岗位分析
- 处理 JD follow-up

核心文件：
- [/Users/superskylark/myproject/ResumeAgent/app/agent/subgraphs/jd_analysis.py](/Users/superskylark/myproject/ResumeAgent/app/agent/subgraphs/jd_analysis.py)
- [/Users/superskylark/myproject/ResumeAgent/app/agent/nodes/extract_jd.py](/Users/superskylark/myproject/ResumeAgent/app/agent/nodes/extract_jd.py)
- [/Users/superskylark/myproject/ResumeAgent/app/agent/nodes/analyze_jd.py](/Users/superskylark/myproject/ResumeAgent/app/agent/nodes/analyze_jd.py)

### 8.2 resume_expert

负责：
- 提取结构化简历
- 复用真实 JD 或知识库召回
- 输出简历评估 / 匹配分析

核心文件：
- [/Users/superskylark/myproject/ResumeAgent/app/agent/subgraphs/resume_analysis.py](/Users/superskylark/myproject/ResumeAgent/app/agent/subgraphs/resume_analysis.py)
- [/Users/superskylark/myproject/ResumeAgent/app/agent/nodes/extract_resume.py](/Users/superskylark/myproject/ResumeAgent/app/agent/nodes/extract_resume.py)
- [/Users/superskylark/myproject/ResumeAgent/app/agent/nodes/retrieve_jd.py](/Users/superskylark/myproject/ResumeAgent/app/agent/nodes/retrieve_jd.py)
- [/Users/superskylark/myproject/ResumeAgent/app/agent/nodes/generate_analysis.py](/Users/superskylark/myproject/ResumeAgent/app/agent/nodes/generate_analysis.py)

### 8.3 interview_expert

负责：
- 基于 JD、简历、历史对话生成模拟面试题
- 对用户作答评分
- 输出逐题反馈与复盘

核心文件：
- [/Users/superskylark/myproject/ResumeAgent/app/agent/agents/interview_expert.py](/Users/superskylark/myproject/ResumeAgent/app/agent/agents/interview_expert.py)

### 8.4 summary_expert

负责：
- 综合 JD、简历、面试数据和已有专家结果
- 生成总评分、雷达图、建议、资源推荐

核心文件：
- [/Users/superskylark/myproject/ResumeAgent/app/agent/agents/summary_expert.py](/Users/superskylark/myproject/ResumeAgent/app/agent/agents/summary_expert.py)

### 8.5 react_fallback

负责：
- 处理非标准请求
- 受控 tool-call
- 必要时 handoff 回标准 expert

核心文件：
- [/Users/superskylark/myproject/ResumeAgent/app/agent/agents/react_fallback.py](/Users/superskylark/myproject/ResumeAgent/app/agent/agents/react_fallback.py)
- [/Users/superskylark/myproject/ResumeAgent/app/agent/agents/react_tools.py](/Users/superskylark/myproject/ResumeAgent/app/agent/agents/react_tools.py)

---

## 9. 知识库设计：FAISS + PostgreSQL Metadata

### 9.1 为什么不是只用 metadata.json

早期方案里，所有 metadata 和 chunk 正文都在 JSON 里，问题是：

- 文件会快速膨胀
- 启动时要整体加载
- 删除 / 重建不优雅

所以现在采用：

- `FAISS`：只存向量
- `PostgreSQL`：存 documents / chunks / metadata
- `row_map.json`：记录 FAISS 行号与 chunk 的映射

### 9.2 关键文件

- [/Users/superskylark/myproject/ResumeAgent/app/repositories/vector_store.py](/Users/superskylark/myproject/ResumeAgent/app/repositories/vector_store.py)
- [/Users/superskylark/myproject/ResumeAgent/app/repositories/metadata_store.py](/Users/superskylark/myproject/ResumeAgent/app/repositories/metadata_store.py)

### 9.3 检索流程

```mermaid
graph TD
    A["query"] --> B["embedding"]
    B --> C["FAISS search"]
    C --> D["row ids"]
    D --> E["row_map.json"]
    E --> F["PostgreSQL chunks/documents"]
    F --> G["返回 content/source/metadata"]
```

---

## 10. 多查询 JD 召回策略

在没有真实 `jd_data` 时，`retrieve_jd.py` 会构造多组查询去扩大召回：

- 岗位 + 技能组合
- 技能子集
- 简历摘要语义补充

核心目标：

- 提升召回率
- 避免只靠单 query 漏召

平衡召回率和精确率的方法：

- 查询分层
- 结果去重
- 全局 score 排序

---

## 11. 缓存设计

### 11.1 会话持久化

通过 LangGraph checkpointer 实现：

- thread 级会话状态保存
- PostgreSQL 持久化

### 11.2 expert cache

当前支持：

- `resume_expert`
- `jd_expert`
- `summary_expert`

核心文件：

- [/Users/superskylark/myproject/ResumeAgent/app/agent/agents/expert_cache.py](/Users/superskylark/myproject/ResumeAgent/app/agent/agents/expert_cache.py)
- [/Users/superskylark/myproject/ResumeAgent/app/agent/agents/expert_nodes.py](/Users/superskylark/myproject/ResumeAgent/app/agent/agents/expert_nodes.py)
- [/Users/superskylark/myproject/ResumeAgent/app/agent/agents/cache_store.py](/Users/superskylark/myproject/ResumeAgent/app/agent/agents/cache_store.py)

### 11.3 tool cache

ReAct fallback 工具层有独立 cache，用于：

- `search_kb`
- `search_web`
- `extract_jd`
- `extract_resume`
- `list_documents`
- `list_sources`
- `filter_kb_by_type`

### 11.4 为什么缓存有效

因为项目存在大量 follow-up 场景：

- “我的简历还差什么？”
- “基于上面的 JD 再说说面试重点”
- “给我一个综合评估”

这些请求天然适合命中已有中间产物。

---

## 12. 低延迟设计

低延迟不是只靠算得快，而是：

1. **SSE 先发状态**
2. **token 流式输出**
3. **缓存 / 复用 / follow-up 短答**

主要事件：

- `planning`
- `agent_start`
- `agent_result`
- `status`
- `sources`
- `tool_start`
- `tool_result`
- `tool_cache_hit`
- `agent_cache_hit`
- `interview_progress`
- `summary_data`
- `token`
- `done`

前端消费逻辑在：

- [/Users/superskylark/myproject/ResumeAgent/static/js/app.js](/Users/superskylark/myproject/ResumeAgent/static/js/app.js)

---

## 13. 会话历史与前端设计

前端是原生实现，但已经具备比较完整的产品能力：

- 多会话列表
- 时间分组（今天 / 昨天 / 最近 7 天 / 更早）
- GPT 风格侧边历史
- 路由标签
- 可观测性标签
- 模拟面试状态
- 综合评估卡片与雷达图
- 导出综合评估 Markdown 报告
- 浏览器语音输入

关键文件：

- [/Users/superskylark/myproject/ResumeAgent/static/index.html](/Users/superskylark/myproject/ResumeAgent/static/index.html)
- [/Users/superskylark/myproject/ResumeAgent/static/js/app.js](/Users/superskylark/myproject/ResumeAgent/static/js/app.js)
- [/Users/superskylark/myproject/ResumeAgent/static/css/style.css](/Users/superskylark/myproject/ResumeAgent/static/css/style.css)

---

## 14. 模拟面试与综合评估设计

### 14.1 模拟面试

入口：

- 顶栏 `🎤 模拟面试`

流程：

1. 用户点击开始
2. `interview_expert` 读取 JD / 简历 / 历史消息
3. 生成题目
4. 用户逐题作答
5. 每题评分
6. 面试结束后可继续追问或复盘

### 14.2 综合评估

入口：

- 顶栏 `📊 综合评估`

流程：

1. 用户点击综合评估
2. `summary_expert` 读取：
   - `jd_data`
   - `resume_data`
   - `interview_data`
   - `agent_outputs`
3. 模型生成结构化 JSON
4. 后端返回：
   - `summary_data`
   - `final_answer`
5. 前端渲染：
   - 总评分
   - 雷达图
   - 建议
   - 推荐资源
6. 用户可点击导出按钮生成 Markdown 报告

---

## 15. API 概览

### 15.1 Agent 对话

- `POST /agent/chat`
- `POST /agent/chat/stream`

### 15.2 简历分析

- `POST /agent/resume-analysis`
- `POST /agent/resume-upload`

### 15.3 JD 分析

- `POST /agent/jd-analysis`
- `POST /agent/jd-upload`

### 15.4 会话管理

- `GET /agent/sessions`
- `GET /agent/sessions/{session_id}/messages`
- `GET /agent/session/{session_id}`
- `DELETE /agent/session/{session_id}`

### 15.5 知识库导入与管理

- `POST /ingest/file`
- `GET /ingest/documents`
- `GET /ingest/sources`
- `POST /ingest/compact`

---

## 16. 环境准备

### 16.1 Python 依赖

建议使用：

- Python 3.12

安装：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 16.2 环境变量

核心变量见：

- [/Users/superskylark/myproject/ResumeAgent/.env.example](/Users/superskylark/myproject/ResumeAgent/.env.example)

重点包括：

```env
ZHIPUAI_API_KEY=
CHECKPOINT_DB_URL=postgresql://resumeagent:password@postgres:5432/resumeagent?sslmode=disable
METADATA_DB_URL=postgresql://resumeagent:password@postgres:5432/resumeagent?sslmode=disable
EXPERT_CACHE_BACKEND=postgres
EXPERT_CACHE_DB_URL=postgresql://resumeagent:password@postgres:5432/resumeagent?sslmode=disable
```

---

## 17. 启动方式

### 17.1 本地开发

```bash
python -m uvicorn app.main:app --reload
```

### 17.2 Docker 启动

```bash
docker compose up -d --build
```

健康检查：

```bash
curl http://localhost:8000/health
```

---

## 18. 复刻步骤

如果你要从零复刻本项目，推荐按下面顺序：

### Step 1：搭 Web API 框架

- 搭建 FastAPI
- 接 `/health`
- 接简单聊天接口

### Step 2：接入模型服务

- 封装 `llm_service.py`
- 支持普通生成
- 支持流式生成
- 支持 tools

### Step 3：做知识库基础能力

- 文件上传
- 文本抽取 / 分块
- embedding
- FAISS index
- PostgreSQL metadata

### Step 4：做 QA / RAG

- `search_kb`
- `normalize`
- `generate`

### Step 5：做 JD / 简历分析

- `extract_jd`
- `extract_resume`
- `retrieve_jd`
- `generate_analysis`

### Step 6：做 LangGraph 主图

- `Supervisor`
- `qa_flow`
- `jd_expert`
- `resume_expert`
- `generate_final`

### Step 7：做缓存与状态持久化

- Checkpointer
- expert cache
- tool cache

### Step 8：做模拟面试

- `interview_expert`
- 面试问答轮次控制
- 评分与复盘

### Step 9：做综合评估

- `summary_expert`
- 雷达图
- 导出报告

### Step 10：做前端会话历史与可观测性

- SSE 状态
- GPT 风格侧边栏
- 会话恢复
- 事件标签

---

## 19. 如何新增一个 Expert

以新增 `summary_expert` 为例，步骤是：

1. 在 `TaskType` 中增加新任务
2. 新建 expert 节点文件
3. 在 `prompts.py` 增加专用 prompt
4. 在 `supervisor.py` 增加路由规则和执行计划
5. 在 `graph.py` 注册节点与边
6. 在 `api/agent.py` 增加 SSE 透传
7. 在前端增加入口与渲染

这也是后续新增：

- `mock_interview_expert`
- `resource_recommend_expert`
- `resume_rewrite_expert`

的统一套路。

---

## 20. 如何演进到 Skill 架构

当前是：

- Supervisor + Expert

建议的演进路径是：

1. 先保持 Expert 主干
2. 把 Expert 内部能力抽成 Skill Definition
3. 让 ReAct fallback 改成 skill-first
4. 再考虑 MCP 化

也就是：

**先 Expert 稳态，再 Skill 化，再 MCP 化。**

---

## 21. 常见问题

### Q1：为什么不用纯 MCP / 纯 Skill

因为当前项目核心任务是：

- 确定性强
- 对质量要求高
- 需要缓存、流式、可观测

所以 `Supervisor + Expert` 更合适。

### Q2：为什么不是纯 ReAct

因为成本更高、路径更不稳定，且会削弱现有工程化收益。

### Q3：为什么保留 FAISS

因为当前真正容易膨胀的是 metadata 层，不是向量索引本身；`FAISS + PostgreSQL Metadata` 已经足够支撑当前阶段。

---

## 22. 推荐联调清单

1. 上传知识库文档
2. 测 QA / RAG
3. 测 JD 分析
4. 测简历分析
5. 测 JD + 简历匹配追问
6. 测模拟面试
7. 测综合评估
8. 测会话切换后历史恢复
9. 测 summary 报告导出
10. 测 expert cache / tool cache 命中

---

## 23. 后续优化建议

### 23.1 后端

- 给 `summary_expert` 增加质量评审与重规划
- 给 interview / summary 结果增加结构化持久化页面
- 增加资源推荐的知识库实时检索

### 23.2 前端

- summary 图表可切换维度
- 支持 PDF 导出
- 模拟面试答题计时器

### 23.3 平台层

- 抽象内部 skill
- 做 skill registry
- 再选择性 MCP 化

---

## 24. 结语

这套项目最核心的价值，不只是“做了一个简历助手”，而是沉淀了一套可以持续扩展的智能体工程骨架：

- 稳定的 `Supervisor + Expert`
- 受控的 `ReAct fallback`
- 可扩展的知识库与 metadata 架构
- 可观测、可缓存、可流式的运行时

如果你按本文档顺序复刻，建议优先做：

1. FastAPI + LLM service
2. FAISS + PostgreSQL metadata
3. QA / JD / Resume 三条主链路
4. 再加 interview 与 summary

这样成本最低，也最容易逐步得到可运行结果。

