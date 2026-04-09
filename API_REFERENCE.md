# ResumeAgent API 接口文档

> **版本**：2.0.0  
> **基础 URL**：`http://{host}:{port}` （默认 `http://localhost:8000`）  
> **CORS**：开发环境允许全部来源（`*`）  

---

## 目录

- [通用说明](#通用说明)
- [1. 健康检查](#1-健康检查)
- [2. 知识库文件管理](#2-知识库文件管理)
  - [2.1 上传文件](#21-上传文件)
  - [2.2 已上传文件列表](#22-已上传文件列表)
  - [2.3 知识库来源列表](#23-知识库来源列表)
  - [2.4 删除知识库文件](#24-删除知识库文件)
- [3. 知识库问答](#3-知识库问答)
  - [3.1 文本问答](#31-文本问答)
  - [3.2 图片即时问答](#32-图片即时问答)
- [4. Agent 多轮对话](#4-agent-多轮对话)
  - [4.1 非流式对话](#41-非流式对话)
  - [4.2 流式对话 (SSE)](#42-流式对话-sse)
- [5. 简历分析](#5-简历分析)
  - [5.1 文本粘贴方式 (SSE)](#51-文本粘贴方式-sse)
  - [5.2 文件上传方式 (SSE)](#52-文件上传方式-sse)
- [6. JD 岗位分析](#6-jd-岗位分析)
  - [6.1 文本粘贴方式 (SSE)](#61-文本粘贴方式-sse)
  - [6.2 文件上传方式 (SSE)](#62-文件上传方式-sse)
- [7. 会话管理](#7-会话管理)
  - [7.1 获取会话信息](#71-获取会话信息)
  - [7.2 清空会话](#72-清空会话)
- [8. 调试接口](#8-调试接口)
  - [8.1 运行时状态快照](#81-运行时状态快照)
  - [8.2 会话状态详情](#82-会话状态详情)
- [附录 A：SSE 事件协议](#附录-assse-事件协议)
- [附录 B：枚举值说明](#附录-b枚举值说明)
- [附录 C：错误码参考](#附录-c错误码参考)

---

## 通用说明

### SSE 流式响应

多个接口使用 **Server-Sent Events (SSE)** 返回流式结果。SSE 响应的通用特征：

| 项目 | 说明 |
|------|------|
| Content-Type | `text/event-stream` |
| 事件格式 | `data: {JSON}\n\n` |
| 事件类型 | 通过 JSON 的 `type` 字段区分（**不是** 标准 SSE `event:` 字段） |
| 响应头 | `Cache-Control: no-cache`、`X-Accel-Buffering: no`、`Connection: keep-alive` |

> ⚠️ **注意**：本服务的 SSE 不使用标准 `event: xxx\n` 字段来区分事件类型，而是所有事件均为 `data:` 行，类型通过 JSON body 的 `type` 字段判断。

### 会话 (Session) 机制

- 每个会话由 `session_id`（即 `thread_id`）唯一标识
- `session_id` 为空时，后端自动生成 UUID hex
- 同一 `session_id` 的历史消息、简历数据 (`resume_data`)、JD 数据 (`jd_data`) 通过 checkpointer 自动持久化
- 多轮对话、简历分析、JD 分析共享同一 session 状态，可实现 **JD分析 → 简历分析 → 追问** 的跨场景联动

### 文件上传限制

| 配置项 | 默认值 |
|--------|--------|
| 最大文件大小 | 50 MB |
| 支持的文件格式 | `.pdf`, `.png`, `.jpg`, `.jpeg`, `.txt`, `.md` |

---

## 1. 健康检查

### `GET /health`

返回服务运行状态和关键后端信息。

#### 请求

无参数。

#### 响应

```json
{
  "status": "ok",
  "index_records": 128,
  "checkpointer_backend": "MemorySaver",
  "expert_cache_backend": "state_checkpointer"
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `status` | string | 固定 `"ok"` |
| `index_records` | int | FAISS 索引中的向量记录数 |
| `checkpointer_backend` | string | Checkpointer 后端类型：`MemorySaver` / `SqliteSaver` / `PostgresSaver` |
| `expert_cache_backend` | string | Expert 缓存后端：`state_checkpointer` / `postgres` |

---

## 2. 知识库文件管理

### 2.1 上传文件

### `POST /ingest/file`

上传文件到知识库，自动完成分类 → 加载 → 分块 → Embedding → 入库。

#### 请求

**Content-Type**: `multipart/form-data`

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `file` | file | ✅ | 知识库文件，支持 `.txt`/`.md`/`.png`/`.jpg`/`.jpeg`/`.pdf` |

#### 响应

```json
{
  "message": "文件导入成功: example.pdf",
  "file_type": "pdf",
  "chunks": 12,
  "pages": 5
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `message` | string | 处理结果描述 |
| `file_type` | string | 文件类型：`text` / `image` / `pdf` |
| `chunks` | int | 生成的文本块数量 |
| `pages` | int\|null | PDF 页数（非 PDF 文件为 `null`） |

#### 错误响应

| HTTP 状态码 | 场景 |
|-------------|------|
| 400 | 不支持的文件类型 / 文件过大 |
| 500 | 文件加载失败 / 向量化入库失败 |
| 503 | 向量存储未初始化 |

---

### 2.2 已上传文件列表

### `GET /ingest/files`

返回当前服务内存中记录的已上传文件清单。

#### 请求

无参数。

#### 响应

```json
{
  "files": [
    {
      "name": "example.pdf",
      "type": "pdf",
      "chunks": 12,
      "pages": 5,
      "size": 204800
    }
  ]
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `files` | array | 文件列表 |
| `files[].name` | string | 原始文件名 |
| `files[].type` | string | 文件类型 |
| `files[].chunks` | int | 分块数量 |
| `files[].pages` | int | PDF 页数 |
| `files[].size` | int | 文件大小（bytes） |

> ⚠️ 注意：此列表存储在内存中，服务重启后会清空（但 FAISS 索引数据仍保留）。

---

### 2.3 知识库来源列表

### `GET /ingest/sources`

返回 FAISS 向量存储中所有不重复的来源文件名。

#### 请求

无参数。

#### 响应

```json
{
  "sources": ["example.pdf", "resume.txt", "job_description.md"]
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `sources` | string[] | 来源文件名列表 |

---

### 2.4 删除知识库文件

### `DELETE /ingest/file/{filename}`

从知识库中删除指定来源的所有向量记录。

#### 路径参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `filename` | string | 要删除的来源文件名（与上传时的原始文件名匹配） |

#### 响应

**成功 (200)**：

```json
{
  "message": "已删除来源 'example.pdf' 的所有记录",
  "deleted": 12
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `message` | string | 操作结果描述 |
| `deleted` | int | 删除的向量记录数 |

**未找到 (404)**：

```json
{
  "message": "未找到来源 'nonexistent.pdf' 的记录",
  "deleted": 0
}
```

---

## 3. 知识库问答

### 3.1 文本问答

### `POST /chat`

基于知识库的 RAG 问答，不使用 Agent 多轮对话能力。

#### 请求

**Content-Type**: `application/json`

```json
{
  "question": "什么是 RAG？"
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `question` | string | ✅ | 用户问题，最少 1 字符 |

#### 响应

```json
{
  "answer": "RAG（Retrieval-Augmented Generation）是一种结合检索和生成的技术...",
  "sources": [
    {
      "content": "RAG 是 Retrieval-Augmented Generation 的缩写...",
      "source": "rag_intro.pdf",
      "page": 3,
      "score": 0.87
    }
  ]
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `answer` | string | 模型回答 |
| `sources` | SourceItem[] | 检索来源引用列表 |
| `sources[].content` | string | 命中的文本片段 |
| `sources[].source` | string | 来源文件名 |
| `sources[].page` | int\|null | PDF 页码 |
| `sources[].score` | float | 相似度分数 |

---

### 3.2 图片即时问答

### `POST /chat/image`

图片即时问答，不走知识库，直接调用视觉模型回答。

#### 请求

**Content-Type**: `application/json`

```json
{
  "question": "请描述这张图片的内容",
  "image_base64": "iVBORw0KGgoAAAANSUhEUgAA..."
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `question` | string | ❌ | 关于图片的问题，默认 `"请描述这张图片的内容"` |
| `image_base64` | string | ✅ | 图片 base64 编码（不含 `data:image/...;base64,` 前缀） |

#### 响应

```json
{
  "answer": "这是一张展示机器学习流程的架构图...",
  "sources": []
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `answer` | string | 视觉模型回答 |
| `sources` | array | 固定为空数组 `[]` |

---

## 4. Agent 多轮对话

### 4.1 非流式对话

### `POST /agent/chat`

Agent 多轮对话接口，等待执行完成后一次性返回。支持路由到 QA、简历分析、JD 分析等不同 Agent。

#### 请求

**Content-Type**: `application/json`

```json
{
  "question": "我的简历和 JD 匹配度如何？",
  "session_id": "a1b2c3d4e5f6"
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `question` | string | ✅ | 用户问题，最少 1 字符 |
| `session_id` | string | ❌ | 会话 ID，为空时自动生成 |

#### 响应

```json
{
  "answer": "根据您的简历和 JD 的对比分析...",
  "sources": [
    {
      "content": "前端开发工程师，3年经验...",
      "source": "frontend_jd.pdf",
      "score": 0.82,
      "type": "kb"
    }
  ],
  "session_id": "a1b2c3d4e5f6",
  "route_type": "resume_analysis",
  "task_type": "resume_analysis"
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `answer` | string | Agent 回答（Markdown 格式） |
| `sources` | AgentSourceItem[] | 来源引用列表 |
| `sources[].content` | string | 来源文本 |
| `sources[].source` | string | 来源 URL 或文件名 |
| `sources[].score` | float | 相似度分数 |
| `sources[].type` | string | 来源类型：`kb` / `web` |
| `session_id` | string | 会话 ID（用于后续多轮对话） |
| `route_type` | string | 实际路由路径，见[附录 B](#附录-b枚举值说明) |
| `task_type` | string | 任务类型，见[附录 B](#附录-b枚举值说明) |

---

### 4.2 流式对话 (SSE)

### `POST /agent/chat/stream`

Agent 多轮对话流式接口，通过 SSE 逐步返回结果。请求体与非流式接口相同。

#### 请求

与 [4.1 非流式对话](#41-非流式对话) 相同。

#### 响应

**Content-Type**: `text/event-stream`

SSE 事件按以下顺序出现：

```
data: {"type": "route", ...}           ← 路由决策
data: {"type": "planning", ...}        ← 调度规划（可选）
data: {"type": "agent_start", ...}     ← Agent 开始执行
data: {"type": "agent_cache_hit", ...} ← 缓存命中（可选）
data: {"type": "extracted", ...}       ← 结构化数据提取完成（可选）
data: {"type": "sources", ...}         ← 检索来源
data: {"type": "status", ...}          ← 状态文本（可多次）
data: {"type": "token", ...}           ← 增量文本（可多次）
data: {"type": "agent_result", ...}    ← Agent 执行完成
data: {"type": "done", ...}            ← 整体完成
data: {"type": "error", ...}           ← 错误（出现则流程终止）
```

#### 各事件详细结构

**`route`** — 路由决策

```json
{"type": "route", "route": "retrieve", "task": "qa"}
```

| 字段 | 说明 |
|------|------|
| `route` | 路由路径：`direct` / `retrieve` |
| `task` | 任务类型：`qa` / `resume_analysis` / `jd_analysis` |

**`planning`** — 调度规划

```json
{
  "type": "planning",
  "task": "resume_analysis",
  "route": "retrieve",
  "question_signature": "简历评估",
  "response_mode": "resume_first",
  "planning_reason": "检测到简历内容，触发简历分析路径"
}
```

| 字段 | 说明 |
|------|------|
| `task` | 任务类型 |
| `route` | 路由路径 |
| `question_signature` | 问题语义签名 |
| `response_mode` | 响应模式 |
| `planning_reason` | 路由决策原因 |

**`agent_start`** — Agent 开始

```json
{"type": "agent_start", "agent": "qa_flow"}
```

| 字段 | 说明 |
|------|------|
| `agent` | Agent 名称：`qa_flow` / `resume_expert` / `jd_expert` |

**`agent_cache_hit`** — 缓存命中

```json
{
  "type": "agent_cache_hit",
  "agent": "resume_expert",
  "task": "resume_analysis",
  "question_signature": "简历评估",
  "response_mode": "resume_first",
  "backend": "state_checkpointer",
  "hit_count": 2,
  "cached_at": "2026-04-09T10:30:00"
}
```

**`extracted`** — 结构化数据提取完成

```json
// 简历提取
{"type": "extracted", "resume_data": {"name": "张三", "skills": [...]}}

// JD 提取
{"type": "extracted", "jd_data": {"position": "前端工程师", "requirements": [...]}}
```

**`sources`** — 检索来源

```json
{
  "type": "sources",
  "sources": [
    {"content": "...", "source": "file.pdf", "score": 0.85, "type": "kb"}
  ]
}
```

**`status`** — 状态文本（人类可读的进度提示）

```json
{"type": "status", "content": "正在评估简历"}
```

**`token`** — 增量文本（可多次，需拼接为完整回答）

```json
{"type": "token", "content": "根据"}
{"type": "token", "content": "您的简历"}
{"type": "token", "content": "分析..."}
```

**`agent_result`** — Agent 执行完成

```json
{"type": "agent_result", "agent": "qa_flow"}
```

**`done`** — 整体完成

```json
{"type": "done", "session_id": "a1b2c3d4e5f6", "answer": "完整回答文本..."}
```

| 字段 | 说明 |
|------|------|
| `session_id` | 会话 ID |
| `answer` | 完整回答文本（也可由 `token` 事件拼接得到） |

**`error`** — 错误

```json
{"type": "error", "message": "Agent 执行出错：..."}
```

> 收到 `error` 事件后，流将终止，不会再有 `done` 事件。

---

## 5. 简历分析

### 5.1 文本粘贴方式 (SSE)

### `POST /agent/resume-analysis`

用户直接粘贴简历文本，后端提取结构化信息 → 检索相关 JD → 流式生成分析报告。

#### 请求

**Content-Type**: `application/json`

```json
{
  "resume_text": "张三 | 前端开发工程师\n3年工作经验\n技能：React, TypeScript...",
  "question": "请对我的简历进行全面分析评估",
  "session_id": "",
  "target_position": "高级前端工程师"
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `resume_text` | string | ✅ | 简历文本内容（直接粘贴） |
| `question` | string | ❌ | 分析要求，默认 `"请对我的简历进行全面分析评估"` |
| `session_id` | string | ❌ | 会话 ID，为空时自动生成 |
| `target_position` | string | ❌ | 目标岗位（可选，帮助更精准匹配） |

#### 响应

**Content-Type**: `text/event-stream`

SSE 事件序列：

```
data: {"type": "extracted", "resume_data": {...}}   ← 简历提取完成
data: {"type": "sources", "sources": [...]}          ← 相关 JD 来源
data: {"type": "status", "content": "..."}           ← 状态提示（可多次）
data: {"type": "token", "content": "..."}            ← 增量文本（可多次）
data: {"type": "done", ...}                          ← 完成
data: {"type": "error", ...}                         ← 错误
```

**`extracted` 事件**：

```json
{
  "type": "extracted",
  "resume_data": {
    "name": "张三",
    "target_position": "高级前端工程师",
    "skills": ["React", "TypeScript", "Node.js"],
    "experience": [...],
    "education": [...]
  }
}
```

> `resume_data` 不含 `raw_text` 字段（已过滤）。

**`done` 事件**：

```json
{
  "type": "done",
  "session_id": "a1b2c3d4e5f6",
  "answer": "## 简历分析报告\n\n### 优势\n...",
  "resume_data": {"name": "张三", "skills": [...]}
}
```

| 字段 | 说明 |
|------|------|
| `session_id` | 会话 ID |
| `answer` | 完整分析报告（Markdown） |
| `resume_data` | 结构化简历信息（不含 `raw_text`） |

---

### 5.2 文件上传方式 (SSE)

### `POST /agent/resume-upload`

上传简历文件（PDF/图片/文本），后端自动提取内容后执行简历分析。

#### 请求

**Content-Type**: `multipart/form-data`

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `file` | file | ✅ | 简历文件（`.pdf`/`.png`/`.jpg`/`.jpeg`/`.txt`/`.md`） |
| `question` | string | ❌ | 分析要求，默认 `"请对我的简历进行全面分析评估"` |
| `session_id` | string | ❌ | 会话 ID，为空时自动生成 |
| `target_position` | string | ❌ | 目标岗位 |

> 文件在内存中处理，不做持久化存储。

#### 响应

与 [5.1 文本粘贴方式](#51-文本粘贴方式-sse) 的 SSE 事件完全相同。

#### 错误场景

| 场景 | SSE error 消息 |
|------|----------------|
| 不支持的文件格式 | `"不支持的文件格式：.docx。请上传 PDF、图片或文本文件。"` |
| 文件过大 | `"文件过大（60.0MB），最大支持 50MB。"` |

---

## 6. JD 岗位分析

### 6.1 文本粘贴方式 (SSE)

### `POST /agent/jd-analysis`

用户直接粘贴 JD 岗位描述文本，后端提取结构化信息 → 流式生成岗位解读 + 简历写作建议。

JD 数据会自动存入 session state，后续同一 session 的简历分析将自动关联该 JD。

#### 请求

**Content-Type**: `application/json`

```json
{
  "jd_text": "职位：高级前端工程师\n要求：5年以上开发经验...",
  "question": "请分析该岗位的核心要求并给出简历写作建议",
  "session_id": ""
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `jd_text` | string | ✅ | JD 岗位描述文本 |
| `question` | string | ❌ | 分析要求，默认 `"请分析该岗位的核心要求并给出简历写作建议"` |
| `session_id` | string | ❌ | 会话 ID，为空时自动生成 |

#### 响应

**Content-Type**: `text/event-stream`

SSE 事件序列：

```
data: {"type": "extracted", "jd_data": {...}}       ← JD 提取完成
data: {"type": "status", "content": "..."}           ← 状态提示（可多次）
data: {"type": "token", "content": "..."}            ← 增量文本（可多次）
data: {"type": "done", ...}                          ← 完成
data: {"type": "error", ...}                         ← 错误
```

**`extracted` 事件**：

```json
{
  "type": "extracted",
  "jd_data": {
    "position": "高级前端工程师",
    "company": "某科技公司",
    "requirements": [...],
    "responsibilities": [...],
    "skills": [...]
  }
}
```

**`done` 事件**：

```json
{
  "type": "done",
  "session_id": "a1b2c3d4e5f6",
  "answer": "## 岗位分析报告\n\n### 核心要求\n...",
  "jd_data": {"position": "高级前端工程师", "skills": [...]}
}
```

| 字段 | 说明 |
|------|------|
| `session_id` | 会话 ID |
| `answer` | 完整分析报告（Markdown） |
| `jd_data` | 结构化 JD 信息（不含 `raw_text`） |

---

### 6.2 文件上传方式 (SSE)

### `POST /agent/jd-upload`

上传 JD 文件（PDF/图片/文本），后端自动提取内容后执行 JD 分析。

#### 请求

**Content-Type**: `multipart/form-data`

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `file` | file | ✅ | JD 文件（`.pdf`/`.png`/`.jpg`/`.jpeg`/`.txt`/`.md`） |
| `question` | string | ❌ | 分析要求，默认 `"请分析该岗位的核心要求并给出简历写作建议"` |
| `session_id` | string | ❌ | 会话 ID，为空时自动生成 |

#### 响应

与 [6.1 文本粘贴方式](#61-文本粘贴方式-sse) 的 SSE 事件完全相同。

#### 错误场景

| 场景 | SSE error 消息 |
|------|----------------|
| 不支持的文件格式 | `"不支持的文件格式：.docx。请上传 PDF、图片或文本文件。"` |
| 文件过大 | `"文件过大（60.0MB），最大支持 50MB。"` |
| 图片文本提取失败 | `"图片 JD 文本提取失败，请尝试粘贴 JD 文本或上传 PDF/文本文件。"` |
| PDF 文件提取失败 | `"文件 JD 文本提取失败，请尝试粘贴 JD 文本或上传其他格式文件。"` |

---

## 7. 会话管理

### 7.1 获取会话信息

### `GET /agent/session/{session_id}`

通过 checkpointer 查询会话的 thread 快照。

#### 路径参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `session_id` | string | 会话 ID |

#### 响应

```json
{
  "session_id": "a1b2c3d4e5f6",
  "message_count": 5
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `session_id` | string | 会话 ID |
| `message_count` | int | 该会话中的消息数量 |

---

### 7.2 清空会话

### `DELETE /agent/session/{session_id}`

清除指定会话的 checkpoint 数据。使用 MemorySaver 时，直接删除内存中的 thread 存储。

#### 路径参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `session_id` | string | 会话 ID |

#### 响应

```json
{
  "status": "ok",
  "session_id": "a1b2c3d4e5f6"
}
```

> ⚠️ 注意：MemorySaver 的清空操作是内存级别的，服务重启后数据本身也会丢失。使用 PostgresSaver 等持久化 checkpointer 时，此接口仅清除内存缓存，不删除数据库记录。

---

## 8. 调试接口

> ⚠️ 调试接口仅在 `debug_mode=True` 时可用，否则返回 HTTP 404。

### 8.1 运行时状态快照

### `GET /debug/runtime`

返回服务运行时状态，包括 checkpointer 后端、缓存后端、活跃会话列表等。

#### 请求

无参数。

#### 响应

```json
{
  "checkpointer": {
    "backend": "MemorySaver",
    "thread_count": 3
  },
  "expert_cache": {
    "backend": "state_checkpointer"
  },
  "threads": [
    {
      "thread_id": "a1b2c3d4e5f6",
      "message_count": 5,
      "has_jd_data": true,
      "has_resume_data": true,
      "expert_cache_entries": 2
    }
  ],
  "config": {
    "app_env": "development",
    "log_level": "INFO",
    "llm_model": "glm-4-flash",
    "debug_mode": true,
    "checkpoint_db_url_set": false,
    "expert_cache_db_url_set": false
  }
}
```

---

### 8.2 会话状态详情

### `GET /debug/session/{session_id}`

返回单个会话的完整 state dump，大字段已截断以避免返回超大 JSON。

#### 路径参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `session_id` | string | 会话 ID |

#### 响应

```json
{
  "session_id": "a1b2c3d4e5f6",
  "state": {
    "message_count": 5,
    "recent_messages": [
      {"type": "human", "content_preview": "我的简历和这个岗位匹配吗..."},
      {"type": "ai", "content_preview": "根据您的简历和岗位要求的对..."}
    ],
    "expert_cache": {
      "resume_expert": [
        {
          "cache_key": "a1b2c3d4..e5f6",
          "hit_count": 2,
          "created_at": "2026-04-09T10:30:00",
          "last_hit_at": "2026-04-09T11:00:00",
          "backend": "state_checkpointer"
        }
      ]
    },
    "resume_data": {"name": "str", "skills": "list", "experience": "list"},
    "jd_data": {"position": "str", "requirements": "list", "skills": "list"},
    "final_answer": "...",
    "_checkpoint": {
      "next": [],
      "created_at": "2026-04-09T10:30:00+00:00",
      "parent_config": "None"
    }
  }
}
```

> `resume_data` 和 `jd_data` 在 debug 接口中只保留 key 名和类型名，不返回完整内容。

#### 错误响应

| HTTP 状态码 | 场景 |
|-------------|------|
| 404 | 会话不存在 |
| 503 | Agent 图未初始化 |

---

## 附录 A：SSE 事件协议

### 事件类型汇总

| 事件类型 | 出现的接口 | 说明 |
|----------|-----------|------|
| `route` | chat/stream | 路由决策结果 |
| `planning` | chat/stream | 调度规划详情 |
| `agent_start` | chat/stream | Agent 开始执行 |
| `agent_result` | chat/stream | Agent 执行完成 |
| `agent_cache_hit` | chat/stream | Expert 缓存命中 |
| `extracted` | chat/stream, resume-analysis, resume-upload, jd-analysis, jd-upload | 结构化数据提取完成 |
| `sources` | chat/stream, resume-analysis, resume-upload | 检索来源 |
| `status` | chat/stream, resume-analysis, resume-upload, jd-analysis, jd-upload | 状态文本（人类可读） |
| `token` | 所有 SSE 接口 | 增量文本片段 |
| `done` | 所有 SSE 接口 | 整体完成 |
| `error` | 所有 SSE 接口 | 错误（终止流） |

### 第三方集成建议

1. **消费 token 事件**：逐个 `token` 事件的 `content` 字段拼接可得到完整回答
2. **获取完整结果**：直接使用 `done` 事件的 `answer` 字段，无需自行拼接
3. **结构化数据**：`extracted` 事件提供的 `resume_data`/`jd_data` 可用于业务逻辑判断
4. **进度展示**：`status` 事件提供人类可读的进度文案（如"正在评估简历"、"正在检索相关岗位"）
5. **来源展示**：`sources` 事件提供的引用信息可展示回答依据
6. **超时处理**：建议 SSE 客户端设置合理的超时时间（分析任务可能耗时 30-120 秒）

### SSE 客户端示例

**Python (httpx-sse)**：

```python
import httpx
from httpx_sse import connect_sse

url = "http://localhost:8000/agent/chat/stream"
payload = {"question": "什么是 RAG？", "session_id": ""}

with httpx.Client() as client:
    with connect_sse("POST", url, json=payload, client=client) as event_source:
        for sse_event in event_source.iter_sse():
            data = sse_event.json()
            if data["type"] == "token":
                print(data["content"], end="", flush=True)
            elif data["type"] == "done":
                print(f"\n完成: session={data['session_id']}")
            elif data["type"] == "error":
                print(f"\n错误: {data['message']}")
```

**JavaScript (EventSource 不支持 POST，需用 fetch)**：

```javascript
const response = await fetch('/agent/chat/stream', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ question: '什么是 RAG？', session_id: '' })
});

const reader = response.body.getReader();
const decoder = new TextDecoder();
let buffer = '';

while (true) {
  const { done, value } = await reader.read();
  if (done) break;

  buffer += decoder.decode(value, { stream: true });
  const lines = buffer.split('\n\n');
  buffer = lines.pop(); // 保留不完整的行

  for (const line of lines) {
    if (line.startsWith('data: ')) {
      const data = JSON.parse(line.slice(6));
      if (data.type === 'token') {
        process.stdout.write(data.content);
      } else if (data.type === 'done') {
        console.log('\n完成:', data.session_id);
      } else if (data.type === 'error') {
        console.error('错误:', data.message);
      }
    }
  }
}
```

**cURL**：

```bash
curl -N -X POST http://localhost:8000/agent/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "什么是RAG？", "session_id": ""}'
```

---

## 附录 B：枚举值说明

### `route_type` — 路由类型

| 值 | 说明 |
|----|------|
| `direct` | 直接回答（无需检索） |
| `retrieve` | 需要检索知识库 |

### `task_type` — 任务类型

| 值 | 说明 |
|----|------|
| `qa` | 通用问答 |
| `resume_analysis` | 简历分析 |
| `jd_analysis` | JD 岗位分析 |

### `source.type` — 来源类型

| 值 | 说明 |
|----|------|
| `kb` | 知识库检索 |
| `web` | Web 搜索 |

### Agent 名称

| 值 | 说明 |
|----|------|
| `qa_flow` | QA 对话专家 |
| `resume_expert` | 简历分析专家 |
| `jd_expert` | JD 分析专家 |

### Checkpointer 后端

| 值 | 说明 |
|----|------|
| `MemorySaver` | 内存存储（默认，重启丢失） |
| `SqliteSaver` | SQLite 持久化 |
| `PostgresSaver` | PostgreSQL 持久化 |

### Expert Cache 后端

| 值 | 说明 |
|----|------|
| `state_checkpointer` | 使用 checkpointer 存储（默认） |
| `postgres` | 独立 PostgreSQL 表存储 |

---

## 附录 C：错误码参考

### HTTP 错误码

| 状态码 | 说明 | 典型场景 |
|--------|------|----------|
| 400 | 请求参数错误 | 不支持的文件格式、文件过大、必填字段缺失 |
| 404 | 资源不存在 | 会话未找到、debug 接口未启用 |
| 500 | 服务内部错误 | LLM 调用失败、文件处理异常 |
| 503 | 服务未就绪 | Agent/RAG/向量存储未初始化 |

### SSE error 事件

所有 SSE 流式接口在发生错误时，会发送一个 `error` 事件并终止流：

```json
{"type": "error", "message": "错误描述文本"}
```

常见错误消息：

| 错误消息 | 场景 |
|----------|------|
| `"Agent 服务未初始化，请检查服务启动日志。"` | Agent 图未成功加载 |
| `"简历分析子图未初始化"` | 简历分析模块加载失败 |
| `"JD 分析子图未初始化"` | JD 分析模块加载失败 |
| `"请提供简历文本内容。"` | resume_text 为空 |
| `"请提供 JD 岗位描述文本内容。"` | jd_text 为空 |
| `"不支持的文件格式：{ext}"` | 上传文件格式不在白名单 |
| `"文件过大（{size}MB），最大支持 {max}MB。"` | 超过文件大小限制 |
| `"图片 JD 文本提取失败"` | 图片 OCR 提取失败 |
| `"文件 JD 文本提取失败"` | PDF 文本提取失败 |
