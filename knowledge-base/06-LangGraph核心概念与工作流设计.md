# LangGraph 核心概念与工作流设计

## 一、LangGraph 概述

LangGraph 是 LangChain 生态中用于构建**有状态、多步骤 AI 应用**的框架。与传统 Chain 不同，LangGraph 基于**图（Graph）**结构编排 LLM 调用，支持条件分支、循环、持久化等复杂拓扑。

### 核心优势

| 特性 | 说明 |
|------|------|
| **图拓扑** | 节点（Node）+ 边（Edge），支持任意有向图结构 |
| **状态管理** | 每个 State 定义图的"记忆"，跨节点传递 |
| **条件边** | 根据运行时状态动态选择下一个节点 |
| **持久化** | Checkpointer 支持 state 持久化，实现多轮对话 |
| **子图** | 图可嵌套，支持模块化设计 |
| **人机协作** | interrupt 机制支持人工审核后继续执行 |

## 二、核心组件

### 2.1 State（状态）

State 是图的核心数据结构，所有节点共享同一个 State 对象。

```python
from typing import TypedDict, Annotated
from langgraph.graph import add_messages

class AgentState(TypedDict, total=False):
    messages: Annotated[list, add_messages]  # 自动追加消息
    context: str                              # 自定义字段
    route_type: str                           # 路由决策
    final_answer: str                         # 最终输出
```

**关键点**：
- `TypedDict` 定义字段，`total=False` 允许字段缺失
- `Annotated[type, reducer]` 指定字段的合并策略（如 `add_messages` 自动追加）
- State 在节点间**不可变传递**，每个节点返回新的 State 片段

### 2.2 Node（节点）

节点是一个接收 State、返回 State 部分更新的函数。

```python
from langchain_core.messages import HumanMessage

def my_node(state: AgentState) -> dict:
    last_msg = state["messages"][-1].content
    # 处理逻辑...
    return {"context": "处理结果", "route_type": "retrieve"}
```

**要点**：
- 节点函数签名：`(state) -> dict`，返回值是 State 的**部分更新**
- 返回的 key 必须在 State 定义中存在
- 不需要返回所有字段，只返回变更的字段

### 2.3 Edge（边）

边定义节点间的连接关系。

| 边类型 | 说明 |
|--------|------|
| **普通边** | `graph.add_edge("node_a", "node_b")`，无条件跳转 |
| **条件边** | `graph.add_conditional_edges("router", route_fn, {"key": "node"})`，根据函数返回值路由 |
| **入口边** | `graph.add_edge(START, "first_node")`，定义起始节点 |
| **出口边** | `graph.add_edge("last_node", END)`，定义终止节点 |

### 2.4 START 和 END

```python
from langgraph.graph import START, END

graph.add_edge(START, "router")
graph.add_edge("generate", END)
```

`START` 不是真正的节点，而是图的入口标记。

## 三、状态图构建

### 3.1 StateGraph 基本模式

```python
from langgraph.graph import StateGraph

# 1. 创建图
graph = StateGraph(AgentState)

# 2. 添加节点
graph.add_node("router", router_node)
graph.add_node("retrieve", retrieve_node)
graph.add_node("generate", generate_node)

# 3. 添加边
graph.add_edge(START, "router")
graph.add_conditional_edges("router", route_function, {
    "retrieve": "retrieve",
    "direct": "generate"
})
graph.add_edge("retrieve", "generate")
graph.add_edge("generate", END)

# 4. 编译
app = graph.compile()
```

### 3.2 条件路由函数

```python
def route_function(state: AgentState) -> str:
    """根据 state 决定下一个节点"""
    if state.get("route_type") == "retrieve":
        return "retrieve"
    return "generate"
```

返回值必须是条件边映射表中的 key。

## 四、Checkpointer 与持久化

### 4.1 MemorySaver

```python
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)
```

### 4.2 Thread 对话

```python
# 通过 thread_id 隔离不同会话
config = {"configurable": {"thread_id": "session_123"}}

result = app.invoke(
    {"messages": [HumanMessage(content="你好")]},
    config=config
)
```

### 4.3 读取历史状态

```python
# LangGraph 1.1.x+
snapshot = checkpointer.get(config)
state = snapshot.channel_values  # 注意：不是 .values
```

**重要**：不同版本 API 不同：
- 1.1.x: `snapshot.channel_values`
- 1.0.x: `snapshot.values`

## 五、子图设计模式

### 5.1 基本嵌套

```python
# 子图
sub_graph = StateGraph(SubState)
sub_graph.add_node("extract", extract_node)
sub_graph.add_node("analyze", analyze_node)
sub_graph.add_edge("extract", "analyze")
sub_graph.add_edge("analyze", END)
sub_app = sub_graph.compile()

# 主图引用
main_graph.add_node("sub_task", sub_app)
main_graph.add_edge("router", "sub_task")
```

### 5.2 子图与主图的状态映射

子图可以使用不同的 State 类型。父节点负责将主图 State 的数据转换为子图输入。

```python
def prepare_sub_input(state: AgentState) -> dict:
    return {
        "resume_data": state["resume_data"],
        "context_sources": state["context_sources"]
    }

def merge_sub_output(state: AgentState, sub_result: dict) -> dict:
    return {"final_answer": sub_result["report"]}
```

## 六、常见工作流模式

### 6.1 Router-Worker 模式

```
START → Router → (条件路由)
  ├── KB检索 → Normalize → Generate → END
  ├── Web搜索 → Normalize → Generate → END
  └── 直接回答 → Generate → END
```

### 6.2 顺序处理链

```
START → Extract → Validate → Transform → Store → END
```

### 6.3 循环迭代模式

```
START → Generate → (质量检查)
  ├── 通过 → END
  └── 不通过 → Generate（循环改进）
```

### 6.4 人工审核模式

```python
def human_review_node(state):
    return interrupt("请审核以下内容")  # 暂停执行

# 审核通过后调用
app.invoke(Command(resume="approved"), config=config)
```

## 七、调试与监控

### 7.1 日志

在节点函数中添加 logger 输出，记录输入 state 和输出。

### 7.2 LangSmith 集成

```bash
export LANGCHAIN_TRACING_V2="true"
export LANGCHAIN_API_KEY="your_key"
```

LangSmith 提供节点级别的执行追踪和可视化。

### 7.3 常见问题

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| State 字段丢失 | 节点返回的 key 不在 State 定义中 | 检查 State TypedDict |
| 条件边死循环 | 路由函数未正确收敛 | 添加最大迭代计数 |
| 内存泄漏 | Checkpointer 无限增长 | 定期清理旧 thread |
| 编译错误 | 节点未连接到 END | 确保所有路径都到达 END |
