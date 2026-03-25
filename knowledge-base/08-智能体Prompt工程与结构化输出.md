# 智能体 Prompt 工程与结构化输出

## 一、Prompt 在 Agent 中的角色

在 LangGraph Agent 中，Prompt 是节点逻辑的**指令核心**。每个节点通过精心设计的 Prompt 驱动 LLM 完成特定任务：路由决策、内容生成、结构化提取等。

### Agent 中 Prompt 的分类

| 类型 | 作用 | 设计要点 |
|------|------|----------|
| **路由 Prompt** | 意图识别 + 任务分类 | 输出结构化 JSON，低温度 |
| **生成 Prompt** | 基于上下文生成回答 | 控制输出风格、引用要求 |
| **提取 Prompt** | 从非结构化文本中提取结构化数据 | 严格 JSON Schema，零幻觉 |
| **分析 Prompt** | 综合分析 + 评估报告 | 评分标准明确，输出格式模板 |

## 二、结构化输出

### 2.1 JSON 输出策略

**方案一：System Prompt 约束 + 后处理解析**

```python
ROUTER_PROMPT = """你是一个意图路由器。根据用户输入，判断应该使用哪种处理方式。

你必须输出以下 JSON 格式（不要输出任何其他内容）：
{
  "reasoning": "你的分析理由",
  "route_type": "retrieve | web | direct",
  "task_type": "qa | resume_analysis"
}

判断标准：
- 包含简历相关关键词（简历、经验、项目经历）→ task_type: "resume_analysis"
- 涉及专业知识、已上传的文档 → route_type: "retrieve"
- 涉及时效性信息、新闻 → route_type: "web"
- 打招呼、闲聊、自我介绍 → route_type: "direct"
"""
```

**方案二：工具调用 / 函数调用**

```python
from langchain_core.tools import tool

@tool
def route_decision(reasoning: str, route_type: str, task_type: str) -> str:
    """根据用户意图进行路由决策"""
    return f"route={route_type}, task={task_type}"

llm.bind_tools([route_decision]).invoke(messages)
```

**方案三：Pydantic OutputParser**

```python
from pydantic import BaseModel
from langchain_core.output_parsers import PydanticOutputParser

class RouteDecision(BaseModel):
    reasoning: str
    route_type: Literal["retrieve", "web", "direct"]
    task_type: Literal["qa", "resume_analysis"]

parser = PydanticOutputParser(pydantic_object=RouteDecision)
prompt = f"{ROUTER_PROMPT}\n{parser.get_format_instructions()}"
```

### 2.2 结构化提取 Prompt

```python
RESUME_EXTRACT_PROMPT = """你是一个专业的简历解析器。请从以下简历内容中提取结构化信息。

你必须严格按以下 JSON Schema 输出：
{
  "name": "姓名",
  "phone": "联系电话",
  "email": "邮箱地址",
  "education": [
    {"school": "学校", "degree": "学位", "major": "专业", "period": "时间段"}
  ],
  "experience": [
    {"company": "公司", "position": "职位", "period": "时间段", "description": "工作描述"}
  ],
  "skills": ["技能1", "技能2"],
  "projects": [
    {"name": "项目名", "role": "角色", "tech_stack": ["技术栈"], "description": "描述"}
  ],
  "summary": "个人简介摘要（50字以内）"
}

规则：
1. 所有字段都必须存在，缺失字段用空字符串或空数组
2. 不要编造简历中没有的信息
3. 技能列表提取技术关键词，而非描述性文字
4. experience.description 要简洁，每个条目不超过 100 字

简历内容：
{resume_text}
"""
```

### 2.3 JSON 解析容错

```python
import json
import re

def parse_json_response(text: str) -> dict | None:
    """健壮的 JSON 解析，兼容 markdown 代码块"""
    # 尝试直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # 尝试提取 markdown 代码块中的 JSON
    match = re.search(r'```(?:json)?\s*\n(.*?)\n```', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    
    # 尝试提取第一个 { } 块
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    
    return None
```

## 三、路由 Prompt 设计

### 3.1 路由器核心原则

| 原则 | 说明 |
|------|------|
| **互斥分类** | 分类之间不应有重叠 |
| **完整覆盖** | 所有可能输入都应被分类 |
| **Fallback** | 不确定时选择最安全的路径 |
| **温度参数** | 使用 temperature=0 保证确定性 |

### 3.2 多级路由示例

```python
MULTI_LEVEL_ROUTER = """第一级：判断输入类型
- 文本对话 → level_1: "text"
- 简历分析 → level_1: "resume"
- 图片理解 → level_1: "image"

第二级：对于 text 类型，判断回答方式
- 需要知识库 → level_2: "retrieve"
- 需要网络搜索 → level_2: "web"
- 直接回答 → level_2: "direct"

输出格式：
{"level_1": "...", "level_2": "...", "reasoning": "..."}
"""
```

### 3.3 路由降级策略

```python
def route_with_fallback(state):
    try:
        decision = llm.generate(ROUTER_PROMPT, temperature=0)
        parsed = parse_json_response(decision)
        if parsed and parsed.get("route_type") in VALID_ROUTES:
            return parsed
    except Exception:
        pass
    
    # Fallback: 检查 web search 可用性
    if not web_search_available:
        return {"route_type": "retrieve", "task_type": "qa"}
    
    return {"route_type": "direct", "task_type": "qa"}
```

## 四、生成 Prompt 设计

### 4.1 RAG 生成 Prompt

```python
RAG_GENERATE_PROMPT = """你是一个专业的问答助手。请基于以下参考内容回答用户问题。

规则：
1. **优先使用参考内容**：回答应主要基于提供的参考内容
2. **标注来源**：使用【来源N】标注信息出处
3. **诚实回答**：如果参考内容不足以回答问题，明确告知用户
4. **不要编造**：绝不编造参考内容中没有的信息
5. **结构化输出**：使用列表、表格等格式使回答更清晰

参考内容：
{context}

用户问题：{question}

请回答："""
```

### 4.2 分析报告 Prompt

```python
ANALYSIS_REPORT_PROMPT = """你是一个专业的{domain}分析师。请基于以下信息生成分析报告。

## 报告结构要求

### 1. 总体评价（评分 + 一句话总结）
评分标准：90+ 优秀 | 75-89 良好 | 60-74 合格 | <60 待改进

### 2. 关键发现
用表格展示，包含维度、评分、说明三列

### 3. 优势亮点
列出 3-5 个核心优势，每条附具体证据

### 4. 待改进项
按优先级排列，给出具体改进建议

### 5. 总结建议
3-5 条可执行的行动建议

参考数据：
{context}

分析对象：
{target}

要求：{question}
"""
```

### 4.3 Web 搜索生成 Prompt

```python
WEB_GENERATE_PROMPT = """你是一个信息整合助手。请基于以下网络搜索结果回答用户问题。

注意：
1. 综合多个搜索结果，给出全面回答
2. 标注信息来源 URL
3. 注意信息的时效性，提醒用户可能已变化
4. 如搜索结果不足，可结合自身知识补充，但要标注

搜索结果：
{context}

用户问题：{question}
"""
```

## 五、Prompt 模板管理

### 5.1 集中管理

```python
# prompts.py

class Prompts:
    ROUTER = "..."
    RAG_GENERATE = "..."
    WEB_GENERATE = "..."
    RESUME_EXTRACT = "..."
    ANALYSIS_REPORT = "..."

# 使用时通过变量引用，方便统一修改和维护
```

### 5.2 模板变量

使用 Python f-string 或 `str.format()` 管理动态变量。

```python
def fill_template(template: str, **kwargs) -> str:
    """填充 Prompt 模板"""
    try:
        return template.format(**kwargs)
    except KeyError as e:
        # 缺失变量时用空字符串替代
        kwargs.setdefault(e.args[0], "")
        return template.format(**kwargs)
```

### 5.3 Prompt 版本管理

```python
# 通过配置切换 Prompt 版本
ROUTER_PROMPT_V2 = """...优化后的路由 prompt..."""

def get_router_prompt(version: str = "v1"):
    prompts = {"v1": ROUTER_PROMPT, "v2": ROUTER_PROMPT_V2}
    return prompts.get(version, ROUTER_PROMPT)
```

## 六、Prompt 调优技巧

### 6.1 Few-Shot 示例

```python
ROUTER_WITH_EXAMPLES = """...路由说明...

示例：
用户输入："什么是微服务架构？"
{"reasoning": "涉及技术知识，应检索知识库", "route_type": "retrieve", "task_type": "qa"}

用户输入："今天天气怎么样？"
{"reasoning": "时效性信息，需要网络搜索", "route_type": "web", "task_type": "qa"}

用户输入："你好，你是谁？"
{"reasoning": "闲聊问候", "route_type": "direct", "task_type": "qa"}

现在请判断：
用户输入："{question}"
"""
```

### 6.2 Chain-of-Thought

```python
COT_ROUTER = """请逐步分析用户意图：

1. 用户想做什么？
2. 这个问题需要什么类型的信息？
3. 知识库中是否可能包含相关信息？
4. 是否需要最新的网络信息？

基于以上分析，输出路由决策 JSON。
"""
```

### 6.3 常见陷阱

| 陷阱 | 说明 | 解决方案 |
|------|------|----------|
| **指令过多** | Prompt 太长导致模型忽略部分指令 | 精简核心指令，非核心放入 examples |
| **格式不一致** | 模型输出格式偶尔偏离 | 使用 JSON Schema + 后处理容错 |
| **角色模糊** | 多角色定义导致混乱 | 每个节点一个明确角色 |
| **上下文溢出** | 参考内容过长截断关键信息 | 设置合理的 max_chars，优先保留高分片段 |
