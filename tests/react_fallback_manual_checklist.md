# ReAct Fallback Manual Checklist

这份清单用于联调当前混合架构中的 ReAct fallback 扩展路径：

- 标准任务：应优先走固定 Expert
- 非标准任务：应进入 `react_fallback`
- fallback 内部：应看到 tool 选择、tool cache、必要时 handoff 回标准 Expert

## 观测点

联调时优先看这几类日志：

- `agent.supervisor`
- `agent.react_fallback`
- `api.agent`
- `obs.request`

前端优先看这些状态：

- `ReAct 兜底`
- `工具运行中`
- `已调用工具`
- `命中工具缓存`
- `识别为标准任务，切换到...`

## Case 1: 普通 QA 不应进入 fallback

问题：

`rag 是如何实现的`

预期：

- `supervisor` 判为 `task=qa`
- 执行 `qa_flow`
- 不出现 `agent.react_fallback | ReAct fallback 开始`

## Case 2: 组合型问题进入 fallback

问题：

`结合知识库，你怎么看待 RAG 的工程实现`

预期：

- `supervisor` 判为 `task=react_fallback`
- 出现 `ReAct fallback 开始`
- 至少调用一个：
  - `search_kb`
  - `filter_kb_by_type`
  - `list_documents`
- 最终有回答，不直接报错

## Case 3: list_sources 工具链

问题：

`先看看当前知识库里都有哪些来源文件`

预期：

- fallback 调用 `list_sources`
- 日志出现：
  - `tool 开始: ... list_sources`
  - `tool 完成: ... list_sources`
- 回答应提到来源文件列表

## Case 4: list_documents + filter_kb_by_type

问题：

`只看 interview_kb 里和 rag 相关的内容，帮我总结一下`

预期：

- fallback 先调用 `list_documents` 或 `filter_kb_by_type`
- 如果命中文档/检索结果，应生成简短总结
- 不应该无休止地继续 search

## Case 5: 工具缓存命中

步骤：

1. 连续两次问：
   `先看看当前知识库里都有哪些来源文件`

预期：

- 第二次出现：
  - `tool 命中缓存`
  - 前端出现 `命中工具缓存`

## Case 6: JD 提取后 handoff 到 jd_expert

问题：

`先帮我提取一下这段岗位描述，再详细分析岗位重点：...`

预期：

- fallback 先调用 `extract_jd`
- 日志出现：
  - `ReAct fallback handoff`
  - `handoff_agent=jd_expert`
- 后续进入标准 `jd_expert`

## Case 7: 简历提取后 handoff 到 resume_expert

问题：

`这是我的简历，先识别结构，再给我完整优化建议：...`

预期：

- fallback 先调用 `extract_resume`
- 日志出现：
  - `ReAct fallback handoff`
  - `handoff_agent=resume_expert`
- 后续进入标准 `resume_expert`

## Case 8: JD + 简历匹配走 match_resume_jd

前置：

- 当前 session 已有 `jd_data`
- 当前 session 已有 `resume_data`

问题：

`不用完整报告，直接说我和这个岗位最核心的差距`

预期：

- fallback 直接或在极少工具后调用 `match_resume_jd`
- 日志出现：
  - `tool 开始: ... match_resume_jd`
  - `ReAct fallback 收口: reason=match_resume_jd_complete`

## Case 9: 非法工具组合被拦截

目标：

验证 fallback 不会乱跳。

重点观测：

- 日志出现：
  - `ReAct fallback 拦截不允许的工具组合`
- 然后直接改为 `generate_report` 收口

如果前端看到：

- `工具组合受限，改为直接生成结果`

说明拦截策略生效。

## Case 10: 网络抖动时的降级

问题：

任意容易进入 fallback 的组合型问题。

预期：

- 如果外部 LLM 连接失败：
  - 不应直接把整条 graph 打炸
- 日志出现：
  - `tool 规划失败，降级到普通回答`
  - 或 `返回安全降级文本`

## Case 11: compact_faiss 工具

问题：

`帮我压缩一下当前索引，清理掉已删除文档的失效向量`

预期：

- fallback 调用 `compact_faiss`
- 日志出现：
  - `tool 完成: ... compact_faiss`
- 回答应带：
  - `before`
  - `after`
  - `removed`

## 回归结论模板

每次联调后建议记录：

- Case 编号
- 是否进入 fallback
- 实际 tool 链
- 是否命中 tool cache
- 是否 handoff
- 是否得到最终答案
- 是否出现异常
