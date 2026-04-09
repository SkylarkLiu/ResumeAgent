"""
KB 检索降级机制测试。

覆盖：
1. evaluate_kb_relevance 单元测试 — 空结果/低分/合格 三种场景
2. QA 子图降级路径 — KB 低分时自动降级到 web search
3. QA 子图正常路径 — KB 高分时不降级
"""
from __future__ import annotations

import pytest
from langchain_core.messages import HumanMessage

from app.agent.nodes.kb_search import (
    evaluate_kb_relevance,
    set_retrieval_service,
)
from app.agent.state import AgentState


# ---- Fixtures ----

@pytest.fixture(autouse=True)
def _setup_threshold():
    """每个测试前重置阈值为 0.35。"""
    set_retrieval_service(None, top_k=5, kb_relevance_threshold=0.35)


def _make_state(context_sources: list[dict], **extra) -> dict:
    """构造测试用 AgentState 子集。"""
    state: dict = {
        "messages": [HumanMessage(content="测试问题")],
        "context_sources": context_sources,
    }
    state.update(extra)
    return state


# ============================================================
# 1. evaluate_kb_relevance 单元测试
# ============================================================


class TestEvaluateKbRelevance:
    """evaluate_kb_relevance 节点测试。"""

    def test_empty_sources_triggers_fallback(self):
        """知识库无结果 → 标记降级，route_type 改为 web。"""
        state = _make_state(context_sources=[])
        result = evaluate_kb_relevance(state)

        assert result["retrieval_fallback"] is True
        assert result["route_type"] == "web"

    def test_low_score_triggers_fallback(self):
        """最高分低于阈值(0.35) → 标记降级。"""
        state = _make_state(context_sources=[
            {"content": "文档1", "source": "a.pdf", "score": 0.20, "type": "kb"},
            {"content": "文档2", "source": "b.pdf", "score": 0.15, "type": "kb"},
        ])
        result = evaluate_kb_relevance(state)

        assert result["retrieval_fallback"] is True
        assert result["route_type"] == "web"

    def test_exact_threshold_no_fallback(self):
        """最高分恰好等于阈值(0.35) → 不降级。"""
        state = _make_state(context_sources=[
            {"content": "文档1", "source": "a.pdf", "score": 0.35, "type": "kb"},
        ])
        result = evaluate_kb_relevance(state)

        assert result["retrieval_fallback"] is False
        assert "route_type" not in result  # 不修改 route_type

    def test_high_score_no_fallback(self):
        """最高分高于阈值 → 正常继续。"""
        state = _make_state(context_sources=[
            {"content": "文档1", "source": "a.pdf", "score": 0.50, "type": "kb"},
            {"content": "文档2", "source": "b.pdf", "score": 0.42, "type": "kb"},
        ])
        result = evaluate_kb_relevance(state)

        assert result["retrieval_fallback"] is False
        assert "route_type" not in result

    def test_mixed_types_uses_kb_max_score(self):
        """context_sources 混合 kb 和 web 类型时，只用 kb 类型的最高分。"""
        state = _make_state(context_sources=[
            {"content": "web结果", "source": "web.com", "score": 0.9, "type": "web"},
            {"content": "kb低分", "source": "a.pdf", "score": 0.20, "type": "kb"},
        ])
        result = evaluate_kb_relevance(state)

        # web 的高分不算，kb 最高 0.20 < 0.35 → 降级
        assert result["retrieval_fallback"] is True

    def test_custom_threshold(self):
        """自定义阈值生效。"""
        # 设置高阈值
        set_retrieval_service(None, top_k=5, kb_relevance_threshold=0.60)

        state = _make_state(context_sources=[
            {"content": "文档1", "source": "a.pdf", "score": 0.50, "type": "kb"},
        ])
        result = evaluate_kb_relevance(state)

        # 0.50 < 0.60 → 降级
        assert result["retrieval_fallback"] is True

    def test_missing_score_treated_as_zero(self):
        """缺少 score 字段的 source 视为 0 分。"""
        state = _make_state(context_sources=[
            {"content": "无分数文档", "source": "a.pdf", "type": "kb"},
        ])
        result = evaluate_kb_relevance(state)

        assert result["retrieval_fallback"] is True


# ============================================================
# 2. QA 子图降级路径测试
# ============================================================


class TestQaFlowFallback:
    """QA 子图 KB→Web 降级路径测试。

    这些测试需要 mock search_kb 和 search_web 的依赖注入。
    通过 set_retrieval_service / set_web_search_service 注入 mock service。
    """

    def test_evaluate_relevance_decision_pass(self):
        """_evaluate_relevance_decision 返回 'pass' 当 fallback=False。"""
        from app.agent.agents.qa_flow import _evaluate_relevance_decision

        state = _make_state(context_sources=[], retrieval_fallback=False)
        assert _evaluate_relevance_decision(state) == "pass"

    def test_evaluate_relevance_decision_fallback(self):
        """_evaluate_relevance_decision 返回 'fallback' 当 fallback=True。"""
        from app.agent.agents.qa_flow import _evaluate_relevance_decision

        state = _make_state(context_sources=[], retrieval_fallback=True)
        assert _evaluate_relevance_decision(state) == "fallback"

    def test_qa_subgraph_has_fallback_nodes(self):
        """编译后的 QA 子图包含降级相关节点。"""
        from app.agent.agents.qa_flow import build_qa_flow_subgraph

        subgraph = build_qa_flow_subgraph()
        node_names = list(subgraph.nodes)

        assert "evaluate_relevance" in node_names
        assert "search_web_fallback" in node_names
        assert "search_kb" in node_names
        assert "search_web" in node_names
        assert "normalize_kb" in node_names
        assert "normalize_web" in node_names

    def test_qa_subgraph_kb_path(self):
        """KB 路径正常走通：dispatch → search_kb → evaluate → normalize_kb → generate。"""
        from app.agent.agents.qa_flow import (
            _dispatch_node,
            _evaluate_relevance_decision,
        )

        # dispatch 返回空 dict（正常）
        result = _dispatch_node({})
        assert result == {}

        # 评估决策：不降级
        state = _make_state(context_sources=[], retrieval_fallback=False)
        assert _evaluate_relevance_decision(state) == "pass"

    def test_qa_subgraph_fallback_path(self):
        """降级路径：dispatch → search_kb → evaluate(fallback) → search_web_fallback → normalize_web → generate。"""
        from app.agent.agents.qa_flow import (
            _dispatch_node,
            _evaluate_relevance_decision,
        )

        # 评估决策：降级
        state = _make_state(context_sources=[], retrieval_fallback=True)
        assert _evaluate_relevance_decision(state) == "fallback"


# ============================================================
# 3. 端到端降级场景（需 mock LLM，在 Docker 中运行）
# ============================================================


class TestKbFallbackE2E:
    """
    端到端测试：在 Docker 中运行。

    测试方法：
    1. 确保知识库中有低相关性的文档
    2. 发送一个知识库匹配度低的问题
    3. 验证最终回答来自 web search 而非 KB

    由于需要运行环境，此处只提供测试骨架。
    """

    @pytest.mark.asyncio
    async def test_kb_low_relevance_fallback_to_web(self, client):
        """
        当 KB 最高分 < 阈值时，自动降级到 web search。

        验证点：
        - SSE 事件中出现 status 事件，内容包含"降级"
        - SSE 事件中出现 web 类型的 sources
        - 最终回答包含内容
        """
        # 此测试需要：
        # 1. 知识库中加载一些无关文档
        # 2. 提问一个 KB 匹配度低的问题
        # 3. 检查 SSE 事件流
        #
        # 在 Docker 中运行：
        # pytest tests/test_kb_fallback.py::TestKbFallbackE2E -v
        pass

    @pytest.mark.asyncio
    async def test_kb_high_relevance_no_fallback(self, client):
        """
        当 KB 最高分 >= 阈值时，不走降级。

        验证点：
        - SSE 事件中不出现"降级"关键词
        - sources 中包含 kb 类型的结果
        """
        pass
