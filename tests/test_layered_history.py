"""
分层裁剪（方案 A）单元测试。

覆盖：
1. token 估算工具
2. 早期消息摘要
3. 分层消息构建
4. generate.py _build_llm_messages 分层裁剪
5. session_manager 分层裁剪
"""
from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.agent.utils.history_utils import (
    build_layered_messages,
    estimate_message_tokens,
    estimate_messages_tokens,
    estimate_tokens,
    summarize_early_messages,
)


# ---- token 估算 ----

class TestEstimateTokens:
    def test_empty_string(self):
        assert estimate_tokens("") == 0

    def test_short_text(self):
        result = estimate_tokens("hello")
        assert result >= 2

    def test_chinese_text(self):
        result = estimate_tokens("你好世界")
        assert result >= 2

    def test_long_text(self):
        result = estimate_tokens("a" * 1000)
        assert result >= 400


class TestEstimateMessageTokens:
    def test_human_message(self):
        msg = HumanMessage(content="测试")
        assert estimate_message_tokens(msg) > 0

    def test_ai_message(self):
        msg = AIMessage(content="回复")
        assert estimate_message_tokens(msg) > 0

    def test_message_overhead(self):
        """消息估算应包含角色开销（约 4 token）"""
        msg = HumanMessage(content="x")
        assert estimate_message_tokens(msg) > estimate_tokens("x")


class TestEstimateMessagesTokens:
    def test_multiple_messages(self):
        msgs = [HumanMessage(content="你好"), AIMessage(content="你好！")]
        total = estimate_messages_tokens(msgs)
        assert total > 0

    def test_empty_list(self):
        assert estimate_messages_tokens([]) == 0


# ---- 早期消息摘要 ----

class TestSummarizeEarlyMessages:
    def test_basic_summary(self):
        msgs = [
            HumanMessage(content="帮我分析简历"),
            AIMessage(content="好的，请提供简历内容"),
        ]
        summary = summarize_early_messages(msgs)
        assert "用户" in summary
        assert "助手" in summary

    def test_route_decision_skipped(self):
        msgs = [
            HumanMessage(content="你好"),
            AIMessage(content="[路由决策] route=direct"),
            AIMessage(content="你好！"),
        ]
        summary = summarize_early_messages(msgs)
        assert "[路由决策]" not in summary

    def test_truncation(self):
        long_content = "很长的内容" * 100
        msgs = [HumanMessage(content=long_content)]
        summary = summarize_early_messages(msgs, max_chars_per_msg=50)
        assert "…" in summary
        # 每条消息截断后不应超过 max_chars + 1(…)
        user_line = [l for l in summary.split("\n") if l.startswith("用户")][0]
        assert len(user_line) <= 70  # "用户: " + 50字符 + "…"

    def test_empty_messages(self):
        assert summarize_early_messages([]) == ""

    def test_custom_max_chars(self):
        msgs = [HumanMessage(content="abcdefghij" * 10)]
        summary = summarize_early_messages(msgs, max_chars_per_msg=20)
        assert "…" in summary


# ---- 分层消息构建 ----

class TestBuildLayeredMessages:
    def test_basic_layered(self):
        msgs = [HumanMessage(content=f"问题{i}") for i in range(15)]
        msgs += [AIMessage(content=f"回答{i}") for i in range(15)]
        complete, summary = build_layered_messages(msgs, recent_count=10)
        assert len(complete) == 10
        assert summary

    def test_few_messages(self):
        msgs = [HumanMessage(content="只有一条")]
        complete, summary = build_layered_messages(msgs, recent_count=10)
        assert len(complete) == 1
        assert summary == ""

    def test_empty_messages(self):
        complete, summary = build_layered_messages([], recent_count=10)
        assert complete == []
        assert summary == ""

    def test_token_budget(self):
        msgs = [HumanMessage(content="很长的消息内容" * 50) for _ in range(5)]
        msgs += [AIMessage(content="回复" * 10) for _ in range(5)]
        _, summary = build_layered_messages(msgs, recent_count=4, token_budget=200)
        # 允许估算误差，但不应超过太多
        assert estimate_tokens(summary) <= 350

    def test_no_token_budget(self):
        msgs = [HumanMessage(content="问题" * 100) for _ in range(10)]
        _, summary = build_layered_messages(msgs, recent_count=2, token_budget=0)
        # token_budget=0 表示不限制，摘要应包含所有早期消息
        assert len(summary) > 100

    def test_route_decision_filtered_in_complete(self):
        msgs = [
            HumanMessage(content="分析简历"),
            AIMessage(content="[路由决策] route=resume"),
            AIMessage(content="好的，我来分析"),
        ]
        complete, _ = build_layered_messages(msgs, recent_count=3)
        route_msgs = [m for m in complete if "路由决策" in m.get("content", "")]
        assert len(route_msgs) == 0


# ---- generate.py _build_llm_messages ----

class TestBuildLlmMessages:
    @pytest.fixture(autouse=True)
    def setup_config(self):
        from app.agent.nodes.generate import set_layered_config
        set_layered_config(recent_count=5, summary_max_chars=120, summary_token_budget=800)

    def _make_state(self, messages, **kwargs):
        from app.agent.state import AgentState
        state = {
            "messages": messages,
            "working_context": "",
            "route_type": "direct",
            "conversation_summary": "",
        }
        state.update(kwargs)
        return state

    def test_layered_with_many_messages(self):
        from app.agent.nodes.generate import _build_llm_messages

        msgs = []
        for i in range(12):
            msgs.append(HumanMessage(content=f"第{i+1}个问题"))
            msgs.append(AIMessage(content=f"第{i+1}个回答"))

        llm_msgs, merged_summary = _build_llm_messages(self._make_state(msgs))
        assert llm_msgs[0]["role"] == "system"
        assert "早期对话摘要" in llm_msgs[0]["content"]
        assert merged_summary  # 应返回非空摘要

    def test_no_summary_for_few_messages(self):
        from app.agent.nodes.generate import _build_llm_messages

        msgs = [HumanMessage(content="你好"), AIMessage(content="你好！")]
        llm_msgs, merged_summary = _build_llm_messages(self._make_state(msgs))
        assert "早期对话摘要" not in llm_msgs[0]["content"]
        assert merged_summary == ""  # 消息少，无摘要

    def test_conversation_summary_merge(self):
        from app.agent.nodes.generate import _build_llm_messages

        msgs = [HumanMessage(content=f"问题{i}") for i in range(8)]
        state = self._make_state(
            msgs,
            conversation_summary="用户: 之前讨论了简历优化\n助手: 建议增加量化指标",
        )
        llm_msgs, merged_summary = _build_llm_messages(state)
        assert "更早" in llm_msgs[0]["content"]
        assert merged_summary  # 合并后应有摘要

    def test_conversation_summary_preserved_when_no_new(self):
        """消息少于 recent_count 时，已有摘要应保留"""
        from app.agent.nodes.generate import _build_llm_messages

        msgs = [HumanMessage(content="你好")]
        state = self._make_state(
            msgs,
            conversation_summary="用户: 之前讨论了简历优化\n助手: 建议增加量化指标",
        )
        llm_msgs, merged_summary = _build_llm_messages(state)
        assert merged_summary == "用户: 之前讨论了简历优化\n助手: 建议增加量化指标"

    def test_working_context_injection(self):
        from app.agent.nodes.generate import _build_llm_messages

        msgs = [
            HumanMessage(content="什么是RAG"),
            AIMessage(content="RAG是检索增强生成"),
            HumanMessage(content="详细说说"),
        ]
        state = self._make_state(msgs, working_context="RAG是一种技术...")
        llm_msgs, _ = _build_llm_messages(state)
        last_user = [m for m in llm_msgs if m["role"] == "user"][-1]
        assert "参考内容" in last_user["content"]

    def test_route_decision_filtered(self):
        from app.agent.nodes.generate import _build_llm_messages

        msgs = [
            HumanMessage(content="分析简历"),
            AIMessage(content="[路由决策] route=resume"),
            AIMessage(content="好的，分析简历"),
        ]
        llm_msgs, _ = _build_llm_messages(self._make_state(msgs))
        route_msgs = [m for m in llm_msgs if "路由决策" in m.get("content", "")]
        assert len(route_msgs) == 0

    def test_web_route_system_prompt(self):
        from app.agent.nodes.generate import _build_llm_messages

        msgs = [HumanMessage(content="搜索")]
        state = self._make_state(msgs, route_type="web")
        llm_msgs, _ = _build_llm_messages(state)
        # WEB_AGENT_SYSTEM_PROMPT 内容应包含 web 相关指令
        assert len(llm_msgs[0]["content"]) > 50


# ---- session_manager 分层裁剪 ----

class TestSessionManagerLayered:
    def test_trim_history(self):
        from app.agent.session_manager import SessionManager

        sm = SessionManager(recent_count=4, summary_max_chars=100, summary_token_budget=400)
        for i in range(10):
            sm.add_message("s1", "user", f"第{i+1}个问题")
            sm.add_message("s1", "assistant", f"第{i+1}个回答")

        session = sm.get_session("s1")
        assert len(session["messages"]) <= 4
        assert session.get("conversation_summary", "")

    def test_few_messages_no_trim(self):
        from app.agent.session_manager import SessionManager

        sm = SessionManager(recent_count=10)
        sm.add_message("s2", "user", "你好")
        sm.add_message("s2", "assistant", "你好！")
        session = sm.get_session("s2")
        assert len(session["messages"]) == 2

    def test_resume_data_preserved(self):
        from app.agent.session_manager import SessionManager

        sm = SessionManager(recent_count=4)
        sm.add_message("s3", "user", "分析简历")
        sm.set_resume_data("s3", {"name": "张三", "skills": ["Python"]})
        assert sm.get_resume_data("s3")["name"] == "张三"

    def test_conversation_summary_accumulated(self):
        from app.agent.session_manager import SessionManager

        sm = SessionManager(recent_count=3, summary_max_chars=80, summary_token_budget=400)
        for i in range(8):
            sm.add_message("s4", "user", f"第{i+1}个问题，关于AI")
            sm.add_message("s4", "assistant", f"第{i+1}个回答")
        summary = sm.get_conversation_summary("s4")
        assert summary

    def test_clear_session(self):
        from app.agent.session_manager import SessionManager

        sm = SessionManager()
        sm.add_message("s5", "user", "测试")
        sm.clear_session("s5")
        assert sm.get_session("s5") is None
