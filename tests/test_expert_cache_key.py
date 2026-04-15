"""
单元测试：expert_cache 缓存键应区分不同用户问题。

修复目标：追问"根据简历内容，帮我设计自我介绍模版"和
"如何将rocketmq接入简历中的格力百通项目"不应命中同一缓存，
因为 question_signature 都是 resume_followup:optimize，
但用户问题内容完全不同。
"""
from __future__ import annotations

from langchain_core.messages import HumanMessage

from app.agent.agents.expert_cache import (
    build_jd_expert_cache_key,
    build_resume_expert_cache_key,
    _latest_user_question,
    _question_hash,
)


def _make_state(question: str, **overrides) -> dict:
    """构造带用户问题的 state dict。"""
    state = {
        "messages": [HumanMessage(content=question)],
        "task_type": "resume_followup",
        "question_signature": "resume_followup:optimize",
        "response_mode": "followup_brief",
        "resume_data": {
            "name": "张三",
            "skills": ["Python", "RocketMQ"],
            "summary": "5年后端经验",
        },
        "jd_data": None,
    }
    state.update(overrides)
    return state


class TestLatestUserQuestion:
    def test_extracts_latest_human_message(self):
        state = _make_state("你好世界")
        assert _latest_user_question(state) == "你好世界"

    def test_picks_last_when_multiple(self):
        state = {
            "messages": [
                HumanMessage(content="第一问"),
                HumanMessage(content="第二问"),
            ],
        }
        assert _latest_user_question(state) == "第二问"

    def test_returns_empty_when_no_messages(self):
        assert _latest_user_question({}) == ""
        assert _latest_user_question({"messages": []}) == ""

    def test_skips_non_human_messages(self):
        from langchain_core.messages import AIMessage
        state = {
            "messages": [
                AIMessage(content="我是AI"),
                HumanMessage(content="我是人"),
            ],
        }
        assert _latest_user_question(state) == "我是人"


class TestQuestionHash:
    def test_deterministic(self):
        h1 = _question_hash("相同问题")
        h2 = _question_hash("相同问题")
        assert h1 == h2

    def test_different_questions_different_hashes(self):
        h1 = _question_hash("自我介绍模版")
        h2 = _question_hash("rocketmq接入格力百通")
        assert h1 != h2

    def test_empty_returns_empty(self):
        assert _question_hash("") == ""


class TestResumeExpertCacheKey:
    def test_different_questions_produce_different_keys(self):
        """核心断言：同一 question_signature 下，不同问题应产生不同缓存键。"""
        state1 = _make_state("根据简历内容，帮我设计自我介绍模版，控制在5分钟左右")
        state2 = _make_state("如何将rocketmq接入简历中的格力百通项目")

        key1 = build_resume_expert_cache_key(state1)
        key2 = build_resume_expert_cache_key(state2)

        assert key1 != key2, (
            "不同用户问题不应命中同一缓存！"
            f"key1={key1[:16]}.. key2={key2[:16]}.."
        )

    def test_same_question_produces_same_key(self):
        """同一问题应产生相同缓存键。"""
        state1 = _make_state("帮我优化简历的技能描述")
        state2 = _make_state("帮我优化简历的技能描述")

        key1 = build_resume_expert_cache_key(state1)
        key2 = build_resume_expert_cache_key(state2)

        assert key1 == key2

    def test_question_signature_alone_is_not_enough(self):
        """仅 question_signature 相同但问题不同，缓存键应不同。"""
        state1 = _make_state(
            "自我介绍模版",
            question_signature="resume_followup:optimize",
        )
        state2 = _make_state(
            "rocketmq接入格力百通",
            question_signature="resume_followup:optimize",
        )

        key1 = build_resume_expert_cache_key(state1)
        key2 = build_resume_expert_cache_key(state2)

        assert key1 != key2

    def test_same_data_same_question_same_key(self):
        """相同简历数据 + 相同问题 = 相同缓存键。"""
        resume = {"name": "李四", "skills": ["Java"], "summary": "3年"}
        state1 = _make_state("我的简历有什么亮点", resume_data=resume)
        state2 = _make_state("我的简历有什么亮点", resume_data=resume)

        assert build_resume_expert_cache_key(state1) == build_resume_expert_cache_key(state2)

    def test_different_resume_data_different_key(self):
        """不同简历数据应产生不同缓存键。"""
        state1 = _make_state("优化简历", resume_data={"name": "张三"})
        state2 = _make_state("优化简历", resume_data={"name": "李四"})

        assert build_resume_expert_cache_key(state1) != build_resume_expert_cache_key(state2)


class TestJdExpertCacheKey:
    def test_different_questions_produce_different_keys(self):
        state1 = {
            "messages": [HumanMessage(content="这个岗位的薪资范围？")],
            "task_type": "jd_followup",
            "question_signature": "jd_followup:detail",
            "response_mode": "followup_brief",
            "jd_data": {"position": "前端工程师", "skills_must": ["React"]},
            "resume_data": None,
        }
        state2 = {
            "messages": [HumanMessage(content="这个岗位需要多少年经验？")],
            "task_type": "jd_followup",
            "question_signature": "jd_followup:detail",
            "response_mode": "followup_brief",
            "jd_data": {"position": "前端工程师", "skills_must": ["React"]},
            "resume_data": None,
        }

        key1 = build_jd_expert_cache_key(state1)
        key2 = build_jd_expert_cache_key(state2)

        assert key1 != key2

    def test_same_question_same_key(self):
        state1 = {
            "messages": [HumanMessage(content="技术栈是什么")],
            "task_type": "jd_followup",
            "question_signature": "jd_followup:detail",
            "response_mode": "followup_brief",
            "jd_data": {"position": "后端工程师"},
            "resume_data": None,
        }
        state2 = {
            "messages": [HumanMessage(content="技术栈是什么")],
            "task_type": "jd_followup",
            "question_signature": "jd_followup:detail",
            "response_mode": "followup_brief",
            "jd_data": {"position": "后端工程师"},
            "resume_data": None,
        }

        assert build_jd_expert_cache_key(state1) == build_jd_expert_cache_key(state2)
