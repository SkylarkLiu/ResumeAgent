"""
ResumeAgent 回归测试 conftest。

提供：
- 测试用 FastAPI app（mock 掉 LLM 调用）
- 内存 checkpointer（不依赖外部 DB）
- httpx AsyncClient
- mock LLM 服务

注意：各模块通过 `from app.services.llm_service import chat_completion` 方式导入，
因此必须 mock 各模块的局部引用，否则 patch 不生效。
"""
from __future__ import annotations

import json
import uuid
from typing import Any
from unittest.mock import patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.core.config import Settings


# ---- 测试专用 Settings ----

class TestSettings(Settings):
    """测试环境配置，不走 .env 文件。"""
    zhipuai_api_key: str = "test-key"
    log_level: str = "WARNING"
    app_env: str = "test"
    debug_mode: bool = False
    checkpoint_db_url: str = ""
    expert_cache_backend: str = "state_checkpointer"
    expert_cache_db_url: str = ""

    class Config:
        env_file = None  # 测试不读 .env


# ---- Mock LLM 服务 ----

MOCK_QA_ANSWER = "这是测试 QA 回答。简历中 STAR 法则是指 Situation（情境）、Task（任务）、Action（行动）、Result（结果）。"

MOCK_RESUME_EXTRACTED = json.dumps({
    "name": "张三",
    "target_position": "前端开发工程师",
    "summary": "3年前端开发经验，熟练掌握 React/Vue",
    "skills": ["React", "Vue", "TypeScript", "CSS"],
    "experience": [
        {"company": "ABC公司", "position": "前端工程师", "duration": "2022-2024", "description": "负责核心业务前端开发"}
    ],
    "education": [
        {"school": "XX大学", "major": "计算机科学", "degree": "本科"}
    ],
    "projects": [
        {"name": "电商后台管理系统", "description": "基于 React + Ant Design 的中后台系统"}
    ],
}, ensure_ascii=False)

MOCK_JD_EXTRACTED = json.dumps({
    "position": "高级前端开发工程师",
    "company": "某大厂",
    "summary": "负责核心业务前端架构设计与开发",
    "skills_must": ["React", "TypeScript", "Node.js"],
    "skills_preferred": ["Vue", "Webpack"],
    "tech_stack": {"frontend": "React", "build": "Webpack"},
    "keywords": ["前端", "React", "架构"],
}, ensure_ascii=False)

MOCK_RESUME_ANALYSIS = "## 简历分析报告\n\n### 整体评价\n该简历整体结构清晰，内容充实。\n\n### 优势\n- 技术栈全面\n- 项目经验丰富\n\n### 改进建议\n- 增加量化成果\n- 补充软技能描述"

MOCK_JD_ANALYSIS = "## 岗位分析报告\n\n### 岗位核心要求\n该岗位要求具备扎实的前端基础和架构设计能力。\n\n### 技术栈解读\n- React 为核心框架\n- TypeScript 必须熟练\n\n### 简历写作建议\n- 突出 React 项目经验\n- 展示架构设计能力"

MOCK_RESUME_FOLLOWUP = "您的简历亮点包括：1) 技术栈全面，覆盖 React/Vue 2) 有电商系统实战经验 3) 具备 TypeScript 工程化能力。"

MOCK_JD_FOLLOWUP = "该岗位面试需要准备：1) React 原理与性能优化 2) TypeScript 类型系统 3) 前端架构设计模式 4) 系统设计题。"

MOCK_MATCH_FOLLOWUP = "您与该岗位的差距主要在：1) 缺少大型项目架构经验 2) Node.js 服务端能力需加强 3) 缺少性能优化案例。建议：补齐 Node.js 经验，增加架构设计项目描述。"


def _mock_chat_completion(messages, *, temperature=0.5, max_tokens=2048, thinking=None, **kwargs):
    """
    根据 system prompt 判断当前是哪个 LLM 调用，返回对应的 mock 结果。
    """
    system_msg = ""
    for msg in messages:
        if msg.get("role") == "system":
            system_msg = msg.get("content", "")
            break

    user_msg = ""
    for msg in messages:
        if msg.get("role") == "user":
            user_msg = msg.get("content", "")
            break

    # Router 分类
    if "Router" in system_msg or "路由" in system_msg or "route_type" in system_msg:
        # 根据用户消息推断分类
        if any(k in user_msg for k in ["简历", "评估", "优化"]):
            return json.dumps({"reasoning": "用户询问简历", "route_type": "direct", "task_type": "resume_analysis"})
        elif any(k in user_msg for k in ["岗位", "JD", "jd"]):
            return json.dumps({"reasoning": "用户询问岗位", "route_type": "direct", "task_type": "jd_analysis"})
        return json.dumps({"reasoning": "普通问答", "route_type": "retrieve", "task_type": "qa"})

    # 简历提取
    if "简历" in system_msg and "提取" in system_msg:
        return MOCK_RESUME_EXTRACTED

    # JD 提取
    if ("JD" in system_msg or "岗位" in system_msg) and "提取" in system_msg:
        return MOCK_JD_EXTRACTED

    # 简历分析
    if "简历" in system_msg and ("分析" in system_msg or "评估" in system_msg):
        return MOCK_RESUME_ANALYSIS

    # JD 分析
    if "JD" in system_msg or "岗位" in system_msg:
        return MOCK_JD_ANALYSIS

    # 追问类
    if "追问" in system_msg or "followup" in system_msg.lower():
        if "简历" in system_msg:
            return MOCK_RESUME_FOLLOWUP
        if "岗位" in system_msg or "JD" in system_msg:
            return MOCK_JD_FOLLOWUP
        if "匹配" in system_msg:
            return MOCK_MATCH_FOLLOWUP
        return MOCK_RESUME_FOLLOWUP

    # 默认 QA 回答
    return MOCK_QA_ANSWER


def _mock_chat_completion_stream(messages, *, temperature=0.5, max_tokens=2048, thinking=None, **kwargs):
    """同步流式 mock：yield 完整回答的每个字符。"""
    answer = _mock_chat_completion(messages, temperature=temperature, max_tokens=max_tokens, thinking=thinking, **kwargs)
    for char in answer:
        yield char


async def _mock_chat_completion_stream_async(messages, *, temperature=0.5, max_tokens=2048, thinking=None, **kwargs):
    """异步流式 mock：yield 完整回答的每个字符。"""
    import asyncio
    answer = _mock_chat_completion(messages, temperature=temperature, max_tokens=max_tokens, thinking=thinking, **kwargs)
    for char in answer:
        yield char
        await asyncio.sleep(0)


# ---- 需要被 mock 的所有模块级 chat_completion 引用路径 ----
# 各模块通过 from app.services.llm_service import chat_completion 导入
# 因此需要 mock 各模块的局部引用才能生效

_CHAT_MOCK_TARGETS = [
    "app.services.llm_service.chat_completion",
    "app.agent.agents.supervisor.chat_completion",
    "app.agent.nodes.analyze_jd.chat_completion",
    "app.agent.nodes.extract_jd.chat_completion",
    "app.agent.nodes.extract_resume.chat_completion",
    "app.agent.nodes.generate_analysis.chat_completion",
    "app.agent.nodes.generate.chat_completion",
    # NOTE: rag_service 已不在 Agent 主流程中使用，其导入依赖 psycopg，
    # 测试环境无法加载，因此从 mock 目标中移除
    # "app.services.rag_service.chat_completion",
]

_STREAM_MOCK_TARGETS = [
    "app.services.llm_service.chat_completion_stream",
]

_STREAM_ASYNC_MOCK_TARGETS = [
    "app.services.llm_service.chat_completion_stream_async",
    "app.agent.nodes.analyze_jd.chat_completion_stream_async",
    "app.agent.nodes.generate_analysis.chat_completion_stream_async",
    "app.agent.nodes.generate.chat_completion_stream_async",
]


# ---- Fixtures ----

@pytest.fixture(scope="session")
def test_settings():
    return TestSettings()


@pytest_asyncio.fixture
async def app(test_settings):
    """构建测试用 FastAPI app，mock 掉 LLM。"""
    mock_patches = []

    # 预导入需要 mock 的模块（跳过依赖缺失的模块）
    import importlib
    for target in _CHAT_MOCK_TARGETS + _STREAM_MOCK_TARGETS + _STREAM_ASYNC_MOCK_TARGETS:
        module_path = target.rsplit(".", 1)[0]
        try:
            importlib.import_module(module_path)
        except Exception:
            pass

    # Mock 所有 chat_completion 引用
    for target in _CHAT_MOCK_TARGETS:
        p = patch(target, side_effect=_mock_chat_completion)
        p.start()
        mock_patches.append(p)

    # Mock 所有 chat_completion_stream 引用
    for target in _STREAM_MOCK_TARGETS:
        p = patch(target, side_effect=_mock_chat_completion_stream)
        p.start()
        mock_patches.append(p)

    # Mock 所有 chat_completion_stream_async 引用
    for target in _STREAM_ASYNC_MOCK_TARGETS:
        p = patch(target, side_effect=_mock_chat_completion_stream_async)
        p.start()
        mock_patches.append(p)

    # Mock settings
    settings_p = patch("app.services.llm_service._settings", test_settings)
    settings_p.start()
    mock_patches.append(settings_p)

    from app.main import app as real_app
    yield real_app

    # Cleanup
    for p in mock_patches:
        p.stop()


@pytest_asyncio.fixture
async def client(app):
    """HTTP 异步测试客户端，自动管理 ASGI lifespan。"""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # 手动触发 app lifespan，确保 Agent 图被初始化
        async with app.router.lifespan_context(app):
            yield ac


@pytest.fixture
def fresh_session_id():
    """生成一个全新的 session ID。"""
    return f"test_{uuid.uuid4().hex[:12]}"


# ---- 可观测性测试 Fixture ----

@pytest.fixture
def obs_log_capture():
    """捕获 obs.request logger 的输出。"""
    import logging
    from io import StringIO

    logger = logging.getLogger("obs.request")
    handler = logging.StreamHandler(StringIO())
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    yield handler.stream

    logger.removeHandler(handler)
