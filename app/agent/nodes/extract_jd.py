"""
JD 提取节点 - 从 JD 文本中提取结构化岗位信息

输入：state["jd_data"]["raw_text"]
输出：state["jd_data"] 填充结构化字段
"""
from __future__ import annotations

import json

from app.agent.prompts import JD_EXTRACT_PROMPT
from app.agent.state import AgentState
from app.core.logger import setup_logger
from app.services.llm_service import chat_completion

logger = setup_logger("agent.extract_jd")


def extract_jd(state: AgentState) -> dict:
    """
    JD 提取节点：从 JD 文本中提取结构化岗位信息。

    读取 state["jd_data"]["raw_text"]，通过 LLM 结构化提取。

    Returns:
        {"jd_data": dict}  结构化 JD 信息
    """
    jd_data = state.get("jd_data") or {}
    raw_text = jd_data.get("raw_text", "")

    if not raw_text or not raw_text.strip():
        logger.warning("无 JD 文本可提取")
        return {
            "jd_data": {
                **jd_data,
                "extract_error": "未提供 JD 内容，请粘贴或上传岗位描述。",
            }
        }

    # ---- LLM 结构化提取 ----
    try:
        llm_messages = [
            {"role": "system", "content": JD_EXTRACT_PROMPT},
            {"role": "user", "content": f"请解析以下岗位描述（JD）：\n\n{raw_text}"},
        ]

        response = chat_completion(
            llm_messages,
            temperature=0,
            max_tokens=4096,
        )

        # 尝试解析 JSON（兼容 markdown 代码块包裹）
        json_str = response.strip()
        if json_str.startswith("```"):
            lines = json_str.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            json_str = "\n".join(lines)

        extracted = json.loads(json_str)

        # 保存 raw_text 以便后续节点使用
        extracted["raw_text"] = raw_text

        logger.info(
            "JD 提取完成: position=%s, skills_must=%d项, skills_preferred=%d项",
            extracted.get("position", ""),
            len(extracted.get("skills_must", [])),
            len(extracted.get("skills_preferred", [])),
        )

        return {"jd_data": extracted}

    except json.JSONDecodeError as e:
        logger.error("JD 提取 JSON 解析失败: %s", e)
        return {
            "jd_data": {
                **jd_data,
                "raw_text": raw_text,
                "extract_error": f"结构化提取失败，已保留原始文本：{e}",
            }
        }
    except Exception as e:
        logger.error("JD 提取异常: %s", e, exc_info=True)
        return {
            "jd_data": {
                **jd_data,
                "extract_error": f"JD 提取出错：{e}",
            }
        }
