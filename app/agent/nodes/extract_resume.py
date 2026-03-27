"""
简历提取节点 - 从文本/PDF/图片中提取结构化简历信息

输入：state["resume_data"]["raw_text"] 或 state["resume_data"]["file_path"]
输出：state["resume_data"] 填充结构化字段
"""
from __future__ import annotations

import base64
import json
import os
import tempfile

from langchain_core.messages import HumanMessage

from app.agent.prompts import RESUME_EXTRACT_PROMPT
from app.agent.state import AgentState
from app.core.logger import setup_logger
from app.services.llm_service import chat_completion

logger = setup_logger("agent.extract_resume")


def extract_resume(state: AgentState) -> dict:
    """
    简历提取节点：从简历文本或文件中提取结构化信息。

    优先使用 state["resume_data"]["raw_text"]（前端粘贴/上传解析后的文本），
    如果没有，则尝试从 state["resume_data"]["file_path"] 读取 PDF/图片文件。

    Returns:
        {"resume_data": dict}  结构化简历信息
    """
    resume_data = state.get("resume_data") or {}
    raw_text = resume_data.get("raw_text", "")
    file_path = resume_data.get("file_path", "")
    file_base64 = resume_data.get("file_base64", "")

    # ---- 确定文本来源 ----
    if raw_text:
        logger.info("使用已有简历文本（%d 字符）", len(raw_text))
        text_to_extract = raw_text
    elif file_base64:
        # base64 图片走视觉模型提取
        text_to_extract = _extract_from_image_base64(file_base64)
    elif file_path and os.path.exists(file_path):
        text_to_extract = _extract_from_file(file_path)
    else:
        logger.warning("无简历文本或文件可提取")
        return {
            "resume_data": {
                **resume_data,
                "extract_error": "未提供简历内容，请上传简历文件或粘贴简历文本。",
            }
        }

    if not text_to_extract:
        return {
            "resume_data": {
                **resume_data,
                "extract_error": "简历内容为空或解析失败。",
            }
        }

    # ---- LLM 结构化提取 ----
    try:
        llm_messages = [
            {"role": "system", "content": RESUME_EXTRACT_PROMPT},
            {"role": "user", "content": f"请解析以下简历内容：\n\n{text_to_extract}"},
        ]

        response = chat_completion(
            llm_messages,
            temperature=0,
            max_tokens=4096,
        )

        # 尝试解析 JSON（兼容 markdown 代码块包裹）
        json_str = response.strip()
        if json_str.startswith("```"):
            # 去掉 markdown 代码块
            lines = json_str.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            json_str = "\n".join(lines)

        extracted = json.loads(json_str)

        # 保存 raw_text 以便后续节点使用
        extracted["raw_text"] = text_to_extract

        logger.info(
            "简历提取完成: name=%s, skills=%d项, experience=%d段, projects=%d个",
            extracted.get("name", ""),
            len(extracted.get("skills", [])),
            len(extracted.get("experience", [])),
            len(extracted.get("projects", [])),
        )

        return {"resume_data": extracted}

    except json.JSONDecodeError as e:
        logger.error("简历提取 JSON 解析失败: %s", e)
        return {
            "resume_data": {
                **resume_data,
                "raw_text": text_to_extract,
                "extract_error": f"结构化提取失败，已保留原始文本：{e}",
            }
        }
    except Exception as e:
        logger.error("简历提取异常: %s", e, exc_info=True)
        return {
            "resume_data": {
                **resume_data,
                "extract_error": f"简历提取出错：{e}",
            }
        }


def _extract_from_file(file_path: str) -> str:
    """从 PDF/图片/文本文件中提取文本"""
    ext = os.path.splitext(file_path)[1].lower()

    if ext in (".txt", ".md"):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    if ext == ".pdf":
        return _extract_from_pdf(file_path)

    if ext in (".png", ".jpg", ".jpeg"):
        with open(file_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        return _extract_from_image_base64(b64)

    logger.warning("不支持的简历文件格式: %s", ext)
    return ""


def _extract_from_pdf(file_path: str) -> str:
    """使用 PyMuPDF 提取 PDF 文本"""
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(file_path)
        texts = []
        for page in doc:
            text = page.get_text("text")
            if text.strip():
                texts.append(text)
        doc.close()

        full_text = "\n\n".join(texts)
        if full_text.strip():
            logger.info("PDF 文本提取成功: %d 页, %d 字符", len(texts), len(full_text))
            return full_text

        # 文本为空 → 可能是扫描件，渲染为图片走视觉模型
        logger.info("PDF 无可提取文本，尝试渲染为图片走视觉理解")
        return _pdf_to_image_extract(file_path)

    except ImportError:
        logger.error("PyMuPDF 未安装，无法解析 PDF")
        return ""
    except Exception as e:
        logger.error("PDF 解析失败: %s", e)
        return ""


def _pdf_to_image_extract(file_path: str) -> str:
    """将 PDF 每页渲染为图片，走视觉模型 OCR"""
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(file_path)
        all_text = []

        for page_num, page in enumerate(doc):
            # 渲染为 PNG（300 DPI 保证清晰度）
            pix = page.get_pixmap(dpi=200)
            img_bytes = pix.tobytes("png")
            b64 = base64.b64encode(img_bytes).decode("utf-8")
            text = _extract_from_image_base64(b64)
            if text:
                all_text.append(f"[第{page_num + 1}页]\n{text}")

        doc.close()
        return "\n\n".join(all_text)

    except Exception as e:
        logger.error("PDF 渲染图片失败: %s", e)
        return ""


def _extract_from_image_base64(image_base64: str) -> str:
    """使用 GLM-4V-Flash 视觉模型从图片中提取文本"""
    try:
        from zai import ZhipuAiClient

        from app.core.config import get_settings
        from app.services.vision_service import compress_image_base64

        settings = get_settings()
        client = ZhipuAiClient(api_key=settings.zhipuai_api_key)

        # 压缩大图，防止超过 API 大小限制（返回 b64 + mime）
        b64, mime = compress_image_base64(image_base64)

        response = client.chat.completions.create(
            model=settings.vision_model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/{mime};base64,{b64}"},
                        },
                        {
                            "type": "text",
                            "text": "请完整提取这张图片中的所有文字内容，保持原始格式和布局。如果是简历，请完整保留所有信息。",
                        },
                    ],
                }
            ],
        )

        text = response.choices[0].message.content or ""
        logger.info("视觉模型提取: %d 字符", len(text))
        return text

    except Exception as e:
        logger.error("视觉模型提取失败: %s", e)
        return ""
