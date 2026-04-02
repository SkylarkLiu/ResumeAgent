"""
分析子图构建入口。
"""
from app.agent.subgraphs.jd_analysis import build_jd_analysis_subgraph
from app.agent.subgraphs.resume_analysis import build_resume_analysis_subgraph

__all__ = [
    "build_resume_analysis_subgraph",
    "build_jd_analysis_subgraph",
]
