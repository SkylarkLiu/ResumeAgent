"""
日志配置

关键设计：
- 首次调用 setup_logger() 时配置 root logger（含 stderr handler）
- 所有子模块 logger 通过 Python 层级机制继承 root 的 handler
- uvicorn --reload 模式下，子进程的日志通过 root logger + stderr 正确输出
"""
import logging
import sys


def setup_logger(name: str = "app", level: str | None = None) -> logging.Logger:
    """创建格式统一的 logger

    首次调用时配置 root logger（含 stderr handler），后续调用复用。
    level 参数仅在首次调用时生效，用于设定 root logger 的日志级别。
    """
    # 首次调用时配置 root logger
    root = logging.getLogger()
    if not root.handlers:
        from app.core.config import get_settings

        settings = get_settings()
        log_level = (level or settings.log_level or "INFO").upper()

        root.setLevel(getattr(logging, log_level, logging.INFO))
        root.propagate = False

        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # stderr handler —— uvicorn reload 子进程下最可靠的输出方式
        console = logging.StreamHandler(sys.stderr)
        console.setLevel(logging.DEBUG)
        console.setFormatter(formatter)
        root.addHandler(console)

    logger = logging.getLogger(name)
    return logger
