"""
Web 搜索服务 - Tavily API 封装
"""
from __future__ import annotations

from app.core.logger import setup_logger

logger = setup_logger("web_search_service")


class WebSearchService:
    """基于 Tavily 的 Web 搜索服务"""

    def __init__(self, api_key: str = ""):
        self.api_key = api_key
        self._client = None

    def _get_client(self):
        """延迟初始化 Tavily 客户端"""
        if self._client is None:
            if not self.api_key:
                logger.warning("TAVILY_API_KEY 未配置，Web 搜索不可用")
                return None
            try:
                from tavily import TavilyClient
                self._client = TavilyClient(api_key=self.api_key)
                logger.info("Tavily 客户端初始化成功")
            except ImportError:
                logger.error("tavily-python 未安装，请执行 pip install tavily-python")
                return None
            except Exception as e:
                logger.error("Tavily 客户端初始化失败: %s", e)
                return None
        return self._client

    def search(self, query: str, max_results: int = 5) -> list[dict]:
        """
        执行 Web 搜索。

        Args:
            query: 搜索关键词
            max_results: 最大返回条数
        Returns:
            [{"content": str, "source": str, "type": "web"}]
        """
        client = self._get_client()
        if client is None:
            return []

        try:
            response = client.search(
                query=query,
                max_results=max_results,
                include_answer=False,
            )

            results = []
            for item in response.get("results", []):
                results.append({
                    "content": item.get("content", ""),
                    "source": item.get("url", ""),
                    "type": "web",
                })

            logger.info("Web 搜索完成: query='%s', 返回 %d 条", query[:50], len(results))
            return results

        except Exception as e:
            logger.error("Web 搜索失败: %s", e)
            return []

    @property
    def is_available(self) -> bool:
        """检查搜索服务是否可用"""
        return bool(self.api_key)
