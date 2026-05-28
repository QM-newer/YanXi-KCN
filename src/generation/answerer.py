"""
答案生成器
==========
基于检索结果生成答案
"""

from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from src.utils.llm_client import QwenClient
from src.utils.logger import get_logger

logger = get_logger(__name__)


# 生成提示词
GENERATE_PROMPT = """你是一个来电助手的智能问答系统。

基于以下通话记录信息，回答用户的问题。

## 通话记录：
{context}

## 用户问题：
{question}

请根据上述信息，给出准确、简洁的回答。如果信息不足以回答，请说明情况。
"""


class Answerer:
    """
    答案生成器

    使用 LLM 基于检索结果生成答案
    """

    def __init__(
        self,
        llm_client: QwenClient,
        temperature: float = 0.3,
        max_tokens: int = 512
    ):
        """
        初始化答案生成器

        Args:
            llm_client: LLM 客户端
            temperature: 温度参数
            max_tokens: 最大 token 数
        """
        self.llm = llm_client
        self.temperature = temperature
        self.max_tokens = max_tokens

    def answer(
        self,
        question: str,
        documents: List[Document],
        route: str = "hybrid"
    ) -> Dict[str, Any]:
        """
        生成答案

        Args:
            question: 用户问题
            documents: 检索到的文档
            route: 检索路由类型

        Returns:
            {"text": str, "citations": dict}
        """
        if not documents:
            return {
                "text": "抱歉，没有找到相关信息来回答您的问题。",
                "citations": {}
            }

        # 构建上下文
        context = self._build_context(documents)

        # 构建提示词
        prompt = GENERATE_PROMPT.format(
            context=context,
            question=question
        )

        # 调用 LLM
        try:
            response = self.llm.call(
                prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            # 构建引用
            citations = self._build_citations(documents)

            return {
                "text": response,
                "citations": citations
            }

        except Exception as e:
            logger.error(f"答案生成失败: {e}")
            return {
                "text": f"生成答案时出错: {str(e)}",
                "citations": {}
            }

    def _build_context(self, documents: List[Document], max_docs: int = 5) -> str:
        """构建上下文"""
        contexts = []

        for i, doc in enumerate(documents[:max_docs], 1):
            content = doc.page_content.strip()
            meta = doc.metadata or {}

            category = meta.get('category', '')
            source = f"[{i}]"

            if category:
                contexts.append(f"{source} [{category}] {content}")
            else:
                contexts.append(f"{source} {content}")

        return "\n\n".join(contexts)

    def _build_citations(self, documents: List[Document]) -> Dict[str, str]:
        """构建引用"""
        citations = {}

        for i, doc in enumerate(documents, 1):
            content = doc.page_content[:100]
            citations[f"[{i}]"] = content

        return citations


def create_answerer(
    llm_client: QwenClient,
    temperature: float = 0.3,
    max_tokens: int = 512
) -> Answerer:
    """工厂函数：创建答案生成器"""
    return Answerer(
        llm_client=llm_client,
        temperature=temperature,
        max_tokens=max_tokens
    )
