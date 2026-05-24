"""
Embedding 模型加载
================
支持两种模式：
1. DashScope API (推荐，无本地模型下载)
2. sentence-transformers (本地 BGE 模型)
"""

import os
from typing import List, Union
import numpy as np
from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODEL = "BAAI/bge-large-zh-v1.5"
DEFAULT_DEVICE = "cpu"


class DashScopeEmbedder:
    """
    DashScope Embedding API 封装

    使用通义千问 Embedding API，无需本地模型
    模型: text-embedding-v3 (1536维) 或 text-embedding-v2 (1512维)
    """

    def __init__(
        self,
        model: str = "text-embedding-v3",
        api_key: str = None,
        dimensions: int = 1536,
        batch_size: int = 25
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("QWEN_API_KEY", "")
        self.dimensions = dimensions
        self.batch_size = batch_size

    def _call_api(self, texts: List[str]) -> List[List[float]]:
        """调用 DashScope Embedding API"""
        try:
            import dashscope
            from dashscope import TextEmbedding

            dashscope.api_key = self.api_key

            embeddings = []
            # 分批处理
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                response = TextEmbedding.call(
                    model=self.model,
                    input=batch
                )

                if response.status_code == 200:
                    for item in response.output['embeddings']:
                        embeddings.append(item['embedding'])
                else:
                    raise Exception(f"API error: {response.code} {response.message}")

            return embeddings

        except ImportError:
            raise ImportError("请安装 dashscope: pip install dashscope")
        except Exception as e:
            raise Exception(f"DashScope API 调用失败: {e}")

    def encode(self, texts: Union[str, List[str]], **kwargs) -> np.ndarray:
        """编码文本为向量"""
        if isinstance(texts, str):
            texts = [texts]

        embeddings = self._call_api(texts)
        return np.array(embeddings)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """LangChain Chroma 接口"""
        return self._call_api(texts)

    def embed_query(self, text: str) -> List[float]:
        """LangChain Chroma 接口"""
        embeddings = self._call_api([text])
        return embeddings[0]

    def __call__(self, texts: Union[str, List[str]], **kwargs) -> np.ndarray:
        """支持直接调用"""
        return self.encode(texts, **kwargs)


class BGEEmbedder:
    """
    BGE Embedding 模型封装

    支持本地 GPU/CPU 推理
    兼容 LangChain Chroma 接口
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = DEFAULT_DEVICE,
        batch_size: int = 32,
        normalize: bool = True
    ):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.normalize = normalize
        self._model = None

    def _load_model(self):
        """延迟加载模型"""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name, device=self.device)
            except ImportError:
                raise ImportError(
                    "请安装 sentence-transformers: pip install sentence-transformers"
                )
        return self._model

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = None,
        show_progress: bool = False,
        **kwargs
    ) -> np.ndarray:
        """
        编码文本为向量

        Args:
            texts: 单个文本或文本列表
            batch_size: 批次大小
            show_progress: 是否显示进度条
            **kwargs: 其他参数

        Returns:
            numpy.ndarray: 文本向量
        """
        model = self._load_model()
        batch_size = batch_size or self.batch_size

        # 单个文本转列表
        if isinstance(texts, str):
            texts = [texts]

        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=self.normalize,
            **kwargs
        )

        return embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        LangChain Chroma 接口所需的方法

        Args:
            texts: 文本列表

        Returns:
            向量列表
        """
        embeddings = self.encode(texts)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """
        LangChain Chroma 接口所需的方法

        Args:
            text: 查询文本

        Returns:
            向量
        """
        embedding = self.encode(text)
        if len(embedding.shape) > 1:
            embedding = embedding[0]
        return embedding.tolist()

    def __call__(self, texts: Union[str, List[str]], **kwargs) -> np.ndarray:
        """支持直接调用"""
        return self.encode(texts, **kwargs)


def load_bge_embedding(
    model_name: str = DEFAULT_MODEL,
    device: str = DEFAULT_DEVICE,
    batch_size: int = 32
) -> BGEEmbedder:
    """
    加载 BGE Embedding 模型

    Args:
        model_name: 模型名称
        device: 设备 (cpu/cuda)
        batch_size: 批次大小

    Returns:
        BGEEmbedder 实例
    """
    return BGEEmbedder(
        model_name=model_name,
        device=device,
        batch_size=batch_size
    )


def load_embedding(
    backend: str = "dashscope",
    model_name: str = None,
    device: str = DEFAULT_DEVICE,
    batch_size: int = 32
) -> Union[DashScopeEmbedder, BGEEmbedder]:
    """
    加载 Embedding 模型（工厂函数）

    Args:
        backend: 后端类型 ("dashscope" / "bge")
        model_name: 模型名称（仅 BGE 使用）
        device: 设备（仅 BGE 使用）
        batch_size: 批次大小

    Returns:
        Embedder 实例
    """
    backend = backend.lower()

    if backend == "dashscope":
        return DashScopeEmbedder(batch_size=batch_size)
    elif backend == "bge":
        return load_bge_embedding(
            model_name=model_name or DEFAULT_MODEL,
            device=device,
            batch_size=batch_size
        )
    else:
        raise ValueError(f"不支持的 backend: {backend}，可选: dashscope, bge")
