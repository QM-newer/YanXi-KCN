"""
LLM 客户端
==========
统一管理通义千问API调用
"""

import os
import json
from typing import Optional, Dict, Any
from src.utils.logger import get_logger

logger = get_logger(__name__)


class QwenClient:
    """通义千问API客户端"""

    def __init__(
        self,
        api_key: str = None,
        model: str = "qwen-turbo",
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        timeout: int = 30,
        max_retries: int = 3
    ):
        self.api_key = api_key or os.environ.get("QWEN_API_KEY", "")
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries

    def call(self, prompt: str, system: str = None, **kwargs) -> str:
        """
        调用LLM生成回复

        Args:
            prompt: 用户输入
            system: 系统提示（可选）
            **kwargs: 其他参数

        Returns:
            生成的文本
        """
        # 优先使用 dashscope 库
        result = self._call_with_dashscope(prompt, system, **kwargs)
        if result:
            return result
        return self._fallback_response(prompt)

    def _call_with_dashscope(self, prompt: str, system: str = None, **kwargs) -> Optional[str]:
        """使用dashscope库调用"""
        try:
            import dashscope
            from dashscope import Generation
            
            dashscope.api_key = self.api_key

            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})

            response = Generation.call(
                model=self.model,
                messages=messages,
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                result_format="message"
            )

            if response.status_code == 200:
                return response.output.choices[0].message.content
            else:
                logger.warning(f"Dashscope错误: {response.code} {response.message}")
                return None

        except ImportError:
            logger.warning("dashscope未安装")
            return None
        except Exception as e:
            logger.warning(f"Dashscope调用异常: {e}")
            return None

    def _fallback_response(self, prompt: str) -> str:
        """降级响应"""
        # 简单的规则匹配降级
        if any(kw in prompt for kw in ["外卖", "餐", "取餐"]):
            return "好的，请稍等。"
        elif any(kw in prompt for kw in ["领导", "老板", "经理"]):
            return "好的，请稍等，我马上过来。"
        elif any(kw in prompt for kw in ["转账", "验证码", "中奖"]):
            return "谢谢，不需要。"
        return "好的，我知道了。"
