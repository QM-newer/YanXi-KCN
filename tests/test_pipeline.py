"""
Pipeline 测试
=============
测试主流程
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from src.pipeline import CallAssistantPipeline, PipelineResult
from src.retrieval.router import CallCategory


class TestPipeline(unittest.TestCase):
    """Pipeline测试用例"""

    def setUp(self):
        """初始化"""
        self.pipeline = CallAssistantPipeline()

    def test_delivery_class_on(self):
        """测试外卖-上课中"""
        result = self.pipeline.query(
            "您好，我是外卖小哥，您的餐到了，取餐码是1234",
            is_class_in_session=True
        )
        self.assertEqual(result.category, "delivery")
        self.assertIn("外卖", result.response)
        print(f"\n[测试] 外卖-上课中:\n{result.response}")

    def test_delivery_class_off(self):
        """测试外卖-下课"""
        result = self.pipeline.query(
            "您好，我是骑手，您的美团外卖到了",
            is_class_in_session=False
        )
        self.assertEqual(result.category, "delivery")
        print(f"\n[测试] 外卖-下课:\n{result.response}")

    def test_normal_leader(self):
        """测试领导来电"""
        result = self.pipeline.query("喂，我是领导，有个紧急的事要跟你说")
        self.assertEqual(result.category, "normal")
        print(f"\n[测试] 领导来电:\n{result.response}")

    def test_normal_family(self):
        """测试家人来电"""
        result = self.pipeline.query("妈妈，你吃饭了吗？")
        self.assertEqual(result.category, "normal")
        print(f"\n[测试] 家人来电:\n{result.response}")

    def test_risk_transfer(self):
        """测试转账诈骗"""
        result = self.pipeline.query("您好，这里是公安局，您涉嫌一起案件，需要转账到安全账户")
        self.assertEqual(result.category, "risk")
        print(f"\n[测试] 转账诈骗:\n{result.response}")

    def test_risk_verify_code(self):
        """测试验证码诈骗"""
        result = self.pipeline.query("您的订单异常，请告诉我验证码，我帮您处理")
        self.assertEqual(result.category, "risk")
        print(f"\n[测试] 验证码诈骗:\n{result.response}")


class TestRouter(unittest.TestCase):
    """路由器测试"""

    def setUp(self):
        from src.retrieval.router import CallRouter
        self.router = CallRouter()

    def test_keyword_match(self):
        """测试关键词匹配"""
        result = self.router.route("外卖到了，取餐码1234")
        self.assertEqual(result.category, "delivery")

    def test_risk_detect(self):
        """测试风险检测"""
        result = self.router.route("请转账到安全账户")
        self.assertEqual(result.category, "risk")


class TestAgents(unittest.TestCase):
    """Agent测试"""

    def test_delivery_agent(self):
        """测试外卖Agent"""
        from src.agents.delivery import DeliveryAgent
        agent = DeliveryAgent(is_class_in_session=True)
        result = agent.process("您好，外卖到了")
        self.assertEqual(result.category, "delivery")
        self.assertIn("上课", result.response)

    def test_normal_agent(self):
        """测试正常来电Agent"""
        from src.agents.normal import NormalCallAgent
        agent = NormalCallAgent()
        result = agent.process("我是领导，有急事")
        self.assertEqual(result.category, "normal")

    def test_risk_agent(self):
        """测试诈骗Agent"""
        from src.agents.risk import RiskAgent
        agent = RiskAgent()
        result = agent.process("恭喜中奖，请转账领取")
        self.assertEqual(result.category, "risk")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("运行测试...")
    print("=" * 60)
    unittest.main(verbosity=2)
