"""电话类型分类 Agent 测试脚本

Usage:
  python scripts/classifier.py                      # 交互模式
  python scripts/classifier.py "外卖到了"           # 单次分类
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.call_classifier import create_classifier


def main():
    print("正在加载分类器...")
    agent = create_classifier()

    if len(sys.argv) > 1:
        # 单次分类
        text = " ".join(sys.argv[1:])
        result = agent.classify(text)
        print(f"\n输入: {text}")
        print(f"类型: {result['category']} (置信度: {result['confidence']:.2f})")
        print(f"处理动作: {result['action']}")
        print(f"处理规则: {result['handling_response']}")

        if result.get("samples"):
            print("\n检索样例:")
            for s in result["samples"]:
                print(f"  [{s['category']}] {s['text']}")
        return

    # 交互模式
    print("=" * 50)
    print("  电话类型分类 Agent (基于 RAG)")
    print("  输入 exit/quit 退出")
    print("=" * 50 + "\n")

    while True:
        try:
            text = input("对话内容 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye~")
            break

        if not text:
            continue
        if text.lower() in {"exit", "quit", "q"}:
            break

        result = agent.classify(text)
        print(f"→ 类型: {result['category']} (置信度: {result['confidence']:.2f}, 方法: {result['method']})")
        print(f"→ 处理动作: {result['action']}")
        print(f"→ 处理规则: {result['handling_response']}")

        if result.get("samples"):
            print("  检索样例:")
            for s in result["samples"]:
                print(f"    [{s['category']}] {s['text']}")
        print()


if __name__ == "__main__":
    main()
