"""交互式 RAG 查询测试
Usage:
  python scripts/query.py                      # 交互模式
  python scripts/query.py "你的问题"           # 单次查询
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.factory import build_call_rag


def _doc_line(i: int, d, max_len: int = 150) -> str:
    """格式化文档行"""
    meta = d.metadata or {}
    category = meta.get("category", "-")
    content = (d.page_content or "").replace("\n", " ")
    if len(content) > max_len:
        content = content[:max_len] + "..."
    return f"  [{i}] [{category}] {content}"


def print_result(q: str, res) -> None:
    """打印查询结果，兼容 dict 和 dataclass 两种返回"""
    # 统一转为 dict 访问
    if hasattr(res, 'to_dict'):
        res = res.to_dict()
    print("\n" + "=" * 70)
    print(f"Q: {q}")
    print("-" * 70)

    # 路由信息
    route = res.get("route", "unknown")
    print(f"[Route] {route}")

    # 向量检索结果
    v_docs = res.get("vector_docs", [])
    print("-" * 70)
    print(f"[Vector 召回] {len(v_docs)} 条 (显示前 5 条)")
    for i, d in enumerate(v_docs[:5], 1):
        print(_doc_line(i, d))

    # 图谱检索结果
    g_docs = res.get("graph_docs", [])
    if g_docs:
        print("-" * 70)
        print(f"[Graph 召回] {len(g_docs)} 条")
        for i, d in enumerate(g_docs[:3], 1):
            print(_doc_line(i, d, max_len=200))

    # 融合结果
    fused = res.get("retrieved_docs", [])
    if fused:
        print("-" * 70)
        print(f"[融合结果] {len(fused)} 条 (显示前 5 条)")
        for i, d in enumerate(fused[:5], 1):
            print(_doc_line(i, d))

    # 最终答案
    print("-" * 70)
    print("A:")
    print(res.get("text", ""))

    # 引用
    if res.get("citations"):
        print("-" * 70)
        print("[Citations]")
        for k, v in res["citations"].items():
            preview = v if isinstance(v, str) else str(v)
            print(f"  {k}: {preview[:100]}...")

    print("=" * 70)


def main() -> None:
    import traceback
    print("正在加载 Call RAG 系统...")
    try:
        rag = build_call_rag(str(PROJECT_ROOT / "configs" / "config.yaml"))
        print("加载完成！\n")
    except Exception as e:
        print(f"[ERROR] 加载失败: {e}")
        traceback.print_exc()
        return

    if len(sys.argv) > 1:
        # 单次查询
        q = " ".join(sys.argv[1:])
        try:
            res = rag.query(q)
            print_result(q, res)
        except Exception as e:
            print(f"[ERROR] 查询失败: {e}")
            traceback.print_exc()
        return

    # 交互模式
    print("=" * 50)
    print("  来电助手 RAG 查询系统")
    print("  输入 exit/quit 退出")
    print("=" * 50 + "\n")

    while True:
        try:
            q = input("Q > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye~")
            break

        if not q:
            continue
        if q.lower() in {"exit", "quit", "q"}:
            break

        try:
            res = rag.query(q)
            print_result(q, res)
        except Exception as e:
            print(f"[ERROR] {type(e).__name__}: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
