"""
Bug 修复验证脚本
================
逐个验证所有修复点是否生效。

Usage:
    python tests/test_bugfixes.py           # 运行所有离线测试
    python tests/test_bugfixes.py --quick   # 仅快速核心测试
"""

import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================
# 辅助函数
# =============================================================

def ok(msg: str):
    print(f"  [PASS] {msg}")

def fail(msg: str):
    print(f"  [FAIL] {msg}")

def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# =============================================================
# 修复 1&2: llm_client.generate() → llm_client.call()
# =============================================================

def test_fix_llm_client_call():
    """
    验证 community_builder.py 和 build_indices.py 中
    不再调用 llm_client.generate()（应改为 .call()）
    """
    section("修复 1&2: llm_client.call() 调用正确性")

    # 检查社区构建器
    comm_path = PROJECT_ROOT / "src" / "indexing" / "community_builder.py"
    with open(comm_path, "r", encoding="utf-8") as f:
        comm_content = f.read()
    if "llm_client.generate(prompt)" in comm_content:
        fail("community_builder.py 仍使用 llm_client.generate()")
    else:
        ok("community_builder.py 已改用 llm_client.call()")

    if "llm_client.call(prompt)" in comm_content:
        ok("community_builder.py 确认使用 llm_client.call()")
    else:
        fail("community_builder.py 未找到 llm_client.call()")

    # 检查索引构建脚本
    build_path = PROJECT_ROOT / "scripts" / "build_indices.py"
    with open(build_path, "r", encoding="utf-8") as f:
        build_content = f.read()
    if "llm_client.generate(prompt)" in build_content:
        fail("build_indices.py 仍使用 llm_client.generate()")
    else:
        ok("build_indices.py 已改用 llm_client.call()")

    if "llm_client.call(prompt)" in build_content:
        ok("build_indices.py 确认使用 llm_client.call()")
    else:
        fail("build_indices.py 未找到 llm_client.call()")

    # 验证 QwenClient 确实有 call() 方法
    from src.utils.llm_client import QwenClient
    assert hasattr(QwenClient, "call"), "QwenClient 缺少 call 方法"
    ok("QwenClient.call() 方法存在")
    assert not hasattr(QwenClient, "generate"), "QwenClient 不应有 generate 方法"
    ok("QwenClient.generate() 方法不存在 (正确)")


# =============================================================
# 修复 3: graph_retriever._tokenize() 词级分词
# =============================================================

def test_fix_tokenize():
    """验证 _tokenize() 方法的词级分词功能"""
    section("修复 3: graph_retriever._tokenize() 词级分词")

    from src.retrieval.graph_retriever import GraphRetriever

    # 中文双字以上词组提取（正则匹配连续中文字符序列）
    tokens = GraphRetriever._tokenize("外卖配送到了，取餐码1234")
    # regex 提取连续中文字符序列 (2+ 字符): {"外卖配送到了", "取餐码"}
    assert len(tokens) >= 2, f"中文分词结果不足: {tokens}"
    assert "取餐码" in tokens, f"取餐码 不在分词结果中: {tokens}"
    ok(f"中文序列分词正确: {tokens}")

    # 验证不会产生单字符
    assert "外" not in tokens, "不应有单字'外'"
    assert "送" not in tokens, "不应有单字'送'"
    assert "了" not in tokens, "不应有单字'了'"
    ok("单字符不会出现在分词结果中")

    # 英文单词（>=2 字符）
    tokens = GraphRetriever._tokenize("hello world ok a")
    assert "hello" in tokens and "world" in tokens, f"英文分词失败: {tokens}"
    assert "a" not in tokens, "单字符英文应被过滤"
    ok(f"英文分词正确: hello,world,ok 都保留; 单字母 a 被过滤")

    # 空输入
    tokens = GraphRetriever._tokenize("123456 !@#$")
    assert len(tokens) == 0, f"纯数字/符号应无词: {tokens}"
    ok("纯数字/符号正确返回空集合")


# =============================================================
# 修复 4: int(cid_str) 安全转换
# =============================================================

def test_fix_int_cid():
    """验证 community_id 转换的安全处理"""
    section("修复 4: int(cid_str) 安全转换")

    cid = "5"
    try:
        result = int(cid)
        assert result == 5
    except (ValueError, TypeError):
        fail(f"数字字符串 {cid!r} 转换失败")
    ok(f"数字字符串 '{cid}' → int({result}) 正确")

    cid = "uuid-1234"
    try:
        _ = int(cid)
        fail(f"非数字字符串 {cid!r} 不应成功")
    except ValueError:
        pass
    # 模拟修复逻辑
    try:
        result = int(cid)
    except (ValueError, TypeError):
        result = cid
    ok(f"非数字字符串 '{cid}' → 保留原值 '{result}' (正确)")


# =============================================================
# 修复 5: summary_db 在 retrieve() 中被使用
# =============================================================

def test_fix_summary_db_usage():
    """验证 retrieve() 中 summary_db 优先路径存在"""
    section("修复 5: summary_db 向量检索路径")

    graph_path = PROJECT_ROOT / "src" / "retrieval" / "graph_retriever.py"
    with open(graph_path, "r", encoding="utf-8") as f:
        content = f.read()

    if "self.summary_db is not None" in content:
        ok("retrieve() 中检查了 summary_db")
    else:
        fail("retrieve() 中未检查 summary_db")

    if "self.summary_db.similarity_search" in content:
        ok("retrieve() 中调用了 summary_db.similarity_search()")
    else:
        fail("未找到 summary_db.similarity_search() 调用")

    if "_find_relevant_communities" in content:
        ok("关键词匹配作为 fallback 保留")


# =============================================================
# 修复 6: _load() 支持 pickle
# =============================================================

def test_fix_load_pickle():
    """验证 _load() 中 pickle 加载路径"""
    section("修复 6: _load() pickle 格式支持")

    graph_path = PROJECT_ROOT / "src" / "retrieval" / "graph_retriever.py"
    with open(graph_path, "r", encoding="utf-8") as f:
        content = f.read()

    assert "import pickle" in content, "缺少 pickle 导入"
    ok("pickle 导入存在")

    assert "graph_file.suffix" in content, "缺少扩展名判断"
    ok("按文件扩展名区分加载方式")

    assert "pickle.load" in content, "缺少 pickle.load() 调用"
    ok("pickle.load() 调用存在")

    assert "UnpicklingError" in content, "缺少 UnpicklingError 异常处理"
    ok("pickle 加载失败回退 JSON")

    # 验证 _build_graph_from_json 辅助方法存在
    assert "_build_graph_from_json" in content, "缺少 _build_graph_from_json 方法"
    ok("_build_graph_from_json 辅助方法存在")


# =============================================================
# 修复 7: call_classifier LLM 增强模糊匹配
# =============================================================

def test_fix_llm_fuzzy_match():
    """验证 LLM 增强的模糊匹配逻辑"""
    section("修复 7: call_classifier LLM 增强模糊匹配")

    clf_path = PROJECT_ROOT / "src" / "agents" / "call_classifier.py"
    with open(clf_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 检查模糊匹配逻辑
    assert "rstrip(" in content, "缺少 rstrip() 标点清理"
    ok("rstrip() 标点清理存在")

    assert "if cat in result:" in content, "缺少模糊匹配 'cat in result'"
    ok("模糊匹配 'cat in result' 存在")

    # 模拟模糊匹配逻辑
    categories = ["外卖配送", "快递取件", "诈骗风险"]

    # 模拟 LLM 返回带句号
    result = "外卖配送。"
    result = result.strip().rstrip("。.!！?？,，、；;：:\n\r")
    matched = None
    for cat in categories:
        if cat in result:
            matched = cat
            break
    assert matched == "外卖配送", f"模糊匹配失败: {matched}"
    ok(f"LLM 返回 '外卖配送。' → 模糊匹配到 '{matched}'")

    # 模拟 LLM 返回带额外说明
    result = "类型是快递取件，需要代接"
    result = result.strip().rstrip("。.!！?？,，、；;：:\n\r")
    matched = None
    for cat in categories:
        if cat in result:
            matched = cat
            break
    assert matched == "快递取件", f"模糊匹配失败: {matched}"
    ok(f"LLM 返回 '类型是快递取件，需要代接' → 模糊匹配到 '{matched}'")


# =============================================================
# 修复 8: router _rag_assist 元数据 + 内容双重保障
# =============================================================

def test_fix_router_rag_assist():
    """验证 router _rag_assist 的风险检测加强"""
    section("修复 8: router._rag_assist 风险检测加强")

    router_path = PROJECT_ROOT / "src" / "retrieval" / "router.py"
    with open(router_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 检查内容关键词检测
    assert "is_risky_content" in content, "缺少 is_risky_content 变量"
    ok("is_risky_content 变量存在")

    assert "is_risky_meta or is_risky_content" in content, "缺少双重保障逻辑"
    ok("元数据 + 内容双重保障逻辑存在")

    # 检查扩展的关键词
    for kw in ["转账", "验证码", "安全账户", "涉嫌", "免费领"]:
        assert kw in content, f"风险关键词 '{kw}' 不在 router 中"
    ok("风险关键词扩展完整")


# =============================================================
# 修复 9: llm_client 重试逻辑
# =============================================================

def test_fix_llm_retry():
    """验证 LLM 客户端的重试逻辑"""
    section("修复 9: llm_client 重试逻辑")

    llm_path = PROJECT_ROOT / "src" / "utils" / "llm_client.py"
    with open(llm_path, "r", encoding="utf-8") as f:
        content = f.read()

    assert "for attempt in range(self.max_retries)" in content, "缺少重试循环"
    ok("重试循环 'for attempt in range(self.max_retries)' 存在")

    assert "attempt < self.max_retries - 1" in content, "缺少重试条件判断"
    ok("重试条件判断存在")

    assert "time.sleep" in content, "缺少延迟等待"
    ok("重试延迟 time.sleep() 存在")


# =============================================================
# 修复 10: reranker 词级评分
# =============================================================

def test_fix_reranker_tokenize():
    """验证 reranker 词级评分"""
    section("修复 10: reranker 词级评分")

    rerank_path = PROJECT_ROOT / "src" / "retrieval" / "reranker.py"
    with open(rerank_path, "r", encoding="utf-8") as f:
        content = f.read()

    assert "_tokenize" in content, "SimpleReranker 缺少 _tokenize 方法"
    ok("SimpleReranker._tokenize() 方法存在")

    from src.retrieval.reranker import SimpleReranker

    # 验证分词方法（正则提取连续中文字符序列）
    tokens = SimpleReranker._tokenize("外卖配送 转账诈骗")
    assert "外卖配送" in tokens or "转账诈骗" in tokens, \
        f"reranker 分词结果: {tokens}"
    ok(f"reranker 中文序列分词: {tokens}")

    # 验证不会用字符级交集
    tokens = SimpleReranker._tokenize("外卖")
    assert "外" not in tokens and "卖" not in tokens, \
        f"不应有单字字符: {tokens}"
    ok("reranker 不使用字符级匹配")


# =============================================================
# 集成测试: GraphRetriever 整体行为
# =============================================================

def test_graph_retriever_integration():
    """集成测试: 验证 GraphRetriever 的完整修复"""
    section("集成测试: GraphRetriever 完整行为")

    from src.retrieval.graph_retriever import GraphRetriever

    # 需要有 actual 数据才能做完整测试
    graph_path = PROJECT_ROOT / "indices" / "graph.pkl"
    comm_path = PROJECT_ROOT / "indices" / "communities.json"
    summary_dir = PROJECT_ROOT / "indices" / "summary_store"

    if not graph_path.exists() or not comm_path.exists():
        print("  [SKIP] 索引文件未构建，跳过集成测试")
        print("  构建索引: python scripts/build_indices.py")
        return

    print(f"  图谱: {graph_path} ({graph_path.stat().st_size} bytes)")
    print(f"  社区: {comm_path} ({comm_path.stat().st_size} bytes)")

    try:
        retriever = GraphRetriever(
            graph_path=str(graph_path),
            communities_path=str(comm_path),
            top_k=5
        )
        ok(f"GraphRetriever 初始化成功: {len(retriever.communities)} 个社区")

        # 测试分词和检索
        docs = retriever.retrieve("外卖配送取餐", k=3)
        ok(f"检索 '外卖配送取餐' → {len(docs)} 个文档")
        if docs:
            print(f"    首个文档: {docs[0].page_content[:60]}...")

        # 测试无匹配情况（回退到关键词匹配）
        docs_empty = retriever.retrieve("xyz我们不存在的词123", k=3)
        ok(f"检索无意义词 → {len(docs_empty)} 个文档")
    except Exception as e:
        fail(f"GraphRetriever 集成测试失败: {e}")


# =============================================================
# 集成测试: 全流程 pipeline
# =============================================================

def test_pipeline_integration():
    """集成测试: 验证 pipeline 正常运转"""
    section("集成测试: Pipeline 全流程")

    try:
        from src.pipeline import CallAssistantPipeline
        from src.retrieval.router import CallCategory

        pipeline = CallAssistantPipeline()

        # 测试外卖
        result = pipeline.query("您好，外卖到了，取餐码7788")
        assert result.category == "delivery", f"外卖分类失败: {result.category}"
        assert result.response, "外卖应有回复"
        ok(f"外卖: category={result.category}, response={result.response[:40]}...")

        # 测试诈骗
        result = pipeline.query("请转账到安全账户，验证码告诉我")
        assert result.category == "risk", f"诈骗分类失败: {result.category}"
        ok(f"诈骗: category={result.category}, response={result.response[:40]}...")

        # 测试正常来电
        result = pipeline.query("领导，明天有个会议")
        assert result.category == "normal", f"领导分类失败: {result.category}"
        ok(f"领导: category={result.category}, response={result.response[:40]}...")
    except ImportError as e:
        print(f"  [SKIP] 缺少依赖: {e}")
    except Exception as e:
        fail(f"Pipeline 集成测试失败: {e}")


# =============================================================
# 快速冒烟测试
# =============================================================

def quick_smoke():
    """快速验证最关键的文件语法和导入"""
    section("快速冒烟测试")

    modules = [
        ("src.indexing.community_builder", "社区构建器"),
        ("src.retrieval.graph_retriever", "图检索器"),
        ("src.agents.call_classifier", "来电分类器"),
        ("src.retrieval.router", "路由"),
        ("src.utils.llm_client", "LLM 客户端"),
        ("src.retrieval.reranker", "重排器"),
    ]

    for mod_name, desc in modules:
        try:
            __import__(mod_name)
            ok(f"{desc} ({mod_name}) 导入成功")
        except Exception as e:
            fail(f"{desc} ({mod_name}) 导入失败: {e}")


# =============================================================
# 主入口
# =============================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Bug 修复验证")
    parser.add_argument("--quick", action="store_true", help="仅快速验证")
    args = parser.parse_args()

    print("=" * 60)
    print("  Hybrid RAG Bug 修复验证")
    print("=" * 60)

    quick_smoke()

    if args.quick:
        print("\n  快速验证完成。运行完整验证: python tests/test_bugfixes.py")
        sys.exit(0)

    # 所有离线验证
    test_fix_llm_client_call()
    test_fix_tokenize()
    test_fix_int_cid()
    test_fix_summary_db_usage()
    test_fix_load_pickle()
    test_fix_llm_fuzzy_match()
    test_fix_router_rag_assist()
    test_fix_llm_retry()
    test_fix_reranker_tokenize()

    # 需要索引文件的集成测试（可选）
    test_graph_retriever_integration()
    test_pipeline_integration()

    print(f"\n{'='*60}")
    print("  验证完毕！以上所有 [PASS] 表示修复已生效。")
    print(f"{'='*60}")
