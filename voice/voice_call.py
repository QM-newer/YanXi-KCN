"""
电话语音分类 - 语音识别 + RAG来电分类

把麦克风录音或音频文件自动转为文字，并分类来电类型。

Usage:
  python voice_call.py mic              # 按键录音模式
  python voice_call.py mic --auto 10    # 自动录音10秒
  python voice_call.py file test.mp3    # 音频文件模式
  python voice_call.py                  # 交互模式（多次录音分类）
"""

import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.call_classifier import create_classifier
from voice.asr import (
    asr,
    asr_from_microphone,
    record_from_microphone,
    MIC_RECORD_SECONDS
)


def classify_text(agent, text: str) -> dict:
    """用 RAG 分类器对文字进行分类"""
    result = agent.classify(text)
    return result


def print_result(text: str, result: dict):
    """美化输出分类结果"""
    category = result.get("category", "未知")
    confidence = result.get("confidence", 0)
    action = result.get("action", "未知")
    handling = result.get("handling_response", "")
    method = result.get("method", "")

    bar = "=" * 50
    print(f"\n{bar}")
    print(f"  识别文本: {text}")
    print(f"  来电类型: {category}")
    print(f"  置信度:   {confidence:.2f}  ({method})")
    print(f"  处理动作: {action}")
    if handling:
        print(f"  处理规则: {handling}")

    if result.get("samples"):
        print(f"  参考样例:")
        for s in result["samples"][:3]:
            print(f"    [{s.get('category', '')}] {s.get('text', '')}")
    print(f"{bar}\n")


def mic_mode(auto: bool = False, duration: int = MIC_RECORD_SECONDS):
    """麦克风录音 + 分类（可多次运行）"""
    agent = create_classifier()
    print(f"\n分类器已就绪，开始识别来电...")
    print("=" * 50)

    count = 0

    while True:
        count += 1
        print(f"\n--- 第 {count} 次录音 ---")

        if auto:
            result_text = asr_from_microphone(duration=duration)
        else:
            print("按 [空格键] 开始录音，再按一次停止")
            audio_path = record_from_microphone(
                duration=duration,
                press_to_start=True,
                press_to_stop=True
            )
            if not audio_path:
                print("没有录到音频")
                continue
            result_text = asr(audio_path)
            try:
                os.unlink(audio_path)
            except:
                pass

        if result_text:
            classify_result = classify_text(agent, result_text)
            print_result(result_text, classify_result)
        else:
            print("未识别到语音内容")

        # 询问是否继续
        try:
            again = input("继续录音？(y/n) ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n退出")
            break

        if again not in {"y", "yes", ""}:
            print(f"共完成 {count} 次分类，退出")
            break


def file_mode(audio_file: str):
    """音频文件 + 分类"""
    if not os.path.exists(audio_file):
        print(f"文件不存在: {audio_file}")
        return

    agent = create_classifier()
    result_text = asr(audio_file)
    classify_result = classify_text(agent, result_text)
    print_result(result_text, classify_result)


def interactive_mode():
    """交互模式：多次录音分类"""
    agent = create_classifier()
    print("=" * 50)
    print("  电话语音分类（语音识别 + RAG 分类）")
    print("  输入 mic 开始录音，file <路径> 识别文件")
    print("  输入 exit/quit 退出")
    print("=" * 50)

    while True:
        try:
            cmd = input("\n>>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n退出")
            break

        if not cmd:
            continue
        if cmd.lower() in {"exit", "quit", "q"}:
            break

        if cmd.lower() == "mic":
            print("按空格键开始/停止录音...")
            audio_path = record_from_microphone(
                duration=MIC_RECORD_SECONDS,
                press_to_start=True,
                press_to_stop=True
            )
            if audio_path:
                try:
                    text = asr(audio_path)
                    classify_result = classify_text(agent, text)
                    print_result(text, classify_result)
                finally:
                    try:
                        os.unlink(audio_path)
                    except:
                        pass

        elif cmd.startswith("file "):
            audio_file = cmd[5:].strip()
            if os.path.exists(audio_file):
                text = asr(audio_file)
                classify_result = classify_text(agent, text)
                print_result(text, classify_result)
            else:
                print(f"文件不存在: {audio_file}")

        else:
            print("未知命令。支持: mic | file <路径> | exit")


def main():
    if len(sys.argv) < 2:
        interactive_mode()
        return

    mode = sys.argv[1].lower()

    if mode == "mic":
        auto = "--auto" in sys.argv
        duration = MIC_RECORD_SECONDS
        for arg in sys.argv[2:]:
            if arg.isdigit():
                duration = int(arg)
                break
        mic_mode(auto=auto, duration=duration)

    elif mode == "file":
        if len(sys.argv) < 3:
            print("请提供音频文件路径")
            return
        file_mode(sys.argv[2])

    else:
        print("用法:")
        print("  python voice_call.py mic              # 按键录音")
        print("  python voice_call.py mic --auto 10    # 自动录音")
        print("  python voice_call.py file test.mp3    # 音频文件")
        print("  python voice_call.py                  # 交互模式")


if __name__ == "__main__":
    main()
