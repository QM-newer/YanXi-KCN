"""
生成测试音频文件
"""

import asyncio
import edge_tts


async def generate_test_audio():
    """生成测试音频"""
    text = "你好，这是一段测试语音。请按下空格键开始录音，然后对着麦克风说话。识别结果会实时显示在屏幕上。"
    
    output_file = "test_audio.mp3"
    
    print(f"正在生成测试音频...")
    print(f"文本内容: {text}")
    
    # 使用微软晓晓的声音
    voice = "zh-CN-XiaoxiaoNeural"
    
    # 生成音频
    await edge_tts.Communicate(text, voice).save(output_file)
    
    print(f"\nOK! Test audio saved: {output_file}")
    return output_file


if __name__ == "__main__":
    asyncio.run(generate_test_audio())
