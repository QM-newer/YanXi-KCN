"""
语音识别模块 - 基于 Whisper
支持文件输入和麦克风实时录音
"""

import whisper
import numpy as np
import wave
import tempfile
import os
from typing import Optional, Union, Any, TYPE_CHECKING

if TYPE_CHECKING:
    import pyaudio

# 录音相关依赖（优先使用 sounddevice，备用 pyaudio）
SOUNDDEVICE_AVAILABLE = False
PYAUDIO_AVAILABLE = False

try:
    import sounddevice as sd
    import soundfile as sf
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    try:
        import pyaudio  # noqa: F401
        PYAUDIO_AVAILABLE = True
    except ImportError:
        pass


# ==================== 配置 ====================
# 模型选择: tiny, base, small, medium, large
# 模型越大，准确率越高，但速度越慢，资源占用越大
WHISPER_MODEL = "small"

# 语言设置: "zh"=中文, "en"=英文, None=自动检测
DEFAULT_LANGUAGE = "zh"

# 提示词，用于提升特定领域的识别准确率
DEFAULT_PROMPT = "以下是普通话的句子，用词准确。"

# 模型缓存（避免重复加载）
_model_cache = {}  # {"model_name": model}

# 麦克风录音配置
MIC_SAMPLE_RATE = 16000  # Whisper 推荐采样率
MIC_CHUNK_SIZE = 1024    # 音频块大小
MIC_CHANNELS = 1         # 单声道
MIC_RECORD_SECONDS = 15  # 录音时长上限（秒）


# ==================== 麦克风录音功能 ====================

def record_from_microphone(duration: int = MIC_RECORD_SECONDS,
                           sample_rate: int = MIC_SAMPLE_RATE,
                           channels: int = MIC_CHANNELS,
                           press_to_start: bool = False,
                           press_to_stop: bool = True) -> Optional[str]:
    """
    从麦克风录音并保存为临时音频文件

    Args:
        duration: 录音时长上限（秒）
        sample_rate: 采样率
        channels: 声道数
        press_to_start: 是否按键开始录音（True=按空格开始，False=立即开始）
        press_to_stop: 是否按键停止录音（True=按空格停止，False=到达时长自动停止）

    Returns:
        临时音频文件路径
    """
    if not SOUNDDEVICE_AVAILABLE and not PYAUDIO_AVAILABLE:
        raise ImportError(
            "需要安装录音库，请运行:\n"
            "  pip install sounddevice  # 推荐\n"
            "或\n"
            "  pip install pyaudio"
        )

    import threading
    import time
    import sys

    is_recording = False
    frames = []
    stop_flag = threading.Event()

    def input_listener():
        """监听键盘输入"""
        nonlocal is_recording
        print("\n📌 按 [空格键] 开始录音，再按一次 [空格键] 停止")
        while not stop_flag.is_set():
            try:
                # Windows 系统使用 msvcrt
                import msvcrt
                if msvcrt.kbhit():
                    key = msvcrt.getch()
                    if key == b' ' or key == b'\r':  # 空格或回车
                        if press_to_start and not is_recording:
                            is_recording = True
                            print("\n🔴 正在录音... (再按空格停止)")
                        elif is_recording:
                            stop_flag.set()
            except Exception:
                pass
            time.sleep(0.05)

    def record_audio():
        """录音线程"""
        if SOUNDDEVICE_AVAILABLE:
            stream = sd.InputStream(
                samplerate=sample_rate,
                channels=channels,
                dtype='int16'
            )
            with stream:
                while not stop_flag.is_set():
                    data, _ = stream.read(MIC_CHUNK_SIZE)
                    if is_recording:
                        frames.append(data)
                    # 超时保护
                    if len(frames) * MIC_CHUNK_SIZE / sample_rate > duration:
                        print(f"\n⏰ 达到最大录音时长 ({duration}秒)，自动停止")
                        stop_flag.set()
        elif PYAUDIO_AVAILABLE:
            import pyaudio
            p = pyaudio.PyAudio()
            stream = p.open(
                format=pyaudio.paInt16,
                channels=channels,
                rate=sample_rate,
                input=True,
                frames_per_buffer=MIC_CHUNK_SIZE
            )
            try:
                while not stop_flag.is_set():
                    data = stream.read(MIC_CHUNK_SIZE)
                    if is_recording:
                        frames.append(data)
                    if len(frames) * MIC_CHUNK_SIZE / sample_rate > duration:
                        stop_flag.set()
            finally:
                stream.stop_stream()
                stream.close()
                p.terminate()

    print("\n" + "="*50)
    print("🎙️ 麦克风录音模式")
    print("="*50)

    if press_to_start:
        print(f"📌 录音时长上限: {duration} 秒")
    else:
        print(f"⏱️ 录音时长: {duration} 秒")

    # 启动键盘监听线程
    listener = threading.Thread(target=input_listener)
    listener.daemon = True
    listener.start()

    # 启动录音线程
    record_thread = threading.Thread(target=record_audio)
    record_thread.start()

    if press_to_start:
        # 等待开始信号
        while not is_recording and not stop_flag.is_set():
            time.sleep(0.1)
        # 等待停止信号
        while not stop_flag.is_set():
            time.sleep(0.1)
    else:
        # 立即开始，等待时长
        is_recording = True
        print("\n🔴 正在录音...")
        try:
            time.sleep(duration)
        except KeyboardInterrupt:
            print("\n   用户中断")
        stop_flag.set()

    record_thread.join()

    if not frames:
        print("❌ 没有录到任何音频")
        return None

    # 保存为临时 WAV 文件
    audio_data = np.concatenate(frames) if frames else np.array([])
    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    
    if SOUNDDEVICE_AVAILABLE:
        sf.write(temp_file.name, audio_data, sample_rate)
    else:
        import wave as wv
        with wv.open(temp_file.name, "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_data.tobytes())

    duration_sec = len(frames) * MIC_CHUNK_SIZE / sample_rate
    print(f"\n✅ 录音完成！时长: {duration_sec:.1f}秒")
    return temp_file.name


def asr_from_microphone(duration: int = MIC_RECORD_SECONDS,
                        language: str = DEFAULT_LANGUAGE,
                        model_name: str = WHISPER_MODEL,
                        prompt: str = DEFAULT_PROMPT) -> str:
    """
    从麦克风录音并实时识别

    Args:
        duration: 录音时长上限（秒）
        language: 语言设置
        model_name: Whisper 模型大小
        prompt: 提示词

    Returns:
        识别的文本内容
    """
    # 录音
    audio_path = record_from_microphone(duration=duration, press_to_start=False)

    try:
        # 识别
        result = asr(audio_path, language, model_name, prompt)
        return result
    finally:
        # 清理临时文件
        if audio_path is not None:
            try:
                os.unlink(audio_path)
            except OSError:
                pass


# ==================== 文件语音识别 ====================

# ----------------------
# 语音识别（精准版）
# ----------------------
def clear_model_cache():
    """清空模型缓存，释放内存"""
    global _model_cache
    _model_cache.clear()
    print("Model cache cleared")


def asr(audio_path: Any,
        language: str = DEFAULT_LANGUAGE,
        model_name: str = WHISPER_MODEL,
        prompt: str = DEFAULT_PROMPT,
        fp16: bool = False) -> str:
    """
    使用 Whisper 进行语音识别

    Args:
        audio_path: 音频文件路径（支持 mp3, wav, m4a, flac 等格式）
        language: 强制指定语言，"zh"=中文，None=自动检测
        model_name: Whisper 模型大小
        prompt: 提示词，用于提升特定领域准确率
        fp16: 是否使用半精度（需要 NVIDIA GPU）

    Returns:
        识别的文本内容
    """
    print(f"\nRecording: {audio_path}")

    # 使用缓存的模型，避免重复加载
    global _model_cache
    if model_name not in _model_cache:
        print(f"Loading Whisper model: {model_name}...")
        _model_cache[model_name] = whisper.load_model(model_name)
    model = _model_cache[model_name]

    # 准备参数
    transcribe_options = {
        "fp16": fp16,
        "temperature": 0.0,  # 降低随机性，更精准
    }

    # 设置语言
    if language:
        transcribe_options["language"] = language
        transcribe_options["initial_prompt"] = prompt

    # 识别
    result = model.transcribe(audio_path, **transcribe_options)

    text = result["text"].strip()
    print(f"✅ 识别结果：{text}")
    return text


# ----------------------
# 语音识别类（可复用模型实例）
# ----------------------
class WhisperASR:
    """语音识别器类，复用模型实例提升效率"""

    def __init__(self,
                 model_name: str = WHISPER_MODEL,
                 language: str = DEFAULT_LANGUAGE,
                 prompt: str = DEFAULT_PROMPT,
                 fp16: bool = False):
        """
        初始化语音识别器

        Args:
            model_name: Whisper 模型大小
            language: 默认语言
            prompt: 默认提示词
            fp16: 是否使用半精度
        """
        self.model_name = model_name
        self.language = language
        self.prompt = prompt
        self.fp16 = fp16
        self._model = None

    @property
    def model(self):
        """懒加载模型（使用全局缓存）"""
        if self._model is None:
            if self.model_name not in _model_cache:
                print(f"Loading Whisper model: {self.model_name}...")
                _model_cache[self.model_name] = whisper.load_model(self.model_name)
            self._model = _model_cache[self.model_name]
        return self._model

    def recognize(self,
                  audio_path: Any,
                  language: Optional[str] = None,
                  prompt: Optional[str] = None) -> str:
        """
        识别单个音频文件

        Args:
            audio_path: 音频文件路径
            language: 覆盖默认语言
            prompt: 覆盖默认提示词

        Returns:
            识别的文本内容
        """
        lang = language or self.language
        pmt = prompt or self.prompt

        transcribe_options = {
            "fp16": self.fp16,
            "temperature": 0.0,
        }

        if lang:
            transcribe_options["language"] = lang
            transcribe_options["initial_prompt"] = pmt

        result = self.model.transcribe(audio_path, **transcribe_options)
        return result["text"].strip()

    def recognize_batch(self, audio_paths: list[str]) -> list[str]:
        """
        批量识别多个音频文件

        Args:
            audio_paths: 音频文件路径列表

        Returns:
            识别结果列表
        """
        results = []
        for path in audio_paths:
            try:
                text = self.recognize(path)
                results.append(text)
            except Exception as e:
                print(f"❌ 识别失败 {path}: {e}")
                results.append("")
        return results

    def unload(self):
        """释放模型内存"""
        if self._model is not None:
            del self._model
            self._model = None
            print("🗑️ Whisper 模型已卸载")


# ----------------------
# 便捷入口
# ----------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("用法:")
        print("  麦克风模式（按键录音）: python asr.py mic")
        print("  麦克风模式（定时录音）: python asr.py mic --auto [秒数]")
        print("  文件模式:              python asr.py file <音频文件>")
        print("\n示例:")
        print("  python asr.py mic           # 按空格开始/停止录音")
        print("  python asr.py mic --auto 5  # 自动录音5秒")
        print("  python asr.py file test.mp3")
        sys.exit(1)

    mode = sys.argv[1].lower()

    if mode == "mic":
        auto_mode = "--auto" in sys.argv
        duration = MIC_RECORD_SECONDS
        
        # 提取数字参数
        for arg in sys.argv[2:]:
            if arg.isdigit():
                duration = int(arg)
                break

        print(f"\n{'='*50}")
        print(f"🎙️ 麦克风语音识别模式")
        print(f"{'='*50}")
        
        if auto_mode:
            print(f"📌 自动录音模式（{duration}秒后自动停止）")
            result = asr_from_microphone(duration=duration)
        else:
            print(f"📌 按键录音模式（空格开始/停止）")
            # 按键模式
            audio_path = record_from_microphone(
                duration=duration,
                press_to_start=True,
                press_to_stop=True
            )
            if audio_path:
                result = asr(audio_path)
                try:
                    os.unlink(audio_path)
                except OSError:
                    pass
            else:
                result = ""
        
        print(f"\n🎯 最终识别结果: {result}")

    elif mode == "file":
        if len(sys.argv) < 3:
            print("错误: 请提供音频文件路径")
            print("示例: python asr.py file test.mp3")
            sys.exit(1)
        audio_file = sys.argv[2]
        result = asr(audio_file)
        print(f"\n🎯 最终识别结果: {result}")

    else:
        print(f"未知模式: {mode}")
        print("请使用 'mic' 或 'file' 模式")
