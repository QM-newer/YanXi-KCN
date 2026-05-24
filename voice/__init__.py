"""
语音模块 - 语音转文字(ASR)和文字转语音(TTS)
"""

from .asr import (
    WhisperASR,
    asr,
    asr_from_microphone,
    record_from_microphone,
    clear_model_cache,
)

from .voice_call import (
    classify_text,
    print_result,
    mic_mode,
    file_mode,
    interactive_mode,
)

__all__ = [
    "WhisperASR",
    "asr",
    "asr_from_microphone",
    "record_from_microphone",
    "clear_model_cache",
    "classify_text",
    "print_result",
    "mic_mode",
    "file_mode",
    "interactive_mode",
]
