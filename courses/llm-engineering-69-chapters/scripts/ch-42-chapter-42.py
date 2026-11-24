#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — معالجة الصوت والكلام — Whisper وTTS وSpeech-to-Speech
Large Language Model Engineering — Whisper

Book slug : llm-engineering-69-chapters
Chapter   : ch-42-chapter-42 (lesson 42)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/llm-engineering-69-chapters/code/ch-42-chapter-42.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — معالجة الصوت والكلام — Whisper وTTS وSpeech-to-Speech
Large Language Model Engineering — Whisper

Book slug : llm-engineering-69-chapters
Chapter   : ch-42-chapter-42 (lesson 42)
Code files: 4 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/llm-engineering-69-chapters/code/ch-42-chapter-42.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — معالجة الصوت والكلام — Whisper وTTS وSpeech-to-Speech
Large Language Model Engineering — Whisper

Book slug : llm-engineering-69-chapters
Chapter   : ch-42-chapter-42 (lesson 42)
Code files: 3 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book_code/notebooks/chapters/chapter_42.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 42: الفصل 42: معالجة الصوت والكلام — Whisper وTTS وSpeech-to-Speech."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_audio() -> None:
    samples, frame_size = ([0.1, 0.2, 0.3, 0.2, 0.0, 0.1, 0.4, 0.5], 4)
    frames = [samples[i:i + frame_size] for i in range(0, len(samples), frame_size)]
    reference, hypothesis = ('the cat sat on mat'.split(), 'the cat is on mat'.split())
    previous = list(range(len(reference) + 1))
    for i, word in enumerate(hypothesis, 1):
        current = [i] + [0] * len(reference)
        for j, target in enumerate(reference, 1):
            current[j] = min(current[j - 1] + 1, previous[j] + 1, previous[j - 1] + (word != target))
        previous = current
    print('audio frames:', frames)
    print('word error rate:', round(previous[-1] / len(reference), 3))

if __name__ == "__main__":
    print("Chapter 42 lab: Frame audio and measure word error rate")
    run_audio()

# ────────────────────────────────────────────────────────────────────────
# [002] book/chapters/part-08-multimodal-models/chapter-42.md
# ────────────────────────────────────────────────────────────────────────
"""
نظام محادثة صوتية كامل: Whisper (ASR) + LLM + TTS
يوضّح معمارية Cascade التقليدية مع قياس زمن الاستجابة لكل مرحلة
"""

import time
from dataclasses import dataclass

import numpy as np
import torch
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    AutoModelForCausalLM,
    AutoTokenizer,
    SpeechT5Processor,
    SpeechT5ForTextToSpeech,
    SpeechT5HifiGan,
)
from datasets import load_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------
# 1) مرحلة Speech-to-Text عبر Whisper
# ---------------------------------------------------------------------
class SpeechToTextEngine:
    def __init__(self, model_name: str = "openai/whisper-small"):
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)
        self.model.eval()

    @torch.no_grad()
    def transcribe(
        self, audio_array: np.ndarray, sampling_rate: int = 16000,
        language: str = "arabic", task: str = "transcribe",
    ) -> str:
        """
        يحوّل موجة صوتية خام إلى نص.
        task يمكن أن تكون "transcribe" (نفس اللغة) أو "translate" (إلى الإنجليزية).
        """
        input_features = self.processor(
            audio_array, sampling_rate=sampling_rate, return_tensors="pt"
        ).input_features.to(device)

        # تحديد رموز المهمة الخاصة بـ Whisper (اللغة + نوع المهمة)
        forced_decoder_ids = self.processor.get_decoder_prompt_ids(
            language=language, task=task
        )

        predicted_ids = self.model.generate(
            input_features, forced_decoder_ids=forced_decoder_ids, max_new_tokens=256
        )
        transcription = self.processor.batch_decode(
            predicted_ids, skip_special_tokens=True
        )[0]
        return transcription.strip()


# ---------------------------------------------------------------------
# 2) مرحلة معالجة اللغة (أي LLM — نستخدم نموذجًا صغيرًا للتوضيح)
# ---------------------------------------------------------------------
class LanguageProcessingEngine:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float32
        ).to(device)
        self.model.eval()

    @torch.no_grad()
    def generate_response(self, user_text: str, max_new_tokens: int = 60) -> str:
        messages = [
            {"role": "system", "content": "أنت مساعد صوتي مفيد وموجز. أجب بجملة قصيرة."},
            {"role": "user", "content": user_text},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        response = self.tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        return response.strip()


# ---------------------------------------------------------------------
# 3) مرحلة Text-to-Speech عبر SpeechT5 (Acoustic Model + Vocoder)
# ---------------------------------------------------------------------
class TextToSpeechEngine:
    def __init__(self, model_name: str = "microsoft/speecht5_tts"):
        self.processor = SpeechT5Processor.from_pretrained(model_name)
        self.model = SpeechT5ForTextToSpeech.from_pretrained(model_name).to(device)
        self.vocoder = SpeechT5HifiGan.from_pretrained(
            "microsoft/speecht5_hifigan"
        ).to(device)
        self.model.eval()
        self.vocoder.eval()

        # تضمين المتحدث (Speaker Embedding) — يحدد هوية/نبرة الصوت المُولَّد
        embeddings_dataset = load_dataset(
            "Matthijs/cmu-arctic-xvectors", split="validation"
        )
        self.speaker_embeddings = torch.tensor(
            embeddings_dataset[7306]["xvector"]
        ).unsqueeze(0).to(device)

    @torch.no_grad()
    def synthesize(self, text: str) -> np.ndarray:
        """يحوّل نصًا إلى موجة صوتية خام جاهزة للتشغيل."""
        inputs = self.processor(text=text, return_tensors="pt").to(device)

        # المرحلة 1: Text -> Mel Spectrogram -> المرحلة 2: Spectrogram -> Waveform
        speech = self.model.generate_speech(
            inputs["input_ids"], self.speaker_embeddings, vocoder=self.vocoder
        )
        return speech.cpu().numpy()


# ---------------------------------------------------------------------
# 4) تجميع النظام الكامل (Cascade Pipeline) مع قياس الزمن
# ---------------------------------------------------------------------
@dataclass
class PipelineTimings:
    asr_seconds: float
    llm_seconds: float
    tts_seconds: float

    @property
    def total_seconds(self) -> float:
        return self.asr_seconds + self.llm_seconds + self.tts_seconds


class VoiceAssistantPipeline:
    def __init__(self):
        print("تحميل النماذج الثلاثة... (قد يستغرق بعض الوقت أول مرة)")
        self.asr_engine = SpeechToTextEngine()
        self.llm_engine = LanguageProcessingEngine()
        self.tts_engine = TextToSpeechEngine()
        print("النظام جاهز.")

    def process(
        self, audio_array: np.ndarray, sampling_rate: int = 16000
    ) -> tuple[np.ndarray, PipelineTimings, dict]:
        # المرحلة 1: الصوت -> نص
        t0 = time.perf_counter()
        user_text = self.asr_engine.transcribe(
            audio_array, sampling_rate=sampling_rate, language="arabic"
        )
        t1 = time.perf_counter()

        # المرحلة 2: نص -> استجابة نصية (LLM)
        response_text = self.llm_engine.generate_response(user_text)
        t2 = time.perf_counter()

        # المرحلة 3: استجابة نصية -> صوت
        response_audio = self.tts_engine.synthesize(response_text)
        t3 = time.perf_counter()

        timings = PipelineTimings(
            asr_seconds=t1 - t0, llm_seconds=t2 - t1, tts_seconds=t3 - t2
        )
        debug_info = {"user_text": user_text, "response_text": response_text}
        return response_audio, timings, debug_info


# ---------------------------------------------------------------------
# 5) التشغيل الكامل (مثال توضيحي بصوت مُسجَّل مسبقًا)
# ---------------------------------------------------------------------
if __name__ == "__main__":
    import soundfile as sf

    pipeline = VoiceAssistantPipeline()

    # مثال: تحميل ملف صوتي (استبدله بمسار ملف حقيقي يحتوي سؤالًا بالعربية)
    audio_array, sampling_rate = sf.read("user_question.wav")
    if audio_array.ndim > 1:
        audio_array = audio_array.mean(axis=1)  # تحويل Stereo إلى Mono

    response_audio, timings, debug_info = pipeline.process(audio_array, sampling_rate)

    print(f"\n📝 النص المُفرَّغ (ASR): {debug_info['user_text']}")
    print(f"🤖 استجابة النموذج (LLM): {debug_info['response_text']}")
    print(f"\n⏱️  زمن ASR: {timings.asr_seconds:.3f}s")
    print(f"⏱️  زمن LLM: {timings.llm_seconds:.3f}s")
    print(f"⏱️  زمن TTS: {timings.tts_seconds:.3f}s")
    print(f"⏱️  الزمن الكلي: {timings.total_seconds:.3f}s")

    # حفظ الصوت المُولَّد (16kHz افتراضيًا لـ SpeechT5)
    sf.write("assistant_response.wav", response_audio, samplerate=16000)
    print("\n✅ تم حفظ الرد الصوتي في assistant_response.wav")

# ────────────────────────────────────────────────────────────────────────
# [003] book/chapters/part-08-multimodal-models/chapter-42.md
# ────────────────────────────────────────────────────────────────────────
def stream_llm_response(self, user_text: str):
    """
    استخدم TextIteratorStreamer من transformers لتوليد الاستجابة
    تدريجيًا جملة بجملة، بدلًا من انتظار الاستجابة الكاملة.
    """
    ...

def synthesize_streaming(self, text_stream):
    """
    فور اكتمال أول جملة من الـ LLM (كشف نقطة/علامة استفهام)،
    ابدأ توليد صوتها فورًا بالتوازي مع توليد باقي الاستجابة النصية.
    """
    ...

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-42-chapter-42.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 42: الفصل 42: معالجة الصوت والكلام — Whisper وTTS وSpeech-to-Speech."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_audio() -> None:
    samples, frame_size = ([0.1, 0.2, 0.3, 0.2, 0.0, 0.1, 0.4, 0.5], 4)
    frames = [samples[i:i + frame_size] for i in range(0, len(samples), frame_size)]
    reference, hypothesis = ('the cat sat on mat'.split(), 'the cat is on mat'.split())
    previous = list(range(len(reference) + 1))
    for i, word in enumerate(hypothesis, 1):
        current = [i] + [0] * len(reference)
        for j, target in enumerate(reference, 1):
            current[j] = min(current[j - 1] + 1, previous[j] + 1, previous[j - 1] + (word != target))
        previous = current
    print('audio frames:', frames)
    print('word error rate:', round(previous[-1] / len(reference), 3))

if __name__ == "__main__":
    print("Chapter 42 lab: Frame audio and measure word error rate")
    run_audio()

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-42-chapter-42.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
"""
نظام محادثة صوتية كامل: Whisper (ASR) + LLM + TTS
يوضّح معمارية Cascade التقليدية مع قياس زمن الاستجابة لكل مرحلة
"""

import time
from dataclasses import dataclass

import numpy as np
import torch
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    AutoModelForCausalLM,
    AutoTokenizer,
    SpeechT5Processor,
    SpeechT5ForTextToSpeech,
    SpeechT5HifiGan,
)
from datasets import load_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------
# 1) مرحلة Speech-to-Text عبر Whisper
# ---------------------------------------------------------------------
class SpeechToTextEngine:
    def __init__(self, model_name: str = "openai/whisper-small"):
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)
        self.model.eval()

    @torch.no_grad()
    def transcribe(
        self, audio_array: np.ndarray, sampling_rate: int = 16000,
        language: str = "arabic", task: str = "transcribe",
    ) -> str:
        """
        يحوّل موجة صوتية خام إلى نص.
        task يمكن أن تكون "transcribe" (نفس اللغة) أو "translate" (إلى الإنجليزية).
        """
        input_features = self.processor(
            audio_array, sampling_rate=sampling_rate, return_tensors="pt"
        ).input_features.to(device)

        # تحديد رموز المهمة الخاصة بـ Whisper (اللغة + نوع المهمة)
        forced_decoder_ids = self.processor.get_decoder_prompt_ids(
            language=language, task=task
        )

        predicted_ids = self.model.generate(
            input_features, forced_decoder_ids=forced_decoder_ids, max_new_tokens=256
        )
        transcription = self.processor.batch_decode(
            predicted_ids, skip_special_tokens=True
        )[0]
        return transcription.strip()


# ---------------------------------------------------------------------
# 2) مرحلة معالجة اللغة (أي LLM — نستخدم نموذجًا صغيرًا للتوضيح)
# ---------------------------------------------------------------------
class LanguageProcessingEngine:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float32
        ).to(device)
        self.model.eval()

    @torch.no_grad()
    def generate_response(self, user_text: str, max_new_tokens: int = 60) -> str:
        messages = [
            {"role": "system", "content": "أنت مساعد صوتي مفيد وموجز. أجب بجملة قصيرة."},
            {"role": "user", "content": user_text},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        response = self.tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        return response.strip()


# ---------------------------------------------------------------------
# 3) مرحلة Text-to-Speech عبر SpeechT5 (Acoustic Model + Vocoder)
# ---------------------------------------------------------------------
class TextToSpeechEngine:
    def __init__(self, model_name: str = "microsoft/speecht5_tts"):
        self.processor = SpeechT5Processor.from_pretrained(model_name)
        self.model = SpeechT5ForTextToSpeech.from_pretrained(model_name).to(device)
        self.vocoder = SpeechT5HifiGan.from_pretrained(
            "microsoft/speecht5_hifigan"
        ).to(device)
        self.model.eval()
        self.vocoder.eval()

        # تضمين المتحدث (Speaker Embedding) — يحدد هوية/نبرة الصوت المُولَّد
        embeddings_dataset = load_dataset(
            "Matthijs/cmu-arctic-xvectors", split="validation"
        )
        self.speaker_embeddings = torch.tensor(
            embeddings_dataset[7306]["xvector"]
        ).unsqueeze(0).to(device)

    @torch.no_grad()
    def synthesize(self, text: str) -> np.ndarray:
        """يحوّل نصًا إلى موجة صوتية خام جاهزة للتشغيل."""
        inputs = self.processor(text=text, return_tensors="pt").to(device)

        # المرحلة 1: Text -> Mel Spectrogram -> المرحلة 2: Spectrogram -> Waveform
        speech = self.model.generate_speech(
            inputs["input_ids"], self.speaker_embeddings, vocoder=self.vocoder
        )
        return speech.cpu().numpy()


# ---------------------------------------------------------------------
# 4) تجميع النظام الكامل (Cascade Pipeline) مع قياس الزمن
# ---------------------------------------------------------------------
@dataclass
class PipelineTimings:
    asr_seconds: float
    llm_seconds: float
    tts_seconds: float

    @property
    def total_seconds(self) -> float:
        return self.asr_seconds + self.llm_seconds + self.tts_seconds


class VoiceAssistantPipeline:
    def __init__(self):
        print("تحميل النماذج الثلاثة... (قد يستغرق بعض الوقت أول مرة)")
        self.asr_engine = SpeechToTextEngine()
        self.llm_engine = LanguageProcessingEngine()
        self.tts_engine = TextToSpeechEngine()
        print("النظام جاهز.")

    def process(
        self, audio_array: np.ndarray, sampling_rate: int = 16000
    ) -> tuple[np.ndarray, PipelineTimings, dict]:
        # المرحلة 1: الصوت -> نص
        t0 = time.perf_counter()
        user_text = self.asr_engine.transcribe(
            audio_array, sampling_rate=sampling_rate, language="arabic"
        )
        t1 = time.perf_counter()

        # المرحلة 2: نص -> استجابة نصية (LLM)
        response_text = self.llm_engine.generate_response(user_text)
        t2 = time.perf_counter()

        # المرحلة 3: استجابة نصية -> صوت
        response_audio = self.tts_engine.synthesize(response_text)
        t3 = time.perf_counter()

        timings = PipelineTimings(
            asr_seconds=t1 - t0, llm_seconds=t2 - t1, tts_seconds=t3 - t2
        )
        debug_info = {"user_text": user_text, "response_text": response_text}
        return response_audio, timings, debug_info


# ---------------------------------------------------------------------
# 5) التشغيل الكامل (مثال توضيحي بصوت مُسجَّل مسبقًا)
# ---------------------------------------------------------------------
if __name__ == "__main__":
    import soundfile as sf

    pipeline = VoiceAssistantPipeline()

    # مثال: تحميل ملف صوتي (استبدله بمسار ملف حقيقي يحتوي سؤالًا بالعربية)
    audio_array, sampling_rate = sf.read("user_question.wav")
    if audio_array.ndim > 1:
        audio_array = audio_array.mean(axis=1)  # تحويل Stereo إلى Mono

    response_audio, timings, debug_info = pipeline.process(audio_array, sampling_rate)

    print(f"\n📝 النص المُفرَّغ (ASR): {debug_info['user_text']}")
    print(f"🤖 استجابة النموذج (LLM): {debug_info['response_text']}")
    print(f"\n⏱️  زمن ASR: {timings.asr_seconds:.3f}s")
    print(f"⏱️  زمن LLM: {timings.llm_seconds:.3f}s")
    print(f"⏱️  زمن TTS: {timings.tts_seconds:.3f}s")
    print(f"⏱️  الزمن الكلي: {timings.total_seconds:.3f}s")

    # حفظ الصوت المُولَّد (16kHz افتراضيًا لـ SpeechT5)
    sf.write("assistant_response.wav", response_audio, samplerate=16000)
    print("\n✅ تم حفظ الرد الصوتي في assistant_response.wav")

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-42-chapter-42.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
def stream_llm_response(self, user_text: str):
    """
    استخدم TextIteratorStreamer من transformers لتوليد الاستجابة
    تدريجيًا جملة بجملة، بدلًا من انتظار الاستجابة الكاملة.
    """
    ...

def synthesize_streaming(self, text_stream):
    """
    فور اكتمال أول جملة من الـ LLM (كشف نقطة/علامة استفهام)،
    ابدأ توليد صوتها فورًا بالتوازي مع توليد باقي الاستجابة النصية.
    """
    ...

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-42-chapter-42.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — معالجة الصوت والكلام — Whisper وTTS وSpeech-to-Speech
Large Language Model Engineering — Whisper

Book slug : llm-engineering-69-chapters
Chapter   : ch-42-chapter-42 (lesson 42)
Code files: 3 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book_code/notebooks/chapters/chapter_42.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 42: الفصل 42: معالجة الصوت والكلام — Whisper وTTS وSpeech-to-Speech."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_audio() -> None:
    samples, frame_size = ([0.1, 0.2, 0.3, 0.2, 0.0, 0.1, 0.4, 0.5], 4)
    frames = [samples[i:i + frame_size] for i in range(0, len(samples), frame_size)]
    reference, hypothesis = ('the cat sat on mat'.split(), 'the cat is on mat'.split())
    previous = list(range(len(reference) + 1))
    for i, word in enumerate(hypothesis, 1):
        current = [i] + [0] * len(reference)
        for j, target in enumerate(reference, 1):
            current[j] = min(current[j - 1] + 1, previous[j] + 1, previous[j - 1] + (word != target))
        previous = current
    print('audio frames:', frames)
    print('word error rate:', round(previous[-1] / len(reference), 3))

if __name__ == "__main__":
    print("Chapter 42 lab: Frame audio and measure word error rate")
    run_audio()

# ────────────────────────────────────────────────────────────────────────
# [002] book/chapters/part-08-multimodal-models/chapter-42.md
# ────────────────────────────────────────────────────────────────────────
"""
نظام محادثة صوتية كامل: Whisper (ASR) + LLM + TTS
يوضّح معمارية Cascade التقليدية مع قياس زمن الاستجابة لكل مرحلة
"""

import time
from dataclasses import dataclass

import numpy as np
import torch
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    AutoModelForCausalLM,
    AutoTokenizer,
    SpeechT5Processor,
    SpeechT5ForTextToSpeech,
    SpeechT5HifiGan,
)
from datasets import load_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------
# 1) مرحلة Speech-to-Text عبر Whisper
# ---------------------------------------------------------------------
class SpeechToTextEngine:
    def __init__(self, model_name: str = "openai/whisper-small"):
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)
        self.model.eval()

    @torch.no_grad()
    def transcribe(
        self, audio_array: np.ndarray, sampling_rate: int = 16000,
        language: str = "arabic", task: str = "transcribe",
    ) -> str:
        """
        يحوّل موجة صوتية خام إلى نص.
        task يمكن أن تكون "transcribe" (نفس اللغة) أو "translate" (إلى الإنجليزية).
        """
        input_features = self.processor(
            audio_array, sampling_rate=sampling_rate, return_tensors="pt"
        ).input_features.to(device)

        # تحديد رموز المهمة الخاصة بـ Whisper (اللغة + نوع المهمة)
        forced_decoder_ids = self.processor.get_decoder_prompt_ids(
            language=language, task=task
        )

        predicted_ids = self.model.generate(
            input_features, forced_decoder_ids=forced_decoder_ids, max_new_tokens=256
        )
        transcription = self.processor.batch_decode(
            predicted_ids, skip_special_tokens=True
        )[0]
        return transcription.strip()


# ---------------------------------------------------------------------
# 2) مرحلة معالجة اللغة (أي LLM — نستخدم نموذجًا صغيرًا للتوضيح)
# ---------------------------------------------------------------------
class LanguageProcessingEngine:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float32
        ).to(device)
        self.model.eval()

    @torch.no_grad()
    def generate_response(self, user_text: str, max_new_tokens: int = 60) -> str:
        messages = [
            {"role": "system", "content": "أنت مساعد صوتي مفيد وموجز. أجب بجملة قصيرة."},
            {"role": "user", "content": user_text},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        response = self.tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        return response.strip()


# ---------------------------------------------------------------------
# 3) مرحلة Text-to-Speech عبر SpeechT5 (Acoustic Model + Vocoder)
# ---------------------------------------------------------------------
class TextToSpeechEngine:
    def __init__(self, model_name: str = "microsoft/speecht5_tts"):
        self.processor = SpeechT5Processor.from_pretrained(model_name)
        self.model = SpeechT5ForTextToSpeech.from_pretrained(model_name).to(device)
        self.vocoder = SpeechT5HifiGan.from_pretrained(
            "microsoft/speecht5_hifigan"
        ).to(device)
        self.model.eval()
        self.vocoder.eval()

        # تضمين المتحدث (Speaker Embedding) — يحدد هوية/نبرة الصوت المُولَّد
        embeddings_dataset = load_dataset(
            "Matthijs/cmu-arctic-xvectors", split="validation"
        )
        self.speaker_embeddings = torch.tensor(
            embeddings_dataset[7306]["xvector"]
        ).unsqueeze(0).to(device)

    @torch.no_grad()
    def synthesize(self, text: str) -> np.ndarray:
        """يحوّل نصًا إلى موجة صوتية خام جاهزة للتشغيل."""
        inputs = self.processor(text=text, return_tensors="pt").to(device)

        # المرحلة 1: Text -> Mel Spectrogram -> المرحلة 2: Spectrogram -> Waveform
        speech = self.model.generate_speech(
            inputs["input_ids"], self.speaker_embeddings, vocoder=self.vocoder
        )
        return speech.cpu().numpy()


# ---------------------------------------------------------------------
# 4) تجميع النظام الكامل (Cascade Pipeline) مع قياس الزمن
# ---------------------------------------------------------------------
@dataclass
class PipelineTimings:
    asr_seconds: float
    llm_seconds: float
    tts_seconds: float

    @property
    def total_seconds(self) -> float:
        return self.asr_seconds + self.llm_seconds + self.tts_seconds


class VoiceAssistantPipeline:
    def __init__(self):
        print("تحميل النماذج الثلاثة... (قد يستغرق بعض الوقت أول مرة)")
        self.asr_engine = SpeechToTextEngine()
        self.llm_engine = LanguageProcessingEngine()
        self.tts_engine = TextToSpeechEngine()
        print("النظام جاهز.")

    def process(
        self, audio_array: np.ndarray, sampling_rate: int = 16000
    ) -> tuple[np.ndarray, PipelineTimings, dict]:
        # المرحلة 1: الصوت -> نص
        t0 = time.perf_counter()
        user_text = self.asr_engine.transcribe(
            audio_array, sampling_rate=sampling_rate, language="arabic"
        )
        t1 = time.perf_counter()

        # المرحلة 2: نص -> استجابة نصية (LLM)
        response_text = self.llm_engine.generate_response(user_text)
        t2 = time.perf_counter()

        # المرحلة 3: استجابة نصية -> صوت
        response_audio = self.tts_engine.synthesize(response_text)
        t3 = time.perf_counter()

        timings = PipelineTimings(
            asr_seconds=t1 - t0, llm_seconds=t2 - t1, tts_seconds=t3 - t2
        )
        debug_info = {"user_text": user_text, "response_text": response_text}
        return response_audio, timings, debug_info


# ---------------------------------------------------------------------
# 5) التشغيل الكامل (مثال توضيحي بصوت مُسجَّل مسبقًا)
# ---------------------------------------------------------------------
if __name__ == "__main__":
    import soundfile as sf

    pipeline = VoiceAssistantPipeline()

    # مثال: تحميل ملف صوتي (استبدله بمسار ملف حقيقي يحتوي سؤالًا بالعربية)
    audio_array, sampling_rate = sf.read("user_question.wav")
    if audio_array.ndim > 1:
        audio_array = audio_array.mean(axis=1)  # تحويل Stereo إلى Mono

    response_audio, timings, debug_info = pipeline.process(audio_array, sampling_rate)

    print(f"\n📝 النص المُفرَّغ (ASR): {debug_info['user_text']}")
    print(f"🤖 استجابة النموذج (LLM): {debug_info['response_text']}")
    print(f"\n⏱️  زمن ASR: {timings.asr_seconds:.3f}s")
    print(f"⏱️  زمن LLM: {timings.llm_seconds:.3f}s")
    print(f"⏱️  زمن TTS: {timings.tts_seconds:.3f}s")
    print(f"⏱️  الزمن الكلي: {timings.total_seconds:.3f}s")

    # حفظ الصوت المُولَّد (16kHz افتراضيًا لـ SpeechT5)
    sf.write("assistant_response.wav", response_audio, samplerate=16000)
    print("\n✅ تم حفظ الرد الصوتي في assistant_response.wav")

# ────────────────────────────────────────────────────────────────────────
# [003] book/chapters/part-08-multimodal-models/chapter-42.md
# ────────────────────────────────────────────────────────────────────────
def stream_llm_response(self, user_text: str):
    """
    استخدم TextIteratorStreamer من transformers لتوليد الاستجابة
    تدريجيًا جملة بجملة، بدلًا من انتظار الاستجابة الكاملة.
    """
    ...

def synthesize_streaming(self, text_stream):
    """
    فور اكتمال أول جملة من الـ LLM (كشف نقطة/علامة استفهام)،
    ابدأ توليد صوتها فورًا بالتوازي مع توليد باقي الاستجابة النصية.
    """
    ...

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-42-chapter-42.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 42: الفصل 42: معالجة الصوت والكلام — Whisper وTTS وSpeech-to-Speech."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_audio() -> None:
    samples, frame_size = ([0.1, 0.2, 0.3, 0.2, 0.0, 0.1, 0.4, 0.5], 4)
    frames = [samples[i:i + frame_size] for i in range(0, len(samples), frame_size)]
    reference, hypothesis = ('the cat sat on mat'.split(), 'the cat is on mat'.split())
    previous = list(range(len(reference) + 1))
    for i, word in enumerate(hypothesis, 1):
        current = [i] + [0] * len(reference)
        for j, target in enumerate(reference, 1):
            current[j] = min(current[j - 1] + 1, previous[j] + 1, previous[j - 1] + (word != target))
        previous = current
    print('audio frames:', frames)
    print('word error rate:', round(previous[-1] / len(reference), 3))

if __name__ == "__main__":
    print("Chapter 42 lab: Frame audio and measure word error rate")
    run_audio()

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-42-chapter-42.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
"""
نظام محادثة صوتية كامل: Whisper (ASR) + LLM + TTS
يوضّح معمارية Cascade التقليدية مع قياس زمن الاستجابة لكل مرحلة
"""

import time
from dataclasses import dataclass

import numpy as np
import torch
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    AutoModelForCausalLM,
    AutoTokenizer,
    SpeechT5Processor,
    SpeechT5ForTextToSpeech,
    SpeechT5HifiGan,
)
from datasets import load_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------
# 1) مرحلة Speech-to-Text عبر Whisper
# ---------------------------------------------------------------------
class SpeechToTextEngine:
    def __init__(self, model_name: str = "openai/whisper-small"):
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)
        self.model.eval()

    @torch.no_grad()
    def transcribe(
        self, audio_array: np.ndarray, sampling_rate: int = 16000,
        language: str = "arabic", task: str = "transcribe",
    ) -> str:
        """
        يحوّل موجة صوتية خام إلى نص.
        task يمكن أن تكون "transcribe" (نفس اللغة) أو "translate" (إلى الإنجليزية).
        """
        input_features = self.processor(
            audio_array, sampling_rate=sampling_rate, return_tensors="pt"
        ).input_features.to(device)

        # تحديد رموز المهمة الخاصة بـ Whisper (اللغة + نوع المهمة)
        forced_decoder_ids = self.processor.get_decoder_prompt_ids(
            language=language, task=task
        )

        predicted_ids = self.model.generate(
            input_features, forced_decoder_ids=forced_decoder_ids, max_new_tokens=256
        )
        transcription = self.processor.batch_decode(
            predicted_ids, skip_special_tokens=True
        )[0]
        return transcription.strip()


# ---------------------------------------------------------------------
# 2) مرحلة معالجة اللغة (أي LLM — نستخدم نموذجًا صغيرًا للتوضيح)
# ---------------------------------------------------------------------
class LanguageProcessingEngine:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float32
        ).to(device)
        self.model.eval()

    @torch.no_grad()
    def generate_response(self, user_text: str, max_new_tokens: int = 60) -> str:
        messages = [
            {"role": "system", "content": "أنت مساعد صوتي مفيد وموجز. أجب بجملة قصيرة."},
            {"role": "user", "content": user_text},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        response = self.tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        return response.strip()


# ---------------------------------------------------------------------
# 3) مرحلة Text-to-Speech عبر SpeechT5 (Acoustic Model + Vocoder)
# ---------------------------------------------------------------------
class TextToSpeechEngine:
    def __init__(self, model_name: str = "microsoft/speecht5_tts"):
        self.processor = SpeechT5Processor.from_pretrained(model_name)
        self.model = SpeechT5ForTextToSpeech.from_pretrained(model_name).to(device)
        self.vocoder = SpeechT5HifiGan.from_pretrained(
            "microsoft/speecht5_hifigan"
        ).to(device)
        self.model.eval()
        self.vocoder.eval()

        # تضمين المتحدث (Speaker Embedding) — يحدد هوية/نبرة الصوت المُولَّد
        embeddings_dataset = load_dataset(
            "Matthijs/cmu-arctic-xvectors", split="validation"
        )
        self.speaker_embeddings = torch.tensor(
            embeddings_dataset[7306]["xvector"]
        ).unsqueeze(0).to(device)

    @torch.no_grad()
    def synthesize(self, text: str) -> np.ndarray:
        """يحوّل نصًا إلى موجة صوتية خام جاهزة للتشغيل."""
        inputs = self.processor(text=text, return_tensors="pt").to(device)

        # المرحلة 1: Text -> Mel Spectrogram -> المرحلة 2: Spectrogram -> Waveform
        speech = self.model.generate_speech(
            inputs["input_ids"], self.speaker_embeddings, vocoder=self.vocoder
        )
        return speech.cpu().numpy()


# ---------------------------------------------------------------------
# 4) تجميع النظام الكامل (Cascade Pipeline) مع قياس الزمن
# ---------------------------------------------------------------------
@dataclass
class PipelineTimings:
    asr_seconds: float
    llm_seconds: float
    tts_seconds: float

    @property
    def total_seconds(self) -> float:
        return self.asr_seconds + self.llm_seconds + self.tts_seconds


class VoiceAssistantPipeline:
    def __init__(self):
        print("تحميل النماذج الثلاثة... (قد يستغرق بعض الوقت أول مرة)")
        self.asr_engine = SpeechToTextEngine()
        self.llm_engine = LanguageProcessingEngine()
        self.tts_engine = TextToSpeechEngine()
        print("النظام جاهز.")

    def process(
        self, audio_array: np.ndarray, sampling_rate: int = 16000
    ) -> tuple[np.ndarray, PipelineTimings, dict]:
        # المرحلة 1: الصوت -> نص
        t0 = time.perf_counter()
        user_text = self.asr_engine.transcribe(
            audio_array, sampling_rate=sampling_rate, language="arabic"
        )
        t1 = time.perf_counter()

        # المرحلة 2: نص -> استجابة نصية (LLM)
        response_text = self.llm_engine.generate_response(user_text)
        t2 = time.perf_counter()

        # المرحلة 3: استجابة نصية -> صوت
        response_audio = self.tts_engine.synthesize(response_text)
        t3 = time.perf_counter()

        timings = PipelineTimings(
            asr_seconds=t1 - t0, llm_seconds=t2 - t1, tts_seconds=t3 - t2
        )
        debug_info = {"user_text": user_text, "response_text": response_text}
        return response_audio, timings, debug_info


# ---------------------------------------------------------------------
# 5) التشغيل الكامل (مثال توضيحي بصوت مُسجَّل مسبقًا)
# ---------------------------------------------------------------------
if __name__ == "__main__":
    import soundfile as sf

    pipeline = VoiceAssistantPipeline()

    # مثال: تحميل ملف صوتي (استبدله بمسار ملف حقيقي يحتوي سؤالًا بالعربية)
    audio_array, sampling_rate = sf.read("user_question.wav")
    if audio_array.ndim > 1:
        audio_array = audio_array.mean(axis=1)  # تحويل Stereo إلى Mono

    response_audio, timings, debug_info = pipeline.process(audio_array, sampling_rate)

    print(f"\n📝 النص المُفرَّغ (ASR): {debug_info['user_text']}")
    print(f"🤖 استجابة النموذج (LLM): {debug_info['response_text']}")
    print(f"\n⏱️  زمن ASR: {timings.asr_seconds:.3f}s")
    print(f"⏱️  زمن LLM: {timings.llm_seconds:.3f}s")
    print(f"⏱️  زمن TTS: {timings.tts_seconds:.3f}s")
    print(f"⏱️  الزمن الكلي: {timings.total_seconds:.3f}s")

    # حفظ الصوت المُولَّد (16kHz افتراضيًا لـ SpeechT5)
    sf.write("assistant_response.wav", response_audio, samplerate=16000)
    print("\n✅ تم حفظ الرد الصوتي في assistant_response.wav")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-42-chapter-42.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
def stream_llm_response(self, user_text: str):
    """
    استخدم TextIteratorStreamer من transformers لتوليد الاستجابة
    تدريجيًا جملة بجملة، بدلًا من انتظار الاستجابة الكاملة.
    """
    ...

def synthesize_streaming(self, text_stream):
    """
    فور اكتمال أول جملة من الـ LLM (كشف نقطة/علامة استفهام)،
    ابدأ توليد صوتها فورًا بالتوازي مع توليد باقي الاستجابة النصية.
    """
    ...
