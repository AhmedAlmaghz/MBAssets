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
