"""Standalone mini-lab for chapter 20: الفصل العشرون: خط أنابيب البيانات — من فوضى الإنترنت إلى وقود تدريب نموذج لغوي عملاق."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_data_pipeline() -> None:
    raw = ['  A useful note. ', 'A useful note.', 'BUY NOW!!!', 'The model learns.', '']
    cleaned = []
    seen = set()
    for text in raw:
        text = re.sub('\\s+', ' ', text).strip().lower()
        text = re.sub('[!]{2,}', '!', text)
        if text and text not in seen:
            seen.add(text)
            cleaned.append(text)
    print('raw / retained:', len(raw), '/', len(cleaned))
    print('deduplicated documents:', cleaned)
    print('split by document before tokenization to prevent leakage.')

if __name__ == "__main__":
    print("Chapter 20 lab: A reproducible text-data cleaning pipeline")
    run_data_pipeline()
