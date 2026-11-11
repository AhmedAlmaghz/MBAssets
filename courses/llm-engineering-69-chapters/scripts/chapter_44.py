"""Standalone mini-lab for chapter 44: الفصل 44: النماذج متعددة الوسائط الموحّدة (Unified Multimodal Models)."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_multimodal() -> None:
    sequence = [('text', 'hello'), ('image', 'patch_01'), ('image', 'patch_02'), ('audio', 'frame_01')]
    modality_ids = {'text': 0, 'image': 1, 'audio': 2}
    encoded = [(modality_ids[modality], payload) for modality, payload in sequence]
    print('typed sequence:', encoded)
    print('modality counts:', dict(Counter((modality for modality, _ in sequence))))
    assert all((0 <= token_id < len(modality_ids) for token_id, _ in encoded))

if __name__ == "__main__":
    print("Chapter 44 lab: Pack text, image, and audio into typed tokens")
    run_multimodal()
