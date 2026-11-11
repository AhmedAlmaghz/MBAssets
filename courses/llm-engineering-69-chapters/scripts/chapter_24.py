"""Standalone mini-lab for chapter 24: الفصل 24: حلقة Pretraining عمليًا — إدارة نقاط الحفظ، تشخيص Loss Spikes، وGradient Checkpointing."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_checkpoint() -> None:
    checkpoint = {'step': 120, 'model': {'w': [0.2, 0.6]}, 'optimizer_step': 120, 'seed': 7}
    encoded = json.dumps(checkpoint, sort_keys=True)
    restored = json.loads(encoded)
    assert restored['step'] == restored['optimizer_step']
    print('checkpoint bytes:', len(encoded.encode('utf-8')))
    print('resume step:', restored['step'], '| seed:', restored['seed'])
    print('integrity check:', restored['model']['w'] == [0.2, 0.6])

if __name__ == "__main__":
    print("Chapter 24 lab: Save, validate, and resume a training checkpoint")
    run_checkpoint()
