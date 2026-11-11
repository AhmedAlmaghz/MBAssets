"""Standalone mini-lab for chapter 60: الفصل 60: خوادم الاستدلال — vLLM وTGI وTensorRT-LLM وllama.cpp."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_serving() -> None:
    arrivals = [0, 1, 2, 8, 9, 10, 20]
    batch_window, max_batch = (3, 3)
    batches, current = ([], [])
    for arrival in arrivals:
        if current and (arrival - current[0] > batch_window or len(current) == max_batch):
            batches.append(current)
            current = []
        current.append(arrival)
    if current:
        batches.append(current)
    print('arrival times:', arrivals)
    print('formed batches:', batches)
    print('throughput (items per batch):', [len(batch) for batch in batches])

if __name__ == "__main__":
    print("Chapter 60 lab: Simulate batching and server throughput")
    run_serving()
