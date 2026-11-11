"""Standalone mini-lab for chapter 48: الفصل 48: تقييم أنظمة RAG (RAG Evaluation)."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_rag_eval() -> None:
    relevant, retrieved, supported = ({'d1', 'd2'}, {'d1', 'd3'}, {'d1'})
    recall = len(relevant & retrieved) / len(relevant)
    precision = len(relevant & retrieved) / len(retrieved)
    grounding = len(supported) / max(len(retrieved), 1)
    print('context recall:', round(recall, 3))
    print('context precision:', round(precision, 3))
    print('supported retrieved evidence:', round(grounding, 3))

if __name__ == "__main__":
    print("Chapter 48 lab: Evaluate retrieval coverage and answer grounding")
    run_rag_eval()
