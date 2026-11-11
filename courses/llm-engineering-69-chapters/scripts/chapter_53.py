"""Standalone mini-lab for chapter 53: الفصل 53: RAG الكامل (Retrieval-Augmented Generation)."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_rag() -> None:
    documents = {'d1': 'attention combines token values using learned weights', 'd2': 'a tokenizer maps text into token identifiers', 'd3': 'retrieval finds relevant source passages'}
    query = set('how does attention combine values'.split())
    ranked = sorted(((len(query & set(text.split())), doc_id, text) for doc_id, text in documents.items()), reverse=True)
    for score, doc_id, text in ranked[:2]:
        print(f'[{score}] [{doc_id}] {text}')
    print('answer policy: cite only retrieved IDs; abstain when evidence is absent.')

if __name__ == "__main__":
    print("Chapter 53 lab: Retrieve evidence and attach source citations")
    run_rag()
