"""Standalone mini-lab for chapter 51: الفصل 51: تحليل المشاعر واستخراج المعلومات (Sentiment Analysis & NER)."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_nlp() -> None:
    text = 'Aisha said the model helped her team in Paris.'
    positive = {'helped', 'excellent', 'useful'}
    negative = {'failed', 'poor', 'harmful'}
    words = re.findall("\\b[\\w']+\\b", text.lower())
    sentiment = sum((word in positive for word in words)) - sum((word in negative for word in words))
    entities = re.findall('\\b[A-Z][a-z]+\\b', text)
    print('lexicon sentiment score:', sentiment)
    print('candidate proper-name entities:', entities)
    print('review ambiguous entities before production use.')

if __name__ == "__main__":
    print("Chapter 51 lab: Combine a sentiment lexicon with entity extraction")
    run_nlp()
