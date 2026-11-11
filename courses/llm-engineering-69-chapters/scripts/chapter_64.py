"""Standalone mini-lab for chapter 64: الفصل 64: هندسة التكلفة — Self-Hosting مقابل API، والتوجيه الذكي للنماذج."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_cost() -> None:
    options = [{'name': 'local-small', 'input_per_m': 0.2, 'output_per_m': 0.4, 'latency': 0.12}, {'name': 'hosted-large', 'input_per_m': 2.0, 'output_per_m': 8.0, 'latency': 0.8}]
    tokens_in, tokens_out = (1000, 300)
    budget, latency_sla = (0.01, 0.5)
    for option in options:
        cost = (tokens_in * option['input_per_m'] + tokens_out * option['output_per_m']) / 1000000.0
        eligible = cost <= budget and option['latency'] <= latency_sla
        print(option['name'], 'cost:', round(cost, 6), '| eligible:', eligible)

if __name__ == "__main__":
    print("Chapter 64 lab: Route requests using latency and cost budgets")
    run_cost()
