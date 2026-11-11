"""Standalone mini-lab for chapter 55: الفصل 55: بناء وكلاء ذكيين (AI Agents)."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_tools() -> None:
    tools = {'add': lambda a, b: a + b, 'multiply': lambda a, b: a * b}
    plan = [{'name': 'add', 'arguments': {'a': 2, 'b': 3}}, {'name': 'multiply', 'arguments': {'a': 4, 'b': 5}}]
    results = []
    for call in plan:
        if call['name'] not in tools or set(call['arguments']) != {'a', 'b'}:
            raise ValueError('tool call rejected by allow-list/schema.')
        results.append(tools[call['name']](**call['arguments']))
    print('bounded tool results:', results)
    print('external network/filesystem access: disabled')

if __name__ == "__main__":
    print("Chapter 55 lab: Run an allow-listed, bounded tool-calling agent")
    run_tools()
