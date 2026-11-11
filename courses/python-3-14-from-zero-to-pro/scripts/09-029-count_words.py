#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.6: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-029-count_words.py
"""

# تمرين 4
def count_words(text: str) -> dict[str, int]:
    words = text.lower().split()
    counts: dict[str, int] = {}
    for word in words:
        counts[word] = counts.get(word, 0) + 1
    return counts

result = count_words("Python is great and Python is fun")
print(result)
# {'python': 2, 'is': 2, 'great': 1, 'and': 1, 'fun': 1}

# تمرين 5
team_a = {"Ali", "Sara", "Omar"}
team_b = {"Omar", "Lina", "Khaled"}

print("في الفريقين:", team_a & team_b)
print("في A فقط:", team_a - team_b)
print("في B فقط:", team_b - team_a)
print("في فريق واحد فقط:", team_a ^ team_b)

# تمرين 6
def merge_dicts(d1: dict, d2: dict) -> dict:
    result = d1.copy()
    for key, value in d2.items():
        result[key] = result.get(key, 0) + value
    return result

a = {"apples": 5, "bananas": 3}
b = {"bananas": 2, "oranges": 7}
print(merge_dicts(a, b))
# {'apples': 5, 'bananas': 5, 'oranges': 7}
