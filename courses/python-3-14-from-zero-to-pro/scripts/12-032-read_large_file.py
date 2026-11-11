#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني عشر: الدوال المتقدمة (Lambda, Decorators, Generators, Closures)
القسم 12.5: المولّدات (Generators): الكفاءة في الذاكرة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 12-032-read_large_file.py
"""

def read_large_file(file_path: str):
    """تقرأ ملفاً ضخماً سطراً سطراً دون تحميله بالكامل في الذاكرة."""
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            yield line.strip()

def count_lines_with_keyword(file_path: str, keyword: str) -> int:
    """تحسب عدد الأسطر المحتوية على كلمة معينة، بكفاءة ذاكرة عالية."""
    count = 0
    for line in read_large_file(file_path):
        if keyword in line:
            count += 1
    return count

# حتى لو كان الملف بحجم عدة جيجابايت، هذا الكود يستهلك ذاكرة ثابتة تقريباً!
# count = count_lines_with_keyword("huge_log.txt", "ERROR")
