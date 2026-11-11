#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس عشر: الملفات وتنسيقات البيانات
القسم 16.8: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 16-035-count_lines.py
"""

# تمرين 1
names: list[str] = [input(f"أدخل الاسم {i + 1}: ") for i in range(3)]

with open("names.txt", "w", encoding="utf-8") as file:
    for name in names:
        file.write(name + "\n")

# تمرين 2
from pathlib import Path

def count_lines(file_path: Path) -> int:
    with open(file_path, "r", encoding="utf-8") as file:
        return sum(1 for _ in file)

print(count_lines(Path("names.txt")))

# تمرين 3
import json
from pathlib import Path

person: dict[str, object] = {"name": "ليلى", "age": 27, "city": "تونس"}

Path("person.json").write_text(
    json.dumps(person, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

loaded = json.loads(Path("person.json").read_text(encoding="utf-8"))
print(loaded)
