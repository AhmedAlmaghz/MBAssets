#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس عشر: الملفات وتنسيقات البيانات
القسم 16.10: ملخص الفصل (Cheat Sheet)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 16-038-content.py
"""

# ===== فتح الملفات النصية =====
with open("file.txt", "r", encoding="utf-8") as f:
    content = f.read()           # الملف كاملاً كنص
    # أو
    lines = f.readlines()        # قائمة بكل الأسطر
    # أو (الأفضل للملفات الكبيرة)
    for line in f:
        print(line.strip())

with open("file.txt", "w", encoding="utf-8") as f:
    f.write("نص جديد\n")         # يمحو المحتوى القديم!

with open("file.txt", "a", encoding="utf-8") as f:
    f.write("إضافة\n")           # يُضيف دون حذف

# ===== pathlib =====
from pathlib import Path

p = Path("data") / "file.txt"
p.exists()          # هل موجود؟
p.is_file()         # هل ملف؟
p.mkdir(parents=True, exist_ok=True)
p.write_text("محتوى", encoding="utf-8")
content = p.read_text(encoding="utf-8")
p.name, p.stem, p.suffix, p.parent

for f in Path(".").glob("*.py"):
    print(f)

# ===== JSON =====
import json

json.dumps(data, ensure_ascii=False, indent=2)   # Python → نص JSON
json.loads(json_text)                             # نص JSON → Python
json.dump(data, file, ensure_ascii=False, indent=2)  # Python → ملف
json.load(file)                                   # ملف → Python

# مع Dataclasses
from dataclasses import asdict
json.dumps(asdict(my_object), ensure_ascii=False)

# ===== CSV =====
import csv

with open("f.csv", "w", encoding="utf-8", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=["a", "b"])
    writer.writeheader()
    writer.writerows(data)

with open("f.csv", "r", encoding="utf-8") as file:
    reader = csv.DictReader(file)
    for row in reader:
        print(row)  # dict[str, str]
