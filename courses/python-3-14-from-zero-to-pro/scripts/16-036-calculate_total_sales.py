#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس عشر: الملفات وتنسيقات البيانات
القسم 16.8: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 16-036-calculate_total_sales.py
"""

# تمرين 4
import csv
from pathlib import Path

def calculate_total_sales(csv_path: Path) -> float:
    if not csv_path.exists():
        print("ملف المبيعات غير موجود!")
        return 0.0

    total = 0.0
    with open(csv_path, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            total += float(row["amount"])
    return total

print(calculate_total_sales(Path("sales.csv")))

# تمرين 5
from pathlib import Path
from datetime import datetime

class Logger:
    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path

    def log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_path, "a", encoding="utf-8") as file:
            file.write(f"[{timestamp}] {message}\n")

logger = Logger(Path("app.log"))
logger.log("بدأ تشغيل التطبيق")
logger.log("تم تسجيل دخول مستخدم جديد")

# تمرين 6
import json
from pathlib import Path

def save_config(config: dict[str, object], path: Path) -> None:
    path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

def load_config(path: Path) -> dict[str, object]:
    default_config: dict[str, object] = {"theme": "light", "language": "ar"}
    if not path.exists():
        return default_config
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default_config

config = load_config(Path("config.json"))
print(config)
save_config({"theme": "dark", "language": "en"}, Path("config.json"))
