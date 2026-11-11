#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني عشر: الدوال المتقدمة (Lambda, Decorators, Generators, Closures)
القسم 12.4: المزخرفات (Decorators): تعديل سلوك الدوال

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 12-021-retry.py
"""

from functools import wraps
import random

def retry(max_attempts: int = 3):
    """Decorator factory: تُرجع decorator حقيقياً حسب max_attempts."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    print(f"⚠️ محاولة {attempt} فشلت: {e}")
                    if attempt == max_attempts:
                        raise
            return None
        return wrapper
    return decorator

@retry(max_attempts=3)
def unreliable_operation():
    if random.random() < 0.7:
        raise ConnectionError("فشل الاتصال!")
    return "✅ نجحت العملية!"

print(unreliable_operation())
