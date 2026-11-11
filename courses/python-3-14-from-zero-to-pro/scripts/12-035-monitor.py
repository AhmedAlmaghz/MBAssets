#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني عشر: الدوال المتقدمة (Lambda, Decorators, Generators, Closures)
القسم 12.7: مشروع تطبيقي: نظام مراقبة أداء مع Decorators وGenerators

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 12-035-monitor.py
"""

import time
import random
from functools import wraps

# --- جزء 1: Decorators للمراقبة ---

call_log: list[dict] = []

def monitor(func):
    """decorator يسجل كل استدعاء: الاسم، الوقت المستغرق، النتيجة."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            status = "نجاح"
        except Exception as e:
            result = str(e)
            status = "فشل"
            raise
        finally:
            duration = time.perf_counter() - start
            call_log.append({
                "function": func.__name__,
                "duration": round(duration, 6),
                "status": status,
            })
        return result
    return wrapper

def retry(max_attempts: int = 3, delay: float = 0.1):
    """decorator factory لإعادة المحاولة عند الفشل."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    time.sleep(delay)
            raise last_error
        return wrapper
    return decorator

# --- جزء 2: دوال العمل الفعلية ---

@monitor
@retry(max_attempts=3)
def fetch_data_from_source(source_id: int) -> dict:
    """تحاكي جلب بيانات من مصدر خارجي، قد يفشل أحياناً."""
    if random.random() < 0.3:
        raise ConnectionError(f"فشل الاتصال بالمصدر {source_id}")
    return {"source_id": source_id, "value": random.randint(1, 100)}

# --- جزء 3: Generator لمعالجة البيانات بكفاءة ---

def data_source_generator(count: int):
    """يولّد معرفات مصادر بيانات واحداً تلو الآخر."""
    for i in range(1, count + 1):
        yield i

def process_all_sources(count: int) -> list[dict]:
    """يعالج كل المصادر، مستخدماً generator لتوفير الذاكرة."""
    results = []
    for source_id in data_source_generator(count):
        try:
            data = fetch_data_from_source(source_id)
            results.append(data)
        except ConnectionError as e:
            print(f"❌ فشل نهائي: {e}")
    return results

# --- جزء 4: تقرير الأداء ---

def print_performance_report() -> None:
    print("\n" + "=" * 50)
    print("📊 تقرير مراقبة الأداء")
    print("=" * 50)

    total_calls = len(call_log)
    successful = sum(1 for log in call_log if log["status"] == "نجاح")
    total_time = sum(log["duration"] for log in call_log)

    print(f"إجمالي الاستدعاءات: {total_calls}")
    print(f"الناجحة: {successful} | الفاشلة: {total_calls - successful}")
    print(f"إجمالي الوقت: {total_time:.4f} ثانية")

    if call_log:
        avg_time = total_time / total_calls
        print(f"متوسط الوقت لكل استدعاء: {avg_time:.6f} ثانية")
    print("=" * 50)

# --- التشغيل ---
if __name__ == "__main__":
    results = process_all_sources(10)

    print(f"\n✅ تم جلب {len(results)} نتيجة بنجاح:")
    for r in results:
        print(f"  مصدر {r['source_id']}: قيمة = {r['value']}")

    print_performance_report()
