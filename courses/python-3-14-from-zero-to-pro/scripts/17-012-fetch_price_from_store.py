#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السابع عشر: البرمجة غير المتزامنة - مقدمة عملية (async/await)
القسم 17.9: مشروع تطبيقي: مُجمِّع أسعار من عدة «متاجر» (محاكاة)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 17-012-fetch_price_from_store.py
"""

"""
مُجمِّع أسعار - مقارنة أداء Sync مقابل Async
"""
from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass

@dataclass
class StoreResult:
    store_name: str
    price: float | None  # None يعني فشل الاستعلام
    error: str | None = None

# ---------------------------------------------------------
# محاكاة استدعاء API لمتجر واحد (غير متزامن)
# ---------------------------------------------------------
async def fetch_price_from_store(store_name: str, base_delay: float) -> StoreResult:
    """يُحاكي استعلام سعر من متجر معيّن، مع احتمال فشل عشوائي."""
    # محاكاة تذبذب في زمن الاستجابة (كالشبكة الحقيقية)
    actual_delay = base_delay + random.uniform(-0.3, 0.3)
    await asyncio.sleep(max(actual_delay, 0.1))

    # محاكاة فشل عشوائي بنسبة 20%
    if random.random() < 0.2:
        return StoreResult(store_name, price=None, error="انتهت مهلة الاتصال")

    price = round(random.uniform(80, 150), 2)
    return StoreResult(store_name, price=price)

# ---------------------------------------------------------
# النسخة غير المتزامنة: كل المتاجر معًا
# ---------------------------------------------------------
async def compare_prices_async(stores: dict[str, float]) -> list[StoreResult]:
    tasks = [
        fetch_price_from_store(name, delay)
        for name, delay in stores.items()
    ]
    return await asyncio.gather(*tasks)

# ---------------------------------------------------------
# النسخة المتزامنة التقليدية (للمقارنة فقط)
# ---------------------------------------------------------
def fetch_price_sync(store_name: str, base_delay: float) -> StoreResult:
    actual_delay = base_delay + random.uniform(-0.3, 0.3)
    time.sleep(max(actual_delay, 0.1))

    if random.random() < 0.2:
        return StoreResult(store_name, price=None, error="انتهت مهلة الاتصال")

    price = round(random.uniform(80, 150), 2)
    return StoreResult(store_name, price=price)

def compare_prices_sync(stores: dict[str, float]) -> list[StoreResult]:
    return [fetch_price_sync(name, delay) for name, delay in stores.items()]

# ---------------------------------------------------------
# تحليل وعرض النتائج
# ---------------------------------------------------------
def print_report(results: list[StoreResult]) -> None:
    print("\n--- تقرير الأسعار ---")
    successful: list[StoreResult] = []

    for result in results:
        if result.price is not None:
            print(f"  ✅ {result.store_name}: {result.price} دولار")
            successful.append(result)
        else:
            print(f"  ❌ {result.store_name}: فشل ({result.error})")

    if not successful:
        print("لم ينجح أي استعلام!")
        return

    prices = [r.price for r in successful if r.price is not None]
    cheapest = min(successful, key=lambda r: r.price or float("inf"))
    most_expensive = max(successful, key=lambda r: r.price or 0)

    print(f"\n💰 الأرخص: {cheapest.store_name} ({cheapest.price} دولار)")
    print(f"💸 الأغلى: {most_expensive.store_name} ({most_expensive.price} دولار)")
    print(f"📊 المتوسط: {sum(prices) / len(prices):.2f} دولار")

# ---------------------------------------------------------
# البرنامج الرئيسي: مقارنة الأداء
# ---------------------------------------------------------
async def main() -> None:
    stores: dict[str, float] = {
        "متجر الأمانة": 1.5,
        "سوق النجاح": 2.0,
        "بازار الخير": 1.0,
        "متجر السرعة": 0.8,
        "سوق التوفير": 1.8,
    }

    print("=" * 50)
    print("🐢 تشغيل النسخة المتزامنة (Sync)...")
    start_sync = time.perf_counter()
    sync_results = compare_prices_sync(stores)
    sync_elapsed = time.perf_counter() - start_sync
    print_report(sync_results)
    print(f"\n⏱️  وقت النسخة المتزامنة: {sync_elapsed:.2f} ثانية")

    print("\n" + "=" * 50)
    print("🚀 تشغيل النسخة غير المتزامنة (Async)...")
    start_async = time.perf_counter()
    async_results = await compare_prices_async(stores)
    async_elapsed = time.perf_counter() - start_async
    print_report(async_results)
    print(f"\n⏱️  وقت النسخة غير المتزامنة: {async_elapsed:.2f} ثانية")

    print("\n" + "=" * 50)
    speedup = sync_elapsed / async_elapsed if async_elapsed > 0 else 0
    print(f"🎯 التسريع المحقق: {speedup:.1f}x أسرع!")

if __name__ == "__main__":
    asyncio.run(main())
