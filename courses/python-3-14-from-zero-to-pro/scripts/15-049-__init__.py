#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس عشر: Type Hints — نحو برمجة أكثر وضوحًا ودقة (باستخدام mypy)
القسم 15.15: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 15-049-__init__.py
"""

from typing import TypedDict, Callable

class CacheStats(TypedDict):
    hits: int
    misses: int

class Cache[K, V]:
    def __init__(self) -> None:
        self._data: dict[K, V] = {}
        self._hits: int = 0
        self._misses: int = 0

    def get(self, key: K) -> V | None:
        if key in self._data:
            self._hits += 1
            return self._data[key]
        self._misses += 1
        return None

    def set(self, key: K, value: V) -> None:
        self._data[key] = value

    def get_or_compute(self, key: K, compute_fn: Callable[[], V]) -> V:
        existing = self.get(key)
        if existing is not None:
            return existing
        new_value = compute_fn()
        self.set(key, new_value)
        return new_value

    def stats(self) -> CacheStats:
        return {"hits": self._hits, "misses": self._misses}

def expensive_computation(n: int) -> int:
    print(f"🔄 جارِ الحساب الباهظ لـ {n} ...")
    return n ** 2

cache: Cache[int, int] = Cache()

result1 = cache.get_or_compute(5, lambda: expensive_computation(5))
print(result1)  # يطبع "جارِ الحساب..." ثم 25

result2 = cache.get_or_compute(5, lambda: expensive_computation(5))
print(result2)  # لا يطبع "جارِ الحساب" لأنها مخزّنة! يطبع 25 مباشرة

print(cache.stats())  # {'hits': 1, 'misses': 1}
