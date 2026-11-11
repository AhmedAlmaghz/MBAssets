#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السابع عشر: البرمجة غير المتزامنة - مقدمة عملية (async/await)
القسم 17.10: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 17-016-check_server.py
"""

import asyncio
from typing import TypedDict

class ServerStatus(TypedDict):
    name: str
    status: str
    response_time: float | None

async def check_server(name: str, is_healthy: bool, delay: float) -> ServerStatus:
    async def do_check() -> float:
        await asyncio.sleep(delay)
        if not is_healthy:
            raise ConnectionError("الخادم لا يستجيب")
        return delay

    try:
        response_time = await asyncio.wait_for(do_check(), timeout=3.0)
        return {"name": name, "status": "healthy", "response_time": response_time}
    except asyncio.TimeoutError:
        return {"name": name, "status": "down", "response_time": None}
    except ConnectionError:
        return {"name": name, "status": "down", "response_time": None}

async def check_all_servers(
    servers: list[tuple[str, bool, float]]
) -> list[ServerStatus]:
    tasks = [check_server(name, healthy, delay) for name, healthy, delay in servers]
    return await asyncio.gather(*tasks)

def analyze_results(results: list[ServerStatus]) -> None:
    healthy = [r for r in results if r["status"] == "healthy"]
    down = [r for r in results if r["status"] == "down"]

    print(f"✅ الخوادم السليمة: {len(healthy)}")
    for server in healthy:
        print(f"   {server['name']} - {server['response_time']:.2f}s")

    print(f"❌ الخوادم المعطلة: {len(down)}")
    for server in down:
        print(f"   {server['name']}")

    if healthy:
        slowest = max(healthy, key=lambda r: r["response_time"] or 0)
        print(f"\n🐢 أبطأ خادم سليم: {slowest['name']} ({slowest['response_time']:.2f}s)")

async def main() -> None:
    servers = [
        ("خادم الويب الرئيسي", True, 0.5),
        ("خادم قاعدة البيانات", True, 1.2),
        ("خادم الكاش", False, 1.0),
        ("خادم البريد", True, 4.0),  # سيتجاوز المهلة (3 ثوانٍ) → down
    ]

    results = await check_all_servers(servers)
    analyze_results(results)

if __name__ == "__main__":
    asyncio.run(main())
