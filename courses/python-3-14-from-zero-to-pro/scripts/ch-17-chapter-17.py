#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
بايثون 3.14 — من الصفر إلى الاحتراف — البرمجة غير المتزامنة - مقدمة عملية (async/await)
Python 3.14 — From Zero to Pro — async

Book slug : python-3-14-from-zero-to-pro
Chapter   : ch-17-chapter-17 (lesson 17)
Code files: 36 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/python-3-14-from-zero-to-pro/code/ch-17-chapter-17.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
بايثون 3.14 — من الصفر إلى الاحتراف — البرمجة غير المتزامنة - مقدمة عملية (async/await)
Python 3.14 — From Zero to Pro — async

Book slug : python-3-14-from-zero-to-pro
Chapter   : ch-17-chapter-17 (lesson 17)
Code files: 35 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] dist/notebooks/chapter-17.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#@title ⚙️ تهيئة البيئة  { display-mode: "form" }
import sys, os, random, pathlib

print("Python:", sys.version.split()[0], "|", sys.platform)
assert sys.version_info >= (3, 10), "هذا الكتاب يفترض Python 3.10 أو أحدث"

SEED = 42
random.seed(SEED)

# تثبيت الحزم الناقصة فقط (سريع إن كانت مثبّتة أصلًا)
import importlib.util as _iu
_need = [m for m in ["aiohttp"] if _iu.find_spec(m) is None]
if _need:
    # %pip install -q aiohttp

# asyncio.run() لا تعمل داخل Jupyter: نشغّل الحلقة في خيط منفصل
import asyncio as _asyncio, threading as _threading
def _run_in_thread(coro):
    box = {}
    def _target():
        loop = _asyncio.new_event_loop()
        try:
            box['value'] = loop.run_until_complete(coro)
        except BaseException as exc:
            box['error'] = exc
        finally:
            loop.close()
    thread = _threading.Thread(target=_target)
    thread.start()
    thread.join()
    if 'error' in box:
        raise box['error']
    return box['value']
_asyncio.run = _run_in_thread

# ────────────────────────────────────────────────────────────────────────
# [002] dist/notebooks/chapter-17.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import time

def fetch_data(source: str, delay: float) -> str:
    """يُحاكي طلب بيانات بطيء من مصدر خارجي (مثل API أو قاعدة بيانات)."""
    print(f"🔄 بدء تحميل البيانات من {source}...")
    time.sleep(delay)  # محاكاة وقت الانتظار (شبكة، قرص صلب، إلخ)
    print(f"✅ انتهى تحميل البيانات من {source}")
    return f"بيانات من {source}"

def main() -> None:
    start = time.perf_counter()

    result1 = fetch_data("الخادم أ", 2)
    result2 = fetch_data("الخادم ب", 2)
    result3 = fetch_data("الخادم ج", 2)

    elapsed = time.perf_counter() - start
    print(f"\nالوقت الإجمالي: {elapsed:.2f} ثانية")

main()

# ────────────────────────────────────────────────────────────────────────
# [003] dist/notebooks/chapter-17.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
async def say_hello() -> None:
    print("مرحبًا!")

# ────────────────────────────────────────────────────────────────────────
# [004] dist/notebooks/chapter-17.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
result = say_hello()
print(result)  # <coroutine object say_hello at 0x...>
# لاحظ: "مرحبًا!" لم تُطبع إطلاقًا!

# ────────────────────────────────────────────────────────────────────────
# [005] dist/notebooks/chapter-17.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
import asyncio

async def say_hello() -> None:
    print("مرحبًا!")

asyncio.run(say_hello())  # الآن تُطبع "مرحبًا!" فعليًا

# ────────────────────────────────────────────────────────────────────────
# [006] dist/notebooks/chapter-17.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
import asyncio

async def greet(name: str) -> None:
    print(f"مرحبًا، {name}!")

async def main() -> None:
    await greet("أحمد")
    await greet("سارة")

asyncio.run(main())

# ────────────────────────────────────────────────────────────────────────
# [007] dist/notebooks/chapter-17.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
import asyncio
import time

async def fetch_data(source: str, delay: float) -> str:
    print(f"🔄 بدء تحميل البيانات من {source}...")
    await asyncio.sleep(delay)  # ← غير متزامن! يسمح لعمليات أخرى بالعمل أثناء الانتظار
    print(f"✅ انتهى تحميل البيانات من {source}")
    return f"بيانات من {source}"

async def main() -> None:
    start = time.perf_counter()

    result1 = await fetch_data("الخادم أ", 2)
    result2 = await fetch_data("الخادم ب", 2)
    result3 = await fetch_data("الخادم ج", 2)

    elapsed = time.perf_counter() - start
    print(f"\nالوقت الإجمالي: {elapsed:.2f} ثانية")

asyncio.run(main())

# ────────────────────────────────────────────────────────────────────────
# [008] dist/notebooks/chapter-17.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
import asyncio
import time

async def fetch_data(source: str, delay: float) -> str:
    print(f"🔄 بدء تحميل البيانات من {source}...")
    await asyncio.sleep(delay)
    print(f"✅ انتهى تحميل البيانات من {source}")
    return f"بيانات من {source}"

async def main() -> None:
    start = time.perf_counter()

    results = await asyncio.gather(
        fetch_data("الخادم أ", 2),
        fetch_data("الخادم ب", 2),
        fetch_data("الخادم ج", 2),
    )

    elapsed = time.perf_counter() - start
    print(f"\nالنتائج: {results}")
    print(f"الوقت الإجمالي: {elapsed:.2f} ثانية")

asyncio.run(main())

# ────────────────────────────────────────────────────────────────────────
# [009] dist/notebooks/chapter-17.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
import asyncio
import time
import aiohttp

async def fetch_url(session: aiohttp.ClientSession, url: str) -> int:
    """يُحمّل صفحة ويب ويُرجع رمز الحالة (Status Code) فقط، للتبسيط."""
    async with session.get(url) as response:
        return response.status

async def main() -> None:
    urls = [
        "https://www.python.org",
        "https://www.github.com",
        "https://www.wikipedia.org",
    ]

    start = time.perf_counter()

    async with aiohttp.ClientSession() as session:
        tasks = [fetch_url(session, url) for url in urls]
        results = await asyncio.gather(*tasks)

    elapsed = time.perf_counter() - start

    for url, status in zip(urls, results):
        print(f"{url} → {status}")

    print(f"\nالوقت الإجمالي: {elapsed:.2f} ثانية")

asyncio.run(main())

# ────────────────────────────────────────────────────────────────────────
# [010] dist/notebooks/chapter-17.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
import asyncio

async def background_job(name: str, delay: float) -> str:
    print(f"🚀 بدأت المهمة: {name}")
    await asyncio.sleep(delay)
    print(f"🏁 انتهت المهمة: {name}")
    return f"نتيجة {name}"

async def main() -> None:
    # إنشاء المهمة تبدأ تنفيذها فورًا "في الخلفية"
    task1 = asyncio.create_task(background_job("أ", 3))
    task2 = asyncio.create_task(background_job("ب", 1))

    print("البرنامج يستمر بعمل أشياء أخرى بينما المهام تعمل في الخلفية...")
    await asyncio.sleep(0.5)
    print("لا يزال البرنامج يعمل...")

    # الآن ننتظر نتائج المهام عند الحاجة الفعلية إليها
    result1 = await task1
    result2 = await task2

    print(result1, result2)

asyncio.run(main())

# ────────────────────────────────────────────────────────────────────────
# [011] dist/notebooks/chapter-17.ipynb#cell-11
# ────────────────────────────────────────────────────────────────────────
import asyncio

async def risky_operation(should_fail: bool) -> str:
    await asyncio.sleep(1)
    if should_fail:
        raise ValueError("حدث خطأ أثناء العملية!")
    return "نجحت العملية"

async def main() -> None:
    try:
        result = await risky_operation(should_fail=True)
        print(result)
    except ValueError as e:
        print(f"تم اصطياد الخطأ: {e}")

asyncio.run(main())

# ────────────────────────────────────────────────────────────────────────
# [012] dist/notebooks/chapter-17.ipynb#cell-12
# ────────────────────────────────────────────────────────────────────────
import asyncio

async def task_that_fails() -> str:
    await asyncio.sleep(1)
    raise ValueError("فشلت هذه المهمة!")

async def task_that_succeeds() -> str:
    await asyncio.sleep(1)
    return "نجحت هذه المهمة"

async def main() -> None:
    results = await asyncio.gather(
        task_that_succeeds(),
        task_that_fails(),
        return_exceptions=True,  # ← مهم جدًا!
    )

    for result in results:
        if isinstance(result, Exception):
            print(f"خطأ: {result}")
        else:
            print(f"نجاح: {result}")

asyncio.run(main())

# ────────────────────────────────────────────────────────────────────────
# [013] dist/notebooks/chapter-17.ipynb#cell-13
# ────────────────────────────────────────────────────────────────────────
async def get_data() -> str:
    return "بيانات"

def main():
    result = get_data()
    print(result)

main()

# ────────────────────────────────────────────────────────────────────────
# [014] dist/notebooks/chapter-17.ipynb#cell-14
# ────────────────────────────────────────────────────────────────────────
# تمرين 1
import asyncio

async def countdown(n: int) -> None:
    for i in range(n, 0, -1):
        print(i)
        await asyncio.sleep(1)
    print("انتهى!")

asyncio.run(countdown(5))

# تمرين 2
# await func(): ينفّذ الـ Coroutine وينتظر نتيجتها فورًا قبل المتابعة.
# asyncio.create_task(func()): يبدأ تنفيذ الـ Coroutine "في الخلفية" فورًا،
# ويسمح لك بمتابعة كود آخر، على أن تنتظر نتيجتها لاحقًا بـ await عند الحاجة.

# تمرين 3 - الكود المصحح
import asyncio

async def get_data() -> str:
    return "بيانات"

async def main() -> None:
    result = await get_data()  # يجب await لتنفيذ الـ Coroutine فعليًا
    print(result)

asyncio.run(main())  # يجب استدعاء main عبر asyncio.run، وليس مباشرة

# ────────────────────────────────────────────────────────────────────────
# [015] dist/notebooks/chapter-17.ipynb#cell-15
# ────────────────────────────────────────────────────────────────────────
# تمرين 4
import asyncio
import time

async def download_image() -> str:
    await asyncio.sleep(3)
    return "تم تحميل الصورة"

async def download_video() -> str:
    await asyncio.sleep(5)
    return "تم تحميل الفيديو"

async def download_document() -> str:
    await asyncio.sleep(1)
    return "تم تحميل المستند"

async def main() -> None:
    start = time.perf_counter()
    results = await asyncio.gather(
        download_image(), download_video(), download_document()
    )
    elapsed = time.perf_counter() - start
    print(results)
    print(f"الوقت: {elapsed:.2f} ثانية")  # قريب من 5 ثوانٍ

asyncio.run(main())

# تمرين 5
import asyncio

async def download_image() -> str:
    await asyncio.sleep(3)
    return "تم تحميل الصورة"

async def download_video() -> str:
    await asyncio.sleep(2)
    raise ConnectionError("فشل الاتصال بخادم الفيديو")

async def download_document() -> str:
    await asyncio.sleep(1)
    return "تم تحميل المستند"

async def main() -> None:
    results = await asyncio.gather(
        download_image(), download_video(), download_document(),
        return_exceptions=True,
    )
    for result in results:
        if isinstance(result, Exception):
            print(f"❌ فشل: {result}")
        else:
            print(f"✅ نجح: {result}")

asyncio.run(main())

# تمرين 6
import asyncio

async def fetch_with_timeout(delay: float, timeout: float) -> str:
    try:
        async def task() -> str:
            await asyncio.sleep(delay)
            return "تم التحميل بنجاح"
        return await asyncio.wait_for(task(), timeout=timeout)
    except asyncio.TimeoutError:
        return "انتهت المهلة"

async def main() -> None:
    print(await fetch_with_timeout(1, 3))   # "تم التحميل بنجاح"
    print(await fetch_with_timeout(5, 2))   # "انتهت المهلة"

asyncio.run(main())

# ────────────────────────────────────────────────────────────────────────
# [016] dist/notebooks/chapter-17.ipynb#cell-16
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [017] code/chapters/ch17/17-001-fetch_data.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السابع عشر: البرمجة غير المتزامنة - مقدمة عملية (async/await)
القسم 17.2: فهم المشكلة أولًا: لماذا الكود العادي «بطيء» أحيانًا؟

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 17-001-fetch_data.py
"""

import time

def fetch_data(source: str, delay: float) -> str:
    """يُحاكي طلب بيانات بطيء من مصدر خارجي (مثل API أو قاعدة بيانات)."""
    print(f"🔄 بدء تحميل البيانات من {source}...")
    time.sleep(delay)  # محاكاة وقت الانتظار (شبكة، قرص صلب، إلخ)
    print(f"✅ انتهى تحميل البيانات من {source}")
    return f"بيانات من {source}"

def main() -> None:
    start = time.perf_counter()

    result1 = fetch_data("الخادم أ", 2)
    result2 = fetch_data("الخادم ب", 2)
    result3 = fetch_data("الخادم ج", 2)

    elapsed = time.perf_counter() - start
    print(f"\nالوقت الإجمالي: {elapsed:.2f} ثانية")

main()

# ────────────────────────────────────────────────────────────────────────
# [018] code/chapters/ch17/17-002-say_hello.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السابع عشر: البرمجة غير المتزامنة - مقدمة عملية (async/await)
القسم 17.3: المفاهيم الأساسية: `async def`, `await`, Coroutines

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 17-002-say_hello.py
"""

async def say_hello() -> None:
    print("مرحبًا!")

# ────────────────────────────────────────────────────────────────────────
# [019] code/chapters/ch17/17-003-result.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السابع عشر: البرمجة غير المتزامنة - مقدمة عملية (async/await)
القسم 17.3: المفاهيم الأساسية: `async def`, `await`, Coroutines

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 17-003-result.py
"""

result = say_hello()
print(result)  # <coroutine object say_hello at 0x...>
# لاحظ: "مرحبًا!" لم تُطبع إطلاقًا!

# ────────────────────────────────────────────────────────────────────────
# [020] code/chapters/ch17/17-004-say_hello.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السابع عشر: البرمجة غير المتزامنة - مقدمة عملية (async/await)
القسم 17.3: المفاهيم الأساسية: `async def`, `await`, Coroutines

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 17-004-say_hello.py
"""

import asyncio

async def say_hello() -> None:
    print("مرحبًا!")

asyncio.run(say_hello())  # الآن تُطبع "مرحبًا!" فعليًا

# ────────────────────────────────────────────────────────────────────────
# [021] code/chapters/ch17/17-005-greet.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السابع عشر: البرمجة غير المتزامنة - مقدمة عملية (async/await)
القسم 17.3: المفاهيم الأساسية: `async def`, `await`, Coroutines

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 17-005-greet.py
"""

import asyncio

async def greet(name: str) -> None:
    print(f"مرحبًا، {name}!")

async def main() -> None:
    await greet("أحمد")
    await greet("سارة")

asyncio.run(main())

# ────────────────────────────────────────────────────────────────────────
# [022] code/chapters/ch17/17-006-fetch_data.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السابع عشر: البرمجة غير المتزامنة - مقدمة عملية (async/await)
القسم 17.3: المفاهيم الأساسية: `async def`, `await`, Coroutines

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 17-006-fetch_data.py
"""

import asyncio
import time

async def fetch_data(source: str, delay: float) -> str:
    print(f"🔄 بدء تحميل البيانات من {source}...")
    await asyncio.sleep(delay)  # ← غير متزامن! يسمح لعمليات أخرى بالعمل أثناء الانتظار
    print(f"✅ انتهى تحميل البيانات من {source}")
    return f"بيانات من {source}"

async def main() -> None:
    start = time.perf_counter()

    result1 = await fetch_data("الخادم أ", 2)
    result2 = await fetch_data("الخادم ب", 2)
    result3 = await fetch_data("الخادم ج", 2)

    elapsed = time.perf_counter() - start
    print(f"\nالوقت الإجمالي: {elapsed:.2f} ثانية")

asyncio.run(main())

# ────────────────────────────────────────────────────────────────────────
# [023] code/chapters/ch17/17-007-fetch_data.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السابع عشر: البرمجة غير المتزامنة - مقدمة عملية (async/await)
القسم 17.4: تحقيق التزامن الحقيقي: `asyncio.gather()`

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 17-007-fetch_data.py
"""

import asyncio
import time

async def fetch_data(source: str, delay: float) -> str:
    print(f"🔄 بدء تحميل البيانات من {source}...")
    await asyncio.sleep(delay)
    print(f"✅ انتهى تحميل البيانات من {source}")
    return f"بيانات من {source}"

async def main() -> None:
    start = time.perf_counter()

    results = await asyncio.gather(
        fetch_data("الخادم أ", 2),
        fetch_data("الخادم ب", 2),
        fetch_data("الخادم ج", 2),
    )

    elapsed = time.perf_counter() - start
    print(f"\nالنتائج: {results}")
    print(f"الوقت الإجمالي: {elapsed:.2f} ثانية")

asyncio.run(main())

# ────────────────────────────────────────────────────────────────────────
# [024] code/chapters/ch17/17-008-fetch_url.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السابع عشر: البرمجة غير المتزامنة - مقدمة عملية (async/await)
القسم 17.5: مثال عملي حقيقي: تحميل بيانات من الإنترنت بـ `aiohttp`

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 17-008-fetch_url.py
"""

import asyncio
import time
import aiohttp

async def fetch_url(session: aiohttp.ClientSession, url: str) -> int:
    """يُحمّل صفحة ويب ويُرجع رمز الحالة (Status Code) فقط، للتبسيط."""
    async with session.get(url) as response:
        return response.status

async def main() -> None:
    urls = [
        "https://www.python.org",
        "https://www.github.com",
        "https://www.wikipedia.org",
    ]

    start = time.perf_counter()

    async with aiohttp.ClientSession() as session:
        tasks = [fetch_url(session, url) for url in urls]
        results = await asyncio.gather(*tasks)

    elapsed = time.perf_counter() - start

    for url, status in zip(urls, results):
        print(f"{url} → {status}")

    print(f"\nالوقت الإجمالي: {elapsed:.2f} ثانية")

asyncio.run(main())

# ────────────────────────────────────────────────────────────────────────
# [025] code/chapters/ch17/17-009-background_job.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السابع عشر: البرمجة غير المتزامنة - مقدمة عملية (async/await)
القسم 17.6: `asyncio.create_task()`: بدء المهام دون انتظار فوري

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 17-009-background_job.py
"""

import asyncio

async def background_job(name: str, delay: float) -> str:
    print(f"🚀 بدأت المهمة: {name}")
    await asyncio.sleep(delay)
    print(f"🏁 انتهت المهمة: {name}")
    return f"نتيجة {name}"

async def main() -> None:
    # إنشاء المهمة تبدأ تنفيذها فورًا "في الخلفية"
    task1 = asyncio.create_task(background_job("أ", 3))
    task2 = asyncio.create_task(background_job("ب", 1))

    print("البرنامج يستمر بعمل أشياء أخرى بينما المهام تعمل في الخلفية...")
    await asyncio.sleep(0.5)
    print("لا يزال البرنامج يعمل...")

    # الآن ننتظر نتائج المهام عند الحاجة الفعلية إليها
    result1 = await task1
    result2 = await task2

    print(result1, result2)

asyncio.run(main())

# ────────────────────────────────────────────────────────────────────────
# [026] code/chapters/ch17/17-010-risky_operation.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السابع عشر: البرمجة غير المتزامنة - مقدمة عملية (async/await)
القسم 17.7: معالجة الأخطاء في الكود غير المتزامن

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 17-010-risky_operation.py
"""

import asyncio

async def risky_operation(should_fail: bool) -> str:
    await asyncio.sleep(1)
    if should_fail:
        raise ValueError("حدث خطأ أثناء العملية!")
    return "نجحت العملية"

async def main() -> None:
    try:
        result = await risky_operation(should_fail=True)
        print(result)
    except ValueError as e:
        print(f"تم اصطياد الخطأ: {e}")

asyncio.run(main())

# ────────────────────────────────────────────────────────────────────────
# [027] code/chapters/ch17/17-011-task_that_fails.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السابع عشر: البرمجة غير المتزامنة - مقدمة عملية (async/await)
القسم 17.7: معالجة الأخطاء في الكود غير المتزامن

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 17-011-task_that_fails.py
"""

import asyncio

async def task_that_fails() -> str:
    await asyncio.sleep(1)
    raise ValueError("فشلت هذه المهمة!")

async def task_that_succeeds() -> str:
    await asyncio.sleep(1)
    return "نجحت هذه المهمة"

async def main() -> None:
    results = await asyncio.gather(
        task_that_succeeds(),
        task_that_fails(),
        return_exceptions=True,  # ← مهم جدًا!
    )

    for result in results:
        if isinstance(result, Exception):
            print(f"خطأ: {result}")
        else:
            print(f"نجاح: {result}")

asyncio.run(main())

# ────────────────────────────────────────────────────────────────────────
# [028] code/chapters/ch17/17-012-fetch_price_from_store.py
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [029] code/chapters/ch17/17-013-get_data.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السابع عشر: البرمجة غير المتزامنة - مقدمة عملية (async/await)
القسم 17.10: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 17-013-get_data.py
"""

async def get_data() -> str:
    return "بيانات"

def main():
    result = get_data()
    print(result)

main()

# ────────────────────────────────────────────────────────────────────────
# [030] code/chapters/ch17/17-014-countdown.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السابع عشر: البرمجة غير المتزامنة - مقدمة عملية (async/await)
القسم 17.10: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 17-014-countdown.py
"""

# تمرين 1
import asyncio

async def countdown(n: int) -> None:
    for i in range(n, 0, -1):
        print(i)
        await asyncio.sleep(1)
    print("انتهى!")

asyncio.run(countdown(5))

# تمرين 2
# await func(): ينفّذ الـ Coroutine وينتظر نتيجتها فورًا قبل المتابعة.
# asyncio.create_task(func()): يبدأ تنفيذ الـ Coroutine "في الخلفية" فورًا،
# ويسمح لك بمتابعة كود آخر، على أن تنتظر نتيجتها لاحقًا بـ await عند الحاجة.

# تمرين 3 - الكود المصحح
import asyncio

async def get_data() -> str:
    return "بيانات"

async def main() -> None:
    result = await get_data()  # يجب await لتنفيذ الـ Coroutine فعليًا
    print(result)

asyncio.run(main())  # يجب استدعاء main عبر asyncio.run، وليس مباشرة

# ────────────────────────────────────────────────────────────────────────
# [031] code/chapters/ch17/17-015-download_image.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السابع عشر: البرمجة غير المتزامنة - مقدمة عملية (async/await)
القسم 17.10: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 17-015-download_image.py
"""

# تمرين 4
import asyncio
import time

async def download_image() -> str:
    await asyncio.sleep(3)
    return "تم تحميل الصورة"

async def download_video() -> str:
    await asyncio.sleep(5)
    return "تم تحميل الفيديو"

async def download_document() -> str:
    await asyncio.sleep(1)
    return "تم تحميل المستند"

async def main() -> None:
    start = time.perf_counter()
    results = await asyncio.gather(
        download_image(), download_video(), download_document()
    )
    elapsed = time.perf_counter() - start
    print(results)
    print(f"الوقت: {elapsed:.2f} ثانية")  # قريب من 5 ثوانٍ

asyncio.run(main())

# تمرين 5
import asyncio

async def download_image() -> str:
    await asyncio.sleep(3)
    return "تم تحميل الصورة"

async def download_video() -> str:
    await asyncio.sleep(2)
    raise ConnectionError("فشل الاتصال بخادم الفيديو")

async def download_document() -> str:
    await asyncio.sleep(1)
    return "تم تحميل المستند"

async def main() -> None:
    results = await asyncio.gather(
        download_image(), download_video(), download_document(),
        return_exceptions=True,
    )
    for result in results:
        if isinstance(result, Exception):
            print(f"❌ فشل: {result}")
        else:
            print(f"✅ نجح: {result}")

asyncio.run(main())

# تمرين 6
import asyncio

async def fetch_with_timeout(delay: float, timeout: float) -> str:
    try:
        async def task() -> str:
            await asyncio.sleep(delay)
            return "تم التحميل بنجاح"
        return await asyncio.wait_for(task(), timeout=timeout)
    except asyncio.TimeoutError:
        return "انتهت المهلة"

async def main() -> None:
    print(await fetch_with_timeout(1, 3))   # "تم التحميل بنجاح"
    print(await fetch_with_timeout(5, 2))   # "انتهت المهلة"

asyncio.run(main())

# ────────────────────────────────────────────────────────────────────────
# [032] code/chapters/ch17/17-016-check_server.py
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [033] code/chapters/ch17/17-017-my_function.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السابع عشر: البرمجة غير المتزامنة - مقدمة عملية (async/await)
القسم 17.12: ملخص الفصل (Cheat Sheet)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 17-017-my_function.py
"""

import asyncio

# ===== تعريف وتشغيل Coroutine =====
async def my_function() -> str:
    await asyncio.sleep(1)   # انتظار غير متزامن (وليس time.sleep!)
    return "النتيجة"

asyncio.run(my_function())   # نقطة الدخول الرئيسية - تُستدعى مرة واحدة فقط

# ===== انتظار متتالٍ (عند الاعتماد الحقيقي بين العمليات) =====
async def main() -> None:
    result1 = await my_function()
    result2 = await my_function()  # يبدأ فقط بعد انتهاء الأولى

# ===== تزامن حقيقي (عمليات مستقلة) =====
async def main() -> None:
    results = await asyncio.gather(
        my_function(),
        my_function(),
        return_exceptions=True,  # لمنع توقف الكل عند فشل واحدة
    )

# ===== بدء مهمة في الخلفية =====
async def main() -> None:
    task = asyncio.create_task(my_function())
    # ... كود آخر يعمل الآن ...
    result = await task  # ننتظر النتيجة عند الحاجة فقط

# ===== فرض مهلة قصوى =====
try:
    result = await asyncio.wait_for(my_function(), timeout=2.0)
except asyncio.TimeoutError:
    print("انتهت المهلة")

# ===== معالجة الأخطاء =====
try:
    result = await risky_coroutine()
except SomeError as e:
    print(f"خطأ: {e}")

# ────────────────────────────────────────────────────────────────────────
# [034] dist/markdown/chapter-17.md
# ────────────────────────────────────────────────────────────────────────
> def regular_function():
>     await something()  # ❌ SyntaxError: 'await' outside async function
> ```

### محاكاة الانتظار غير المتزامن: `asyncio.sleep()`

تذكر أننا استخدمنا `time.sleep()` في المثال الأول لمحاكاة بطء الشبكة. لكن `time.sleep()` هي دالة **متزامنة (Blocking)** — تُجمّد **كل البرنامج بالكامل**، ولا تسمح لأي كود آخر بالعمل أثناء انتظارها، حتى لو كان الكود الآخر Coroutine!

الحل: `asyncio.sleep()` — نسخة غير متزامنة من `sleep()`، مصممة خصيصًا للعمل مع `async/await`:

# ────────────────────────────────────────────────────────────────────────
# [035] dist/markdown/chapter-17.md
# ────────────────────────────────────────────────────────────────────────
لاحظ النقاط التالية:
- `async with` تُستخدم مع `aiohttp.ClientSession()` — نسخة غير متزامنة من `with` التي تعلمناها في الفصل 16، تعمل مع Context Managers غير متزامنة (`__aenter__`/`__aexit__`).
- `tasks = [fetch_url(session, url) for url in urls]`: قائمة من Coroutines (لم تُشغَّل بعد).
- `*tasks`: نستخدم عامل فك التغليف (من الفصل 10) لتمرير كل عناصر القائمة كمعاملات منفصلة لـ `gather()`.

تشغيل هذا البرنامج سيُحمّل الصفحات الثلاث **في نفس الوقت تقريبًا**، بدلًا من تحميلها واحدة تلو الأخرى، مما يُقلل الوقت الإجمالي بشكل ملحوظ (خاصة مع عدد أكبر من الروابط).

> 💡 **نصيحة الخبير:** كل مكتبة I/O «عادية» (متزامنة) لها عادة «نسخة توأم» غير متزامنة: `requests` ↔ `aiohttp` أو `httpx` (التي تدعم الاثنين معًا)، `psycopg2` ↔ `asyncpg` لقواعد بيانات PostgreSQL، إلخ. عند العمل في بيئة `async` (مثل تطبيق FastAPI)، **يجب** استخدام النسخ غير المتزامنة من كل مكتبة، وإلا ستُجمّد استدعاءاتها المتزامنة حلقة الأحداث بالكامل وتُلغي كل فائدة `asyncio`!

> ⚠️ **خطأ فادح وشائع جدًا:** استخدام دالة متزامنة بطيئة (مثل `time.sleep()` أو `requests.get()`) داخل Coroutine! هذا يُجمّد **حلقة الأحداث بأكملها**، فتتوقف كل المهام الأخرى المنتظرة أيضًا، مما يُلغي كل فائدة `async` تمامًا:
> ```python
> import time
>
> async def bad_example() -> None:
>     time.sleep(2)  # ❌ خطأ فادح! يُجمّد حلقة الأحداث بأكملها
>
> async def good_example() -> None:
>     await asyncio.sleep(2)  # ✅ صحيح، يسمح بتبديل المهام
> ```

---

## 17.6 `asyncio.create_task()`: بدء المهام دون انتظار فوري

أحيانًا تريد أن «تبدأ» عملية غير متزامنة، لكن تريد الاستمرار في عمل آخر، والعودة لنتيجتها لاحقًا. هنا تأتي `asyncio.create_task()`:

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-17-chapter-17.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#@title ⚙️ تهيئة البيئة  { display-mode: "form" }
import sys, os, random, pathlib

print("Python:", sys.version.split()[0], "|", sys.platform)
assert sys.version_info >= (3, 10), "هذا الكتاب يفترض Python 3.10 أو أحدث"

SEED = 42
random.seed(SEED)

# تثبيت الحزم الناقصة فقط (سريع إن كانت مثبّتة أصلًا)
import importlib.util as _iu
_need = [m for m in ["aiohttp"] if _iu.find_spec(m) is None]
if _need:
    # %pip install -q aiohttp

# asyncio.run() لا تعمل داخل Jupyter: نشغّل الحلقة في خيط منفصل
import asyncio as _asyncio, threading as _threading
def _run_in_thread(coro):
    box = {}
    def _target():
        loop = _asyncio.new_event_loop()
        try:
            box['value'] = loop.run_until_complete(coro)
        except BaseException as exc:
            box['error'] = exc
        finally:
            loop.close()
    thread = _threading.Thread(target=_target)
    thread.start()
    thread.join()
    if 'error' in box:
        raise box['error']
    return box['value']
_asyncio.run = _run_in_thread

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-17-chapter-17.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import time

def fetch_data(source: str, delay: float) -> str:
    """يُحاكي طلب بيانات بطيء من مصدر خارجي (مثل API أو قاعدة بيانات)."""
    print(f"🔄 بدء تحميل البيانات من {source}...")
    time.sleep(delay)  # محاكاة وقت الانتظار (شبكة، قرص صلب، إلخ)
    print(f"✅ انتهى تحميل البيانات من {source}")
    return f"بيانات من {source}"

def main() -> None:
    start = time.perf_counter()

    result1 = fetch_data("الخادم أ", 2)
    result2 = fetch_data("الخادم ب", 2)
    result3 = fetch_data("الخادم ج", 2)

    elapsed = time.perf_counter() - start
    print(f"\nالوقت الإجمالي: {elapsed:.2f} ثانية")

main()

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-17-chapter-17.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
async def say_hello() -> None:
    print("مرحبًا!")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-17-chapter-17.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
result = say_hello()
print(result)  # <coroutine object say_hello at 0x...>
# لاحظ: "مرحبًا!" لم تُطبع إطلاقًا!

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-17-chapter-17.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
import asyncio

async def say_hello() -> None:
    print("مرحبًا!")

asyncio.run(say_hello())  # الآن تُطبع "مرحبًا!" فعليًا

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-17-chapter-17.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
import asyncio

async def greet(name: str) -> None:
    print(f"مرحبًا، {name}!")

async def main() -> None:
    await greet("أحمد")
    await greet("سارة")

asyncio.run(main())

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-17-chapter-17.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
import asyncio
import time

async def fetch_data(source: str, delay: float) -> str:
    print(f"🔄 بدء تحميل البيانات من {source}...")
    await asyncio.sleep(delay)  # ← غير متزامن! يسمح لعمليات أخرى بالعمل أثناء الانتظار
    print(f"✅ انتهى تحميل البيانات من {source}")
    return f"بيانات من {source}"

async def main() -> None:
    start = time.perf_counter()

    result1 = await fetch_data("الخادم أ", 2)
    result2 = await fetch_data("الخادم ب", 2)
    result3 = await fetch_data("الخادم ج", 2)

    elapsed = time.perf_counter() - start
    print(f"\nالوقت الإجمالي: {elapsed:.2f} ثانية")

asyncio.run(main())

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-17-chapter-17.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
import asyncio
import time

async def fetch_data(source: str, delay: float) -> str:
    print(f"🔄 بدء تحميل البيانات من {source}...")
    await asyncio.sleep(delay)
    print(f"✅ انتهى تحميل البيانات من {source}")
    return f"بيانات من {source}"

async def main() -> None:
    start = time.perf_counter()

    results = await asyncio.gather(
        fetch_data("الخادم أ", 2),
        fetch_data("الخادم ب", 2),
        fetch_data("الخادم ج", 2),
    )

    elapsed = time.perf_counter() - start
    print(f"\nالنتائج: {results}")
    print(f"الوقت الإجمالي: {elapsed:.2f} ثانية")

asyncio.run(main())

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-17-chapter-17.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
import asyncio
import time
import aiohttp

async def fetch_url(session: aiohttp.ClientSession, url: str) -> int:
    """يُحمّل صفحة ويب ويُرجع رمز الحالة (Status Code) فقط، للتبسيط."""
    async with session.get(url) as response:
        return response.status

async def main() -> None:
    urls = [
        "https://www.python.org",
        "https://www.github.com",
        "https://www.wikipedia.org",
    ]

    start = time.perf_counter()

    async with aiohttp.ClientSession() as session:
        tasks = [fetch_url(session, url) for url in urls]
        results = await asyncio.gather(*tasks)

    elapsed = time.perf_counter() - start

    for url, status in zip(urls, results):
        print(f"{url} → {status}")

    print(f"\nالوقت الإجمالي: {elapsed:.2f} ثانية")

asyncio.run(main())

# ────────────────────────────────────────────────────────────────────────
# [011] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-17-chapter-17.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
import asyncio

async def background_job(name: str, delay: float) -> str:
    print(f"🚀 بدأت المهمة: {name}")
    await asyncio.sleep(delay)
    print(f"🏁 انتهت المهمة: {name}")
    return f"نتيجة {name}"

async def main() -> None:
    # إنشاء المهمة تبدأ تنفيذها فورًا "في الخلفية"
    task1 = asyncio.create_task(background_job("أ", 3))
    task2 = asyncio.create_task(background_job("ب", 1))

    print("البرنامج يستمر بعمل أشياء أخرى بينما المهام تعمل في الخلفية...")
    await asyncio.sleep(0.5)
    print("لا يزال البرنامج يعمل...")

    # الآن ننتظر نتائج المهام عند الحاجة الفعلية إليها
    result1 = await task1
    result2 = await task2

    print(result1, result2)

asyncio.run(main())

# ────────────────────────────────────────────────────────────────────────
# [012] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-17-chapter-17.ipynb#cell-11
# ────────────────────────────────────────────────────────────────────────
import asyncio

async def risky_operation(should_fail: bool) -> str:
    await asyncio.sleep(1)
    if should_fail:
        raise ValueError("حدث خطأ أثناء العملية!")
    return "نجحت العملية"

async def main() -> None:
    try:
        result = await risky_operation(should_fail=True)
        print(result)
    except ValueError as e:
        print(f"تم اصطياد الخطأ: {e}")

asyncio.run(main())

# ────────────────────────────────────────────────────────────────────────
# [013] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-17-chapter-17.ipynb#cell-12
# ────────────────────────────────────────────────────────────────────────
import asyncio

async def task_that_fails() -> str:
    await asyncio.sleep(1)
    raise ValueError("فشلت هذه المهمة!")

async def task_that_succeeds() -> str:
    await asyncio.sleep(1)
    return "نجحت هذه المهمة"

async def main() -> None:
    results = await asyncio.gather(
        task_that_succeeds(),
        task_that_fails(),
        return_exceptions=True,  # ← مهم جدًا!
    )

    for result in results:
        if isinstance(result, Exception):
            print(f"خطأ: {result}")
        else:
            print(f"نجاح: {result}")

asyncio.run(main())

# ────────────────────────────────────────────────────────────────────────
# [014] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-17-chapter-17.ipynb#cell-13
# ────────────────────────────────────────────────────────────────────────
async def get_data() -> str:
    return "بيانات"

def main():
    result = get_data()
    print(result)

main()

# ────────────────────────────────────────────────────────────────────────
# [015] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-17-chapter-17.ipynb#cell-14
# ────────────────────────────────────────────────────────────────────────
# تمرين 1
import asyncio

async def countdown(n: int) -> None:
    for i in range(n, 0, -1):
        print(i)
        await asyncio.sleep(1)
    print("انتهى!")

asyncio.run(countdown(5))

# تمرين 2
# await func(): ينفّذ الـ Coroutine وينتظر نتيجتها فورًا قبل المتابعة.
# asyncio.create_task(func()): يبدأ تنفيذ الـ Coroutine "في الخلفية" فورًا،
# ويسمح لك بمتابعة كود آخر، على أن تنتظر نتيجتها لاحقًا بـ await عند الحاجة.

# تمرين 3 - الكود المصحح
import asyncio

async def get_data() -> str:
    return "بيانات"

async def main() -> None:
    result = await get_data()  # يجب await لتنفيذ الـ Coroutine فعليًا
    print(result)

asyncio.run(main())  # يجب استدعاء main عبر asyncio.run، وليس مباشرة

# ────────────────────────────────────────────────────────────────────────
# [016] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-17-chapter-17.ipynb#cell-15
# ────────────────────────────────────────────────────────────────────────
# تمرين 4
import asyncio
import time

async def download_image() -> str:
    await asyncio.sleep(3)
    return "تم تحميل الصورة"

async def download_video() -> str:
    await asyncio.sleep(5)
    return "تم تحميل الفيديو"

async def download_document() -> str:
    await asyncio.sleep(1)
    return "تم تحميل المستند"

async def main() -> None:
    start = time.perf_counter()
    results = await asyncio.gather(
        download_image(), download_video(), download_document()
    )
    elapsed = time.perf_counter() - start
    print(results)
    print(f"الوقت: {elapsed:.2f} ثانية")  # قريب من 5 ثوانٍ

asyncio.run(main())

# تمرين 5
import asyncio

async def download_image() -> str:
    await asyncio.sleep(3)
    return "تم تحميل الصورة"

async def download_video() -> str:
    await asyncio.sleep(2)
    raise ConnectionError("فشل الاتصال بخادم الفيديو")

async def download_document() -> str:
    await asyncio.sleep(1)
    return "تم تحميل المستند"

async def main() -> None:
    results = await asyncio.gather(
        download_image(), download_video(), download_document(),
        return_exceptions=True,
    )
    for result in results:
        if isinstance(result, Exception):
            print(f"❌ فشل: {result}")
        else:
            print(f"✅ نجح: {result}")

asyncio.run(main())

# تمرين 6
import asyncio

async def fetch_with_timeout(delay: float, timeout: float) -> str:
    try:
        async def task() -> str:
            await asyncio.sleep(delay)
            return "تم التحميل بنجاح"
        return await asyncio.wait_for(task(), timeout=timeout)
    except asyncio.TimeoutError:
        return "انتهت المهلة"

async def main() -> None:
    print(await fetch_with_timeout(1, 3))   # "تم التحميل بنجاح"
    print(await fetch_with_timeout(5, 2))   # "انتهت المهلة"

asyncio.run(main())

# ────────────────────────────────────────────────────────────────────────
# [017] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-17-chapter-17.ipynb#cell-16
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [018] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-17-chapter-17.ipynb#cell-17
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السابع عشر: البرمجة غير المتزامنة - مقدمة عملية (async/await)
القسم 17.2: فهم المشكلة أولًا: لماذا الكود العادي «بطيء» أحيانًا؟

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 17-001-fetch_data.py
"""

import time

def fetch_data(source: str, delay: float) -> str:
    """يُحاكي طلب بيانات بطيء من مصدر خارجي (مثل API أو قاعدة بيانات)."""
    print(f"🔄 بدء تحميل البيانات من {source}...")
    time.sleep(delay)  # محاكاة وقت الانتظار (شبكة، قرص صلب، إلخ)
    print(f"✅ انتهى تحميل البيانات من {source}")
    return f"بيانات من {source}"

def main() -> None:
    start = time.perf_counter()

    result1 = fetch_data("الخادم أ", 2)
    result2 = fetch_data("الخادم ب", 2)
    result3 = fetch_data("الخادم ج", 2)

    elapsed = time.perf_counter() - start
    print(f"\nالوقت الإجمالي: {elapsed:.2f} ثانية")

main()

# ────────────────────────────────────────────────────────────────────────
# [019] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-17-chapter-17.ipynb#cell-18
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السابع عشر: البرمجة غير المتزامنة - مقدمة عملية (async/await)
القسم 17.3: المفاهيم الأساسية: `async def`, `await`, Coroutines

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 17-002-say_hello.py
"""

async def say_hello() -> None:
    print("مرحبًا!")

# ────────────────────────────────────────────────────────────────────────
# [020] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-17-chapter-17.ipynb#cell-19
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السابع عشر: البرمجة غير المتزامنة - مقدمة عملية (async/await)
القسم 17.3: المفاهيم الأساسية: `async def`, `await`, Coroutines

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 17-003-result.py
"""

result = say_hello()
print(result)  # <coroutine object say_hello at 0x...>
# لاحظ: "مرحبًا!" لم تُطبع إطلاقًا!

# ────────────────────────────────────────────────────────────────────────
# [021] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-17-chapter-17.ipynb#cell-20
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السابع عشر: البرمجة غير المتزامنة - مقدمة عملية (async/await)
القسم 17.3: المفاهيم الأساسية: `async def`, `await`, Coroutines

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 17-004-say_hello.py
"""

import asyncio

async def say_hello() -> None:
    print("مرحبًا!")

asyncio.run(say_hello())  # الآن تُطبع "مرحبًا!" فعليًا

# ────────────────────────────────────────────────────────────────────────
# [022] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-17-chapter-17.ipynb#cell-21
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السابع عشر: البرمجة غير المتزامنة - مقدمة عملية (async/await)
القسم 17.3: المفاهيم الأساسية: `async def`, `await`, Coroutines

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 17-005-greet.py
"""

import asyncio

async def greet(name: str) -> None:
    print(f"مرحبًا، {name}!")

async def main() -> None:
    await greet("أحمد")
    await greet("سارة")

asyncio.run(main())

# ────────────────────────────────────────────────────────────────────────
# [023] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-17-chapter-17.ipynb#cell-22
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السابع عشر: البرمجة غير المتزامنة - مقدمة عملية (async/await)
القسم 17.3: المفاهيم الأساسية: `async def`, `await`, Coroutines

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 17-006-fetch_data.py
"""

import asyncio
import time

async def fetch_data(source: str, delay: float) -> str:
    print(f"🔄 بدء تحميل البيانات من {source}...")
    await asyncio.sleep(delay)  # ← غير متزامن! يسمح لعمليات أخرى بالعمل أثناء الانتظار
    print(f"✅ انتهى تحميل البيانات من {source}")
    return f"بيانات من {source}"

async def main() -> None:
    start = time.perf_counter()

    result1 = await fetch_data("الخادم أ", 2)
    result2 = await fetch_data("الخادم ب", 2)
    result3 = await fetch_data("الخادم ج", 2)

    elapsed = time.perf_counter() - start
    print(f"\nالوقت الإجمالي: {elapsed:.2f} ثانية")

asyncio.run(main())

# ────────────────────────────────────────────────────────────────────────
# [024] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-17-chapter-17.ipynb#cell-23
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السابع عشر: البرمجة غير المتزامنة - مقدمة عملية (async/await)
القسم 17.4: تحقيق التزامن الحقيقي: `asyncio.gather()`

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 17-007-fetch_data.py
"""

import asyncio
import time

async def fetch_data(source: str, delay: float) -> str:
    print(f"🔄 بدء تحميل البيانات من {source}...")
    await asyncio.sleep(delay)
    print(f"✅ انتهى تحميل البيانات من {source}")
    return f"بيانات من {source}"

async def main() -> None:
    start = time.perf_counter()

    results = await asyncio.gather(
        fetch_data("الخادم أ", 2),
        fetch_data("الخادم ب", 2),
        fetch_data("الخادم ج", 2),
    )

    elapsed = time.perf_counter() - start
    print(f"\nالنتائج: {results}")
    print(f"الوقت الإجمالي: {elapsed:.2f} ثانية")

asyncio.run(main())

# ────────────────────────────────────────────────────────────────────────
# [025] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-17-chapter-17.ipynb#cell-24
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السابع عشر: البرمجة غير المتزامنة - مقدمة عملية (async/await)
القسم 17.5: مثال عملي حقيقي: تحميل بيانات من الإنترنت بـ `aiohttp`

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 17-008-fetch_url.py
"""

import asyncio
import time
import aiohttp

async def fetch_url(session: aiohttp.ClientSession, url: str) -> int:
    """يُحمّل صفحة ويب ويُرجع رمز الحالة (Status Code) فقط، للتبسيط."""
    async with session.get(url) as response:
        return response.status

async def main() -> None:
    urls = [
        "https://www.python.org",
        "https://www.github.com",
        "https://www.wikipedia.org",
    ]

    start = time.perf_counter()

    async with aiohttp.ClientSession() as session:
        tasks = [fetch_url(session, url) for url in urls]
        results = await asyncio.gather(*tasks)

    elapsed = time.perf_counter() - start

    for url, status in zip(urls, results):
        print(f"{url} → {status}")

    print(f"\nالوقت الإجمالي: {elapsed:.2f} ثانية")

asyncio.run(main())

# ────────────────────────────────────────────────────────────────────────
# [026] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-17-chapter-17.ipynb#cell-25
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السابع عشر: البرمجة غير المتزامنة - مقدمة عملية (async/await)
القسم 17.6: `asyncio.create_task()`: بدء المهام دون انتظار فوري

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 17-009-background_job.py
"""

import asyncio

async def background_job(name: str, delay: float) -> str:
    print(f"🚀 بدأت المهمة: {name}")
    await asyncio.sleep(delay)
    print(f"🏁 انتهت المهمة: {name}")
    return f"نتيجة {name}"

async def main() -> None:
    # إنشاء المهمة تبدأ تنفيذها فورًا "في الخلفية"
    task1 = asyncio.create_task(background_job("أ", 3))
    task2 = asyncio.create_task(background_job("ب", 1))

    print("البرنامج يستمر بعمل أشياء أخرى بينما المهام تعمل في الخلفية...")
    await asyncio.sleep(0.5)
    print("لا يزال البرنامج يعمل...")

    # الآن ننتظر نتائج المهام عند الحاجة الفعلية إليها
    result1 = await task1
    result2 = await task2

    print(result1, result2)

asyncio.run(main())

# ────────────────────────────────────────────────────────────────────────
# [027] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-17-chapter-17.ipynb#cell-26
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السابع عشر: البرمجة غير المتزامنة - مقدمة عملية (async/await)
القسم 17.7: معالجة الأخطاء في الكود غير المتزامن

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 17-010-risky_operation.py
"""

import asyncio

async def risky_operation(should_fail: bool) -> str:
    await asyncio.sleep(1)
    if should_fail:
        raise ValueError("حدث خطأ أثناء العملية!")
    return "نجحت العملية"

async def main() -> None:
    try:
        result = await risky_operation(should_fail=True)
        print(result)
    except ValueError as e:
        print(f"تم اصطياد الخطأ: {e}")

asyncio.run(main())

# ────────────────────────────────────────────────────────────────────────
# [028] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-17-chapter-17.ipynb#cell-27
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السابع عشر: البرمجة غير المتزامنة - مقدمة عملية (async/await)
القسم 17.7: معالجة الأخطاء في الكود غير المتزامن

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 17-011-task_that_fails.py
"""

import asyncio

async def task_that_fails() -> str:
    await asyncio.sleep(1)
    raise ValueError("فشلت هذه المهمة!")

async def task_that_succeeds() -> str:
    await asyncio.sleep(1)
    return "نجحت هذه المهمة"

async def main() -> None:
    results = await asyncio.gather(
        task_that_succeeds(),
        task_that_fails(),
        return_exceptions=True,  # ← مهم جدًا!
    )

    for result in results:
        if isinstance(result, Exception):
            print(f"خطأ: {result}")
        else:
            print(f"نجاح: {result}")

asyncio.run(main())

# ────────────────────────────────────────────────────────────────────────
# [029] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-17-chapter-17.ipynb#cell-28
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [030] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-17-chapter-17.ipynb#cell-29
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السابع عشر: البرمجة غير المتزامنة - مقدمة عملية (async/await)
القسم 17.10: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 17-013-get_data.py
"""

async def get_data() -> str:
    return "بيانات"

def main():
    result = get_data()
    print(result)

main()

# ────────────────────────────────────────────────────────────────────────
# [031] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-17-chapter-17.ipynb#cell-30
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السابع عشر: البرمجة غير المتزامنة - مقدمة عملية (async/await)
القسم 17.10: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 17-014-countdown.py
"""

# تمرين 1
import asyncio

async def countdown(n: int) -> None:
    for i in range(n, 0, -1):
        print(i)
        await asyncio.sleep(1)
    print("انتهى!")

asyncio.run(countdown(5))

# تمرين 2
# await func(): ينفّذ الـ Coroutine وينتظر نتيجتها فورًا قبل المتابعة.
# asyncio.create_task(func()): يبدأ تنفيذ الـ Coroutine "في الخلفية" فورًا،
# ويسمح لك بمتابعة كود آخر، على أن تنتظر نتيجتها لاحقًا بـ await عند الحاجة.

# تمرين 3 - الكود المصحح
import asyncio

async def get_data() -> str:
    return "بيانات"

async def main() -> None:
    result = await get_data()  # يجب await لتنفيذ الـ Coroutine فعليًا
    print(result)

asyncio.run(main())  # يجب استدعاء main عبر asyncio.run، وليس مباشرة

# ────────────────────────────────────────────────────────────────────────
# [032] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-17-chapter-17.ipynb#cell-31
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السابع عشر: البرمجة غير المتزامنة - مقدمة عملية (async/await)
القسم 17.10: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 17-015-download_image.py
"""

# تمرين 4
import asyncio
import time

async def download_image() -> str:
    await asyncio.sleep(3)
    return "تم تحميل الصورة"

async def download_video() -> str:
    await asyncio.sleep(5)
    return "تم تحميل الفيديو"

async def download_document() -> str:
    await asyncio.sleep(1)
    return "تم تحميل المستند"

async def main() -> None:
    start = time.perf_counter()
    results = await asyncio.gather(
        download_image(), download_video(), download_document()
    )
    elapsed = time.perf_counter() - start
    print(results)
    print(f"الوقت: {elapsed:.2f} ثانية")  # قريب من 5 ثوانٍ

asyncio.run(main())

# تمرين 5
import asyncio

async def download_image() -> str:
    await asyncio.sleep(3)
    return "تم تحميل الصورة"

async def download_video() -> str:
    await asyncio.sleep(2)
    raise ConnectionError("فشل الاتصال بخادم الفيديو")

async def download_document() -> str:
    await asyncio.sleep(1)
    return "تم تحميل المستند"

async def main() -> None:
    results = await asyncio.gather(
        download_image(), download_video(), download_document(),
        return_exceptions=True,
    )
    for result in results:
        if isinstance(result, Exception):
            print(f"❌ فشل: {result}")
        else:
            print(f"✅ نجح: {result}")

asyncio.run(main())

# تمرين 6
import asyncio

async def fetch_with_timeout(delay: float, timeout: float) -> str:
    try:
        async def task() -> str:
            await asyncio.sleep(delay)
            return "تم التحميل بنجاح"
        return await asyncio.wait_for(task(), timeout=timeout)
    except asyncio.TimeoutError:
        return "انتهت المهلة"

async def main() -> None:
    print(await fetch_with_timeout(1, 3))   # "تم التحميل بنجاح"
    print(await fetch_with_timeout(5, 2))   # "انتهت المهلة"

asyncio.run(main())

# ────────────────────────────────────────────────────────────────────────
# [033] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-17-chapter-17.ipynb#cell-32
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [034] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-17-chapter-17.ipynb#cell-33
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السابع عشر: البرمجة غير المتزامنة - مقدمة عملية (async/await)
القسم 17.12: ملخص الفصل (Cheat Sheet)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 17-017-my_function.py
"""

import asyncio

# ===== تعريف وتشغيل Coroutine =====
async def my_function() -> str:
    await asyncio.sleep(1)   # انتظار غير متزامن (وليس time.sleep!)
    return "النتيجة"

asyncio.run(my_function())   # نقطة الدخول الرئيسية - تُستدعى مرة واحدة فقط

# ===== انتظار متتالٍ (عند الاعتماد الحقيقي بين العمليات) =====
async def main() -> None:
    result1 = await my_function()
    result2 = await my_function()  # يبدأ فقط بعد انتهاء الأولى

# ===== تزامن حقيقي (عمليات مستقلة) =====
async def main() -> None:
    results = await asyncio.gather(
        my_function(),
        my_function(),
        return_exceptions=True,  # لمنع توقف الكل عند فشل واحدة
    )

# ===== بدء مهمة في الخلفية =====
async def main() -> None:
    task = asyncio.create_task(my_function())
    # ... كود آخر يعمل الآن ...
    result = await task  # ننتظر النتيجة عند الحاجة فقط

# ===== فرض مهلة قصوى =====
try:
    result = await asyncio.wait_for(my_function(), timeout=2.0)
except asyncio.TimeoutError:
    print("انتهت المهلة")

# ===== معالجة الأخطاء =====
try:
    result = await risky_coroutine()
except SomeError as e:
    print(f"خطأ: {e}")

# ────────────────────────────────────────────────────────────────────────
# [035] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-17-chapter-17.ipynb#cell-34
# ────────────────────────────────────────────────────────────────────────
> def regular_function():
>     await something()  # ❌ SyntaxError: 'await' outside async function
> ```

### محاكاة الانتظار غير المتزامن: `asyncio.sleep()`

تذكر أننا استخدمنا `time.sleep()` في المثال الأول لمحاكاة بطء الشبكة. لكن `time.sleep()` هي دالة **متزامنة (Blocking)** — تُجمّد **كل البرنامج بالكامل**، ولا تسمح لأي كود آخر بالعمل أثناء انتظارها، حتى لو كان الكود الآخر Coroutine!

الحل: `asyncio.sleep()` — نسخة غير متزامنة من `sleep()`، مصممة خصيصًا للعمل مع `async/await`:

# ────────────────────────────────────────────────────────────────────────
# [036] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-17-chapter-17.ipynb#cell-35
# ────────────────────────────────────────────────────────────────────────
لاحظ النقاط التالية:
- `async with` تُستخدم مع `aiohttp.ClientSession()` — نسخة غير متزامنة من `with` التي تعلمناها في الفصل 16، تعمل مع Context Managers غير متزامنة (`__aenter__`/`__aexit__`).
- `tasks = [fetch_url(session, url) for url in urls]`: قائمة من Coroutines (لم تُشغَّل بعد).
- `*tasks`: نستخدم عامل فك التغليف (من الفصل 10) لتمرير كل عناصر القائمة كمعاملات منفصلة لـ `gather()`.

تشغيل هذا البرنامج سيُحمّل الصفحات الثلاث **في نفس الوقت تقريبًا**، بدلًا من تحميلها واحدة تلو الأخرى، مما يُقلل الوقت الإجمالي بشكل ملحوظ (خاصة مع عدد أكبر من الروابط).

> 💡 **نصيحة الخبير:** كل مكتبة I/O «عادية» (متزامنة) لها عادة «نسخة توأم» غير متزامنة: `requests` ↔ `aiohttp` أو `httpx` (التي تدعم الاثنين معًا)، `psycopg2` ↔ `asyncpg` لقواعد بيانات PostgreSQL، إلخ. عند العمل في بيئة `async` (مثل تطبيق FastAPI)، **يجب** استخدام النسخ غير المتزامنة من كل مكتبة، وإلا ستُجمّد استدعاءاتها المتزامنة حلقة الأحداث بالكامل وتُلغي كل فائدة `asyncio`!

> ⚠️ **خطأ فادح وشائع جدًا:** استخدام دالة متزامنة بطيئة (مثل `time.sleep()` أو `requests.get()`) داخل Coroutine! هذا يُجمّد **حلقة الأحداث بأكملها**، فتتوقف كل المهام الأخرى المنتظرة أيضًا، مما يُلغي كل فائدة `async` تمامًا:
> ```python
> import time
>
> async def bad_example() -> None:
>     time.sleep(2)  # ❌ خطأ فادح! يُجمّد حلقة الأحداث بأكملها
>
> async def good_example() -> None:
>     await asyncio.sleep(2)  # ✅ صحيح، يسمح بتبديل المهام
> ```

---

## 17.6 `asyncio.create_task()`: بدء المهام دون انتظار فوري

أحيانًا تريد أن «تبدأ» عملية غير متزامنة، لكن تريد الاستمرار في عمل آخر، والعودة لنتيجتها لاحقًا. هنا تأتي `asyncio.create_task()`:
