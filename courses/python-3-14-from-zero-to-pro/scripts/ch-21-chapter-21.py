#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
بايثون 3.14 — من الصفر إلى الاحتراف — مسار الأتمتة — استخراج البيانات من الويب (Web Scraping)
Python 3.14 — From Zero to Pro — Web

Book slug : python-3-14-from-zero-to-pro
Chapter   : ch-21-chapter-21 (lesson 21)
Code files: 47 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/python-3-14-from-zero-to-pro/code/ch-21-chapter-21.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
بايثون 3.14 — من الصفر إلى الاحتراف — مسار الأتمتة — استخراج البيانات من الويب (Web Scraping)
Python 3.14 — From Zero to Pro — Web

Book slug : python-3-14-from-zero-to-pro
Chapter   : ch-21-chapter-21 (lesson 21)
Code files: 46 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] dist/notebooks/chapter-21.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#@title ⚙️ تهيئة البيئة  { display-mode: "form" }
import sys, os, random, pathlib

print("Python:", sys.version.split()[0], "|", sys.platform)
assert sys.version_info >= (3, 10), "هذا الكتاب يفترض Python 3.10 أو أحدث"

SEED = 42
random.seed(SEED)

# تثبيت الحزم الناقصة فقط (سريع إن كانت مثبّتة أصلًا)
import importlib.util as _iu
_need = [m for m in ["requests", "bs4", "lxml"] if _iu.find_spec(m) is None]
if _need:
    # %pip install -q requests beautifulsoup4 lxml

# مجلد عمل معزول حتى لا تتناثر الملفات في جذر الدفتر
WORK = pathlib.Path("book_runtime")
WORK.mkdir(exist_ok=True)
os.chdir(WORK)
print("مجلد العمل:", pathlib.Path.cwd().name)

# إعداد خاص بهذا الفصل
# مهلة افتراضية لكل طلب حتى لا تعلّق الخلية إن تعذّر الوصول للشبكة
import requests as _rq
_orig_request = _rq.Session.request
def _request(self, method, url, **kw):
    kw.setdefault('timeout', 10)
    return _orig_request(self, method, url, **kw)
_rq.Session.request = _request

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
# [002] dist/notebooks/chapter-21.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import requests

response = requests.get("https://books.toscrape.com/")

print(response.status_code)   # 200 يعني نجاح (كما تعلمنا في الفصل 19)
print(response.text[:500])    # أول 500 حرف من محتوى HTML الخام

# ────────────────────────────────────────────────────────────────────────
# [003] dist/notebooks/chapter-21.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import requests

response = requests.get("https://books.toscrape.com/")

if response.status_code == 200:
    print("✅ نجح الطلب!")
elif response.status_code == 404:
    print("❌ الصفحة غير موجودة")
else:
    print(f"⚠️ حدث خطأ: {response.status_code}")

# طريقة أنظف وأكثر احترافية
try:
    response.raise_for_status()  # يرفع استثناء تلقائيًا إذا كان status_code خطأ
    print("✅ نجح الطلب!")
except requests.HTTPError as e:
    print(f"❌ خطأ HTTP: {e}")

# ────────────────────────────────────────────────────────────────────────
# [004] dist/notebooks/chapter-21.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import requests

headers = {
    "User-Agent": "Mozilla/5.0 (Educational Scraper Bot - contact@example.com)"
}

response = requests.get("https://books.toscrape.com/", headers=headers)
print(response.status_code)

# ────────────────────────────────────────────────────────────────────────
# [005] dist/notebooks/chapter-21.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
import requests

try:
    response = requests.get(
        "https://books.toscrape.com/",
        timeout=10,  # ثوانٍ -- إذا لم يرد الخادم خلالها، يُرفع استثناء
    )
except requests.Timeout:
    print("⏱️ انتهت المهلة الزمنية -- الخادم بطيء جدًا أو لا يستجيب")
except requests.ConnectionError:
    print("🔌 تعذّر الاتصال -- تحقق من الإنترنت أو الرابط")

# ────────────────────────────────────────────────────────────────────────
# [006] dist/notebooks/chapter-21.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
import requests

response = requests.get("https://api.github.com/users/python", timeout=10)
data = response.json()  # تحويل تلقائي من JSON إلى dict/list بايثون

print(data["name"])
print(data["public_repos"])

# ────────────────────────────────────────────────────────────────────────
# [007] dist/notebooks/chapter-21.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
import requests

# Session يحافظ تلقائيًا على الكوكيز بين الطلبات (مفيد بعد تسجيل الدخول مثلاً)
with requests.Session() as session:
    session.headers.update({"User-Agent": "Educational Bot"})

    login_response = session.post(
        "https://example.com/login",
        data={"user": "ahmad", "pass": "1234"},
        timeout=10,
    )

    # هذا الطلب سيستخدم نفس الكوكيز تلقائيًا
    profile_response = session.get("https://example.com/profile", timeout=10)

# ────────────────────────────────────────────────────────────────────────
# [008] dist/notebooks/chapter-21.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
import requests
from bs4 import BeautifulSoup

response = requests.get("https://books.toscrape.com/", timeout=10)
soup = BeautifulSoup(response.text, "lxml")  # "lxml" هو المحلل (parser) المستخدم

print(soup.title)        # <title>All products | Books to Scrape...</title>
print(soup.title.text)   # All products | Books to Scrape - Sandbox

# ────────────────────────────────────────────────────────────────────────
# [009] dist/notebooks/chapter-21.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
# العثور على أول عنصر مطابق
first_product = soup.find("article", class_="product_pod")

# العثور على كل العناصر المطابقة -> تُرجع قائمة
all_products = soup.find_all("article", class_="product_pod")
print(f"عدد المنتجات: {len(all_products)}")

# البحث بالوسم فقط
all_headings = soup.find_all("h3")

# البحث بخاصية id
header = soup.find(id="messages")

# البحث بعدة شروط
links = soup.find_all("a", href=True)  # كل روابط <a> التي تملك خاصية href

# ────────────────────────────────────────────────────────────────────────
# [010] dist/notebooks/chapter-21.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
product = soup.find("article", class_="product_pod")

title_tag = product.find("h3").find("a")
title = title_tag["title"]           # الوصول لخاصية HTML (title attribute)
# أو
title = title_tag.get("title")       # أكثر أمانًا -- يُرجع None إن لم توجد بدلاً من خطأ

price_tag = product.find("p", class_="price_color")
price_text = price_tag.text          # النص الظاهر للمستخدم: "£51.77"
price_text_stripped = price_tag.get_text(strip=True)  # يزيل المسافات الزائدة تلقائيًا

# ────────────────────────────────────────────────────────────────────────
# [011] dist/notebooks/chapter-21.ipynb#cell-11
# ────────────────────────────────────────────────────────────────────────
# محدد CSS: وسم.class
titles = soup.select("h3 a")               # كل وسم <a> داخل <h3>
prices = soup.select("p.price_color")       # كل <p> بخاصية class="price_color"
first_price = soup.select_one("p.price_color")  # أول عنصر مطابق فقط

# محددات أكثر تعقيدًا
all_product_links = soup.select("article.product_pod h3 a")

for link in all_product_links:
    print(link.get("title"), "->", link.get("href"))

# ────────────────────────────────────────────────────────────────────────
# [012] dist/notebooks/chapter-21.ipynb#cell-12
# ────────────────────────────────────────────────────────────────────────
product = soup.find("article", class_="product_pod")

print(product.parent)         # العنصر الأب
print(product.find_next_sibling())  # العنصر الشقيق التالي
print(product.contents)       # كل العناصر الفرعية المباشرة كقائمة

# ────────────────────────────────────────────────────────────────────────
# [013] dist/notebooks/chapter-21.ipynb#cell-13
# ────────────────────────────────────────────────────────────────────────
import requests
from bs4 import BeautifulSoup

url = "https://books.toscrape.com/"
headers = {"User-Agent": "Educational Scraper (learning purposes)"}

response = requests.get(url, headers=headers, timeout=10)
response.raise_for_status()

soup = BeautifulSoup(response.text, "lxml")

books = soup.select("article.product_pod")
print(f"تم العثور على {len(books)} كتاب في الصفحة")

for book in books[:3]:  # أول 3 كتب فقط للعرض
    title = book.select_one("h3 a").get("title")
    price = book.select_one("p.price_color").get_text(strip=True)
    availability = book.select_one("p.instock.availability").get_text(strip=True)
    rating_classes = book.select_one("p.star-rating")["class"]
    rating = [c for c in rating_classes if c != "star-rating"][0]  # مثل "Three"

    print(f"📖 {title}")
    print(f"   السعر: {price}")
    print(f"   التوفر: {availability}")
    print(f"   التقييم: {rating}")
    print("-" * 40)

# ────────────────────────────────────────────────────────────────────────
# [014] dist/notebooks/chapter-21.ipynb#cell-14
# ────────────────────────────────────────────────────────────────────────
import time

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://books.toscrape.com/catalogue/page-{}.html"
headers = {"User-Agent": "Educational Scraper"}

all_titles = []
page = 1

while True:
    url = BASE_URL.format(page)
    response = requests.get(url, headers=headers, timeout=10)

    if response.status_code == 404:
        print(f"انتهت الصفحات عند الصفحة {page - 1}")
        break

    soup = BeautifulSoup(response.text, "lxml")
    books = soup.select("article.product_pod")

    for book in books:
        title = book.select_one("h3 a").get("title")
        all_titles.append(title)

    print(f"✅ تمت معالجة الصفحة {page} ({len(books)} كتاب)")
    page += 1
    time.sleep(1)  # ⚠️ احترام الخادم -- تأخير بين الطلبات إلزامي

print(f"\nإجمالي الكتب المُستخرجة: {len(all_titles)}")

# ────────────────────────────────────────────────────────────────────────
# [015] dist/notebooks/chapter-21.ipynb#cell-15
# ────────────────────────────────────────────────────────────────────────
import csv

data = [
    {"title": "A Light in the Attic", "price": "£51.77", "rating": "Three"},
    {"title": "Tipping the Velvet", "price": "£53.74", "rating": "One"},
]

with open("books.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["title", "price", "rating"])
    writer.writeheader()
    writer.writerows(data)

# ────────────────────────────────────────────────────────────────────────
# [016] dist/notebooks/chapter-21.ipynb#cell-16
# ────────────────────────────────────────────────────────────────────────
import pandas as pd

df = pd.DataFrame(data)
df.to_csv("books.csv", index=False, encoding="utf-8-sig")
print(df.describe())

# ────────────────────────────────────────────────────────────────────────
# [017] dist/notebooks/chapter-21.ipynb#cell-17
# ────────────────────────────────────────────────────────────────────────
import time

import requests
from bs4 import BeautifulSoup

def fetch_page(url: str, retries: int = 3) -> requests.Response | None:
    """جلب صفحة مع إعادة محاولة عند الفشل.

    Args:
        url: الرابط المطلوب جلبه.
        retries: عدد محاولات إعادة الطلب عند الفشل.

    Returns:
        كائن Response عند النجاح، أو None عند فشل كل المحاولات.
    """
    headers = {"User-Agent": "Educational Scraper"}

    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            return response
        except requests.Timeout:
            print(f"⏱️ محاولة {attempt}: انتهت المهلة الزمنية لـ {url}")
        except requests.HTTPError as e:
            print(f"❌ محاولة {attempt}: خطأ HTTP {e}")
        except requests.ConnectionError:
            print(f"🔌 محاولة {attempt}: تعذّر الاتصال بـ {url}")

        if attempt < retries:
            time.sleep(2 * attempt)  # تأخير متزايد (Exponential backoff مبسّط)

    print(f"🚫 فشلت كل المحاولات لـ {url}")
    return None

def extract_book_data(book_element) -> dict[str, str] | None:
    """استخراج بيانات كتاب واحد بأمان، مع التعامل مع عناصر مفقودة.

    Args:
        book_element: عنصر BeautifulSoup يمثّل بطاقة كتاب واحدة.

    Returns:
        قاموس ببيانات الكتاب، أو None إذا كانت البيانات الأساسية مفقودة.
    """
    title_tag = book_element.select_one("h3 a")
    price_tag = book_element.select_one("p.price_color")

    if title_tag is None or price_tag is None:
        return None  # تخطَّ هذا العنصر بدلاً من رفع خطأ يوقف البرنامج كله

    return {
        "title": title_tag.get("title", "غير معروف"),
        "price": price_tag.get_text(strip=True),
    }

# ────────────────────────────────────────────────────────────────────────
# [018] dist/notebooks/chapter-21.ipynb#cell-18
# ────────────────────────────────────────────────────────────────────────
import requests
from bs4 import BeautifulSoup

response = requests.get("https://books.toscrape.com/", timeout=10)
print(response.status_code)
print(len(response.text))

soup = BeautifulSoup(response.text, "lxml")
print(soup.title.get_text(strip=True))

links = soup.find_all("a", href=True)
for link in links[:5]:
    print(link["href"])

# ────────────────────────────────────────────────────────────────────────
# [019] dist/notebooks/chapter-21.ipynb#cell-19
# ────────────────────────────────────────────────────────────────────────
import asyncio

import aiohttp

SEMAPHORE_LIMIT = 3  # عدد الطلبات المتزامنة القصوى -- احترامًا للخادم

async def fetch_page_async(session: aiohttp.ClientSession, url: str, sem: asyncio.Semaphore) -> str | None:
    """جلب صفحة بشكل غير متزامن مع تحديد التزامن عبر Semaphore."""
    async with sem:
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status == 404:
                    return None
                response.raise_for_status()
                await asyncio.sleep(1)  # تأخير مهذّب حتى في الوضع المتزامن
                return await response.text()
        except aiohttp.ClientError as e:
            print(f"خطأ: {e}")
            return None

async def scrape_pages_async(urls: list[str]) -> list[str | None]:
    """جلب عدة صفحات بالتوازي مع تحديد عدد الطلبات المتزامنة."""
    sem = asyncio.Semaphore(SEMAPHORE_LIMIT)
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_page_async(session, url, sem) for url in urls]
        return await asyncio.gather(*tasks)

# urls = [BASE_URL.format(i) for i in range(1, 6)]
# results = asyncio.run(scrape_pages_async(urls))

# ────────────────────────────────────────────────────────────────────────
# [020] code/chapters/ch21/21-001-response.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي والعشرون: مسار الأتمتة — استخراج البيانات من الويب (Web Scraping)
القسم 21.4: مكتبة `requests`: طلب الصفحات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 21-001-response.py
"""

import requests

response = requests.get("https://books.toscrape.com/")

print(response.status_code)   # 200 يعني نجاح (كما تعلمنا في الفصل 19)
print(response.text[:500])    # أول 500 حرف من محتوى HTML الخام

# ────────────────────────────────────────────────────────────────────────
# [021] code/chapters/ch21/21-002-response.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي والعشرون: مسار الأتمتة — استخراج البيانات من الويب (Web Scraping)
القسم 21.4: مكتبة `requests`: طلب الصفحات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 21-002-response.py
"""

import requests

response = requests.get("https://books.toscrape.com/")

if response.status_code == 200:
    print("✅ نجح الطلب!")
elif response.status_code == 404:
    print("❌ الصفحة غير موجودة")
else:
    print(f"⚠️ حدث خطأ: {response.status_code}")

# طريقة أنظف وأكثر احترافية
try:
    response.raise_for_status()  # يرفع استثناء تلقائيًا إذا كان status_code خطأ
    print("✅ نجح الطلب!")
except requests.HTTPError as e:
    print(f"❌ خطأ HTTP: {e}")

# ────────────────────────────────────────────────────────────────────────
# [022] code/chapters/ch21/21-003-headers.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي والعشرون: مسار الأتمتة — استخراج البيانات من الويب (Web Scraping)
القسم 21.4: مكتبة `requests`: طلب الصفحات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 21-003-headers.py
"""

import requests

headers = {
    "User-Agent": "Mozilla/5.0 (Educational Scraper Bot - contact@example.com)"
}

response = requests.get("https://books.toscrape.com/", headers=headers)
print(response.status_code)

# ────────────────────────────────────────────────────────────────────────
# [023] code/chapters/ch21/21-004-response.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي والعشرون: مسار الأتمتة — استخراج البيانات من الويب (Web Scraping)
القسم 21.4: مكتبة `requests`: طلب الصفحات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 21-004-response.py
"""

import requests

try:
    response = requests.get(
        "https://books.toscrape.com/",
        timeout=10,  # ثوانٍ -- إذا لم يرد الخادم خلالها، يُرفع استثناء
    )
except requests.Timeout:
    print("⏱️ انتهت المهلة الزمنية -- الخادم بطيء جدًا أو لا يستجيب")
except requests.ConnectionError:
    print("🔌 تعذّر الاتصال -- تحقق من الإنترنت أو الرابط")

# ────────────────────────────────────────────────────────────────────────
# [024] code/chapters/ch21/21-005-response.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي والعشرون: مسار الأتمتة — استخراج البيانات من الويب (Web Scraping)
القسم 21.4: مكتبة `requests`: طلب الصفحات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 21-005-response.py
"""

import requests

# GET مع query parameters (مثل الفصل 19)
response = requests.get(
    "https://api.example.com/search",
    params={"q": "python", "page": 1},
    timeout=10,
)
# الرابط الفعلي سيكون: https://api.example.com/search?q=python&page=1

# POST لإرسال بيانات (مثلاً تسجيل الدخول أو إرسال نموذج)
response = requests.post(
    "https://api.example.com/login",
    json={"username": "ahmad", "password": "secret"},
    timeout=10,
)

# ────────────────────────────────────────────────────────────────────────
# [025] code/chapters/ch21/21-006-response.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي والعشرون: مسار الأتمتة — استخراج البيانات من الويب (Web Scraping)
القسم 21.4: مكتبة `requests`: طلب الصفحات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 21-006-response.py
"""

import requests

response = requests.get("https://api.github.com/users/python", timeout=10)
data = response.json()  # تحويل تلقائي من JSON إلى dict/list بايثون

print(data["name"])
print(data["public_repos"])

# ────────────────────────────────────────────────────────────────────────
# [026] code/chapters/ch21/21-007-login_response.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي والعشرون: مسار الأتمتة — استخراج البيانات من الويب (Web Scraping)
القسم 21.4: مكتبة `requests`: طلب الصفحات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 21-007-login_response.py
"""

import requests

# Session يحافظ تلقائيًا على الكوكيز بين الطلبات (مفيد بعد تسجيل الدخول مثلاً)
with requests.Session() as session:
    session.headers.update({"User-Agent": "Educational Bot"})

    login_response = session.post(
        "https://example.com/login",
        data={"user": "ahmad", "pass": "1234"},
        timeout=10,
    )

    # هذا الطلب سيستخدم نفس الكوكيز تلقائيًا
    profile_response = session.get("https://example.com/profile", timeout=10)

# ────────────────────────────────────────────────────────────────────────
# [027] code/chapters/ch21/21-008-response.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي والعشرون: مسار الأتمتة — استخراج البيانات من الويب (Web Scraping)
القسم 21.5: مكتبة BeautifulSoup: تحليل واستخراج البيانات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 21-008-response.py
"""

import requests
from bs4 import BeautifulSoup

response = requests.get("https://books.toscrape.com/", timeout=10)
soup = BeautifulSoup(response.text, "lxml")  # "lxml" هو المحلل (parser) المستخدم

print(soup.title)        # <title>All products | Books to Scrape...</title>
print(soup.title.text)   # All products | Books to Scrape - Sandbox

# ────────────────────────────────────────────────────────────────────────
# [028] code/chapters/ch21/21-009-first_product.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي والعشرون: مسار الأتمتة — استخراج البيانات من الويب (Web Scraping)
القسم 21.5: مكتبة BeautifulSoup: تحليل واستخراج البيانات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 21-009-first_product.py
"""

# العثور على أول عنصر مطابق
first_product = soup.find("article", class_="product_pod")

# العثور على كل العناصر المطابقة -> تُرجع قائمة
all_products = soup.find_all("article", class_="product_pod")
print(f"عدد المنتجات: {len(all_products)}")

# البحث بالوسم فقط
all_headings = soup.find_all("h3")

# البحث بخاصية id
header = soup.find(id="messages")

# البحث بعدة شروط
links = soup.find_all("a", href=True)  # كل روابط <a> التي تملك خاصية href

# ────────────────────────────────────────────────────────────────────────
# [029] code/chapters/ch21/21-010-product.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي والعشرون: مسار الأتمتة — استخراج البيانات من الويب (Web Scraping)
القسم 21.5: مكتبة BeautifulSoup: تحليل واستخراج البيانات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 21-010-product.py
"""

product = soup.find("article", class_="product_pod")

title_tag = product.find("h3").find("a")
title = title_tag["title"]           # الوصول لخاصية HTML (title attribute)
# أو
title = title_tag.get("title")       # أكثر أمانًا -- يُرجع None إن لم توجد بدلاً من خطأ

price_tag = product.find("p", class_="price_color")
price_text = price_tag.text          # النص الظاهر للمستخدم: "£51.77"
price_text_stripped = price_tag.get_text(strip=True)  # يزيل المسافات الزائدة تلقائيًا

# ────────────────────────────────────────────────────────────────────────
# [030] code/chapters/ch21/21-011-titles.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي والعشرون: مسار الأتمتة — استخراج البيانات من الويب (Web Scraping)
القسم 21.5: مكتبة BeautifulSoup: تحليل واستخراج البيانات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 21-011-titles.py
"""

# محدد CSS: وسم.class
titles = soup.select("h3 a")               # كل وسم <a> داخل <h3>
prices = soup.select("p.price_color")       # كل <p> بخاصية class="price_color"
first_price = soup.select_one("p.price_color")  # أول عنصر مطابق فقط

# محددات أكثر تعقيدًا
all_product_links = soup.select("article.product_pod h3 a")

for link in all_product_links:
    print(link.get("title"), "->", link.get("href"))

# ────────────────────────────────────────────────────────────────────────
# [031] code/chapters/ch21/21-012-product.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي والعشرون: مسار الأتمتة — استخراج البيانات من الويب (Web Scraping)
القسم 21.5: مكتبة BeautifulSoup: تحليل واستخراج البيانات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 21-012-product.py
"""

product = soup.find("article", class_="product_pod")

print(product.parent)         # العنصر الأب
print(product.find_next_sibling())  # العنصر الشقيق التالي
print(product.contents)       # كل العناصر الفرعية المباشرة كقائمة

# ────────────────────────────────────────────────────────────────────────
# [032] code/chapters/ch21/21-013-url.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي والعشرون: مسار الأتمتة — استخراج البيانات من الويب (Web Scraping)
القسم 21.6: نمط العمل الكامل: من الطلب إلى البيانات المنظّمة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 21-013-url.py
"""

import requests
from bs4 import BeautifulSoup

url = "https://books.toscrape.com/"
headers = {"User-Agent": "Educational Scraper (learning purposes)"}

response = requests.get(url, headers=headers, timeout=10)
response.raise_for_status()

soup = BeautifulSoup(response.text, "lxml")

books = soup.select("article.product_pod")
print(f"تم العثور على {len(books)} كتاب في الصفحة")

for book in books[:3]:  # أول 3 كتب فقط للعرض
    title = book.select_one("h3 a").get("title")
    price = book.select_one("p.price_color").get_text(strip=True)
    availability = book.select_one("p.instock.availability").get_text(strip=True)
    rating_classes = book.select_one("p.star-rating")["class"]
    rating = [c for c in rating_classes if c != "star-rating"][0]  # مثل "Three"

    print(f"📖 {title}")
    print(f"   السعر: {price}")
    print(f"   التوفر: {availability}")
    print(f"   التقييم: {rating}")
    print("-" * 40)

# ────────────────────────────────────────────────────────────────────────
# [033] code/chapters/ch21/21-014-base_url.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي والعشرون: مسار الأتمتة — استخراج البيانات من الويب (Web Scraping)
القسم 21.7: التعامل مع التصفّح متعدد الصفحات (Pagination)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 21-014-base_url.py
"""

import time

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://books.toscrape.com/catalogue/page-{}.html"
headers = {"User-Agent": "Educational Scraper"}

all_titles = []
page = 1

while True:
    url = BASE_URL.format(page)
    response = requests.get(url, headers=headers, timeout=10)

    if response.status_code == 404:
        print(f"انتهت الصفحات عند الصفحة {page - 1}")
        break

    soup = BeautifulSoup(response.text, "lxml")
    books = soup.select("article.product_pod")

    for book in books:
        title = book.select_one("h3 a").get("title")
        all_titles.append(title)

    print(f"✅ تمت معالجة الصفحة {page} ({len(books)} كتاب)")
    page += 1
    time.sleep(1)  # ⚠️ احترام الخادم -- تأخير بين الطلبات إلزامي

print(f"\nإجمالي الكتب المُستخرجة: {len(all_titles)}")

# ────────────────────────────────────────────────────────────────────────
# [034] code/chapters/ch21/21-015-data.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي والعشرون: مسار الأتمتة — استخراج البيانات من الويب (Web Scraping)
القسم 21.8: حفظ البيانات المُستخرجة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 21-015-data.py
"""

import csv

data = [
    {"title": "A Light in the Attic", "price": "£51.77", "rating": "Three"},
    {"title": "Tipping the Velvet", "price": "£53.74", "rating": "One"},
]

with open("books.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["title", "price", "rating"])
    writer.writeheader()
    writer.writerows(data)

# ────────────────────────────────────────────────────────────────────────
# [035] code/chapters/ch21/21-016-ex21_8.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي والعشرون: مسار الأتمتة — استخراج البيانات من الويب (Web Scraping)
القسم 21.8: حفظ البيانات المُستخرجة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 21-016-ex21_8.py
"""

import pandas as pd

df = pd.DataFrame(data)
df.to_csv("books.csv", index=False, encoding="utf-8-sig")
print(df.describe())

# ────────────────────────────────────────────────────────────────────────
# [036] code/chapters/ch21/21-017-fetch_page.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي والعشرون: مسار الأتمتة — استخراج البيانات من الويب (Web Scraping)
القسم 21.9: معالجة الأخطاء بصلابة (Robust Scraping)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 21-017-fetch_page.py
"""

import time

import requests
from bs4 import BeautifulSoup

def fetch_page(url: str, retries: int = 3) -> requests.Response | None:
    """جلب صفحة مع إعادة محاولة عند الفشل.

    Args:
        url: الرابط المطلوب جلبه.
        retries: عدد محاولات إعادة الطلب عند الفشل.

    Returns:
        كائن Response عند النجاح، أو None عند فشل كل المحاولات.
    """
    headers = {"User-Agent": "Educational Scraper"}

    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            return response
        except requests.Timeout:
            print(f"⏱️ محاولة {attempt}: انتهت المهلة الزمنية لـ {url}")
        except requests.HTTPError as e:
            print(f"❌ محاولة {attempt}: خطأ HTTP {e}")
        except requests.ConnectionError:
            print(f"🔌 محاولة {attempt}: تعذّر الاتصال بـ {url}")

        if attempt < retries:
            time.sleep(2 * attempt)  # تأخير متزايد (Exponential backoff مبسّط)

    print(f"🚫 فشلت كل المحاولات لـ {url}")
    return None

def extract_book_data(book_element) -> dict[str, str] | None:
    """استخراج بيانات كتاب واحد بأمان، مع التعامل مع عناصر مفقودة.

    Args:
        book_element: عنصر BeautifulSoup يمثّل بطاقة كتاب واحدة.

    Returns:
        قاموس ببيانات الكتاب، أو None إذا كانت البيانات الأساسية مفقودة.
    """
    title_tag = book_element.select_one("h3 a")
    price_tag = book_element.select_one("p.price_color")

    if title_tag is None or price_tag is None:
        return None  # تخطَّ هذا العنصر بدلاً من رفع خطأ يوقف البرنامج كله

    return {
        "title": title_tag.get("title", "غير معروف"),
        "price": price_tag.get_text(strip=True),
    }

# ────────────────────────────────────────────────────────────────────────
# [037] code/chapters/ch21/21-018-fetch_page.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي والعشرون: مسار الأتمتة — استخراج البيانات من الويب (Web Scraping)
القسم 21.11: مشروع تطبيقي متكامل: مستخرج بيانات كتب شامل

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 21-018-fetch_page.py
"""

"""وحدة جلب صفحات الويب مع معالجة الأخطاء وإعادة المحاولة."""

import time

import requests

USER_AGENT = "Educational Book Scraper (learning purposes, github.com/example)"
REQUEST_TIMEOUT = 10
DELAY_BETWEEN_REQUESTS = 1.0  # ثانية -- احترام الخادم

def fetch_page(url: str, retries: int = 3) -> str | None:
    """جلب محتوى HTML لصفحة، مع إعادة محاولة عند الفشل.

    Args:
        url: رابط الصفحة المطلوبة.
        retries: عدد محاولات إعادة الطلب عند حدوث خطأ مؤقت.

    Returns:
        محتوى HTML كنص، أو None إذا فشلت كل المحاولات أو الصفحة غير موجودة.
    """
    headers = {"User-Agent": USER_AGENT}

    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)

            if response.status_code == 404:
                return None  # نهاية الصفحات -- ليس خطأ، بل إشارة توقف طبيعية

            response.raise_for_status()
            return response.text

        except requests.Timeout:
            print(f"⏱️ [محاولة {attempt}/{retries}] انتهت المهلة الزمنية: {url}")
        except requests.ConnectionError:
            print(f"🔌 [محاولة {attempt}/{retries}] تعذّر الاتصال: {url}")
        except requests.HTTPError as e:
            print(f"❌ [محاولة {attempt}/{retries}] خطأ HTTP: {e}")

        if attempt < retries:
            time.sleep(2 * attempt)

    print(f"🚫 فشلت كل المحاولات لجلب: {url}")
    return None

def polite_delay() -> None:
    """تأخير مهذّب بين الطلبات المتتالية لاحترام خادم الموقع."""
    time.sleep(DELAY_BETWEEN_REQUESTS)

# ────────────────────────────────────────────────────────────────────────
# [038] code/chapters/ch21/21-019-_parse_price.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي والعشرون: مسار الأتمتة — استخراج البيانات من الويب (Web Scraping)
القسم 21.11: مشروع تطبيقي متكامل: مستخرج بيانات كتب شامل

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 21-019-_parse_price.py
"""

"""وحدة تحليل صفحات الكتب واستخراج البيانات منها."""

from dataclasses import dataclass

from bs4 import BeautifulSoup

RATING_WORDS_TO_NUMBERS = {
    "One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5,
}

@dataclass
class Book:
    """يمثّل بيانات كتاب واحد مُستخرجة من الموقع."""

    title: str
    price: float
    rating: int
    availability: str

def _parse_price(price_text: str) -> float:
    """تحويل نص السعر مثل '£51.77' إلى رقم عشري."""
    cleaned = price_text.replace("£", "").strip()
    return float(cleaned)

def _parse_rating(star_rating_tag) -> int:
    """استخراج رقم التقييم من اسم الـ class (مثل 'Three' -> 3)."""
    classes = star_rating_tag.get("class", [])
    for cls in classes:
        if cls in RATING_WORDS_TO_NUMBERS:
            return RATING_WORDS_TO_NUMBERS[cls]
    return 0

def extract_books_from_page(html: str) -> list[Book]:
    """استخراج كل الكتب من صفحة HTML واحدة.

    Args:
        html: محتوى HTML الخام للصفحة.

    Returns:
        قائمة من كائنات Book، مع تخطي أي عنصر بيانات ناقصة.
    """
    soup = BeautifulSoup(html, "lxml")
    book_elements = soup.select("article.product_pod")

    books: list[Book] = []

    for element in book_elements:
        title_tag = element.select_one("h3 a")
        price_tag = element.select_one("p.price_color")
        rating_tag = element.select_one("p.star-rating")
        availability_tag = element.select_one("p.instock.availability")

        if not all([title_tag, price_tag, rating_tag, availability_tag]):
            continue  # تخطَّ أي كتاب ببيانات ناقصة بدلاً من إيقاف البرنامج

        book = Book(
            title=title_tag.get("title", "غير معروف"),
            price=_parse_price(price_tag.get_text(strip=True)),
            rating=_parse_rating(rating_tag),
            availability=availability_tag.get_text(strip=True),
        )
        books.append(book)

    return books

def has_next_page(html: str) -> bool:
    """التحقق من وجود صفحة تالية في نظام التصفّح."""
    soup = BeautifulSoup(html, "lxml")
    return soup.select_one("li.next") is not None

# ────────────────────────────────────────────────────────────────────────
# [039] code/chapters/ch21/21-020-save_books_to_csv.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي والعشرون: مسار الأتمتة — استخراج البيانات من الويب (Web Scraping)
القسم 21.11: مشروع تطبيقي متكامل: مستخرج بيانات كتب شامل

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 21-020-save_books_to_csv.py
"""

"""وحدة حفظ بيانات الكتب المُستخرجة."""

from dataclasses import asdict
from pathlib import Path

import pandas as pd

from book_scraper.parser import Book

def save_books_to_csv(books: list[Book], file_path: str | Path) -> None:
    """حفظ قائمة الكتب في ملف CSV باستخدام pandas.

    Args:
        books: قائمة كائنات Book المُستخرجة.
        file_path: مسار ملف الحفظ.
    """
    records = [asdict(book) for book in books]
    df = pd.DataFrame(records)
    df.to_csv(file_path, index=False, encoding="utf-8-sig")

# ────────────────────────────────────────────────────────────────────────
# [040] code/chapters/ch21/21-021-scrape_all_books.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي والعشرون: مسار الأتمتة — استخراج البيانات من الويب (Web Scraping)
القسم 21.11: مشروع تطبيقي متكامل: مستخرج بيانات كتب شامل

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 21-021-scrape_all_books.py
"""

"""نقطة الدخول الرئيسية لأداة استخراج بيانات الكتب."""

from pathlib import Path

import pandas as pd

from book_scraper.fetcher import fetch_page, polite_delay
from book_scraper.parser import Book, extract_books_from_page, has_next_page
from book_scraper.storage import save_books_to_csv

BASE_URL = "https://books.toscrape.com/catalogue/page-{}.html"
OUTPUT_FILE = Path("output/books.csv")
MAX_PAGES = 5  # نحدد سقفًا احترازيًا لتجنب حلقة لا نهائية في حال خطأ غير متوقع

def scrape_all_books() -> list[Book]:
    """استخراج كل الكتب من كل الصفحات المتاحة.

    Returns:
        قائمة شاملة بكل الكتب المُستخرجة من كل الصفحات.
    """
    all_books: list[Book] = []
    page_number = 1

    while page_number <= MAX_PAGES:
        url = BASE_URL.format(page_number)
        print(f"📄 جلب الصفحة {page_number}...")

        html = fetch_page(url)
        if html is None:
            print("🏁 لا مزيد من الصفحات أو حدث خطأ غير قابل للتعافي. توقف.")
            break

        books_in_page = extract_books_from_page(html)
        all_books.extend(books_in_page)
        print(f"   ✅ تم استخراج {len(books_in_page)} كتاب")

        if not has_next_page(html):
            print("🏁 تم الوصول لآخر صفحة.")
            break

        page_number += 1
        polite_delay()

    return all_books

def print_analysis(books: list[Book]) -> None:
    """طباعة تحليل سريع لبيانات الكتب المُستخرجة."""
    if not books:
        print("لا توجد بيانات للتحليل.")
        return

    df = pd.DataFrame([{"title": b.title, "price": b.price, "rating": b.rating} for b in books])

    print("\n" + "=" * 50)
    print(" " * 12 + "تحليل بيانات الكتب المُستخرجة")
    print("=" * 50)
    print(f"إجمالي الكتب: {len(df)}")
    print(f"متوسط السعر: £{df['price'].mean():.2f}")
    print(f"أغلى كتاب: £{df['price'].max():.2f}")
    print(f"أرخص كتاب: £{df['price'].min():.2f}")
    print(f"متوسط التقييم: {df['rating'].mean():.1f} / 5")
    print("=" * 50)

def main() -> None:
    """تشغيل عملية الاستخراج الكاملة."""
    print("🚀 بدء استخراج بيانات الكتب من books.toscrape.com\n")

    books = scrape_all_books()

    print(f"\n💾 حفظ {len(books)} كتاب في {OUTPUT_FILE}...")
    OUTPUT_FILE.parent.mkdir(exist_ok=True)
    save_books_to_csv(books, OUTPUT_FILE)

    print_analysis(books)
    print(f"\n✅ اكتمل! راجع الملف: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()

# ────────────────────────────────────────────────────────────────────────
# [041] code/chapters/ch21/21-022-response.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي والعشرون: مسار الأتمتة — استخراج البيانات من الويب (Web Scraping)
القسم 21.12: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 21-022-response.py
"""

import requests
from bs4 import BeautifulSoup

response = requests.get("https://books.toscrape.com/", timeout=10)
print(response.status_code)
print(len(response.text))

soup = BeautifulSoup(response.text, "lxml")
print(soup.title.get_text(strip=True))

links = soup.find_all("a", href=True)
for link in links[:5]:
    print(link["href"])

# ────────────────────────────────────────────────────────────────────────
# [042] code/chapters/ch21/21-023-filter_books_by_rating.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي والعشرون: مسار الأتمتة — استخراج البيانات من الويب (Web Scraping)
القسم 21.12: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 21-023-filter_books_by_rating.py
"""

# تمرين 5 -- سهل التطبيق مباشرة على كائنات Book
def filter_books_by_rating(books: list[Book], min_rating: int) -> list[Book]:
    """إرجاع الكتب التي تقييمها أكبر من أو يساوي الحد الأدنى المطلوب."""
    return [book for book in books if book.rating >= min_rating]

# مثال استخدام
high_rated = filter_books_by_rating(books, min_rating=4)
print(f"عدد الكتب ذات التقييم 4+: {len(high_rated)}")

# ────────────────────────────────────────────────────────────────────────
# [043] code/chapters/ch21/21-024-fetch_page_async.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي والعشرون: مسار الأتمتة — استخراج البيانات من الويب (Web Scraping)
القسم 21.12: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 21-024-fetch_page_async.py
"""

import asyncio

import aiohttp

SEMAPHORE_LIMIT = 3  # عدد الطلبات المتزامنة القصوى -- احترامًا للخادم

async def fetch_page_async(session: aiohttp.ClientSession, url: str, sem: asyncio.Semaphore) -> str | None:
    """جلب صفحة بشكل غير متزامن مع تحديد التزامن عبر Semaphore."""
    async with sem:
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status == 404:
                    return None
                response.raise_for_status()
                await asyncio.sleep(1)  # تأخير مهذّب حتى في الوضع المتزامن
                return await response.text()
        except aiohttp.ClientError as e:
            print(f"خطأ: {e}")
            return None

async def scrape_pages_async(urls: list[str]) -> list[str | None]:
    """جلب عدة صفحات بالتوازي مع تحديد عدد الطلبات المتزامنة."""
    sem = asyncio.Semaphore(SEMAPHORE_LIMIT)
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_page_async(session, url, sem) for url in urls]
        return await asyncio.gather(*tasks)

# urls = [BASE_URL.format(i) for i in range(1, 6)]
# results = asyncio.run(scrape_pages_async(urls))

# ────────────────────────────────────────────────────────────────────────
# [044] code/chapters/ch21/21-025-response.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي والعشرون: مسار الأتمتة — استخراج البيانات من الويب (Web Scraping)
القسم 21.14: ملخص الفصل (Cheat Sheet)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 21-025-response.py
"""

# ===== requests: الطلبات الأساسية =====
import requests
response = requests.get(url, headers={"User-Agent": "..."}, timeout=10)
response.status_code         # 200, 404, 500...
response.text                 # محتوى HTML/نص خام
response.json()               # تحويل تلقائي من JSON
response.raise_for_status()   # يرفع استثناء عند 4xx/5xx

try:
    response = requests.get(url, timeout=10)
except requests.Timeout:
    ...
except requests.ConnectionError:
    ...

# ===== BeautifulSoup: التحليل =====
from bs4 import BeautifulSoup
soup = BeautifulSoup(response.text, "lxml")

soup.find("div", class_="price")       # أول عنصر مطابق
soup.find_all("article")               # كل العناصر المطابقة -> قائمة
soup.select("article.product_pod h3 a")  # محدد CSS -> قائمة
soup.select_one("title")               # محدد CSS -> أول عنصر

tag.text / tag.get_text(strip=True)    # النص الظاهر
tag.get("href")                        # خاصية HTML بأمان (None إن لم توجد)
tag["href"]                            # خاصية HTML (يرفع خطأ إن لم توجد)

# ===== الأتمتة الآمنة =====
import time
time.sleep(1)                          # تأخير بين الطلبات (إلزامي أخلاقيًا)

# ────────────────────────────────────────────────────────────────────────
# [045] dist/markdown/chapter-21.md
# ────────────────────────────────────────────────────────────────────────
import requests

# GET مع query parameters (مثل الفصل 19)
response = requests.get(
    "https://api.example.com/search",
    params={"q": "python", "page": 1},
    timeout=10,
)
# الرابط الفعلي سيكون: https://api.example.com/search?q=python&page=1

# POST لإرسال بيانات (مثلاً تسجيل الدخول أو إرسال نموذج)
response = requests.post(
    "https://api.example.com/login",
    json={"username": "ahmad", "password": "secret"},
    timeout=10,
)

# ────────────────────────────────────────────────────────────────────────
# [046] dist/markdown/chapter-21.md
# ────────────────────────────────────────────────────────────────────────
> soup.find("div", class_="price")   # ✅ صحيح
> ```

### استخراج النص والخصائص

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-21-chapter-21.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#@title ⚙️ تهيئة البيئة  { display-mode: "form" }
import sys, os, random, pathlib

print("Python:", sys.version.split()[0], "|", sys.platform)
assert sys.version_info >= (3, 10), "هذا الكتاب يفترض Python 3.10 أو أحدث"

SEED = 42
random.seed(SEED)

# تثبيت الحزم الناقصة فقط (سريع إن كانت مثبّتة أصلًا)
import importlib.util as _iu
_need = [m for m in ["requests", "bs4", "lxml"] if _iu.find_spec(m) is None]
if _need:
    # %pip install -q requests beautifulsoup4 lxml

# مجلد عمل معزول حتى لا تتناثر الملفات في جذر الدفتر
WORK = pathlib.Path("book_runtime")
WORK.mkdir(exist_ok=True)
os.chdir(WORK)
print("مجلد العمل:", pathlib.Path.cwd().name)

# إعداد خاص بهذا الفصل
# مهلة افتراضية لكل طلب حتى لا تعلّق الخلية إن تعذّر الوصول للشبكة
import requests as _rq
_orig_request = _rq.Session.request
def _request(self, method, url, **kw):
    kw.setdefault('timeout', 10)
    return _orig_request(self, method, url, **kw)
_rq.Session.request = _request

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
# [003] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-21-chapter-21.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import requests

response = requests.get("https://books.toscrape.com/")

print(response.status_code)   # 200 يعني نجاح (كما تعلمنا في الفصل 19)
print(response.text[:500])    # أول 500 حرف من محتوى HTML الخام

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-21-chapter-21.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import requests

response = requests.get("https://books.toscrape.com/")

if response.status_code == 200:
    print("✅ نجح الطلب!")
elif response.status_code == 404:
    print("❌ الصفحة غير موجودة")
else:
    print(f"⚠️ حدث خطأ: {response.status_code}")

# طريقة أنظف وأكثر احترافية
try:
    response.raise_for_status()  # يرفع استثناء تلقائيًا إذا كان status_code خطأ
    print("✅ نجح الطلب!")
except requests.HTTPError as e:
    print(f"❌ خطأ HTTP: {e}")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-21-chapter-21.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import requests

headers = {
    "User-Agent": "Mozilla/5.0 (Educational Scraper Bot - contact@example.com)"
}

response = requests.get("https://books.toscrape.com/", headers=headers)
print(response.status_code)

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-21-chapter-21.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
import requests

try:
    response = requests.get(
        "https://books.toscrape.com/",
        timeout=10,  # ثوانٍ -- إذا لم يرد الخادم خلالها، يُرفع استثناء
    )
except requests.Timeout:
    print("⏱️ انتهت المهلة الزمنية -- الخادم بطيء جدًا أو لا يستجيب")
except requests.ConnectionError:
    print("🔌 تعذّر الاتصال -- تحقق من الإنترنت أو الرابط")

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-21-chapter-21.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
import requests

response = requests.get("https://api.github.com/users/python", timeout=10)
data = response.json()  # تحويل تلقائي من JSON إلى dict/list بايثون

print(data["name"])
print(data["public_repos"])

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-21-chapter-21.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
import requests

# Session يحافظ تلقائيًا على الكوكيز بين الطلبات (مفيد بعد تسجيل الدخول مثلاً)
with requests.Session() as session:
    session.headers.update({"User-Agent": "Educational Bot"})

    login_response = session.post(
        "https://example.com/login",
        data={"user": "ahmad", "pass": "1234"},
        timeout=10,
    )

    # هذا الطلب سيستخدم نفس الكوكيز تلقائيًا
    profile_response = session.get("https://example.com/profile", timeout=10)

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-21-chapter-21.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
import requests
from bs4 import BeautifulSoup

response = requests.get("https://books.toscrape.com/", timeout=10)
soup = BeautifulSoup(response.text, "lxml")  # "lxml" هو المحلل (parser) المستخدم

print(soup.title)        # <title>All products | Books to Scrape...</title>
print(soup.title.text)   # All products | Books to Scrape - Sandbox

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-21-chapter-21.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
# العثور على أول عنصر مطابق
first_product = soup.find("article", class_="product_pod")

# العثور على كل العناصر المطابقة -> تُرجع قائمة
all_products = soup.find_all("article", class_="product_pod")
print(f"عدد المنتجات: {len(all_products)}")

# البحث بالوسم فقط
all_headings = soup.find_all("h3")

# البحث بخاصية id
header = soup.find(id="messages")

# البحث بعدة شروط
links = soup.find_all("a", href=True)  # كل روابط <a> التي تملك خاصية href

# ────────────────────────────────────────────────────────────────────────
# [011] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-21-chapter-21.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
product = soup.find("article", class_="product_pod")

title_tag = product.find("h3").find("a")
title = title_tag["title"]           # الوصول لخاصية HTML (title attribute)
# أو
title = title_tag.get("title")       # أكثر أمانًا -- يُرجع None إن لم توجد بدلاً من خطأ

price_tag = product.find("p", class_="price_color")
price_text = price_tag.text          # النص الظاهر للمستخدم: "£51.77"
price_text_stripped = price_tag.get_text(strip=True)  # يزيل المسافات الزائدة تلقائيًا

# ────────────────────────────────────────────────────────────────────────
# [012] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-21-chapter-21.ipynb#cell-11
# ────────────────────────────────────────────────────────────────────────
# محدد CSS: وسم.class
titles = soup.select("h3 a")               # كل وسم <a> داخل <h3>
prices = soup.select("p.price_color")       # كل <p> بخاصية class="price_color"
first_price = soup.select_one("p.price_color")  # أول عنصر مطابق فقط

# محددات أكثر تعقيدًا
all_product_links = soup.select("article.product_pod h3 a")

for link in all_product_links:
    print(link.get("title"), "->", link.get("href"))

# ────────────────────────────────────────────────────────────────────────
# [013] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-21-chapter-21.ipynb#cell-12
# ────────────────────────────────────────────────────────────────────────
product = soup.find("article", class_="product_pod")

print(product.parent)         # العنصر الأب
print(product.find_next_sibling())  # العنصر الشقيق التالي
print(product.contents)       # كل العناصر الفرعية المباشرة كقائمة

# ────────────────────────────────────────────────────────────────────────
# [014] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-21-chapter-21.ipynb#cell-13
# ────────────────────────────────────────────────────────────────────────
import requests
from bs4 import BeautifulSoup

url = "https://books.toscrape.com/"
headers = {"User-Agent": "Educational Scraper (learning purposes)"}

response = requests.get(url, headers=headers, timeout=10)
response.raise_for_status()

soup = BeautifulSoup(response.text, "lxml")

books = soup.select("article.product_pod")
print(f"تم العثور على {len(books)} كتاب في الصفحة")

for book in books[:3]:  # أول 3 كتب فقط للعرض
    title = book.select_one("h3 a").get("title")
    price = book.select_one("p.price_color").get_text(strip=True)
    availability = book.select_one("p.instock.availability").get_text(strip=True)
    rating_classes = book.select_one("p.star-rating")["class"]
    rating = [c for c in rating_classes if c != "star-rating"][0]  # مثل "Three"

    print(f"📖 {title}")
    print(f"   السعر: {price}")
    print(f"   التوفر: {availability}")
    print(f"   التقييم: {rating}")
    print("-" * 40)

# ────────────────────────────────────────────────────────────────────────
# [015] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-21-chapter-21.ipynb#cell-14
# ────────────────────────────────────────────────────────────────────────
import time

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://books.toscrape.com/catalogue/page-{}.html"
headers = {"User-Agent": "Educational Scraper"}

all_titles = []
page = 1

while True:
    url = BASE_URL.format(page)
    response = requests.get(url, headers=headers, timeout=10)

    if response.status_code == 404:
        print(f"انتهت الصفحات عند الصفحة {page - 1}")
        break

    soup = BeautifulSoup(response.text, "lxml")
    books = soup.select("article.product_pod")

    for book in books:
        title = book.select_one("h3 a").get("title")
        all_titles.append(title)

    print(f"✅ تمت معالجة الصفحة {page} ({len(books)} كتاب)")
    page += 1
    time.sleep(1)  # ⚠️ احترام الخادم -- تأخير بين الطلبات إلزامي

print(f"\nإجمالي الكتب المُستخرجة: {len(all_titles)}")

# ────────────────────────────────────────────────────────────────────────
# [016] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-21-chapter-21.ipynb#cell-15
# ────────────────────────────────────────────────────────────────────────
import csv

data = [
    {"title": "A Light in the Attic", "price": "£51.77", "rating": "Three"},
    {"title": "Tipping the Velvet", "price": "£53.74", "rating": "One"},
]

with open("books.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["title", "price", "rating"])
    writer.writeheader()
    writer.writerows(data)

# ────────────────────────────────────────────────────────────────────────
# [017] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-21-chapter-21.ipynb#cell-16
# ────────────────────────────────────────────────────────────────────────
import pandas as pd

df = pd.DataFrame(data)
df.to_csv("books.csv", index=False, encoding="utf-8-sig")
print(df.describe())

# ────────────────────────────────────────────────────────────────────────
# [018] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-21-chapter-21.ipynb#cell-17
# ────────────────────────────────────────────────────────────────────────
import time

import requests
from bs4 import BeautifulSoup

def fetch_page(url: str, retries: int = 3) -> requests.Response | None:
    """جلب صفحة مع إعادة محاولة عند الفشل.

    Args:
        url: الرابط المطلوب جلبه.
        retries: عدد محاولات إعادة الطلب عند الفشل.

    Returns:
        كائن Response عند النجاح، أو None عند فشل كل المحاولات.
    """
    headers = {"User-Agent": "Educational Scraper"}

    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            return response
        except requests.Timeout:
            print(f"⏱️ محاولة {attempt}: انتهت المهلة الزمنية لـ {url}")
        except requests.HTTPError as e:
            print(f"❌ محاولة {attempt}: خطأ HTTP {e}")
        except requests.ConnectionError:
            print(f"🔌 محاولة {attempt}: تعذّر الاتصال بـ {url}")

        if attempt < retries:
            time.sleep(2 * attempt)  # تأخير متزايد (Exponential backoff مبسّط)

    print(f"🚫 فشلت كل المحاولات لـ {url}")
    return None

def extract_book_data(book_element) -> dict[str, str] | None:
    """استخراج بيانات كتاب واحد بأمان، مع التعامل مع عناصر مفقودة.

    Args:
        book_element: عنصر BeautifulSoup يمثّل بطاقة كتاب واحدة.

    Returns:
        قاموس ببيانات الكتاب، أو None إذا كانت البيانات الأساسية مفقودة.
    """
    title_tag = book_element.select_one("h3 a")
    price_tag = book_element.select_one("p.price_color")

    if title_tag is None or price_tag is None:
        return None  # تخطَّ هذا العنصر بدلاً من رفع خطأ يوقف البرنامج كله

    return {
        "title": title_tag.get("title", "غير معروف"),
        "price": price_tag.get_text(strip=True),
    }

# ────────────────────────────────────────────────────────────────────────
# [019] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-21-chapter-21.ipynb#cell-18
# ────────────────────────────────────────────────────────────────────────
import requests
from bs4 import BeautifulSoup

response = requests.get("https://books.toscrape.com/", timeout=10)
print(response.status_code)
print(len(response.text))

soup = BeautifulSoup(response.text, "lxml")
print(soup.title.get_text(strip=True))

links = soup.find_all("a", href=True)
for link in links[:5]:
    print(link["href"])

# ────────────────────────────────────────────────────────────────────────
# [020] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-21-chapter-21.ipynb#cell-19
# ────────────────────────────────────────────────────────────────────────
import asyncio

import aiohttp

SEMAPHORE_LIMIT = 3  # عدد الطلبات المتزامنة القصوى -- احترامًا للخادم

async def fetch_page_async(session: aiohttp.ClientSession, url: str, sem: asyncio.Semaphore) -> str | None:
    """جلب صفحة بشكل غير متزامن مع تحديد التزامن عبر Semaphore."""
    async with sem:
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status == 404:
                    return None
                response.raise_for_status()
                await asyncio.sleep(1)  # تأخير مهذّب حتى في الوضع المتزامن
                return await response.text()
        except aiohttp.ClientError as e:
            print(f"خطأ: {e}")
            return None

async def scrape_pages_async(urls: list[str]) -> list[str | None]:
    """جلب عدة صفحات بالتوازي مع تحديد عدد الطلبات المتزامنة."""
    sem = asyncio.Semaphore(SEMAPHORE_LIMIT)
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_page_async(session, url, sem) for url in urls]
        return await asyncio.gather(*tasks)

# urls = [BASE_URL.format(i) for i in range(1, 6)]
# results = asyncio.run(scrape_pages_async(urls))

# ────────────────────────────────────────────────────────────────────────
# [021] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-21-chapter-21.ipynb#cell-20
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي والعشرون: مسار الأتمتة — استخراج البيانات من الويب (Web Scraping)
القسم 21.4: مكتبة `requests`: طلب الصفحات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 21-001-response.py
"""

import requests

response = requests.get("https://books.toscrape.com/")

print(response.status_code)   # 200 يعني نجاح (كما تعلمنا في الفصل 19)
print(response.text[:500])    # أول 500 حرف من محتوى HTML الخام

# ────────────────────────────────────────────────────────────────────────
# [022] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-21-chapter-21.ipynb#cell-21
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي والعشرون: مسار الأتمتة — استخراج البيانات من الويب (Web Scraping)
القسم 21.4: مكتبة `requests`: طلب الصفحات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 21-002-response.py
"""

import requests

response = requests.get("https://books.toscrape.com/")

if response.status_code == 200:
    print("✅ نجح الطلب!")
elif response.status_code == 404:
    print("❌ الصفحة غير موجودة")
else:
    print(f"⚠️ حدث خطأ: {response.status_code}")

# طريقة أنظف وأكثر احترافية
try:
    response.raise_for_status()  # يرفع استثناء تلقائيًا إذا كان status_code خطأ
    print("✅ نجح الطلب!")
except requests.HTTPError as e:
    print(f"❌ خطأ HTTP: {e}")

# ────────────────────────────────────────────────────────────────────────
# [023] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-21-chapter-21.ipynb#cell-22
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي والعشرون: مسار الأتمتة — استخراج البيانات من الويب (Web Scraping)
القسم 21.4: مكتبة `requests`: طلب الصفحات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 21-003-headers.py
"""

import requests

headers = {
    "User-Agent": "Mozilla/5.0 (Educational Scraper Bot - contact@example.com)"
}

response = requests.get("https://books.toscrape.com/", headers=headers)
print(response.status_code)

# ────────────────────────────────────────────────────────────────────────
# [024] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-21-chapter-21.ipynb#cell-23
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي والعشرون: مسار الأتمتة — استخراج البيانات من الويب (Web Scraping)
القسم 21.4: مكتبة `requests`: طلب الصفحات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 21-004-response.py
"""

import requests

try:
    response = requests.get(
        "https://books.toscrape.com/",
        timeout=10,  # ثوانٍ -- إذا لم يرد الخادم خلالها، يُرفع استثناء
    )
except requests.Timeout:
    print("⏱️ انتهت المهلة الزمنية -- الخادم بطيء جدًا أو لا يستجيب")
except requests.ConnectionError:
    print("🔌 تعذّر الاتصال -- تحقق من الإنترنت أو الرابط")

# ────────────────────────────────────────────────────────────────────────
# [025] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-21-chapter-21.ipynb#cell-24
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي والعشرون: مسار الأتمتة — استخراج البيانات من الويب (Web Scraping)
القسم 21.4: مكتبة `requests`: طلب الصفحات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 21-005-response.py
"""

import requests

# GET مع query parameters (مثل الفصل 19)
response = requests.get(
    "https://api.example.com/search",
    params={"q": "python", "page": 1},
    timeout=10,
)
# الرابط الفعلي سيكون: https://api.example.com/search?q=python&page=1

# POST لإرسال بيانات (مثلاً تسجيل الدخول أو إرسال نموذج)
response = requests.post(
    "https://api.example.com/login",
    json={"username": "ahmad", "password": "secret"},
    timeout=10,
)

# ────────────────────────────────────────────────────────────────────────
# [026] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-21-chapter-21.ipynb#cell-25
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي والعشرون: مسار الأتمتة — استخراج البيانات من الويب (Web Scraping)
القسم 21.4: مكتبة `requests`: طلب الصفحات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 21-006-response.py
"""

import requests

response = requests.get("https://api.github.com/users/python", timeout=10)
data = response.json()  # تحويل تلقائي من JSON إلى dict/list بايثون

print(data["name"])
print(data["public_repos"])

# ────────────────────────────────────────────────────────────────────────
# [027] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-21-chapter-21.ipynb#cell-26
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي والعشرون: مسار الأتمتة — استخراج البيانات من الويب (Web Scraping)
القسم 21.4: مكتبة `requests`: طلب الصفحات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 21-007-login_response.py
"""

import requests

# Session يحافظ تلقائيًا على الكوكيز بين الطلبات (مفيد بعد تسجيل الدخول مثلاً)
with requests.Session() as session:
    session.headers.update({"User-Agent": "Educational Bot"})

    login_response = session.post(
        "https://example.com/login",
        data={"user": "ahmad", "pass": "1234"},
        timeout=10,
    )

    # هذا الطلب سيستخدم نفس الكوكيز تلقائيًا
    profile_response = session.get("https://example.com/profile", timeout=10)

# ────────────────────────────────────────────────────────────────────────
# [028] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-21-chapter-21.ipynb#cell-27
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي والعشرون: مسار الأتمتة — استخراج البيانات من الويب (Web Scraping)
القسم 21.5: مكتبة BeautifulSoup: تحليل واستخراج البيانات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 21-008-response.py
"""

import requests
from bs4 import BeautifulSoup

response = requests.get("https://books.toscrape.com/", timeout=10)
soup = BeautifulSoup(response.text, "lxml")  # "lxml" هو المحلل (parser) المستخدم

print(soup.title)        # <title>All products | Books to Scrape...</title>
print(soup.title.text)   # All products | Books to Scrape - Sandbox

# ────────────────────────────────────────────────────────────────────────
# [029] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-21-chapter-21.ipynb#cell-28
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي والعشرون: مسار الأتمتة — استخراج البيانات من الويب (Web Scraping)
القسم 21.5: مكتبة BeautifulSoup: تحليل واستخراج البيانات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 21-009-first_product.py
"""

# العثور على أول عنصر مطابق
first_product = soup.find("article", class_="product_pod")

# العثور على كل العناصر المطابقة -> تُرجع قائمة
all_products = soup.find_all("article", class_="product_pod")
print(f"عدد المنتجات: {len(all_products)}")

# البحث بالوسم فقط
all_headings = soup.find_all("h3")

# البحث بخاصية id
header = soup.find(id="messages")

# البحث بعدة شروط
links = soup.find_all("a", href=True)  # كل روابط <a> التي تملك خاصية href

# ────────────────────────────────────────────────────────────────────────
# [030] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-21-chapter-21.ipynb#cell-29
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي والعشرون: مسار الأتمتة — استخراج البيانات من الويب (Web Scraping)
القسم 21.5: مكتبة BeautifulSoup: تحليل واستخراج البيانات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 21-010-product.py
"""

product = soup.find("article", class_="product_pod")

title_tag = product.find("h3").find("a")
title = title_tag["title"]           # الوصول لخاصية HTML (title attribute)
# أو
title = title_tag.get("title")       # أكثر أمانًا -- يُرجع None إن لم توجد بدلاً من خطأ

price_tag = product.find("p", class_="price_color")
price_text = price_tag.text          # النص الظاهر للمستخدم: "£51.77"
price_text_stripped = price_tag.get_text(strip=True)  # يزيل المسافات الزائدة تلقائيًا

# ────────────────────────────────────────────────────────────────────────
# [031] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-21-chapter-21.ipynb#cell-30
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي والعشرون: مسار الأتمتة — استخراج البيانات من الويب (Web Scraping)
القسم 21.5: مكتبة BeautifulSoup: تحليل واستخراج البيانات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 21-011-titles.py
"""

# محدد CSS: وسم.class
titles = soup.select("h3 a")               # كل وسم <a> داخل <h3>
prices = soup.select("p.price_color")       # كل <p> بخاصية class="price_color"
first_price = soup.select_one("p.price_color")  # أول عنصر مطابق فقط

# محددات أكثر تعقيدًا
all_product_links = soup.select("article.product_pod h3 a")

for link in all_product_links:
    print(link.get("title"), "->", link.get("href"))

# ────────────────────────────────────────────────────────────────────────
# [032] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-21-chapter-21.ipynb#cell-31
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي والعشرون: مسار الأتمتة — استخراج البيانات من الويب (Web Scraping)
القسم 21.5: مكتبة BeautifulSoup: تحليل واستخراج البيانات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 21-012-product.py
"""

product = soup.find("article", class_="product_pod")

print(product.parent)         # العنصر الأب
print(product.find_next_sibling())  # العنصر الشقيق التالي
print(product.contents)       # كل العناصر الفرعية المباشرة كقائمة

# ────────────────────────────────────────────────────────────────────────
# [033] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-21-chapter-21.ipynb#cell-32
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي والعشرون: مسار الأتمتة — استخراج البيانات من الويب (Web Scraping)
القسم 21.6: نمط العمل الكامل: من الطلب إلى البيانات المنظّمة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 21-013-url.py
"""

import requests
from bs4 import BeautifulSoup

url = "https://books.toscrape.com/"
headers = {"User-Agent": "Educational Scraper (learning purposes)"}

response = requests.get(url, headers=headers, timeout=10)
response.raise_for_status()

soup = BeautifulSoup(response.text, "lxml")

books = soup.select("article.product_pod")
print(f"تم العثور على {len(books)} كتاب في الصفحة")

for book in books[:3]:  # أول 3 كتب فقط للعرض
    title = book.select_one("h3 a").get("title")
    price = book.select_one("p.price_color").get_text(strip=True)
    availability = book.select_one("p.instock.availability").get_text(strip=True)
    rating_classes = book.select_one("p.star-rating")["class"]
    rating = [c for c in rating_classes if c != "star-rating"][0]  # مثل "Three"

    print(f"📖 {title}")
    print(f"   السعر: {price}")
    print(f"   التوفر: {availability}")
    print(f"   التقييم: {rating}")
    print("-" * 40)

# ────────────────────────────────────────────────────────────────────────
# [034] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-21-chapter-21.ipynb#cell-33
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي والعشرون: مسار الأتمتة — استخراج البيانات من الويب (Web Scraping)
القسم 21.7: التعامل مع التصفّح متعدد الصفحات (Pagination)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 21-014-base_url.py
"""

import time

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://books.toscrape.com/catalogue/page-{}.html"
headers = {"User-Agent": "Educational Scraper"}

all_titles = []
page = 1

while True:
    url = BASE_URL.format(page)
    response = requests.get(url, headers=headers, timeout=10)

    if response.status_code == 404:
        print(f"انتهت الصفحات عند الصفحة {page - 1}")
        break

    soup = BeautifulSoup(response.text, "lxml")
    books = soup.select("article.product_pod")

    for book in books:
        title = book.select_one("h3 a").get("title")
        all_titles.append(title)

    print(f"✅ تمت معالجة الصفحة {page} ({len(books)} كتاب)")
    page += 1
    time.sleep(1)  # ⚠️ احترام الخادم -- تأخير بين الطلبات إلزامي

print(f"\nإجمالي الكتب المُستخرجة: {len(all_titles)}")

# ────────────────────────────────────────────────────────────────────────
# [035] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-21-chapter-21.ipynb#cell-34
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي والعشرون: مسار الأتمتة — استخراج البيانات من الويب (Web Scraping)
القسم 21.8: حفظ البيانات المُستخرجة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 21-015-data.py
"""

import csv

data = [
    {"title": "A Light in the Attic", "price": "£51.77", "rating": "Three"},
    {"title": "Tipping the Velvet", "price": "£53.74", "rating": "One"},
]

with open("books.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["title", "price", "rating"])
    writer.writeheader()
    writer.writerows(data)

# ────────────────────────────────────────────────────────────────────────
# [036] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-21-chapter-21.ipynb#cell-35
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي والعشرون: مسار الأتمتة — استخراج البيانات من الويب (Web Scraping)
القسم 21.8: حفظ البيانات المُستخرجة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 21-016-ex21_8.py
"""

import pandas as pd

df = pd.DataFrame(data)
df.to_csv("books.csv", index=False, encoding="utf-8-sig")
print(df.describe())

# ────────────────────────────────────────────────────────────────────────
# [037] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-21-chapter-21.ipynb#cell-36
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي والعشرون: مسار الأتمتة — استخراج البيانات من الويب (Web Scraping)
القسم 21.9: معالجة الأخطاء بصلابة (Robust Scraping)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 21-017-fetch_page.py
"""

import time

import requests
from bs4 import BeautifulSoup

def fetch_page(url: str, retries: int = 3) -> requests.Response | None:
    """جلب صفحة مع إعادة محاولة عند الفشل.

    Args:
        url: الرابط المطلوب جلبه.
        retries: عدد محاولات إعادة الطلب عند الفشل.

    Returns:
        كائن Response عند النجاح، أو None عند فشل كل المحاولات.
    """
    headers = {"User-Agent": "Educational Scraper"}

    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            return response
        except requests.Timeout:
            print(f"⏱️ محاولة {attempt}: انتهت المهلة الزمنية لـ {url}")
        except requests.HTTPError as e:
            print(f"❌ محاولة {attempt}: خطأ HTTP {e}")
        except requests.ConnectionError:
            print(f"🔌 محاولة {attempt}: تعذّر الاتصال بـ {url}")

        if attempt < retries:
            time.sleep(2 * attempt)  # تأخير متزايد (Exponential backoff مبسّط)

    print(f"🚫 فشلت كل المحاولات لـ {url}")
    return None

def extract_book_data(book_element) -> dict[str, str] | None:
    """استخراج بيانات كتاب واحد بأمان، مع التعامل مع عناصر مفقودة.

    Args:
        book_element: عنصر BeautifulSoup يمثّل بطاقة كتاب واحدة.

    Returns:
        قاموس ببيانات الكتاب، أو None إذا كانت البيانات الأساسية مفقودة.
    """
    title_tag = book_element.select_one("h3 a")
    price_tag = book_element.select_one("p.price_color")

    if title_tag is None or price_tag is None:
        return None  # تخطَّ هذا العنصر بدلاً من رفع خطأ يوقف البرنامج كله

    return {
        "title": title_tag.get("title", "غير معروف"),
        "price": price_tag.get_text(strip=True),
    }

# ────────────────────────────────────────────────────────────────────────
# [038] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-21-chapter-21.ipynb#cell-37
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي والعشرون: مسار الأتمتة — استخراج البيانات من الويب (Web Scraping)
القسم 21.11: مشروع تطبيقي متكامل: مستخرج بيانات كتب شامل

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 21-018-fetch_page.py
"""

"""وحدة جلب صفحات الويب مع معالجة الأخطاء وإعادة المحاولة."""

import time

import requests

USER_AGENT = "Educational Book Scraper (learning purposes, github.com/example)"
REQUEST_TIMEOUT = 10
DELAY_BETWEEN_REQUESTS = 1.0  # ثانية -- احترام الخادم

def fetch_page(url: str, retries: int = 3) -> str | None:
    """جلب محتوى HTML لصفحة، مع إعادة محاولة عند الفشل.

    Args:
        url: رابط الصفحة المطلوبة.
        retries: عدد محاولات إعادة الطلب عند حدوث خطأ مؤقت.

    Returns:
        محتوى HTML كنص، أو None إذا فشلت كل المحاولات أو الصفحة غير موجودة.
    """
    headers = {"User-Agent": USER_AGENT}

    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)

            if response.status_code == 404:
                return None  # نهاية الصفحات -- ليس خطأ، بل إشارة توقف طبيعية

            response.raise_for_status()
            return response.text

        except requests.Timeout:
            print(f"⏱️ [محاولة {attempt}/{retries}] انتهت المهلة الزمنية: {url}")
        except requests.ConnectionError:
            print(f"🔌 [محاولة {attempt}/{retries}] تعذّر الاتصال: {url}")
        except requests.HTTPError as e:
            print(f"❌ [محاولة {attempt}/{retries}] خطأ HTTP: {e}")

        if attempt < retries:
            time.sleep(2 * attempt)

    print(f"🚫 فشلت كل المحاولات لجلب: {url}")
    return None

def polite_delay() -> None:
    """تأخير مهذّب بين الطلبات المتتالية لاحترام خادم الموقع."""
    time.sleep(DELAY_BETWEEN_REQUESTS)

# ────────────────────────────────────────────────────────────────────────
# [039] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-21-chapter-21.ipynb#cell-38
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي والعشرون: مسار الأتمتة — استخراج البيانات من الويب (Web Scraping)
القسم 21.11: مشروع تطبيقي متكامل: مستخرج بيانات كتب شامل

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 21-019-_parse_price.py
"""

"""وحدة تحليل صفحات الكتب واستخراج البيانات منها."""

from dataclasses import dataclass

from bs4 import BeautifulSoup

RATING_WORDS_TO_NUMBERS = {
    "One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5,
}

@dataclass
class Book:
    """يمثّل بيانات كتاب واحد مُستخرجة من الموقع."""

    title: str
    price: float
    rating: int
    availability: str

def _parse_price(price_text: str) -> float:
    """تحويل نص السعر مثل '£51.77' إلى رقم عشري."""
    cleaned = price_text.replace("£", "").strip()
    return float(cleaned)

def _parse_rating(star_rating_tag) -> int:
    """استخراج رقم التقييم من اسم الـ class (مثل 'Three' -> 3)."""
    classes = star_rating_tag.get("class", [])
    for cls in classes:
        if cls in RATING_WORDS_TO_NUMBERS:
            return RATING_WORDS_TO_NUMBERS[cls]
    return 0

def extract_books_from_page(html: str) -> list[Book]:
    """استخراج كل الكتب من صفحة HTML واحدة.

    Args:
        html: محتوى HTML الخام للصفحة.

    Returns:
        قائمة من كائنات Book، مع تخطي أي عنصر بيانات ناقصة.
    """
    soup = BeautifulSoup(html, "lxml")
    book_elements = soup.select("article.product_pod")

    books: list[Book] = []

    for element in book_elements:
        title_tag = element.select_one("h3 a")
        price_tag = element.select_one("p.price_color")
        rating_tag = element.select_one("p.star-rating")
        availability_tag = element.select_one("p.instock.availability")

        if not all([title_tag, price_tag, rating_tag, availability_tag]):
            continue  # تخطَّ أي كتاب ببيانات ناقصة بدلاً من إيقاف البرنامج

        book = Book(
            title=title_tag.get("title", "غير معروف"),
            price=_parse_price(price_tag.get_text(strip=True)),
            rating=_parse_rating(rating_tag),
            availability=availability_tag.get_text(strip=True),
        )
        books.append(book)

    return books

def has_next_page(html: str) -> bool:
    """التحقق من وجود صفحة تالية في نظام التصفّح."""
    soup = BeautifulSoup(html, "lxml")
    return soup.select_one("li.next") is not None

# ────────────────────────────────────────────────────────────────────────
# [040] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-21-chapter-21.ipynb#cell-39
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي والعشرون: مسار الأتمتة — استخراج البيانات من الويب (Web Scraping)
القسم 21.11: مشروع تطبيقي متكامل: مستخرج بيانات كتب شامل

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 21-020-save_books_to_csv.py
"""

"""وحدة حفظ بيانات الكتب المُستخرجة."""

from dataclasses import asdict
from pathlib import Path

import pandas as pd

from book_scraper.parser import Book

def save_books_to_csv(books: list[Book], file_path: str | Path) -> None:
    """حفظ قائمة الكتب في ملف CSV باستخدام pandas.

    Args:
        books: قائمة كائنات Book المُستخرجة.
        file_path: مسار ملف الحفظ.
    """
    records = [asdict(book) for book in books]
    df = pd.DataFrame(records)
    df.to_csv(file_path, index=False, encoding="utf-8-sig")

# ────────────────────────────────────────────────────────────────────────
# [041] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-21-chapter-21.ipynb#cell-40
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي والعشرون: مسار الأتمتة — استخراج البيانات من الويب (Web Scraping)
القسم 21.11: مشروع تطبيقي متكامل: مستخرج بيانات كتب شامل

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 21-021-scrape_all_books.py
"""

"""نقطة الدخول الرئيسية لأداة استخراج بيانات الكتب."""

from pathlib import Path

import pandas as pd

from book_scraper.fetcher import fetch_page, polite_delay
from book_scraper.parser import Book, extract_books_from_page, has_next_page
from book_scraper.storage import save_books_to_csv

BASE_URL = "https://books.toscrape.com/catalogue/page-{}.html"
OUTPUT_FILE = Path("output/books.csv")
MAX_PAGES = 5  # نحدد سقفًا احترازيًا لتجنب حلقة لا نهائية في حال خطأ غير متوقع

def scrape_all_books() -> list[Book]:
    """استخراج كل الكتب من كل الصفحات المتاحة.

    Returns:
        قائمة شاملة بكل الكتب المُستخرجة من كل الصفحات.
    """
    all_books: list[Book] = []
    page_number = 1

    while page_number <= MAX_PAGES:
        url = BASE_URL.format(page_number)
        print(f"📄 جلب الصفحة {page_number}...")

        html = fetch_page(url)
        if html is None:
            print("🏁 لا مزيد من الصفحات أو حدث خطأ غير قابل للتعافي. توقف.")
            break

        books_in_page = extract_books_from_page(html)
        all_books.extend(books_in_page)
        print(f"   ✅ تم استخراج {len(books_in_page)} كتاب")

        if not has_next_page(html):
            print("🏁 تم الوصول لآخر صفحة.")
            break

        page_number += 1
        polite_delay()

    return all_books

def print_analysis(books: list[Book]) -> None:
    """طباعة تحليل سريع لبيانات الكتب المُستخرجة."""
    if not books:
        print("لا توجد بيانات للتحليل.")
        return

    df = pd.DataFrame([{"title": b.title, "price": b.price, "rating": b.rating} for b in books])

    print("\n" + "=" * 50)
    print(" " * 12 + "تحليل بيانات الكتب المُستخرجة")
    print("=" * 50)
    print(f"إجمالي الكتب: {len(df)}")
    print(f"متوسط السعر: £{df['price'].mean():.2f}")
    print(f"أغلى كتاب: £{df['price'].max():.2f}")
    print(f"أرخص كتاب: £{df['price'].min():.2f}")
    print(f"متوسط التقييم: {df['rating'].mean():.1f} / 5")
    print("=" * 50)

def main() -> None:
    """تشغيل عملية الاستخراج الكاملة."""
    print("🚀 بدء استخراج بيانات الكتب من books.toscrape.com\n")

    books = scrape_all_books()

    print(f"\n💾 حفظ {len(books)} كتاب في {OUTPUT_FILE}...")
    OUTPUT_FILE.parent.mkdir(exist_ok=True)
    save_books_to_csv(books, OUTPUT_FILE)

    print_analysis(books)
    print(f"\n✅ اكتمل! راجع الملف: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()

# ────────────────────────────────────────────────────────────────────────
# [042] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-21-chapter-21.ipynb#cell-41
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي والعشرون: مسار الأتمتة — استخراج البيانات من الويب (Web Scraping)
القسم 21.12: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 21-022-response.py
"""

import requests
from bs4 import BeautifulSoup

response = requests.get("https://books.toscrape.com/", timeout=10)
print(response.status_code)
print(len(response.text))

soup = BeautifulSoup(response.text, "lxml")
print(soup.title.get_text(strip=True))

links = soup.find_all("a", href=True)
for link in links[:5]:
    print(link["href"])

# ────────────────────────────────────────────────────────────────────────
# [043] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-21-chapter-21.ipynb#cell-42
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي والعشرون: مسار الأتمتة — استخراج البيانات من الويب (Web Scraping)
القسم 21.12: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 21-023-filter_books_by_rating.py
"""

# تمرين 5 -- سهل التطبيق مباشرة على كائنات Book
def filter_books_by_rating(books: list[Book], min_rating: int) -> list[Book]:
    """إرجاع الكتب التي تقييمها أكبر من أو يساوي الحد الأدنى المطلوب."""
    return [book for book in books if book.rating >= min_rating]

# مثال استخدام
high_rated = filter_books_by_rating(books, min_rating=4)
print(f"عدد الكتب ذات التقييم 4+: {len(high_rated)}")

# ────────────────────────────────────────────────────────────────────────
# [044] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-21-chapter-21.ipynb#cell-43
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي والعشرون: مسار الأتمتة — استخراج البيانات من الويب (Web Scraping)
القسم 21.12: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 21-024-fetch_page_async.py
"""

import asyncio

import aiohttp

SEMAPHORE_LIMIT = 3  # عدد الطلبات المتزامنة القصوى -- احترامًا للخادم

async def fetch_page_async(session: aiohttp.ClientSession, url: str, sem: asyncio.Semaphore) -> str | None:
    """جلب صفحة بشكل غير متزامن مع تحديد التزامن عبر Semaphore."""
    async with sem:
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status == 404:
                    return None
                response.raise_for_status()
                await asyncio.sleep(1)  # تأخير مهذّب حتى في الوضع المتزامن
                return await response.text()
        except aiohttp.ClientError as e:
            print(f"خطأ: {e}")
            return None

async def scrape_pages_async(urls: list[str]) -> list[str | None]:
    """جلب عدة صفحات بالتوازي مع تحديد عدد الطلبات المتزامنة."""
    sem = asyncio.Semaphore(SEMAPHORE_LIMIT)
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_page_async(session, url, sem) for url in urls]
        return await asyncio.gather(*tasks)

# urls = [BASE_URL.format(i) for i in range(1, 6)]
# results = asyncio.run(scrape_pages_async(urls))

# ────────────────────────────────────────────────────────────────────────
# [045] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-21-chapter-21.ipynb#cell-44
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي والعشرون: مسار الأتمتة — استخراج البيانات من الويب (Web Scraping)
القسم 21.14: ملخص الفصل (Cheat Sheet)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 21-025-response.py
"""

# ===== requests: الطلبات الأساسية =====
import requests
response = requests.get(url, headers={"User-Agent": "..."}, timeout=10)
response.status_code         # 200, 404, 500...
response.text                 # محتوى HTML/نص خام
response.json()               # تحويل تلقائي من JSON
response.raise_for_status()   # يرفع استثناء عند 4xx/5xx

try:
    response = requests.get(url, timeout=10)
except requests.Timeout:
    ...
except requests.ConnectionError:
    ...

# ===== BeautifulSoup: التحليل =====
from bs4 import BeautifulSoup
soup = BeautifulSoup(response.text, "lxml")

soup.find("div", class_="price")       # أول عنصر مطابق
soup.find_all("article")               # كل العناصر المطابقة -> قائمة
soup.select("article.product_pod h3 a")  # محدد CSS -> قائمة
soup.select_one("title")               # محدد CSS -> أول عنصر

tag.text / tag.get_text(strip=True)    # النص الظاهر
tag.get("href")                        # خاصية HTML بأمان (None إن لم توجد)
tag["href"]                            # خاصية HTML (يرفع خطأ إن لم توجد)

# ===== الأتمتة الآمنة =====
import time
time.sleep(1)                          # تأخير بين الطلبات (إلزامي أخلاقيًا)

# ────────────────────────────────────────────────────────────────────────
# [046] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-21-chapter-21.ipynb#cell-45
# ────────────────────────────────────────────────────────────────────────
import requests

# GET مع query parameters (مثل الفصل 19)
response = requests.get(
    "https://api.example.com/search",
    params={"q": "python", "page": 1},
    timeout=10,
)
# الرابط الفعلي سيكون: https://api.example.com/search?q=python&page=1

# POST لإرسال بيانات (مثلاً تسجيل الدخول أو إرسال نموذج)
response = requests.post(
    "https://api.example.com/login",
    json={"username": "ahmad", "password": "secret"},
    timeout=10,
)

# ────────────────────────────────────────────────────────────────────────
# [047] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-21-chapter-21.ipynb#cell-46
# ────────────────────────────────────────────────────────────────────────
> soup.find("div", class_="price")   # ✅ صحيح
> ```

### استخراج النص والخصائص
