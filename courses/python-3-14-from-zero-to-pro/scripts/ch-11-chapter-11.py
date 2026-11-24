#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
بايثون 3.14 — من الصفر إلى الاحتراف — معالجة الأخطاء (Error Handling)
Python 3.14 — From Zero to Pro — Error

Book slug : python-3-14-from-zero-to-pro
Chapter   : ch-11-chapter-11 (lesson 11)
Code files: 71 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/python-3-14-from-zero-to-pro/code/ch-11-chapter-11.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
بايثون 3.14 — من الصفر إلى الاحتراف — معالجة الأخطاء (Error Handling)
Python 3.14 — From Zero to Pro — Error

Book slug : python-3-14-from-zero-to-pro
Chapter   : ch-11-chapter-11 (lesson 11)
Code files: 70 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/python-3-14-from-zero-to-pro/code/ch-11-chapter-11.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
بايثون 3.14 — من الصفر إلى الاحتراف — معالجة الأخطاء (Error Handling)
Python 3.14 — From Zero to Pro — Error

Book slug : python-3-14-from-zero-to-pro
Chapter   : ch-11-chapter-11 (lesson 11)
Code files: 69 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] dist/notebooks/chapter-11.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#@title ⚙️ تهيئة البيئة  { display-mode: "form" }
import sys, os, random, pathlib

print("Python:", sys.version.split()[0], "|", sys.platform)
assert sys.version_info >= (3, 10), "هذا الكتاب يفترض Python 3.10 أو أحدث"

SEED = 42
random.seed(SEED)

# ────────────────────────────────────────────────────────────────────────
# [002] dist/notebooks/chapter-11.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
# ⚠️ مثال يُظهر خطأً مقصودًا: نلتقطه هنا ليكتمل «تشغيل الكل»،
# وسترى نوع الخطأ ورسالته في المخرجات كما في الطرفية.
try:
    def divide(a, b):
        return a / b

    result = divide(10, 0)
except (Exception, SystemExit) as _err:
    print(f"⚠️ {type(_err).__name__}: {_err}")

# ────────────────────────────────────────────────────────────────────────
# [003] dist/notebooks/chapter-11.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# 📝 نسخة الدفتر: حقل نموذج بدل الإدخال اليدوي — عدّل القيمة ثم أعد التشغيل
try:
    number = 10  #@param {type:"integer"}
    print(f"ضعف الرقم: {number * 2}")
except ValueError:
    print("❌ خطأ: الرجاء إدخال رقم صحيح فقط.")

print("البرنامج استمر في العمل!")

# ────────────────────────────────────────────────────────────────────────
# [004] dist/notebooks/chapter-11.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# ⚠️ مثال يُظهر خطأً مقصودًا: نلتقطه هنا ليكتمل «تشغيل الكل»،
# وسترى نوع الخطأ ورسالته في المخرجات كما في الطرفية.
try:
    def safe_divide(a: float, b: float) -> float | None:
        try:
            return a / b
        except ZeroDivisionError:
            print("❌ لا يمكن القسمة على صفر.")
            return None
        except TypeError:
            print("❌ يجب أن تكون القيم أرقاماً.")
            return None

    print(safe_divide(10, 0))        # ❌ لا يمكن القسمة على صفر. → None
    print(safe_divide(10, "abc"))    # ❌ يجب أن تكون القيم أرقاماً. → None
    print(safe_divide(10, 2))        # 5.0
except (Exception, SystemExit) as _err:
    print(f"⚠️ {type(_err).__name__}: {_err}")

# ────────────────────────────────────────────────────────────────────────
# [005] dist/notebooks/chapter-11.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
try:
    x = 1 / 0
except Exception:
    print("خطأ عام")  # هذا ما سيُطبع
except ZeroDivisionError:  # ⚠️ ميت! لن يُنفَّذ أبداً
    print("قسمة على صفر")

# ────────────────────────────────────────────────────────────────────────
# [006] dist/notebooks/chapter-11.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
def read_and_process_number(text: str) -> None:
    try:
        number = int(text)
    except ValueError:
        print(f"❌ '{text}' ليس رقماً صحيحاً.")
    else:
        # يُنفَّذ فقط إذا نجح التحويل بدون خطأ
        print(f"✅ تم تحويل الرقم بنجاح: {number}")
        print(f"   ضعفه: {number * 2}")
    finally:
        # يُنفَّذ دائماً - مثالي للتنظيف
        print("--- انتهت المحاولة ---")

read_and_process_number("42")
print()
read_and_process_number("abc")

# ────────────────────────────────────────────────────────────────────────
# [007] dist/notebooks/chapter-11.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
def process_file(filename: str) -> None:
    file = None
    try:
        file = open(filename, "r", encoding="utf-8")
        content = file.read()
        print(f"تم قراءة {len(content)} حرف.")
    except FileNotFoundError:
        print(f"❌ الملف '{filename}' غير موجود.")
    finally:
        if file is not None:
            file.close()
            print("✅ تم إغلاق الملف.")  # يحدث دائماً!

process_file("not_exist.txt")

# ────────────────────────────────────────────────────────────────────────
# [008] dist/notebooks/chapter-11.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
def withdraw(balance: float, amount: float) -> float:
    if amount <= 0:
        raise ValueError("مبلغ السحب يجب أن يكون موجباً.")
    if amount > balance:
        raise ValueError("الرصيد غير كافٍ لإتمام العملية.")
    return balance - amount

try:
    new_balance = withdraw(1000, 1500)
except ValueError as e:
    print(f"❌ فشلت العملية: {e}")

# ────────────────────────────────────────────────────────────────────────
# [009] dist/notebooks/chapter-11.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
def process_data(data: list) -> float:
    try:
        return sum(data) / len(data)
    except ZeroDivisionError:
        print("⚠️ تحذير: القائمة فارغة، سيتم إعادة إطلاق الخطأ.")
        raise  # raise بدون معامل = إعادة إطلاق نفس الخطأ الملتقط

try:
    process_data([])
except ZeroDivisionError:
    print("❌ المستوى الأعلى: تم التعامل مع القائمة الفارغة.")

# ────────────────────────────────────────────────────────────────────────
# [010] dist/notebooks/chapter-11.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
def parse_age(text: str) -> int:
    try:
        return int(text)
    except ValueError as e:
        raise ValueError(f"فشل تحليل العمر من النص: '{text}'") from e

try:
    parse_age("twenty")
except ValueError as e:
    print(e)
    print(f"السبب الأصلي: {e.__cause__}")

# ────────────────────────────────────────────────────────────────────────
# [011] dist/notebooks/chapter-11.ipynb#cell-11
# ────────────────────────────────────────────────────────────────────────
class InsufficientFundsError(Exception):
    """تُطلق عندما يحاول المستخدم سحب مبلغ أكبر من رصيده."""
    pass

class InvalidAmountError(Exception):
    """تُطلق عندما يكون المبلغ المُدخل غير منطقي (سالب أو صفر)."""
    pass

def withdraw(balance: float, amount: float) -> float:
    if amount <= 0:
        raise InvalidAmountError(f"المبلغ {amount} غير صالح؛ يجب أن يكون موجباً.")
    if amount > balance:
        raise InsufficientFundsError(
            f"طلبت سحب {amount}، لكن رصيدك الحالي {balance} فقط."
        )
    return balance - amount

try:
    withdraw(500, 1000)
except InsufficientFundsError as e:
    print(f"💰 خطأ في الرصيد: {e}")
except InvalidAmountError as e:
    print(f"🔢 خطأ في المبلغ: {e}")

# ────────────────────────────────────────────────────────────────────────
# [012] dist/notebooks/chapter-11.ipynb#cell-12
# ────────────────────────────────────────────────────────────────────────
class BankError(Exception):
    """الفئة الأساسية لكل أخطاء النظام البنكي."""
    pass

class InsufficientFundsError(BankError):
    """رصيد غير كافٍ."""
    pass

class InvalidAmountError(BankError):
    """مبلغ غير صالح."""
    pass

class AccountFrozenError(BankError):
    """الحساب مجمّد."""
    pass

def process_transaction(account_status: str, balance: float, amount: float) -> float:
    if account_status == "frozen":
        raise AccountFrozenError("هذا الحساب مجمّد حالياً.")
    if amount <= 0:
        raise InvalidAmountError("المبلغ يجب أن يكون موجباً.")
    if amount > balance:
        raise InsufficientFundsError("الرصيد غير كافٍ.")
    return balance - amount

try:
    process_transaction("frozen", 1000, 100)
except BankError as e:  # التقاط أي خطأ بنكي بغض النظر عن نوعه الدقيق
    print(f"🏦 خطأ في النظام البنكي: {type(e).__name__}: {e}")

# ────────────────────────────────────────────────────────────────────────
# [013] dist/notebooks/chapter-11.ipynb#cell-13
# ────────────────────────────────────────────────────────────────────────
class ValidationError(Exception):
    """خطأ تحقق من صحة البيانات، مع تفاصيل الحقل المُخالف."""

    def __init__(self, field: str, message: str):
        self.field = field
        self.message = message
        super().__init__(f"خطأ في حقل '{field}': {message}")

def validate_age(age: int) -> None:
    if age < 0 or age > 150:
        raise ValidationError(field="age", message="يجب أن يكون بين 0 و150")

try:
    validate_age(-5)
except ValidationError as e:
    print(f"الحقل المُخالف: {e.field}")
    print(f"الرسالة: {e.message}")

# ────────────────────────────────────────────────────────────────────────
# [014] dist/notebooks/chapter-11.ipynb#cell-14
# ────────────────────────────────────────────────────────────────────────
# تحقق أولاً، ثم نفّذ
import os

if os.path.exists("data.txt"):
    with open("data.txt") as f:
        content = f.read()
else:
    print("الملف غير موجود")

# ────────────────────────────────────────────────────────────────────────
# [015] dist/notebooks/chapter-11.ipynb#cell-15
# ────────────────────────────────────────────────────────────────────────
# نفّذ مباشرة، وتعامل مع الخطأ إن حدث
try:
    with open("data.txt") as f:
        content = f.read()
except FileNotFoundError:
    print("الملف غير موجود")

# ────────────────────────────────────────────────────────────────────────
# [016] dist/notebooks/chapter-11.ipynb#cell-16
# ────────────────────────────────────────────────────────────────────────
# ⚠️ مثال يُظهر خطأً مقصودًا: نلتقطه هنا ليكتمل «تشغيل الكل»،
# وسترى نوع الخطأ ورسالته في المخرجات كما في الطرفية.
try:
    def calculate_average(numbers: list[float]) -> float:
        assert len(numbers) > 0, "القائمة يجب ألا تكون فارغة"
        return sum(numbers) / len(numbers)

    print(calculate_average([10, 20, 30]))  # 20.0
    # print(calculate_average([]))
    # 💥 AssertionError: القائمة يجب ألا تكون فارغة
except (Exception, SystemExit) as _err:
    print(f"⚠️ {type(_err).__name__}: {_err}")

# ────────────────────────────────────────────────────────────────────────
# [017] dist/notebooks/chapter-11.ipynb#cell-17
# ────────────────────────────────────────────────────────────────────────
# ⚠️ مثال يُظهر خطأً مقصودًا: نلتقطه هنا ليكتمل «تشغيل الكل»،
# وسترى نوع الخطأ ورسالته في المخرجات كما في الطرفية.
try:
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    def divide(a: float, b: float) -> float | None:
        try:
            return a / b
        except ZeroDivisionError:
            logging.error(f"محاولة قسمة {a} على صفر!")
            return None

    divide(10, 0)
except (Exception, SystemExit) as _err:
    print(f"⚠️ {type(_err).__name__}: {_err}")

# ────────────────────────────────────────────────────────────────────────
# [018] dist/notebooks/chapter-11.ipynb#cell-18
# ────────────────────────────────────────────────────────────────────────
class AuthError(Exception):
    """الفئة الأساسية لكل أخطاء المصادقة."""
    pass

class UserNotFoundError(AuthError):
    """المستخدم غير موجود في النظام."""
    pass

class IncorrectPasswordError(AuthError):
    """كلمة المرور غير صحيحة."""
    pass

class AccountLockedError(AuthError):
    """الحساب مقفل بسبب محاولات فاشلة متكررة."""
    pass

class WeakPasswordError(AuthError):
    """كلمة مرور ضعيفة عند التسجيل."""
    pass

# قاعدة بيانات وهمية (في الواقع ستكون قاعدة بيانات حقيقية)
users_db: dict[str, dict] = {
    "ahmed@test.com": {"password": "Secure123", "failed_attempts": 0, "locked": False},
}

MAX_FAILED_ATTEMPTS = 3

def register(email: str, password: str) -> None:
    """تسجيل مستخدم جديد مع التحقق من قوة كلمة المرور."""
    if len(password) < 8:
        raise WeakPasswordError("كلمة المرور يجب أن تكون 8 أحرف على الأقل.")
    if not any(char.isdigit() for char in password):
        raise WeakPasswordError("كلمة المرور يجب أن تحتوي رقماً واحداً على الأقل.")

    if email in users_db:
        raise ValueError("هذا البريد الإلكتروني مسجّل مسبقاً.")

    users_db[email] = {"password": password, "failed_attempts": 0, "locked": False}
    print(f"✅ تم تسجيل المستخدم {email} بنجاح.")

def login(email: str, password: str) -> bool:
    """محاولة تسجيل دخول، مع حماية من المحاولات المتكررة."""
    try:
        user = users_db[email]
    except KeyError:
        raise UserNotFoundError(f"لا يوجد مستخدم بالبريد: {email}")

    if user["locked"]:
        raise AccountLockedError("هذا الحساب مقفل. تواصل مع الدعم الفني.")

    if user["password"] != password:
        user["failed_attempts"] += 1
        if user["failed_attempts"] >= MAX_FAILED_ATTEMPTS:
            user["locked"] = True
            raise AccountLockedError(
                "تم قفل الحساب بسبب تجاوز عدد المحاولات الفاشلة المسموحة."
            )
        remaining = MAX_FAILED_ATTEMPTS - user["failed_attempts"]
        raise IncorrectPasswordError(
            f"كلمة المرور خاطئة. المحاولات المتبقية: {remaining}"
        )

    user["failed_attempts"] = 0  # إعادة تصفير عند النجاح
    return True

def run_login_attempt(email: str, password: str) -> None:
    """واجهة تفاعلية تلتقط كل أنواع أخطاء المصادقة بوضوح."""
    try:
        login(email, password)
    except UserNotFoundError as e:
        print(f"👤 {e}")
    except AccountLockedError as e:
        print(f"🔒 {e}")
    except IncorrectPasswordError as e:
        print(f"🔑 {e}")
    except AuthError as e:
        print(f"⚠️ خطأ مصادقة غير متوقع: {e}")
    else:
        print(f"✅ تم تسجيل الدخول بنجاح! أهلاً بك {email}")
    finally:
        print("-" * 40)

# --- سيناريوهات تجريبية ---
if __name__ == "__main__":
    try:
        register("sara@test.com", "weak")
    except WeakPasswordError as e:
        print(f"❌ فشل التسجيل: {e}")
    print("-" * 40)

    register("sara@test.com", "Strong123")
    print("-" * 40)

    run_login_attempt("unknown@test.com", "anything")   # UserNotFoundError
    run_login_attempt("ahmed@test.com", "wrong_pass")   # IncorrectPasswordError (محاولة 1)
    run_login_attempt("ahmed@test.com", "wrong_pass")   # IncorrectPasswordError (محاولة 2)
    run_login_attempt("ahmed@test.com", "wrong_pass")   # AccountLockedError (محاولة 3 - قفل!)
    run_login_attempt("ahmed@test.com", "Secure123")    # AccountLockedError (مقفل مسبقاً)
    run_login_attempt("sara@test.com", "Strong123")     # ✅ نجاح

# ────────────────────────────────────────────────────────────────────────
# [019] dist/notebooks/chapter-11.ipynb#cell-19
# ────────────────────────────────────────────────────────────────────────
def reset_password(email: str, new_password: str) -> None:
    """إعادة تعيين كلمة مرور، مع إلغاء القفل إن وُجد."""
    try:
        user = users_db[email]
    except KeyError:
        raise UserNotFoundError(f"لا يوجد مستخدم بالبريد: {email}")

    if len(new_password) < 8 or not any(c.isdigit() for c in new_password):
        raise WeakPasswordError("كلمة المرور الجديدة ضعيفة جداً.")

    user["password"] = new_password
    user["failed_attempts"] = 0
    user["locked"] = False
    print(f"🔓 تم إعادة تعيين كلمة المرور وفك قفل الحساب لـ {email}")

try:
    reset_password("ahmed@test.com", "NewPass123")
except AuthError as e:
    print(f"❌ {e}")

# ────────────────────────────────────────────────────────────────────────
# [020] dist/notebooks/chapter-11.ipynb#cell-20
# ────────────────────────────────────────────────────────────────────────
# 📝 نسخة الدفتر: حقل نموذج بدل الإدخال اليدوي — عدّل القيمة ثم أعد التشغيل
# تمرين 1
try:
    num = 10  #@param {type:"integer"}
    print("رقم صحيح")
except ValueError:
    print("خطأ: ليس رقماً")

# تمرين 2
def safe_list_access(lst: list, index: int):
    try:
        return lst[index]
    except IndexError:
        return None

print(safe_list_access([1, 2, 3], 1))   # 2
print(safe_list_access([1, 2, 3], 10))  # None

# ────────────────────────────────────────────────────────────────────────
# [021] dist/notebooks/chapter-11.ipynb#cell-21
# ────────────────────────────────────────────────────────────────────────
import math

# تمرين 3
class NegativeValueError(Exception):
    pass

def calculate_square_root(n: float) -> float:
    if n < 0:
        raise NegativeValueError(f"لا يمكن حساب جذر رقم سالب: {n}")
    return math.sqrt(n)

try:
    print(calculate_square_root(-9))
except NegativeValueError as e:
    print(f"❌ {e}")

# تمرين 4
class MissingConfigError(Exception):
    def __init__(self, key: str):
        self.key = key
        super().__init__(f"الإعداد المطلوب '{key}' غير موجود.")

def parse_config(data: dict) -> tuple[str, int]:
    try:
        host = data["host"]
    except KeyError:
        raise MissingConfigError("host")

    try:
        port = data["port"]
    except KeyError:
        raise MissingConfigError("port")

    return host, port

try:
    parse_config({"host": "localhost"})
except MissingConfigError as e:
    print(f"❌ {e}")  # ❌ الإعداد المطلوب 'port' غير موجود.

# ────────────────────────────────────────────────────────────────────────
# [022] dist/notebooks/chapter-11.ipynb#cell-22
# ────────────────────────────────────────────────────────────────────────
class APIError(Exception):
    """الفئة الأساسية لكل أخطاء API."""
    pass

class EndpointNotFoundError(APIError):
    def __init__(self, endpoint: str):
        self.endpoint = endpoint
        super().__init__(f"المسار '{endpoint}' غير موجود.")

class MissingFieldError(APIError):
    def __init__(self, field: str):
        self.field = field
        super().__init__(f"الحقل المطلوب '{field}' مفقود.")

class InvalidFieldError(APIError):
    def __init__(self, field: str, reason: str):
        self.field = field
        self.reason = reason
        super().__init__(f"الحقل '{field}' غير صالح: {reason}")

VALID_ENDPOINTS = {"/users", "/orders", "/products"}

def process_api_request(endpoint: str, data: dict) -> dict:
    if endpoint not in VALID_ENDPOINTS:
        raise EndpointNotFoundError(endpoint)

    if "user_id" not in data:
        raise MissingFieldError("user_id")

    user_id = data["user_id"]
    if not isinstance(user_id, int) or user_id <= 0:
        raise InvalidFieldError("user_id", "يجب أن يكون رقماً صحيحاً موجباً")

    return {"status": "success", "endpoint": endpoint, "user_id": user_id}

def handle_request(endpoint: str, data: dict) -> None:
    try:
        result = process_api_request(endpoint, data)
    except EndpointNotFoundError as e:
        print(f"404 - {e}")
    except MissingFieldError as e:
        print(f"400 - حقل مفقود: {e}")
    except InvalidFieldError as e:
        print(f"400 - حقل غير صالح: {e}")
    except APIError as e:
        print(f"500 - خطأ API غير متوقع: {e}")
    else:
        print(f"200 - {result}")

handle_request("/comments", {"user_id": 1})         # 404
handle_request("/users", {})                         # 400 مفقود
handle_request("/users", {"user_id": -5})            # 400 غير صالح
handle_request("/users", {"user_id": 42})            # 200 نجاح

# ────────────────────────────────────────────────────────────────────────
# [023] code/chapters/ch11/11-001-balance.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.1: لماذا يحتاج كل برنامج احترافي لمعالجة الأخطاء؟

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-001-balance.py
"""

balance = 1000
withdrawal = input("أدخل مبلغ السحب: ")  # المستخدم يكتب "مئة"
balance -= int(withdrawal)  # 💥 ValueError: invalid literal for int()
# البرنامج ينهار، والمستخدم يرى رسالة تقنية غير مفهومة

# ────────────────────────────────────────────────────────────────────────
# [024] code/chapters/ch11/11-002-ex11_2.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.2: فهم الاستثناءات (Exceptions)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-002-ex11_2.py
"""

print(10 / 0)        # ZeroDivisionError
print(int("abc"))    # ValueError
print(my_list[99])   # IndexError
print(my_dict["x"])  # KeyError
print(undefined_var) # NameError

# ────────────────────────────────────────────────────────────────────────
# [025] code/chapters/ch11/11-003-divide.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.2: فهم الاستثناءات (Exceptions)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-003-divide.py
"""

def divide(a, b):
    return a / b

result = divide(10, 0)

# ────────────────────────────────────────────────────────────────────────
# [026] code/chapters/ch11/11-004-number.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.3: الصيغة الأساسية: try / except

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-004-number.py
"""

try:
    number = int(input("أدخل رقماً: "))
    print(f"ضعف الرقم: {number * 2}")
except ValueError:
    print("❌ خطأ: الرجاء إدخال رقم صحيح فقط.")

print("البرنامج استمر في العمل!")

# ────────────────────────────────────────────────────────────────────────
# [027] code/chapters/ch11/11-005-safe_divide.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.3: الصيغة الأساسية: try / except

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-005-safe_divide.py
"""

def safe_divide(a: float, b: float) -> float | None:
    try:
        return a / b
    except ZeroDivisionError:
        print("❌ لا يمكن القسمة على صفر.")
        return None
    except TypeError:
        print("❌ يجب أن تكون القيم أرقاماً.")
        return None

print(safe_divide(10, 0))        # ❌ لا يمكن القسمة على صفر. → None
print(safe_divide(10, "abc"))    # ❌ يجب أن تكون القيم أرقاماً. → None
print(safe_divide(10, 2))        # 5.0

# ────────────────────────────────────────────────────────────────────────
# [028] code/chapters/ch11/11-006-value.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.3: الصيغة الأساسية: try / except

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-006-value.py
"""

try:
    value = my_dict[key]
    result = 100 / value
except (KeyError, ZeroDivisionError) as e:
    print(f"حدث خطأ: {e}")

# ────────────────────────────────────────────────────────────────────────
# [029] code/chapters/ch11/11-007-ex11_3.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.3: الصيغة الأساسية: try / except

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-007-ex11_3.py
"""

try:
    risky_operation()
except Exception as e:
    print(f"حدث خطأ غير متوقع: {e}")

# ────────────────────────────────────────────────────────────────────────
# [030] code/chapters/ch11/11-008-ex11_3.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.3: الصيغة الأساسية: try / except

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-008-ex11_3.py
"""

# ❌ الأسوأ على الإطلاق - لا تفعل هذا أبداً!
try:
    risky_operation()
except:  # bare except - يلتقط كل شيء حتى KeyboardInterrupt!
    pass  # يُخفي الخطأ تماماً دون أي أثر!

# ────────────────────────────────────────────────────────────────────────
# [031] code/chapters/ch11/11-009-value.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.3: الصيغة الأساسية: try / except

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-009-value.py
"""

# ترتيب except مهم جداً! الأكثر تحديداً أولاً
try:
    value = my_list[10]
except IndexError:
    print("فهرس خارج النطاق")
except LookupError:  # تأتي بعد IndexError لأنها أعم منها
    print("خطأ بحث عام")

# ────────────────────────────────────────────────────────────────────────
# [032] code/chapters/ch11/11-010-ex11_3.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.3: الصيغة الأساسية: try / except

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-010-ex11_3.py
"""

try:
    x = 1 / 0
except Exception:
    print("خطأ عام")  # هذا ما سيُطبع
except ZeroDivisionError:  # ⚠️ ميت! لن يُنفَّذ أبداً
    print("قسمة على صفر")

# ────────────────────────────────────────────────────────────────────────
# [033] code/chapters/ch11/11-011-ex11_4.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.4: else و finally: إكمال الصورة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-011-ex11_4.py
"""

try:
    # الكود الذي قد يُطلق خطأ
    pass
except SomeError:
    # يُنفَّذ فقط إذا حدث هذا الخطأ المحدد
    pass
else:
    # يُنفَّذ فقط إذا لم يحدث أي خطأ في try
    pass
finally:
    # يُنفَّذ دائماً، سواء حدث خطأ أم لا
    pass

# ────────────────────────────────────────────────────────────────────────
# [034] code/chapters/ch11/11-012-read_and_process_number.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.4: else و finally: إكمال الصورة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-012-read_and_process_number.py
"""

def read_and_process_number(text: str) -> None:
    try:
        number = int(text)
    except ValueError:
        print(f"❌ '{text}' ليس رقماً صحيحاً.")
    else:
        # يُنفَّذ فقط إذا نجح التحويل بدون خطأ
        print(f"✅ تم تحويل الرقم بنجاح: {number}")
        print(f"   ضعفه: {number * 2}")
    finally:
        # يُنفَّذ دائماً - مثالي للتنظيف
        print("--- انتهت المحاولة ---")

read_and_process_number("42")
print()
read_and_process_number("abc")

# ────────────────────────────────────────────────────────────────────────
# [035] code/chapters/ch11/11-013-number.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.4: else و finally: إكمال الصورة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-013-number.py
"""

# أقل وضوحاً - أي سطر بالضبط قد يُطلق الخطأ؟
try:
    number = int(text)
    print(f"ضعفه: {number * 2}")  # ماذا لو هذا السطر نفسه فشل لسبب آخر؟
except ValueError:
    print("خطأ")

# أوضح بكثير - try تحتوي فقط على الكود المحتمل فشله
try:
    number = int(text)
except ValueError:
    print("خطأ")
else:
    print(f"ضعفه: {number * 2}")  # نعلم بالتأكيد أن number تم تحويله بنجاح

# ────────────────────────────────────────────────────────────────────────
# [036] code/chapters/ch11/11-014-process_file.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.4: else و finally: إكمال الصورة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-014-process_file.py
"""

def process_file(filename: str) -> None:
    file = None
    try:
        file = open(filename, "r", encoding="utf-8")
        content = file.read()
        print(f"تم قراءة {len(content)} حرف.")
    except FileNotFoundError:
        print(f"❌ الملف '{filename}' غير موجود.")
    finally:
        if file is not None:
            file.close()
            print("✅ تم إغلاق الملف.")  # يحدث دائماً!

process_file("not_exist.txt")

# ────────────────────────────────────────────────────────────────────────
# [037] code/chapters/ch11/11-015-withdraw.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.5: إطلاق الأخطاء بنفسك: raise

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-015-withdraw.py
"""

def withdraw(balance: float, amount: float) -> float:
    if amount <= 0:
        raise ValueError("مبلغ السحب يجب أن يكون موجباً.")
    if amount > balance:
        raise ValueError("الرصيد غير كافٍ لإتمام العملية.")
    return balance - amount

try:
    new_balance = withdraw(1000, 1500)
except ValueError as e:
    print(f"❌ فشلت العملية: {e}")

# ────────────────────────────────────────────────────────────────────────
# [038] code/chapters/ch11/11-016-process_data.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.5: إطلاق الأخطاء بنفسك: raise

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-016-process_data.py
"""

def process_data(data: list) -> float:
    try:
        return sum(data) / len(data)
    except ZeroDivisionError:
        print("⚠️ تحذير: القائمة فارغة، سيتم إعادة إطلاق الخطأ.")
        raise  # raise بدون معامل = إعادة إطلاق نفس الخطأ الملتقط

try:
    process_data([])
except ZeroDivisionError:
    print("❌ المستوى الأعلى: تم التعامل مع القائمة الفارغة.")

# ────────────────────────────────────────────────────────────────────────
# [039] code/chapters/ch11/11-017-parse_age.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.5: إطلاق الأخطاء بنفسك: raise

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-017-parse_age.py
"""

def parse_age(text: str) -> int:
    try:
        return int(text)
    except ValueError as e:
        raise ValueError(f"فشل تحليل العمر من النص: '{text}'") from e

try:
    parse_age("twenty")
except ValueError as e:
    print(e)
    print(f"السبب الأصلي: {e.__cause__}")

# ────────────────────────────────────────────────────────────────────────
# [040] code/chapters/ch11/11-018-withdraw.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.6: إنشاء استثناءات مخصصة (Custom Exceptions)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-018-withdraw.py
"""

class InsufficientFundsError(Exception):
    """تُطلق عندما يحاول المستخدم سحب مبلغ أكبر من رصيده."""
    pass

class InvalidAmountError(Exception):
    """تُطلق عندما يكون المبلغ المُدخل غير منطقي (سالب أو صفر)."""
    pass

def withdraw(balance: float, amount: float) -> float:
    if amount <= 0:
        raise InvalidAmountError(f"المبلغ {amount} غير صالح؛ يجب أن يكون موجباً.")
    if amount > balance:
        raise InsufficientFundsError(
            f"طلبت سحب {amount}، لكن رصيدك الحالي {balance} فقط."
        )
    return balance - amount

try:
    withdraw(500, 1000)
except InsufficientFundsError as e:
    print(f"💰 خطأ في الرصيد: {e}")
except InvalidAmountError as e:
    print(f"🔢 خطأ في المبلغ: {e}")

# ────────────────────────────────────────────────────────────────────────
# [041] code/chapters/ch11/11-019-process_transaction.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.6: إنشاء استثناءات مخصصة (Custom Exceptions)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-019-process_transaction.py
"""

class BankError(Exception):
    """الفئة الأساسية لكل أخطاء النظام البنكي."""
    pass

class InsufficientFundsError(BankError):
    """رصيد غير كافٍ."""
    pass

class InvalidAmountError(BankError):
    """مبلغ غير صالح."""
    pass

class AccountFrozenError(BankError):
    """الحساب مجمّد."""
    pass

def process_transaction(account_status: str, balance: float, amount: float) -> float:
    if account_status == "frozen":
        raise AccountFrozenError("هذا الحساب مجمّد حالياً.")
    if amount <= 0:
        raise InvalidAmountError("المبلغ يجب أن يكون موجباً.")
    if amount > balance:
        raise InsufficientFundsError("الرصيد غير كافٍ.")
    return balance - amount

try:
    process_transaction("frozen", 1000, 100)
except BankError as e:  # التقاط أي خطأ بنكي بغض النظر عن نوعه الدقيق
    print(f"🏦 خطأ في النظام البنكي: {type(e).__name__}: {e}")

# ────────────────────────────────────────────────────────────────────────
# [042] code/chapters/ch11/11-020-__init__.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.6: إنشاء استثناءات مخصصة (Custom Exceptions)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-020-__init__.py
"""

class ValidationError(Exception):
    """خطأ تحقق من صحة البيانات، مع تفاصيل الحقل المُخالف."""

    def __init__(self, field: str, message: str):
        self.field = field
        self.message = message
        super().__init__(f"خطأ في حقل '{field}': {message}")

def validate_age(age: int) -> None:
    if age < 0 or age > 150:
        raise ValidationError(field="age", message="يجب أن يكون بين 0 و150")

try:
    validate_age(-5)
except ValidationError as e:
    print(f"الحقل المُخالف: {e.field}")
    print(f"الرسالة: {e.message}")

# ────────────────────────────────────────────────────────────────────────
# [043] code/chapters/ch11/11-021-content.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.7: فلسفتان في البرمجة: EAFP مقابل LBYL

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-021-content.py
"""

# تحقق أولاً، ثم نفّذ
import os

if os.path.exists("data.txt"):
    with open("data.txt") as f:
        content = f.read()
else:
    print("الملف غير موجود")

# ────────────────────────────────────────────────────────────────────────
# [044] code/chapters/ch11/11-022-content.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.7: فلسفتان في البرمجة: EAFP مقابل LBYL

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-022-content.py
"""

# نفّذ مباشرة، وتعامل مع الخطأ إن حدث
try:
    with open("data.txt") as f:
        content = f.read()
except FileNotFoundError:
    print("الملف غير موجود")

# ────────────────────────────────────────────────────────────────────────
# [045] code/chapters/ch11/11-023-calculate_average.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.8: دالة assert: التحقق أثناء التطوير

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-023-calculate_average.py
"""

def calculate_average(numbers: list[float]) -> float:
    assert len(numbers) > 0, "القائمة يجب ألا تكون فارغة"
    return sum(numbers) / len(numbers)

print(calculate_average([10, 20, 30]))  # 20.0
# print(calculate_average([]))
# 💥 AssertionError: القائمة يجب ألا تكون فارغة

# ────────────────────────────────────────────────────────────────────────
# [046] code/chapters/ch11/11-024-set_age.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.8: دالة assert: التحقق أثناء التطوير

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-024-set_age.py
"""

# ❌ خطأ: استخدام assert للتحقق من مدخلات مستخدم
def set_age(age: int) -> None:
    assert age >= 0, "العمر يجب أن يكون موجباً"  # قد يُحذف بالكامل!

# ✅ صحيح: استخدام raise للتحقق من مدخلات مستخدم
def set_age(age: int) -> None:
    if age < 0:
        raise ValueError("العمر يجب أن يكون موجباً")

# ────────────────────────────────────────────────────────────────────────
# [047] code/chapters/ch11/11-025-divide.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.9: مقدمة مختصرة لـ logging بدلاً من print

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-025-divide.py
"""

import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def divide(a: float, b: float) -> float | None:
    try:
        return a / b
    except ZeroDivisionError:
        logging.error(f"محاولة قسمة {a} على صفر!")
        return None

divide(10, 0)

# ────────────────────────────────────────────────────────────────────────
# [048] code/chapters/ch11/11-026-register.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.10: مشروع تطبيقي: نظام تسجيل دخول آمن

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-026-register.py
"""

class AuthError(Exception):
    """الفئة الأساسية لكل أخطاء المصادقة."""
    pass

class UserNotFoundError(AuthError):
    """المستخدم غير موجود في النظام."""
    pass

class IncorrectPasswordError(AuthError):
    """كلمة المرور غير صحيحة."""
    pass

class AccountLockedError(AuthError):
    """الحساب مقفل بسبب محاولات فاشلة متكررة."""
    pass

class WeakPasswordError(AuthError):
    """كلمة مرور ضعيفة عند التسجيل."""
    pass

# قاعدة بيانات وهمية (في الواقع ستكون قاعدة بيانات حقيقية)
users_db: dict[str, dict] = {
    "ahmed@test.com": {"password": "Secure123", "failed_attempts": 0, "locked": False},
}

MAX_FAILED_ATTEMPTS = 3

def register(email: str, password: str) -> None:
    """تسجيل مستخدم جديد مع التحقق من قوة كلمة المرور."""
    if len(password) < 8:
        raise WeakPasswordError("كلمة المرور يجب أن تكون 8 أحرف على الأقل.")
    if not any(char.isdigit() for char in password):
        raise WeakPasswordError("كلمة المرور يجب أن تحتوي رقماً واحداً على الأقل.")

    if email in users_db:
        raise ValueError("هذا البريد الإلكتروني مسجّل مسبقاً.")

    users_db[email] = {"password": password, "failed_attempts": 0, "locked": False}
    print(f"✅ تم تسجيل المستخدم {email} بنجاح.")

def login(email: str, password: str) -> bool:
    """محاولة تسجيل دخول، مع حماية من المحاولات المتكررة."""
    try:
        user = users_db[email]
    except KeyError:
        raise UserNotFoundError(f"لا يوجد مستخدم بالبريد: {email}")

    if user["locked"]:
        raise AccountLockedError("هذا الحساب مقفل. تواصل مع الدعم الفني.")

    if user["password"] != password:
        user["failed_attempts"] += 1
        if user["failed_attempts"] >= MAX_FAILED_ATTEMPTS:
            user["locked"] = True
            raise AccountLockedError(
                "تم قفل الحساب بسبب تجاوز عدد المحاولات الفاشلة المسموحة."
            )
        remaining = MAX_FAILED_ATTEMPTS - user["failed_attempts"]
        raise IncorrectPasswordError(
            f"كلمة المرور خاطئة. المحاولات المتبقية: {remaining}"
        )

    user["failed_attempts"] = 0  # إعادة تصفير عند النجاح
    return True

def run_login_attempt(email: str, password: str) -> None:
    """واجهة تفاعلية تلتقط كل أنواع أخطاء المصادقة بوضوح."""
    try:
        login(email, password)
    except UserNotFoundError as e:
        print(f"👤 {e}")
    except AccountLockedError as e:
        print(f"🔒 {e}")
    except IncorrectPasswordError as e:
        print(f"🔑 {e}")
    except AuthError as e:
        print(f"⚠️ خطأ مصادقة غير متوقع: {e}")
    else:
        print(f"✅ تم تسجيل الدخول بنجاح! أهلاً بك {email}")
    finally:
        print("-" * 40)

# --- سيناريوهات تجريبية ---
if __name__ == "__main__":
    try:
        register("sara@test.com", "weak")
    except WeakPasswordError as e:
        print(f"❌ فشل التسجيل: {e}")
    print("-" * 40)

    register("sara@test.com", "Strong123")
    print("-" * 40)

    run_login_attempt("unknown@test.com", "anything")   # UserNotFoundError
    run_login_attempt("ahmed@test.com", "wrong_pass")   # IncorrectPasswordError (محاولة 1)
    run_login_attempt("ahmed@test.com", "wrong_pass")   # IncorrectPasswordError (محاولة 2)
    run_login_attempt("ahmed@test.com", "wrong_pass")   # AccountLockedError (محاولة 3 - قفل!)
    run_login_attempt("ahmed@test.com", "Secure123")    # AccountLockedError (مقفل مسبقاً)
    run_login_attempt("sara@test.com", "Strong123")     # ✅ نجاح

# ────────────────────────────────────────────────────────────────────────
# [049] code/chapters/ch11/11-027-reset_password.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.10: مشروع تطبيقي: نظام تسجيل دخول آمن

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-027-reset_password.py
"""

def reset_password(email: str, new_password: str) -> None:
    """إعادة تعيين كلمة مرور، مع إلغاء القفل إن وُجد."""
    try:
        user = users_db[email]
    except KeyError:
        raise UserNotFoundError(f"لا يوجد مستخدم بالبريد: {email}")

    if len(new_password) < 8 or not any(c.isdigit() for c in new_password):
        raise WeakPasswordError("كلمة المرور الجديدة ضعيفة جداً.")

    user["password"] = new_password
    user["failed_attempts"] = 0
    user["locked"] = False
    print(f"🔓 تم إعادة تعيين كلمة المرور وفك قفل الحساب لـ {email}")

try:
    reset_password("ahmed@test.com", "NewPass123")
except AuthError as e:
    print(f"❌ {e}")

# ────────────────────────────────────────────────────────────────────────
# [050] code/chapters/ch11/11-028-safe_list_access.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.11: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-028-safe_list_access.py
"""

# تمرين 1
try:
    num = int(input("أدخل رقماً: "))
    print("رقم صحيح")
except ValueError:
    print("خطأ: ليس رقماً")

# تمرين 2
def safe_list_access(lst: list, index: int):
    try:
        return lst[index]
    except IndexError:
        return None

print(safe_list_access([1, 2, 3], 1))   # 2
print(safe_list_access([1, 2, 3], 10))  # None

# ────────────────────────────────────────────────────────────────────────
# [051] code/chapters/ch11/11-029-calculate_square_root.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.11: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-029-calculate_square_root.py
"""

import math

# تمرين 3
class NegativeValueError(Exception):
    pass

def calculate_square_root(n: float) -> float:
    if n < 0:
        raise NegativeValueError(f"لا يمكن حساب جذر رقم سالب: {n}")
    return math.sqrt(n)

try:
    print(calculate_square_root(-9))
except NegativeValueError as e:
    print(f"❌ {e}")

# تمرين 4
class MissingConfigError(Exception):
    def __init__(self, key: str):
        self.key = key
        super().__init__(f"الإعداد المطلوب '{key}' غير موجود.")

def parse_config(data: dict) -> tuple[str, int]:
    try:
        host = data["host"]
    except KeyError:
        raise MissingConfigError("host")

    try:
        port = data["port"]
    except KeyError:
        raise MissingConfigError("port")

    return host, port

try:
    parse_config({"host": "localhost"})
except MissingConfigError as e:
    print(f"❌ {e}")  # ❌ الإعداد المطلوب 'port' غير موجود.

# ────────────────────────────────────────────────────────────────────────
# [052] code/chapters/ch11/11-030-__init__.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.11: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-030-__init__.py
"""

class APIError(Exception):
    """الفئة الأساسية لكل أخطاء API."""
    pass

class EndpointNotFoundError(APIError):
    def __init__(self, endpoint: str):
        self.endpoint = endpoint
        super().__init__(f"المسار '{endpoint}' غير موجود.")

class MissingFieldError(APIError):
    def __init__(self, field: str):
        self.field = field
        super().__init__(f"الحقل المطلوب '{field}' مفقود.")

class InvalidFieldError(APIError):
    def __init__(self, field: str, reason: str):
        self.field = field
        self.reason = reason
        super().__init__(f"الحقل '{field}' غير صالح: {reason}")

VALID_ENDPOINTS = {"/users", "/orders", "/products"}

def process_api_request(endpoint: str, data: dict) -> dict:
    if endpoint not in VALID_ENDPOINTS:
        raise EndpointNotFoundError(endpoint)

    if "user_id" not in data:
        raise MissingFieldError("user_id")

    user_id = data["user_id"]
    if not isinstance(user_id, int) or user_id <= 0:
        raise InvalidFieldError("user_id", "يجب أن يكون رقماً صحيحاً موجباً")

    return {"status": "success", "endpoint": endpoint, "user_id": user_id}

def handle_request(endpoint: str, data: dict) -> None:
    try:
        result = process_api_request(endpoint, data)
    except EndpointNotFoundError as e:
        print(f"404 - {e}")
    except MissingFieldError as e:
        print(f"400 - حقل مفقود: {e}")
    except InvalidFieldError as e:
        print(f"400 - حقل غير صالح: {e}")
    except APIError as e:
        print(f"500 - خطأ API غير متوقع: {e}")
    else:
        print(f"200 - {result}")

handle_request("/comments", {"user_id": 1})         # 404
handle_request("/users", {})                         # 400 مفقود
handle_request("/users", {"user_id": -5})            # 400 غير صالح
handle_request("/users", {"user_id": 42})            # 200 نجاح

# ────────────────────────────────────────────────────────────────────────
# [053] code/chapters/ch11/11-031-__init__.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.13: ملخص الفصل (Cheat Sheet)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-031-__init__.py
"""

# الصيغة الكاملة
try:
    risky_code()
except SpecificError as e:
    handle_specific_case(e)
except (ErrorA, ErrorB) as e:
    handle_multiple_cases(e)
except Exception as e:      # الأعم - دائماً في الأخير
    handle_general_case(e)
else:
    runs_if_no_error()
finally:
    always_runs()            # تنظيف الموارد

# إطلاق خطأ
raise ValueError("رسالة واضحة")

# إعادة إطلاق
except SomeError:
    raise  # بدون معامل

# تسلسل الاستثناءات
raise NewError("...") from original_error

# استثناء مخصص
class MyAppError(Exception):
    pass

class SpecificError(MyAppError):
    def __init__(self, detail: str):
        self.detail = detail
        super().__init__(f"فشل بسبب: {detail}")

# assert - للتطوير فقط، ليس لمدخلات المستخدم
assert condition, "رسالة إذا فشل الشرط"

# ────────────────────────────────────────────────────────────────────────
# [054] dist/markdown/chapter-11.md
# ────────────────────────────────────────────────────────────────────────
balance = 1000
withdrawal = input("أدخل مبلغ السحب: ")  # المستخدم يكتب "مئة"
balance -= int(withdrawal)  # 💥 ValueError: invalid literal for int()
# البرنامج ينهار، والمستخدم يرى رسالة تقنية غير مفهومة

# ────────────────────────────────────────────────────────────────────────
# [055] dist/markdown/chapter-11.md
# ────────────────────────────────────────────────────────────────────────
print(10 / 0)        # ZeroDivisionError
print(int("abc"))    # ValueError
print(my_list[99])   # IndexError
print(my_dict["x"])  # KeyError
print(undefined_var) # NameError

# ────────────────────────────────────────────────────────────────────────
# [056] dist/markdown/chapter-11.md
# ────────────────────────────────────────────────────────────────────────
def divide(a, b):
    return a / b

result = divide(10, 0)

# ────────────────────────────────────────────────────────────────────────
# [057] dist/markdown/chapter-11.md
# ────────────────────────────────────────────────────────────────────────
try:
    number = int(input("أدخل رقماً: "))
    print(f"ضعف الرقم: {number * 2}")
except ValueError:
    print("❌ خطأ: الرجاء إدخال رقم صحيح فقط.")

print("البرنامج استمر في العمل!")

# ────────────────────────────────────────────────────────────────────────
# [058] dist/markdown/chapter-11.md
# ────────────────────────────────────────────────────────────────────────
def safe_divide(a: float, b: float) -> float | None:
    try:
        return a / b
    except ZeroDivisionError:
        print("❌ لا يمكن القسمة على صفر.")
        return None
    except TypeError:
        print("❌ يجب أن تكون القيم أرقاماً.")
        return None

print(safe_divide(10, 0))        # ❌ لا يمكن القسمة على صفر. → None
print(safe_divide(10, "abc"))    # ❌ يجب أن تكون القيم أرقاماً. → None
print(safe_divide(10, 2))        # 5.0

# ────────────────────────────────────────────────────────────────────────
# [059] dist/markdown/chapter-11.md
# ────────────────────────────────────────────────────────────────────────
try:
    value = my_dict[key]
    result = 100 / value
except (KeyError, ZeroDivisionError) as e:
    print(f"حدث خطأ: {e}")

# ────────────────────────────────────────────────────────────────────────
# [060] dist/markdown/chapter-11.md
# ────────────────────────────────────────────────────────────────────────
try:
    risky_operation()
except Exception as e:
    print(f"حدث خطأ غير متوقع: {e}")

# ────────────────────────────────────────────────────────────────────────
# [061] dist/markdown/chapter-11.md
# ────────────────────────────────────────────────────────────────────────
# ❌ الأسوأ على الإطلاق - لا تفعل هذا أبداً!
try:
    risky_operation()
except:  # bare except - يلتقط كل شيء حتى KeyboardInterrupt!
    pass  # يُخفي الخطأ تماماً دون أي أثر!

# ────────────────────────────────────────────────────────────────────────
# [062] dist/markdown/chapter-11.md
# ────────────────────────────────────────────────────────────────────────
# ترتيب except مهم جداً! الأكثر تحديداً أولاً
try:
    value = my_list[10]
except IndexError:
    print("فهرس خارج النطاق")
except LookupError:  # تأتي بعد IndexError لأنها أعم منها
    print("خطأ بحث عام")

# ────────────────────────────────────────────────────────────────────────
# [063] dist/markdown/chapter-11.md
# ────────────────────────────────────────────────────────────────────────
try:
    # الكود الذي قد يُطلق خطأ
    pass
except SomeError:
    # يُنفَّذ فقط إذا حدث هذا الخطأ المحدد
    pass
else:
    # يُنفَّذ فقط إذا لم يحدث أي خطأ في try
    pass
finally:
    # يُنفَّذ دائماً، سواء حدث خطأ أم لا
    pass

# ────────────────────────────────────────────────────────────────────────
# [064] dist/markdown/chapter-11.md
# ────────────────────────────────────────────────────────────────────────
# أقل وضوحاً - أي سطر بالضبط قد يُطلق الخطأ؟
try:
    number = int(text)
    print(f"ضعفه: {number * 2}")  # ماذا لو هذا السطر نفسه فشل لسبب آخر؟
except ValueError:
    print("خطأ")

# أوضح بكثير - try تحتوي فقط على الكود المحتمل فشله
try:
    number = int(text)
except ValueError:
    print("خطأ")
else:
    print(f"ضعفه: {number * 2}")  # نعلم بالتأكيد أن number تم تحويله بنجاح

# ────────────────────────────────────────────────────────────────────────
# [065] dist/markdown/chapter-11.md
# ────────────────────────────────────────────────────────────────────────
def calculate_average(numbers: list[float]) -> float:
    assert len(numbers) > 0, "القائمة يجب ألا تكون فارغة"
    return sum(numbers) / len(numbers)

print(calculate_average([10, 20, 30]))  # 20.0
# print(calculate_average([]))
# 💥 AssertionError: القائمة يجب ألا تكون فارغة

# ────────────────────────────────────────────────────────────────────────
# [066] dist/markdown/chapter-11.md
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: استخدام assert للتحقق من مدخلات مستخدم
def set_age(age: int) -> None:
    assert age >= 0, "العمر يجب أن يكون موجباً"  # قد يُحذف بالكامل!

# ✅ صحيح: استخدام raise للتحقق من مدخلات مستخدم
def set_age(age: int) -> None:
    if age < 0:
        raise ValueError("العمر يجب أن يكون موجباً")

# ────────────────────────────────────────────────────────────────────────
# [067] dist/markdown/chapter-11.md
# ────────────────────────────────────────────────────────────────────────
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def divide(a: float, b: float) -> float | None:
    try:
        return a / b
    except ZeroDivisionError:
        logging.error(f"محاولة قسمة {a} على صفر!")
        return None

divide(10, 0)

# ────────────────────────────────────────────────────────────────────────
# [068] dist/markdown/chapter-11.md
# ────────────────────────────────────────────────────────────────────────
# تمرين 1
try:
    num = int(input("أدخل رقماً: "))
    print("رقم صحيح")
except ValueError:
    print("خطأ: ليس رقماً")

# تمرين 2
def safe_list_access(lst: list, index: int):
    try:
        return lst[index]
    except IndexError:
        return None

print(safe_list_access([1, 2, 3], 1))   # 2
print(safe_list_access([1, 2, 3], 10))  # None

# ────────────────────────────────────────────────────────────────────────
# [069] dist/markdown/chapter-11.md
# ────────────────────────────────────────────────────────────────────────
# الصيغة الكاملة
try:
    risky_code()
except SpecificError as e:
    handle_specific_case(e)
except (ErrorA, ErrorB) as e:
    handle_multiple_cases(e)
except Exception as e:      # الأعم - دائماً في الأخير
    handle_general_case(e)
else:
    runs_if_no_error()
finally:
    always_runs()            # تنظيف الموارد

# إطلاق خطأ
raise ValueError("رسالة واضحة")

# إعادة إطلاق
except SomeError:
    raise  # بدون معامل

# تسلسل الاستثناءات
raise NewError("...") from original_error

# استثناء مخصص
class MyAppError(Exception):
    pass

class SpecificError(MyAppError):
    def __init__(self, detail: str):
        self.detail = detail
        super().__init__(f"فشل بسبب: {detail}")

# assert - للتطوير فقط، ليس لمدخلات المستخدم
assert condition, "رسالة إذا فشل الشرط"

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-11-chapter-11.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#@title ⚙️ تهيئة البيئة  { display-mode: "form" }
import sys, os, random, pathlib

print("Python:", sys.version.split()[0], "|", sys.platform)
assert sys.version_info >= (3, 10), "هذا الكتاب يفترض Python 3.10 أو أحدث"

SEED = 42
random.seed(SEED)

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-11-chapter-11.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
# ⚠️ مثال يُظهر خطأً مقصودًا: نلتقطه هنا ليكتمل «تشغيل الكل»،
# وسترى نوع الخطأ ورسالته في المخرجات كما في الطرفية.
try:
    def divide(a, b):
        return a / b

    result = divide(10, 0)
except (Exception, SystemExit) as _err:
    print(f"⚠️ {type(_err).__name__}: {_err}")

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-11-chapter-11.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# 📝 نسخة الدفتر: حقل نموذج بدل الإدخال اليدوي — عدّل القيمة ثم أعد التشغيل
try:
    number = 10  #@param {type:"integer"}
    print(f"ضعف الرقم: {number * 2}")
except ValueError:
    print("❌ خطأ: الرجاء إدخال رقم صحيح فقط.")

print("البرنامج استمر في العمل!")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-11-chapter-11.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# ⚠️ مثال يُظهر خطأً مقصودًا: نلتقطه هنا ليكتمل «تشغيل الكل»،
# وسترى نوع الخطأ ورسالته في المخرجات كما في الطرفية.
try:
    def safe_divide(a: float, b: float) -> float | None:
        try:
            return a / b
        except ZeroDivisionError:
            print("❌ لا يمكن القسمة على صفر.")
            return None
        except TypeError:
            print("❌ يجب أن تكون القيم أرقاماً.")
            return None

    print(safe_divide(10, 0))        # ❌ لا يمكن القسمة على صفر. → None
    print(safe_divide(10, "abc"))    # ❌ يجب أن تكون القيم أرقاماً. → None
    print(safe_divide(10, 2))        # 5.0
except (Exception, SystemExit) as _err:
    print(f"⚠️ {type(_err).__name__}: {_err}")

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-11-chapter-11.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
try:
    x = 1 / 0
except Exception:
    print("خطأ عام")  # هذا ما سيُطبع
except ZeroDivisionError:  # ⚠️ ميت! لن يُنفَّذ أبداً
    print("قسمة على صفر")

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-11-chapter-11.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
def read_and_process_number(text: str) -> None:
    try:
        number = int(text)
    except ValueError:
        print(f"❌ '{text}' ليس رقماً صحيحاً.")
    else:
        # يُنفَّذ فقط إذا نجح التحويل بدون خطأ
        print(f"✅ تم تحويل الرقم بنجاح: {number}")
        print(f"   ضعفه: {number * 2}")
    finally:
        # يُنفَّذ دائماً - مثالي للتنظيف
        print("--- انتهت المحاولة ---")

read_and_process_number("42")
print()
read_and_process_number("abc")

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-11-chapter-11.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
def process_file(filename: str) -> None:
    file = None
    try:
        file = open(filename, "r", encoding="utf-8")
        content = file.read()
        print(f"تم قراءة {len(content)} حرف.")
    except FileNotFoundError:
        print(f"❌ الملف '{filename}' غير موجود.")
    finally:
        if file is not None:
            file.close()
            print("✅ تم إغلاق الملف.")  # يحدث دائماً!

process_file("not_exist.txt")

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-11-chapter-11.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
def withdraw(balance: float, amount: float) -> float:
    if amount <= 0:
        raise ValueError("مبلغ السحب يجب أن يكون موجباً.")
    if amount > balance:
        raise ValueError("الرصيد غير كافٍ لإتمام العملية.")
    return balance - amount

try:
    new_balance = withdraw(1000, 1500)
except ValueError as e:
    print(f"❌ فشلت العملية: {e}")

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-11-chapter-11.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
def process_data(data: list) -> float:
    try:
        return sum(data) / len(data)
    except ZeroDivisionError:
        print("⚠️ تحذير: القائمة فارغة، سيتم إعادة إطلاق الخطأ.")
        raise  # raise بدون معامل = إعادة إطلاق نفس الخطأ الملتقط

try:
    process_data([])
except ZeroDivisionError:
    print("❌ المستوى الأعلى: تم التعامل مع القائمة الفارغة.")

# ────────────────────────────────────────────────────────────────────────
# [011] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-11-chapter-11.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
def parse_age(text: str) -> int:
    try:
        return int(text)
    except ValueError as e:
        raise ValueError(f"فشل تحليل العمر من النص: '{text}'") from e

try:
    parse_age("twenty")
except ValueError as e:
    print(e)
    print(f"السبب الأصلي: {e.__cause__}")

# ────────────────────────────────────────────────────────────────────────
# [012] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-11-chapter-11.ipynb#cell-11
# ────────────────────────────────────────────────────────────────────────
class InsufficientFundsError(Exception):
    """تُطلق عندما يحاول المستخدم سحب مبلغ أكبر من رصيده."""
    pass

class InvalidAmountError(Exception):
    """تُطلق عندما يكون المبلغ المُدخل غير منطقي (سالب أو صفر)."""
    pass

def withdraw(balance: float, amount: float) -> float:
    if amount <= 0:
        raise InvalidAmountError(f"المبلغ {amount} غير صالح؛ يجب أن يكون موجباً.")
    if amount > balance:
        raise InsufficientFundsError(
            f"طلبت سحب {amount}، لكن رصيدك الحالي {balance} فقط."
        )
    return balance - amount

try:
    withdraw(500, 1000)
except InsufficientFundsError as e:
    print(f"💰 خطأ في الرصيد: {e}")
except InvalidAmountError as e:
    print(f"🔢 خطأ في المبلغ: {e}")

# ────────────────────────────────────────────────────────────────────────
# [013] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-11-chapter-11.ipynb#cell-12
# ────────────────────────────────────────────────────────────────────────
class BankError(Exception):
    """الفئة الأساسية لكل أخطاء النظام البنكي."""
    pass

class InsufficientFundsError(BankError):
    """رصيد غير كافٍ."""
    pass

class InvalidAmountError(BankError):
    """مبلغ غير صالح."""
    pass

class AccountFrozenError(BankError):
    """الحساب مجمّد."""
    pass

def process_transaction(account_status: str, balance: float, amount: float) -> float:
    if account_status == "frozen":
        raise AccountFrozenError("هذا الحساب مجمّد حالياً.")
    if amount <= 0:
        raise InvalidAmountError("المبلغ يجب أن يكون موجباً.")
    if amount > balance:
        raise InsufficientFundsError("الرصيد غير كافٍ.")
    return balance - amount

try:
    process_transaction("frozen", 1000, 100)
except BankError as e:  # التقاط أي خطأ بنكي بغض النظر عن نوعه الدقيق
    print(f"🏦 خطأ في النظام البنكي: {type(e).__name__}: {e}")

# ────────────────────────────────────────────────────────────────────────
# [014] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-11-chapter-11.ipynb#cell-13
# ────────────────────────────────────────────────────────────────────────
class ValidationError(Exception):
    """خطأ تحقق من صحة البيانات، مع تفاصيل الحقل المُخالف."""

    def __init__(self, field: str, message: str):
        self.field = field
        self.message = message
        super().__init__(f"خطأ في حقل '{field}': {message}")

def validate_age(age: int) -> None:
    if age < 0 or age > 150:
        raise ValidationError(field="age", message="يجب أن يكون بين 0 و150")

try:
    validate_age(-5)
except ValidationError as e:
    print(f"الحقل المُخالف: {e.field}")
    print(f"الرسالة: {e.message}")

# ────────────────────────────────────────────────────────────────────────
# [015] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-11-chapter-11.ipynb#cell-14
# ────────────────────────────────────────────────────────────────────────
# تحقق أولاً، ثم نفّذ
import os

if os.path.exists("data.txt"):
    with open("data.txt") as f:
        content = f.read()
else:
    print("الملف غير موجود")

# ────────────────────────────────────────────────────────────────────────
# [016] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-11-chapter-11.ipynb#cell-15
# ────────────────────────────────────────────────────────────────────────
# نفّذ مباشرة، وتعامل مع الخطأ إن حدث
try:
    with open("data.txt") as f:
        content = f.read()
except FileNotFoundError:
    print("الملف غير موجود")

# ────────────────────────────────────────────────────────────────────────
# [017] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-11-chapter-11.ipynb#cell-16
# ────────────────────────────────────────────────────────────────────────
# ⚠️ مثال يُظهر خطأً مقصودًا: نلتقطه هنا ليكتمل «تشغيل الكل»،
# وسترى نوع الخطأ ورسالته في المخرجات كما في الطرفية.
try:
    def calculate_average(numbers: list[float]) -> float:
        assert len(numbers) > 0, "القائمة يجب ألا تكون فارغة"
        return sum(numbers) / len(numbers)

    print(calculate_average([10, 20, 30]))  # 20.0
    # print(calculate_average([]))
    # 💥 AssertionError: القائمة يجب ألا تكون فارغة
except (Exception, SystemExit) as _err:
    print(f"⚠️ {type(_err).__name__}: {_err}")

# ────────────────────────────────────────────────────────────────────────
# [018] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-11-chapter-11.ipynb#cell-17
# ────────────────────────────────────────────────────────────────────────
# ⚠️ مثال يُظهر خطأً مقصودًا: نلتقطه هنا ليكتمل «تشغيل الكل»،
# وسترى نوع الخطأ ورسالته في المخرجات كما في الطرفية.
try:
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    def divide(a: float, b: float) -> float | None:
        try:
            return a / b
        except ZeroDivisionError:
            logging.error(f"محاولة قسمة {a} على صفر!")
            return None

    divide(10, 0)
except (Exception, SystemExit) as _err:
    print(f"⚠️ {type(_err).__name__}: {_err}")

# ────────────────────────────────────────────────────────────────────────
# [019] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-11-chapter-11.ipynb#cell-18
# ────────────────────────────────────────────────────────────────────────
class AuthError(Exception):
    """الفئة الأساسية لكل أخطاء المصادقة."""
    pass

class UserNotFoundError(AuthError):
    """المستخدم غير موجود في النظام."""
    pass

class IncorrectPasswordError(AuthError):
    """كلمة المرور غير صحيحة."""
    pass

class AccountLockedError(AuthError):
    """الحساب مقفل بسبب محاولات فاشلة متكررة."""
    pass

class WeakPasswordError(AuthError):
    """كلمة مرور ضعيفة عند التسجيل."""
    pass

# قاعدة بيانات وهمية (في الواقع ستكون قاعدة بيانات حقيقية)
users_db: dict[str, dict] = {
    "ahmed@test.com": {"password": "Secure123", "failed_attempts": 0, "locked": False},
}

MAX_FAILED_ATTEMPTS = 3

def register(email: str, password: str) -> None:
    """تسجيل مستخدم جديد مع التحقق من قوة كلمة المرور."""
    if len(password) < 8:
        raise WeakPasswordError("كلمة المرور يجب أن تكون 8 أحرف على الأقل.")
    if not any(char.isdigit() for char in password):
        raise WeakPasswordError("كلمة المرور يجب أن تحتوي رقماً واحداً على الأقل.")

    if email in users_db:
        raise ValueError("هذا البريد الإلكتروني مسجّل مسبقاً.")

    users_db[email] = {"password": password, "failed_attempts": 0, "locked": False}
    print(f"✅ تم تسجيل المستخدم {email} بنجاح.")

def login(email: str, password: str) -> bool:
    """محاولة تسجيل دخول، مع حماية من المحاولات المتكررة."""
    try:
        user = users_db[email]
    except KeyError:
        raise UserNotFoundError(f"لا يوجد مستخدم بالبريد: {email}")

    if user["locked"]:
        raise AccountLockedError("هذا الحساب مقفل. تواصل مع الدعم الفني.")

    if user["password"] != password:
        user["failed_attempts"] += 1
        if user["failed_attempts"] >= MAX_FAILED_ATTEMPTS:
            user["locked"] = True
            raise AccountLockedError(
                "تم قفل الحساب بسبب تجاوز عدد المحاولات الفاشلة المسموحة."
            )
        remaining = MAX_FAILED_ATTEMPTS - user["failed_attempts"]
        raise IncorrectPasswordError(
            f"كلمة المرور خاطئة. المحاولات المتبقية: {remaining}"
        )

    user["failed_attempts"] = 0  # إعادة تصفير عند النجاح
    return True

def run_login_attempt(email: str, password: str) -> None:
    """واجهة تفاعلية تلتقط كل أنواع أخطاء المصادقة بوضوح."""
    try:
        login(email, password)
    except UserNotFoundError as e:
        print(f"👤 {e}")
    except AccountLockedError as e:
        print(f"🔒 {e}")
    except IncorrectPasswordError as e:
        print(f"🔑 {e}")
    except AuthError as e:
        print(f"⚠️ خطأ مصادقة غير متوقع: {e}")
    else:
        print(f"✅ تم تسجيل الدخول بنجاح! أهلاً بك {email}")
    finally:
        print("-" * 40)

# --- سيناريوهات تجريبية ---
if __name__ == "__main__":
    try:
        register("sara@test.com", "weak")
    except WeakPasswordError as e:
        print(f"❌ فشل التسجيل: {e}")
    print("-" * 40)

    register("sara@test.com", "Strong123")
    print("-" * 40)

    run_login_attempt("unknown@test.com", "anything")   # UserNotFoundError
    run_login_attempt("ahmed@test.com", "wrong_pass")   # IncorrectPasswordError (محاولة 1)
    run_login_attempt("ahmed@test.com", "wrong_pass")   # IncorrectPasswordError (محاولة 2)
    run_login_attempt("ahmed@test.com", "wrong_pass")   # AccountLockedError (محاولة 3 - قفل!)
    run_login_attempt("ahmed@test.com", "Secure123")    # AccountLockedError (مقفل مسبقاً)
    run_login_attempt("sara@test.com", "Strong123")     # ✅ نجاح

# ────────────────────────────────────────────────────────────────────────
# [020] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-11-chapter-11.ipynb#cell-19
# ────────────────────────────────────────────────────────────────────────
def reset_password(email: str, new_password: str) -> None:
    """إعادة تعيين كلمة مرور، مع إلغاء القفل إن وُجد."""
    try:
        user = users_db[email]
    except KeyError:
        raise UserNotFoundError(f"لا يوجد مستخدم بالبريد: {email}")

    if len(new_password) < 8 or not any(c.isdigit() for c in new_password):
        raise WeakPasswordError("كلمة المرور الجديدة ضعيفة جداً.")

    user["password"] = new_password
    user["failed_attempts"] = 0
    user["locked"] = False
    print(f"🔓 تم إعادة تعيين كلمة المرور وفك قفل الحساب لـ {email}")

try:
    reset_password("ahmed@test.com", "NewPass123")
except AuthError as e:
    print(f"❌ {e}")

# ────────────────────────────────────────────────────────────────────────
# [021] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-11-chapter-11.ipynb#cell-20
# ────────────────────────────────────────────────────────────────────────
# 📝 نسخة الدفتر: حقل نموذج بدل الإدخال اليدوي — عدّل القيمة ثم أعد التشغيل
# تمرين 1
try:
    num = 10  #@param {type:"integer"}
    print("رقم صحيح")
except ValueError:
    print("خطأ: ليس رقماً")

# تمرين 2
def safe_list_access(lst: list, index: int):
    try:
        return lst[index]
    except IndexError:
        return None

print(safe_list_access([1, 2, 3], 1))   # 2
print(safe_list_access([1, 2, 3], 10))  # None

# ────────────────────────────────────────────────────────────────────────
# [022] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-11-chapter-11.ipynb#cell-21
# ────────────────────────────────────────────────────────────────────────
import math

# تمرين 3
class NegativeValueError(Exception):
    pass

def calculate_square_root(n: float) -> float:
    if n < 0:
        raise NegativeValueError(f"لا يمكن حساب جذر رقم سالب: {n}")
    return math.sqrt(n)

try:
    print(calculate_square_root(-9))
except NegativeValueError as e:
    print(f"❌ {e}")

# تمرين 4
class MissingConfigError(Exception):
    def __init__(self, key: str):
        self.key = key
        super().__init__(f"الإعداد المطلوب '{key}' غير موجود.")

def parse_config(data: dict) -> tuple[str, int]:
    try:
        host = data["host"]
    except KeyError:
        raise MissingConfigError("host")

    try:
        port = data["port"]
    except KeyError:
        raise MissingConfigError("port")

    return host, port

try:
    parse_config({"host": "localhost"})
except MissingConfigError as e:
    print(f"❌ {e}")  # ❌ الإعداد المطلوب 'port' غير موجود.

# ────────────────────────────────────────────────────────────────────────
# [023] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-11-chapter-11.ipynb#cell-22
# ────────────────────────────────────────────────────────────────────────
class APIError(Exception):
    """الفئة الأساسية لكل أخطاء API."""
    pass

class EndpointNotFoundError(APIError):
    def __init__(self, endpoint: str):
        self.endpoint = endpoint
        super().__init__(f"المسار '{endpoint}' غير موجود.")

class MissingFieldError(APIError):
    def __init__(self, field: str):
        self.field = field
        super().__init__(f"الحقل المطلوب '{field}' مفقود.")

class InvalidFieldError(APIError):
    def __init__(self, field: str, reason: str):
        self.field = field
        self.reason = reason
        super().__init__(f"الحقل '{field}' غير صالح: {reason}")

VALID_ENDPOINTS = {"/users", "/orders", "/products"}

def process_api_request(endpoint: str, data: dict) -> dict:
    if endpoint not in VALID_ENDPOINTS:
        raise EndpointNotFoundError(endpoint)

    if "user_id" not in data:
        raise MissingFieldError("user_id")

    user_id = data["user_id"]
    if not isinstance(user_id, int) or user_id <= 0:
        raise InvalidFieldError("user_id", "يجب أن يكون رقماً صحيحاً موجباً")

    return {"status": "success", "endpoint": endpoint, "user_id": user_id}

def handle_request(endpoint: str, data: dict) -> None:
    try:
        result = process_api_request(endpoint, data)
    except EndpointNotFoundError as e:
        print(f"404 - {e}")
    except MissingFieldError as e:
        print(f"400 - حقل مفقود: {e}")
    except InvalidFieldError as e:
        print(f"400 - حقل غير صالح: {e}")
    except APIError as e:
        print(f"500 - خطأ API غير متوقع: {e}")
    else:
        print(f"200 - {result}")

handle_request("/comments", {"user_id": 1})         # 404
handle_request("/users", {})                         # 400 مفقود
handle_request("/users", {"user_id": -5})            # 400 غير صالح
handle_request("/users", {"user_id": 42})            # 200 نجاح

# ────────────────────────────────────────────────────────────────────────
# [024] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-11-chapter-11.ipynb#cell-23
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.1: لماذا يحتاج كل برنامج احترافي لمعالجة الأخطاء؟

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-001-balance.py
"""

balance = 1000
withdrawal = input("أدخل مبلغ السحب: ")  # المستخدم يكتب "مئة"
balance -= int(withdrawal)  # 💥 ValueError: invalid literal for int()
# البرنامج ينهار، والمستخدم يرى رسالة تقنية غير مفهومة

# ────────────────────────────────────────────────────────────────────────
# [025] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-11-chapter-11.ipynb#cell-24
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.2: فهم الاستثناءات (Exceptions)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-002-ex11_2.py
"""

print(10 / 0)        # ZeroDivisionError
print(int("abc"))    # ValueError
print(my_list[99])   # IndexError
print(my_dict["x"])  # KeyError
print(undefined_var) # NameError

# ────────────────────────────────────────────────────────────────────────
# [026] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-11-chapter-11.ipynb#cell-25
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.2: فهم الاستثناءات (Exceptions)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-003-divide.py
"""

def divide(a, b):
    return a / b

result = divide(10, 0)

# ────────────────────────────────────────────────────────────────────────
# [027] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-11-chapter-11.ipynb#cell-26
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.3: الصيغة الأساسية: try / except

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-004-number.py
"""

try:
    number = int(input("أدخل رقماً: "))
    print(f"ضعف الرقم: {number * 2}")
except ValueError:
    print("❌ خطأ: الرجاء إدخال رقم صحيح فقط.")

print("البرنامج استمر في العمل!")

# ────────────────────────────────────────────────────────────────────────
# [028] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-11-chapter-11.ipynb#cell-27
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.3: الصيغة الأساسية: try / except

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-005-safe_divide.py
"""

def safe_divide(a: float, b: float) -> float | None:
    try:
        return a / b
    except ZeroDivisionError:
        print("❌ لا يمكن القسمة على صفر.")
        return None
    except TypeError:
        print("❌ يجب أن تكون القيم أرقاماً.")
        return None

print(safe_divide(10, 0))        # ❌ لا يمكن القسمة على صفر. → None
print(safe_divide(10, "abc"))    # ❌ يجب أن تكون القيم أرقاماً. → None
print(safe_divide(10, 2))        # 5.0

# ────────────────────────────────────────────────────────────────────────
# [029] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-11-chapter-11.ipynb#cell-28
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.3: الصيغة الأساسية: try / except

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-006-value.py
"""

try:
    value = my_dict[key]
    result = 100 / value
except (KeyError, ZeroDivisionError) as e:
    print(f"حدث خطأ: {e}")

# ────────────────────────────────────────────────────────────────────────
# [030] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-11-chapter-11.ipynb#cell-29
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.3: الصيغة الأساسية: try / except

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-007-ex11_3.py
"""

try:
    risky_operation()
except Exception as e:
    print(f"حدث خطأ غير متوقع: {e}")

# ────────────────────────────────────────────────────────────────────────
# [031] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-11-chapter-11.ipynb#cell-30
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.3: الصيغة الأساسية: try / except

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-008-ex11_3.py
"""

# ❌ الأسوأ على الإطلاق - لا تفعل هذا أبداً!
try:
    risky_operation()
except:  # bare except - يلتقط كل شيء حتى KeyboardInterrupt!
    pass  # يُخفي الخطأ تماماً دون أي أثر!

# ────────────────────────────────────────────────────────────────────────
# [032] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-11-chapter-11.ipynb#cell-31
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.3: الصيغة الأساسية: try / except

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-009-value.py
"""

# ترتيب except مهم جداً! الأكثر تحديداً أولاً
try:
    value = my_list[10]
except IndexError:
    print("فهرس خارج النطاق")
except LookupError:  # تأتي بعد IndexError لأنها أعم منها
    print("خطأ بحث عام")

# ────────────────────────────────────────────────────────────────────────
# [033] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-11-chapter-11.ipynb#cell-32
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.3: الصيغة الأساسية: try / except

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-010-ex11_3.py
"""

try:
    x = 1 / 0
except Exception:
    print("خطأ عام")  # هذا ما سيُطبع
except ZeroDivisionError:  # ⚠️ ميت! لن يُنفَّذ أبداً
    print("قسمة على صفر")

# ────────────────────────────────────────────────────────────────────────
# [034] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-11-chapter-11.ipynb#cell-33
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.4: else و finally: إكمال الصورة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-011-ex11_4.py
"""

try:
    # الكود الذي قد يُطلق خطأ
    pass
except SomeError:
    # يُنفَّذ فقط إذا حدث هذا الخطأ المحدد
    pass
else:
    # يُنفَّذ فقط إذا لم يحدث أي خطأ في try
    pass
finally:
    # يُنفَّذ دائماً، سواء حدث خطأ أم لا
    pass

# ────────────────────────────────────────────────────────────────────────
# [035] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-11-chapter-11.ipynb#cell-34
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.4: else و finally: إكمال الصورة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-012-read_and_process_number.py
"""

def read_and_process_number(text: str) -> None:
    try:
        number = int(text)
    except ValueError:
        print(f"❌ '{text}' ليس رقماً صحيحاً.")
    else:
        # يُنفَّذ فقط إذا نجح التحويل بدون خطأ
        print(f"✅ تم تحويل الرقم بنجاح: {number}")
        print(f"   ضعفه: {number * 2}")
    finally:
        # يُنفَّذ دائماً - مثالي للتنظيف
        print("--- انتهت المحاولة ---")

read_and_process_number("42")
print()
read_and_process_number("abc")

# ────────────────────────────────────────────────────────────────────────
# [036] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-11-chapter-11.ipynb#cell-35
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.4: else و finally: إكمال الصورة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-013-number.py
"""

# أقل وضوحاً - أي سطر بالضبط قد يُطلق الخطأ؟
try:
    number = int(text)
    print(f"ضعفه: {number * 2}")  # ماذا لو هذا السطر نفسه فشل لسبب آخر؟
except ValueError:
    print("خطأ")

# أوضح بكثير - try تحتوي فقط على الكود المحتمل فشله
try:
    number = int(text)
except ValueError:
    print("خطأ")
else:
    print(f"ضعفه: {number * 2}")  # نعلم بالتأكيد أن number تم تحويله بنجاح

# ────────────────────────────────────────────────────────────────────────
# [037] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-11-chapter-11.ipynb#cell-36
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.4: else و finally: إكمال الصورة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-014-process_file.py
"""

def process_file(filename: str) -> None:
    file = None
    try:
        file = open(filename, "r", encoding="utf-8")
        content = file.read()
        print(f"تم قراءة {len(content)} حرف.")
    except FileNotFoundError:
        print(f"❌ الملف '{filename}' غير موجود.")
    finally:
        if file is not None:
            file.close()
            print("✅ تم إغلاق الملف.")  # يحدث دائماً!

process_file("not_exist.txt")

# ────────────────────────────────────────────────────────────────────────
# [038] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-11-chapter-11.ipynb#cell-37
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.5: إطلاق الأخطاء بنفسك: raise

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-015-withdraw.py
"""

def withdraw(balance: float, amount: float) -> float:
    if amount <= 0:
        raise ValueError("مبلغ السحب يجب أن يكون موجباً.")
    if amount > balance:
        raise ValueError("الرصيد غير كافٍ لإتمام العملية.")
    return balance - amount

try:
    new_balance = withdraw(1000, 1500)
except ValueError as e:
    print(f"❌ فشلت العملية: {e}")

# ────────────────────────────────────────────────────────────────────────
# [039] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-11-chapter-11.ipynb#cell-38
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.5: إطلاق الأخطاء بنفسك: raise

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-016-process_data.py
"""

def process_data(data: list) -> float:
    try:
        return sum(data) / len(data)
    except ZeroDivisionError:
        print("⚠️ تحذير: القائمة فارغة، سيتم إعادة إطلاق الخطأ.")
        raise  # raise بدون معامل = إعادة إطلاق نفس الخطأ الملتقط

try:
    process_data([])
except ZeroDivisionError:
    print("❌ المستوى الأعلى: تم التعامل مع القائمة الفارغة.")

# ────────────────────────────────────────────────────────────────────────
# [040] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-11-chapter-11.ipynb#cell-39
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.5: إطلاق الأخطاء بنفسك: raise

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-017-parse_age.py
"""

def parse_age(text: str) -> int:
    try:
        return int(text)
    except ValueError as e:
        raise ValueError(f"فشل تحليل العمر من النص: '{text}'") from e

try:
    parse_age("twenty")
except ValueError as e:
    print(e)
    print(f"السبب الأصلي: {e.__cause__}")

# ────────────────────────────────────────────────────────────────────────
# [041] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-11-chapter-11.ipynb#cell-40
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.6: إنشاء استثناءات مخصصة (Custom Exceptions)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-018-withdraw.py
"""

class InsufficientFundsError(Exception):
    """تُطلق عندما يحاول المستخدم سحب مبلغ أكبر من رصيده."""
    pass

class InvalidAmountError(Exception):
    """تُطلق عندما يكون المبلغ المُدخل غير منطقي (سالب أو صفر)."""
    pass

def withdraw(balance: float, amount: float) -> float:
    if amount <= 0:
        raise InvalidAmountError(f"المبلغ {amount} غير صالح؛ يجب أن يكون موجباً.")
    if amount > balance:
        raise InsufficientFundsError(
            f"طلبت سحب {amount}، لكن رصيدك الحالي {balance} فقط."
        )
    return balance - amount

try:
    withdraw(500, 1000)
except InsufficientFundsError as e:
    print(f"💰 خطأ في الرصيد: {e}")
except InvalidAmountError as e:
    print(f"🔢 خطأ في المبلغ: {e}")

# ────────────────────────────────────────────────────────────────────────
# [042] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-11-chapter-11.ipynb#cell-41
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.6: إنشاء استثناءات مخصصة (Custom Exceptions)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-019-process_transaction.py
"""

class BankError(Exception):
    """الفئة الأساسية لكل أخطاء النظام البنكي."""
    pass

class InsufficientFundsError(BankError):
    """رصيد غير كافٍ."""
    pass

class InvalidAmountError(BankError):
    """مبلغ غير صالح."""
    pass

class AccountFrozenError(BankError):
    """الحساب مجمّد."""
    pass

def process_transaction(account_status: str, balance: float, amount: float) -> float:
    if account_status == "frozen":
        raise AccountFrozenError("هذا الحساب مجمّد حالياً.")
    if amount <= 0:
        raise InvalidAmountError("المبلغ يجب أن يكون موجباً.")
    if amount > balance:
        raise InsufficientFundsError("الرصيد غير كافٍ.")
    return balance - amount

try:
    process_transaction("frozen", 1000, 100)
except BankError as e:  # التقاط أي خطأ بنكي بغض النظر عن نوعه الدقيق
    print(f"🏦 خطأ في النظام البنكي: {type(e).__name__}: {e}")

# ────────────────────────────────────────────────────────────────────────
# [043] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-11-chapter-11.ipynb#cell-42
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.6: إنشاء استثناءات مخصصة (Custom Exceptions)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-020-__init__.py
"""

class ValidationError(Exception):
    """خطأ تحقق من صحة البيانات، مع تفاصيل الحقل المُخالف."""

    def __init__(self, field: str, message: str):
        self.field = field
        self.message = message
        super().__init__(f"خطأ في حقل '{field}': {message}")

def validate_age(age: int) -> None:
    if age < 0 or age > 150:
        raise ValidationError(field="age", message="يجب أن يكون بين 0 و150")

try:
    validate_age(-5)
except ValidationError as e:
    print(f"الحقل المُخالف: {e.field}")
    print(f"الرسالة: {e.message}")

# ────────────────────────────────────────────────────────────────────────
# [044] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-11-chapter-11.ipynb#cell-43
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.7: فلسفتان في البرمجة: EAFP مقابل LBYL

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-021-content.py
"""

# تحقق أولاً، ثم نفّذ
import os

if os.path.exists("data.txt"):
    with open("data.txt") as f:
        content = f.read()
else:
    print("الملف غير موجود")

# ────────────────────────────────────────────────────────────────────────
# [045] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-11-chapter-11.ipynb#cell-44
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.7: فلسفتان في البرمجة: EAFP مقابل LBYL

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-022-content.py
"""

# نفّذ مباشرة، وتعامل مع الخطأ إن حدث
try:
    with open("data.txt") as f:
        content = f.read()
except FileNotFoundError:
    print("الملف غير موجود")

# ────────────────────────────────────────────────────────────────────────
# [046] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-11-chapter-11.ipynb#cell-45
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.8: دالة assert: التحقق أثناء التطوير

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-023-calculate_average.py
"""

def calculate_average(numbers: list[float]) -> float:
    assert len(numbers) > 0, "القائمة يجب ألا تكون فارغة"
    return sum(numbers) / len(numbers)

print(calculate_average([10, 20, 30]))  # 20.0
# print(calculate_average([]))
# 💥 AssertionError: القائمة يجب ألا تكون فارغة

# ────────────────────────────────────────────────────────────────────────
# [047] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-11-chapter-11.ipynb#cell-46
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.8: دالة assert: التحقق أثناء التطوير

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-024-set_age.py
"""

# ❌ خطأ: استخدام assert للتحقق من مدخلات مستخدم
def set_age(age: int) -> None:
    assert age >= 0, "العمر يجب أن يكون موجباً"  # قد يُحذف بالكامل!

# ✅ صحيح: استخدام raise للتحقق من مدخلات مستخدم
def set_age(age: int) -> None:
    if age < 0:
        raise ValueError("العمر يجب أن يكون موجباً")

# ────────────────────────────────────────────────────────────────────────
# [048] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-11-chapter-11.ipynb#cell-47
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.9: مقدمة مختصرة لـ logging بدلاً من print

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-025-divide.py
"""

import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def divide(a: float, b: float) -> float | None:
    try:
        return a / b
    except ZeroDivisionError:
        logging.error(f"محاولة قسمة {a} على صفر!")
        return None

divide(10, 0)

# ────────────────────────────────────────────────────────────────────────
# [049] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-11-chapter-11.ipynb#cell-48
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.10: مشروع تطبيقي: نظام تسجيل دخول آمن

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-026-register.py
"""

class AuthError(Exception):
    """الفئة الأساسية لكل أخطاء المصادقة."""
    pass

class UserNotFoundError(AuthError):
    """المستخدم غير موجود في النظام."""
    pass

class IncorrectPasswordError(AuthError):
    """كلمة المرور غير صحيحة."""
    pass

class AccountLockedError(AuthError):
    """الحساب مقفل بسبب محاولات فاشلة متكررة."""
    pass

class WeakPasswordError(AuthError):
    """كلمة مرور ضعيفة عند التسجيل."""
    pass

# قاعدة بيانات وهمية (في الواقع ستكون قاعدة بيانات حقيقية)
users_db: dict[str, dict] = {
    "ahmed@test.com": {"password": "Secure123", "failed_attempts": 0, "locked": False},
}

MAX_FAILED_ATTEMPTS = 3

def register(email: str, password: str) -> None:
    """تسجيل مستخدم جديد مع التحقق من قوة كلمة المرور."""
    if len(password) < 8:
        raise WeakPasswordError("كلمة المرور يجب أن تكون 8 أحرف على الأقل.")
    if not any(char.isdigit() for char in password):
        raise WeakPasswordError("كلمة المرور يجب أن تحتوي رقماً واحداً على الأقل.")

    if email in users_db:
        raise ValueError("هذا البريد الإلكتروني مسجّل مسبقاً.")

    users_db[email] = {"password": password, "failed_attempts": 0, "locked": False}
    print(f"✅ تم تسجيل المستخدم {email} بنجاح.")

def login(email: str, password: str) -> bool:
    """محاولة تسجيل دخول، مع حماية من المحاولات المتكررة."""
    try:
        user = users_db[email]
    except KeyError:
        raise UserNotFoundError(f"لا يوجد مستخدم بالبريد: {email}")

    if user["locked"]:
        raise AccountLockedError("هذا الحساب مقفل. تواصل مع الدعم الفني.")

    if user["password"] != password:
        user["failed_attempts"] += 1
        if user["failed_attempts"] >= MAX_FAILED_ATTEMPTS:
            user["locked"] = True
            raise AccountLockedError(
                "تم قفل الحساب بسبب تجاوز عدد المحاولات الفاشلة المسموحة."
            )
        remaining = MAX_FAILED_ATTEMPTS - user["failed_attempts"]
        raise IncorrectPasswordError(
            f"كلمة المرور خاطئة. المحاولات المتبقية: {remaining}"
        )

    user["failed_attempts"] = 0  # إعادة تصفير عند النجاح
    return True

def run_login_attempt(email: str, password: str) -> None:
    """واجهة تفاعلية تلتقط كل أنواع أخطاء المصادقة بوضوح."""
    try:
        login(email, password)
    except UserNotFoundError as e:
        print(f"👤 {e}")
    except AccountLockedError as e:
        print(f"🔒 {e}")
    except IncorrectPasswordError as e:
        print(f"🔑 {e}")
    except AuthError as e:
        print(f"⚠️ خطأ مصادقة غير متوقع: {e}")
    else:
        print(f"✅ تم تسجيل الدخول بنجاح! أهلاً بك {email}")
    finally:
        print("-" * 40)

# --- سيناريوهات تجريبية ---
if __name__ == "__main__":
    try:
        register("sara@test.com", "weak")
    except WeakPasswordError as e:
        print(f"❌ فشل التسجيل: {e}")
    print("-" * 40)

    register("sara@test.com", "Strong123")
    print("-" * 40)

    run_login_attempt("unknown@test.com", "anything")   # UserNotFoundError
    run_login_attempt("ahmed@test.com", "wrong_pass")   # IncorrectPasswordError (محاولة 1)
    run_login_attempt("ahmed@test.com", "wrong_pass")   # IncorrectPasswordError (محاولة 2)
    run_login_attempt("ahmed@test.com", "wrong_pass")   # AccountLockedError (محاولة 3 - قفل!)
    run_login_attempt("ahmed@test.com", "Secure123")    # AccountLockedError (مقفل مسبقاً)
    run_login_attempt("sara@test.com", "Strong123")     # ✅ نجاح

# ────────────────────────────────────────────────────────────────────────
# [050] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-11-chapter-11.ipynb#cell-49
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.10: مشروع تطبيقي: نظام تسجيل دخول آمن

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-027-reset_password.py
"""

def reset_password(email: str, new_password: str) -> None:
    """إعادة تعيين كلمة مرور، مع إلغاء القفل إن وُجد."""
    try:
        user = users_db[email]
    except KeyError:
        raise UserNotFoundError(f"لا يوجد مستخدم بالبريد: {email}")

    if len(new_password) < 8 or not any(c.isdigit() for c in new_password):
        raise WeakPasswordError("كلمة المرور الجديدة ضعيفة جداً.")

    user["password"] = new_password
    user["failed_attempts"] = 0
    user["locked"] = False
    print(f"🔓 تم إعادة تعيين كلمة المرور وفك قفل الحساب لـ {email}")

try:
    reset_password("ahmed@test.com", "NewPass123")
except AuthError as e:
    print(f"❌ {e}")

# ────────────────────────────────────────────────────────────────────────
# [051] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-11-chapter-11.ipynb#cell-50
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.11: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-028-safe_list_access.py
"""

# تمرين 1
try:
    num = int(input("أدخل رقماً: "))
    print("رقم صحيح")
except ValueError:
    print("خطأ: ليس رقماً")

# تمرين 2
def safe_list_access(lst: list, index: int):
    try:
        return lst[index]
    except IndexError:
        return None

print(safe_list_access([1, 2, 3], 1))   # 2
print(safe_list_access([1, 2, 3], 10))  # None

# ────────────────────────────────────────────────────────────────────────
# [052] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-11-chapter-11.ipynb#cell-51
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.11: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-029-calculate_square_root.py
"""

import math

# تمرين 3
class NegativeValueError(Exception):
    pass

def calculate_square_root(n: float) -> float:
    if n < 0:
        raise NegativeValueError(f"لا يمكن حساب جذر رقم سالب: {n}")
    return math.sqrt(n)

try:
    print(calculate_square_root(-9))
except NegativeValueError as e:
    print(f"❌ {e}")

# تمرين 4
class MissingConfigError(Exception):
    def __init__(self, key: str):
        self.key = key
        super().__init__(f"الإعداد المطلوب '{key}' غير موجود.")

def parse_config(data: dict) -> tuple[str, int]:
    try:
        host = data["host"]
    except KeyError:
        raise MissingConfigError("host")

    try:
        port = data["port"]
    except KeyError:
        raise MissingConfigError("port")

    return host, port

try:
    parse_config({"host": "localhost"})
except MissingConfigError as e:
    print(f"❌ {e}")  # ❌ الإعداد المطلوب 'port' غير موجود.

# ────────────────────────────────────────────────────────────────────────
# [053] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-11-chapter-11.ipynb#cell-52
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.11: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-030-__init__.py
"""

class APIError(Exception):
    """الفئة الأساسية لكل أخطاء API."""
    pass

class EndpointNotFoundError(APIError):
    def __init__(self, endpoint: str):
        self.endpoint = endpoint
        super().__init__(f"المسار '{endpoint}' غير موجود.")

class MissingFieldError(APIError):
    def __init__(self, field: str):
        self.field = field
        super().__init__(f"الحقل المطلوب '{field}' مفقود.")

class InvalidFieldError(APIError):
    def __init__(self, field: str, reason: str):
        self.field = field
        self.reason = reason
        super().__init__(f"الحقل '{field}' غير صالح: {reason}")

VALID_ENDPOINTS = {"/users", "/orders", "/products"}

def process_api_request(endpoint: str, data: dict) -> dict:
    if endpoint not in VALID_ENDPOINTS:
        raise EndpointNotFoundError(endpoint)

    if "user_id" not in data:
        raise MissingFieldError("user_id")

    user_id = data["user_id"]
    if not isinstance(user_id, int) or user_id <= 0:
        raise InvalidFieldError("user_id", "يجب أن يكون رقماً صحيحاً موجباً")

    return {"status": "success", "endpoint": endpoint, "user_id": user_id}

def handle_request(endpoint: str, data: dict) -> None:
    try:
        result = process_api_request(endpoint, data)
    except EndpointNotFoundError as e:
        print(f"404 - {e}")
    except MissingFieldError as e:
        print(f"400 - حقل مفقود: {e}")
    except InvalidFieldError as e:
        print(f"400 - حقل غير صالح: {e}")
    except APIError as e:
        print(f"500 - خطأ API غير متوقع: {e}")
    else:
        print(f"200 - {result}")

handle_request("/comments", {"user_id": 1})         # 404
handle_request("/users", {})                         # 400 مفقود
handle_request("/users", {"user_id": -5})            # 400 غير صالح
handle_request("/users", {"user_id": 42})            # 200 نجاح

# ────────────────────────────────────────────────────────────────────────
# [054] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-11-chapter-11.ipynb#cell-53
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.13: ملخص الفصل (Cheat Sheet)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-031-__init__.py
"""

# الصيغة الكاملة
try:
    risky_code()
except SpecificError as e:
    handle_specific_case(e)
except (ErrorA, ErrorB) as e:
    handle_multiple_cases(e)
except Exception as e:      # الأعم - دائماً في الأخير
    handle_general_case(e)
else:
    runs_if_no_error()
finally:
    always_runs()            # تنظيف الموارد

# إطلاق خطأ
raise ValueError("رسالة واضحة")

# إعادة إطلاق
except SomeError:
    raise  # بدون معامل

# تسلسل الاستثناءات
raise NewError("...") from original_error

# استثناء مخصص
class MyAppError(Exception):
    pass

class SpecificError(MyAppError):
    def __init__(self, detail: str):
        self.detail = detail
        super().__init__(f"فشل بسبب: {detail}")

# assert - للتطوير فقط، ليس لمدخلات المستخدم
assert condition, "رسالة إذا فشل الشرط"

# ────────────────────────────────────────────────────────────────────────
# [055] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-11-chapter-11.ipynb#cell-54
# ────────────────────────────────────────────────────────────────────────
balance = 1000
withdrawal = input("أدخل مبلغ السحب: ")  # المستخدم يكتب "مئة"
balance -= int(withdrawal)  # 💥 ValueError: invalid literal for int()
# البرنامج ينهار، والمستخدم يرى رسالة تقنية غير مفهومة

# ────────────────────────────────────────────────────────────────────────
# [056] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-11-chapter-11.ipynb#cell-55
# ────────────────────────────────────────────────────────────────────────
print(10 / 0)        # ZeroDivisionError
print(int("abc"))    # ValueError
print(my_list[99])   # IndexError
print(my_dict["x"])  # KeyError
print(undefined_var) # NameError

# ────────────────────────────────────────────────────────────────────────
# [057] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-11-chapter-11.ipynb#cell-56
# ────────────────────────────────────────────────────────────────────────
def divide(a, b):
    return a / b

result = divide(10, 0)

# ────────────────────────────────────────────────────────────────────────
# [058] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-11-chapter-11.ipynb#cell-57
# ────────────────────────────────────────────────────────────────────────
try:
    number = int(input("أدخل رقماً: "))
    print(f"ضعف الرقم: {number * 2}")
except ValueError:
    print("❌ خطأ: الرجاء إدخال رقم صحيح فقط.")

print("البرنامج استمر في العمل!")

# ────────────────────────────────────────────────────────────────────────
# [059] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-11-chapter-11.ipynb#cell-58
# ────────────────────────────────────────────────────────────────────────
def safe_divide(a: float, b: float) -> float | None:
    try:
        return a / b
    except ZeroDivisionError:
        print("❌ لا يمكن القسمة على صفر.")
        return None
    except TypeError:
        print("❌ يجب أن تكون القيم أرقاماً.")
        return None

print(safe_divide(10, 0))        # ❌ لا يمكن القسمة على صفر. → None
print(safe_divide(10, "abc"))    # ❌ يجب أن تكون القيم أرقاماً. → None
print(safe_divide(10, 2))        # 5.0

# ────────────────────────────────────────────────────────────────────────
# [060] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-11-chapter-11.ipynb#cell-59
# ────────────────────────────────────────────────────────────────────────
try:
    value = my_dict[key]
    result = 100 / value
except (KeyError, ZeroDivisionError) as e:
    print(f"حدث خطأ: {e}")

# ────────────────────────────────────────────────────────────────────────
# [061] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-11-chapter-11.ipynb#cell-60
# ────────────────────────────────────────────────────────────────────────
try:
    risky_operation()
except Exception as e:
    print(f"حدث خطأ غير متوقع: {e}")

# ────────────────────────────────────────────────────────────────────────
# [062] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-11-chapter-11.ipynb#cell-61
# ────────────────────────────────────────────────────────────────────────
# ❌ الأسوأ على الإطلاق - لا تفعل هذا أبداً!
try:
    risky_operation()
except:  # bare except - يلتقط كل شيء حتى KeyboardInterrupt!
    pass  # يُخفي الخطأ تماماً دون أي أثر!

# ────────────────────────────────────────────────────────────────────────
# [063] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-11-chapter-11.ipynb#cell-62
# ────────────────────────────────────────────────────────────────────────
# ترتيب except مهم جداً! الأكثر تحديداً أولاً
try:
    value = my_list[10]
except IndexError:
    print("فهرس خارج النطاق")
except LookupError:  # تأتي بعد IndexError لأنها أعم منها
    print("خطأ بحث عام")

# ────────────────────────────────────────────────────────────────────────
# [064] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-11-chapter-11.ipynb#cell-63
# ────────────────────────────────────────────────────────────────────────
try:
    # الكود الذي قد يُطلق خطأ
    pass
except SomeError:
    # يُنفَّذ فقط إذا حدث هذا الخطأ المحدد
    pass
else:
    # يُنفَّذ فقط إذا لم يحدث أي خطأ في try
    pass
finally:
    # يُنفَّذ دائماً، سواء حدث خطأ أم لا
    pass

# ────────────────────────────────────────────────────────────────────────
# [065] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-11-chapter-11.ipynb#cell-64
# ────────────────────────────────────────────────────────────────────────
# أقل وضوحاً - أي سطر بالضبط قد يُطلق الخطأ؟
try:
    number = int(text)
    print(f"ضعفه: {number * 2}")  # ماذا لو هذا السطر نفسه فشل لسبب آخر؟
except ValueError:
    print("خطأ")

# أوضح بكثير - try تحتوي فقط على الكود المحتمل فشله
try:
    number = int(text)
except ValueError:
    print("خطأ")
else:
    print(f"ضعفه: {number * 2}")  # نعلم بالتأكيد أن number تم تحويله بنجاح

# ────────────────────────────────────────────────────────────────────────
# [066] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-11-chapter-11.ipynb#cell-65
# ────────────────────────────────────────────────────────────────────────
def calculate_average(numbers: list[float]) -> float:
    assert len(numbers) > 0, "القائمة يجب ألا تكون فارغة"
    return sum(numbers) / len(numbers)

print(calculate_average([10, 20, 30]))  # 20.0
# print(calculate_average([]))
# 💥 AssertionError: القائمة يجب ألا تكون فارغة

# ────────────────────────────────────────────────────────────────────────
# [067] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-11-chapter-11.ipynb#cell-66
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: استخدام assert للتحقق من مدخلات مستخدم
def set_age(age: int) -> None:
    assert age >= 0, "العمر يجب أن يكون موجباً"  # قد يُحذف بالكامل!

# ✅ صحيح: استخدام raise للتحقق من مدخلات مستخدم
def set_age(age: int) -> None:
    if age < 0:
        raise ValueError("العمر يجب أن يكون موجباً")

# ────────────────────────────────────────────────────────────────────────
# [068] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-11-chapter-11.ipynb#cell-67
# ────────────────────────────────────────────────────────────────────────
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def divide(a: float, b: float) -> float | None:
    try:
        return a / b
    except ZeroDivisionError:
        logging.error(f"محاولة قسمة {a} على صفر!")
        return None

divide(10, 0)

# ────────────────────────────────────────────────────────────────────────
# [069] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-11-chapter-11.ipynb#cell-68
# ────────────────────────────────────────────────────────────────────────
# تمرين 1
try:
    num = int(input("أدخل رقماً: "))
    print("رقم صحيح")
except ValueError:
    print("خطأ: ليس رقماً")

# تمرين 2
def safe_list_access(lst: list, index: int):
    try:
        return lst[index]
    except IndexError:
        return None

print(safe_list_access([1, 2, 3], 1))   # 2
print(safe_list_access([1, 2, 3], 10))  # None

# ────────────────────────────────────────────────────────────────────────
# [070] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-11-chapter-11.ipynb#cell-69
# ────────────────────────────────────────────────────────────────────────
# الصيغة الكاملة
try:
    risky_code()
except SpecificError as e:
    handle_specific_case(e)
except (ErrorA, ErrorB) as e:
    handle_multiple_cases(e)
except Exception as e:      # الأعم - دائماً في الأخير
    handle_general_case(e)
else:
    runs_if_no_error()
finally:
    always_runs()            # تنظيف الموارد

# إطلاق خطأ
raise ValueError("رسالة واضحة")

# إعادة إطلاق
except SomeError:
    raise  # بدون معامل

# تسلسل الاستثناءات
raise NewError("...") from original_error

# استثناء مخصص
class MyAppError(Exception):
    pass

class SpecificError(MyAppError):
    def __init__(self, detail: str):
        self.detail = detail
        super().__init__(f"فشل بسبب: {detail}")

# assert - للتطوير فقط، ليس لمدخلات المستخدم
assert condition, "رسالة إذا فشل الشرط"

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-11-chapter-11.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
بايثون 3.14 — من الصفر إلى الاحتراف — معالجة الأخطاء (Error Handling)
Python 3.14 — From Zero to Pro — Error

Book slug : python-3-14-from-zero-to-pro
Chapter   : ch-11-chapter-11 (lesson 11)
Code files: 69 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] dist/notebooks/chapter-11.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#@title ⚙️ تهيئة البيئة  { display-mode: "form" }
import sys, os, random, pathlib

print("Python:", sys.version.split()[0], "|", sys.platform)
assert sys.version_info >= (3, 10), "هذا الكتاب يفترض Python 3.10 أو أحدث"

SEED = 42
random.seed(SEED)

# ────────────────────────────────────────────────────────────────────────
# [002] dist/notebooks/chapter-11.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
# ⚠️ مثال يُظهر خطأً مقصودًا: نلتقطه هنا ليكتمل «تشغيل الكل»،
# وسترى نوع الخطأ ورسالته في المخرجات كما في الطرفية.
try:
    def divide(a, b):
        return a / b

    result = divide(10, 0)
except (Exception, SystemExit) as _err:
    print(f"⚠️ {type(_err).__name__}: {_err}")

# ────────────────────────────────────────────────────────────────────────
# [003] dist/notebooks/chapter-11.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# 📝 نسخة الدفتر: حقل نموذج بدل الإدخال اليدوي — عدّل القيمة ثم أعد التشغيل
try:
    number = 10  #@param {type:"integer"}
    print(f"ضعف الرقم: {number * 2}")
except ValueError:
    print("❌ خطأ: الرجاء إدخال رقم صحيح فقط.")

print("البرنامج استمر في العمل!")

# ────────────────────────────────────────────────────────────────────────
# [004] dist/notebooks/chapter-11.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# ⚠️ مثال يُظهر خطأً مقصودًا: نلتقطه هنا ليكتمل «تشغيل الكل»،
# وسترى نوع الخطأ ورسالته في المخرجات كما في الطرفية.
try:
    def safe_divide(a: float, b: float) -> float | None:
        try:
            return a / b
        except ZeroDivisionError:
            print("❌ لا يمكن القسمة على صفر.")
            return None
        except TypeError:
            print("❌ يجب أن تكون القيم أرقاماً.")
            return None

    print(safe_divide(10, 0))        # ❌ لا يمكن القسمة على صفر. → None
    print(safe_divide(10, "abc"))    # ❌ يجب أن تكون القيم أرقاماً. → None
    print(safe_divide(10, 2))        # 5.0
except (Exception, SystemExit) as _err:
    print(f"⚠️ {type(_err).__name__}: {_err}")

# ────────────────────────────────────────────────────────────────────────
# [005] dist/notebooks/chapter-11.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
try:
    x = 1 / 0
except Exception:
    print("خطأ عام")  # هذا ما سيُطبع
except ZeroDivisionError:  # ⚠️ ميت! لن يُنفَّذ أبداً
    print("قسمة على صفر")

# ────────────────────────────────────────────────────────────────────────
# [006] dist/notebooks/chapter-11.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
def read_and_process_number(text: str) -> None:
    try:
        number = int(text)
    except ValueError:
        print(f"❌ '{text}' ليس رقماً صحيحاً.")
    else:
        # يُنفَّذ فقط إذا نجح التحويل بدون خطأ
        print(f"✅ تم تحويل الرقم بنجاح: {number}")
        print(f"   ضعفه: {number * 2}")
    finally:
        # يُنفَّذ دائماً - مثالي للتنظيف
        print("--- انتهت المحاولة ---")

read_and_process_number("42")
print()
read_and_process_number("abc")

# ────────────────────────────────────────────────────────────────────────
# [007] dist/notebooks/chapter-11.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
def process_file(filename: str) -> None:
    file = None
    try:
        file = open(filename, "r", encoding="utf-8")
        content = file.read()
        print(f"تم قراءة {len(content)} حرف.")
    except FileNotFoundError:
        print(f"❌ الملف '{filename}' غير موجود.")
    finally:
        if file is not None:
            file.close()
            print("✅ تم إغلاق الملف.")  # يحدث دائماً!

process_file("not_exist.txt")

# ────────────────────────────────────────────────────────────────────────
# [008] dist/notebooks/chapter-11.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
def withdraw(balance: float, amount: float) -> float:
    if amount <= 0:
        raise ValueError("مبلغ السحب يجب أن يكون موجباً.")
    if amount > balance:
        raise ValueError("الرصيد غير كافٍ لإتمام العملية.")
    return balance - amount

try:
    new_balance = withdraw(1000, 1500)
except ValueError as e:
    print(f"❌ فشلت العملية: {e}")

# ────────────────────────────────────────────────────────────────────────
# [009] dist/notebooks/chapter-11.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
def process_data(data: list) -> float:
    try:
        return sum(data) / len(data)
    except ZeroDivisionError:
        print("⚠️ تحذير: القائمة فارغة، سيتم إعادة إطلاق الخطأ.")
        raise  # raise بدون معامل = إعادة إطلاق نفس الخطأ الملتقط

try:
    process_data([])
except ZeroDivisionError:
    print("❌ المستوى الأعلى: تم التعامل مع القائمة الفارغة.")

# ────────────────────────────────────────────────────────────────────────
# [010] dist/notebooks/chapter-11.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
def parse_age(text: str) -> int:
    try:
        return int(text)
    except ValueError as e:
        raise ValueError(f"فشل تحليل العمر من النص: '{text}'") from e

try:
    parse_age("twenty")
except ValueError as e:
    print(e)
    print(f"السبب الأصلي: {e.__cause__}")

# ────────────────────────────────────────────────────────────────────────
# [011] dist/notebooks/chapter-11.ipynb#cell-11
# ────────────────────────────────────────────────────────────────────────
class InsufficientFundsError(Exception):
    """تُطلق عندما يحاول المستخدم سحب مبلغ أكبر من رصيده."""
    pass

class InvalidAmountError(Exception):
    """تُطلق عندما يكون المبلغ المُدخل غير منطقي (سالب أو صفر)."""
    pass

def withdraw(balance: float, amount: float) -> float:
    if amount <= 0:
        raise InvalidAmountError(f"المبلغ {amount} غير صالح؛ يجب أن يكون موجباً.")
    if amount > balance:
        raise InsufficientFundsError(
            f"طلبت سحب {amount}، لكن رصيدك الحالي {balance} فقط."
        )
    return balance - amount

try:
    withdraw(500, 1000)
except InsufficientFundsError as e:
    print(f"💰 خطأ في الرصيد: {e}")
except InvalidAmountError as e:
    print(f"🔢 خطأ في المبلغ: {e}")

# ────────────────────────────────────────────────────────────────────────
# [012] dist/notebooks/chapter-11.ipynb#cell-12
# ────────────────────────────────────────────────────────────────────────
class BankError(Exception):
    """الفئة الأساسية لكل أخطاء النظام البنكي."""
    pass

class InsufficientFundsError(BankError):
    """رصيد غير كافٍ."""
    pass

class InvalidAmountError(BankError):
    """مبلغ غير صالح."""
    pass

class AccountFrozenError(BankError):
    """الحساب مجمّد."""
    pass

def process_transaction(account_status: str, balance: float, amount: float) -> float:
    if account_status == "frozen":
        raise AccountFrozenError("هذا الحساب مجمّد حالياً.")
    if amount <= 0:
        raise InvalidAmountError("المبلغ يجب أن يكون موجباً.")
    if amount > balance:
        raise InsufficientFundsError("الرصيد غير كافٍ.")
    return balance - amount

try:
    process_transaction("frozen", 1000, 100)
except BankError as e:  # التقاط أي خطأ بنكي بغض النظر عن نوعه الدقيق
    print(f"🏦 خطأ في النظام البنكي: {type(e).__name__}: {e}")

# ────────────────────────────────────────────────────────────────────────
# [013] dist/notebooks/chapter-11.ipynb#cell-13
# ────────────────────────────────────────────────────────────────────────
class ValidationError(Exception):
    """خطأ تحقق من صحة البيانات، مع تفاصيل الحقل المُخالف."""

    def __init__(self, field: str, message: str):
        self.field = field
        self.message = message
        super().__init__(f"خطأ في حقل '{field}': {message}")

def validate_age(age: int) -> None:
    if age < 0 or age > 150:
        raise ValidationError(field="age", message="يجب أن يكون بين 0 و150")

try:
    validate_age(-5)
except ValidationError as e:
    print(f"الحقل المُخالف: {e.field}")
    print(f"الرسالة: {e.message}")

# ────────────────────────────────────────────────────────────────────────
# [014] dist/notebooks/chapter-11.ipynb#cell-14
# ────────────────────────────────────────────────────────────────────────
# تحقق أولاً، ثم نفّذ
import os

if os.path.exists("data.txt"):
    with open("data.txt") as f:
        content = f.read()
else:
    print("الملف غير موجود")

# ────────────────────────────────────────────────────────────────────────
# [015] dist/notebooks/chapter-11.ipynb#cell-15
# ────────────────────────────────────────────────────────────────────────
# نفّذ مباشرة، وتعامل مع الخطأ إن حدث
try:
    with open("data.txt") as f:
        content = f.read()
except FileNotFoundError:
    print("الملف غير موجود")

# ────────────────────────────────────────────────────────────────────────
# [016] dist/notebooks/chapter-11.ipynb#cell-16
# ────────────────────────────────────────────────────────────────────────
# ⚠️ مثال يُظهر خطأً مقصودًا: نلتقطه هنا ليكتمل «تشغيل الكل»،
# وسترى نوع الخطأ ورسالته في المخرجات كما في الطرفية.
try:
    def calculate_average(numbers: list[float]) -> float:
        assert len(numbers) > 0, "القائمة يجب ألا تكون فارغة"
        return sum(numbers) / len(numbers)

    print(calculate_average([10, 20, 30]))  # 20.0
    # print(calculate_average([]))
    # 💥 AssertionError: القائمة يجب ألا تكون فارغة
except (Exception, SystemExit) as _err:
    print(f"⚠️ {type(_err).__name__}: {_err}")

# ────────────────────────────────────────────────────────────────────────
# [017] dist/notebooks/chapter-11.ipynb#cell-17
# ────────────────────────────────────────────────────────────────────────
# ⚠️ مثال يُظهر خطأً مقصودًا: نلتقطه هنا ليكتمل «تشغيل الكل»،
# وسترى نوع الخطأ ورسالته في المخرجات كما في الطرفية.
try:
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    def divide(a: float, b: float) -> float | None:
        try:
            return a / b
        except ZeroDivisionError:
            logging.error(f"محاولة قسمة {a} على صفر!")
            return None

    divide(10, 0)
except (Exception, SystemExit) as _err:
    print(f"⚠️ {type(_err).__name__}: {_err}")

# ────────────────────────────────────────────────────────────────────────
# [018] dist/notebooks/chapter-11.ipynb#cell-18
# ────────────────────────────────────────────────────────────────────────
class AuthError(Exception):
    """الفئة الأساسية لكل أخطاء المصادقة."""
    pass

class UserNotFoundError(AuthError):
    """المستخدم غير موجود في النظام."""
    pass

class IncorrectPasswordError(AuthError):
    """كلمة المرور غير صحيحة."""
    pass

class AccountLockedError(AuthError):
    """الحساب مقفل بسبب محاولات فاشلة متكررة."""
    pass

class WeakPasswordError(AuthError):
    """كلمة مرور ضعيفة عند التسجيل."""
    pass

# قاعدة بيانات وهمية (في الواقع ستكون قاعدة بيانات حقيقية)
users_db: dict[str, dict] = {
    "ahmed@test.com": {"password": "Secure123", "failed_attempts": 0, "locked": False},
}

MAX_FAILED_ATTEMPTS = 3

def register(email: str, password: str) -> None:
    """تسجيل مستخدم جديد مع التحقق من قوة كلمة المرور."""
    if len(password) < 8:
        raise WeakPasswordError("كلمة المرور يجب أن تكون 8 أحرف على الأقل.")
    if not any(char.isdigit() for char in password):
        raise WeakPasswordError("كلمة المرور يجب أن تحتوي رقماً واحداً على الأقل.")

    if email in users_db:
        raise ValueError("هذا البريد الإلكتروني مسجّل مسبقاً.")

    users_db[email] = {"password": password, "failed_attempts": 0, "locked": False}
    print(f"✅ تم تسجيل المستخدم {email} بنجاح.")

def login(email: str, password: str) -> bool:
    """محاولة تسجيل دخول، مع حماية من المحاولات المتكررة."""
    try:
        user = users_db[email]
    except KeyError:
        raise UserNotFoundError(f"لا يوجد مستخدم بالبريد: {email}")

    if user["locked"]:
        raise AccountLockedError("هذا الحساب مقفل. تواصل مع الدعم الفني.")

    if user["password"] != password:
        user["failed_attempts"] += 1
        if user["failed_attempts"] >= MAX_FAILED_ATTEMPTS:
            user["locked"] = True
            raise AccountLockedError(
                "تم قفل الحساب بسبب تجاوز عدد المحاولات الفاشلة المسموحة."
            )
        remaining = MAX_FAILED_ATTEMPTS - user["failed_attempts"]
        raise IncorrectPasswordError(
            f"كلمة المرور خاطئة. المحاولات المتبقية: {remaining}"
        )

    user["failed_attempts"] = 0  # إعادة تصفير عند النجاح
    return True

def run_login_attempt(email: str, password: str) -> None:
    """واجهة تفاعلية تلتقط كل أنواع أخطاء المصادقة بوضوح."""
    try:
        login(email, password)
    except UserNotFoundError as e:
        print(f"👤 {e}")
    except AccountLockedError as e:
        print(f"🔒 {e}")
    except IncorrectPasswordError as e:
        print(f"🔑 {e}")
    except AuthError as e:
        print(f"⚠️ خطأ مصادقة غير متوقع: {e}")
    else:
        print(f"✅ تم تسجيل الدخول بنجاح! أهلاً بك {email}")
    finally:
        print("-" * 40)

# --- سيناريوهات تجريبية ---
if __name__ == "__main__":
    try:
        register("sara@test.com", "weak")
    except WeakPasswordError as e:
        print(f"❌ فشل التسجيل: {e}")
    print("-" * 40)

    register("sara@test.com", "Strong123")
    print("-" * 40)

    run_login_attempt("unknown@test.com", "anything")   # UserNotFoundError
    run_login_attempt("ahmed@test.com", "wrong_pass")   # IncorrectPasswordError (محاولة 1)
    run_login_attempt("ahmed@test.com", "wrong_pass")   # IncorrectPasswordError (محاولة 2)
    run_login_attempt("ahmed@test.com", "wrong_pass")   # AccountLockedError (محاولة 3 - قفل!)
    run_login_attempt("ahmed@test.com", "Secure123")    # AccountLockedError (مقفل مسبقاً)
    run_login_attempt("sara@test.com", "Strong123")     # ✅ نجاح

# ────────────────────────────────────────────────────────────────────────
# [019] dist/notebooks/chapter-11.ipynb#cell-19
# ────────────────────────────────────────────────────────────────────────
def reset_password(email: str, new_password: str) -> None:
    """إعادة تعيين كلمة مرور، مع إلغاء القفل إن وُجد."""
    try:
        user = users_db[email]
    except KeyError:
        raise UserNotFoundError(f"لا يوجد مستخدم بالبريد: {email}")

    if len(new_password) < 8 or not any(c.isdigit() for c in new_password):
        raise WeakPasswordError("كلمة المرور الجديدة ضعيفة جداً.")

    user["password"] = new_password
    user["failed_attempts"] = 0
    user["locked"] = False
    print(f"🔓 تم إعادة تعيين كلمة المرور وفك قفل الحساب لـ {email}")

try:
    reset_password("ahmed@test.com", "NewPass123")
except AuthError as e:
    print(f"❌ {e}")

# ────────────────────────────────────────────────────────────────────────
# [020] dist/notebooks/chapter-11.ipynb#cell-20
# ────────────────────────────────────────────────────────────────────────
# 📝 نسخة الدفتر: حقل نموذج بدل الإدخال اليدوي — عدّل القيمة ثم أعد التشغيل
# تمرين 1
try:
    num = 10  #@param {type:"integer"}
    print("رقم صحيح")
except ValueError:
    print("خطأ: ليس رقماً")

# تمرين 2
def safe_list_access(lst: list, index: int):
    try:
        return lst[index]
    except IndexError:
        return None

print(safe_list_access([1, 2, 3], 1))   # 2
print(safe_list_access([1, 2, 3], 10))  # None

# ────────────────────────────────────────────────────────────────────────
# [021] dist/notebooks/chapter-11.ipynb#cell-21
# ────────────────────────────────────────────────────────────────────────
import math

# تمرين 3
class NegativeValueError(Exception):
    pass

def calculate_square_root(n: float) -> float:
    if n < 0:
        raise NegativeValueError(f"لا يمكن حساب جذر رقم سالب: {n}")
    return math.sqrt(n)

try:
    print(calculate_square_root(-9))
except NegativeValueError as e:
    print(f"❌ {e}")

# تمرين 4
class MissingConfigError(Exception):
    def __init__(self, key: str):
        self.key = key
        super().__init__(f"الإعداد المطلوب '{key}' غير موجود.")

def parse_config(data: dict) -> tuple[str, int]:
    try:
        host = data["host"]
    except KeyError:
        raise MissingConfigError("host")

    try:
        port = data["port"]
    except KeyError:
        raise MissingConfigError("port")

    return host, port

try:
    parse_config({"host": "localhost"})
except MissingConfigError as e:
    print(f"❌ {e}")  # ❌ الإعداد المطلوب 'port' غير موجود.

# ────────────────────────────────────────────────────────────────────────
# [022] dist/notebooks/chapter-11.ipynb#cell-22
# ────────────────────────────────────────────────────────────────────────
class APIError(Exception):
    """الفئة الأساسية لكل أخطاء API."""
    pass

class EndpointNotFoundError(APIError):
    def __init__(self, endpoint: str):
        self.endpoint = endpoint
        super().__init__(f"المسار '{endpoint}' غير موجود.")

class MissingFieldError(APIError):
    def __init__(self, field: str):
        self.field = field
        super().__init__(f"الحقل المطلوب '{field}' مفقود.")

class InvalidFieldError(APIError):
    def __init__(self, field: str, reason: str):
        self.field = field
        self.reason = reason
        super().__init__(f"الحقل '{field}' غير صالح: {reason}")

VALID_ENDPOINTS = {"/users", "/orders", "/products"}

def process_api_request(endpoint: str, data: dict) -> dict:
    if endpoint not in VALID_ENDPOINTS:
        raise EndpointNotFoundError(endpoint)

    if "user_id" not in data:
        raise MissingFieldError("user_id")

    user_id = data["user_id"]
    if not isinstance(user_id, int) or user_id <= 0:
        raise InvalidFieldError("user_id", "يجب أن يكون رقماً صحيحاً موجباً")

    return {"status": "success", "endpoint": endpoint, "user_id": user_id}

def handle_request(endpoint: str, data: dict) -> None:
    try:
        result = process_api_request(endpoint, data)
    except EndpointNotFoundError as e:
        print(f"404 - {e}")
    except MissingFieldError as e:
        print(f"400 - حقل مفقود: {e}")
    except InvalidFieldError as e:
        print(f"400 - حقل غير صالح: {e}")
    except APIError as e:
        print(f"500 - خطأ API غير متوقع: {e}")
    else:
        print(f"200 - {result}")

handle_request("/comments", {"user_id": 1})         # 404
handle_request("/users", {})                         # 400 مفقود
handle_request("/users", {"user_id": -5})            # 400 غير صالح
handle_request("/users", {"user_id": 42})            # 200 نجاح

# ────────────────────────────────────────────────────────────────────────
# [023] code/chapters/ch11/11-001-balance.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.1: لماذا يحتاج كل برنامج احترافي لمعالجة الأخطاء؟

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-001-balance.py
"""

balance = 1000
withdrawal = input("أدخل مبلغ السحب: ")  # المستخدم يكتب "مئة"
balance -= int(withdrawal)  # 💥 ValueError: invalid literal for int()
# البرنامج ينهار، والمستخدم يرى رسالة تقنية غير مفهومة

# ────────────────────────────────────────────────────────────────────────
# [024] code/chapters/ch11/11-002-ex11_2.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.2: فهم الاستثناءات (Exceptions)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-002-ex11_2.py
"""

print(10 / 0)        # ZeroDivisionError
print(int("abc"))    # ValueError
print(my_list[99])   # IndexError
print(my_dict["x"])  # KeyError
print(undefined_var) # NameError

# ────────────────────────────────────────────────────────────────────────
# [025] code/chapters/ch11/11-003-divide.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.2: فهم الاستثناءات (Exceptions)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-003-divide.py
"""

def divide(a, b):
    return a / b

result = divide(10, 0)

# ────────────────────────────────────────────────────────────────────────
# [026] code/chapters/ch11/11-004-number.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.3: الصيغة الأساسية: try / except

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-004-number.py
"""

try:
    number = int(input("أدخل رقماً: "))
    print(f"ضعف الرقم: {number * 2}")
except ValueError:
    print("❌ خطأ: الرجاء إدخال رقم صحيح فقط.")

print("البرنامج استمر في العمل!")

# ────────────────────────────────────────────────────────────────────────
# [027] code/chapters/ch11/11-005-safe_divide.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.3: الصيغة الأساسية: try / except

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-005-safe_divide.py
"""

def safe_divide(a: float, b: float) -> float | None:
    try:
        return a / b
    except ZeroDivisionError:
        print("❌ لا يمكن القسمة على صفر.")
        return None
    except TypeError:
        print("❌ يجب أن تكون القيم أرقاماً.")
        return None

print(safe_divide(10, 0))        # ❌ لا يمكن القسمة على صفر. → None
print(safe_divide(10, "abc"))    # ❌ يجب أن تكون القيم أرقاماً. → None
print(safe_divide(10, 2))        # 5.0

# ────────────────────────────────────────────────────────────────────────
# [028] code/chapters/ch11/11-006-value.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.3: الصيغة الأساسية: try / except

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-006-value.py
"""

try:
    value = my_dict[key]
    result = 100 / value
except (KeyError, ZeroDivisionError) as e:
    print(f"حدث خطأ: {e}")

# ────────────────────────────────────────────────────────────────────────
# [029] code/chapters/ch11/11-007-ex11_3.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.3: الصيغة الأساسية: try / except

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-007-ex11_3.py
"""

try:
    risky_operation()
except Exception as e:
    print(f"حدث خطأ غير متوقع: {e}")

# ────────────────────────────────────────────────────────────────────────
# [030] code/chapters/ch11/11-008-ex11_3.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.3: الصيغة الأساسية: try / except

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-008-ex11_3.py
"""

# ❌ الأسوأ على الإطلاق - لا تفعل هذا أبداً!
try:
    risky_operation()
except:  # bare except - يلتقط كل شيء حتى KeyboardInterrupt!
    pass  # يُخفي الخطأ تماماً دون أي أثر!

# ────────────────────────────────────────────────────────────────────────
# [031] code/chapters/ch11/11-009-value.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.3: الصيغة الأساسية: try / except

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-009-value.py
"""

# ترتيب except مهم جداً! الأكثر تحديداً أولاً
try:
    value = my_list[10]
except IndexError:
    print("فهرس خارج النطاق")
except LookupError:  # تأتي بعد IndexError لأنها أعم منها
    print("خطأ بحث عام")

# ────────────────────────────────────────────────────────────────────────
# [032] code/chapters/ch11/11-010-ex11_3.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.3: الصيغة الأساسية: try / except

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-010-ex11_3.py
"""

try:
    x = 1 / 0
except Exception:
    print("خطأ عام")  # هذا ما سيُطبع
except ZeroDivisionError:  # ⚠️ ميت! لن يُنفَّذ أبداً
    print("قسمة على صفر")

# ────────────────────────────────────────────────────────────────────────
# [033] code/chapters/ch11/11-011-ex11_4.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.4: else و finally: إكمال الصورة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-011-ex11_4.py
"""

try:
    # الكود الذي قد يُطلق خطأ
    pass
except SomeError:
    # يُنفَّذ فقط إذا حدث هذا الخطأ المحدد
    pass
else:
    # يُنفَّذ فقط إذا لم يحدث أي خطأ في try
    pass
finally:
    # يُنفَّذ دائماً، سواء حدث خطأ أم لا
    pass

# ────────────────────────────────────────────────────────────────────────
# [034] code/chapters/ch11/11-012-read_and_process_number.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.4: else و finally: إكمال الصورة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-012-read_and_process_number.py
"""

def read_and_process_number(text: str) -> None:
    try:
        number = int(text)
    except ValueError:
        print(f"❌ '{text}' ليس رقماً صحيحاً.")
    else:
        # يُنفَّذ فقط إذا نجح التحويل بدون خطأ
        print(f"✅ تم تحويل الرقم بنجاح: {number}")
        print(f"   ضعفه: {number * 2}")
    finally:
        # يُنفَّذ دائماً - مثالي للتنظيف
        print("--- انتهت المحاولة ---")

read_and_process_number("42")
print()
read_and_process_number("abc")

# ────────────────────────────────────────────────────────────────────────
# [035] code/chapters/ch11/11-013-number.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.4: else و finally: إكمال الصورة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-013-number.py
"""

# أقل وضوحاً - أي سطر بالضبط قد يُطلق الخطأ؟
try:
    number = int(text)
    print(f"ضعفه: {number * 2}")  # ماذا لو هذا السطر نفسه فشل لسبب آخر؟
except ValueError:
    print("خطأ")

# أوضح بكثير - try تحتوي فقط على الكود المحتمل فشله
try:
    number = int(text)
except ValueError:
    print("خطأ")
else:
    print(f"ضعفه: {number * 2}")  # نعلم بالتأكيد أن number تم تحويله بنجاح

# ────────────────────────────────────────────────────────────────────────
# [036] code/chapters/ch11/11-014-process_file.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.4: else و finally: إكمال الصورة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-014-process_file.py
"""

def process_file(filename: str) -> None:
    file = None
    try:
        file = open(filename, "r", encoding="utf-8")
        content = file.read()
        print(f"تم قراءة {len(content)} حرف.")
    except FileNotFoundError:
        print(f"❌ الملف '{filename}' غير موجود.")
    finally:
        if file is not None:
            file.close()
            print("✅ تم إغلاق الملف.")  # يحدث دائماً!

process_file("not_exist.txt")

# ────────────────────────────────────────────────────────────────────────
# [037] code/chapters/ch11/11-015-withdraw.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.5: إطلاق الأخطاء بنفسك: raise

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-015-withdraw.py
"""

def withdraw(balance: float, amount: float) -> float:
    if amount <= 0:
        raise ValueError("مبلغ السحب يجب أن يكون موجباً.")
    if amount > balance:
        raise ValueError("الرصيد غير كافٍ لإتمام العملية.")
    return balance - amount

try:
    new_balance = withdraw(1000, 1500)
except ValueError as e:
    print(f"❌ فشلت العملية: {e}")

# ────────────────────────────────────────────────────────────────────────
# [038] code/chapters/ch11/11-016-process_data.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.5: إطلاق الأخطاء بنفسك: raise

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-016-process_data.py
"""

def process_data(data: list) -> float:
    try:
        return sum(data) / len(data)
    except ZeroDivisionError:
        print("⚠️ تحذير: القائمة فارغة، سيتم إعادة إطلاق الخطأ.")
        raise  # raise بدون معامل = إعادة إطلاق نفس الخطأ الملتقط

try:
    process_data([])
except ZeroDivisionError:
    print("❌ المستوى الأعلى: تم التعامل مع القائمة الفارغة.")

# ────────────────────────────────────────────────────────────────────────
# [039] code/chapters/ch11/11-017-parse_age.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.5: إطلاق الأخطاء بنفسك: raise

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-017-parse_age.py
"""

def parse_age(text: str) -> int:
    try:
        return int(text)
    except ValueError as e:
        raise ValueError(f"فشل تحليل العمر من النص: '{text}'") from e

try:
    parse_age("twenty")
except ValueError as e:
    print(e)
    print(f"السبب الأصلي: {e.__cause__}")

# ────────────────────────────────────────────────────────────────────────
# [040] code/chapters/ch11/11-018-withdraw.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.6: إنشاء استثناءات مخصصة (Custom Exceptions)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-018-withdraw.py
"""

class InsufficientFundsError(Exception):
    """تُطلق عندما يحاول المستخدم سحب مبلغ أكبر من رصيده."""
    pass

class InvalidAmountError(Exception):
    """تُطلق عندما يكون المبلغ المُدخل غير منطقي (سالب أو صفر)."""
    pass

def withdraw(balance: float, amount: float) -> float:
    if amount <= 0:
        raise InvalidAmountError(f"المبلغ {amount} غير صالح؛ يجب أن يكون موجباً.")
    if amount > balance:
        raise InsufficientFundsError(
            f"طلبت سحب {amount}، لكن رصيدك الحالي {balance} فقط."
        )
    return balance - amount

try:
    withdraw(500, 1000)
except InsufficientFundsError as e:
    print(f"💰 خطأ في الرصيد: {e}")
except InvalidAmountError as e:
    print(f"🔢 خطأ في المبلغ: {e}")

# ────────────────────────────────────────────────────────────────────────
# [041] code/chapters/ch11/11-019-process_transaction.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.6: إنشاء استثناءات مخصصة (Custom Exceptions)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-019-process_transaction.py
"""

class BankError(Exception):
    """الفئة الأساسية لكل أخطاء النظام البنكي."""
    pass

class InsufficientFundsError(BankError):
    """رصيد غير كافٍ."""
    pass

class InvalidAmountError(BankError):
    """مبلغ غير صالح."""
    pass

class AccountFrozenError(BankError):
    """الحساب مجمّد."""
    pass

def process_transaction(account_status: str, balance: float, amount: float) -> float:
    if account_status == "frozen":
        raise AccountFrozenError("هذا الحساب مجمّد حالياً.")
    if amount <= 0:
        raise InvalidAmountError("المبلغ يجب أن يكون موجباً.")
    if amount > balance:
        raise InsufficientFundsError("الرصيد غير كافٍ.")
    return balance - amount

try:
    process_transaction("frozen", 1000, 100)
except BankError as e:  # التقاط أي خطأ بنكي بغض النظر عن نوعه الدقيق
    print(f"🏦 خطأ في النظام البنكي: {type(e).__name__}: {e}")

# ────────────────────────────────────────────────────────────────────────
# [042] code/chapters/ch11/11-020-__init__.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.6: إنشاء استثناءات مخصصة (Custom Exceptions)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-020-__init__.py
"""

class ValidationError(Exception):
    """خطأ تحقق من صحة البيانات، مع تفاصيل الحقل المُخالف."""

    def __init__(self, field: str, message: str):
        self.field = field
        self.message = message
        super().__init__(f"خطأ في حقل '{field}': {message}")

def validate_age(age: int) -> None:
    if age < 0 or age > 150:
        raise ValidationError(field="age", message="يجب أن يكون بين 0 و150")

try:
    validate_age(-5)
except ValidationError as e:
    print(f"الحقل المُخالف: {e.field}")
    print(f"الرسالة: {e.message}")

# ────────────────────────────────────────────────────────────────────────
# [043] code/chapters/ch11/11-021-content.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.7: فلسفتان في البرمجة: EAFP مقابل LBYL

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-021-content.py
"""

# تحقق أولاً، ثم نفّذ
import os

if os.path.exists("data.txt"):
    with open("data.txt") as f:
        content = f.read()
else:
    print("الملف غير موجود")

# ────────────────────────────────────────────────────────────────────────
# [044] code/chapters/ch11/11-022-content.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.7: فلسفتان في البرمجة: EAFP مقابل LBYL

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-022-content.py
"""

# نفّذ مباشرة، وتعامل مع الخطأ إن حدث
try:
    with open("data.txt") as f:
        content = f.read()
except FileNotFoundError:
    print("الملف غير موجود")

# ────────────────────────────────────────────────────────────────────────
# [045] code/chapters/ch11/11-023-calculate_average.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.8: دالة assert: التحقق أثناء التطوير

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-023-calculate_average.py
"""

def calculate_average(numbers: list[float]) -> float:
    assert len(numbers) > 0, "القائمة يجب ألا تكون فارغة"
    return sum(numbers) / len(numbers)

print(calculate_average([10, 20, 30]))  # 20.0
# print(calculate_average([]))
# 💥 AssertionError: القائمة يجب ألا تكون فارغة

# ────────────────────────────────────────────────────────────────────────
# [046] code/chapters/ch11/11-024-set_age.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.8: دالة assert: التحقق أثناء التطوير

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-024-set_age.py
"""

# ❌ خطأ: استخدام assert للتحقق من مدخلات مستخدم
def set_age(age: int) -> None:
    assert age >= 0, "العمر يجب أن يكون موجباً"  # قد يُحذف بالكامل!

# ✅ صحيح: استخدام raise للتحقق من مدخلات مستخدم
def set_age(age: int) -> None:
    if age < 0:
        raise ValueError("العمر يجب أن يكون موجباً")

# ────────────────────────────────────────────────────────────────────────
# [047] code/chapters/ch11/11-025-divide.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.9: مقدمة مختصرة لـ logging بدلاً من print

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-025-divide.py
"""

import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def divide(a: float, b: float) -> float | None:
    try:
        return a / b
    except ZeroDivisionError:
        logging.error(f"محاولة قسمة {a} على صفر!")
        return None

divide(10, 0)

# ────────────────────────────────────────────────────────────────────────
# [048] code/chapters/ch11/11-026-register.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.10: مشروع تطبيقي: نظام تسجيل دخول آمن

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-026-register.py
"""

class AuthError(Exception):
    """الفئة الأساسية لكل أخطاء المصادقة."""
    pass

class UserNotFoundError(AuthError):
    """المستخدم غير موجود في النظام."""
    pass

class IncorrectPasswordError(AuthError):
    """كلمة المرور غير صحيحة."""
    pass

class AccountLockedError(AuthError):
    """الحساب مقفل بسبب محاولات فاشلة متكررة."""
    pass

class WeakPasswordError(AuthError):
    """كلمة مرور ضعيفة عند التسجيل."""
    pass

# قاعدة بيانات وهمية (في الواقع ستكون قاعدة بيانات حقيقية)
users_db: dict[str, dict] = {
    "ahmed@test.com": {"password": "Secure123", "failed_attempts": 0, "locked": False},
}

MAX_FAILED_ATTEMPTS = 3

def register(email: str, password: str) -> None:
    """تسجيل مستخدم جديد مع التحقق من قوة كلمة المرور."""
    if len(password) < 8:
        raise WeakPasswordError("كلمة المرور يجب أن تكون 8 أحرف على الأقل.")
    if not any(char.isdigit() for char in password):
        raise WeakPasswordError("كلمة المرور يجب أن تحتوي رقماً واحداً على الأقل.")

    if email in users_db:
        raise ValueError("هذا البريد الإلكتروني مسجّل مسبقاً.")

    users_db[email] = {"password": password, "failed_attempts": 0, "locked": False}
    print(f"✅ تم تسجيل المستخدم {email} بنجاح.")

def login(email: str, password: str) -> bool:
    """محاولة تسجيل دخول، مع حماية من المحاولات المتكررة."""
    try:
        user = users_db[email]
    except KeyError:
        raise UserNotFoundError(f"لا يوجد مستخدم بالبريد: {email}")

    if user["locked"]:
        raise AccountLockedError("هذا الحساب مقفل. تواصل مع الدعم الفني.")

    if user["password"] != password:
        user["failed_attempts"] += 1
        if user["failed_attempts"] >= MAX_FAILED_ATTEMPTS:
            user["locked"] = True
            raise AccountLockedError(
                "تم قفل الحساب بسبب تجاوز عدد المحاولات الفاشلة المسموحة."
            )
        remaining = MAX_FAILED_ATTEMPTS - user["failed_attempts"]
        raise IncorrectPasswordError(
            f"كلمة المرور خاطئة. المحاولات المتبقية: {remaining}"
        )

    user["failed_attempts"] = 0  # إعادة تصفير عند النجاح
    return True

def run_login_attempt(email: str, password: str) -> None:
    """واجهة تفاعلية تلتقط كل أنواع أخطاء المصادقة بوضوح."""
    try:
        login(email, password)
    except UserNotFoundError as e:
        print(f"👤 {e}")
    except AccountLockedError as e:
        print(f"🔒 {e}")
    except IncorrectPasswordError as e:
        print(f"🔑 {e}")
    except AuthError as e:
        print(f"⚠️ خطأ مصادقة غير متوقع: {e}")
    else:
        print(f"✅ تم تسجيل الدخول بنجاح! أهلاً بك {email}")
    finally:
        print("-" * 40)

# --- سيناريوهات تجريبية ---
if __name__ == "__main__":
    try:
        register("sara@test.com", "weak")
    except WeakPasswordError as e:
        print(f"❌ فشل التسجيل: {e}")
    print("-" * 40)

    register("sara@test.com", "Strong123")
    print("-" * 40)

    run_login_attempt("unknown@test.com", "anything")   # UserNotFoundError
    run_login_attempt("ahmed@test.com", "wrong_pass")   # IncorrectPasswordError (محاولة 1)
    run_login_attempt("ahmed@test.com", "wrong_pass")   # IncorrectPasswordError (محاولة 2)
    run_login_attempt("ahmed@test.com", "wrong_pass")   # AccountLockedError (محاولة 3 - قفل!)
    run_login_attempt("ahmed@test.com", "Secure123")    # AccountLockedError (مقفل مسبقاً)
    run_login_attempt("sara@test.com", "Strong123")     # ✅ نجاح

# ────────────────────────────────────────────────────────────────────────
# [049] code/chapters/ch11/11-027-reset_password.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.10: مشروع تطبيقي: نظام تسجيل دخول آمن

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-027-reset_password.py
"""

def reset_password(email: str, new_password: str) -> None:
    """إعادة تعيين كلمة مرور، مع إلغاء القفل إن وُجد."""
    try:
        user = users_db[email]
    except KeyError:
        raise UserNotFoundError(f"لا يوجد مستخدم بالبريد: {email}")

    if len(new_password) < 8 or not any(c.isdigit() for c in new_password):
        raise WeakPasswordError("كلمة المرور الجديدة ضعيفة جداً.")

    user["password"] = new_password
    user["failed_attempts"] = 0
    user["locked"] = False
    print(f"🔓 تم إعادة تعيين كلمة المرور وفك قفل الحساب لـ {email}")

try:
    reset_password("ahmed@test.com", "NewPass123")
except AuthError as e:
    print(f"❌ {e}")

# ────────────────────────────────────────────────────────────────────────
# [050] code/chapters/ch11/11-028-safe_list_access.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.11: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-028-safe_list_access.py
"""

# تمرين 1
try:
    num = int(input("أدخل رقماً: "))
    print("رقم صحيح")
except ValueError:
    print("خطأ: ليس رقماً")

# تمرين 2
def safe_list_access(lst: list, index: int):
    try:
        return lst[index]
    except IndexError:
        return None

print(safe_list_access([1, 2, 3], 1))   # 2
print(safe_list_access([1, 2, 3], 10))  # None

# ────────────────────────────────────────────────────────────────────────
# [051] code/chapters/ch11/11-029-calculate_square_root.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.11: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-029-calculate_square_root.py
"""

import math

# تمرين 3
class NegativeValueError(Exception):
    pass

def calculate_square_root(n: float) -> float:
    if n < 0:
        raise NegativeValueError(f"لا يمكن حساب جذر رقم سالب: {n}")
    return math.sqrt(n)

try:
    print(calculate_square_root(-9))
except NegativeValueError as e:
    print(f"❌ {e}")

# تمرين 4
class MissingConfigError(Exception):
    def __init__(self, key: str):
        self.key = key
        super().__init__(f"الإعداد المطلوب '{key}' غير موجود.")

def parse_config(data: dict) -> tuple[str, int]:
    try:
        host = data["host"]
    except KeyError:
        raise MissingConfigError("host")

    try:
        port = data["port"]
    except KeyError:
        raise MissingConfigError("port")

    return host, port

try:
    parse_config({"host": "localhost"})
except MissingConfigError as e:
    print(f"❌ {e}")  # ❌ الإعداد المطلوب 'port' غير موجود.

# ────────────────────────────────────────────────────────────────────────
# [052] code/chapters/ch11/11-030-__init__.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.11: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-030-__init__.py
"""

class APIError(Exception):
    """الفئة الأساسية لكل أخطاء API."""
    pass

class EndpointNotFoundError(APIError):
    def __init__(self, endpoint: str):
        self.endpoint = endpoint
        super().__init__(f"المسار '{endpoint}' غير موجود.")

class MissingFieldError(APIError):
    def __init__(self, field: str):
        self.field = field
        super().__init__(f"الحقل المطلوب '{field}' مفقود.")

class InvalidFieldError(APIError):
    def __init__(self, field: str, reason: str):
        self.field = field
        self.reason = reason
        super().__init__(f"الحقل '{field}' غير صالح: {reason}")

VALID_ENDPOINTS = {"/users", "/orders", "/products"}

def process_api_request(endpoint: str, data: dict) -> dict:
    if endpoint not in VALID_ENDPOINTS:
        raise EndpointNotFoundError(endpoint)

    if "user_id" not in data:
        raise MissingFieldError("user_id")

    user_id = data["user_id"]
    if not isinstance(user_id, int) or user_id <= 0:
        raise InvalidFieldError("user_id", "يجب أن يكون رقماً صحيحاً موجباً")

    return {"status": "success", "endpoint": endpoint, "user_id": user_id}

def handle_request(endpoint: str, data: dict) -> None:
    try:
        result = process_api_request(endpoint, data)
    except EndpointNotFoundError as e:
        print(f"404 - {e}")
    except MissingFieldError as e:
        print(f"400 - حقل مفقود: {e}")
    except InvalidFieldError as e:
        print(f"400 - حقل غير صالح: {e}")
    except APIError as e:
        print(f"500 - خطأ API غير متوقع: {e}")
    else:
        print(f"200 - {result}")

handle_request("/comments", {"user_id": 1})         # 404
handle_request("/users", {})                         # 400 مفقود
handle_request("/users", {"user_id": -5})            # 400 غير صالح
handle_request("/users", {"user_id": 42})            # 200 نجاح

# ────────────────────────────────────────────────────────────────────────
# [053] code/chapters/ch11/11-031-__init__.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.13: ملخص الفصل (Cheat Sheet)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-031-__init__.py
"""

# الصيغة الكاملة
try:
    risky_code()
except SpecificError as e:
    handle_specific_case(e)
except (ErrorA, ErrorB) as e:
    handle_multiple_cases(e)
except Exception as e:      # الأعم - دائماً في الأخير
    handle_general_case(e)
else:
    runs_if_no_error()
finally:
    always_runs()            # تنظيف الموارد

# إطلاق خطأ
raise ValueError("رسالة واضحة")

# إعادة إطلاق
except SomeError:
    raise  # بدون معامل

# تسلسل الاستثناءات
raise NewError("...") from original_error

# استثناء مخصص
class MyAppError(Exception):
    pass

class SpecificError(MyAppError):
    def __init__(self, detail: str):
        self.detail = detail
        super().__init__(f"فشل بسبب: {detail}")

# assert - للتطوير فقط، ليس لمدخلات المستخدم
assert condition, "رسالة إذا فشل الشرط"

# ────────────────────────────────────────────────────────────────────────
# [054] dist/markdown/chapter-11.md
# ────────────────────────────────────────────────────────────────────────
balance = 1000
withdrawal = input("أدخل مبلغ السحب: ")  # المستخدم يكتب "مئة"
balance -= int(withdrawal)  # 💥 ValueError: invalid literal for int()
# البرنامج ينهار، والمستخدم يرى رسالة تقنية غير مفهومة

# ────────────────────────────────────────────────────────────────────────
# [055] dist/markdown/chapter-11.md
# ────────────────────────────────────────────────────────────────────────
print(10 / 0)        # ZeroDivisionError
print(int("abc"))    # ValueError
print(my_list[99])   # IndexError
print(my_dict["x"])  # KeyError
print(undefined_var) # NameError

# ────────────────────────────────────────────────────────────────────────
# [056] dist/markdown/chapter-11.md
# ────────────────────────────────────────────────────────────────────────
def divide(a, b):
    return a / b

result = divide(10, 0)

# ────────────────────────────────────────────────────────────────────────
# [057] dist/markdown/chapter-11.md
# ────────────────────────────────────────────────────────────────────────
try:
    number = int(input("أدخل رقماً: "))
    print(f"ضعف الرقم: {number * 2}")
except ValueError:
    print("❌ خطأ: الرجاء إدخال رقم صحيح فقط.")

print("البرنامج استمر في العمل!")

# ────────────────────────────────────────────────────────────────────────
# [058] dist/markdown/chapter-11.md
# ────────────────────────────────────────────────────────────────────────
def safe_divide(a: float, b: float) -> float | None:
    try:
        return a / b
    except ZeroDivisionError:
        print("❌ لا يمكن القسمة على صفر.")
        return None
    except TypeError:
        print("❌ يجب أن تكون القيم أرقاماً.")
        return None

print(safe_divide(10, 0))        # ❌ لا يمكن القسمة على صفر. → None
print(safe_divide(10, "abc"))    # ❌ يجب أن تكون القيم أرقاماً. → None
print(safe_divide(10, 2))        # 5.0

# ────────────────────────────────────────────────────────────────────────
# [059] dist/markdown/chapter-11.md
# ────────────────────────────────────────────────────────────────────────
try:
    value = my_dict[key]
    result = 100 / value
except (KeyError, ZeroDivisionError) as e:
    print(f"حدث خطأ: {e}")

# ────────────────────────────────────────────────────────────────────────
# [060] dist/markdown/chapter-11.md
# ────────────────────────────────────────────────────────────────────────
try:
    risky_operation()
except Exception as e:
    print(f"حدث خطأ غير متوقع: {e}")

# ────────────────────────────────────────────────────────────────────────
# [061] dist/markdown/chapter-11.md
# ────────────────────────────────────────────────────────────────────────
# ❌ الأسوأ على الإطلاق - لا تفعل هذا أبداً!
try:
    risky_operation()
except:  # bare except - يلتقط كل شيء حتى KeyboardInterrupt!
    pass  # يُخفي الخطأ تماماً دون أي أثر!

# ────────────────────────────────────────────────────────────────────────
# [062] dist/markdown/chapter-11.md
# ────────────────────────────────────────────────────────────────────────
# ترتيب except مهم جداً! الأكثر تحديداً أولاً
try:
    value = my_list[10]
except IndexError:
    print("فهرس خارج النطاق")
except LookupError:  # تأتي بعد IndexError لأنها أعم منها
    print("خطأ بحث عام")

# ────────────────────────────────────────────────────────────────────────
# [063] dist/markdown/chapter-11.md
# ────────────────────────────────────────────────────────────────────────
try:
    # الكود الذي قد يُطلق خطأ
    pass
except SomeError:
    # يُنفَّذ فقط إذا حدث هذا الخطأ المحدد
    pass
else:
    # يُنفَّذ فقط إذا لم يحدث أي خطأ في try
    pass
finally:
    # يُنفَّذ دائماً، سواء حدث خطأ أم لا
    pass

# ────────────────────────────────────────────────────────────────────────
# [064] dist/markdown/chapter-11.md
# ────────────────────────────────────────────────────────────────────────
# أقل وضوحاً - أي سطر بالضبط قد يُطلق الخطأ؟
try:
    number = int(text)
    print(f"ضعفه: {number * 2}")  # ماذا لو هذا السطر نفسه فشل لسبب آخر؟
except ValueError:
    print("خطأ")

# أوضح بكثير - try تحتوي فقط على الكود المحتمل فشله
try:
    number = int(text)
except ValueError:
    print("خطأ")
else:
    print(f"ضعفه: {number * 2}")  # نعلم بالتأكيد أن number تم تحويله بنجاح

# ────────────────────────────────────────────────────────────────────────
# [065] dist/markdown/chapter-11.md
# ────────────────────────────────────────────────────────────────────────
def calculate_average(numbers: list[float]) -> float:
    assert len(numbers) > 0, "القائمة يجب ألا تكون فارغة"
    return sum(numbers) / len(numbers)

print(calculate_average([10, 20, 30]))  # 20.0
# print(calculate_average([]))
# 💥 AssertionError: القائمة يجب ألا تكون فارغة

# ────────────────────────────────────────────────────────────────────────
# [066] dist/markdown/chapter-11.md
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: استخدام assert للتحقق من مدخلات مستخدم
def set_age(age: int) -> None:
    assert age >= 0, "العمر يجب أن يكون موجباً"  # قد يُحذف بالكامل!

# ✅ صحيح: استخدام raise للتحقق من مدخلات مستخدم
def set_age(age: int) -> None:
    if age < 0:
        raise ValueError("العمر يجب أن يكون موجباً")

# ────────────────────────────────────────────────────────────────────────
# [067] dist/markdown/chapter-11.md
# ────────────────────────────────────────────────────────────────────────
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def divide(a: float, b: float) -> float | None:
    try:
        return a / b
    except ZeroDivisionError:
        logging.error(f"محاولة قسمة {a} على صفر!")
        return None

divide(10, 0)

# ────────────────────────────────────────────────────────────────────────
# [068] dist/markdown/chapter-11.md
# ────────────────────────────────────────────────────────────────────────
# تمرين 1
try:
    num = int(input("أدخل رقماً: "))
    print("رقم صحيح")
except ValueError:
    print("خطأ: ليس رقماً")

# تمرين 2
def safe_list_access(lst: list, index: int):
    try:
        return lst[index]
    except IndexError:
        return None

print(safe_list_access([1, 2, 3], 1))   # 2
print(safe_list_access([1, 2, 3], 10))  # None

# ────────────────────────────────────────────────────────────────────────
# [069] dist/markdown/chapter-11.md
# ────────────────────────────────────────────────────────────────────────
# الصيغة الكاملة
try:
    risky_code()
except SpecificError as e:
    handle_specific_case(e)
except (ErrorA, ErrorB) as e:
    handle_multiple_cases(e)
except Exception as e:      # الأعم - دائماً في الأخير
    handle_general_case(e)
else:
    runs_if_no_error()
finally:
    always_runs()            # تنظيف الموارد

# إطلاق خطأ
raise ValueError("رسالة واضحة")

# إعادة إطلاق
except SomeError:
    raise  # بدون معامل

# تسلسل الاستثناءات
raise NewError("...") from original_error

# استثناء مخصص
class MyAppError(Exception):
    pass

class SpecificError(MyAppError):
    def __init__(self, detail: str):
        self.detail = detail
        super().__init__(f"فشل بسبب: {detail}")

# assert - للتطوير فقط، ليس لمدخلات المستخدم
assert condition, "رسالة إذا فشل الشرط"

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-11-chapter-11.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
#@title ⚙️ تهيئة البيئة  { display-mode: "form" }
import sys, os, random, pathlib

print("Python:", sys.version.split()[0], "|", sys.platform)
assert sys.version_info >= (3, 10), "هذا الكتاب يفترض Python 3.10 أو أحدث"

SEED = 42
random.seed(SEED)

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-11-chapter-11.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# ⚠️ مثال يُظهر خطأً مقصودًا: نلتقطه هنا ليكتمل «تشغيل الكل»،
# وسترى نوع الخطأ ورسالته في المخرجات كما في الطرفية.
try:
    def divide(a, b):
        return a / b

    result = divide(10, 0)
except (Exception, SystemExit) as _err:
    print(f"⚠️ {type(_err).__name__}: {_err}")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-11-chapter-11.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# 📝 نسخة الدفتر: حقل نموذج بدل الإدخال اليدوي — عدّل القيمة ثم أعد التشغيل
try:
    number = 10  #@param {type:"integer"}
    print(f"ضعف الرقم: {number * 2}")
except ValueError:
    print("❌ خطأ: الرجاء إدخال رقم صحيح فقط.")

print("البرنامج استمر في العمل!")

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-11-chapter-11.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# ⚠️ مثال يُظهر خطأً مقصودًا: نلتقطه هنا ليكتمل «تشغيل الكل»،
# وسترى نوع الخطأ ورسالته في المخرجات كما في الطرفية.
try:
    def safe_divide(a: float, b: float) -> float | None:
        try:
            return a / b
        except ZeroDivisionError:
            print("❌ لا يمكن القسمة على صفر.")
            return None
        except TypeError:
            print("❌ يجب أن تكون القيم أرقاماً.")
            return None

    print(safe_divide(10, 0))        # ❌ لا يمكن القسمة على صفر. → None
    print(safe_divide(10, "abc"))    # ❌ يجب أن تكون القيم أرقاماً. → None
    print(safe_divide(10, 2))        # 5.0
except (Exception, SystemExit) as _err:
    print(f"⚠️ {type(_err).__name__}: {_err}")

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-11-chapter-11.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
try:
    x = 1 / 0
except Exception:
    print("خطأ عام")  # هذا ما سيُطبع
except ZeroDivisionError:  # ⚠️ ميت! لن يُنفَّذ أبداً
    print("قسمة على صفر")

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-11-chapter-11.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
def read_and_process_number(text: str) -> None:
    try:
        number = int(text)
    except ValueError:
        print(f"❌ '{text}' ليس رقماً صحيحاً.")
    else:
        # يُنفَّذ فقط إذا نجح التحويل بدون خطأ
        print(f"✅ تم تحويل الرقم بنجاح: {number}")
        print(f"   ضعفه: {number * 2}")
    finally:
        # يُنفَّذ دائماً - مثالي للتنظيف
        print("--- انتهت المحاولة ---")

read_and_process_number("42")
print()
read_and_process_number("abc")

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-11-chapter-11.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
def process_file(filename: str) -> None:
    file = None
    try:
        file = open(filename, "r", encoding="utf-8")
        content = file.read()
        print(f"تم قراءة {len(content)} حرف.")
    except FileNotFoundError:
        print(f"❌ الملف '{filename}' غير موجود.")
    finally:
        if file is not None:
            file.close()
            print("✅ تم إغلاق الملف.")  # يحدث دائماً!

process_file("not_exist.txt")

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-11-chapter-11.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
def withdraw(balance: float, amount: float) -> float:
    if amount <= 0:
        raise ValueError("مبلغ السحب يجب أن يكون موجباً.")
    if amount > balance:
        raise ValueError("الرصيد غير كافٍ لإتمام العملية.")
    return balance - amount

try:
    new_balance = withdraw(1000, 1500)
except ValueError as e:
    print(f"❌ فشلت العملية: {e}")

# ────────────────────────────────────────────────────────────────────────
# [011] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-11-chapter-11.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
def process_data(data: list) -> float:
    try:
        return sum(data) / len(data)
    except ZeroDivisionError:
        print("⚠️ تحذير: القائمة فارغة، سيتم إعادة إطلاق الخطأ.")
        raise  # raise بدون معامل = إعادة إطلاق نفس الخطأ الملتقط

try:
    process_data([])
except ZeroDivisionError:
    print("❌ المستوى الأعلى: تم التعامل مع القائمة الفارغة.")

# ────────────────────────────────────────────────────────────────────────
# [012] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-11-chapter-11.ipynb#cell-11
# ────────────────────────────────────────────────────────────────────────
def parse_age(text: str) -> int:
    try:
        return int(text)
    except ValueError as e:
        raise ValueError(f"فشل تحليل العمر من النص: '{text}'") from e

try:
    parse_age("twenty")
except ValueError as e:
    print(e)
    print(f"السبب الأصلي: {e.__cause__}")

# ────────────────────────────────────────────────────────────────────────
# [013] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-11-chapter-11.ipynb#cell-12
# ────────────────────────────────────────────────────────────────────────
class InsufficientFundsError(Exception):
    """تُطلق عندما يحاول المستخدم سحب مبلغ أكبر من رصيده."""
    pass

class InvalidAmountError(Exception):
    """تُطلق عندما يكون المبلغ المُدخل غير منطقي (سالب أو صفر)."""
    pass

def withdraw(balance: float, amount: float) -> float:
    if amount <= 0:
        raise InvalidAmountError(f"المبلغ {amount} غير صالح؛ يجب أن يكون موجباً.")
    if amount > balance:
        raise InsufficientFundsError(
            f"طلبت سحب {amount}، لكن رصيدك الحالي {balance} فقط."
        )
    return balance - amount

try:
    withdraw(500, 1000)
except InsufficientFundsError as e:
    print(f"💰 خطأ في الرصيد: {e}")
except InvalidAmountError as e:
    print(f"🔢 خطأ في المبلغ: {e}")

# ────────────────────────────────────────────────────────────────────────
# [014] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-11-chapter-11.ipynb#cell-13
# ────────────────────────────────────────────────────────────────────────
class BankError(Exception):
    """الفئة الأساسية لكل أخطاء النظام البنكي."""
    pass

class InsufficientFundsError(BankError):
    """رصيد غير كافٍ."""
    pass

class InvalidAmountError(BankError):
    """مبلغ غير صالح."""
    pass

class AccountFrozenError(BankError):
    """الحساب مجمّد."""
    pass

def process_transaction(account_status: str, balance: float, amount: float) -> float:
    if account_status == "frozen":
        raise AccountFrozenError("هذا الحساب مجمّد حالياً.")
    if amount <= 0:
        raise InvalidAmountError("المبلغ يجب أن يكون موجباً.")
    if amount > balance:
        raise InsufficientFundsError("الرصيد غير كافٍ.")
    return balance - amount

try:
    process_transaction("frozen", 1000, 100)
except BankError as e:  # التقاط أي خطأ بنكي بغض النظر عن نوعه الدقيق
    print(f"🏦 خطأ في النظام البنكي: {type(e).__name__}: {e}")

# ────────────────────────────────────────────────────────────────────────
# [015] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-11-chapter-11.ipynb#cell-14
# ────────────────────────────────────────────────────────────────────────
class ValidationError(Exception):
    """خطأ تحقق من صحة البيانات، مع تفاصيل الحقل المُخالف."""

    def __init__(self, field: str, message: str):
        self.field = field
        self.message = message
        super().__init__(f"خطأ في حقل '{field}': {message}")

def validate_age(age: int) -> None:
    if age < 0 or age > 150:
        raise ValidationError(field="age", message="يجب أن يكون بين 0 و150")

try:
    validate_age(-5)
except ValidationError as e:
    print(f"الحقل المُخالف: {e.field}")
    print(f"الرسالة: {e.message}")

# ────────────────────────────────────────────────────────────────────────
# [016] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-11-chapter-11.ipynb#cell-15
# ────────────────────────────────────────────────────────────────────────
# تحقق أولاً، ثم نفّذ
import os

if os.path.exists("data.txt"):
    with open("data.txt") as f:
        content = f.read()
else:
    print("الملف غير موجود")

# ────────────────────────────────────────────────────────────────────────
# [017] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-11-chapter-11.ipynb#cell-16
# ────────────────────────────────────────────────────────────────────────
# نفّذ مباشرة، وتعامل مع الخطأ إن حدث
try:
    with open("data.txt") as f:
        content = f.read()
except FileNotFoundError:
    print("الملف غير موجود")

# ────────────────────────────────────────────────────────────────────────
# [018] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-11-chapter-11.ipynb#cell-17
# ────────────────────────────────────────────────────────────────────────
# ⚠️ مثال يُظهر خطأً مقصودًا: نلتقطه هنا ليكتمل «تشغيل الكل»،
# وسترى نوع الخطأ ورسالته في المخرجات كما في الطرفية.
try:
    def calculate_average(numbers: list[float]) -> float:
        assert len(numbers) > 0, "القائمة يجب ألا تكون فارغة"
        return sum(numbers) / len(numbers)

    print(calculate_average([10, 20, 30]))  # 20.0
    # print(calculate_average([]))
    # 💥 AssertionError: القائمة يجب ألا تكون فارغة
except (Exception, SystemExit) as _err:
    print(f"⚠️ {type(_err).__name__}: {_err}")

# ────────────────────────────────────────────────────────────────────────
# [019] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-11-chapter-11.ipynb#cell-18
# ────────────────────────────────────────────────────────────────────────
# ⚠️ مثال يُظهر خطأً مقصودًا: نلتقطه هنا ليكتمل «تشغيل الكل»،
# وسترى نوع الخطأ ورسالته في المخرجات كما في الطرفية.
try:
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    def divide(a: float, b: float) -> float | None:
        try:
            return a / b
        except ZeroDivisionError:
            logging.error(f"محاولة قسمة {a} على صفر!")
            return None

    divide(10, 0)
except (Exception, SystemExit) as _err:
    print(f"⚠️ {type(_err).__name__}: {_err}")

# ────────────────────────────────────────────────────────────────────────
# [020] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-11-chapter-11.ipynb#cell-19
# ────────────────────────────────────────────────────────────────────────
class AuthError(Exception):
    """الفئة الأساسية لكل أخطاء المصادقة."""
    pass

class UserNotFoundError(AuthError):
    """المستخدم غير موجود في النظام."""
    pass

class IncorrectPasswordError(AuthError):
    """كلمة المرور غير صحيحة."""
    pass

class AccountLockedError(AuthError):
    """الحساب مقفل بسبب محاولات فاشلة متكررة."""
    pass

class WeakPasswordError(AuthError):
    """كلمة مرور ضعيفة عند التسجيل."""
    pass

# قاعدة بيانات وهمية (في الواقع ستكون قاعدة بيانات حقيقية)
users_db: dict[str, dict] = {
    "ahmed@test.com": {"password": "Secure123", "failed_attempts": 0, "locked": False},
}

MAX_FAILED_ATTEMPTS = 3

def register(email: str, password: str) -> None:
    """تسجيل مستخدم جديد مع التحقق من قوة كلمة المرور."""
    if len(password) < 8:
        raise WeakPasswordError("كلمة المرور يجب أن تكون 8 أحرف على الأقل.")
    if not any(char.isdigit() for char in password):
        raise WeakPasswordError("كلمة المرور يجب أن تحتوي رقماً واحداً على الأقل.")

    if email in users_db:
        raise ValueError("هذا البريد الإلكتروني مسجّل مسبقاً.")

    users_db[email] = {"password": password, "failed_attempts": 0, "locked": False}
    print(f"✅ تم تسجيل المستخدم {email} بنجاح.")

def login(email: str, password: str) -> bool:
    """محاولة تسجيل دخول، مع حماية من المحاولات المتكررة."""
    try:
        user = users_db[email]
    except KeyError:
        raise UserNotFoundError(f"لا يوجد مستخدم بالبريد: {email}")

    if user["locked"]:
        raise AccountLockedError("هذا الحساب مقفل. تواصل مع الدعم الفني.")

    if user["password"] != password:
        user["failed_attempts"] += 1
        if user["failed_attempts"] >= MAX_FAILED_ATTEMPTS:
            user["locked"] = True
            raise AccountLockedError(
                "تم قفل الحساب بسبب تجاوز عدد المحاولات الفاشلة المسموحة."
            )
        remaining = MAX_FAILED_ATTEMPTS - user["failed_attempts"]
        raise IncorrectPasswordError(
            f"كلمة المرور خاطئة. المحاولات المتبقية: {remaining}"
        )

    user["failed_attempts"] = 0  # إعادة تصفير عند النجاح
    return True

def run_login_attempt(email: str, password: str) -> None:
    """واجهة تفاعلية تلتقط كل أنواع أخطاء المصادقة بوضوح."""
    try:
        login(email, password)
    except UserNotFoundError as e:
        print(f"👤 {e}")
    except AccountLockedError as e:
        print(f"🔒 {e}")
    except IncorrectPasswordError as e:
        print(f"🔑 {e}")
    except AuthError as e:
        print(f"⚠️ خطأ مصادقة غير متوقع: {e}")
    else:
        print(f"✅ تم تسجيل الدخول بنجاح! أهلاً بك {email}")
    finally:
        print("-" * 40)

# --- سيناريوهات تجريبية ---
if __name__ == "__main__":
    try:
        register("sara@test.com", "weak")
    except WeakPasswordError as e:
        print(f"❌ فشل التسجيل: {e}")
    print("-" * 40)

    register("sara@test.com", "Strong123")
    print("-" * 40)

    run_login_attempt("unknown@test.com", "anything")   # UserNotFoundError
    run_login_attempt("ahmed@test.com", "wrong_pass")   # IncorrectPasswordError (محاولة 1)
    run_login_attempt("ahmed@test.com", "wrong_pass")   # IncorrectPasswordError (محاولة 2)
    run_login_attempt("ahmed@test.com", "wrong_pass")   # AccountLockedError (محاولة 3 - قفل!)
    run_login_attempt("ahmed@test.com", "Secure123")    # AccountLockedError (مقفل مسبقاً)
    run_login_attempt("sara@test.com", "Strong123")     # ✅ نجاح

# ────────────────────────────────────────────────────────────────────────
# [021] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-11-chapter-11.ipynb#cell-20
# ────────────────────────────────────────────────────────────────────────
def reset_password(email: str, new_password: str) -> None:
    """إعادة تعيين كلمة مرور، مع إلغاء القفل إن وُجد."""
    try:
        user = users_db[email]
    except KeyError:
        raise UserNotFoundError(f"لا يوجد مستخدم بالبريد: {email}")

    if len(new_password) < 8 or not any(c.isdigit() for c in new_password):
        raise WeakPasswordError("كلمة المرور الجديدة ضعيفة جداً.")

    user["password"] = new_password
    user["failed_attempts"] = 0
    user["locked"] = False
    print(f"🔓 تم إعادة تعيين كلمة المرور وفك قفل الحساب لـ {email}")

try:
    reset_password("ahmed@test.com", "NewPass123")
except AuthError as e:
    print(f"❌ {e}")

# ────────────────────────────────────────────────────────────────────────
# [022] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-11-chapter-11.ipynb#cell-21
# ────────────────────────────────────────────────────────────────────────
# 📝 نسخة الدفتر: حقل نموذج بدل الإدخال اليدوي — عدّل القيمة ثم أعد التشغيل
# تمرين 1
try:
    num = 10  #@param {type:"integer"}
    print("رقم صحيح")
except ValueError:
    print("خطأ: ليس رقماً")

# تمرين 2
def safe_list_access(lst: list, index: int):
    try:
        return lst[index]
    except IndexError:
        return None

print(safe_list_access([1, 2, 3], 1))   # 2
print(safe_list_access([1, 2, 3], 10))  # None

# ────────────────────────────────────────────────────────────────────────
# [023] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-11-chapter-11.ipynb#cell-22
# ────────────────────────────────────────────────────────────────────────
import math

# تمرين 3
class NegativeValueError(Exception):
    pass

def calculate_square_root(n: float) -> float:
    if n < 0:
        raise NegativeValueError(f"لا يمكن حساب جذر رقم سالب: {n}")
    return math.sqrt(n)

try:
    print(calculate_square_root(-9))
except NegativeValueError as e:
    print(f"❌ {e}")

# تمرين 4
class MissingConfigError(Exception):
    def __init__(self, key: str):
        self.key = key
        super().__init__(f"الإعداد المطلوب '{key}' غير موجود.")

def parse_config(data: dict) -> tuple[str, int]:
    try:
        host = data["host"]
    except KeyError:
        raise MissingConfigError("host")

    try:
        port = data["port"]
    except KeyError:
        raise MissingConfigError("port")

    return host, port

try:
    parse_config({"host": "localhost"})
except MissingConfigError as e:
    print(f"❌ {e}")  # ❌ الإعداد المطلوب 'port' غير موجود.

# ────────────────────────────────────────────────────────────────────────
# [024] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-11-chapter-11.ipynb#cell-23
# ────────────────────────────────────────────────────────────────────────
class APIError(Exception):
    """الفئة الأساسية لكل أخطاء API."""
    pass

class EndpointNotFoundError(APIError):
    def __init__(self, endpoint: str):
        self.endpoint = endpoint
        super().__init__(f"المسار '{endpoint}' غير موجود.")

class MissingFieldError(APIError):
    def __init__(self, field: str):
        self.field = field
        super().__init__(f"الحقل المطلوب '{field}' مفقود.")

class InvalidFieldError(APIError):
    def __init__(self, field: str, reason: str):
        self.field = field
        self.reason = reason
        super().__init__(f"الحقل '{field}' غير صالح: {reason}")

VALID_ENDPOINTS = {"/users", "/orders", "/products"}

def process_api_request(endpoint: str, data: dict) -> dict:
    if endpoint not in VALID_ENDPOINTS:
        raise EndpointNotFoundError(endpoint)

    if "user_id" not in data:
        raise MissingFieldError("user_id")

    user_id = data["user_id"]
    if not isinstance(user_id, int) or user_id <= 0:
        raise InvalidFieldError("user_id", "يجب أن يكون رقماً صحيحاً موجباً")

    return {"status": "success", "endpoint": endpoint, "user_id": user_id}

def handle_request(endpoint: str, data: dict) -> None:
    try:
        result = process_api_request(endpoint, data)
    except EndpointNotFoundError as e:
        print(f"404 - {e}")
    except MissingFieldError as e:
        print(f"400 - حقل مفقود: {e}")
    except InvalidFieldError as e:
        print(f"400 - حقل غير صالح: {e}")
    except APIError as e:
        print(f"500 - خطأ API غير متوقع: {e}")
    else:
        print(f"200 - {result}")

handle_request("/comments", {"user_id": 1})         # 404
handle_request("/users", {})                         # 400 مفقود
handle_request("/users", {"user_id": -5})            # 400 غير صالح
handle_request("/users", {"user_id": 42})            # 200 نجاح

# ────────────────────────────────────────────────────────────────────────
# [025] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-11-chapter-11.ipynb#cell-24
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.1: لماذا يحتاج كل برنامج احترافي لمعالجة الأخطاء؟

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-001-balance.py
"""

balance = 1000
withdrawal = input("أدخل مبلغ السحب: ")  # المستخدم يكتب "مئة"
balance -= int(withdrawal)  # 💥 ValueError: invalid literal for int()
# البرنامج ينهار، والمستخدم يرى رسالة تقنية غير مفهومة

# ────────────────────────────────────────────────────────────────────────
# [026] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-11-chapter-11.ipynb#cell-25
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.2: فهم الاستثناءات (Exceptions)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-002-ex11_2.py
"""

print(10 / 0)        # ZeroDivisionError
print(int("abc"))    # ValueError
print(my_list[99])   # IndexError
print(my_dict["x"])  # KeyError
print(undefined_var) # NameError

# ────────────────────────────────────────────────────────────────────────
# [027] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-11-chapter-11.ipynb#cell-26
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.2: فهم الاستثناءات (Exceptions)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-003-divide.py
"""

def divide(a, b):
    return a / b

result = divide(10, 0)

# ────────────────────────────────────────────────────────────────────────
# [028] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-11-chapter-11.ipynb#cell-27
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.3: الصيغة الأساسية: try / except

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-004-number.py
"""

try:
    number = int(input("أدخل رقماً: "))
    print(f"ضعف الرقم: {number * 2}")
except ValueError:
    print("❌ خطأ: الرجاء إدخال رقم صحيح فقط.")

print("البرنامج استمر في العمل!")

# ────────────────────────────────────────────────────────────────────────
# [029] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-11-chapter-11.ipynb#cell-28
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.3: الصيغة الأساسية: try / except

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-005-safe_divide.py
"""

def safe_divide(a: float, b: float) -> float | None:
    try:
        return a / b
    except ZeroDivisionError:
        print("❌ لا يمكن القسمة على صفر.")
        return None
    except TypeError:
        print("❌ يجب أن تكون القيم أرقاماً.")
        return None

print(safe_divide(10, 0))        # ❌ لا يمكن القسمة على صفر. → None
print(safe_divide(10, "abc"))    # ❌ يجب أن تكون القيم أرقاماً. → None
print(safe_divide(10, 2))        # 5.0

# ────────────────────────────────────────────────────────────────────────
# [030] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-11-chapter-11.ipynb#cell-29
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.3: الصيغة الأساسية: try / except

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-006-value.py
"""

try:
    value = my_dict[key]
    result = 100 / value
except (KeyError, ZeroDivisionError) as e:
    print(f"حدث خطأ: {e}")

# ────────────────────────────────────────────────────────────────────────
# [031] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-11-chapter-11.ipynb#cell-30
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.3: الصيغة الأساسية: try / except

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-007-ex11_3.py
"""

try:
    risky_operation()
except Exception as e:
    print(f"حدث خطأ غير متوقع: {e}")

# ────────────────────────────────────────────────────────────────────────
# [032] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-11-chapter-11.ipynb#cell-31
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.3: الصيغة الأساسية: try / except

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-008-ex11_3.py
"""

# ❌ الأسوأ على الإطلاق - لا تفعل هذا أبداً!
try:
    risky_operation()
except:  # bare except - يلتقط كل شيء حتى KeyboardInterrupt!
    pass  # يُخفي الخطأ تماماً دون أي أثر!

# ────────────────────────────────────────────────────────────────────────
# [033] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-11-chapter-11.ipynb#cell-32
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.3: الصيغة الأساسية: try / except

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-009-value.py
"""

# ترتيب except مهم جداً! الأكثر تحديداً أولاً
try:
    value = my_list[10]
except IndexError:
    print("فهرس خارج النطاق")
except LookupError:  # تأتي بعد IndexError لأنها أعم منها
    print("خطأ بحث عام")

# ────────────────────────────────────────────────────────────────────────
# [034] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-11-chapter-11.ipynb#cell-33
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.3: الصيغة الأساسية: try / except

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-010-ex11_3.py
"""

try:
    x = 1 / 0
except Exception:
    print("خطأ عام")  # هذا ما سيُطبع
except ZeroDivisionError:  # ⚠️ ميت! لن يُنفَّذ أبداً
    print("قسمة على صفر")

# ────────────────────────────────────────────────────────────────────────
# [035] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-11-chapter-11.ipynb#cell-34
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.4: else و finally: إكمال الصورة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-011-ex11_4.py
"""

try:
    # الكود الذي قد يُطلق خطأ
    pass
except SomeError:
    # يُنفَّذ فقط إذا حدث هذا الخطأ المحدد
    pass
else:
    # يُنفَّذ فقط إذا لم يحدث أي خطأ في try
    pass
finally:
    # يُنفَّذ دائماً، سواء حدث خطأ أم لا
    pass

# ────────────────────────────────────────────────────────────────────────
# [036] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-11-chapter-11.ipynb#cell-35
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.4: else و finally: إكمال الصورة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-012-read_and_process_number.py
"""

def read_and_process_number(text: str) -> None:
    try:
        number = int(text)
    except ValueError:
        print(f"❌ '{text}' ليس رقماً صحيحاً.")
    else:
        # يُنفَّذ فقط إذا نجح التحويل بدون خطأ
        print(f"✅ تم تحويل الرقم بنجاح: {number}")
        print(f"   ضعفه: {number * 2}")
    finally:
        # يُنفَّذ دائماً - مثالي للتنظيف
        print("--- انتهت المحاولة ---")

read_and_process_number("42")
print()
read_and_process_number("abc")

# ────────────────────────────────────────────────────────────────────────
# [037] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-11-chapter-11.ipynb#cell-36
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.4: else و finally: إكمال الصورة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-013-number.py
"""

# أقل وضوحاً - أي سطر بالضبط قد يُطلق الخطأ؟
try:
    number = int(text)
    print(f"ضعفه: {number * 2}")  # ماذا لو هذا السطر نفسه فشل لسبب آخر؟
except ValueError:
    print("خطأ")

# أوضح بكثير - try تحتوي فقط على الكود المحتمل فشله
try:
    number = int(text)
except ValueError:
    print("خطأ")
else:
    print(f"ضعفه: {number * 2}")  # نعلم بالتأكيد أن number تم تحويله بنجاح

# ────────────────────────────────────────────────────────────────────────
# [038] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-11-chapter-11.ipynb#cell-37
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.4: else و finally: إكمال الصورة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-014-process_file.py
"""

def process_file(filename: str) -> None:
    file = None
    try:
        file = open(filename, "r", encoding="utf-8")
        content = file.read()
        print(f"تم قراءة {len(content)} حرف.")
    except FileNotFoundError:
        print(f"❌ الملف '{filename}' غير موجود.")
    finally:
        if file is not None:
            file.close()
            print("✅ تم إغلاق الملف.")  # يحدث دائماً!

process_file("not_exist.txt")

# ────────────────────────────────────────────────────────────────────────
# [039] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-11-chapter-11.ipynb#cell-38
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.5: إطلاق الأخطاء بنفسك: raise

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-015-withdraw.py
"""

def withdraw(balance: float, amount: float) -> float:
    if amount <= 0:
        raise ValueError("مبلغ السحب يجب أن يكون موجباً.")
    if amount > balance:
        raise ValueError("الرصيد غير كافٍ لإتمام العملية.")
    return balance - amount

try:
    new_balance = withdraw(1000, 1500)
except ValueError as e:
    print(f"❌ فشلت العملية: {e}")

# ────────────────────────────────────────────────────────────────────────
# [040] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-11-chapter-11.ipynb#cell-39
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.5: إطلاق الأخطاء بنفسك: raise

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-016-process_data.py
"""

def process_data(data: list) -> float:
    try:
        return sum(data) / len(data)
    except ZeroDivisionError:
        print("⚠️ تحذير: القائمة فارغة، سيتم إعادة إطلاق الخطأ.")
        raise  # raise بدون معامل = إعادة إطلاق نفس الخطأ الملتقط

try:
    process_data([])
except ZeroDivisionError:
    print("❌ المستوى الأعلى: تم التعامل مع القائمة الفارغة.")

# ────────────────────────────────────────────────────────────────────────
# [041] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-11-chapter-11.ipynb#cell-40
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.5: إطلاق الأخطاء بنفسك: raise

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-017-parse_age.py
"""

def parse_age(text: str) -> int:
    try:
        return int(text)
    except ValueError as e:
        raise ValueError(f"فشل تحليل العمر من النص: '{text}'") from e

try:
    parse_age("twenty")
except ValueError as e:
    print(e)
    print(f"السبب الأصلي: {e.__cause__}")

# ────────────────────────────────────────────────────────────────────────
# [042] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-11-chapter-11.ipynb#cell-41
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.6: إنشاء استثناءات مخصصة (Custom Exceptions)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-018-withdraw.py
"""

class InsufficientFundsError(Exception):
    """تُطلق عندما يحاول المستخدم سحب مبلغ أكبر من رصيده."""
    pass

class InvalidAmountError(Exception):
    """تُطلق عندما يكون المبلغ المُدخل غير منطقي (سالب أو صفر)."""
    pass

def withdraw(balance: float, amount: float) -> float:
    if amount <= 0:
        raise InvalidAmountError(f"المبلغ {amount} غير صالح؛ يجب أن يكون موجباً.")
    if amount > balance:
        raise InsufficientFundsError(
            f"طلبت سحب {amount}، لكن رصيدك الحالي {balance} فقط."
        )
    return balance - amount

try:
    withdraw(500, 1000)
except InsufficientFundsError as e:
    print(f"💰 خطأ في الرصيد: {e}")
except InvalidAmountError as e:
    print(f"🔢 خطأ في المبلغ: {e}")

# ────────────────────────────────────────────────────────────────────────
# [043] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-11-chapter-11.ipynb#cell-42
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.6: إنشاء استثناءات مخصصة (Custom Exceptions)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-019-process_transaction.py
"""

class BankError(Exception):
    """الفئة الأساسية لكل أخطاء النظام البنكي."""
    pass

class InsufficientFundsError(BankError):
    """رصيد غير كافٍ."""
    pass

class InvalidAmountError(BankError):
    """مبلغ غير صالح."""
    pass

class AccountFrozenError(BankError):
    """الحساب مجمّد."""
    pass

def process_transaction(account_status: str, balance: float, amount: float) -> float:
    if account_status == "frozen":
        raise AccountFrozenError("هذا الحساب مجمّد حالياً.")
    if amount <= 0:
        raise InvalidAmountError("المبلغ يجب أن يكون موجباً.")
    if amount > balance:
        raise InsufficientFundsError("الرصيد غير كافٍ.")
    return balance - amount

try:
    process_transaction("frozen", 1000, 100)
except BankError as e:  # التقاط أي خطأ بنكي بغض النظر عن نوعه الدقيق
    print(f"🏦 خطأ في النظام البنكي: {type(e).__name__}: {e}")

# ────────────────────────────────────────────────────────────────────────
# [044] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-11-chapter-11.ipynb#cell-43
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.6: إنشاء استثناءات مخصصة (Custom Exceptions)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-020-__init__.py
"""

class ValidationError(Exception):
    """خطأ تحقق من صحة البيانات، مع تفاصيل الحقل المُخالف."""

    def __init__(self, field: str, message: str):
        self.field = field
        self.message = message
        super().__init__(f"خطأ في حقل '{field}': {message}")

def validate_age(age: int) -> None:
    if age < 0 or age > 150:
        raise ValidationError(field="age", message="يجب أن يكون بين 0 و150")

try:
    validate_age(-5)
except ValidationError as e:
    print(f"الحقل المُخالف: {e.field}")
    print(f"الرسالة: {e.message}")

# ────────────────────────────────────────────────────────────────────────
# [045] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-11-chapter-11.ipynb#cell-44
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.7: فلسفتان في البرمجة: EAFP مقابل LBYL

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-021-content.py
"""

# تحقق أولاً، ثم نفّذ
import os

if os.path.exists("data.txt"):
    with open("data.txt") as f:
        content = f.read()
else:
    print("الملف غير موجود")

# ────────────────────────────────────────────────────────────────────────
# [046] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-11-chapter-11.ipynb#cell-45
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.7: فلسفتان في البرمجة: EAFP مقابل LBYL

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-022-content.py
"""

# نفّذ مباشرة، وتعامل مع الخطأ إن حدث
try:
    with open("data.txt") as f:
        content = f.read()
except FileNotFoundError:
    print("الملف غير موجود")

# ────────────────────────────────────────────────────────────────────────
# [047] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-11-chapter-11.ipynb#cell-46
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.8: دالة assert: التحقق أثناء التطوير

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-023-calculate_average.py
"""

def calculate_average(numbers: list[float]) -> float:
    assert len(numbers) > 0, "القائمة يجب ألا تكون فارغة"
    return sum(numbers) / len(numbers)

print(calculate_average([10, 20, 30]))  # 20.0
# print(calculate_average([]))
# 💥 AssertionError: القائمة يجب ألا تكون فارغة

# ────────────────────────────────────────────────────────────────────────
# [048] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-11-chapter-11.ipynb#cell-47
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.8: دالة assert: التحقق أثناء التطوير

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-024-set_age.py
"""

# ❌ خطأ: استخدام assert للتحقق من مدخلات مستخدم
def set_age(age: int) -> None:
    assert age >= 0, "العمر يجب أن يكون موجباً"  # قد يُحذف بالكامل!

# ✅ صحيح: استخدام raise للتحقق من مدخلات مستخدم
def set_age(age: int) -> None:
    if age < 0:
        raise ValueError("العمر يجب أن يكون موجباً")

# ────────────────────────────────────────────────────────────────────────
# [049] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-11-chapter-11.ipynb#cell-48
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.9: مقدمة مختصرة لـ logging بدلاً من print

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-025-divide.py
"""

import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def divide(a: float, b: float) -> float | None:
    try:
        return a / b
    except ZeroDivisionError:
        logging.error(f"محاولة قسمة {a} على صفر!")
        return None

divide(10, 0)

# ────────────────────────────────────────────────────────────────────────
# [050] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-11-chapter-11.ipynb#cell-49
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.10: مشروع تطبيقي: نظام تسجيل دخول آمن

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-026-register.py
"""

class AuthError(Exception):
    """الفئة الأساسية لكل أخطاء المصادقة."""
    pass

class UserNotFoundError(AuthError):
    """المستخدم غير موجود في النظام."""
    pass

class IncorrectPasswordError(AuthError):
    """كلمة المرور غير صحيحة."""
    pass

class AccountLockedError(AuthError):
    """الحساب مقفل بسبب محاولات فاشلة متكررة."""
    pass

class WeakPasswordError(AuthError):
    """كلمة مرور ضعيفة عند التسجيل."""
    pass

# قاعدة بيانات وهمية (في الواقع ستكون قاعدة بيانات حقيقية)
users_db: dict[str, dict] = {
    "ahmed@test.com": {"password": "Secure123", "failed_attempts": 0, "locked": False},
}

MAX_FAILED_ATTEMPTS = 3

def register(email: str, password: str) -> None:
    """تسجيل مستخدم جديد مع التحقق من قوة كلمة المرور."""
    if len(password) < 8:
        raise WeakPasswordError("كلمة المرور يجب أن تكون 8 أحرف على الأقل.")
    if not any(char.isdigit() for char in password):
        raise WeakPasswordError("كلمة المرور يجب أن تحتوي رقماً واحداً على الأقل.")

    if email in users_db:
        raise ValueError("هذا البريد الإلكتروني مسجّل مسبقاً.")

    users_db[email] = {"password": password, "failed_attempts": 0, "locked": False}
    print(f"✅ تم تسجيل المستخدم {email} بنجاح.")

def login(email: str, password: str) -> bool:
    """محاولة تسجيل دخول، مع حماية من المحاولات المتكررة."""
    try:
        user = users_db[email]
    except KeyError:
        raise UserNotFoundError(f"لا يوجد مستخدم بالبريد: {email}")

    if user["locked"]:
        raise AccountLockedError("هذا الحساب مقفل. تواصل مع الدعم الفني.")

    if user["password"] != password:
        user["failed_attempts"] += 1
        if user["failed_attempts"] >= MAX_FAILED_ATTEMPTS:
            user["locked"] = True
            raise AccountLockedError(
                "تم قفل الحساب بسبب تجاوز عدد المحاولات الفاشلة المسموحة."
            )
        remaining = MAX_FAILED_ATTEMPTS - user["failed_attempts"]
        raise IncorrectPasswordError(
            f"كلمة المرور خاطئة. المحاولات المتبقية: {remaining}"
        )

    user["failed_attempts"] = 0  # إعادة تصفير عند النجاح
    return True

def run_login_attempt(email: str, password: str) -> None:
    """واجهة تفاعلية تلتقط كل أنواع أخطاء المصادقة بوضوح."""
    try:
        login(email, password)
    except UserNotFoundError as e:
        print(f"👤 {e}")
    except AccountLockedError as e:
        print(f"🔒 {e}")
    except IncorrectPasswordError as e:
        print(f"🔑 {e}")
    except AuthError as e:
        print(f"⚠️ خطأ مصادقة غير متوقع: {e}")
    else:
        print(f"✅ تم تسجيل الدخول بنجاح! أهلاً بك {email}")
    finally:
        print("-" * 40)

# --- سيناريوهات تجريبية ---
if __name__ == "__main__":
    try:
        register("sara@test.com", "weak")
    except WeakPasswordError as e:
        print(f"❌ فشل التسجيل: {e}")
    print("-" * 40)

    register("sara@test.com", "Strong123")
    print("-" * 40)

    run_login_attempt("unknown@test.com", "anything")   # UserNotFoundError
    run_login_attempt("ahmed@test.com", "wrong_pass")   # IncorrectPasswordError (محاولة 1)
    run_login_attempt("ahmed@test.com", "wrong_pass")   # IncorrectPasswordError (محاولة 2)
    run_login_attempt("ahmed@test.com", "wrong_pass")   # AccountLockedError (محاولة 3 - قفل!)
    run_login_attempt("ahmed@test.com", "Secure123")    # AccountLockedError (مقفل مسبقاً)
    run_login_attempt("sara@test.com", "Strong123")     # ✅ نجاح

# ────────────────────────────────────────────────────────────────────────
# [051] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-11-chapter-11.ipynb#cell-50
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.10: مشروع تطبيقي: نظام تسجيل دخول آمن

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-027-reset_password.py
"""

def reset_password(email: str, new_password: str) -> None:
    """إعادة تعيين كلمة مرور، مع إلغاء القفل إن وُجد."""
    try:
        user = users_db[email]
    except KeyError:
        raise UserNotFoundError(f"لا يوجد مستخدم بالبريد: {email}")

    if len(new_password) < 8 or not any(c.isdigit() for c in new_password):
        raise WeakPasswordError("كلمة المرور الجديدة ضعيفة جداً.")

    user["password"] = new_password
    user["failed_attempts"] = 0
    user["locked"] = False
    print(f"🔓 تم إعادة تعيين كلمة المرور وفك قفل الحساب لـ {email}")

try:
    reset_password("ahmed@test.com", "NewPass123")
except AuthError as e:
    print(f"❌ {e}")

# ────────────────────────────────────────────────────────────────────────
# [052] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-11-chapter-11.ipynb#cell-51
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.11: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-028-safe_list_access.py
"""

# تمرين 1
try:
    num = int(input("أدخل رقماً: "))
    print("رقم صحيح")
except ValueError:
    print("خطأ: ليس رقماً")

# تمرين 2
def safe_list_access(lst: list, index: int):
    try:
        return lst[index]
    except IndexError:
        return None

print(safe_list_access([1, 2, 3], 1))   # 2
print(safe_list_access([1, 2, 3], 10))  # None

# ────────────────────────────────────────────────────────────────────────
# [053] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-11-chapter-11.ipynb#cell-52
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.11: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-029-calculate_square_root.py
"""

import math

# تمرين 3
class NegativeValueError(Exception):
    pass

def calculate_square_root(n: float) -> float:
    if n < 0:
        raise NegativeValueError(f"لا يمكن حساب جذر رقم سالب: {n}")
    return math.sqrt(n)

try:
    print(calculate_square_root(-9))
except NegativeValueError as e:
    print(f"❌ {e}")

# تمرين 4
class MissingConfigError(Exception):
    def __init__(self, key: str):
        self.key = key
        super().__init__(f"الإعداد المطلوب '{key}' غير موجود.")

def parse_config(data: dict) -> tuple[str, int]:
    try:
        host = data["host"]
    except KeyError:
        raise MissingConfigError("host")

    try:
        port = data["port"]
    except KeyError:
        raise MissingConfigError("port")

    return host, port

try:
    parse_config({"host": "localhost"})
except MissingConfigError as e:
    print(f"❌ {e}")  # ❌ الإعداد المطلوب 'port' غير موجود.

# ────────────────────────────────────────────────────────────────────────
# [054] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-11-chapter-11.ipynb#cell-53
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.11: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-030-__init__.py
"""

class APIError(Exception):
    """الفئة الأساسية لكل أخطاء API."""
    pass

class EndpointNotFoundError(APIError):
    def __init__(self, endpoint: str):
        self.endpoint = endpoint
        super().__init__(f"المسار '{endpoint}' غير موجود.")

class MissingFieldError(APIError):
    def __init__(self, field: str):
        self.field = field
        super().__init__(f"الحقل المطلوب '{field}' مفقود.")

class InvalidFieldError(APIError):
    def __init__(self, field: str, reason: str):
        self.field = field
        self.reason = reason
        super().__init__(f"الحقل '{field}' غير صالح: {reason}")

VALID_ENDPOINTS = {"/users", "/orders", "/products"}

def process_api_request(endpoint: str, data: dict) -> dict:
    if endpoint not in VALID_ENDPOINTS:
        raise EndpointNotFoundError(endpoint)

    if "user_id" not in data:
        raise MissingFieldError("user_id")

    user_id = data["user_id"]
    if not isinstance(user_id, int) or user_id <= 0:
        raise InvalidFieldError("user_id", "يجب أن يكون رقماً صحيحاً موجباً")

    return {"status": "success", "endpoint": endpoint, "user_id": user_id}

def handle_request(endpoint: str, data: dict) -> None:
    try:
        result = process_api_request(endpoint, data)
    except EndpointNotFoundError as e:
        print(f"404 - {e}")
    except MissingFieldError as e:
        print(f"400 - حقل مفقود: {e}")
    except InvalidFieldError as e:
        print(f"400 - حقل غير صالح: {e}")
    except APIError as e:
        print(f"500 - خطأ API غير متوقع: {e}")
    else:
        print(f"200 - {result}")

handle_request("/comments", {"user_id": 1})         # 404
handle_request("/users", {})                         # 400 مفقود
handle_request("/users", {"user_id": -5})            # 400 غير صالح
handle_request("/users", {"user_id": 42})            # 200 نجاح

# ────────────────────────────────────────────────────────────────────────
# [055] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-11-chapter-11.ipynb#cell-54
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.13: ملخص الفصل (Cheat Sheet)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-031-__init__.py
"""

# الصيغة الكاملة
try:
    risky_code()
except SpecificError as e:
    handle_specific_case(e)
except (ErrorA, ErrorB) as e:
    handle_multiple_cases(e)
except Exception as e:      # الأعم - دائماً في الأخير
    handle_general_case(e)
else:
    runs_if_no_error()
finally:
    always_runs()            # تنظيف الموارد

# إطلاق خطأ
raise ValueError("رسالة واضحة")

# إعادة إطلاق
except SomeError:
    raise  # بدون معامل

# تسلسل الاستثناءات
raise NewError("...") from original_error

# استثناء مخصص
class MyAppError(Exception):
    pass

class SpecificError(MyAppError):
    def __init__(self, detail: str):
        self.detail = detail
        super().__init__(f"فشل بسبب: {detail}")

# assert - للتطوير فقط، ليس لمدخلات المستخدم
assert condition, "رسالة إذا فشل الشرط"

# ────────────────────────────────────────────────────────────────────────
# [056] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-11-chapter-11.ipynb#cell-55
# ────────────────────────────────────────────────────────────────────────
balance = 1000
withdrawal = input("أدخل مبلغ السحب: ")  # المستخدم يكتب "مئة"
balance -= int(withdrawal)  # 💥 ValueError: invalid literal for int()
# البرنامج ينهار، والمستخدم يرى رسالة تقنية غير مفهومة

# ────────────────────────────────────────────────────────────────────────
# [057] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-11-chapter-11.ipynb#cell-56
# ────────────────────────────────────────────────────────────────────────
print(10 / 0)        # ZeroDivisionError
print(int("abc"))    # ValueError
print(my_list[99])   # IndexError
print(my_dict["x"])  # KeyError
print(undefined_var) # NameError

# ────────────────────────────────────────────────────────────────────────
# [058] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-11-chapter-11.ipynb#cell-57
# ────────────────────────────────────────────────────────────────────────
def divide(a, b):
    return a / b

result = divide(10, 0)

# ────────────────────────────────────────────────────────────────────────
# [059] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-11-chapter-11.ipynb#cell-58
# ────────────────────────────────────────────────────────────────────────
try:
    number = int(input("أدخل رقماً: "))
    print(f"ضعف الرقم: {number * 2}")
except ValueError:
    print("❌ خطأ: الرجاء إدخال رقم صحيح فقط.")

print("البرنامج استمر في العمل!")

# ────────────────────────────────────────────────────────────────────────
# [060] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-11-chapter-11.ipynb#cell-59
# ────────────────────────────────────────────────────────────────────────
def safe_divide(a: float, b: float) -> float | None:
    try:
        return a / b
    except ZeroDivisionError:
        print("❌ لا يمكن القسمة على صفر.")
        return None
    except TypeError:
        print("❌ يجب أن تكون القيم أرقاماً.")
        return None

print(safe_divide(10, 0))        # ❌ لا يمكن القسمة على صفر. → None
print(safe_divide(10, "abc"))    # ❌ يجب أن تكون القيم أرقاماً. → None
print(safe_divide(10, 2))        # 5.0

# ────────────────────────────────────────────────────────────────────────
# [061] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-11-chapter-11.ipynb#cell-60
# ────────────────────────────────────────────────────────────────────────
try:
    value = my_dict[key]
    result = 100 / value
except (KeyError, ZeroDivisionError) as e:
    print(f"حدث خطأ: {e}")

# ────────────────────────────────────────────────────────────────────────
# [062] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-11-chapter-11.ipynb#cell-61
# ────────────────────────────────────────────────────────────────────────
try:
    risky_operation()
except Exception as e:
    print(f"حدث خطأ غير متوقع: {e}")

# ────────────────────────────────────────────────────────────────────────
# [063] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-11-chapter-11.ipynb#cell-62
# ────────────────────────────────────────────────────────────────────────
# ❌ الأسوأ على الإطلاق - لا تفعل هذا أبداً!
try:
    risky_operation()
except:  # bare except - يلتقط كل شيء حتى KeyboardInterrupt!
    pass  # يُخفي الخطأ تماماً دون أي أثر!

# ────────────────────────────────────────────────────────────────────────
# [064] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-11-chapter-11.ipynb#cell-63
# ────────────────────────────────────────────────────────────────────────
# ترتيب except مهم جداً! الأكثر تحديداً أولاً
try:
    value = my_list[10]
except IndexError:
    print("فهرس خارج النطاق")
except LookupError:  # تأتي بعد IndexError لأنها أعم منها
    print("خطأ بحث عام")

# ────────────────────────────────────────────────────────────────────────
# [065] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-11-chapter-11.ipynb#cell-64
# ────────────────────────────────────────────────────────────────────────
try:
    # الكود الذي قد يُطلق خطأ
    pass
except SomeError:
    # يُنفَّذ فقط إذا حدث هذا الخطأ المحدد
    pass
else:
    # يُنفَّذ فقط إذا لم يحدث أي خطأ في try
    pass
finally:
    # يُنفَّذ دائماً، سواء حدث خطأ أم لا
    pass

# ────────────────────────────────────────────────────────────────────────
# [066] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-11-chapter-11.ipynb#cell-65
# ────────────────────────────────────────────────────────────────────────
# أقل وضوحاً - أي سطر بالضبط قد يُطلق الخطأ؟
try:
    number = int(text)
    print(f"ضعفه: {number * 2}")  # ماذا لو هذا السطر نفسه فشل لسبب آخر؟
except ValueError:
    print("خطأ")

# أوضح بكثير - try تحتوي فقط على الكود المحتمل فشله
try:
    number = int(text)
except ValueError:
    print("خطأ")
else:
    print(f"ضعفه: {number * 2}")  # نعلم بالتأكيد أن number تم تحويله بنجاح

# ────────────────────────────────────────────────────────────────────────
# [067] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-11-chapter-11.ipynb#cell-66
# ────────────────────────────────────────────────────────────────────────
def calculate_average(numbers: list[float]) -> float:
    assert len(numbers) > 0, "القائمة يجب ألا تكون فارغة"
    return sum(numbers) / len(numbers)

print(calculate_average([10, 20, 30]))  # 20.0
# print(calculate_average([]))
# 💥 AssertionError: القائمة يجب ألا تكون فارغة

# ────────────────────────────────────────────────────────────────────────
# [068] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-11-chapter-11.ipynb#cell-67
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: استخدام assert للتحقق من مدخلات مستخدم
def set_age(age: int) -> None:
    assert age >= 0, "العمر يجب أن يكون موجباً"  # قد يُحذف بالكامل!

# ✅ صحيح: استخدام raise للتحقق من مدخلات مستخدم
def set_age(age: int) -> None:
    if age < 0:
        raise ValueError("العمر يجب أن يكون موجباً")

# ────────────────────────────────────────────────────────────────────────
# [069] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-11-chapter-11.ipynb#cell-68
# ────────────────────────────────────────────────────────────────────────
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def divide(a: float, b: float) -> float | None:
    try:
        return a / b
    except ZeroDivisionError:
        logging.error(f"محاولة قسمة {a} على صفر!")
        return None

divide(10, 0)

# ────────────────────────────────────────────────────────────────────────
# [070] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-11-chapter-11.ipynb#cell-69
# ────────────────────────────────────────────────────────────────────────
# تمرين 1
try:
    num = int(input("أدخل رقماً: "))
    print("رقم صحيح")
except ValueError:
    print("خطأ: ليس رقماً")

# تمرين 2
def safe_list_access(lst: list, index: int):
    try:
        return lst[index]
    except IndexError:
        return None

print(safe_list_access([1, 2, 3], 1))   # 2
print(safe_list_access([1, 2, 3], 10))  # None

# ────────────────────────────────────────────────────────────────────────
# [071] maghzbook-download/python-3-14-from-zero-to-pro/notebooks/ch-11-chapter-11.ipynb#cell-70
# ────────────────────────────────────────────────────────────────────────
# الصيغة الكاملة
try:
    risky_code()
except SpecificError as e:
    handle_specific_case(e)
except (ErrorA, ErrorB) as e:
    handle_multiple_cases(e)
except Exception as e:      # الأعم - دائماً في الأخير
    handle_general_case(e)
else:
    runs_if_no_error()
finally:
    always_runs()            # تنظيف الموارد

# إطلاق خطأ
raise ValueError("رسالة واضحة")

# إعادة إطلاق
except SomeError:
    raise  # بدون معامل

# تسلسل الاستثناءات
raise NewError("...") from original_error

# استثناء مخصص
class MyAppError(Exception):
    pass

class SpecificError(MyAppError):
    def __init__(self, detail: str):
        self.detail = detail
        super().__init__(f"فشل بسبب: {detail}")

# assert - للتطوير فقط، ليس لمدخلات المستخدم
assert condition, "رسالة إذا فشل الشرط"
