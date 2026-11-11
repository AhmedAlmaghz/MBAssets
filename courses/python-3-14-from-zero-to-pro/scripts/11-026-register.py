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
