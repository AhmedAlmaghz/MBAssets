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
