#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.5: مشروع تطبيقي: نظام إدارة مخزون بسيط

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-025-add_product.py
"""

inventory: dict[str, dict[str, object]] = {}

def add_product(name: str, price: float, quantity: int, category: str) -> None:
    """إضافة منتج جديد إلى المخزون."""
    if name in inventory:
        print(f"⚠️  المنتج '{name}' موجود مسبقاً. استخدم update_quantity للتعديل.")
        return
    inventory[name] = {
        "price": price,
        "quantity": quantity,
        "category": category
    }
    print(f"✅ تمت إضافة '{name}' بنجاح.")

def update_quantity(name: str, change: int) -> None:
    """تحديث كمية منتج (موجب للشراء، سالب للبيع)."""
    if name not in inventory:
        print(f"❌ المنتج '{name}' غير موجود.")
        return

    new_quantity = inventory[name]["quantity"] + change
    if new_quantity < 0:
        print(f"❌ لا توجد كمية كافية من '{name}'.")
        return

    inventory[name]["quantity"] = new_quantity
    print(f"✅ كمية '{name}' الآن: {new_quantity}")

def low_stock_alert(threshold: int = 5) -> list[str]:
    """إرجاع قائمة بأسماء المنتجات التي كميتها أقل من الحد المسموح."""
    low_items = [
        name for name, info in inventory.items()
        if info["quantity"] < threshold
    ]
    return low_items

def unique_categories() -> set[str]:
    """إرجاع كل الفئات الفريدة الموجودة في المخزون."""
    return {info["category"] for info in inventory.values()}

def total_inventory_value() -> float:
    """حساب القيمة الإجمالية لكل المخزون (السعر × الكمية)."""
    total = sum(
        info["price"] * info["quantity"]
        for info in inventory.values()
    )
    return total

def print_report() -> None:
    """طباعة تقرير شامل عن حالة المخزون."""
    print("\n" + "=" * 40)
    print("📊 تقرير المخزون")
    print("=" * 40)

    for name, info in inventory.items():
        print(f"- {name}: {info['quantity']} قطعة | "
              f"{info['price']:.2f} ر.س | الفئة: {info['category']}")

    print("-" * 40)
    print(f"💰 القيمة الإجمالية: {total_inventory_value():,.2f} ر.س")
    print(f"🏷️  الفئات المتاحة: {', '.join(unique_categories())}")

    low_items = low_stock_alert(threshold=5)
    if low_items:
        print(f"⚠️  تنبيه نفاد المخزون: {', '.join(low_items)}")
    else:
        print("✅ لا توجد منتجات على وشك النفاد.")
    print("=" * 40)

# --- تشغيل تجريبي للنظام ---
if __name__ == "__main__":
    add_product("لابتوب", 3500.0, 10, "إلكترونيات")
    add_product("ماوس", 50.0, 3, "إلكترونيات")
    add_product("كرسي مكتب", 450.0, 8, "أثاث")
    add_product("طاولة", 700.0, 2, "أثاث")

    update_quantity("لابتوب", -2)   # بيع لابتوبين
    update_quantity("ماوس", 10)     # شراء 10 ماوسات جديدة

    print_report()
