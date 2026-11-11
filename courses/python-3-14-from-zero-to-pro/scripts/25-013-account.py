#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس والعشرون: الاختبار (Testing) باستخدام pytest
القسم 25.7: اختبار الأصناف (Classes) والكود الكائني

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 25-013-account.py
"""

# tests/test_bank_account.py
import pytest
from bank_account import BankAccount, InsufficientFundsError

@pytest.fixture
def account():
    """حساب بنكي تجريبي برصيد ابتدائي 100."""
    return BankAccount(owner="أحمد", balance=100)

class TestDeposit:
    """تجميع كل اختبارات عملية الإيداع في صنف واحد."""

    def test_deposit_increases_balance(self, account):
        account.deposit(50)
        assert account.balance == 150

    def test_deposit_negative_amount_raises_error(self, account):
        with pytest.raises(ValueError):
            account.deposit(-10)

    def test_deposit_zero_raises_error(self, account):
        with pytest.raises(ValueError):
            account.deposit(0)

class TestWithdraw:
    """تجميع كل اختبارات عملية السحب في صنف واحد."""

    def test_withdraw_decreases_balance(self, account):
        account.withdraw(30)
        assert account.balance == 70

    def test_withdraw_more_than_balance_raises_error(self, account):
        with pytest.raises(InsufficientFundsError):
            account.withdraw(200)

    def test_withdraw_exact_balance_succeeds(self, account):
        account.withdraw(100)
        assert account.balance == 0
