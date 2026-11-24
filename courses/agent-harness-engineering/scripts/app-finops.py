#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
هندسة وكلاء الذكاء الاصطناعي — المعادلات الهندسية وحاسبة اقتصاديات التوكنات
AI Agent Engineering — FinOps & Capacity Planning

Book slug : agent-harness-engineering
Chapter   : app-finops (lesson finops)
Code files: 3 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/agent-harness-engineering/code/app-finops.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
هندسة وكلاء الذكاء الاصطناعي — المعادلات الهندسية وحاسبة اقتصاديات التوكنات
AI Agent Engineering — FinOps & Capacity Planning

Book slug : agent-harness-engineering
Chapter   : app-finops (lesson finops)
Code files: 2 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/agent-harness-engineering/code/app-finops.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
هندسة وكلاء الذكاء الاصطناعي — المعادلات الهندسية وحاسبة اقتصاديات التوكنات
AI Agent Engineering — FinOps & Capacity Planning

Book slug : agent-harness-engineering
Chapter   : app-finops (lesson finops)
Code files: 1 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] dist/md/app-finops.md
# ────────────────────────────────────────────────────────────────────────
from pydantic import BaseModel, ConfigDict, Field


class AgentWorkloadProfile(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    system_prefix_tokens: int = Field(default=2500, ge=100)
    avg_observation_tokens_per_turn: int = Field(default=800, ge=50)
    avg_output_tokens_per_turn: int = Field(default=220, ge=10)
    avg_turns_per_run: int = Field(default=5, ge=1)
    concurrent_users: int = Field(default=50, ge=1)
    avg_run_duration_sec: float = Field(default=12.0, gt=0.5)
    daily_total_runs: int = Field(default=10_000, ge=1)

    # التسعير لكل مليون توكن (مثال: نموذج رائد متوسط التكلفة)
    input_price_per_1m_usd: float = Field(default=3.00, gt=0.0)
    output_price_per_1m_usd: float = Field(default=15.00, gt=0.0)
    cache_discount_ratio: float = Field(default=0.90, ge=0.0, le=1.0)


class FinOpsCapacityReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    total_input_tokens_per_run: int
    total_output_tokens_per_run: int
    cost_per_run_no_cache_usd: float
    cost_per_run_with_80pct_cache_usd: float
    monthly_cost_no_cache_usd: float
    monthly_cost_with_cache_usd: float
    monthly_savings_usd: float
    required_provider_rpm_quota: int
    required_provider_tpm_quota: int


def calculate_finops_and_capacity(
    profile: AgentWorkloadProfile,
    expected_cache_hit_ratio: float = 0.80,
    safety_headroom: float = 1.30,
) -> FinOpsCapacityReport:
    s = profile.avg_turns_per_run
    # حساب التوكنات التراكمية عبر S دورات
    total_in = (s * profile.system_prefix_tokens) + (
        (s * (s - 1)) // 2
    ) * profile.avg_observation_tokens_per_turn
    total_out = s * profile.avg_output_tokens_per_turn

    in_unit = profile.input_price_per_1m_usd / 1_000_000
    out_unit = profile.output_price_per_1m_usd / 1_000_000

    # التكلفة بدون KV-Cache
    cost_no_cache = (total_in * in_unit) + (total_out * out_unit)

    # التكلفة مع KV-Cache (Chapter 5 Prefix Engineering)
    effective_in_multiplier = (1.0 - expected_cache_hit_ratio) + (
        expected_cache_hit_ratio * (1.0 - profile.cache_discount_ratio)
    )
    cost_with_cache = (total_in * in_unit * effective_in_multiplier) + (
        total_out * out_unit
    )

    monthly_runs = profile.daily_total_runs * 30
    monthly_no_cache = cost_no_cache * monthly_runs
    monthly_with_cache = cost_with_cache * monthly_runs

    # حساب السعة المطلوبة من المزود في الدقيقة (RPM & TPM)
    runs_per_minute_at_peak = profile.concurrent_users * (
        60.0 / profile.avg_run_duration_sec
    )
    req_rpm = int(runs_per_minute_at_peak * s * safety_headroom)
    req_tpm = int(
        runs_per_minute_at_peak * (total_in + total_out) * safety_headroom
    )

    return FinOpsCapacityReport(
        total_input_tokens_per_run=total_in,
        total_output_tokens_per_run=total_out,
        cost_per_run_no_cache_usd=round(cost_no_cache, 5),
        cost_per_run_with_80pct_cache_usd=round(cost_with_cache, 5),
        monthly_cost_no_cache_usd=round(monthly_no_cache, 2),
        monthly_cost_with_cache_usd=round(monthly_with_cache, 2),
        monthly_savings_usd=round(monthly_no_cache - monthly_with_cache, 2),
        required_provider_rpm_quota=req_rpm,
        required_provider_tpm_quota=req_tpm,
    )


if __name__ == "__main__":
    workload = AgentWorkloadProfile()
    report = calculate_finops_and_capacity(workload)
    print("=" * 75)
    print("📊 تقرير حاسبة FinOps وتخطيط السعة (10,000 تشغيل يومياً / 5 دورات للتشغيل)")
    print("=" * 75)
    print(f"• إجمالي توكنات الإدخال للتشغيل الواحد : {report.total_input_tokens_per_run:,} tokens")
    print(f"• التكلفة الشهرية (بدون هندسة الكاش)   : ${report.monthly_cost_no_cache_usd:,.2f} / month")
    print(f"• التكلفة الشهرية (مع 80% KV-Cache)    : ${report.monthly_cost_with_cache_usd:,.2f} / month")
    print(f"💰 الوفر المالي الشهري المباشر         : ${report.monthly_savings_usd:,.2f} / month!")
    print(f"⚡ حصة المزود المطلوبة في الذروة       : {report.required_provider_rpm_quota:,} RPM | {report.required_provider_tpm_quota:,} TPM")

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/agent-harness-engineering/notebooks/app-finops.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
from pydantic import BaseModel, ConfigDict, Field


class AgentWorkloadProfile(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    system_prefix_tokens: int = Field(default=2500, ge=100)
    avg_observation_tokens_per_turn: int = Field(default=800, ge=50)
    avg_output_tokens_per_turn: int = Field(default=220, ge=10)
    avg_turns_per_run: int = Field(default=5, ge=1)
    concurrent_users: int = Field(default=50, ge=1)
    avg_run_duration_sec: float = Field(default=12.0, gt=0.5)
    daily_total_runs: int = Field(default=10_000, ge=1)

    # التسعير لكل مليون توكن (مثال: نموذج رائد متوسط التكلفة)
    input_price_per_1m_usd: float = Field(default=3.00, gt=0.0)
    output_price_per_1m_usd: float = Field(default=15.00, gt=0.0)
    cache_discount_ratio: float = Field(default=0.90, ge=0.0, le=1.0)


class FinOpsCapacityReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    total_input_tokens_per_run: int
    total_output_tokens_per_run: int
    cost_per_run_no_cache_usd: float
    cost_per_run_with_80pct_cache_usd: float
    monthly_cost_no_cache_usd: float
    monthly_cost_with_cache_usd: float
    monthly_savings_usd: float
    required_provider_rpm_quota: int
    required_provider_tpm_quota: int


def calculate_finops_and_capacity(
    profile: AgentWorkloadProfile,
    expected_cache_hit_ratio: float = 0.80,
    safety_headroom: float = 1.30,
) -> FinOpsCapacityReport:
    s = profile.avg_turns_per_run
    # حساب التوكنات التراكمية عبر S دورات
    total_in = (s * profile.system_prefix_tokens) + (
        (s * (s - 1)) // 2
    ) * profile.avg_observation_tokens_per_turn
    total_out = s * profile.avg_output_tokens_per_turn

    in_unit = profile.input_price_per_1m_usd / 1_000_000
    out_unit = profile.output_price_per_1m_usd / 1_000_000

    # التكلفة بدون KV-Cache
    cost_no_cache = (total_in * in_unit) + (total_out * out_unit)

    # التكلفة مع KV-Cache (Chapter 5 Prefix Engineering)
    effective_in_multiplier = (1.0 - expected_cache_hit_ratio) + (
        expected_cache_hit_ratio * (1.0 - profile.cache_discount_ratio)
    )
    cost_with_cache = (total_in * in_unit * effective_in_multiplier) + (
        total_out * out_unit
    )

    monthly_runs = profile.daily_total_runs * 30
    monthly_no_cache = cost_no_cache * monthly_runs
    monthly_with_cache = cost_with_cache * monthly_runs

    # حساب السعة المطلوبة من المزود في الدقيقة (RPM & TPM)
    runs_per_minute_at_peak = profile.concurrent_users * (
        60.0 / profile.avg_run_duration_sec
    )
    req_rpm = int(runs_per_minute_at_peak * s * safety_headroom)
    req_tpm = int(
        runs_per_minute_at_peak * (total_in + total_out) * safety_headroom
    )

    return FinOpsCapacityReport(
        total_input_tokens_per_run=total_in,
        total_output_tokens_per_run=total_out,
        cost_per_run_no_cache_usd=round(cost_no_cache, 5),
        cost_per_run_with_80pct_cache_usd=round(cost_with_cache, 5),
        monthly_cost_no_cache_usd=round(monthly_no_cache, 2),
        monthly_cost_with_cache_usd=round(monthly_with_cache, 2),
        monthly_savings_usd=round(monthly_no_cache - monthly_with_cache, 2),
        required_provider_rpm_quota=req_rpm,
        required_provider_tpm_quota=req_tpm,
    )


if __name__ == "__main__":
    workload = AgentWorkloadProfile()
    report = calculate_finops_and_capacity(workload)
    print("=" * 75)
    print("📊 تقرير حاسبة FinOps وتخطيط السعة (10,000 تشغيل يومياً / 5 دورات للتشغيل)")
    print("=" * 75)
    print(f"• إجمالي توكنات الإدخال للتشغيل الواحد : {report.total_input_tokens_per_run:,} tokens")
    print(f"• التكلفة الشهرية (بدون هندسة الكاش)   : ${report.monthly_cost_no_cache_usd:,.2f} / month")
    print(f"• التكلفة الشهرية (مع 80% KV-Cache)    : ${report.monthly_cost_with_cache_usd:,.2f} / month")
    print(f"💰 الوفر المالي الشهري المباشر         : ${report.monthly_savings_usd:,.2f} / month!")
    print(f"⚡ حصة المزود المطلوبة في الذروة       : {report.required_provider_rpm_quota:,} RPM | {report.required_provider_tpm_quota:,} TPM")

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/agent-harness-engineering/notebooks/app-finops.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
هندسة وكلاء الذكاء الاصطناعي — المعادلات الهندسية وحاسبة اقتصاديات التوكنات
AI Agent Engineering — FinOps & Capacity Planning

Book slug : agent-harness-engineering
Chapter   : app-finops (lesson finops)
Code files: 1 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] dist/md/app-finops.md
# ────────────────────────────────────────────────────────────────────────
from pydantic import BaseModel, ConfigDict, Field


class AgentWorkloadProfile(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    system_prefix_tokens: int = Field(default=2500, ge=100)
    avg_observation_tokens_per_turn: int = Field(default=800, ge=50)
    avg_output_tokens_per_turn: int = Field(default=220, ge=10)
    avg_turns_per_run: int = Field(default=5, ge=1)
    concurrent_users: int = Field(default=50, ge=1)
    avg_run_duration_sec: float = Field(default=12.0, gt=0.5)
    daily_total_runs: int = Field(default=10_000, ge=1)

    # التسعير لكل مليون توكن (مثال: نموذج رائد متوسط التكلفة)
    input_price_per_1m_usd: float = Field(default=3.00, gt=0.0)
    output_price_per_1m_usd: float = Field(default=15.00, gt=0.0)
    cache_discount_ratio: float = Field(default=0.90, ge=0.0, le=1.0)


class FinOpsCapacityReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    total_input_tokens_per_run: int
    total_output_tokens_per_run: int
    cost_per_run_no_cache_usd: float
    cost_per_run_with_80pct_cache_usd: float
    monthly_cost_no_cache_usd: float
    monthly_cost_with_cache_usd: float
    monthly_savings_usd: float
    required_provider_rpm_quota: int
    required_provider_tpm_quota: int


def calculate_finops_and_capacity(
    profile: AgentWorkloadProfile,
    expected_cache_hit_ratio: float = 0.80,
    safety_headroom: float = 1.30,
) -> FinOpsCapacityReport:
    s = profile.avg_turns_per_run
    # حساب التوكنات التراكمية عبر S دورات
    total_in = (s * profile.system_prefix_tokens) + (
        (s * (s - 1)) // 2
    ) * profile.avg_observation_tokens_per_turn
    total_out = s * profile.avg_output_tokens_per_turn

    in_unit = profile.input_price_per_1m_usd / 1_000_000
    out_unit = profile.output_price_per_1m_usd / 1_000_000

    # التكلفة بدون KV-Cache
    cost_no_cache = (total_in * in_unit) + (total_out * out_unit)

    # التكلفة مع KV-Cache (Chapter 5 Prefix Engineering)
    effective_in_multiplier = (1.0 - expected_cache_hit_ratio) + (
        expected_cache_hit_ratio * (1.0 - profile.cache_discount_ratio)
    )
    cost_with_cache = (total_in * in_unit * effective_in_multiplier) + (
        total_out * out_unit
    )

    monthly_runs = profile.daily_total_runs * 30
    monthly_no_cache = cost_no_cache * monthly_runs
    monthly_with_cache = cost_with_cache * monthly_runs

    # حساب السعة المطلوبة من المزود في الدقيقة (RPM & TPM)
    runs_per_minute_at_peak = profile.concurrent_users * (
        60.0 / profile.avg_run_duration_sec
    )
    req_rpm = int(runs_per_minute_at_peak * s * safety_headroom)
    req_tpm = int(
        runs_per_minute_at_peak * (total_in + total_out) * safety_headroom
    )

    return FinOpsCapacityReport(
        total_input_tokens_per_run=total_in,
        total_output_tokens_per_run=total_out,
        cost_per_run_no_cache_usd=round(cost_no_cache, 5),
        cost_per_run_with_80pct_cache_usd=round(cost_with_cache, 5),
        monthly_cost_no_cache_usd=round(monthly_no_cache, 2),
        monthly_cost_with_cache_usd=round(monthly_with_cache, 2),
        monthly_savings_usd=round(monthly_no_cache - monthly_with_cache, 2),
        required_provider_rpm_quota=req_rpm,
        required_provider_tpm_quota=req_tpm,
    )


if __name__ == "__main__":
    workload = AgentWorkloadProfile()
    report = calculate_finops_and_capacity(workload)
    print("=" * 75)
    print("📊 تقرير حاسبة FinOps وتخطيط السعة (10,000 تشغيل يومياً / 5 دورات للتشغيل)")
    print("=" * 75)
    print(f"• إجمالي توكنات الإدخال للتشغيل الواحد : {report.total_input_tokens_per_run:,} tokens")
    print(f"• التكلفة الشهرية (بدون هندسة الكاش)   : ${report.monthly_cost_no_cache_usd:,.2f} / month")
    print(f"• التكلفة الشهرية (مع 80% KV-Cache)    : ${report.monthly_cost_with_cache_usd:,.2f} / month")
    print(f"💰 الوفر المالي الشهري المباشر         : ${report.monthly_savings_usd:,.2f} / month!")
    print(f"⚡ حصة المزود المطلوبة في الذروة       : {report.required_provider_rpm_quota:,} RPM | {report.required_provider_tpm_quota:,} TPM")

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/agent-harness-engineering/notebooks/app-finops.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
from pydantic import BaseModel, ConfigDict, Field


class AgentWorkloadProfile(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    system_prefix_tokens: int = Field(default=2500, ge=100)
    avg_observation_tokens_per_turn: int = Field(default=800, ge=50)
    avg_output_tokens_per_turn: int = Field(default=220, ge=10)
    avg_turns_per_run: int = Field(default=5, ge=1)
    concurrent_users: int = Field(default=50, ge=1)
    avg_run_duration_sec: float = Field(default=12.0, gt=0.5)
    daily_total_runs: int = Field(default=10_000, ge=1)

    # التسعير لكل مليون توكن (مثال: نموذج رائد متوسط التكلفة)
    input_price_per_1m_usd: float = Field(default=3.00, gt=0.0)
    output_price_per_1m_usd: float = Field(default=15.00, gt=0.0)
    cache_discount_ratio: float = Field(default=0.90, ge=0.0, le=1.0)


class FinOpsCapacityReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    total_input_tokens_per_run: int
    total_output_tokens_per_run: int
    cost_per_run_no_cache_usd: float
    cost_per_run_with_80pct_cache_usd: float
    monthly_cost_no_cache_usd: float
    monthly_cost_with_cache_usd: float
    monthly_savings_usd: float
    required_provider_rpm_quota: int
    required_provider_tpm_quota: int


def calculate_finops_and_capacity(
    profile: AgentWorkloadProfile,
    expected_cache_hit_ratio: float = 0.80,
    safety_headroom: float = 1.30,
) -> FinOpsCapacityReport:
    s = profile.avg_turns_per_run
    # حساب التوكنات التراكمية عبر S دورات
    total_in = (s * profile.system_prefix_tokens) + (
        (s * (s - 1)) // 2
    ) * profile.avg_observation_tokens_per_turn
    total_out = s * profile.avg_output_tokens_per_turn

    in_unit = profile.input_price_per_1m_usd / 1_000_000
    out_unit = profile.output_price_per_1m_usd / 1_000_000

    # التكلفة بدون KV-Cache
    cost_no_cache = (total_in * in_unit) + (total_out * out_unit)

    # التكلفة مع KV-Cache (Chapter 5 Prefix Engineering)
    effective_in_multiplier = (1.0 - expected_cache_hit_ratio) + (
        expected_cache_hit_ratio * (1.0 - profile.cache_discount_ratio)
    )
    cost_with_cache = (total_in * in_unit * effective_in_multiplier) + (
        total_out * out_unit
    )

    monthly_runs = profile.daily_total_runs * 30
    monthly_no_cache = cost_no_cache * monthly_runs
    monthly_with_cache = cost_with_cache * monthly_runs

    # حساب السعة المطلوبة من المزود في الدقيقة (RPM & TPM)
    runs_per_minute_at_peak = profile.concurrent_users * (
        60.0 / profile.avg_run_duration_sec
    )
    req_rpm = int(runs_per_minute_at_peak * s * safety_headroom)
    req_tpm = int(
        runs_per_minute_at_peak * (total_in + total_out) * safety_headroom
    )

    return FinOpsCapacityReport(
        total_input_tokens_per_run=total_in,
        total_output_tokens_per_run=total_out,
        cost_per_run_no_cache_usd=round(cost_no_cache, 5),
        cost_per_run_with_80pct_cache_usd=round(cost_with_cache, 5),
        monthly_cost_no_cache_usd=round(monthly_no_cache, 2),
        monthly_cost_with_cache_usd=round(monthly_with_cache, 2),
        monthly_savings_usd=round(monthly_no_cache - monthly_with_cache, 2),
        required_provider_rpm_quota=req_rpm,
        required_provider_tpm_quota=req_tpm,
    )


if __name__ == "__main__":
    workload = AgentWorkloadProfile()
    report = calculate_finops_and_capacity(workload)
    print("=" * 75)
    print("📊 تقرير حاسبة FinOps وتخطيط السعة (10,000 تشغيل يومياً / 5 دورات للتشغيل)")
    print("=" * 75)
    print(f"• إجمالي توكنات الإدخال للتشغيل الواحد : {report.total_input_tokens_per_run:,} tokens")
    print(f"• التكلفة الشهرية (بدون هندسة الكاش)   : ${report.monthly_cost_no_cache_usd:,.2f} / month")
    print(f"• التكلفة الشهرية (مع 80% KV-Cache)    : ${report.monthly_cost_with_cache_usd:,.2f} / month")
    print(f"💰 الوفر المالي الشهري المباشر         : ${report.monthly_savings_usd:,.2f} / month!")
    print(f"⚡ حصة المزود المطلوبة في الذروة       : {report.required_provider_rpm_quota:,} RPM | {report.required_provider_tpm_quota:,} TPM")
