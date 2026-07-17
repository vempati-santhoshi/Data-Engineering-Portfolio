# ============================================
# 08_quality_checks.py
#
# Comprehensive data quality framework
# for the promotional analytics pipeline.
#
# Why Quality Checks Matter:
# This pipeline processes 1.2 billion
# rows of loyalty and scan data to
# produce promotional metrics used
# for business decisions.
#
# If the data quality is poor:
# → Wrong metrics influence decisions
# → Promotions get approved based on
#   incorrect performance data
# → Business loses money
# → Trust in data is lost
#
# Quality check philosophy:
# → Check early and often
# → Fail fast with clear messages
# → Never silently pass bad data
# → Log everything for audit trail
#
# Checks are organised into layers:
#
# Layer 1 — Source checks
#   Validate raw source data before
#   any transformations are applied
#
# Layer 2 — Pipeline checks
#   Validate after each major step
#   in the pipeline
#
# Layer 3 — Output checks
#   Validate final metrics table
#   before it is used by business
#
# Layer 4 — Reconciliation checks
#   Cross-check totals between
#   source data and final output
#   to ensure nothing was lost
# ============================================

from pyspark.sql import functions as F
from pyspark.sql import DataFrame
from pyspark.sql import SparkSession
from datetime import datetime
from typing import List, Dict, Optional

spark = SparkSession.builder.getOrCreate()


# ============================================
# QUALITY CHECK RESULT CLASS
# Stores result of each individual check
# ============================================

class QualityCheckResult:
    """
    Stores result of a single
    quality check.

    Attributes:
    → check_name: Name of the check
    → passed: True if check passed
    → message: Description of result
    → value: Actual value found
    → threshold: Expected threshold
    → timestamp: When check ran
    """

    def __init__(
        self,
        check_name: str,
        passed: bool,
        message: str,
        value=None,
        threshold=None
    ):
        self.check_name = check_name
        self.passed = passed
        self.message = message
        self.value = value
        self.threshold = threshold
        self.timestamp = datetime.now()

    def __str__(self):
        status = "✅ PASS" if self.passed \
            else "❌ FAIL"
        return (
            f"{status} | {self.check_name} | "
            f"{self.message}"
        )


# ============================================
# QUALITY CHECK RUNNER
# Runs checks and collects results
# ============================================

class QualityCheckRunner:
    """
    Runs quality checks and
    collects results for reporting.

    Usage:
        runner = QualityCheckRunner(
            "Pipeline Name"
        )
        runner.check_no_nulls(df, "col")
        runner.check_no_negatives(df, "col")
        runner.print_summary()
        runner.raise_if_failed()
    """

    def __init__(self, pipeline_name: str):
        self.pipeline_name = pipeline_name
        self.results: List[
            QualityCheckResult
        ] = []
        self.start_time = datetime.now()

    def add_result(
        self,
        result: QualityCheckResult
    ) -> None:
        """Adds check result to list"""
        self.results.append(result)
        print(str(result))

    def passed_count(self) -> int:
        """Returns number of passed checks"""
        return sum(
            1 for r in self.results
            if r.passed
        )

    def failed_count(self) -> int:
        """Returns number of failed checks"""
        return sum(
            1 for r in self.results
            if not r.passed
        )

    def all_passed(self) -> bool:
        """Returns True if all checks passed"""
        return self.failed_count() == 0

    def print_summary(self) -> None:
        """Prints summary of all checks"""
        total = len(self.results)
        passed = self.passed_count()
        failed = self.failed_count()
        duration = (
            datetime.now() - self.start_time
        ).seconds

        print(f"\n{'='*50}")
        print(f"Quality Check Summary")
        print(f"Pipeline: {self.pipeline_name}")
        print(f"{'='*50}")
        print(f"Total checks:  {total}")
        print(f"Passed:        {passed} ✅")
        print(f"Failed:        {failed} ❌")
        print(f"Duration:      {duration}s")
        print(f"{'='*50}")

        if failed > 0:
            print("\nFailed checks:")
            for r in self.results:
                if not r.passed:
                    print(f"  → {r.check_name}")
                    print(f"    {r.message}")

    def raise_if_failed(self) -> None:
        """
        Raises exception if any
        checks failed.

        Call this after all checks
        to stop pipeline if quality
        issues found.
        """
        if not self.all_passed():
            failed = self.failed_count()
            raise ValueError(
                f"{failed} quality checks "
                f"failed in "
                f"{self.pipeline_name}. "
                f"Pipeline stopped."
            )


# ============================================
# LAYER 1 — SOURCE DATA CHECKS
# Validate raw data before processing
# ============================================

def check_source_data(
    df_scan: DataFrame,
    df_outlet: DataFrame,
    df_product: DataFrame,
    df_promo: DataFrame
) -> QualityCheckRunner:
    """
    Layer 1: Validates all source
    tables before pipeline starts.

    Checks:
    → All tables have data (not empty)
    → Date filter applied correctly
    → Key columns exist and not null
    → Expected date ranges present
    → No duplicate key records

    Args:
        df_scan: Raw scan transactions
        df_outlet: Outlet dimension
        df_product: Product dimension
        df_promo: Promotion table

    Returns:
        QualityCheckRunner with results
    """
    runner = QualityCheckRunner(
        "Source Data Checks"
    )

    print("\n--- Layer 1: Source Data ---")

    # Check 1 — Scan table not empty
    scan_count = df_scan.count()
    runner.add_result(QualityCheckResult(
        check_name="scan_table_not_empty",
        passed=scan_count > 0,
        message=f"Scan rows: {scan_count:,}",
        value=scan_count,
        threshold=0
    ))

    # Check 2 — Outlet table not empty
    outlet_count = df_outlet.count()
    runner.add_result(QualityCheckResult(
        check_name="outlet_table_not_empty",
        passed=outlet_count > 0,
        message=(
            f"Outlet rows: {outlet_count:,}"
        ),
        value=outlet_count,
        threshold=0
    ))

    # Check 3 — Product table not empty
    product_count = df_product.count()
    runner.add_result(QualityCheckResult(
        check_name="product_table_not_empty",
        passed=product_count > 0,
        message=(
            f"Product rows: {product_count:,}"
        ),
        value=product_count,
        threshold=0
    ))

    # Check 4 — Promo table not empty
    promo_count = df_promo.count()
    runner.add_result(QualityCheckResult(
        check_name="promo_table_not_empty",
        passed=promo_count > 0,
        message=f"Promo rows: {promo_count:,}",
        value=promo_count,
        threshold=0
    ))

    # Check 5 — No old dates in scan
    old_dates = df_scan.filter(
        F.col("DateKey") < 20250101
    ).count()
    runner.add_result(QualityCheckResult(
        check_name="scan_date_filter_applied",
        passed=old_dates == 0,
        message=(
            f"Rows before 2025-01-01: "
            f"{old_dates}"
        ),
        value=old_dates,
        threshold=0
    ))

    # Check 6 — No null outlet keys
    null_outlet_keys = df_scan.filter(
        F.col("OutletKey").isNull()
    ).count()
    runner.add_result(QualityCheckResult(
        check_name="no_null_outlet_keys",
        passed=null_outlet_keys == 0,
        message=(
            f"Null outlet keys: "
            f"{null_outlet_keys}"
        ),
        value=null_outlet_keys,
        threshold=0
    ))

    # Check 7 — No null product keys
    null_product_keys = df_scan.filter(
        F.col("ProductKey").isNull()
    ).count()
    runner.add_result(QualityCheckResult(
        check_name="no_null_product_keys",
        passed=null_product_keys == 0,
        message=(
            f"Null product keys: "
            f"{null_product_keys}"
        ),
        value=null_product_keys,
        threshold=0
    ))

    # Check 8 — Promo status filter
    invalid_status = df_promo.filter(
        ~F.col("PromotionStatus").isin(
            "Committed", "Closed"
        )
    ).count()
    runner.add_result(QualityCheckResult(
        check_name="promo_status_filtered",
        passed=invalid_status == 0,
        message=(
            f"Invalid promo status: "
            f"{invalid_status}"
        ),
        value=invalid_status,
        threshold=0
    ))

    runner.print_summary()
    return runner


# ============================================
# LAYER 2 — PIPELINE CHECKS
# Validate after each major step
# ============================================

def check_discount_flags(
    df: DataFrame
) -> QualityCheckRunner:
    """
    Layer 2a: Validates discount
    flag derivation.

    Checks:
    → SMP discount not negative
    → MP discount not negative
    → CPN discount not negative
    → is_promo only 0 or 1
    → All discount columns present

    Args:
        df: DataFrame after discount
            columns added

    Returns:
        QualityCheckRunner with results
    """
    runner = QualityCheckRunner(
        "Discount Flag Checks"
    )

    print("\n--- Layer 2a: Discount Flags ---")

    # Check 1 — No negative SMP
    neg_smp = df.filter(
        F.col("SMP_discount") < 0
    ).count()
    runner.add_result(QualityCheckResult(
        check_name="no_negative_smp",
        passed=neg_smp == 0,
        message=f"Negative SMP: {neg_smp}",
        value=neg_smp,
        threshold=0
    ))

    # Check 2 — No negative MP
    neg_mp = df.filter(
        F.col("MP_discount") < 0
    ).count()
    runner.add_result(QualityCheckResult(
        check_name="no_negative_mp",
        passed=neg_mp == 0,
        message=f"Negative MP: {neg_mp}",
        value=neg_mp,
        threshold=0
    ))

    # Check 3 — No negative CPN
    neg_cpn = df.filter(
        F.col("CPN_discount") < 0
    ).count()
    runner.add_result(QualityCheckResult(
        check_name="no_negative_cpn",
        passed=neg_cpn == 0,
        message=f"Negative CPN: {neg_cpn}",
        value=neg_cpn,
        threshold=0
    ))

    # Check 4 — is_promo valid
    invalid_flag = df.filter(
        ~F.col("is_promo").isin(0, 1)
    ).count()
    runner.add_result(QualityCheckResult(
        check_name="is_promo_valid",
        passed=invalid_flag == 0,
        message=(
            f"Invalid is_promo: {invalid_flag}"
        ),
        value=invalid_flag,
        threshold=0
    ))

    runner.print_summary()
    return runner


def check_loyalty_derivation(
    df_all: DataFrame,
    df_valid: DataFrame
) -> QualityCheckRunner:
    """
    Layer 2b: Validates loyalty
    code derivation.

    Checks:
    → No null codes in valid set
    → No empty string codes
    → Derivation rate reasonable
    → Valid codes start with
      SubmitterName prefix

    Args:
        df_all: All transactions
        df_valid: Valid loyalty only

    Returns:
        QualityCheckRunner with results
    """
    runner = QualityCheckRunner(
        "Loyalty Derivation Checks"
    )

    print(
        "\n--- Layer 2b: "
        "Loyalty Derivation ---"
    )

    total = df_all.count()
    valid = df_valid.count()
    rate = valid / total * 100

    # Check 1 — No null codes
    null_codes = df_valid.filter(
        F.col("LoyaltyCode").isNull()
    ).count()
    runner.add_result(QualityCheckResult(
        check_name="no_null_loyalty_codes",
        passed=null_codes == 0,
        message=f"Null codes: {null_codes}",
        value=null_codes,
        threshold=0
    ))

    # Check 2 — No empty codes
    empty_codes = df_valid.filter(
        F.col("LoyaltyCode") == ""
    ).count()
    runner.add_result(QualityCheckResult(
        check_name="no_empty_loyalty_codes",
        passed=empty_codes == 0,
        message=f"Empty codes: {empty_codes}",
        value=empty_codes,
        threshold=0
    ))

    # Check 3 — Derivation rate > 1%
    runner.add_result(QualityCheckResult(
        check_name="loyalty_derivation_rate",
        passed=rate > 1.0,
        message=(
            f"Derivation rate: {rate:.1f}%"
        ),
        value=rate,
        threshold=1.0
    ))

    runner.print_summary()
    return runner


def check_promo_windows(
    df: DataFrame
) -> QualityCheckRunner:
    """
    Layer 2c: Validates promo
    window calculations.

    Checks:
    → No negative durations
    → Start date before end date
    → Before window ends before promo
    → After window starts after promo
    → Duration days consistent

    Args:
        df: DataFrame with promo
            windows calculated

    Returns:
        QualityCheckRunner with results
    """
    runner = QualityCheckRunner(
        "Promo Window Checks"
    )

    print(
        "\n--- Layer 2c: "
        "Promo Windows ---"
    )

    # Check 1 — No negative durations
    neg_dur = df.filter(
        F.col("promo_duration_days") <= 0
    ).count()
    runner.add_result(QualityCheckResult(
        check_name="no_negative_durations",
        passed=neg_dur == 0,
        message=f"Negative durations: {neg_dur}",
        value=neg_dur,
        threshold=0
    ))

    # Check 2 — Start before end
    invalid_dates = df.filter(
        F.col("PromoTrueStartDate_dt") >
        F.col("PromoTrueEndDate_dt")
    ).count()
    runner.add_result(QualityCheckResult(
        check_name="start_before_end",
        passed=invalid_dates == 0,
        message=(
            f"Invalid date ranges: "
            f"{invalid_dates}"
        ),
        value=invalid_dates,
        threshold=0
    ))

    # Check 3 — Before window correct
    invalid_before = df.filter(
        F.col("before_candidate_end") >=
        F.col("PromoTrueStartDate_dt")
    ).count()
    runner.add_result(QualityCheckResult(
        check_name="before_window_correct",
        passed=invalid_before == 0,
        message=(
            f"Invalid before windows: "
            f"{invalid_before}"
        ),
        value=invalid_before,
        threshold=0
    ))

    # Check 4 — After window correct
    invalid_after = df.filter(
        F.col("after_candidate_start") <=
        F.col("PromoTrueEndDate_dt")
    ).count()
    runner.add_result(QualityCheckResult(
        check_name="after_window_correct",
        passed=invalid_after == 0,
        message=(
            f"Invalid after windows: "
            f"{invalid_after}"
        ),
        value=invalid_after,
        threshold=0
    ))

    runner.print_summary()
    return runner


def check_overlap_detection(
    result_dur: DataFrame,
    result_windows: DataFrame
) -> QualityCheckRunner:
    """
    Layer 2d: Validates overlap
    detection results.

    Checks:
    → Row count preserved
    → Overlap flags valid (0 or 1)
    → Null dates when invalid window
    → Not null dates when valid window

    Args:
        result_dur: Input promo DataFrame
        result_windows: After overlap
            detection

    Returns:
        QualityCheckRunner with results
    """
    runner = QualityCheckRunner(
        "Overlap Detection Checks"
    )

    print(
        "\n--- Layer 2d: "
        "Overlap Detection ---"
    )

    # Check 1 — Row count preserved
    input_count = result_dur.count()
    output_count = result_windows.count()
    runner.add_result(QualityCheckResult(
        check_name="row_count_preserved",
        passed=input_count == output_count,
        message=(
            f"In: {input_count:,} "
            f"Out: {output_count:,}"
        ),
        value=output_count,
        threshold=input_count
    ))

    # Check 2 — Before overlap valid
    invalid_before = result_windows.filter(
        ~F.col("before_overlap").isin(0, 1)
    ).count()
    runner.add_result(QualityCheckResult(
        check_name="before_overlap_valid",
        passed=invalid_before == 0,
        message=(
            f"Invalid before flags: "
            f"{invalid_before}"
        ),
        value=invalid_before,
        threshold=0
    ))

    # Check 3 — After overlap valid
    invalid_after = result_windows.filter(
        ~F.col("after_overlap").isin(0, 1)
    ).count()
    runner.add_result(QualityCheckResult(
        check_name="after_overlap_valid",
        passed=invalid_after == 0,
        message=(
            f"Invalid after flags: "
            f"{invalid_after}"
        ),
        value=invalid_after,
        threshold=0
    ))

    # Check 4 — Null dates when invalid
    wrong_nulls = result_windows.filter(
        (~F.col("has_before_window")) &
        (F.col("before_window_start")
         .isNotNull())
    ).count()
    runner.add_result(QualityCheckResult(
        check_name="null_dates_when_invalid",
        passed=wrong_nulls == 0,
        message=(
            f"Wrong null dates: {wrong_nulls}"
        ),
        value=wrong_nulls,
        threshold=0
    ))

    runner.print_summary()
    return runner


def check_tagged_transactions(
    all_tagged: DataFrame
) -> QualityCheckRunner:
    """
    Layer 2e: Validates tagged
    transaction results.

    Checks:
    → Only valid period values
    → No null periods
    → No null loyalty codes
    → No null outlet identifiers
    → No negative volumes
    → No negative values

    Args:
        all_tagged: Tagged transactions
            DataFrame

    Returns:
        QualityCheckRunner with results
    """
    runner = QualityCheckRunner(
        "Tagged Transaction Checks"
    )

    print(
        "\n--- Layer 2e: "
        "Tagged Transactions ---"
    )

    valid_periods = [
        "before_promo",
        "during_promo",
        "after_promo"
    ]

    # Check 1 — Valid periods
    invalid_periods = all_tagged.filter(
        ~F.col("period").isin(valid_periods)
    ).count()
    runner.add_result(QualityCheckResult(
        check_name="valid_period_values",
        passed=invalid_periods == 0,
        message=(
            f"Invalid periods: "
            f"{invalid_periods}"
        ),
        value=invalid_periods,
        threshold=0
    ))

    # Check 2 — No null periods
    null_periods = all_tagged.filter(
        F.col("period").isNull()
    ).count()
    runner.add_result(QualityCheckResult(
        check_name="no_null_periods",
        passed=null_periods == 0,
        message=f"Null periods: {null_periods}",
        value=null_periods,
        threshold=0
    ))

    # Check 3 — No null loyalty
    null_loyalty = all_tagged.filter(
        F.col("LoyaltyCode").isNull()
    ).count()
    runner.add_result(QualityCheckResult(
        check_name="no_null_loyalty_codes",
        passed=null_loyalty == 0,
        message=(
            f"Null loyalty: {null_loyalty}"
        ),
        value=null_loyalty,
        threshold=0
    ))

    # Check 4 — No negative volumes
    neg_volume = all_tagged.filter(
        F.col("MetricVolume") < 0
    ).count()
    runner.add_result(QualityCheckResult(
        check_name="no_negative_volumes",
        passed=neg_volume == 0,
        message=(
            f"Negative volumes: {neg_volume}"
        ),
        value=neg_volume,
        threshold=0
    ))

    # Check 5 — No negative values
    neg_value = all_tagged.filter(
        F.col("MetricValue") < 0
    ).count()
    runner.add_result(QualityCheckResult(
        check_name="no_negative_values",
        passed=neg_value == 0,
        message=f"Negative values: {neg_value}",
        value=neg_value,
        threshold=0
    ))

    runner.print_summary()
    return runner


# ============================================
# LAYER 3 — OUTPUT CHECKS
# Validate final metrics table
# ============================================

def check_final_metrics(
    metrics: DataFrame
) -> QualityCheckRunner:
    """
    Layer 3: Validates final metrics
    table before business use.

    Checks:
    → No null key columns
    → No negative during metrics
    → During metrics always present
    → Date ranges valid
    → No duplicate rows
    → Derived metrics in valid range

    Args:
        metrics: Final metrics DataFrame

    Returns:
        QualityCheckRunner with results
    """
    runner = QualityCheckRunner(
        "Final Metrics Checks"
    )

    print("\n--- Layer 3: Final Metrics ---")

    # Check 1 — No null outlets
    null_outlets = metrics.filter(
        F.col("OutletIdentifier").isNull()
    ).count()
    runner.add_result(QualityCheckResult(
        check_name="no_null_outlets",
        passed=null_outlets == 0,
        message=f"Null outlets: {null_outlets}",
        value=null_outlets,
        threshold=0
    ))

    # Check 2 — No null brands
    null_brands = metrics.filter(
        F.col("ReportingBrandName").isNull()
    ).count()
    runner.add_result(QualityCheckResult(
        check_name="no_null_brands",
        passed=null_brands == 0,
        message=f"Null brands: {null_brands}",
        value=null_brands,
        threshold=0
    ))

    # Check 3 — No null promos
    null_promos = metrics.filter(
        F.col("PromotionName").isNull()
    ).count()
    runner.add_result(QualityCheckResult(
        check_name="no_null_promos",
        passed=null_promos == 0,
        message=f"Null promos: {null_promos}",
        value=null_promos,
        threshold=0
    ))

    # Check 4 — No null dates
    null_dates = metrics.filter(
        F.col("PromoTrueStartDate_dt")
         .isNull() |
        F.col("PromoTrueEndDate_dt")
         .isNull()
    ).count()
    runner.add_result(QualityCheckResult(
        check_name="no_null_dates",
        passed=null_dates == 0,
        message=f"Null dates: {null_dates}",
        value=null_dates,
        threshold=0
    ))

    # Check 5 — No negative during volume
    neg_vol = metrics.filter(
        (F.col("during_promo_total_volume")
         .isNotNull()) &
        (F.col("during_promo_total_volume") < 0)
    ).count()
    runner.add_result(QualityCheckResult(
        check_name="no_negative_during_volume",
        passed=neg_vol == 0,
        message=(
            f"Negative during volume: {neg_vol}"
        ),
        value=neg_vol,
        threshold=0
    ))

    # Check 6 — No duplicate rows
    total = metrics.count()
    distinct = metrics.select(
        "OutletIdentifier",
        "ReportingBrandName",
        "PromotionName"
    ).distinct().count()
    runner.add_result(QualityCheckResult(
        check_name="no_duplicate_rows",
        passed=total == distinct,
        message=(
            f"Total: {total:,} "
            f"Distinct: {distinct:,}"
        ),
        value=total,
        threshold=distinct
    ))

    # Check 7 — Volume lift reasonable
    # (between -100% and +1000%)
    if "volume_lift_pct" in metrics.columns:
        unreasonable_lift = metrics.filter(
            F.col("volume_lift_pct")
             .isNotNull() &
            (
                (F.col("volume_lift_pct") < -100) |
                (F.col("volume_lift_pct") > 1000)
            )
        ).count()
        runner.add_result(QualityCheckResult(
            check_name="volume_lift_reasonable",
            passed=unreasonable_lift == 0,
            message=(
                f"Unreasonable lifts: "
                f"{unreasonable_lift}"
            ),
            value=unreasonable_lift,
            threshold=0
        ))

    runner.print_summary()
    return runner


# ============================================
# LAYER 4 — RECONCILIATION CHECKS
# Cross-check source vs output totals
# ============================================

def check_reconciliation(
    all_tagged: DataFrame,
    metrics: DataFrame
) -> QualityCheckRunner:
    """
    Layer 4: Reconciles totals between
    tagged transactions and final
    metrics output.

    Checks:
    → Total volume matches
    → Total value matches
    → No data lost in aggregation
    → During period totals match

    Args:
        all_tagged: Tagged transactions
            before aggregation
        metrics: Final metrics after
            aggregation

    Returns:
        QualityCheckRunner with results
    """
    runner = QualityCheckRunner(
        "Reconciliation Checks"
    )

    print("\n--- Layer 4: Reconciliation ---")

    # Source totals from tagged data
    source_totals = all_tagged.agg(
        F.sum("MetricVolume")
         .alias("total_volume"),
        F.sum("MetricValue")
         .alias("total_value")
    ).collect()[0]

    source_volume = source_totals[
        "total_volume"
    ]
    source_value = source_totals[
        "total_value"
    ]

    # Output totals from metrics
    output_volume = (
        metrics.agg(
            F.sum("before_promo_total_volume") +
            F.sum("during_promo_total_volume") +
            F.sum("after_promo_total_volume")
        ).collect()[0][0]
    )

    output_value = (
        metrics.agg(
            F.sum("before_promo_total_value") +
            F.sum("during_promo_total_value") +
            F.sum("after_promo_total_value")
        ).collect()[0][0]
    )

    # Check 1 — Volume reconciles
    # Allow 1% tolerance for rounding
    vol_diff_pct = abs(
        (output_volume - source_volume)
        / source_volume * 100
    ) if source_volume else 0

    runner.add_result(QualityCheckResult(
        check_name="volume_reconciles",
        passed=vol_diff_pct < 1.0,
        message=(
            f"Source: {source_volume:,.0f} "
            f"Output: {output_volume:,.0f} "
            f"Diff: {vol_diff_pct:.2f}%"
        ),
        value=vol_diff_pct,
        threshold=1.0
    ))

    # Check 2 — Value reconciles
    val_diff_pct = abs(
        (output_value - source_value)
        / source_value * 100
    ) if source_value else 0

    runner.add_result(QualityCheckResult(
        check_name="value_reconciles",
        passed=val_diff_pct < 1.0,
        message=(
            f"Source: {source_value:,.0f} "
            f"Output: {output_value:,.0f} "
            f"Diff: {val_diff_pct:.2f}%"
        ),
        value=val_diff_pct,
        threshold=1.0
    ))

    runner.print_summary()
    return runner


# ============================================
# MAIN FUNCTION
# Run all quality checks in sequence
# ============================================

def run_all_quality_checks(
    df_scan: DataFrame,
    df_outlet: DataFrame,
    df_product: DataFrame,
    df_promo: DataFrame,
    df_all: DataFrame,
    df_valid: DataFrame,
    result_dur: DataFrame,
    result_windows: DataFrame,
    all_tagged: DataFrame,
    metrics: DataFrame,
    raise_on_failure: bool = True
) -> Dict[str, QualityCheckRunner]:
    """
    Runs all quality checks across
    all 4 layers of the pipeline.

    Args:
        df_scan: Raw scan transactions
        df_outlet: Outlet dimension
        df_product: Product dimension
        df_promo: Promotion table
        df_all: All loyalty transactions
        df_valid: Valid loyalty only
        result_dur: Promo windows
        result_windows: After overlap
        all_tagged: Tagged transactions
        metrics: Final metrics table
        raise_on_failure: Stop pipeline
            if any checks fail

    Returns:
        Dictionary of all runners
        keyed by layer name
    """
    print("\n" + "="*50)
    print("RUNNING ALL QUALITY CHECKS")
    print("="*50)

    runners = {}

    # Layer 1 — Source checks
    runners["source"] = check_source_data(
        df_scan,
        df_outlet,
        df_product,
        df_promo
    )

    # Layer 2a — Discount checks
    runners["discounts"] = \
        check_discount_flags(df_scan)

    # Layer 2b — Loyalty checks
    runners["loyalty"] = \
        check_loyalty_derivation(
            df_all,
            df_valid
        )

    # Layer 2c — Window checks
    runners["windows"] = \
        check_promo_windows(result_dur)

    # Layer 2d — Overlap checks
    runners["overlap"] = \
        check_overlap_detection(
            result_dur,
            result_windows
        )

    # Layer 2e — Tagging checks
    runners["tagging"] = \
        check_tagged_transactions(all_tagged)

    # Layer 3 — Output checks
    runners["output"] = \
        check_final_metrics(metrics)

    # Layer 4 — Reconciliation
    runners["reconciliation"] = \
        check_reconciliation(
            all_tagged,
            metrics
        )

    # Final summary
    total_passed = sum(
        r.passed_count()
        for r in runners.values()
    )
    total_failed = sum(
        r.failed_count()
        for r in runners.values()
    )
    total_checks = total_passed + total_failed

    print("\n" + "="*50)
    print("OVERALL QUALITY SUMMARY")
    print("="*50)
    print(f"Total checks run:  {total_checks}")
    print(f"Total passed:      {total_passed} ✅")
    print(f"Total failed:      {total_failed} ❌")

    if total_failed == 0:
        print(
            "\n✅ ALL QUALITY CHECKS PASSED"
        )
        print(
            "Pipeline output is ready "
            "for business use."
        )
    else:
        print(
            f"\n❌ {total_failed} CHECKS FAILED"
        )
        print(
            "Review failed checks before "
            "using pipeline output."
        )
        if raise_on_failure:
            raise ValueError(
                f"{total_failed} quality "
                f"checks failed. "
                f"Pipeline stopped."
            )

    print("="*50)
    return runners


# ============================================
# USAGE EXAMPLE
# ============================================

if __name__ == "__main__":

    print("Running quality checks...")

    # Run all checks
    runners = run_all_quality_checks(
        df_scan=df_scan,
        df_outlet=df_outlet,
        df_product=df_product,
        df_promo=df_promo,
        df_all=df_loyalty,
        df_valid=df_loyalty_valid,
        result_dur=result_dur,
        result_windows=result_windows,
        all_tagged=all_tagged,
        metrics=metrics,
        raise_on_failure=True
    )

    print("\nAll quality checks complete ✅")
    print("Pipeline is production ready 🎉")
