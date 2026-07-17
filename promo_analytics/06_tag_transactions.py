# ============================================
# 06_tag_transactions.py
#
# Tags each loyalty transaction as
# before, during or after promotion.
#
# Business Problem:
# We have 67 million loyalty transactions
# and 782,707 promotions.
#
# We need to label each transaction
# with which promotional period it
# falls into so we can compare
# customer behaviour across periods.
#
# Challenge:
# A naive join of all transactions
# against all promotions would create:
# 67M x 4 promos per outlet = 268M rows
# before filtering → memory explosion
#
# Solution:
# 3 separate targeted joins:
# → Join 1: loyalty x during windows
#   filter date IN promo period
# → Join 2: loyalty x before windows
#   filter date IN before window
# → Join 3: loyalty x after windows
#   filter date IN after window
#
# Each join filters immediately after
# joining → kills explosion early
# → Much smaller intermediate results
# → No cluster memory issues
#
# Period Definitions:
# during_promo:
# → DateKey >= PromoTrueStartDate_dt
# → DateKey <= PromoTrueEndDate_dt
#
# before_promo:
# → DateKey >= before_window_start
# → DateKey <= before_window_end
# → Only for promos with valid
#   before window (no overlap)
#
# after_promo:
# → DateKey >= after_window_start
# → DateKey <= after_window_end
# → Only for promos with valid
#   after window (no overlap)
# ============================================

from pyspark.sql import functions as F
from pyspark.sql import DataFrame
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()


# ============================================
# STEP 1 — PREPARE LOYALTY DATA
# Convert DateKey to proper date type
# for date comparison joins
# ============================================

def prepare_loyalty_dates(
    df_loyalty: DataFrame
) -> DataFrame:
    """
    Converts DateKey integer column
    to proper date type for joining
    with promo window dates.

    Source column:
    → Promo_DateKey (integer yyyyMMdd)
      Example: 20250330

    Output column:
    → DateKey_dt (date type)
      Example: 2025-03-30

    Why convert?
    → Promo window dates are date type
    → Date comparisons need same type
    → Cannot compare integer to date

    Args:
        df_loyalty: Loyalty transaction
            DataFrame with Promo_DateKey
            integer column

    Returns:
        DataFrame with DateKey_dt
        date column added
    """
    return df_loyalty.withColumn(
        "DateKey_dt",
        F.to_date(
            F.col("Promo_DateKey")
             .cast("string"),
            "yyyyMMdd"
        )
    )


# ============================================
# STEP 2 — BUILD WINDOW TABLES
# Create slim versions of promo windows
# for each period join
# Only keep promos with valid windows
# ============================================

def get_during_windows(
    result_windows: DataFrame
) -> DataFrame:
    """
    Gets promo window table for
    during promo join.

    Includes ALL promotions since
    every promo has a during period.

    Columns kept:
    → OutletIdentifier (join key)
    → ReportingBrandName (join key)
    → PromotionName
    → PromoTrueStartDate_dt (filter)
    → PromoTrueEndDate_dt (filter)

    Args:
        result_windows: Promo windows
            DataFrame from
            05_overlap_detection.py

    Returns:
        Slim DataFrame for during join
    """
    return result_windows.select(
        "OutletIdentifier",
        "ReportingBrandName",
        "PromotionName",
        "PromoTrueStartDate_dt",
        "PromoTrueEndDate_dt"
    )


def get_before_windows(
    result_windows: DataFrame
) -> DataFrame:
    """
    Gets promo window table for
    before promo join.

    ONLY includes promos with
    valid before windows.
    (has_before_window == True)

    Filtering here reduces the
    broadcast table size and
    prevents false matches.

    Columns kept:
    → OutletIdentifier (join key)
    → ReportingBrandName (join key)
    → PromotionName
    → PromoTrueStartDate_dt
    → PromoTrueEndDate_dt
    → before_window_start (filter)
    → before_window_end (filter)

    Args:
        result_windows: Promo windows
            DataFrame from
            05_overlap_detection.py

    Returns:
        Slim DataFrame for before join
        containing only valid windows
    """
    return result_windows.filter(
        F.col("has_before_window") == True
    ).select(
        "OutletIdentifier",
        "ReportingBrandName",
        "PromotionName",
        "PromoTrueStartDate_dt",
        "PromoTrueEndDate_dt",
        "before_window_start",
        "before_window_end"
    )


def get_after_windows(
    result_windows: DataFrame
) -> DataFrame:
    """
    Gets promo window table for
    after promo join.

    ONLY includes promos with
    valid after windows.
    (has_after_window == True)

    Columns kept:
    → OutletIdentifier (join key)
    → ReportingBrandName (join key)
    → PromotionName
    → PromoTrueStartDate_dt
    → PromoTrueEndDate_dt
    → after_window_start (filter)
    → after_window_end (filter)

    Args:
        result_windows: Promo windows
            DataFrame from
            05_overlap_detection.py

    Returns:
        Slim DataFrame for after join
        containing only valid windows
    """
    return result_windows.filter(
        F.col("has_after_window") == True
    ).select(
        "OutletIdentifier",
        "ReportingBrandName",
        "PromotionName",
        "PromoTrueStartDate_dt",
        "PromoTrueEndDate_dt",
        "after_window_start",
        "after_window_end"
    )


# ============================================
# STEP 3 — THREE TARGETED JOINS
# One join per period
# Filter immediately after join
# to prevent data explosion
# ============================================

def tag_during_transactions(
    df_loyalty_dt: DataFrame,
    during_windows: DataFrame
) -> DataFrame:
    """
    Tags loyalty transactions that
    fall during the promotion period.

    Join type: INNER
    → Only keep loyalty transactions
      for outlets that have promotions

    Join keys:
    → OutletIdentifier
    → ReportingBrandName

    Filter immediately after join:
    → DateKey_dt >= PromoTrueStartDate_dt
    → DateKey_dt <= PromoTrueEndDate_dt

    Why broadcast?
    → during_windows is small
      (782K rows)
    → df_loyalty_dt is large
      (67M+ rows)
    → Broadcasting small table
      avoids expensive shuffle
      of large table

    Args:
        df_loyalty_dt: Loyalty DataFrame
            with DateKey_dt column
        during_windows: Slim promo
            windows for during period

    Returns:
        DataFrame of transactions
        tagged as during_promo
    """
    return df_loyalty_dt.join(
        F.broadcast(during_windows),
        on=[
            "OutletIdentifier",
            "ReportingBrandName"
        ],
        how="inner"
    ).filter(
        # Transaction falls within
        # promotion period
        (F.col("DateKey_dt") >=
         F.col("PromoTrueStartDate_dt")) &
        (F.col("DateKey_dt") <=
         F.col("PromoTrueEndDate_dt"))
    ).withColumn(
        "period",
        F.lit("during_promo")
    )


def tag_before_transactions(
    df_loyalty_dt: DataFrame,
    before_windows: DataFrame
) -> DataFrame:
    """
    Tags loyalty transactions that
    fall in the before promo window.

    Only joins against promos that
    have valid before windows.
    (no overlap, data available)

    Filter immediately after join:
    → DateKey_dt >= before_window_start
    → DateKey_dt <= before_window_end

    Args:
        df_loyalty_dt: Loyalty DataFrame
            with DateKey_dt column
        before_windows: Slim promo windows
            for before period
            (only valid windows)

    Returns:
        DataFrame of transactions
        tagged as before_promo
    """
    return df_loyalty_dt.join(
        F.broadcast(before_windows),
        on=[
            "OutletIdentifier",
            "ReportingBrandName"
        ],
        how="inner"
    ).filter(
        # Transaction falls within
        # before window
        (F.col("DateKey_dt") >=
         F.col("before_window_start")) &
        (F.col("DateKey_dt") <=
         F.col("before_window_end"))
    ).withColumn(
        "period",
        F.lit("before_promo")
    )


def tag_after_transactions(
    df_loyalty_dt: DataFrame,
    after_windows: DataFrame
) -> DataFrame:
    """
    Tags loyalty transactions that
    fall in the after promo window.

    Only joins against promos that
    have valid after windows.
    (no overlap, data available)

    Filter immediately after join:
    → DateKey_dt >= after_window_start
    → DateKey_dt <= after_window_end

    Args:
        df_loyalty_dt: Loyalty DataFrame
            with DateKey_dt column
        after_windows: Slim promo windows
            for after period
            (only valid windows)

    Returns:
        DataFrame of transactions
        tagged as after_promo
    """
    return df_loyalty_dt.join(
        F.broadcast(after_windows),
        on=[
            "OutletIdentifier",
            "ReportingBrandName"
        ],
        how="inner"
    ).filter(
        # Transaction falls within
        # after window
        (F.col("DateKey_dt") >=
         F.col("after_window_start")) &
        (F.col("DateKey_dt") <=
         F.col("after_window_end"))
    ).withColumn(
        "period",
        F.lit("after_promo")
    )


# ============================================
# STEP 4 — UNION ALL PERIODS
# Combine all 3 tagged datasets
# into single dataset
# ============================================

# Columns to keep in final union
# All other columns dropped
COMMON_COLS = [
    "OutletIdentifier",
    "ReportingBrandName",
    "PromotionName",
    "PromoTrueStartDate_dt",
    "PromoTrueEndDate_dt",
    "LoyaltyCode",
    "MetricVolume",
    "MetricValue",
    "period"
]


def union_all_periods(
    during_tagged: DataFrame,
    before_tagged: DataFrame,
    after_tagged: DataFrame
) -> DataFrame:
    """
    Unions all three period datasets
    into a single tagged dataset.

    Selects only COMMON_COLS before
    union to ensure consistent schema
    across all three DataFrames.

    Uses unionByName to match columns
    by name not position — safer than
    standard union.

    Final schema:
    → OutletIdentifier
    → ReportingBrandName
    → PromotionName
    → PromoTrueStartDate_dt
    → PromoTrueEndDate_dt
    → LoyaltyCode
    → MetricVolume
    → MetricValue
    → period (before/during/after_promo)

    Args:
        during_tagged: Transactions tagged
            as during_promo
        before_tagged: Transactions tagged
            as before_promo
        after_tagged: Transactions tagged
            as after_promo

    Returns:
        Single unified DataFrame with
        all transactions tagged by period
    """
    return during_tagged.select(COMMON_COLS) \
        .unionByName(
            before_tagged.select(COMMON_COLS)
        ).unionByName(
            after_tagged.select(COMMON_COLS)
        )


# ============================================
# MAIN FUNCTION
# Complete transaction tagging pipeline
# ============================================

def tag_transactions(
    df_loyalty: DataFrame,
    result_windows: DataFrame
) -> DataFrame:
    """
    Complete transaction tagging pipeline.

    Steps:
    1. Convert DateKey to date type
    2. Build slim window tables
       for each period
    3. Run 3 separate targeted joins
       (during, before, after)
    4. Union all periods together

    Memory strategy:
    → 3 separate joins instead of 1
    → Broadcast small window tables
    → Filter immediately after each join
    → Only union clean small results

    This approach successfully processes
    1.2 billion row datasets without
    cluster memory issues.

    Args:
        df_loyalty: Loyalty transactions
            DataFrame (67M+ rows)
        result_windows: Promo windows
            DataFrame from
            05_overlap_detection.py

    Returns:
        Single DataFrame with all
        transactions tagged by period

    Expected counts (approximate):
    → during_promo: largest
    → before_promo: smaller
    → after_promo:  smallest
    """
    print("Starting transaction tagging...")

    # Step 1 — Convert date
    print("Converting date columns...")
    df_loyalty_dt = prepare_loyalty_dates(
        df_loyalty
    )

    # Step 2 — Build window tables
    print("Building window tables...")
    during_windows = get_during_windows(
        result_windows
    )
    before_windows = get_before_windows(
        result_windows
    )
    after_windows = get_after_windows(
        result_windows
    )

    # Step 3 — Three targeted joins
    print("Running during promo join...")
    during_tagged = tag_during_transactions(
        df_loyalty_dt,
        during_windows
    )

    print("Running before promo join...")
    before_tagged = tag_before_transactions(
        df_loyalty_dt,
        before_windows
    )

    print("Running after promo join...")
    after_tagged = tag_after_transactions(
        df_loyalty_dt,
        after_windows
    )

    # Step 4 — Union all periods
    print("Unioning all periods...")
    all_tagged = union_all_periods(
        during_tagged,
        before_tagged,
        after_tagged
    )

    return all_tagged


# ============================================
# VALIDATION
# ============================================

def validate_tagged_transactions(
    all_tagged: DataFrame
) -> DataFrame:
    """
    Validates tagged transaction results.

    Checks:
    → Only valid period values exist
    → No null period values
    → No null loyalty codes
    → No null outlet identifiers
    → No negative volumes

    Args:
        all_tagged: Tagged transactions
            DataFrame

    Returns:
        Same DataFrame if checks pass
        Raises AssertionError if fail
    """
    print("\nValidating tagged transactions...")

    valid_periods = [
        "before_promo",
        "during_promo",
        "after_promo"
    ]

    # Check 1 — Valid period values
    invalid_periods = all_tagged.filter(
        ~F.col("period").isin(valid_periods)
    ).count()
    assert invalid_periods == 0, \
        f"Invalid periods: {invalid_periods}"
    print(f"Invalid periods: {invalid_periods} ✅")

    # Check 2 — No null periods
    null_periods = all_tagged.filter(
        F.col("period").isNull()
    ).count()
    assert null_periods == 0, \
        f"Null periods: {null_periods}"
    print(f"Null periods: {null_periods} ✅")

    # Check 3 — No null loyalty codes
    null_loyalty = all_tagged.filter(
        F.col("LoyaltyCode").isNull()
    ).count()
    assert null_loyalty == 0, \
        f"Null loyalty codes: {null_loyalty}"
    print(f"Null loyalty codes: {null_loyalty} ✅")

    # Check 4 — No null outlets
    null_outlets = all_tagged.filter(
        F.col("OutletIdentifier").isNull()
    ).count()
    assert null_outlets == 0, \
        f"Null outlets: {null_outlets}"
    print(f"Null outlets: {null_outlets} ✅")

    # Check 5 — No negative volumes
    neg_volume = all_tagged.filter(
        F.col("MetricVolume") < 0
    ).count()
    assert neg_volume == 0, \
        f"Negative volumes: {neg_volume}"
    print(f"Negative volumes: {neg_volume} ✅")

    print("All tagging validations passed! ✅")
    return all_tagged


# ============================================
# SUMMARY
# ============================================

def tagging_summary(
    all_tagged: DataFrame
) -> None:
    """
    Prints summary of tagged transactions
    broken down by period.

    Shows:
    → Total tagged transactions
    → Count per period
    → Unique outlets per period
    → Unique loyalty customers per period

    Args:
        all_tagged: Tagged transactions
            DataFrame
    """
    print("\n=== Transaction Tagging Summary ===")

    total = all_tagged.count()
    print(f"Total tagged: {total:,}")

    print("\nBreakdown by period:")
    all_tagged.groupBy("period").agg(
        F.count("*").alias("transactions"),
        F.countDistinct("OutletIdentifier")
         .alias("unique_outlets"),
        F.countDistinct("LoyaltyCode")
         .alias("unique_customers"),
        F.sum("MetricVolume")
         .alias("total_volume"),
        F.sum("MetricValue")
         .alias("total_value")
    ).orderBy("period").display()

    print("===================================")


# ============================================
# USAGE EXAMPLE
# ============================================

if __name__ == "__main__":

    # df_loyalty from 03_loyalty_derivation.py
    # result_windows from 05_overlap_detection.py

    print("Tagging loyalty transactions...")

    # Run complete tagging pipeline
    all_tagged = tag_transactions(
        df_loyalty_valid,
        result_windows
    )

    # Cache — used in metrics calculation
    all_tagged.cache()
    print("Transaction tagging complete ✅")

    # Validate
    all_tagged = validate_tagged_transactions(
        all_tagged
    )

    # Print summary
    tagging_summary(all_tagged)

    print("\nReady for metrics calculation ✅")
    
