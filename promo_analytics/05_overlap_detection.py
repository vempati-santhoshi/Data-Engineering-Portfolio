# ============================================
# 05_overlap_detection.py
#
# Detects if any other promotion is
# running during the before or after
# candidate windows.
#
# Business Problem:
# When calculating before and after
# promo performance we need clean
# baseline periods with no other
# promotional activity.
#
# If another promotion is running
# during the before window:
# → The "baseline" is contaminated
# → We cannot accurately measure
#   the true before performance
# → That window must be excluded
#
# Same applies to after window:
# → If another promo starts immediately
#   after the first one ends
# → We cannot measure true post-promo
#   customer behaviour
# → That window must be excluded
#
# Real World Example:
# Outlet 001817 / Brand Kool:
#
# |--P1 182d--|--P2 91d--|--P3 91d--|--P4 91d--|
#
# P1 after window → blocked by P2 ❌
# P2 before window → blocked by P1 ❌
# P2 after window → blocked by P3 ❌
# P3 before window → blocked by P2 ❌
# P3 after window → blocked by P4 ❌
# P4 before window → blocked by P3 ❌
# P4 after window → no promo after ✅
#
# Solution:
# Self join each promo against all
# OTHER promos for same outlet+brand
# and check for date range overlap
# ============================================

from pyspark.sql import functions as F
from pyspark.sql import DataFrame
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

TODAY = F.current_date()


# ============================================
# STEP 1 — CREATE OTHER PROMOS TABLE
# Slim version used for self join
# ============================================

def get_other_promos(
    result_dur: DataFrame
) -> DataFrame:
    """
    Creates slim version of promo
    dataframe for self join.

    Only keeps columns needed to
    check date overlap:
    → OutletIdentifier
    → ReportingBrandName
    → PromotionName
    → PromoTrueStartDate_dt
    → PromoTrueEndDate_dt

    This reduces shuffle size during
    the self join on large datasets.

    Args:
        result_dur: Promo windows DataFrame
            from 04_promo_windows.py

    Returns:
        Slim DataFrame for self join
    """
    return result_dur.select(
        "OutletIdentifier",
        "ReportingBrandName",
        "PromotionName",
        "PromoTrueStartDate_dt",
        "PromoTrueEndDate_dt"
    )


# ============================================
# STEP 2 — SELF JOIN
# Compare each promo against all other
# promos for same outlet + brand
# ============================================

def self_join_promos(
    result_dur: DataFrame,
    other_promos: DataFrame
) -> DataFrame:
    """
    Performs self join to compare each
    promotion against all other promotions
    for the same outlet and brand.

    Join conditions:
    → Same OutletIdentifier
    → Same ReportingBrandName
    → DIFFERENT PromotionName
      (exclude comparing promo to itself)

    Join type: LEFT
    → Keeps promos with no other promos
      (single promo outlets)
    → Those will have null "other" columns
    → Treated as no overlap (overlap = 0)

    Why self join?
    → We need to check each promo's
      candidate windows against ALL
      other promos for same outlet+brand
    → Self join is the most efficient
      way to do this in Spark

    Args:
        result_dur: Main promo DataFrame
            aliased as "main"
        other_promos: Slim promo DataFrame
            aliased as "other"

    Returns:
        Joined DataFrame before aggregation
    """
    return result_dur.alias("main").join(
        other_promos.alias("other"),
        on=[
            # Same outlet
            F.col("main.OutletIdentifier") ==
            F.col("other.OutletIdentifier"),

            # Same brand
            F.col("main.ReportingBrandName") ==
            F.col("other.ReportingBrandName"),

            # Different promotion
            # (don't compare promo to itself)
            F.col("main.PromotionName") !=
            F.col("other.PromotionName")
        ],
        how="left"
    )


# ============================================
# STEP 3 — DETECT OVERLAPS
# Check if other promo dates overlap
# with candidate windows
# ============================================

def detect_before_overlap(
    joined_df: DataFrame
) -> str:
    """
    Overlap detection logic for
    before candidate window.

    A before window overlap occurs when:
    → other promo start <= before window end
      AND
    → other promo end >= before window start

    This is the standard date range
    overlap check:
    Two ranges [A,B] and [C,D] overlap
    when A <= D AND B >= C

    Returns:
    → 1 if any other promo overlaps
    → 0 if no overlap found
    → 0 if no other promos exist
      (null from left join = no overlap)
    """
    return F.max(
        (
            (
                F.col("other.PromoTrueStartDate_dt")
                <= F.col("main.before_candidate_end")
            ) &
            (
                F.col("other.PromoTrueEndDate_dt")
                >= F.col("main.before_candidate_start")
            )
        ).cast("int")
    ).alias("before_overlap")


def detect_after_overlap(
    joined_df: DataFrame
) -> str:
    """
    Overlap detection logic for
    after candidate window.

    An after window overlap occurs when:
    → other promo start <= after window end
      AND
    → other promo end >= after window start

    Returns:
    → 1 if any other promo overlaps
    → 0 if no overlap found
    → 0 if no other promos exist
    """
    return F.max(
        (
            (
                F.col("other.PromoTrueStartDate_dt")
                <= F.col("main.after_candidate_end")
            ) &
            (
                F.col("other.PromoTrueEndDate_dt")
                >= F.col("main.after_candidate_start")
            )
        ).cast("int")
    ).alias("after_overlap")


# ============================================
# STEP 4 — AGGREGATE OVERLAP RESULTS
# Group back to one row per promo
# with overlap flags
# ============================================

def aggregate_overlaps(
    joined_df: DataFrame
) -> DataFrame:
    """
    Aggregates self join results back
    to one row per outlet+brand+promo.

    Uses MAX aggregation for overlap flags:
    → If ANY other promo overlaps → 1
    → If NO other promo overlaps → 0

    Keeps all original promo window
    columns for downstream processing.

    Args:
        joined_df: Self joined DataFrame

    Returns:
        DataFrame with one row per promo
        and overlap flags added
    """
    return joined_df.groupBy(
        "main.OutletIdentifier",
        "main.ReportingBrandName",
        "main.PromotionName",
        "main.PromoTrueStartDate_dt",
        "main.PromoTrueEndDate_dt",
        "main.promo_duration_days",
        "main.before_candidate_start",
        "main.before_candidate_end",
        "main.after_candidate_start",
        "main.after_candidate_end",
        "main.before_data_ok",
        "main.after_data_ok"
    ).agg(
        detect_before_overlap(joined_df),
        detect_after_overlap(joined_df)
    )


# ============================================
# STEP 5 — MARK VALID WINDOWS
# Combine overlap and data availability
# to determine final valid windows
# ============================================

def mark_valid_windows(
    result_overlap: DataFrame
) -> DataFrame:
    """
    Marks which before and after windows
    are actually calculable.

    A window is VALID when BOTH:
    → No overlap (overlap_flag == 0)
    → Data available (data_ok == True)

    A window is INVALID when EITHER:
    → Another promo overlaps it
    → Data not available

    Sets actual window dates:
    → Valid window → dates populated
    → Invalid window → dates are null
      (null signals "not calculable")

    After window end is clipped to
    today to avoid future dates.

    Args:
        result_overlap: DataFrame with
            overlap flags from aggregation

    Returns:
        DataFrame with final valid window
        flags and actual window dates
    """
    return result_overlap \
        .withColumn(
            # Valid before = data ok
            # AND no overlap
            "has_before_window",
            F.col("before_data_ok") &
            (F.col("before_overlap") == 0)
        ).withColumn(
            # Valid after = data ok
            # AND no overlap
            "has_after_window",
            F.col("after_data_ok") &
            (F.col("after_overlap") == 0)
        ).withColumn(
            # Only set dates if valid
            "before_window_start",
            F.when(
                F.col("has_before_window"),
                F.col("before_candidate_start")
            )
        ).withColumn(
            "before_window_end",
            F.when(
                F.col("has_before_window"),
                F.col("before_candidate_end")
            )
        ).withColumn(
            "after_window_start",
            F.when(
                F.col("has_after_window"),
                F.col("after_candidate_start")
            )
        ).withColumn(
            # Clip to today if extends
            # into future
            "after_window_end",
            F.when(
                F.col("has_after_window"),
                F.least(
                    F.col("after_candidate_end"),
                    TODAY
                )
            )
        )


# ============================================
# MAIN FUNCTION
# Complete overlap detection pipeline
# ============================================

def detect_overlaps(
    result_dur: DataFrame
) -> DataFrame:
    """
    Complete overlap detection pipeline.

    Steps:
    1. Create slim other_promos table
    2. Self join main vs other promos
    3. Detect before window overlaps
    4. Detect after window overlaps
    5. Aggregate to one row per promo
    6. Mark valid windows

    This is the most computationally
    intensive step in the pipeline.

    On 782,707 promotions this typically
    takes 3-8 minutes depending on
    cluster size.

    Args:
        result_dur: Promo windows DataFrame
            from 04_promo_windows.py

    Returns:
        DataFrame with one row per promo
        with valid window flags and dates

    Output columns (additional):
    → before_overlap: 1 if blocked
    → after_overlap:  1 if blocked
    → has_before_window: True if valid
    → has_after_window:  True if valid
    → before_window_start: date or null
    → before_window_end:   date or null
    → after_window_start:  date or null
    → after_window_end:    date or null
    """
    print("Starting overlap detection...")
    print("This may take 3-8 minutes...")

    # Step 1 — Create slim other promos
    other_promos = get_other_promos(result_dur)

    # Step 2 — Self join
    joined = self_join_promos(
        result_dur,
        other_promos
    )

    # Steps 3 & 4 — Detect overlaps
    # Steps 5 — Aggregate
    result_overlap = aggregate_overlaps(joined)

    # Step 6 — Mark valid windows
    result_windows = mark_valid_windows(
        result_overlap
    )

    return result_windows


# ============================================
# VALIDATION
# ============================================

def validate_overlap_detection(
    result_dur: DataFrame,
    result_windows: DataFrame
) -> None:
    """
    Validates overlap detection results.

    Checks:
    → Row count preserved after self join
    → Overlap flags are only 0 or 1
    → Window dates are null when
      has_window is False
    → Window dates are not null when
      has_window is True

    Args:
        result_dur: Input promo DataFrame
        result_windows: Output DataFrame
            after overlap detection
    """
    print("\nValidating overlap detection...")

    input_count = result_dur.count()
    output_count = result_windows.count()

    # Check 1 — Row count preserved
    assert input_count == output_count, \
        f"Row count changed! " \
        f"Input: {input_count} " \
        f"Output: {output_count}"
    print(f"Row count preserved: {output_count:,} ✅")

    # Check 2 — Overlap flags valid
    invalid_before_flag = result_windows.filter(
        ~F.col("before_overlap").isin(0, 1)
    ).count()
    assert invalid_before_flag == 0, \
        f"Invalid before flags: {invalid_before_flag}"
    print(
        f"Invalid before flags: "
        f"{invalid_before_flag} ✅"
    )

    invalid_after_flag = result_windows.filter(
        ~F.col("after_overlap").isin(0, 1)
    ).count()
    assert invalid_after_flag == 0, \
        f"Invalid after flags: {invalid_after_flag}"
    print(
        f"Invalid after flags: "
        f"{invalid_after_flag} ✅"
    )

    # Check 3 — Null dates when invalid
    wrong_before_null = result_windows.filter(
        (~F.col("has_before_window")) &
        (F.col("before_window_start").isNotNull())
    ).count()
    assert wrong_before_null == 0, \
        f"Before dates set when invalid: " \
        f"{wrong_before_null}"
    print(
        f"Wrong before nulls: "
        f"{wrong_before_null} ✅"
    )

    wrong_after_null = result_windows.filter(
        (~F.col("has_after_window")) &
        (F.col("after_window_start").isNotNull())
    ).count()
    assert wrong_after_null == 0, \
        f"After dates set when invalid: " \
        f"{wrong_after_null}"
    print(
        f"Wrong after nulls: "
        f"{wrong_after_null} ✅"
    )

    print("All overlap validations passed! ✅")


# ============================================
# SUMMARY
# ============================================

def overlap_summary(
    result_windows: DataFrame
) -> None:
    """
    Prints summary of overlap detection
    results for QA and reporting.

    Shows breakdown of calculable vs
    non-calculable windows across
    all promotions.

    Args:
        result_windows: DataFrame after
            overlap detection
    """
    total = result_windows.count()

    both = result_windows.filter(
        F.col("has_before_window") &
        F.col("has_after_window")
    ).count()

    before_only = result_windows.filter(
        F.col("has_before_window") &
        ~F.col("has_after_window")
    ).count()

    after_only = result_windows.filter(
        ~F.col("has_before_window") &
        F.col("has_after_window")
    ).count()

    during_only = result_windows.filter(
        ~F.col("has_before_window") &
        ~F.col("has_after_window")
    ).count()

    print("\n=== Overlap Detection Summary ===")
    print(f"Total promotions:      {total:,}")
    print(
        f"Both windows valid:    "
        f"{both:,} "
        f"({both/total*100:.1f}%)"
    )
    print(
        f"Before only:           "
        f"{before_only:,} "
        f"({before_only/total*100:.1f}%)"
    )
    print(
        f"After only:            "
        f"{after_only:,} "
        f"({after_only/total*100:.1f}%)"
    )
    print(
        f"During only:           "
        f"{during_only:,} "
        f"({during_only/total*100:.1f}%)"
    )
    print("=================================")


# ============================================
# VALIDATION HELPER
# Check specific outlet and brand
# ============================================

def validate_specific_outlet(
    result_windows: DataFrame,
    outlet_id: str,
    brand_name: str
) -> None:
    """
    Validates overlap logic for a
    specific outlet and brand.

    Use this during development to
    verify the logic is working
    correctly before running the
    full pipeline.

    Args:
        result_windows: DataFrame after
            overlap detection
        outlet_id: OutletIdentifier
            to check
        brand_name: ReportingBrandName
            to check

    Example:
        validate_specific_outlet(
            result_windows,
            "your_outlet_id",
            "your_brand_name"
        )
    """
    print(
        f"\nChecking outlet: {outlet_id} "
        f"brand: {brand_name}"
    )

    result_windows.filter(
        (F.col("OutletIdentifier") == outlet_id) &
        (F.col("ReportingBrandName") == brand_name)
    ).select(
        "PromotionName",
        "PromoTrueStartDate_dt",
        "PromoTrueEndDate_dt",
        "promo_duration_days",
        "before_overlap",
        "after_overlap",
        "has_before_window",
        "has_after_window",
        "before_window_start",
        "before_window_end",
        "after_window_start",
        "after_window_end"
    ).orderBy(
        "PromoTrueStartDate_dt"
    ).display()


# ============================================
# USAGE EXAMPLE
# ============================================

if __name__ == "__main__":

    # result_dur from 04_promo_windows.py
    print("Running overlap detection...")

    # Run complete overlap detection
    result_windows = detect_overlaps(result_dur)

    # Cache — used in multiple next steps
    result_windows.cache()
    print("Overlap detection complete ✅")

    # Validate results
    validate_overlap_detection(
        result_dur,
        result_windows
    )

    # Print summary
    overlap_summary(result_windows)

    # Validate specific outlet
    # Replace with real values to check
    validate_specific_outlet(
        result_windows,
        outlet_id="your_outlet_id",
        brand_name="your_brand_name"
    )

    print("\nReady for transaction tagging ✅")
    print("Next: 06_tag_transactions.py")
