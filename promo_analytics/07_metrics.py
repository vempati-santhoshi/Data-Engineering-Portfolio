# ============================================
# 07_metrics.py
#
# Calculates final promotional metrics
# for before, during and after periods.
#
# Business Problem:
# After tagging each transaction with
# its promotional period we need to
# aggregate into summary metrics that
# answer the core business questions:
#
# 1. How much volume was sold?
#    → Compare before vs during vs after
#    → Did the promotion drive volume?
#
# 2. How much revenue was generated?
#    → Total sales value per period
#    → Was the promotion profitable?
#
# 3. How many loyal customers bought?
#    → Unique loyalty customer count
#    → Did promo attract new loyal buyers?
#    → Did loyal buyers return after promo?
#
# Output Format:
# Wide table — one row per promotion
# with separate columns for each metric
# and each period.
#
# Example output row:
# Outlet: 001817
# Brand:  your_brand
# Promo:  Promo_name
# Start:  2025-03-30
# End:    2025-09-27
# before_promo_total_volume:          1153
# before_promo_total_value:           8579
# before_promo_total_loyalty_customers: 45
# during_promo_total_volume:          5000
# during_promo_total_value:          12000
# during_promo_total_loyalty_customers: 230
# after_promo_total_volume:           null
# after_promo_total_value:            null
# after_promo_total_loyalty_customers: null
#
# Null values mean that period was not
# calculable for that promotion.
# (overlap detected or no data available)
# ============================================

from pyspark.sql import functions as F
from pyspark.sql import DataFrame
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()


# ============================================
# CONFIGURATION
# ============================================

# Output table name
OUTPUT_TABLE = "your_schema.promo_metrics"

# Period order for pivot
PERIODS = [
    "before_promo",
    "during_promo",
    "after_promo"
]

# Group by columns for aggregation
GROUP_BY_COLS = [
    "OutletIdentifier",
    "ReportingBrandName",
    "PromotionName",
    "PromoTrueStartDate_dt",
    "PromoTrueEndDate_dt",
    "period"
]

# Group by columns for pivot
PIVOT_GROUP_BY_COLS = [
    "OutletIdentifier",
    "ReportingBrandName",
    "PromotionName",
    "PromoTrueStartDate_dt",
    "PromoTrueEndDate_dt"
]


# ============================================
# STEP 1 — AGGREGATE BY PERIOD
# Calculate metrics for each
# outlet + brand + promo + period
# combination
# ============================================

def aggregate_by_period(
    all_tagged: DataFrame
) -> DataFrame:
    """
    Aggregates tagged transactions
    into metrics by period.

    Groups by:
    → OutletIdentifier
    → ReportingBrandName
    → PromotionName
    → PromoTrueStartDate_dt
    → PromoTrueEndDate_dt
    → period

    Calculates:
    → total_volume: sum of MetricVolume
      Total units sold in period

    → total_value: sum of MetricValue
      Total sales revenue in period

    → total_loyalty_customers:
      count distinct LoyaltyCode
      Unique loyal customers in period

    Why countDistinct for customers?
    → Same customer can buy multiple
      times in same period
    → We want unique customer count
    → Not transaction count

    Args:
        all_tagged: Tagged transactions
            DataFrame from
            06_tag_transactions.py

    Returns:
        Long format DataFrame with
        one row per outlet+brand+
        promo+period combination
    """
    return all_tagged.groupBy(
        GROUP_BY_COLS
    ).agg(
        F.sum("MetricVolume")
         .alias("total_volume"),
        F.sum("MetricValue")
         .alias("total_value"),
        F.countDistinct("LoyaltyCode")
         .alias("total_loyalty_customers")
    )


# ============================================
# STEP 2 — PIVOT TO WIDE FORMAT
# Transform from long to wide format
# One row per outlet + brand + promo
# ============================================

def pivot_to_wide_format(
    metrics_long: DataFrame
) -> DataFrame:
    """
    Pivots metrics from long to wide
    format for reporting.

    Long format (before pivot):
    OutletId | Brand | Promo | Period        | Volume
    001817xx   | your_brand1  | P-001 | before_promo  | 1153
    001817xx   | your_brand1  | P-001 | during_promo  | 5000
    001817xx   | your_brand1  | P-001 | after_promo   | null

    Wide format (after pivot):
    OutletId | Brand | Promo | before_volume | during_volume | after_volume
    001817xx   | your_brand1  | P-001 | 1153          | 5000          | null

    Why wide format?
    → Easier for business users
    → Can be consumed by AI agents
    → Simple to compare periods
      side by side in a table
    → Standard format for reports

    Null values in output:
    → Period not calculable
    → Overlap was detected
    → Or data not available

    Args:
        metrics_long: Long format metrics
            DataFrame from
            aggregate_by_period()

    Returns:
        Wide format DataFrame with
        one row per outlet+brand+promo
        and separate columns per period
    """
    return metrics_long.groupBy(
        PIVOT_GROUP_BY_COLS
    ).pivot(
        "period",
        PERIODS
    ).agg(
        F.first("total_volume")
         .alias("total_volume"),
        F.first("total_value")
         .alias("total_value"),
        F.first("total_loyalty_customers")
         .alias("total_loyalty_customers")
    )


# ============================================
# STEP 3 — ADD DERIVED METRICS
# Calculate additional metrics from
# the aggregated data
# ============================================

def add_derived_metrics(
    metrics_wide: DataFrame
) -> DataFrame:
    """
    Adds derived comparison metrics
    to the wide format table.

    Derived metrics:
    → volume_lift: during vs before
      How much did volume increase?
      Formula: during / before - 1
      Example: 5000/1153 - 1 = 3.34
               = 334% volume lift

    → customer_retention_rate:
      after vs during loyal customers
      What % of promo customers stayed?
      Formula: after / during
      Example: 45/230 = 0.196
               = 19.6% retention

    → value_per_customer_during:
      Average spend per loyal customer
      during the promotion
      Formula: during_value / during_customers
      Example: 12000/230 = $52.17

    Note: All derived metrics are null
    when source periods are null
    (safe division handles this)

    Args:
        metrics_wide: Wide format metrics
            DataFrame from
            pivot_to_wide_format()

    Returns:
        DataFrame with additional
        derived metric columns
    """
    return metrics_wide \
        .withColumn(
            "volume_lift_pct",
            F.when(
                F.col("before_promo_total_volume")
                .isNotNull() &
                (F.col("before_promo_total_volume")
                 > 0),
                (
                    F.col("during_promo_total_volume")
                    / F.col("before_promo_total_volume")
                    - 1
                ) * 100
            )
        ).withColumn(
            "customer_retention_rate",
            F.when(
                F.col("during_promo_total_loyalty_customers")
                .isNotNull() &
                (F.col("during_promo_total_loyalty_customers")
                 > 0),
                F.col("after_promo_total_loyalty_customers")
                / F.col(
                    "during_promo_total_loyalty_customers"
                ) * 100
            )
        ).withColumn(
            "value_per_loyal_customer_during",
            F.when(
                F.col(
                    "during_promo_total_loyalty_customers"
                ).isNotNull() &
                (F.col(
                    "during_promo_total_loyalty_customers"
                ) > 0),
                F.col("during_promo_total_value")
                / F.col(
                    "during_promo_total_loyalty_customers"
                )
            )
        )


# ============================================
# STEP 4 — SAVE TO TABLE
# ============================================

def save_metrics(
    metrics: DataFrame,
    table_name: str = OUTPUT_TABLE
) -> None:
    """
    Saves final metrics to Delta table.

    Uses:
    → mode overwrite: replaces existing
      table with new data
    → mergeSchema true: allows adding
      new columns without error

    Args:
        metrics: Final metrics DataFrame
        table_name: Target Delta table
            name to save to

    Returns:
        None
    """
    print(f"\nSaving to: {table_name}")

    metrics.write \
        .mode("overwrite") \
        .option("mergeSchema", "true") \
        .saveAsTable(table_name)

    print(f"Saved successfully ✅")


# ============================================
# MAIN FUNCTION
# Complete metrics calculation pipeline
# ============================================

def calculate_metrics(
    all_tagged: DataFrame,
    include_derived: bool = True,
    save_to_table: bool = True
) -> DataFrame:
    """
    Complete metrics calculation pipeline.

    Steps:
    1. Aggregate tagged transactions
       by period
    2. Pivot to wide format
    3. Add derived metrics (optional)
    4. Save to Delta table (optional)

    Args:
        all_tagged: Tagged transactions
            DataFrame from
            06_tag_transactions.py
        include_derived: Whether to add
            derived metrics like
            volume lift and retention
            Default: True
        save_to_table: Whether to save
            to Delta table
            Default: True

    Returns:
        Final metrics DataFrame in
        wide format

    Output table columns:
    → OutletIdentifier
    → ReportingBrandName
    → PromotionName
    → PromoTrueStartDate_dt
    → PromoTrueEndDate_dt
    → before_promo_total_volume
    → before_promo_total_value
    → before_promo_total_loyalty_customers
    → during_promo_total_volume
    → during_promo_total_value
    → during_promo_total_loyalty_customers
    → after_promo_total_volume
    → after_promo_total_value
    → after_promo_total_loyalty_customers
    → volume_lift_pct (if derived=True)
    → customer_retention_rate (if derived=True)
    → value_per_loyal_customer_during (if derived=True)
    """
    print("Calculating promo metrics...")

    # Step 1 — Aggregate by period
    print("Aggregating by period...")
    metrics_long = aggregate_by_period(
        all_tagged
    )

    # Step 2 — Pivot to wide format
    print("Pivoting to wide format...")
    metrics_wide = pivot_to_wide_format(
        metrics_long
    )

    # Step 3 — Add derived metrics
    if include_derived:
        print("Adding derived metrics...")
        metrics_wide = add_derived_metrics(
            metrics_wide
        )

    # Step 4 — Save to table
    if save_to_table:
        save_metrics(metrics_wide)

    return metrics_wide


# ============================================
# VALIDATION
# ============================================

def validate_metrics(
    metrics: DataFrame
) -> DataFrame:
    """
    Validates final metrics table
    before saving to production.

    Checks:
    → No null outlet identifiers
    → No null promotion names
    → No null brand names
    → No null promo dates
    → No negative volumes
    → During metrics always present
      (every promo has during period)
    → Volume lift is reasonable range

    Args:
        metrics: Final metrics DataFrame

    Returns:
        Same DataFrame if checks pass
        Raises AssertionError if fail
    """
    print("\nValidating final metrics...")

    # Check 1 — No null outlets
    null_outlets = metrics.filter(
        F.col("OutletIdentifier").isNull()
    ).count()
    assert null_outlets == 0, \
        f"Null outlets: {null_outlets}"
    print(f"Null outlets: {null_outlets} ✅")

    # Check 2 — No null promos
    null_promos = metrics.filter(
        F.col("PromotionName").isNull()
    ).count()
    assert null_promos == 0, \
        f"Null promos: {null_promos}"
    print(f"Null promos: {null_promos} ✅")

    # Check 3 — No null brands
    null_brands = metrics.filter(
        F.col("ReportingBrandName").isNull()
    ).count()
    assert null_brands == 0, \
        f"Null brands: {null_brands}"
    print(f"Null brands: {null_brands} ✅")

    # Check 4 — No null promo dates
    null_dates = metrics.filter(
        F.col("PromoTrueStartDate_dt").isNull() |
        F.col("PromoTrueEndDate_dt").isNull()
    ).count()
    assert null_dates == 0, \
        f"Null dates: {null_dates}"
    print(f"Null dates: {null_dates} ✅")

    # Check 5 — No negative during volume
    neg_volume = metrics.filter(
        (F.col("during_promo_total_volume")
         .isNotNull()) &
        (F.col("during_promo_total_volume") < 0)
    ).count()
    assert neg_volume == 0, \
        f"Negative volumes: {neg_volume}"
    print(f"Negative volumes: {neg_volume} ✅")

    # Check 6 — During always present
    null_during = metrics.filter(
        F.col("during_promo_total_volume")
         .isNull()
    ).count()
    print(
        f"Promos with no during data: "
        f"{null_during}"
    )

    print("All metric validations passed! ✅")
    return metrics


# ============================================
# SUMMARY
# ============================================

def metrics_summary(
    metrics: DataFrame
) -> None:
    """
    Prints summary of final metrics
    for QA and business reporting.

    Shows:
    → Total promotions in table
    → Promotions with before metrics
    → Promotions with after metrics
    → Average volume lift
    → Average customer retention
    → Top brands by during volume

    Args:
        metrics: Final metrics DataFrame
    """
    total = metrics.count()

    before_count = metrics.filter(
        F.col("before_promo_total_volume")
         .isNotNull()
    ).count()

    after_count = metrics.filter(
        F.col("after_promo_total_volume")
         .isNotNull()
    ).count()

    print("\n=== Final Metrics Summary ===")
    print(f"Total promotions:         {total:,}")
    print(
        f"With before metrics:      "
        f"{before_count:,} "
        f"({before_count/total*100:.1f}%)"
    )
    print(
        f"With after metrics:       "
        f"{after_count:,} "
        f"({after_count/total*100:.1f}%)"
    )

    # Average volume lift
    avg_lift = metrics.filter(
        F.col("volume_lift_pct").isNotNull()
    ).agg(
        F.avg("volume_lift_pct").alias("avg_lift")
    ).collect()[0]["avg_lift"]

    if avg_lift:
        print(
            f"Avg volume lift:          "
            f"{avg_lift:.1f}%"
        )

    # Average retention
    avg_retention = metrics.filter(
        F.col("customer_retention_rate")
         .isNotNull()
    ).agg(
        F.avg("customer_retention_rate")
         .alias("avg_retention")
    ).collect()[0]["avg_retention"]

    if avg_retention:
        print(
            f"Avg customer retention:   "
            f"{avg_retention:.1f}%"
        )

    print("\nTop brands by during volume:")
    metrics.groupBy(
        "ReportingBrandName"
    ).agg(
        F.sum("during_promo_total_volume")
         .alias("total_volume"),
        F.sum("during_promo_total_value")
         .alias("total_value"),
        F.sum(
            "during_promo_total_loyalty_customers"
        ).alias("total_customers")
    ).orderBy(
        F.col("total_volume").desc()
    ).display()

    print("=============================")


# ============================================
# USAGE EXAMPLE
# ============================================

if __name__ == "__main__":

    # all_tagged from 06_tag_transactions.py

    print("Calculating final metrics...")

    # Run complete metrics pipeline
    metrics = calculate_metrics(
        all_tagged,
        include_derived=True,
        save_to_table=True
    )

    # Validate
    metrics = validate_metrics(metrics)

    # Print summary
    metrics_summary(metrics)

    # Preview results
    print("\nSample output:")
    metrics.limit(20).display()

    print("\nPipeline complete! ✅")
    print(f"Table: {OUTPUT_TABLE}")
    
