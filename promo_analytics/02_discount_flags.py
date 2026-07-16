# ============================================
# 02_discount_flags.py
#
# Classifies each transaction into
# discount types and identifies
# promotional transactions.
#
# Discount Types:
# SMP = Single Manufacturer Promotion
#       Manufacturer pays discount
#       directly to retailer
#
# MP  = Multipack Promotion
#       Discount applied when buying
#       multiple units together
#
# CPN = Coupon Promotion
#       Discount applied via coupon
#
# is_promo flag:
#       1 = transaction has any discount
#       0 = no discount applied
# ============================================

from pyspark.sql import functions as F
from pyspark.sql import DataFrame


# ============================================
# DISCOUNT CLASSIFICATION
# ============================================

def add_smp_discount(df: DataFrame) -> DataFrame:
    """
    Adds Single Manufacturer Promotion
    discount column.

    SMP discount = manufacturer pays
    a fixed amount per unit sold
    directly to the retailer.

    Source column:
    → ManufacturerDiscountValue
    """
    return df.withColumn(
        "SMP_discount",
        F.col("ManufacturerDiscountValue")
    )


def add_mp_discount(df: DataFrame) -> DataFrame:
    """
    Adds Multipack Promotion
    discount column.

    MP discount logic:
    → If ManufacturerMultipackAmount <= 0
      AND OutletMultipackDiscountAmount > 0
      → Use OutletMultipackValue
        (outlet funded multipack)

    → If ManufacturerMultipackAmount > 0
      → Use ManufacturerMultipackValue
        (manufacturer funded multipack)

    → Otherwise → 0 (no multipack discount)
    """
    return df.withColumn(
        "MP_discount",
        F.when(
            (F.col("ManufacturerMultipackAmount")
             <= 0) &
            (F.col("OutletMultipackDiscountAmount")
             > 0),
            F.col("OutletMultipackValue")
        ).when(
            F.col("ManufacturerMultipackAmount") > 0,
            F.col("ManufacturerMultipackValue")
        ).otherwise(0)
    )


def add_cpn_discount(df: DataFrame) -> DataFrame:
    """
    Adds Coupon Promotion
    discount column.

    CPN discount = discount applied
    when customer redeems a coupon
    at point of sale.

    Source column:
    → CouponDiscountValue
    """
    return df.withColumn(
        "CPN_discount",
        F.col("CouponDiscountValue")
    )


def add_is_promo_flag(df: DataFrame) -> DataFrame:
    """
    Adds is_promo flag to identify
    any promotional transaction.

    is_promo = 1 when ANY of:
    → SMP_discount > 0
    → MP_discount > 0
    → CPN_discount > 0

    is_promo = 0 when:
    → All discounts are zero
    → No promotion applied

    Requires:
    → SMP_discount column
    → MP_discount column
    → CPN_discount column

    Run after add_smp_discount,
    add_mp_discount, add_cpn_discount
    """
    return df.withColumn(
        "is_promo",
        F.when(
            (F.col("SMP_discount") > 0) |
            (F.col("MP_discount") > 0) |
            (F.col("CPN_discount") > 0),
            1
        ).otherwise(0)
    )


# ============================================
# MAIN FUNCTION
# Applies all discount classifications
# in correct order
# ============================================

def add_discount_columns(df: DataFrame) -> DataFrame:
    """
    Applies all discount classifications
    to transaction dataframe.

    Adds 4 new columns:
    → SMP_discount: manufacturer discount
    → MP_discount:  multipack discount
    → CPN_discount: coupon discount
    → is_promo:     any promo flag (0/1)

    Args:
        df: Transaction DataFrame with
            discount source columns

    Returns:
        DataFrame with discount columns added

    Example:
        df_scan = add_discount_columns(df_scan)
    """
    return df \
        .transform(add_smp_discount) \
        .transform(add_mp_discount) \
        .transform(add_cpn_discount) \
        .transform(add_is_promo_flag)


# ============================================
# VALIDATION
# ============================================

def validate_discount_columns(
    df: DataFrame
) -> DataFrame:
    """
    Validates discount columns after
    classification.

    Checks:
    → No negative discount values
    → is_promo is only 0 or 1
    → All discount columns exist

    Args:
        df: DataFrame after discount
            columns added

    Returns:
        Same DataFrame if all checks pass
        Raises AssertionError if checks fail
    """
    print("\nValidating discount columns...")

    # Check 1 — No negative SMP
    neg_smp = df.filter(
        F.col("SMP_discount") < 0
    ).count()
    assert neg_smp == 0, \
        f"Negative SMP discounts: {neg_smp}"
    print(f"Negative SMP: {neg_smp} ✅")

    # Check 2 — No negative MP
    neg_mp = df.filter(
        F.col("MP_discount") < 0
    ).count()
    assert neg_mp == 0, \
        f"Negative MP discounts: {neg_mp}"
    print(f"Negative MP: {neg_mp} ✅")

    # Check 3 — No negative CPN
    neg_cpn = df.filter(
        F.col("CPN_discount") < 0
    ).count()
    assert neg_cpn == 0, \
        f"Negative CPN discounts: {neg_cpn}"
    print(f"Negative CPN: {neg_cpn} ✅")

    # Check 4 — is_promo only 0 or 1
    invalid_flag = df.filter(
        ~F.col("is_promo").isin(0, 1)
    ).count()
    assert invalid_flag == 0, \
        f"Invalid is_promo values: {invalid_flag}"
    print(f"Invalid promo flags: {invalid_flag} ✅")

    print("All discount validations passed! ✅")
    return df


# ============================================
# SUMMARY
# ============================================

def discount_summary(df: DataFrame) -> None:
    """
    Prints summary of discount
    classifications for QA purposes.

    Shows:
    → Total transactions
    → SMP transaction count
    → MP transaction count
    → CPN transaction count
    → Total promo transactions
    → Promo percentage

    Args:
        df: DataFrame with discount
            columns added
    """
    total = df.count()
    smp_count = df.filter(
        F.col("SMP_discount") > 0
    ).count()
    mp_count = df.filter(
        F.col("MP_discount") > 0
    ).count()
    cpn_count = df.filter(
        F.col("CPN_discount") > 0
    ).count()
    promo_count = df.filter(
        F.col("is_promo") == 1
    ).count()

    print("\n=== Discount Summary ===")
    print(f"Total transactions:  {total:,}")
    print(f"SMP transactions:    {smp_count:,}")
    print(f"MP transactions:     {mp_count:,}")
    print(f"CPN transactions:    {cpn_count:,}")
    print(f"Total promo:         {promo_count:,}")
    print(
        f"Promo percentage:    "
        f"{(promo_count/total*100):.1f}%"
    )
    print("========================")


# ============================================
# USAGE EXAMPLE
# ============================================

if __name__ == "__main__":

    # Load scan data
    # (from 01_load_data.py)
    from load_data import load_scan_table
    df_scan = load_scan_table()

    print("Adding discount columns...")

    # Apply discount classifications
    df_scan = add_discount_columns(df_scan)
    print("Discount columns added ✅")

    # Validate
    df_scan = validate_discount_columns(df_scan)

    # Print summary
    discount_summary(df_scan)

    # Cache for next steps
    df_scan.cache()
    print("\nReady for next step ✅")
