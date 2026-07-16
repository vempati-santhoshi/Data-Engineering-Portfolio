# ============================================
# 03_loyalty_derivation.py
#
# Derives unique loyalty codes for
# each customer transaction.
#
# Business Problem:
# Raw data contains a LoyaltyNumber
# field but it is not unique across
# different retail chains.
#
# The same LoyaltyNumber can exist
# in multiple chains meaning two
# different customers could have
# the same number.
#
# Solution:
# Combine SubmitterName (chain identifier)
# with LoyaltyNumber to create a
# truly unique customer identifier.
#
# Example:
# Chain A + 12345 = "ChainA12345"
# Chain B + 12345 = "ChainB12345"
# These are now two different customers
# ============================================

from pyspark.sql import functions as F
from pyspark.sql import DataFrame


# ============================================
# VALIDATION RULES
# A loyalty code is only valid when:
# → LoyaltyNumber is not null
# → LoyaltyNumber is not empty string
# → LoyaltyNumber is not "0"
# → SubmitterName is not null
# ============================================

def is_valid_loyalty_number(
    df: DataFrame
) -> DataFrame:
    """
    Adds a flag to identify valid
    loyalty numbers before derivation.

    Valid loyalty number requires:
    → Not null
    → Not empty string ""
    → Not "0" (default/missing value)

    Args:
        df: Transaction DataFrame with
            LoyaltyNumber column

    Returns:
        DataFrame with is_valid_loyalty
        boolean column added
    """
    return df.withColumn(
        "is_valid_loyalty",
        F.col("LoyaltyNumber").isNotNull() &
        ~F.col("LoyaltyNumber").isin("", "0")
    )


def is_valid_submitter(
    df: DataFrame
) -> DataFrame:
    """
    Adds a flag to identify valid
    submitter names before derivation.

    Valid submitter requires:
    → SubmitterName is not null

    Args:
        df: Transaction DataFrame with
            SubmitterName column

    Returns:
        DataFrame with is_valid_submitter
        boolean column added
    """
    return df.withColumn(
        "is_valid_submitter",
        F.col("SubmitterName").isNotNull()
    )


# ============================================
# LOYALTY CODE DERIVATION
# ============================================

def derive_loyalty_code(
    df: DataFrame
) -> DataFrame:
    """
    Creates unique loyalty code by
    combining SubmitterName and
    LoyaltyNumber.

    Logic:
    → If LoyaltyNumber is valid
      AND SubmitterName is valid
      → LoyaltyCode = SubmitterName
                    + LoyaltyNumber
    → Otherwise
      → LoyaltyCode = null

    Example:
    SubmitterName = "STORE_A"
    LoyaltyNumber = "12345"
    LoyaltyCode   = "STORE_A12345"

    Args:
        df: Transaction DataFrame with
            LoyaltyNumber and
            SubmitterName columns

    Returns:
        DataFrame with LoyaltyCode
        column added
    """
    return df.withColumn(
        "LoyaltyCode",
        F.when(
            F.col("LoyaltyNumber").isNotNull() &
            ~F.col("LoyaltyNumber").isin("", "0") &
            F.col("SubmitterName").isNotNull(),
            F.concat(
                F.col("SubmitterName"),
                F.col("LoyaltyNumber")
            )
        )
    )


def filter_valid_loyalty(
    df: DataFrame
) -> DataFrame:
    """
    Filters dataframe to only rows
    with valid derived loyalty codes.

    Removes:
    → Transactions with no loyalty number
    → Transactions with invalid numbers
    → Transactions with no submitter

    This is the ITGB loyalty dataset
    used for all downstream analysis.

    Args:
        df: DataFrame with LoyaltyCode
            column derived

    Returns:
        DataFrame with only valid
        loyalty transactions
    """
    return df.filter(
        F.col("LoyaltyCode").isNotNull()
    )


# ============================================
# BRAND NAME STANDARDISATION
#
# Some brand names need standardisation
# for consistent reporting.
# Applied after loyalty derivation.
# ============================================

def standardise_brand_names(
    df: DataFrame
) -> DataFrame:
    """
    Standardises ReportingBrandName
    for consistent reporting.

    Rules applied:
    → Brand X + Variant A
      → Standardised Name A
    → Brand X + Variant B
      → Standardised Name B
    → All others → unchanged

    Args:
        df: DataFrame with
            ReportingBrandName and
            Brand columns

    Returns:
        DataFrame with standardised
        ReportingBrandName column
    """
    return df \
        .withColumn(
            "ReportingBrandName",
            F.when(
                (F.col("ReportingBrandName")
                 == "Brand_X") &
                (F.col("Brand") == "Variant_A"),
                "Standardised_Name_A"
            ).when(
                (F.col("ReportingBrandName")
                 == "Brand_X") &
                (F.col("Brand") == "Variant_B"),
                "Standardised_Name_B"
            ).otherwise(
                F.col("ReportingBrandName")
            )
        )


# ============================================
# MAIN FUNCTION
# Applies full loyalty derivation
# pipeline in correct order
# ============================================

def process_loyalty_data(
    df: DataFrame
) -> tuple:
    """
    Applies complete loyalty derivation
    pipeline to transaction data.

    Steps:
    1. Derive loyalty code
    2. Standardise brand names
    3. Split into two datasets:
       → df_loyalty: all transactions
         with loyalty code derived
         (includes nulls)
       → df_loyalty_valid: only
         transactions with valid
         loyalty codes (no nulls)

    Args:
        df: Transaction DataFrame with
            all dimension joins applied
            and discount columns added

    Returns:
        Tuple of (df_loyalty,
                  df_loyalty_valid)

    Example:
        df_loyalty, df_itgb = \
            process_loyalty_data(df_full)
    """
    # Step 1 — Derive loyalty code
    df_loyalty = derive_loyalty_code(df)

    # Step 2 — Standardise brand names
    df_loyalty = standardise_brand_names(
        df_loyalty
    )

    # Step 3 — Filter to valid only
    df_loyalty_valid = filter_valid_loyalty(
        df_loyalty
    )

    return df_loyalty, df_loyalty_valid


# ============================================
# VALIDATION
# ============================================

def validate_loyalty_derivation(
    df_all: DataFrame,
    df_valid: DataFrame
) -> None:
    """
    Validates loyalty derivation results.

    Checks:
    → Valid dataset has no null codes
    → LoyaltyCode format is correct
    → No empty string loyalty codes
    → Derivation rate is reasonable

    Args:
        df_all: Full dataset before filter
        df_valid: Filtered valid dataset
    """
    print("\nValidating loyalty derivation...")

    total = df_all.count()
    valid = df_valid.count()

    # Check 1 — No null codes in valid set
    null_codes = df_valid.filter(
        F.col("LoyaltyCode").isNull()
    ).count()
    assert null_codes == 0, \
        f"Null loyalty codes found: {null_codes}"
    print(f"Null codes in valid set: {null_codes} ✅")

    # Check 2 — No empty strings
    empty_codes = df_valid.filter(
        F.col("LoyaltyCode") == ""
    ).count()
    assert empty_codes == 0, \
        f"Empty loyalty codes: {empty_codes}"
    print(f"Empty codes: {empty_codes} ✅")

    # Check 3 — Derivation rate
    rate = (valid / total * 100)
    print(f"Total transactions:  {total:,}")
    print(f"Valid loyalty:       {valid:,}")
    print(f"Derivation rate:     {rate:.1f}%")

    print("Loyalty validation passed! ✅")


# ============================================
# SUMMARY
# ============================================

def loyalty_summary(
    df_all: DataFrame,
    df_valid: DataFrame
) -> None:
    """
    Prints loyalty derivation summary
    for QA and reporting purposes.

    Shows:
    → Total transactions
    → Valid loyalty transactions
    → Null loyalty transactions
    → Unique loyal customers
    → Derivation rate

    Args:
        df_all: Full dataset
        df_valid: Valid loyalty dataset
    """
    total = df_all.count()
    valid = df_valid.count()
    null_count = total - valid
    unique_customers = df_valid.select(
        "LoyaltyCode"
    ).distinct().count()

    print("\n=== Loyalty Derivation Summary ===")
    print(f"Total transactions:    {total:,}")
    print(f"Valid loyalty:         {valid:,}")
    print(f"No loyalty:            {null_count:,}")
    print(f"Unique customers:      {unique_customers:,}")
    print(
        f"Derivation rate:       "
        f"{(valid/total*100):.1f}%"
    )
    print("==================================")


# ============================================
# USAGE EXAMPLE
# ============================================

if _name_ == "_main_":

    # Load full joined dataset
    # (from 01_load_data.py
    #  and 02_discount_flags.py)
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.getOrCreate()

    # Assumes df_full already created
    # from previous pipeline steps
    print("Processing loyalty derivation...")

    # Apply full loyalty pipeline
    df_loyalty, df_loyalty_valid = \
        process_loyalty_data(df_full)

    print(f"Loyalty derivation complete ✅")

    # Validate results
    validate_loyalty_derivation(
        df_loyalty,
        df_loyalty_valid
    )

    # Print summary
    loyalty_summary(
        df_loyalty,
        df_loyalty_valid
    )

    # Cache both datasets
    df_loyalty.cache()
    df_loyalty_valid.cache()
