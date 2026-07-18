# ============================================
# 01_load_data.py
# Load and filter all source tables
# needed for promotional loyalty analysis
# ============================================

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import DataFrame
from pyspark.sql.types import (
    IntegerType,
    StringType,
    BooleanType
)
import datetime

spark = SparkSession.builder.getOrCreate()

# ============================================
# CONFIGURATION
# ============================================

SOURCE_SYSTEM = "your_source_system"
ACTIVE_FLAG = "A"
PRODUCT_CATEGORY = "your_product_category"
DATE_FILTER = 20250101

# ============================================
# LOAD FUNCTIONS
# ============================================
""" Loads main transaction scan table. Contains all outlet level sales transactions with loyalty information. Filters: DateKey >= 2025-01-01
    Key columns:
    → OutletKey: outlet identifier
    → ProductKey: product identifier
    → DateKey: transaction date
    → MetricVolume: units sold
    → MetricValue: sales value
    → LoyaltyNumber: customer ID
    → PromotionFlag: promo indicator
    → ManufacturerDiscountValue: SMP discount
    → ManufacturerMultipackAmount: MP amount
    → OutletMultipackDiscountAmount: MP discount
    → OutletMultipackValue: MP value
    → ManufacturerMultipackValue: MP manuf value
    → CouponDiscountValue: coupon discount
    """
def load_scan_table() -> DataFrame:
    return spark.table(
        "your_database.fact_scan_transactions"
    ).select(
        "OutletKey",
        "ProductKey",
        "DateKey",
        "MetricPrice",
        "MetricVolume",
        "MetricValue",
        "LoyaltyNumber",
        "PromotionFlag",
        "AccountPromotionName",
        "TransactionID",
        "ManufacturerDiscountValue",
        "ManufacturerMultipackAmount",
        "OutletMultipackDiscountAmount",
        "OutletMultipackValue",
        "ManufacturerMultipackValue",
        "CouponDiscountValue"
    ).filter(
        F.col("DateKey") >= DATE_FILTER
    )

""" Loads outlet dimension table. Contains store/outlet information.
Filters:
    → ActiveFlag = A (active outlets only)
    → SourceSystem = your source system
    Key columns:
    → OutletKey: join key
    → OutletIdentifier: unique outlet ID
    → State: outlet state
    → SubmitterName: loyalty submitter
    → ChainId: retail chain ID
    → ChainName: retail chain name
    """
def load_outlet_table() -> DataFrame:
    return spark.table(
        "your_database.dim_outlet"
    ).filter(
        (F.col("ActiveFlag") == ACTIVE_FLAG) &
        (F.col("SourceSystem") == SOURCE_SYSTEM)
    ).select(
        "OutletKey",
        "OutletIdentifier",
        "State",
        "SubmitterName",
        "ChainId",
        "ChainName"
    )

 """ Loads product dimension table. Filtered to specific product category.
    Filters:
    → Category = target category
    → ActiveFlag = A
    → SourceSystem = your source system

    Key columns:
    → ProductKey: join key
    → Manufacturer: brand owner
    → Brand: product brand
    → BrandVariant: specific variant
    → ReportingBrandName: reporting name
    → ProductGroup: product grouping
    → Category: product category
    → IscompanyFlag: company product flag
    → EVPSegmentName: segment name
    """
def load_product_table() -> DataFrame:
    return spark.table(
        "your_database.dim_product"
    ).filter(
        (F.col("Category") == PRODUCT_CATEGORY) &
        (F.col("ActiveFlag") == ACTIVE_FLAG) &
        (F.col("SourceSystem") == SOURCE_SYSTEM)
    ).select(
        "ProductKey",
        "Manufacturer",
        "PackSize",
        "Brand",
        "BrandVariant",
        "Flavour",
        "ProductGroup",
        "UnitOfMeasure",
        "ServingUnitFactor",
        "PurchaseUnitFactor",
        "Category",
        "TarLevelName",
        "IscompanyFlag",
        "ActiveFlag",
        "EVPSegmentName",
        "ReportingBrandName"
    )
""" Loads date dimension table. Used for week level aggregations.
    Key columns:
    → DateKey: join key
    → WeekStartDate: week start
    → WeekEndDate: week end
    → CalendarYearNumber: year
    → CalendarYearMonthNumber: year month
    """

def load_date_table() -> DataFrame:
    return spark.table(
        "your_database.dim_date"
    ).select(
        "DateKey",
        "WeekStartDate",
        "WeekEndDate",
        "CalendarYearNumber",
        "CalendarYearMonthNumber"
    )

""" Loads secondary outlet reference table for additional outlet matching. Used for your_source system outlets.
    Filters:
    → SourceSystem = your_source
    → ActiveFlag = A
    """
def load_outlet_reference_table() -> DataFrame:
    return spark.table(
        "your_database.dim_outlet_reference"
    ).filter(
        (F.col("SourceSystem") == "your_source") &
        (F.col("ActiveFlag") == ACTIVE_FLAG)
    ).select(
        "OutletKey",
        "OutletIdentifier"
    ).distinct()

""" Loads your_source outlet reference table. Used for regional outlet matching.
    Filters:
    → SourceSystem = your_source
    """
def load_outlet_your_source_table() -> DataFrame:
    return spark.table(
        "your_database.dim_outlet_reference"
    ).filter(
        F.col("SourceSystem") == "your_source"
    ).select(
        "OutletIdentifier",
        "State"
    ).distinct()

 """ Loads product reference table for your_source system products.
    Filters:
    → SourceSystem = your_source
    → ActiveFlag = A
    """
def load_product_reference_table() -> DataFrame:
    return spark.table(
        "your_database.dim_product_reference"
    ).filter(
        (F.col("SourceSystem") == "your_source") &
        (F.col("ActiveFlag") == ACTIVE_FLAG)
    ).select(
        "ProductKey",
        "ReportingBrandName",
        "ExternalBrandCode"
    ).distinct()

""" Loads your_source product reference table.
    Filters:
    → SourceSystem = your_source system
    """
def load_product_your_source_table() -> DataFrame:
    return spark.table(
        "your_database.dim_product_reference"
    ).filter(
        F.col("SourceSystem") == "your_source"
    ).select(
        "ExternalBrandCode",
        "BrandVariant"
    ).distinct()

""" Loads promotion hierarchy table. Contains outlet level promo assignments.
    Key columns:
    → DateKey: date of promotion
    → OutletKey: outlet identifier
    → PromotionKey: promo identifier
    → PromotionCategory: buydown/multipack
    → PromotionSubCategory: sub type
    → DiscountLevel: discount amount
    """
def load_promo_hierarchy() -> DataFrame:
    return spark.table(
        "your_database.promo_hierarchy"
    )

""" Loads promotion master table. Contains promotion metadata.
    Filters:
    → ActiveFlag = A
    → PromotionStatus in
      Committed or Closed

    Key columns:
    → PromotionName: unique promo ID
    → PromotionType: type of promotion
    → PromotionCategory: category
    → PromotionStartDateKey: start date
    → PromotionEndDateKey: end date
    → DiscountLevel: discount amount
    → TargetProductCategory: target
    → TargetGeography: target geography
    → TargetSegmentation: target segment
    """
def load_promo_master() -> DataFrame:
    return spark.table(
        "your_database.dim_promotion"
    ).filter(
        (F.col("ActiveFlag") == ACTIVE_FLAG) &
        (F.col("PromotionStatus").isin(
            "Committed", "Closed"
        ))
    ).select(
        "PromotionName",
        "PromotionSlogan",
        "PromotionType",
        "PromotionCategory",
        "PromotionSubCategory",
        "PromotionStartDateKey",
        "PromotionEndDateKey",
        "DiscountLevel",
        "ActiveFlag",
        "PromotionStatus",
        "TargetProductCategory",
        "TargetGeography",
        "TargetSegmentation"
    )


# ============================================
# LOAD SAVED TABLES
# (Built in previous pipeline run)
# ============================================
"""
    Loads pre-built loyalty scan table.
    Contains cleaned loyalty transactions
    with derived loyalty codes.
    """
def load_loyalty_scan() -> DataFrame:
    return spark.table(
        "your_schema.loyalty_scan"
    )
"""
    Loads pre-built promotion table.
    Contains promotion assignments
    at outlet and brand level.

    Filters:
    → DateKey >= 2025-01-01
    """
def load_single_promo() -> DataFrame:
    return spark.table(
        "your_schema.single_promo"
    ).select(
        "ProductKey",
        "OutletKey",
        "DateKey",
        "OutletIdentifier",
        "PromotionName",
        "PromotionSlogan",
        "PromotionType",
        "PromotionCategory",
        "PromotionSubCategory",
        "PromotionStartDateKey",
        "PromotionEndDateKey",
        "PromotionRateStartDateKey",
        "PromotionRateEndDateKey",
        "DiscountLevel",
        "ActiveFlag",
        "Promo_BrandVariant",
        "ReportingBrandName",
        "PromotionStatus",
        "TargetProductCategory",
        "TargetGeography",
        "TargetSegmentation"
    ).filter(
        F.col("DateKey") >= DATE_FILTER
    )


# ============================================
# MAIN — LOAD ALL TABLES
# ============================================

if __name__ == "__main__":

    print("Loading source tables...")

    # Raw source tables
    df_scan = load_scan_table()
    print(f"Scan table loaded ✅")

    df_outlet = load_outlet_table()
    print(f"Outlet table loaded ✅")

    df_product = load_product_table()
    print(f"Product table loaded ✅")

    df_date = load_date_table()
    print(f"Date table loaded ✅")

    df_outlet_ref = load_outlet_reference_table()
    print(f"Outlet reference loaded ✅")

    df_outlet_rsd = load_outlet_your_source_table()
    print(f"Outlet RSD loaded ✅")

    df_product_ref = load_product_reference_table()
    print(f"Product reference loaded ✅")

    df_product_rsd = load_product_your_source_table()
    print(f"Product RSD loaded ✅")

    df_promo_hierarchy = load_promo_hierarchy()
    print(f"Promo hierarchy loaded ✅")

    df_promo_master = load_promo_master()
    print(f"Promo master loaded ✅")

    # Pre-built tables
    df_loyalty_table = load_loyalty_scan()
    print(f"Loyalty scan loaded ✅")

    df_promo_table = load_single_promo()
    print(f"Single promo loaded ✅")

    print("\nAll tables loaded successfully! ✅")
