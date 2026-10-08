# Databricks notebook source
# MAGIC %md
# MAGIC # 04 · Rule-based cleaning and quantity parsing
# MAGIC
# MAGIC The first governance stage, and the first one that changes values rather than just
# MAGIC moving them. Two jobs, both pure rules — no API calls, no cost:
# MAGIC
# MAGIC - **Clean the text.** Trim, collapse whitespace, turn empty strings into NULL. Rows
# MAGIC   with no usable product name go to a quarantine table instead of the main flow
# MAGIC - **Parse the quantity.** Turn free text like `6x2000ml` into structured
# MAGIC   `pack_count=6, size_value=2000, size_unit=ml`, and mark the rows the rules cannot
# MAGIC   handle so a later stage can spend LLM budget on those alone
# MAGIC
# MAGIC The second job is where the cost discipline of the whole project starts. Profiling
# MAGIC measured that 93.8% of non-empty quantity text is regular enough for a regex. Sending
# MAGIC all of it to an LLM would pay for work a pattern match does for free, so the rules
# MAGIC run first and only the remainder becomes an LLM problem.

# COMMAND ----------

CATALOG = "govpipe"
SOURCE = f"{CATALOG}.raw.merchant_skus"
TARGET = f"{CATALOG}.curated.sku_normalized"
QUARANTINE = f"{CATALOG}.curated.quarantine"

import pyspark.sql.functions as F
from pyspark.sql.types import DoubleType, IntegerType

raw = spark.table(SOURCE)
print(f"{SOURCE}: {raw.count():,} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Text cleaning
# MAGIC
# MAGIC Merchant text arrives with leading and trailing spaces, runs of internal whitespace,
# MAGIC and empty strings where a NULL was meant. Left alone, these produce phantom
# MAGIC distinctions downstream: `"Coca-Cola "` and `"Coca-Cola"` count as two brands, and
# MAGIC `""` passes an `IS NOT NULL` check while carrying no information.
# MAGIC
# MAGIC The cleaning is deliberately conservative — whitespace and empty-string handling
# MAGIC only. Case is left alone, because case carries signal at this point: m02 scrambled
# MAGIC it and the brand-matching stage needs to see that. Punctuation is left alone because
# MAGIC no rule can tell a stray hyphen from part of a brand name.

# COMMAND ----------

def clean_text(col):
    """Trim, collapse internal whitespace runs, empty string -> NULL."""
    c = F.trim(F.regexp_replace(F.col(col), r"\s+", " "))
    return F.when(c == "", None).otherwise(c)

cleaned = (raw
           .withColumn("name_clean", clean_text("raw_name"))
           .withColumn("brand_clean", clean_text("raw_brand"))
           .withColumn("category_clean", clean_text("raw_category"))
           .withColumn("unit_clean", clean_text("unit_raw"))
           .withColumn("barcode_clean", clean_text("barcode")))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Quarantine: rows with no product name
# MAGIC
# MAGIC A row without a name cannot be extracted from, matched against, or shown to anyone.
# MAGIC Carrying it through the pipeline costs LLM calls and pollutes every accuracy metric
# MAGIC with rows that were never answerable.
# MAGIC
# MAGIC It gets set aside rather than deleted. The quarantine table records why, so the
# MAGIC volume of unusable input per merchant is a measurable thing — that is a data quality
# MAGIC signal worth reporting back upstream, and deleting the rows would erase it.

# COMMAND ----------

quarantined = (cleaned.where(F.col("name_clean").isNull())
               .select("sku_key", "merchant_id", "merchant_sku_id",
                       F.lit("empty_name").alias("reason"),
                       "source_payload", "batch_id", "ingest_date",
                       F.current_timestamp().alias("quarantined_ts")))

survivors = cleaned.where(F.col("name_clean").isNotNull())

n_quarantined = quarantined.count()
print(f"quarantined: {n_quarantined:,}")
quarantined.groupBy("merchant_id").count().orderBy("merchant_id").show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Quantity parsing
# MAGIC
# MAGIC The quantity field arrives as free text in a handful of recognisable shapes:
# MAGIC
# MAGIC | Input | pack_count | size_value | size_unit |
# MAGIC |---|---|---|---|
# MAGIC | `500 ml` | 1 | 500 | ml |
# MAGIC | `6x2000ml` | 6 | 2000 | ml |
# MAGIC | `2 l` | 1 | 2000 | ml |
# MAGIC | `3.7 g` | 1 | 3.7 | g |
# MAGIC | `500 gr` | 1 | 500 | g |
# MAGIC | `1,5 kg` | 1 | 1500 | g |
# MAGIC
# MAGIC Units are normalised to two bases, `ml` for volume and `g` for weight, so that two
# MAGIC listings of the same product written as `2 l` and `2000ml` land on the same numbers.
# MAGIC Without this, the deduplication stage would treat them as different products.
# MAGIC `size_value` is the size of one unit; multiply by `pack_count` for the total.
# MAGIC
# MAGIC Every row gets a `spec_status` recording how it was resolved:
# MAGIC
# MAGIC - `rule_parsed` — the regex matched, values are structured
# MAGIC - `needs_llm` — text is present but does not match any known shape
# MAGIC   (`400 g (4 x 100 g)`, a bare `kg` with no number, Cyrillic units)
# MAGIC - `absent` — no quantity text at all, nothing to parse
# MAGIC
# MAGIC This column is what makes the cost ladder work: the LLM stage reads only the
# MAGIC `needs_llm` rows.

# COMMAND ----------

# Optional "N x" prefix, a number with , or . as decimal separator, then a unit.
QTY_RE = (r"(?i)^\s*(?:(\d+)\s*[x×]\s*)?"
          r"(\d+(?:[.,]\d+)?)\s*"
          r"(ml|millilitre|milliliter|cl|dl|l|litre|liter|g|gr|gram|gramme|kg|mg|oz|lb)\s*$")

# Multiplier to the base unit, and which base
TO_ML = {"ml": 1.0, "millilitre": 1.0, "milliliter": 1.0, "cl": 10.0, "dl": 100.0,
         "l": 1000.0, "litre": 1000.0, "liter": 1000.0}
TO_G = {"g": 1.0, "gr": 1.0, "gram": 1.0, "gramme": 1.0, "kg": 1000.0,
        "mg": 0.001, "oz": 28.3495, "lb": 453.592}

def unit_factor(unit_col):
    expr = F.lit(None).cast(DoubleType())
    for u, f in {**TO_ML, **TO_G}.items():
        expr = F.when(unit_col == u, F.lit(f)).otherwise(expr)
    return expr

def unit_base(unit_col):
    expr = F.lit(None).cast("string")
    for u in TO_ML:
        expr = F.when(unit_col == u, F.lit("ml")).otherwise(expr)
    for u in TO_G:
        expr = F.when(unit_col == u, F.lit("g")).otherwise(expr)
    return expr

u = F.lower(F.regexp_extract("unit_clean", QTY_RE, 3))
num = F.regexp_replace(F.regexp_extract("unit_clean", QTY_RE, 2), ",", ".").cast(DoubleType())
pack = F.regexp_extract("unit_clean", QTY_RE, 1)

parsed = (survivors
          .withColumn("_unit", F.when(u == "", None).otherwise(u))
          .withColumn("_num", num)
          .withColumn("_pack", F.when(pack == "", F.lit(1)).otherwise(pack.cast(IntegerType()))))

parsed = (parsed
          .withColumn("size_value",
                      F.round(F.col("_num") * unit_factor(F.col("_unit")), 4))
          .withColumn("size_unit", unit_base(F.col("_unit")))
          .withColumn("pack_count", F.when(F.col("_unit").isNotNull(), F.col("_pack")))
          .withColumn("spec_status",
                      F.when(F.col("unit_clean").isNull(), F.lit("absent"))
                       .when(F.col("_unit").isNotNull(), F.lit("rule_parsed"))
                       .otherwise(F.lit("needs_llm")))
          .drop("_unit", "_num", "_pack"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write
# MAGIC
# MAGIC `overwrite` here, unlike the `MERGE` in 03. The difference is what the input is: 03
# MAGIC reads external files that may contain only some merchants, so it has to preserve
# MAGIC whatever is already in the table. This stage reads the complete raw table and is a
# MAGIC pure function of it — recomputing from scratch gives the identical result, so
# MAGIC replacing the output wholesale is both correct and simpler.

# COMMAND ----------

# Optional "N x" prefix, a number with , or . as decimal separator, then a unit.
QTY_RE = (r"(?i)^\s*(?:(\d+)\s*[x×]\s*)?"
          r"(\d+(?:[.,]\d+)?)\s*"
          r"(ml|millilitre|milliliter|cl|dl|l|litre|liter|g|gr|gram|gramme|kg|mg|oz|lb)\s*$")

# Multiplier to the base unit, and which base
TO_ML = {"ml": 1.0, "millilitre": 1.0, "milliliter": 1.0, "cl": 10.0, "dl": 100.0,
         "l": 1000.0, "litre": 1000.0, "liter": 1000.0}
TO_G = {"g": 1.0, "gr": 1.0, "gram": 1.0, "gramme": 1.0, "kg": 1000.0,
        "mg": 0.001, "oz": 28.3495, "lb": 453.592}

def unit_factor(unit_col):
    expr = F.lit(None).cast(DoubleType())
    for u, f in {**TO_ML, **TO_G}.items():
        expr = F.when(unit_col == u, F.lit(f)).otherwise(expr)
    return expr

def unit_base(unit_col):
    expr = F.lit(None).cast("string")
    for u in TO_ML:
        expr = F.when(unit_col == u, F.lit("ml")).otherwise(expr)
    for u in TO_G:
        expr = F.when(unit_col == u, F.lit("g")).otherwise(expr)
    return expr

# regexp_extract yields "" when the pattern does not match, not NULL. Under ANSI mode
# casting "" to a numeric type raises rather than returning NULL, and every needs_llm
# row hits that path — so blanks become NULL before any cast.
def blank_to_null(c):
    return F.when(c == "", None).otherwise(c)

_pack = blank_to_null(F.regexp_extract("unit_clean", QTY_RE, 1))
_num = blank_to_null(F.regexp_replace(F.regexp_extract("unit_clean", QTY_RE, 2), ",", "."))
_unit = blank_to_null(F.lower(F.regexp_extract("unit_clean", QTY_RE, 3)))

parsed = (survivors
          .withColumn("_unit", _unit)
          .withColumn("_num", _num.cast(DoubleType()))
          .withColumn("_pack", F.coalesce(_pack.cast(IntegerType()), F.lit(1))))

parsed = (parsed
          .withColumn("size_value",
                      F.round(F.col("_num") * unit_factor(F.col("_unit")), 4))
          .withColumn("size_unit", unit_base(F.col("_unit")))
          .withColumn("pack_count", F.when(F.col("_unit").isNotNull(), F.col("_pack")))
          .withColumn("spec_status",
                      F.when(F.col("unit_clean").isNull(), F.lit("absent"))
                       .when(F.col("_unit").isNotNull(), F.lit("rule_parsed"))
                       .otherwise(F.lit("needs_llm")))
          .drop("_unit", "_num", "_pack"))

# COMMAND ----------

out = parsed.select(
    "sku_key", "merchant_id", "merchant_sku_id",
    "name_clean", "brand_clean", "category_clean", "unit_clean", "barcode_clean",
    "price", "size_value", "size_unit", "pack_count", "spec_status",
    "batch_id", "ingest_date",
    F.current_timestamp().alias("normalized_ts"),
)
out.write.mode("overwrite").saveAsTable(TARGET)
quarantined.write.mode("overwrite").saveAsTable(QUARANTINE)

print(f"{TARGET}: {spark.table(TARGET).count():,} rows")
print(f"{QUARANTINE}: {spark.table(QUARANTINE).count():,} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify
# MAGIC
# MAGIC The number that matters is the rule coverage: of the rows that actually carry
# MAGIC quantity text, what fraction did the regex resolve. Profiling predicted 93.8% on the
# MAGIC raw OFF data. The feeds are dirtier than that — m03 compacted its quantities and m06
# MAGIC localised its units — so a lower figure here is expected, and the gap is exactly the
# MAGIC work the LLM stage inherits.

# COMMAND ----------

norm = spark.table(TARGET)
total = norm.count()

dist = {r["spec_status"]: r["n"] for r in
        norm.groupBy("spec_status").agg(F.count("*").alias("n")).collect()}
have_text = dist.get("rule_parsed", 0) + dist.get("needs_llm", 0)

print("spec_status:")
for s in ("rule_parsed", "needs_llm", "absent"):
    n = dist.get(s, 0)
    print(f" {s:12s} {n:>7,}  ({n / total:.1%})")
print(f"\nrule coverage on rows with quantity text: "
      f"{dist.get('rule_parsed', 0) / have_text:.1%}  ({have_text:,} rows)")

assert dist.get("rule_parsed", 0) + dist.get("needs_llm", 0) + dist.get("absent", 0) == total
assert norm.where("spec_status = 'rule_parsed' AND size_value IS NULL").count() == 0, \
    "rule_parsed rows must all carry a size_value"
assert norm.where("size_unit IS NOT NULL AND size_unit NOT IN ('ml', 'g')").count() == 0, \
    "size_unit must be normalised to ml or g"
print("\nchecks passed")

# COMMAND ----------

# MAGIC %md
# MAGIC ## What the rules could not handle
# MAGIC
# MAGIC These are the rows the LLM stage will be given. Worth reading — if a recognisable
# MAGIC pattern shows up repeatedly here, adding one regex is cheaper than paying for those
# MAGIC rows on every run.

# COMMAND ----------

display(norm.where("spec_status = 'needs_llm'")
            .groupBy("unit_clean").count().orderBy(F.desc("count")).limit(25))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Parsing samples

# COMMAND ----------

display(norm.where("spec_status = 'rule_parsed'")
            .select("merchant_id", "unit_clean", "pack_count", "size_value", "size_unit")
            .limit(15))