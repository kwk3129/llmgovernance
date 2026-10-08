# Databricks notebook source
# MAGIC %md
# MAGIC # 04b · Rule-based cleaning and quantity parsing, revised
# MAGIC
# MAGIC Same stage as `04`, with three changes that came out of reading `04`'s own output.
# MAGIC Both notebooks are kept: `04` is what the rules looked like before anyone had seen
# MAGIC the failure list, and running it is how the problems below become visible. Editing
# MAGIC `04` in place would delete the evidence.
# MAGIC
# MAGIC ### What `04` reported
# MAGIC
# MAGIC `rule_parsed` 15,902 · `needs_llm` 3,131 · `absent` 26,053 · coverage 83.5%
# MAGIC
# MAGIC Those three numbers say the rules resolved a bit over eight tenths of the rows that
# MAGIC carry quantity text. They say nothing about what the remaining 3,131 rows contain,
# MAGIC which is what the last two cells of `04` print. Read together they expose three
# MAGIC problems, each less visible than the one before.
# MAGIC
# MAGIC **First, the top of the failure list is not a parsing failure at all.** `1pcs` 750
# MAGIC rows, `1 serving(s)` 286, `1serving(s)` 34, `2pcs` 23 — over 1,100 rows, more than a
# MAGIC third of the failures. `pcs` abbreviates *pieces* and `serving(s)` means portions:
# MAGIC these state a count, not a size. There is no millilitre or gram anywhere in the
# MAGIC text, so no parser can produce one and neither can a model. Leaving them in
# MAGIC `needs_llm` bills a thousand-odd calls per run for answers that cannot exist.
# MAGIC
# MAGIC Tracing that back exposes a flaw in how the statuses were split. `04` used one
# MAGIC dimension — parsed or not — where two are needed: whether quantity text is present,
# MAGIC and whether that text contains a measurable size. `1pcs` fails the second and was
# MAGIC filed under the third status. The name `needs_llm` compounds it by asserting the
# MAGIC disposition up front, and that assertion was never checked.
# MAGIC
# MAGIC **Second, further down the list, `16 fl oz` / `12 fl oz` / `20 fl oz`** total sixty
# MAGIC odd rows. Fluid ounces are a standard unit, 29.5735 ml, missing from the unit table
# MAGIC because that table was written from memory rather than from a count of the units
# MAGIC actually present in the data.
# MAGIC
# MAGIC **Third, and only visible by looking back at the successes**: `14 oz → 396.893 g`
# MAGIC uses the avoirdupois ounce, 28.3495 g. So `oz` carries two meanings in this data,
# MAGIC and `16 fl oz` escaped being read as weight only because the pattern requires the
# MAGIC unit to end the string and `fl` blocked the match — a side effect, not a safeguard.
# MAGIC A merchant abbreviating a drink's fluid ounces to `16 oz` would get 453.592 grams,
# MAGIC while the same product listed elsewhere as `473 ml` parses to 473.176 millilitres.
# MAGIC One row measured in grams, one in millilitres, values unequal: deduplication treats
# MAGIC them as two products. The numbers differ by only about 4%, `g` is a legal unit, and
# MAGIC every assertion passes — which is what makes it hard to find.
# MAGIC
# MAGIC ### What changed here
# MAGIC
# MAGIC 1. New status `count_only` for piece and serving counts, so only rows that contain a
# MAGIC    size but defeat the rules reach the LLM stage
# MAGIC 2. `fl oz` added to the unit table, matched ahead of and separately from `oz`
# MAGIC 3. Coverage denominator excludes `count_only`, and the row count going to the LLM is
# MAGIC    printed on its own line — that number is the one that costs money
# MAGIC
# MAGIC The third problem is **not** fixed here. Telling weight from volume needs a
# MAGIC cross-field check (category is a beverage, resolved unit is grams), which cannot be
# MAGIC done from the quantity text alone. It belongs to the QC stage; logged as pending.

# COMMAND ----------

CATALOG = "govpipe"
SOURCE = f"{CATALOG}.raw.merchant_skus"
TARGET = f"{CATALOG}.curated.sku_normalized"
QUARANTINE = f"{CATALOG}.curated.quarantine"

import pyspark.sql.functions as F
from pyspark.sql.types import DoubleType, IntegerType

# Capture 04's distribution before this notebook overwrites the table, so the final cell
# can compare against a measured baseline instead of hardcoded numbers.
before = ({r["spec_status"]: r["n"] for r in
           spark.table(TARGET).groupBy("spec_status").agg(F.count("*").alias("n")).collect()}
          if spark.catalog.tableExists(TARGET) else {})
print("previous run:", before or "none — 04 has not been run")

raw = spark.table(SOURCE)
print(f"{SOURCE}: {raw.count():,} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Text cleaning
# MAGIC
# MAGIC Unchanged from `04`. Trim, collapse internal whitespace runs, empty string to NULL.
# MAGIC
# MAGIC Case is deliberately left alone — m02 scrambled it and the brand-matching stage
# MAGIC needs to see that. Punctuation is left alone because no rule can tell a stray hyphen
# MAGIC from part of a brand name.

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
# MAGIC Unchanged from `04`. A row without a name cannot be extracted from, matched against,
# MAGIC or shown to anyone; carrying it through costs LLM calls and pollutes every accuracy
# MAGIC metric with rows that were never answerable. Set aside rather than deleted, so the
# MAGIC volume of unusable input per merchant stays measurable.

# COMMAND ----------

quarantined = (cleaned.where(F.col("name_clean").isNull())
               .select("sku_key", "merchant_id", "merchant_sku_id",
                       F.lit("empty_name").alias("reason"),
                       "source_payload", "batch_id", "ingest_date",
                       F.current_timestamp().alias("quarantined_ts")))

survivors = cleaned.where(F.col("name_clean").isNotNull())

print(f"quarantined: {quarantined.count():,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Quantity parsing
# MAGIC
# MAGIC Units normalise to two bases, `ml` for volume and `g` for weight, so `2 l` and
# MAGIC `2000ml` land on identical numbers — without that, deduplication would treat two
# MAGIC listings of the same product as different. `size_value` is the size of one unit;
# MAGIC multiply by `pack_count` for the total.
# MAGIC
# MAGIC Four statuses now, up from three in `04`:
# MAGIC
# MAGIC | status | meaning | goes to the LLM |
# MAGIC |---|---|---|
# MAGIC | `rule_parsed` | regex matched, values structured | no |
# MAGIC | `needs_llm` | text contains a size the rules cannot read — `400 g (4 x 100 g)`, a bare `kg`, Cyrillic units | **yes** |
# MAGIC | `count_only` | text states a count, not a size — `1pcs`, `1 serving(s)` | no |
# MAGIC | `absent` | no quantity text at all | no |
# MAGIC
# MAGIC The split between the middle two is the change that matters. Both are "the rules
# MAGIC failed", but only one of them describes work a model can do.

# COMMAND ----------

# Optional "N x" prefix, a number with , or . as decimal separator, then a unit.
# "fl oz" is listed ahead of "oz" and matched as a whole: fluid ounces measure volume
# (29.57 ml) while ounces measure weight (28.35 g), and the two share a word.
QTY_RE = (r"(?i)^\s*(?:(\d+)\s*[x×]\s*)?"
          r"(\d+(?:[.,]\d+)?)\s*"
          r"(fl\s*oz|ml|millilitre|milliliter|cl|dl|l|litre|liter"
          r"|g|gr|gram|gramme|kg|mg|oz|lb)\s*$")

# Piece and serving counts: a real value, but not a size. "1pcs" carries no volume or
# weight, so an LLM cannot resolve it either — these are separated out rather than
# billed for on every run. Found by reading 04's failed-parse frequency list.
COUNT_RE = r"(?i)^\s*\d+\s*(?:pcs?|pieces?|units?|ct|servings?(?:\(s\))?)\s*$"

# Multiplier to the base unit, and which base
TO_ML = {"ml": 1.0, "millilitre": 1.0, "milliliter": 1.0, "cl": 10.0, "dl": 100.0,
         "l": 1000.0, "litre": 1000.0, "liter": 1000.0, "floz": 29.5735}
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
# whitespace stripped so "fl oz" and "floz" reach the lookup table as one key
_unit = blank_to_null(F.regexp_replace(
    F.lower(F.regexp_extract("unit_clean", QTY_RE, 3)), r"\s+", ""))

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
                       .when(F.col("unit_clean").rlike(COUNT_RE), F.lit("count_only"))
                       .otherwise(F.lit("needs_llm")))
          .drop("_unit", "_num", "_pack"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write
# MAGIC
# MAGIC `overwrite`, unlike the `MERGE` in 03. The difference is what the input is: 03 reads
# MAGIC external files that may contain only some merchants, so it has to preserve whatever
# MAGIC is already in the table. This stage reads the complete raw table and is a pure
# MAGIC function of it — recomputing from scratch gives the identical result.
# MAGIC
# MAGIC Same target table as `04`. This notebook supersedes it; downstream stages read one
# MAGIC table either way.

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
# MAGIC The coverage denominator excludes `count_only`: those rows carry a value but no
# MAGIC size, so no parser could resolve them and counting them as failures understates what
# MAGIC the rules do. The row count heading for the LLM is printed separately because that
# MAGIC is the figure attached to a bill; the percentages are context.

# COMMAND ----------

norm = spark.table(TARGET)
total = norm.count()

STATUSES = ("rule_parsed", "needs_llm", "count_only", "absent")
dist = {r["spec_status"]: r["n"] for r in
        norm.groupBy("spec_status").agg(F.count("*").alias("n")).collect()}
resolvable = dist.get("rule_parsed", 0) + dist.get("needs_llm", 0)

print("spec_status:")
for s in STATUSES:
    n = dist.get(s, 0)
    print(f"  {s:12s} {n:>7,}  ({n / total:.1%})")

print(f"\nrule coverage on rows that contain a size: "
      f"{dist.get('rule_parsed', 0) / resolvable:.1%}  ({resolvable:,} rows)")
print(f"rows going to the LLM stage: {dist.get('needs_llm', 0):,}")

assert sum(dist.get(s, 0) for s in STATUSES) == total
assert norm.where("spec_status = 'rule_parsed' AND size_value IS NULL").count() == 0, \
    "rule_parsed rows must all carry a size_value"
assert norm.where("size_unit IS NOT NULL AND size_unit NOT IN ('ml', 'g')").count() == 0, \
    "size_unit must be normalised to ml or g"
print("\nchecks passed")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Against the previous run
# MAGIC
# MAGIC The baseline was read off the table at the top of this notebook, before it was
# MAGIC overwritten — so this comparison is measured, not asserted. It is empty if `04`
# MAGIC was never run.

# COMMAND ----------

if before:
    print(f"  {'status':12s} {'04':>9s} {'04b':>9s} {'delta':>9s}")
    for s in sorted(set(before) | set(dist)):
        b, a = before.get(s, 0), dist.get(s, 0)
        print(f"  {s:12s} {b:>9,} {a:>9,} {a - b:>+9,}")
    removed = before.get("needs_llm", 0) - dist.get("needs_llm", 0)
    if before.get("needs_llm"):
        print(f"\nrows no longer sent to the LLM: {removed:,} "
              f"({removed / before['needs_llm']:.1%} of the previous workload)")
        print(f"batches saved per run at 20 rows/call: ~{removed // 20:,}")
else:
    print("no earlier run to compare against")

# COMMAND ----------

# MAGIC %md
# MAGIC ## What still defeats the rules
# MAGIC
# MAGIC The list `04` produced is what drove this revision. Reading it again after the
# MAGIC change is the point: whatever now sits at the top is the next candidate for either
# MAGIC a new rule or an accepted limit.

# COMMAND ----------

display(norm.where("spec_status = 'needs_llm'")
            .groupBy("unit_clean").count().orderBy(F.desc("count")).limit(25))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Rows now classified as count_only

# COMMAND ----------

display(norm.where("spec_status = 'count_only'")
            .groupBy("unit_clean").count().orderBy(F.desc("count")).limit(15))

# COMMAND ----------

display(spark.sql("""
    SELECT sku_name, usage_unit, count(*) AS n, round(sum(usage_quantity), 4) AS total
    FROM system.billing.usage
    GROUP BY sku_name, usage_unit
    ORDER BY total DESC
"""))