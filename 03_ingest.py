# Databricks notebook source
# MAGIC %md
# MAGIC # 03 · Ingest merchant feeds into the raw layer
# MAGIC
# MAGIC Six feed files, three formats' worth of quirks, one target table:
# MAGIC `govpipe.raw.merchant_skus`.
# MAGIC
# MAGIC Three things this step is responsible for:
# MAGIC - **Flatten each merchant's own column names** into one canonical schema, so every
# MAGIC   downstream stage reads the same fields regardless of who uploaded the row
# MAGIC - **Keep the merchant's original row verbatim** in `source_payload` before any
# MAGIC   renaming happens — once columns are renamed the original shape is unrecoverable,
# MAGIC   and any dispute about "what did the merchant actually send us" has no answer
# MAGIC - **Merge idempotently** on `(merchant_id, merchant_sku_id)`. A feed gets re-uploaded
# MAGIC   all the time: same file resent after a failed run, a nightly batch that overlaps
# MAGIC   yesterday's. Ingesting the same batch twice must not change the row count
# MAGIC
# MAGIC The raw layer stays faithful to what arrived. No cleaning, no parsing, no dropping
# MAGIC of malformed values — that starts in the next stage.

# COMMAND ----------

CATALOG = "govpipe"
VOLUME = f"/Volumes/{CATALOG}/raw/landing"
TARGET = f"{CATALOG}.raw.merchant_skus"

# The batch id identifies one upload cycle. The ingest date is pinned to the batch
# rather than to "today" — a rerun of the same batch has to land in the same place,
# otherwise re-ingestion silently accumulates rows across dates.
BATCH_ID = "b20260725"
INGEST_DATE = "2026-07-25"

import pyspark.sql.functions as F
from delta.tables import DeltaTable
from pyspark.sql.types import DoubleType, StringType

# COMMAND ----------

# MAGIC %md
# MAGIC ## Merchant I/O configuration
# MAGIC
# MAGIC Only the three keys ingestion needs: file format and the mapping from each
# MAGIC merchant's column names to our canonical field names.
# MAGIC
# MAGIC These `field_map` entries must stay identical to the ones in `02_make_feeds`.
# MAGIC They are duplicated here because notebooks do not share state — once there are
# MAGIC more than two notebooks reading them, they should move into a shared config
# MAGIC notebook pulled in with `%run`.

# COMMAND ----------

MERCHANTS_IO = [
    {"id": "m01_grocermart", "format": "csv",
     "field_map": {"sku_id": "item_no", "name": "item_name", "brand": "brand", "category": "dept",
                   "quantity": "size", "barcode": "upc", "price": "unit_price", "description": "notes"}},
    {"id": "m02_quickshop", "format": "jsonl",
     "field_map": {"sku_id": "sid", "name": "title", "brand": "brand_name", "category": "cat",
                   "quantity": "pack_size", "barcode": "ean", "price": "price", "description": "desc"}},
    {"id": "m03_liquorbarn", "format": "csv",
     "field_map": {"sku_id": "sku", "name": "product", "brand": "label", "category": "aisle",
                   "quantity": "vol", "barcode": "barcode", "price": "rrp", "description": "memo"}},
    {"id": "m04_cornerstore", "format": "jsonl",
     "field_map": {"sku_id": "id", "name": "name", "brand": "brand", "category": "category",
                   "quantity": "quantity", "barcode": "code", "price": "amount", "description": "description"}},
    {"id": "m05_megamart", "format": "csv",
     "field_map": {"sku_id": "sku_code", "name": "display_name", "brand": "brand_name",
                   "category": "category_name", "quantity": "net_content", "barcode": "gtin",
                   "price": "retail_price", "description": "long_desc"}},
    {"id": "m06_worldbites", "format": "jsonl",
     "field_map": {"sku_id": "ref", "name": "nom", "brand": "marque", "category": "rayon",
                   "quantity": "contenance", "barcode": "cab", "price": "prix", "description": "infos"}},
]

# Canonical feed field -> raw table column. The `raw_` prefix marks values that arrived
# as-is from the merchant and have not been touched.
CANONICAL_TO_RAW = {
    "sku_id": "merchant_sku_id",
    "name": "raw_name",
    "description": "raw_description",
    "brand": "raw_brand",
    "category": "raw_category",
    "quantity": "unit_raw",
    "barcode": "barcode",
    "price": "price",
}

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read and flatten one merchant
# MAGIC
# MAGIC Two details that are easy to get wrong.
# MAGIC
# MAGIC **CSV parsing needs `multiLine` and `escape`.** Description fields carry embedded
# MAGIC newlines. Without these options Spark treats a newline inside a quoted field as a
# MAGIC record boundary and the row count comes out slightly too high — the local run of
# MAGIC this pipeline read 46,410 rows where the generator had written 46,408, and the two
# MAGIC extra rows were exactly this.
# MAGIC
# MAGIC **`source_payload` is captured before renaming.** It serializes the merchant's row
# MAGIC with their own column names intact. Renaming is lossy: after it, nothing in the
# MAGIC table can answer what the merchant literally sent.

# COMMAND ----------

def read_merchant(m: dict):
    path = f"{VOLUME}/feeds/{m['id']}.{m['format']}"
    if m["format"] == "csv":
        df = (spark.read.option("header", True)
              .option("multiLine", True).option("escape", '"').csv(path))
    else:
        df = spark.read.json(path)

    df = df.withColumn("source_payload", F.to_json(F.struct(*df.columns)))
    for canonical, merchant_field in m["field_map"].items():
        df = df.withColumnRenamed(merchant_field, canonical)

    # Cast every canonical field to string before the union. CSV yields all-string
    # columns while JSON gets its types inferred, so the same field can arrive as
    # string from one merchant and double from another — unionByName would fail on that.
    out = df.select(
        F.lit(m["id"]).alias("merchant_id"),
        *[F.col(c).cast(StringType()).alias(raw) for c, raw in CANONICAL_TO_RAW.items()],
        "source_payload",
    )
    return out.withColumn("price", F.col("price").cast(DoubleType()))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Union and derive keys
# MAGIC
# MAGIC `sku_key` is the surrogate key every downstream stage joins on. It hashes
# MAGIC `merchant_id:merchant_sku_id` because neither alone is unique — SKU numbers are
# MAGIC assigned by each merchant independently and collide across merchants by design.
# MAGIC The barcode is deliberately not used as a key: half of m04's rows do not have one.

# COMMAND ----------

dfs = [read_merchant(m) for m in MERCHANTS_IO]
src = dfs[0]
for d in dfs[1:]:
    src = src.unionByName(d)

src = (src
       .withColumn("sku_key", F.sha1(F.concat_ws(":", "merchant_id", "merchant_sku_id")))
       .withColumn("batch_id", F.lit(BATCH_ID))
       .withColumn("ingest_ts", F.current_timestamp())
       .withColumn("ingest_date", F.to_date(F.lit(INGEST_DATE))))

print(f"staged rows: {src.count():,}")
src.groupBy("merchant_id").count().orderBy("merchant_id").show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Idempotent merge
# MAGIC
# MAGIC `MERGE` on the natural key, updating the whole row on a match and inserting on a
# MAGIC miss. Re-running the same batch refreshes `ingest_ts` on existing rows and adds
# MAGIC nothing — which is the property this stage is judged on.
# MAGIC
# MAGIC An `append` would double the table on every rerun; an `overwrite` would work for a
# MAGIC single full feed but destroys any merchant absent from the current batch. `MERGE`
# MAGIC handles both a partial batch and a repeated one.

# COMMAND ----------

if not spark.catalog.tableExists(TARGET):
    src.write.saveAsTable(TARGET)
    print(f"created {TARGET} with {spark.table(TARGET).count():,} rows")
else:
    (DeltaTable.forName(spark, TARGET).alias("t")
     .merge(src.alias("s"),
            "t.merchant_id = s.merchant_id AND t.merchant_sku_id = s.merchant_sku_id")
     .whenMatchedUpdateAll()
     .whenNotMatchedInsertAll()
     .execute())
    print(f"merged into {TARGET}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify

# COMMAND ----------

raw = spark.table(TARGET)
total = raw.count()

print("rows per merchant:")
for r in raw.groupBy("merchant_id").count().orderBy("merchant_id").collect():
    print(f"  {r['merchant_id']:16s} {r['count']:>7,}")
print(f"\nTOTAL {total:,}")

# sku_key feeds every downstream join, so a collision here corrupts everything after it
distinct_keys = raw.select("sku_key").distinct().count()
assert distinct_keys == total, f"sku_key collision: {total:,} rows, {distinct_keys:,} distinct keys"
print(f"sku_key unique: {distinct_keys:,}")

# What actually arrived, before any cleaning
print("\nnull rate on raw fields:")
for c in ("raw_name", "raw_brand", "raw_category", "unit_raw", "barcode"):
    n = raw.where(F.col(c).isNull()).count()
    print(f"  {c:16s} {n:>7,}  ({n / total:.1%})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## What a preserved original row looks like
# MAGIC
# MAGIC `source_payload` still carries the merchant's own column names. Compare `item_no`
# MAGIC and `item_name` inside the JSON against the flattened `merchant_sku_id` /
# MAGIC `raw_name` columns beside it.

# COMMAND ----------

display(raw.where("merchant_id = 'm01_grocermart'")
           .select("merchant_sku_id", "raw_name", "unit_raw", "source_payload")
           .limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Idempotency check
# MAGIC
# MAGIC Run the merge cell again, then this one. The row count has to be identical.
# MAGIC That is the whole acceptance criterion for this stage.

# COMMAND ----------

print(f"{TARGET} rows = {spark.table(TARGET).count():,}")