# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # 05 · LLM attribute extraction
# MAGIC
# MAGIC The first stage that costs money. It asks the model for three things per SKU:
# MAGIC
# MAGIC - **brand** — 34% of rows have no brand field at all, and the ones that do are
# MAGIC   spelled every which way. The model reads it off the product name and returns a
# MAGIC   canonical spelling
# MAGIC - **category** — merchants report categories in their own vocabulary (m03 sends
# MAGIC   "Booze Corner"), which has to map onto one platform taxonomy
# MAGIC - **quantity, for the 1,736 rows the rules could not parse** — rows already resolved
# MAGIC   by 04b carry a `rule_hint` the model is told to copy verbatim; rules win over the
# MAGIC   model wherever rules produced an answer
# MAGIC
# MAGIC ### Four properties this stage has to have
# MAGIC
# MAGIC **Batched.** 20 SKUs per request instead of one. 45,086 rows become roughly 2,255
# MAGIC calls rather than 45,086.
# MAGIC
# MAGIC **Incremental.** Results are keyed by `(sku_key, model_version, prompt_version)`. A
# MAGIC rerun anti-joins what is already stored and only fills gaps. Changing either version
# MAGIC invalidates nothing — it produces a second, separately addressable set of results, so
# MAGIC two prompts can be compared on the same SKUs.
# MAGIC
# MAGIC **Checkpointed.** Rows are appended every `FLUSH_EVERY` batches instead of once at
# MAGIC the end. Free Edition shuts compute down for the rest of the day when an account
# MAGIC exceeds its quota, so a run that dies halfway has to leave its completed work behind.
# MAGIC The incremental key then makes the next run pick up where this one stopped.
# MAGIC
# MAGIC **Metered.** Every response's token usage is accumulated, so cost is a number this
# MAGIC notebook prints rather than something discovered on a bill.
# MAGIC
# MAGIC ### Choosing the provider
# MAGIC
# MAGIC What this stage asks of a model is undemanding: short inputs, a fixed output schema,
# MAGIC no multi-step reasoning, no long context. Mainstream models all clear that bar, so
# MAGIC capability is a threshold here rather than a differentiator — and once past it, the
# MAGIC gap between providers is far smaller than the gap between their prices. Cost
# MAGIC therefore decides.
# MAGIC
# MAGIC Databricks' built-in `ai_query` needs no key at all, which matters for anyone
# MAGIC reproducing this. But it is a SQL function invoked per row: 45,086 rows means 45,086
# MAGIC requests, and the prompt amortisation that batching buys is unavailable by
# MAGIC construction.
# MAGIC
# MAGIC Among the API providers, the same open-weight model costs very different amounts
# MAGIC depending on who hosts it — third-party hosts price output around ¥8 per million
# MAGIC tokens against DeepSeek's own ¥2. Since output is roughly three quarters of the bill,
# MAGIC that single number moves the full-run total from about ¥46 to about ¥14. DeepSeek
# MAGIC also applies context caching automatically, which drops the repeated System Prompt
# MAGIC to a fiftieth of the uncached input rate.
# MAGIC
# MAGIC So this notebook calls DeepSeek directly. Two costs come with that: a key to manage,
# MAGIC and stage 06's embeddings will need a separate provider. Both are cheap next to a
# MAGIC threefold difference in the bill, and the choice is reversible — endpoint, model name
# MAGIC and prices are all constants.

# COMMAND ----------

CATALOG = "govpipe"
SOURCE = f"{CATALOG}.curated.sku_normalized"
TARGET = f"{CATALOG}.curated.sku_extractions"

# Two names for the model, on purpose. API_MODEL is what the request body needs;
# MODEL_VERSION is what gets stored on every extracted row and includes the provider,
# because the same weights served by two providers can differ in quantisation, context
# window and sampling. Results are only reproducible if the provider is part of the key.

API_MODEL = "deepseek-chat"
MODEL_VERSION = "deepseek/deepseek-chat"

# API_MODEL = "deepseek-v4-flash"   # deepseek-chat is an alias for this and is being retired
# MODEL_VERSION = "deepseek/deepseek-v4-flash"
PROMPT_VERSION = "extract_v2"     # v2: size_unit unified to ml/g, see the prompt cell


BATCH_SIZE = 20
WORKERS = 24
FLUSH_EVERY = 10
MAX_ROWS = 0


BATCH_SIZE = 20        # SKUs per request
WORKERS = 8            # concurrent requests
FLUSH_EVERY = 25       # batches between writes — 500 rows per checkpoint
MAX_ROWS = 0        # 0 = no limit. Start small, confirm the output, then raise.

# Clear this line before exporting or sharing the notebook.
API_KEY = "xxxx"
assert API_KEY, "paste the DeepSeek API key into API_KEY above"

import json, time, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import pyspark.sql.functions as F
from pyspark.sql.types import (StructType, StructField, StringType, DoubleType,
                               IntegerType, MapType, TimestampType)

print(f"model={MODEL_VERSION}  prompt={PROMPT_VERSION}  batch={BATCH_SIZE}  "
      f"workers={WORKERS}  max_rows={MAX_ROWS or 'all'}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Taxonomy and prompt
# MAGIC
# MAGIC Both are versioned inputs, not incidental strings. The taxonomy is the closed set
# MAGIC the model must choose from — an open-ended category field would defeat the purpose,
# MAGIC since the whole point is collapsing merchant vocabularies onto one.
# MAGIC
# MAGIC The prompt is `extract_v2`. It differs from the local `extract_v1` in one respect:
# MAGIC `size_unit` must be `ml` or `g`, matching what 04b's rules produce. `v1` allowed
# MAGIC `L`/`g`/`count`, which put two unit systems in one column — `2 l` from the rules
# MAGIC became `2000 ml` while the same value from the model stayed `2 L`, and the dedup
# MAGIC stage compares those numbers directly.
# MAGIC
# MAGIC Any further edit to this text requires a new version string. The version is stored
# MAGIC on every extracted row, so results stay attributable and two prompts can be scored
# MAGIC against the same SKUs.

# COMMAND ----------

TAXONOMY = [
    "beverages", "alcoholic-beverages", "dairy", "snacks", "sweet-snacks",
    "cereals-and-grains", "breads-and-bakery", "meats", "seafood",
    "fruits-and-vegetables", "frozen-foods", "desserts", "condiments-and-sauces",
    "spreads", "canned-and-preserved", "breakfast", "baby-food", "plant-based",
    "dietary-supplements", "pet-food", "other",
]

SYSTEM_PROMPT = """You are a product-catalog data specialist for an instant-retail platform.
You receive a JSON array of raw merchant SKU records. Merchant data is dirty:
names contain abbreviations ("CC Zero" = Coca-Cola Zero), sizes embedded in
names, missing brands, categories in merchant-specific vocabulary, mixed
languages (English, French, Spanish, German, Italian, Russian and others).

For each record, extract:

- brand: the consumer-facing brand name, canonical spelling (e.g. "Coca-Cola",
  not "CC" or "coca cola"). null if no brand can be identified. Never use a
  manufacturer legal-entity name as the brand.
- name_normalized: a clean product name in the record's original language,
  without size/pack info, without the brand duplicated at the start,
  proper capitalization.
- category: exactly one value from the platform taxonomy provided in the input.
- size_value / size_unit / pack_count: only if visible in the record and the
  provided rule_hint is null. If rule_hint is present, copy it unchanged —
  rules take precedence over your own parsing.
- attributes: flavor, sugar_free, alcohol_free, organic and similar
  product-level flags you can identify. Empty object if none.
- confidence: your overall confidence in this record's extraction, 0.0-1.0.
  Be honest: missing brand you guessed from a vague name = low; explicit
  brand and clear name = high.

Return one output item per input record, same sku_key, same order.

Respond with a single JSON object exactly in this shape, no other text:

{"items": [
  {"sku_key": "...", "brand": "Coca-Cola", "name_normalized": "Zero Sugar Cola",
   "category": "beverages", "size_value": 2000, "size_unit": "ml", "pack_count": 6,
   "attributes": {"sugar_free": "true", "flavor": "cola"}, "confidence": 0.93}
]}

size_unit must be "ml", "g" or null — never "L", "kg", "oz" or "count".
Convert litres to millilitres and kilograms to grams before answering.
category must be exactly one of the taxonomy values from the input.
Records you cannot process still get an item with your best effort and a low confidence."""

print(f"taxonomy: {len(TAXONOMY)} categories")
print(f"prompt: {len(SYSTEM_PROMPT)} chars")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Which rows still need extracting
# MAGIC
# MAGIC Everything in `sku_normalized` minus whatever this exact `(model, prompt)` pair has
# MAGIC already produced. On a first run that is all 45,086 rows; on a rerun after a
# MAGIC quota-induced shutdown it is only the remainder.
# MAGIC
# MAGIC `rule_hint` is attached here rather than inside the prompt builder so that the
# MAGIC precedence — rules first, model second — is visible in the data going out.

# COMMAND ----------

EXTRACTIONS_SCHEMA = StructType([
    StructField("sku_key", StringType(), False),
    StructField("brand_extracted", StringType(), True),
    StructField("name_normalized", StringType(), True),
    StructField("category", StringType(), True),
    StructField("size_value", DoubleType(), True),
    StructField("size_unit", StringType(), True),
    StructField("pack_count", IntegerType(), True),
    StructField("attributes", MapType(StringType(), StringType()), True),
    StructField("confidence", DoubleType(), True),
    StructField("extraction_status", StringType(), False),
    StructField("model_version", StringType(), False),
    StructField("prompt_version", StringType(), False),
    StructField("extraction_ts", TimestampType(), False),
])

if not spark.catalog.tableExists(TARGET):
    spark.createDataFrame([], EXTRACTIONS_SCHEMA).write.saveAsTable(TARGET)
    print(f"created empty {TARGET}")

done = (spark.table(TARGET)
        .where((F.col("model_version") == MODEL_VERSION)
               & (F.col("prompt_version") == PROMPT_VERSION)
               & (F.col("extraction_status") == "ok"))
        .select("sku_key"))

todo = (spark.table(SOURCE)
        .join(done, "sku_key", "left_anti")
        .select("sku_key", "name_clean", "brand_clean", "category_clean",
                "unit_clean", "size_value", "size_unit", "pack_count", "spec_status")
        .orderBy("sku_key"))

n_todo = todo.count()
print(f"already extracted: {spark.table(TARGET).count():,}")
print(f"still to do:       {n_todo:,}")

pending = todo.limit(MAX_ROWS).collect() if MAX_ROWS else todo.collect()
print(f"this run:          {len(pending):,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## The API client
# MAGIC
# MAGIC Plain `urllib` — no third-party package to install on serverless. Retries use
# MAGIC exponential backoff on 429 and 5xx, which are the transient ones; a 400 or 401 is a
# MAGIC bug or a bad key and retrying it just wastes time.
# MAGIC
# MAGIC Token usage accumulates into a counter so the cost of a run is printed rather than
# MAGIC discovered afterwards. The per-token prices are the published rates at time of
# MAGIC writing and are an estimate — the authoritative number is on the provider's bill.

# COMMAND ----------

ENDPOINT = "https://api.deepseek.com/chat/completions"

# CNY per million tokens. These are the one input here that cannot be verified from the
# API response, so they get checked against the provider's invoice after the first run —
# a wrong constant produces a plausible number forever and never raises anything.
# Providers also change prices; DeepSeek announced an increase in August 2026.
PRICE_IN_HIT_PER_M = 0.02     # cached input
PRICE_IN_MISS_PER_M = 1.0     # uncached input
PRICE_OUT_PER_M = 2.0         # output

class Usage:
    def __init__(self):
        self.calls = self.retries = self.failures = 0
        self.in_hit = self.in_miss = self.out = 0

    def add(self, u):
        self.calls += 1
        self.in_hit += u.get("prompt_cache_hit_tokens", 0)
        self.in_miss += u.get("prompt_cache_miss_tokens", u.get("prompt_tokens", 0))
        self.out += u.get("completion_tokens", 0)

    @property
    def cost(self):
        return (self.in_hit * PRICE_IN_HIT_PER_M + self.in_miss * PRICE_IN_MISS_PER_M
                + self.out * PRICE_OUT_PER_M) / 1_000_000

    def __str__(self):
        return (f"{self.calls:,} calls  {self.retries} retries  {self.failures} failures  "
                f"{self.in_hit + self.in_miss:,} in / {self.out:,} out tokens  "
                f"¥{self.cost:.4f}")

usage = Usage()

def complete_json(system: str, body: str, attempts: int = 5) -> dict:
    payload = json.dumps({
        "model": API_MODEL,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": body}],
        "response_format": {"type": "json_object"},
        "temperature": 0,
    }).encode()
    req = urllib.request.Request(
        ENDPOINT, data=payload,
        headers={"Authorization": f"Bearer {API_KEY}",
                 "Content-Type": "application/json"})

    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                out = json.loads(r.read())
            usage.add(out.get("usage", {}))
            return json.loads(out["choices"][0]["message"]["content"])
        except urllib.error.HTTPError as e:
            # 429 and 5xx are transient; 4xx otherwise is a bad request or bad key
            if e.code != 429 and e.code < 500:
                raise
            if attempt == attempts - 1:
                raise
            usage.retries += 1
            time.sleep(2 ** attempt)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            if attempt == attempts - 1:
                raise
            usage.retries += 1
            time.sleep(2 ** attempt)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Building a request, reading a response
# MAGIC
# MAGIC The model is asked to echo `sku_key` on every item, and results are matched back by
# MAGIC that key rather than by position. A model that drops or reorders an item would
# MAGIC otherwise silently shift every field onto the wrong product — the kind of corruption
# MAGIC that produces plausible data and no error.
# MAGIC
# MAGIC Anything that fails validation is written with `extraction_status = 'parse_failed'`
# MAGIC rather than dropped. A missing row is invisible; a row marked failed is countable,
# MAGIC and the incremental filter only skips rows marked `ok`, so failures retry next run.

# COMMAND ----------

def payload_for(r):
    hint = ({"size_value": r["size_value"], "size_unit": r["size_unit"],
             "pack_count": r["pack_count"]}
            if r["spec_status"] == "rule_parsed" else None)
    return {"sku_key": r["sku_key"], "name": r["name_clean"],
            "brand": r["brand_clean"], "merchant_category": r["category_clean"],
            "quantity_text": r["unit_clean"], "rule_hint": hint}

def validate(item, r):
    """Coerce one model item into table shape, or None if it is unusable."""
    if not isinstance(item, dict):
        return None
    cat = item.get("category")
    if cat not in TAXONOMY:
        cat = None                      # outside the closed set -> treated as missing
    unit = item.get("size_unit")
    if unit not in ("ml", "g", None):
        unit = None
    try:
        conf = float(item.get("confidence", 0.0))
    except (TypeError, ValueError):
        conf = 0.0
    # rules win: whatever 04b resolved is copied through untouched
    if r["spec_status"] == "rule_parsed":
        sv, su, pc = r["size_value"], r["size_unit"], r["pack_count"]
    else:
        sv = item.get("size_value")
        sv = float(sv) if isinstance(sv, (int, float)) else None
        su = unit
        pc = item.get("pack_count")
        pc = int(pc) if isinstance(pc, int) else None
    attrs = item.get("attributes")
    attrs = ({str(k): str(v) for k, v in attrs.items()}
             if isinstance(attrs, dict) else {})
    return (r["sku_key"], item.get("brand"), item.get("name_normalized"), cat,
            sv, su, pc, attrs, max(0.0, min(1.0, conf)), "ok",
            MODEL_VERSION, PROMPT_VERSION, datetime.utcnow())

def run_batch(batch):
    body = json.dumps({"taxonomy": TAXONOMY,
                       "records": [payload_for(r) for r in batch]}, ensure_ascii=False)
    try:
        out = complete_json(SYSTEM_PROMPT, body)
        by_key = {i.get("sku_key"): i for i in out.get("items", [])
                  if isinstance(i, dict)}
    except Exception as e:
        usage.failures += 1
        print(f"  batch failed: {type(e).__name__} {str(e)[:120]}")
        by_key = {}
    rows = []
    for r in batch:
        row = validate(by_key.get(r["sku_key"]), r)
        rows.append(row if row else
                    (r["sku_key"], None, None, None, None, None, None, None, None,
                     "parse_failed", MODEL_VERSION, PROMPT_VERSION, datetime.utcnow()))
    return rows

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run, checkpointing as it goes
# MAGIC
# MAGIC Every `FLUSH_EVERY` batches the accumulated rows are appended and progress is
# MAGIC printed with a running cost. If the account hits its quota and compute is pulled,
# MAGIC everything up to the last checkpoint survives and the next run resumes from there.

# COMMAND ----------

batches = [pending[i:i + BATCH_SIZE] for i in range(0, len(pending), BATCH_SIZE)]
print(f"{len(batches)} batches of up to {BATCH_SIZE}\n")

started = time.time()
written = 0

for start in range(0, len(batches), FLUSH_EVERY):
    chunk = batches[start:start + FLUSH_EVERY]
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        rows = [row for batch_rows in pool.map(run_batch, chunk) for row in batch_rows]

    (spark.createDataFrame(rows, EXTRACTIONS_SCHEMA)
     .write.mode("append").saveAsTable(TARGET))
    written += len(rows)

    elapsed = time.time() - started
    rate = written / elapsed if elapsed else 0
    remaining = (len(pending) - written) / rate if rate else 0
    print(f"  {written:>6,}/{len(pending):,} rows  |  {usage}  |  "
          f"{elapsed / 60:.1f} min elapsed, ~{remaining / 60:.1f} min left")

print(f"\ndone in {(time.time() - started) / 60:.1f} min")
print(usage)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify
# MAGIC
# MAGIC Three things worth knowing before this feeds anything downstream: how much of the
# MAGIC input actually came back usable, whether the model stayed inside the taxonomy, and
# MAGIC how the confidence it reports is distributed — that last one is what the routing
# MAGIC stage will threshold on.

# COMMAND ----------

ext = spark.table(TARGET).where(
    (F.col("model_version") == MODEL_VERSION) & (F.col("prompt_version") == PROMPT_VERSION))
n = ext.count()

status = {r["extraction_status"]: r["n"] for r in
          ext.groupBy("extraction_status").agg(F.count("*").alias("n")).collect()}
print("extraction_status:", status)

ok = ext.where("extraction_status = 'ok'")
n_ok = ok.count()
print(f"\nbrand present:    {ok.where('brand_extracted IS NOT NULL').count():>6,} / {n_ok:,}")
print(f"category present: {ok.where('category IS NOT NULL').count():>6,} / {n_ok:,}")
print(f"size present:     {ok.where('size_value IS NOT NULL').count():>6,} / {n_ok:,}")

assert ext.where("size_unit IS NOT NULL AND size_unit NOT IN ('ml','g')").count() == 0, \
    "size_unit outside the ml/g convention"
assert ext.where(f"category IS NOT NULL AND category NOT IN "
                 f"({','.join(repr(c) for c in TAXONOMY)})").count() == 0, \
    "category outside the taxonomy"
assert ext.groupBy("sku_key").count().where("count > 1").count() == 0, \
    "duplicate sku_key for this (model, prompt) pair"
print("\nchecks passed")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Confidence distribution
# MAGIC
# MAGIC The routing stage sends high-confidence rows straight through and low-confidence
# MAGIC ones to human review. Where to put those thresholds is decided by calibration
# MAGIC against the golden set later — this histogram is just the raw shape.

# COMMAND ----------

display(ok.select(F.floor(F.col("confidence") * 20).alias("bucket"))
          .groupBy("bucket").count()
          .withColumn("confidence_range",
                      F.concat(F.round(F.col("bucket") / 20, 2), F.lit(" – "),
                               F.round((F.col("bucket") + 1) / 20, 2)))
          .orderBy("bucket").select("confidence_range", "count"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Sample output
# MAGIC
# MAGIC Worth reading before scaling up. The abbreviated names from m01 are the interesting
# MAGIC ones — `Qs frsc de ovj` was `Queso fresco de oveja` before m01 dropped the vowels.

# COMMAND ----------

display(spark.table(SOURCE).alias("s")
        .join(ok.alias("e"), "sku_key")
        .select("s.merchant_id", "s.name_clean", "s.brand_clean",
                "e.brand_extracted", "e.name_normalized", "e.category",
                "e.size_value", "e.size_unit", "e.confidence")
        .limit(25))

# COMMAND ----------

# MAGIC %sql
# MAGIC DELETE FROM govpipe.curated.sku_extractions AS e
# MAGIC WHERE e.extraction_status = 'parse_failed'
# MAGIC   AND EXISTS (
# MAGIC     SELECT 1 FROM govpipe.curated.sku_extractions AS o
# MAGIC     WHERE o.sku_key = e.sku_key
# MAGIC       AND o.model_version = e.model_version
# MAGIC       AND o.prompt_version = e.prompt_version
# MAGIC       AND o.extraction_status = 'ok'
# MAGIC   )

# COMMAND ----------

