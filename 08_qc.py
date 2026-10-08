# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # 08 · Catalog quality check
# MAGIC
# MAGIC Every check so far has been about form: units must be `ml` or `g`, categories must
# MAGIC come from the taxonomy, product ids must be unique. Those pass on `Pollo asado, 19 g`
# MAGIC — a 19-gram roast chicken satisfies every one of them.
# MAGIC
# MAGIC This stage looks at content instead. A size that is physically implausible, a
# MAGIC category that contradicts the name, a brand field holding a flavour rather than a
# MAGIC brand: none of these can be caught by a rule about formats, because the record is
# MAGIC perfectly well-formed. What makes it wrong is what it means.
# MAGIC
# MAGIC ### Three sources of findings
# MAGIC
# MAGIC **Rules find suspects, the model judges them.** Sending 10,026 products to a model
# MAGIC would cost more than it is worth and most of them are fine, so cheap rules do a first
# MAGIC pass tuned for recall, and only what they flag reaches the model. Rules are good at
# MAGIC "this looks unusual" and bad at "this is wrong" — a rule saying beverages hold between
# MAGIC 50 ml and 20 L flags a 25 L party keg, and only a model tells that apart from a typo.
# MAGIC
# MAGIC **Cluster members check each other.** Six merchants listing the same product should
# MAGIC agree on its size. When five say 500 g and one says 7 g, the odd one out is wrong, and
# MAGIC establishing that needs no domain knowledge and no model call at all.
# MAGIC
# MAGIC ### Findings are attributed, not just counted
# MAGIC
# MAGIC Every finding records which merchant and which SKU it came from. Fixing the data is
# MAGIC only half the job — a merchant who sends `1g` for a bottle of whiskey will send it
# MAGIC again next week, and silently correcting it downstream means nobody ever tells them.
# MAGIC The findings table is what turns cleaning into feedback.
# MAGIC
# MAGIC Nothing is removed from the catalog. Problems are marked, the serious ones go to the
# MAGIC review queue, and delivery is never blocked — a quality gate that stops the pipeline
# MAGIC is a quality gate that gets switched off.

# COMMAND ----------

CATALOG = "govpipe"
NORM = f"{CATALOG}.curated.sku_normalized"
PRODUCTS = f"{CATALOG}.serving.products"
ALIASES = f"{CATALOG}.serving.product_aliases"
QUEUE = f"{CATALOG}.ops.review_queue"
TARGET = f"{CATALOG}.curated.qc_findings"

ENDPOINT = "https://api.deepseek.com/chat/completions"
API_MODEL = "deepseek-chat"
MODEL_VERSION = "deepseek/deepseek-chat"
PROMPT_VERSION = "qc_v2"      # v2: refuses findings whose own rationale says "plausible"

PRICE_IN_PER_M, PRICE_OUT_PER_M = 1.0, 2.0

BATCH_SIZE = 15
MAX_CALLS = 200        # ceiling; anything beyond it waits for the next run
WORKERS = 24

OUTLIER_MIN_MEMBERS = 3    # a cluster needs this many sized members before voting
OUTLIER_RATIO = 5.0        # further than this from the median, in either direction

# Clear this line before exporting or sharing the notebook.
API_KEY = "xxxx"
assert API_KEY, "paste the DeepSeek API key into API_KEY above"

import hashlib, json, time, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import pyspark.sql.functions as F
from delta.tables import DeltaTable
from pyspark.sql.types import StructType, StructField, StringType, TimestampType

CHECKS = {"name_size_conflict", "category_mismatch", "brand_in_name", "unit_implausible",
          "cluster_size_outlier"}
SEVERITIES = {"high", "medium", "low"}

print(f"model={MODEL_VERSION}  prompt={PROMPT_VERSION}  batch={BATCH_SIZE}  budget={MAX_CALLS} calls")

# COMMAND ----------

# MAGIC %md
# MAGIC ## The findings table
# MAGIC
# MAGIC `merchant_id` and `sku_key` are what make a finding actionable. A row saying "this
# MAGIC product has an implausible size" is a data point; a row saying "m06 sent 7 g for a
# MAGIC pack of bacon" is something you can take to the merchant. Findings that are genuinely
# MAGIC product-level rather than record-level leave both null.
# MAGIC
# MAGIC `finding_id` is `sha1(product|merchant|check)`, so rerunning cannot duplicate a
# MAGIC finding, and a problem that persists across runs stays one row.

# COMMAND ----------

FINDINGS_SCHEMA = StructType([
    StructField("finding_id", StringType(), False),      # sha1(product|merchant|check)
    StructField("product_id", StringType(), False),
    StructField("merchant_id", StringType(), True),      # null for product-level findings
    StructField("sku_key", StringType(), True),
    StructField("check_type", StringType(), False),
    StructField("severity", StringType(), False),
    StructField("details", StringType(), True),
    StructField("source", StringType(), False),          # llm_audit / cluster_vote
    StructField("model_version", StringType(), True),
    StructField("prompt_version", StringType(), True),
    StructField("status", StringType(), False),          # open / resolved
    StructField("found_ts", TimestampType(), False),
])

# The schema gained columns; drop and rebuild rather than migrate — findings are cheap to
# regenerate and there is nothing here worth preserving across a schema change.
if spark.catalog.tableExists(TARGET):
    have = {f.name for f in spark.table(TARGET).schema}
    if not {f.name for f in FINDINGS_SCHEMA}.issubset(have):
        spark.sql(f"DROP TABLE {TARGET}")
        print(f"dropped {TARGET} — schema changed")

if not spark.catalog.tableExists(TARGET):
    spark.createDataFrame([], FINDINGS_SCHEMA).write.saveAsTable(TARGET)
    print(f"created {TARGET}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Check 1 · cluster members vote on size
# MAGIC
# MAGIC This one costs nothing and needs no model. For every product with at least three
# MAGIC members carrying a size, take the median and flag any member more than five times
# MAGIC away from it in either direction.
# MAGIC
# MAGIC The bounds are deliberately loose. Merchants legitimately stock different pack sizes
# MAGIC of the same product, and a 5× gap is well past that — it catches `7 g` against a
# MAGIC median of 1360 g, not 500 g against 750 g.
# MAGIC
# MAGIC What makes this check different from everything else here is that it needs no
# MAGIC knowledge of what the product is. It does not matter whether bacon should weigh 7
# MAGIC grams; what matters is that five other merchants disagree.

# COMMAND ----------

now = datetime.now(timezone.utc)

members = (spark.table(ALIASES).alias("a")
           .join(spark.table(NORM).alias("n"),
                 (F.col("a.merchant_id") == F.col("n.merchant_id"))
                 & (F.col("a.raw_name") == F.col("n.name_clean")))
           .select(F.col("a.product_id"), F.col("n.merchant_id"), F.col("n.sku_key"),
                   F.col("n.name_clean"), F.col("n.unit_clean"),
                   F.col("n.size_value"), F.col("n.size_unit")))

sized = members.where(F.col("size_value").isNotNull() & (F.col("size_value") > 0))

med = (sized.groupBy("product_id")
       .agg(F.expr("percentile_approx(size_value, 0.5)").alias("median_size"),
            F.count("*").alias("n_sized")))

outliers = (sized.join(med, "product_id")
            .where((F.col("n_sized") >= OUTLIER_MIN_MEMBERS)
                   & ((F.col("size_value") < F.col("median_size") / OUTLIER_RATIO)
                      | (F.col("size_value") > F.col("median_size") * OUTLIER_RATIO)))
            .select(
                F.sha1(F.concat_ws("|", "product_id", "merchant_id",
                                   F.lit("cluster_size_outlier"))).alias("finding_id"),
                "product_id", "merchant_id", "sku_key",
                F.lit("cluster_size_outlier").alias("check_type"),
                F.lit("high").alias("severity"),
                F.concat_ws(" ", F.lit("size"), F.col("size_value"), F.col("size_unit"),
                            F.lit("vs cluster median"), F.round("median_size", 1),
                            F.lit("from raw"), F.coalesce("unit_clean", F.lit("?"))).alias("details"),
                F.lit("cluster_vote").alias("source"),
                F.lit(None).cast("string").alias("model_version"),
                F.lit(None).cast("string").alias("prompt_version"),
                F.lit("open").alias("status"),
                F.lit(now).alias("found_ts")))

n_outliers = outliers.count()
print(f"products with >= {OUTLIER_MIN_MEMBERS} sized members: "
      f"{med.where(F.col('n_sized') >= OUTLIER_MIN_MEMBERS).count():,}")
print(f"members flagged as size outliers:  {n_outliers:,}")

display(outliers.select("merchant_id", "details").limit(20))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Check 2 · rule pre-filter for the model
# MAGIC
# MAGIC Six patterns, all cross-field contradictions — one field alone is almost never
# MAGIC provably wrong, but two fields disagreeing is a signal.
# MAGIC
# MAGIC Note that rules A, D and E look for size information inside the product name, and the
# MAGIC names here are `name_normalized` from 05, which was explicitly asked to strip size and
# MAGIC pack info. Those three barely fire as a result. Left in place because a future run
# MAGIC against raw merchant names would need them, but they contribute almost nothing today.

# COMMAND ----------

PLAUSIBLE = {
    "beverages": (50, 20000), "alcoholic-beverages": (20, 20000), "dairy": (10, 20000),
    "snacks": (5, 5000), "sweet-snacks": (5, 5000), "condiments-and-sauces": (5, 5000),
    "spreads": (10, 5000), "meats": (10, 20000), "seafood": (10, 20000),
    "fruits-and-vegetables": (10, 20000), "frozen-foods": (10, 20000),
    "cereals-and-grains": (10, 10000), "breads-and-bakery": (20, 5000),
}

NON_BRAND_WORDS = ("biscuit", "cookie", "sauce", "milk", "water", "juice", "bio",
                   "organic", "sliced", "frozen", "fresh", "natural", "original",
                   "classic", "premium", "family", "pack", "assorted", "variety",
                   "chocolate", "vanilla", "strawberry", "nature")

CATEGORY_KEYWORDS = {
    "beer": ("alcoholic-beverages",), "lager": ("alcoholic-beverages",),
    "wine": ("alcoholic-beverages",), "whisky": ("alcoholic-beverages",),
    "vodka": ("alcoholic-beverages",), "cider": ("alcoholic-beverages",),
    "milk": ("dairy", "plant-based", "baby-food"),
    "yoghurt": ("dairy", "desserts"), "yogurt": ("dairy", "desserts"),
    "cheese": ("dairy",), "butter": ("dairy", "spreads"),
    "spaghetti": ("cereals-and-grains",), "pasta": ("cereals-and-grains",),
    "coffee": ("beverages",), "cola": ("beverages",), "juice": ("beverages",),
    "water": ("beverages",), "chocolate": ("sweet-snacks",),
    "bread": ("breads-and-bakery",), "salmon": ("seafood",), "tuna": ("seafood",),
    "chicken": ("meats", "frozen-foods"),
}

name_has_size = F.col("canonical_name").rlike(r"(?i)\d+\s*(ml|l|cl|dl|kg|g|oz)\b")
name_has_pack = F.col("canonical_name").rlike(r"(?i)(\bx\s*\d+|\d+\s*x\b|\d+\s*pack)")

kw_conflict = F.lit(False)
for kw, cats in CATEGORY_KEYWORDS.items():
    kw_conflict = kw_conflict | (F.col("canonical_name").rlike(f"(?i)\\b{kw}")
                                 & ~F.col("category").isin(list(cats)))

brand_suspect = F.lower(F.coalesce("brand_id", F.lit(""))).rlike(
    "^(" + "|".join(NON_BRAND_WORDS) + ")$")

ranges = F.create_map(*[x for k, (lo, hi) in PLAUSIBLE.items()
                        for x in (F.lit(k), F.array(F.lit(float(lo)), F.lit(float(hi))))])
lo = F.element_at(ranges[F.col("category")], 1)
hi = F.element_at(ranges[F.col("category")], 2)
implausible = (F.col("size_value").isNotNull() & lo.isNotNull()
               & ((F.col("size_value") < lo) | (F.col("size_value") > hi)))

hint = F.concat_ws(
    ",",
    F.when(name_has_size & F.col("size_value").isNull(), F.lit("A:name_size_missing")),
    F.when(brand_suspect, F.lit("B:brand_suspect")),
    F.when(implausible, F.lit("C:size_out_of_range")),
    F.when(name_has_pack & (F.coalesce("pack_count", F.lit(1)) == 1), F.lit("D:pack_missing")),
    F.when(name_has_size & F.col("size_value").isNotNull(), F.lit("E:name_size_both")),
    F.when(kw_conflict, F.lit("F:category_keyword_conflict")),
)

products = spark.table(PRODUCTS)
checked = spark.table(TARGET).where(F.col("source") == "llm_audit").select("product_id").distinct()

cand = (products.withColumn("rule_hint", hint)
        .where(F.col("rule_hint") != "")
        .join(checked, "product_id", "left_anti"))

n_products = products.count()
n_cand = cand.count()
print(f"products:         {n_products:,}")
print(f"flagged by rules: {n_cand:,}  ({n_cand / n_products:.1%})\n")
for r in (cand.select(F.explode(F.split("rule_hint", ",")).alias("rule"))
          .groupBy("rule").agg(F.count("*").alias("n")).orderBy(F.desc("n")).collect()):
    print(f"  {r['rule']:32s} {r['n']:>6,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ask the model about the suspects
# MAGIC
# MAGIC `qc_v1` produced findings whose own rationale said the record was fine — *"10 litres
# MAGIC of oil is plausible but large; no conflict"* reported as a high-severity issue. The
# MAGIC instruction to report only genuine problems was not enough, because the pre-filter
# MAGIC already biases the input toward looking odd and an auditor asked to audit will find
# MAGIC something.
# MAGIC
# MAGIC `qc_v2` closes that specific hole: if the reasoning would contain a word like
# MAGIC *plausible* or *reasonable*, there is no finding to report.

# COMMAND ----------

QC_PROMPT = """You are a product-catalog quality auditor for an instant-retail platform.
You receive product master records that a rule-based pre-filter flagged as
possibly inconsistent. For each record, check for genuine internal
contradictions between its fields:

- name_size_conflict: the name states a size or pack that contradicts the
  size_value / size_unit / pack_count fields (e.g. name says "2L" but
  size_value is 500 ml). A name with no size info is NOT a conflict.
- category_mismatch: the category is clearly wrong for the product
  (e.g. a beer categorised as "dairy"). Borderline or arguable
  categorisations are NOT a mismatch.
- brand_in_name: the brand field contains something that is not a brand
  (a category word, a flavour, a size, a packaging term).
- unit_implausible: the size is physically impossible or absurd for this
  product type (e.g. 1 g of whiskey, 19 g of roast chicken, 5000 L of yoghurt).

Report ONLY genuine problems. The pre-filter is deliberately noisy, so most
records you receive are in fact correct — an empty issues array is the
expected outcome, not a failure to be thorough.

Hard rule: if your explanation would describe the value as plausible,
reasonable, acceptable, fine, or merely unusual/large/small, then it is NOT
an issue. Report nothing for that record. Only report values that are wrong.

severity: "high" if the record would mislead search or pricing,
"medium" if it degrades quality, "low" if cosmetic.

Respond with a single JSON object exactly in this shape, no other text:

{"items": [
  {"product_id": "abc123", "issues": [
    {"check_type": "unit_implausible", "severity": "high",
     "details": "1 g for a bottle of whiskey"}
  ]}
]}

check_type must be one of: name_size_conflict, category_mismatch,
brand_in_name, unit_implausible. Keep details under 25 words.
One item per input record, same product_id."""

class Usage:
    def __init__(self):
        self.calls = self.failures = 0
        self.tok_in = self.tok_out = 0

    def add(self, u):
        self.calls += 1
        self.tok_in += u.get("prompt_tokens", 0)
        self.tok_out += u.get("completion_tokens", 0)

    @property
    def cost(self):
        return (self.tok_in * PRICE_IN_PER_M + self.tok_out * PRICE_OUT_PER_M) / 1_000_000

    def __str__(self):
        return (f"{self.calls:,} calls  {self.failures} failures  "
                f"{self.tok_in:,} in / {self.tok_out:,} out  ¥{self.cost:.4f}")

usage = Usage()

def audit_batch(batch):
    body = {"records": [{"product_id": r["product_id"], "name": r["canonical_name"],
                         "brand": r["brand_id"], "category": r["category"],
                         "size_value": r["size_value"], "size_unit": r["size_unit"],
                         "pack_count": r["pack_count"], "rule_hint": r["rule_hint"]}
                        for r in batch]}
    payload = json.dumps({
        "model": API_MODEL,
        "messages": [{"role": "system", "content": QC_PROMPT},
                     {"role": "user", "content": json.dumps(body, ensure_ascii=False)}],
        "response_format": {"type": "json_object"},
        "temperature": 0,
    }).encode()
    req = urllib.request.Request(
        ENDPOINT, data=payload,
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                out = json.loads(r.read())
            usage.add(out.get("usage", {}))
            items = json.loads(out["choices"][0]["message"]["content"]).get("items", [])
            return batch, {i.get("product_id"): i for i in items if isinstance(i, dict)}
        except urllib.error.HTTPError as e:
            if e.code in (401, 402):
                raise RuntimeError(f"HTTP {e.code} — key rejected or out of credit") from e
            if (e.code != 429 and e.code < 500) or attempt == 4:
                usage.failures += 1
                print(f"  batch failed: HTTP {e.code}")
                return batch, {}
            time.sleep(2 ** attempt)
        except Exception as e:
            if attempt == 4:
                usage.failures += 1
                print(f"  batch failed: {type(e).__name__} {str(e)[:100]}")
                return batch, {}
            time.sleep(2 ** attempt)

rows = (cand.select("product_id", "canonical_name", "brand_id", "category",
                    "size_value", "size_unit", "pack_count", "rule_hint")
        .orderBy("product_id").limit(MAX_CALLS * BATCH_SIZE).collect())

deferred = max(0, n_cand - len(rows))
batches = [rows[i:i + BATCH_SIZE] for i in range(0, len(rows), BATCH_SIZE)]
print(f"auditing {len(rows):,} products in {len(batches)} batches"
      + (f"  ({deferred:,} deferred to next run)" if deferred else ""))

started = time.time()
with ThreadPoolExecutor(max_workers=WORKERS) as pool:
    outcomes = list(pool.map(audit_batch, batches))
print(f"\ndone in {(time.time() - started) / 60:.1f} min  |  {usage}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Record everything
# MAGIC
# MAGIC Both sources land in one table, distinguished by `source`. Cluster votes carry a
# MAGIC merchant; model findings are product-level and leave it null — the model sees the
# MAGIC merged record, not who contributed which field.
# MAGIC
# MAGIC `high` findings additionally go to the review queue. The rest stay queryable as a
# MAGIC record of catalog health without demanding anyone's attention.

# COMMAND ----------

llm_findings = []
n_clean = n_failed = 0

for batch, out in outcomes:
    if not out:
        n_failed += 1
        continue
    for r in batch:
        issues = (out.get(r["product_id"]) or {}).get("issues") or []
        if not isinstance(issues, list) or not issues:
            n_clean += 1
            continue
        for iss in issues:
            if not isinstance(iss, dict):
                continue
            ct, sev = iss.get("check_type"), iss.get("severity")
            if ct not in CHECKS or sev not in SEVERITIES:
                continue
            fid = hashlib.sha1(f"{r['product_id']}||{ct}".encode()).hexdigest()[:16]
            llm_findings.append((fid, r["product_id"], None, None, ct, sev,
                                 str(iss.get("details", ""))[:200], "llm_audit",
                                 MODEL_VERSION, PROMPT_VERSION, "open", now))

print(f"model — clean:    {n_clean:,}")
print(f"model — findings: {len(llm_findings):,}")
if n_failed:
    print(f"failed batches:   {n_failed:,}  ({n_failed * BATCH_SIZE:,} products -> next run)")

all_df = outliers
if llm_findings:
    all_df = outliers.unionByName(spark.createDataFrame(llm_findings, FINDINGS_SCHEMA))
all_df = all_df.dropDuplicates(["finding_id"])

(DeltaTable.forName(spark, TARGET).alias("t")
 .merge(all_df.alias("s"), "t.finding_id = s.finding_id")
 .whenNotMatchedInsertAll()
 .execute())

high = (all_df.where(F.col("severity") == "high")
        .select(F.sha1(F.concat_ws("|", "product_id", F.coalesce("merchant_id", F.lit("")),
                                   F.lit("qc_flag"))).alias("queue_id"),
                F.coalesce("sku_key", "product_id").alias("sku_key"),
                F.lit("qc_flag").alias("reason"),
                F.to_json(F.struct("product_id", "merchant_id", "check_type",
                                   "details")).alias("payload"))
        .dropDuplicates(["queue_id"])
        .withColumn("status", F.lit("pending"))
        .withColumn("corrected_payload", F.lit(None).cast("string"))
        .withColumn("reviewer", F.lit(None).cast("string"))
        .withColumn("review_ts", F.lit(None).cast("timestamp"))
        .withColumn("created_ts", F.lit(now)))

(DeltaTable.forName(spark, QUEUE).alias("t")
 .merge(high.alias("s"), "t.queue_id = s.queue_id")
 .whenNotMatchedInsertAll()
 .execute())

print(f"\nqueued as high:   {high.count():,}")
print(f"cost this run:    ¥{usage.cost:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Catalog health
# MAGIC
# MAGIC Counts by type and severity say how healthy the catalog is; the samples say whether
# MAGIC the audit itself is working. Both matter — an audit reporting thousands of `low`
# MAGIC findings nobody acts on is indistinguishable from an audit that found nothing.

# COMMAND ----------

f = spark.table(TARGET)
print(f"findings:  {f.count():,}  over {f.select('product_id').distinct().count():,} products")
print(f"of catalog: {f.select('product_id').distinct().count() / n_products:.1%}\n")

for r in (f.groupBy("source", "check_type", "severity").agg(F.count("*").alias("n"))
          .orderBy("source", "check_type", "severity").collect()):
    print(f"  {r['source']:12s} {r['check_type']:22s} {r['severity']:8s} {r['n']:>6,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data quality by merchant
# MAGIC
# MAGIC This is what attribution buys. Cleaning a bad size once helps this run; telling a
# MAGIC merchant that they send bad sizes helps every run after it. The rate matters more
# MAGIC than the count — a merchant with more SKUs will naturally have more findings.

# COMMAND ----------

per_merchant = (f.where(F.col("merchant_id").isNotNull())
                .groupBy("merchant_id")
                .agg(F.count("*").alias("findings"))
                .join(spark.table(NORM).groupBy("merchant_id").agg(F.count("*").alias("skus")),
                      "merchant_id")
                .withColumn("rate", F.round(F.col("findings") / F.col("skus") * 100, 3))
                .orderBy(F.desc("rate")))

display(per_merchant)

# COMMAND ----------

display(f.alias("f")
        .join(products.alias("p"), "product_id")
        .select("f.source", "f.check_type", "f.severity", "f.merchant_id", "f.details",
                "p.canonical_name", "p.brand_id", "p.category",
                "p.size_value", "p.size_unit", "p.pack_count")
        .orderBy("f.source", "f.severity")
        .limit(40))