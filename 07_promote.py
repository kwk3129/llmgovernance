# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # 07 · Routing and promotion
# MAGIC
# MAGIC 05 described each SKU, 06 worked out which ones are the same product. Both wrote to
# MAGIC `curated`, which nothing outside this pipeline reads. This stage produces the tables
# MAGIC that search, recommendation and price comparison actually query.
# MAGIC
# MAGIC Two things happen here, and the order matters.
# MAGIC
# MAGIC **Routing decides what is trustworthy enough to publish.** Extraction reported a
# MAGIC confidence per row; dedup left some pairs at `uncertain`. Rows above the auto
# MAGIC threshold go straight through, rows in the middle band go to a human queue, rows below
# MAGIC the lower threshold are held back entirely. Nothing that a model was unsure about
# MAGIC reaches the serving tables without a person having looked at it.
# MAGIC
# MAGIC **Promotion collapses clusters into products.** Each cluster from 06 becomes one row
# MAGIC in `products`; every merchant name that mapped into it becomes a row in
# MAGIC `product_aliases`. That alias table is what makes the next run cheaper — a name seen
# MAGIC before resolves by lookup instead of by embedding and a model call.

# COMMAND ----------

CATALOG = "govpipe"
NORM = f"{CATALOG}.curated.sku_normalized"
EXT = f"{CATALOG}.curated.sku_extractions"
MATCHES = f"{CATALOG}.curated.entity_matches"

PRODUCTS = f"{CATALOG}.serving.products"
BRANDS = f"{CATALOG}.serving.brands"
ALIASES = f"{CATALOG}.serving.product_aliases"
QUEUE = f"{CATALOG}.ops.review_queue"

EXTRACT_MODEL = "deepseek/deepseek-chat"
EXTRACT_PROMPT = "extract_v2"

# Routing thresholds, set from the confidence histogram rather than picked round.
# The model's scores pile up between 0.85 and 0.95, so a 0.90 cut lands mid-peak and
# sends half the catalog to review — 23,109 rows, which no team can work through.
# At 0.75 roughly 60% clears automatically and the queue drops to about 8,000.
# Still provisional: calibrating properly needs a labelled sample to measure how
# accurate the auto band actually is.
AUTO_THRESHOLD = 0.75
REVIEW_THRESHOLD = 0.60

# Only merge on an LLM edge at least this similar. Below it the verdict stands in the
# audit table but does not join two SKUs into one product.
MERGE_SIM_FLOOR = 0.85

import hashlib, json, re, unicodedata
from datetime import datetime, timezone

import pyspark.sql.functions as F
from delta.tables import DeltaTable
from pyspark.sql.types import (StructType, StructField, StringType, DoubleType,
                               IntegerType, MapType, TimestampType)

print(f"auto >= {AUTO_THRESHOLD}   review >= {REVIEW_THRESHOLD}   merge_sim >= {MERGE_SIM_FLOOR}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Tables
# MAGIC
# MAGIC `products` is the master record: one row per real product, whatever the merchants
# MAGIC called it. `brands` normalises brand spellings so `Coca Cola`, `coca-cola` and
# MAGIC `Coca-Cola` resolve to one id, with the variants kept as aliases. `product_aliases`
# MAGIC maps each merchant's own name onto a product id.
# MAGIC
# MAGIC `review_queue` lives in `ops` rather than `serving` because it is pipeline state, not
# MAGIC product data. Its `queue_id` is `sha1(sku_key|reason)`, which makes enqueueing
# MAGIC idempotent: rerunning this notebook cannot duplicate an item, and cannot reset one a
# MAGIC reviewer has already decided.

# COMMAND ----------

PRODUCTS_SCHEMA = StructType([
    StructField("product_id", StringType(), False),
    StructField("canonical_name", StringType(), True),
    StructField("brand_id", StringType(), True),
    StructField("category", StringType(), True),
    StructField("size_value", DoubleType(), True),
    StructField("size_unit", StringType(), True),
    StructField("pack_count", IntegerType(), True),
    StructField("attributes", MapType(StringType(), StringType()), True),
    StructField("member_count", IntegerType(), True),
    StructField("created_from", StringType(), True),      # auto_promote / human_review
    StructField("first_seen_ts", TimestampType(), False),
    StructField("last_updated_ts", TimestampType(), False),
])

BRANDS_SCHEMA = StructType([
    StructField("brand_id", StringType(), False),
    StructField("canonical_brand", StringType(), True),
    StructField("aliases", StringType(), True),           # JSON array
    StructField("last_updated_ts", TimestampType(), False),
])

ALIASES_SCHEMA = StructType([
    StructField("merchant_id", StringType(), False),
    StructField("raw_name", StringType(), False),
    StructField("product_id", StringType(), False),
    StructField("created_ts", TimestampType(), False),
])

QUEUE_SCHEMA = StructType([
    StructField("queue_id", StringType(), False),         # sha1(sku_key|reason)
    StructField("sku_key", StringType(), False),
    StructField("reason", StringType(), False),           # low_confidence / uncertain_match
    StructField("payload", StringType(), True),           # context snapshot for the reviewer
    StructField("status", StringType(), False),           # pending / approved / corrected / rejected
    StructField("corrected_payload", StringType(), True),
    StructField("reviewer", StringType(), True),
    StructField("review_ts", TimestampType(), True),
    StructField("created_ts", TimestampType(), False),
])

for name, schema in ((PRODUCTS, PRODUCTS_SCHEMA), (BRANDS, BRANDS_SCHEMA),
                     (ALIASES, ALIASES_SCHEMA), (QUEUE, QUEUE_SCHEMA)):
    if not spark.catalog.tableExists(name):
        spark.createDataFrame([], schema).write.saveAsTable(name)
        print(f"created {name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Route by confidence
# MAGIC
# MAGIC The model was asked to report how sure it was, and 05 stored that number. Here it
# MAGIC becomes a decision: publish, ask a human, or hold back.
# MAGIC
# MAGIC The thresholds above are a starting point, not a result. They should be calibrated
# MAGIC against labelled data — pick the point where accuracy inside the auto band is high
# MAGIC enough to publish unreviewed — and against the size of the review queue, which is the
# MAGIC real constraint.

# COMMAND ----------

ext = (spark.table(EXT)
       .where((F.col("model_version") == EXTRACT_MODEL)
              & (F.col("prompt_version") == EXTRACT_PROMPT)
              & (F.col("extraction_status") == "ok"))
       .select("sku_key", "brand_extracted", "name_normalized", "category",
               F.col("size_value").alias("x_size"), F.col("size_unit").alias("x_unit"),
               F.col("pack_count").alias("x_pack"), "attributes", "confidence"))

base = (spark.table(NORM)
        .select("sku_key", "merchant_id", "name_clean", "brand_clean",
                "size_value", "size_unit", "pack_count")
        .join(ext, "sku_key", "left")
        .withColumn("route",
                    F.when(F.col("confidence").isNull(), "no_extraction")
                     .when(F.col("confidence") >= AUTO_THRESHOLD, "auto")
                     .when(F.col("confidence") >= REVIEW_THRESHOLD, "review")
                     .otherwise("reject")))

routes = {r["route"]: r["n"] for r in
          base.groupBy("route").agg(F.count("*").alias("n")).collect()}
total = sum(routes.values())
for r in ("auto", "review", "reject", "no_extraction"):
    n = routes.get(r, 0)
    print(f"{r:14s} {n:>7,}  ({n / total:.1%})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Fill the review queue
# MAGIC
# MAGIC Two sources feed it. Extraction confidence in the middle band is one. Dedup verdicts
# MAGIC of `uncertain` are the other — that includes every pair the brand gate refused to
# MAGIC judge, which is the larger share.
# MAGIC
# MAGIC Each item carries a payload: enough context for a reviewer to decide without opening
# MAGIC three other tables. The insert is `whenNotMatchedInsert` only, so an item a reviewer
# MAGIC has already resolved is never reset by a later run.

# COMMAND ----------

now = datetime.now(timezone.utc)

low_conf = (base.where(F.col("route") == "review")
            .select(
                F.sha1(F.concat_ws("|", "sku_key", F.lit("low_confidence"))).alias("queue_id"),
                "sku_key",
                F.lit("low_confidence").alias("reason"),
                F.to_json(F.struct("merchant_id", "name_clean", "brand_clean",
                                   "brand_extracted", "name_normalized", "category",
                                   "confidence")).alias("payload")))

uncertain = (spark.table(MATCHES)
             .where(F.col("llm_verdict") == "uncertain")
             .groupBy("sku_key")
             .agg(F.to_json(F.collect_list(F.struct("candidate_product_id", "method",
                                                    "similarity", "llm_rationale"))).alias("payload"))
             .select(
                 F.sha1(F.concat_ws("|", "sku_key", F.lit("uncertain_match"))).alias("queue_id"),
                 "sku_key",
                 F.lit("uncertain_match").alias("reason"),
                 "payload"))

enq = (low_conf.unionByName(uncertain)
       .dropDuplicates(["queue_id"])
       .withColumn("status", F.lit("pending"))
       .withColumn("corrected_payload", F.lit(None).cast("string"))
       .withColumn("reviewer", F.lit(None).cast("string"))
       .withColumn("review_ts", F.lit(None).cast("timestamp"))
       .withColumn("created_ts", F.lit(now)))

(DeltaTable.forName(spark, QUEUE).alias("t")
 .merge(enq.select(*[f.name for f in QUEUE_SCHEMA]).alias("s"), "t.queue_id = s.queue_id")
 .whenNotMatchedInsertAll()
 .execute())

q = spark.table(QUEUE)
print(f"queue total:   {q.count():,}")
for r in q.groupBy("reason", "status").agg(F.count("*").alias("n")).collect():
    print(f"  {r['reason']:18s} {r['status']:10s} {r['n']:>7,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Who is eligible to publish
# MAGIC
# MAGIC `auto` rows, minus anything still sitting in the queue unresolved. A row can be
# MAGIC confidently extracted and still be queued because its dedup verdict was uncertain —
# MAGIC publishing it would mean guessing which product it belongs to, so it waits.
# MAGIC
# MAGIC On a first run nothing has been reviewed, so the queue holds everything it received.
# MAGIC As reviewers work through it, `approved` and `corrected` rows join the eligible set on
# MAGIC the next run without anything else changing. That is the flywheel: the queue drains,
# MAGIC the catalog grows, and the corrections become labelled data.

# COMMAND ----------

pending = q.where(F.col("status") == "pending").select("sku_key").distinct()
resolved = q.where(F.col("status").isin("approved", "corrected")).select("sku_key").distinct()

corrections = {r["sku_key"]: json.loads(r["corrected_payload"])
               for r in q.where((F.col("status") == "corrected")
                                & F.col("corrected_payload").isNotNull()).collect()}

auto_part = base.where(F.col("route") == "auto").join(pending, "sku_key", "left_anti")
review_part = (base.where(F.col("route") == "review")
               .join(pending, "sku_key", "left_anti")
               .join(resolved, "sku_key"))

eligible = auto_part.unionByName(review_part)
rows = eligible.collect()

print(f"auto, not queued:      {auto_part.count():,}")
print(f"review, resolved:      {review_part.count():,}")
print(f"eligible to publish:   {len(rows):,}")
print(f"held in queue:         {pending.count():,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Rebuild the clusters
# MAGIC
# MAGIC Same union-find as 06, with one difference: an LLM `duplicate` edge only counts if
# MAGIC its similarity clears `MERGE_SIM_FLOOR`. The verdict stays in `entity_matches` either
# MAGIC way — the audit trail keeps everything — but a weak edge does not get to fuse two
# MAGIC products in the published catalog.
# MAGIC
# MAGIC The reason for the extra floor here is transitivity. A=B and B=C silently merges A
# MAGIC with C, so one bad edge can chain unrelated clusters together, and this is the last
# MAGIC point where that can be contained.

# COMMAND ----------

class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        p = self.parent.setdefault(x, x)
        if p != x:
            self.parent[x] = p = self.find(p)
        return p

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra

matches = spark.table(MATCHES).select("sku_key", "candidate_product_id", "method",
                                      "llm_verdict", "similarity").collect()

edges, best = [], {}
for r in matches:
    if r["candidate_product_id"] is None:
        continue
    if r["method"] == "barcode_exact":
        edges.append((r["sku_key"], r["candidate_product_id"]))
    elif r["method"] == "rag_llm" and r["llm_verdict"] == "duplicate":
        sim = r["similarity"] or 0.0
        cur = best.get(r["sku_key"])
        if cur is None or sim > cur[1]:
            best[r["sku_key"]] = (r["candidate_product_id"], sim)
edges += [(sk, c) for sk, (c, sim) in best.items() if sim >= MERGE_SIM_FLOOR]

uf = UnionFind()
for a, b in edges:
    uf.union(a, b)

comps = {}
for node in list(uf.parent):
    comps.setdefault(uf.find(node), []).append(node)

assign = {}
for members_ in comps.values():
    bc = sorted(m for m in members_ if m.startswith("bc_"))
    skus = sorted(set([m for m in members_ if not m.startswith(("bc_", "em_"))]
                      + [m[3:] for m in members_ if m.startswith("em_")]))
    if not skus:
        continue
    cid = bc[0] if bc else f"em_{skus[0]}"
    for s in skus:
        assign[s] = cid

print(f"edges kept:        {len(edges):,}")
print(f"SKUs in a cluster: {len(assign):,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Collapse clusters into products
# MAGIC
# MAGIC One row per cluster. Where cluster members disagree on a field — different merchants
# MAGIC extracted different brand spellings — the member with the highest extraction
# MAGIC confidence wins. A SKU with no cluster becomes a product of its own.
# MAGIC
# MAGIC Brand ids are normalised: lowercased, accents folded, legal suffixes stripped,
# MAGIC everything else hyphenated. So `Coca-Cola`, `coca cola` and `Coca Cola S.A.` all land
# MAGIC on `coca-cola`, and each spelling is kept in that brand's alias list. Without this
# MAGIC step the brand dimension fragments into near-duplicates and no query can group by it.

# COMMAND ----------

LEGAL = re.compile(r"\b(s\.?a\.?s?|s\.?l\.?|s\.?r\.?l\.?|gmbh|ag|ltd|limited|inc|llc|"
                   r"co|corp|company|bv|nv|plc|kg|oy|ab)\b\.?")

def norm_brand(b):
    """Fold accents, drop legal suffixes, hyphenate the rest. Returns None when nothing
    usable survives — stripping to ASCII would erase Japanese, Chinese and Cyrillic
    brands entirely, so only combining marks are removed and other scripts stay."""
    t = (b or "").strip().lower()
    t = "".join(c for c in unicodedata.normalize("NFKD", t) if not unicodedata.combining(c))
    t = LEGAL.sub(" ", t)
    return re.sub(r"[^\w]+", "-", t, flags=re.UNICODE).strip("-") or None

members, best_of = {}, {}
for r in rows:
    d = r.asDict()
    if d["sku_key"] in corrections:
        d.update({k: v for k, v in corrections[d["sku_key"]].items() if v is not None})
        d["created_from"] = "human_review"
    else:
        d["created_from"] = "auto_promote"
    cid = assign.get(d["sku_key"]) or f"solo_{d['sku_key']}"
    members.setdefault(cid, []).append(d)
    cur = best_of.get(cid)
    if cur is None or (d["confidence"] or 0) > (cur["confidence"] or 0):
        best_of[cid] = d

products, brands, aliases = [], {}, []
existing = {r["brand_id"]: json.loads(r["aliases"] or "[]")
            for r in spark.table(BRANDS).collect()}

for cid, rep in best_of.items():
    pid = hashlib.sha1(cid.encode()).hexdigest()[:16]
    brand = rep.get("brand_extracted")
    bid = norm_brand(brand) if brand else None
    if bid:
        seen = set(existing.get(bid, [])) | {m["brand_extracted"] for m in members[cid]
                                             if m.get("brand_extracted")}
        brands[bid] = (brand, sorted(seen))
    products.append((
        pid, rep.get("name_normalized"), bid, rep.get("category"),
        rep.get("x_size") if rep.get("x_size") is not None else rep.get("size_value"),
        rep.get("x_unit") or rep.get("size_unit"),
        rep.get("x_pack") or rep.get("pack_count"),
        rep.get("attributes"), len(members[cid]), rep["created_from"],
    ))
    for m in members[cid]:
        if m.get("name_clean"):
            aliases.append((m["merchant_id"], m["name_clean"], pid))

print(f"products:  {len(products):,}")
print(f"brands:    {len(brands):,}")
print(f"aliases:   {len(aliases):,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write the serving tables
# MAGIC
# MAGIC All three use `MERGE`, not `append`. This notebook will run again — after reviewers
# MAGIC drain the queue, after a new merchant feed arrives — and a second run must update
# MAGIC what exists rather than write a second copy of it. `first_seen_ts` is set once on
# MAGIC insert and never touched again; `last_updated_ts` moves on every update.

# COMMAND ----------

ts = F.lit(now)

prod_df = (spark.createDataFrame(
    products,
    "product_id string, canonical_name string, brand_id string, category string, "
    "size_value double, size_unit string, pack_count int, "
    "attributes map<string,string>, member_count int, created_from string")
    .dropDuplicates(["product_id"])
    .withColumn("first_seen_ts", ts).withColumn("last_updated_ts", ts))

(DeltaTable.forName(spark, PRODUCTS).alias("t")
 .merge(prod_df.alias("s"), "t.product_id = s.product_id")
 .whenMatchedUpdate(set={c: f"s.{c}" for c in
                         ("canonical_name", "brand_id", "category", "size_value",
                          "size_unit", "pack_count", "attributes", "member_count",
                          "created_from", "last_updated_ts")})
 .whenNotMatchedInsertAll()
 .execute())

brand_df = (spark.createDataFrame(
    [(bid, b, json.dumps(al, ensure_ascii=False)) for bid, (b, al) in brands.items()],
    "brand_id string, canonical_brand string, aliases string")
    .withColumn("last_updated_ts", ts))

(DeltaTable.forName(spark, BRANDS).alias("t")
 .merge(brand_df.alias("s"), "t.brand_id = s.brand_id")
 .whenMatchedUpdate(set={"aliases": "s.aliases", "last_updated_ts": "s.last_updated_ts"})
 .whenNotMatchedInsertAll()
 .execute())

alias_df = (spark.createDataFrame(
    aliases, "merchant_id string, raw_name string, product_id string")
    .dropDuplicates(["merchant_id", "raw_name"])
    .withColumn("created_ts", ts))

(DeltaTable.forName(spark, ALIASES).alias("t")
 .merge(alias_df.alias("s"), "t.merchant_id = s.merchant_id AND t.raw_name = s.raw_name")
 .whenMatchedUpdate(set={"product_id": "s.product_id"})
 .whenNotMatchedInsertAll()
 .execute())

print(f"{PRODUCTS}: {spark.table(PRODUCTS).count():,} rows")
print(f"{BRANDS}:   {spark.table(BRANDS).count():,} rows")
print(f"{ALIASES}:  {spark.table(ALIASES).count():,} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify
# MAGIC
# MAGIC Four things have to hold before anything downstream reads these tables: product ids
# MAGIC are unique, every alias points at a product that exists, every brand referenced by a
# MAGIC product exists in the brand table, and nothing still sitting in the review queue has
# MAGIC leaked into the catalog.
# MAGIC
# MAGIC The last one is the one worth stating explicitly. The queue is a gate; if a pending
# MAGIC row can also be published, the gate is decorative.

# COMMAND ----------

prod = spark.table(PRODUCTS)
al = spark.table(ALIASES)
br = spark.table(BRANDS)

n_prod = prod.count()
assert prod.select("product_id").distinct().count() == n_prod, "duplicate product_id"

orphan_alias = al.join(prod, al.product_id == prod.product_id, "left_anti").count()
assert orphan_alias == 0, f"{orphan_alias} aliases point at a missing product"

missing_brand = (prod.where(F.col("brand_id").isNotNull())
                 .join(br, "brand_id", "left_anti").count())
assert missing_brand == 0, f"{missing_brand} products reference a missing brand"

eligible_keys = {r["sku_key"] for r in rows}
pending_keys = {r["sku_key"] for r in pending.collect()}
leaked = eligible_keys & pending_keys
assert not leaked, f"{len(leaked)} pending SKUs were eligible for publication"

print("checks passed\n")
print(f"products:        {n_prod:,}")
print(f"  with a brand:  {prod.where('brand_id IS NOT NULL').count():,}")
print(f"  multi-SKU:     {prod.where('member_count > 1').count():,}")
print(f"brands:          {br.count():,}")
print(f"aliases:         {al.count():,}")
print(f"queue pending:   {pending.count():,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## What the catalog looks like
# MAGIC
# MAGIC The products with the most merchants behind them. This is the output the whole
# MAGIC pipeline exists for: one row, one real product, and the list of names six different
# MAGIC merchants used for it.

# COMMAND ----------

display(prod.alias("p")
        .join(al.alias("a"), "product_id")
        .groupBy("p.product_id", "p.canonical_name", "p.brand_id",
                 "p.category", "p.size_value", "p.size_unit")
        .agg(F.count("*").alias("merchants"),
             F.collect_list("a.raw_name").alias("merchant_names"))
        .orderBy(F.desc("merchants"))
        .limit(25))

# COMMAND ----------

