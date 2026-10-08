# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # 06 · Entity resolution
# MAGIC
# MAGIC 05 answered *what is this SKU* — brand, category, size, one row at a time. After it,
# MAGIC six merchants' Coca-Cola Zero 2L six-packs all carry `Coca-Cola` as the brand and the
# MAGIC same cleaned-up name. The `CC Zero 2L x6` spellings are gone.
# MAGIC
# MAGIC They are still six separate rows, though, each with its own `sku_key`. A search now
# MAGIC matches all six and returns six entries that look identical, differing only in
# MAGIC merchant and price — where what the user wants is one product with six prices under
# MAGIC it. Nothing in 05 could have known they belong together: it read one row at a time
# MAGIC and never compared any two.
# MAGIC
# MAGIC So 05 made them look the same. This stage works out that they *are* the same, and
# MAGIC groups them.

# COMMAND ----------

CATALOG = "govpipe"
NORM = f"{CATALOG}.curated.sku_normalized"
EXT = f"{CATALOG}.curated.sku_extractions"
TARGET = f"{CATALOG}.curated.entity_matches"

# Which extraction run to read. 05 stored three (SiliconFlow V3, DeepSeek thinking,
# DeepSeek chat); only the last one is complete.
EXTRACT_MODEL = "deepseek/deepseek-chat"
EXTRACT_PROMPT = "extract_v2"

EMBED_MODEL = "BAAI/bge-m3"
EMBED_VERSION = "siliconflow/BAAI/bge-m3"
JUDGE_MODEL = "deepseek-chat"
JUDGE_VERSION = "deepseek/deepseek-chat"
JUDGE_PROMPT = "dedup_v1"

TOP_K = 5              # neighbours retrieved per query
SIM_FLOOR = 0.60       # below this, not even a candidate
MAX_CANDIDATES = 3     # pairs sent to the model per query, best-first
PAIRS_PER_CALL = 20
MAX_LLM_CALLS = 800    # budget ceiling; pairs beyond it are left for the next run
EMBED_BATCH = 32
WORKERS = 24

# Clear these lines before exporting or sharing the notebook.
EMBED_KEY = "xxxx"         # SiliconFlow — DeepSeek serves no embedding model
JUDGE_KEY = "xxxx"         # DeepSeek
assert EMBED_KEY and JUDGE_KEY, "paste both API keys above"

import json, time, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import numpy as np
import pyspark.sql.functions as F
from pyspark.sql.types import (StructType, StructField, StringType, DoubleType,
                               TimestampType)

print(f"embed={EMBED_VERSION}  judge={JUDGE_VERSION}  prompt={JUDGE_PROMPT}")
print(f"top_k={TOP_K}  sim_floor={SIM_FLOOR}  budget={MAX_LLM_CALLS} calls")

# COMMAND ----------

# MAGIC %md
# MAGIC ## What gets compared
# MAGIC
# MAGIC The normalized rows carry merchant, name and barcode; the extraction rows carry the
# MAGIC model's brand, canonical name and size. Both are needed — the barcode decides tier 1,
# MAGIC and the extracted fields are what tiers 2 and 3 read, because raw merchant names like
# MAGIC `Qs frsc de ovj` retrieve nothing useful.
# MAGIC
# MAGIC Rows already adjudicated in an earlier run are excluded, same incremental pattern as
# MAGIC 05: this stage spends money, so a rerun must not pay twice for the same SKU.

# COMMAND ----------

MATCHES_SCHEMA = StructType([
    StructField("sku_key", StringType(), False),
    StructField("candidate_product_id", StringType(), True),
    StructField("method", StringType(), False),        # barcode_exact / rag_llm / brand_gate / no_candidate
    StructField("similarity", DoubleType(), True),
    StructField("llm_verdict", StringType(), True),    # duplicate / unique / uncertain
    StructField("llm_rationale", StringType(), True),
    StructField("model_version", StringType(), True),
    StructField("prompt_version", StringType(), True),
    StructField("matched_ts", TimestampType(), False),
])

if not spark.catalog.tableExists(TARGET):
    spark.createDataFrame([], MATCHES_SCHEMA).write.saveAsTable(TARGET)
    print(f"created empty {TARGET}")

ext = (spark.table(EXT)
       .where((F.col("model_version") == EXTRACT_MODEL)
              & (F.col("prompt_version") == EXTRACT_PROMPT)
              & (F.col("extraction_status") == "ok"))
       .select("sku_key", "brand_extracted", "name_normalized",
               F.col("size_value").alias("x_size"),
               F.col("size_unit").alias("x_unit"),
               F.col("pack_count").alias("x_pack")))

base = (spark.table(NORM)
        .select("sku_key", "merchant_id", "name_clean", "brand_clean", "barcode_clean",
                "size_value", "size_unit", "pack_count")
        .join(ext, "sku_key", "left"))

done = spark.table(TARGET).select("sku_key").distinct()
pend = base.join(done, "sku_key", "left_anti")

n_pend = pend.count()
print(f"already adjudicated: {done.count():,}")
print(f"to adjudicate:       {n_pend:,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Tier 1 · barcode equality
# MAGIC
# MAGIC A barcode is a global product identifier, so two rows sharing one are the same
# MAGIC product with no further argument. This tier is a `GROUP BY` — free, exact, and it
# MAGIC resolves the bulk of cross-merchant duplication.
# MAGIC
# MAGIC Only 8, 12 and 13 digit all-numeric values count. m04 mistypes 4% of its barcodes,
# MAGIC and a mistyped code usually still passes that check — it forms a singleton cluster
# MAGIC that looks perfectly legitimate. Nothing here can catch that; tier 3 is what
# MAGIC eventually pulls those rows back into their real cluster.

# COMMAND ----------

valid_bc = (F.length("barcode_clean").isin(8, 12, 13)
            & F.col("barcode_clean").rlike("^[0-9]+$"))

coded = pend.where(F.col("barcode_clean").isNotNull() & valid_bc)
clusters = (coded.groupBy("barcode_clean").agg(F.count("*").alias("n"))
            .where("n >= 2"))

bc_hit = (coded.join(clusters, "barcode_clean")
          .select("sku_key",
                  F.concat(F.lit("bc_"), F.col("barcode_clean")).alias("candidate_product_id")))

n_bc = bc_hit.count()
n_bc_clusters = clusters.count()
print(f"barcode clusters:  {n_bc_clusters:,}")
print(f"rows clustered:    {n_bc:,}")

remaining = pend.join(bc_hit.select("sku_key"), "sku_key", "left_anti")
n_rem = remaining.count()
print(f"needs retrieval:   {n_rem:,}  ({n_rem / n_pend:.1%} of pending)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Tier 2 · embedding retrieval
# MAGIC
# MAGIC What is left has no usable barcode, or a barcode nobody else shares. Comparing them
# MAGIC exhaustively is 45,086² ≈ 2 billion pairs, so instead every record becomes a vector
# MAGIC and each one retrieves its nearest neighbours.
# MAGIC
# MAGIC The text embedded is `brand | name | size`, built from the extracted fields with the
# MAGIC raw ones as fallback. Size belongs in the string because 500 ml and 2 L are different
# MAGIC products, and a model comparing names alone would call them identical.
# MAGIC
# MAGIC The index holds one representative per barcode cluster plus every unresolved row, so
# MAGIC a row whose barcode was dropped can still find the cluster it belongs to. Retrieval
# MAGIC decides nothing — anything at or above `SIM_FLOOR` becomes a candidate pair and the
# MAGIC verdict is tier 3's job. A neighbour below the floor is dropped without being seen by
# MAGIC any model, and that is where the cost saving lives.
# MAGIC
# MAGIC Similarities are computed with a dense matrix multiply on the driver. At this scale
# MAGIC that is a few hundred megabytes and about a minute; a real catalog would need a
# MAGIC vector index instead.

# COMMAND ----------

EMBED_ENDPOINT = "https://api.siliconflow.cn/v1/embeddings"
PRICE_EMBED_PER_M = 0.5     # CNY per million tokens — check against the invoice

class Usage:
    def __init__(self, price_in, price_out=0.0):
        self.calls = self.retries = self.failures = 0
        self.tok_in = self.tok_out = 0
        self.price_in, self.price_out = price_in, price_out

    def add(self, u):
        self.calls += 1
        self.tok_in += u.get("prompt_tokens", u.get("total_tokens", 0))
        self.tok_out += u.get("completion_tokens", 0)

    @property
    def cost(self):
        return (self.tok_in * self.price_in + self.tok_out * self.price_out) / 1_000_000

    def __str__(self):
        return (f"{self.calls:,} calls  {self.retries} retries  {self.failures} failures  "
                f"{self.tok_in:,} in / {self.tok_out:,} out  ¥{self.cost:.4f}")

embed_usage = Usage(PRICE_EMBED_PER_M)

def post_json(endpoint, key, payload, attempts=5, timeout=180):
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        endpoint, data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            # 401 and 402 stay broken until a human fixes them — an exhausted balance or a
            # dead key will fail every remaining batch, so stop the run instead of
            # printing the same error several hundred times.
            if e.code in (401, 402):
                raise RuntimeError(
                    f"HTTP {e.code} — key rejected or account out of credit. "
                    "Nothing further will succeed; fix the account and rerun.") from e
            # 429 and 5xx are transient; anything else is a bad request
            if e.code != 429 and e.code < 500:
                raise
            if attempt == attempts - 1:
                raise
            time.sleep(2 ** attempt)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            if attempt == attempts - 1:
                raise
            time.sleep(2 ** attempt)

def embed_chunk(chunk):
    out = post_json(EMBED_ENDPOINT, EMBED_KEY, {"model": EMBED_MODEL, "input": chunk})
    embed_usage.add(out.get("usage", {}))
    return [d["embedding"] for d in sorted(out["data"], key=lambda d: d["index"])]

def rag_text(r):
    brand = r["brand_extracted"] or r["brand_clean"] or ""
    name = r["name_normalized"] or r["name_clean"] or ""
    sv = r["x_size"] if r["x_size"] is not None else r["size_value"]
    su = r["x_unit"] or r["size_unit"] or ""
    pc = r["x_pack"] or r["pack_count"]
    size = f"{sv:g}{su}" if sv is not None and su else "?"
    if pc and pc > 1:
        size = f"{size} x{pc}"
    return f"{brand} | {name} | {size}", brand

# COMMAND ----------

# One representative per barcode cluster (smallest sku_key), plus everything unresolved.
rep = (bc_hit.groupBy("candidate_product_id").agg(F.min("sku_key").alias("sku_key"))
       .join(base, "sku_key")
       .collect())
rem_rows = remaining.orderBy("sku_key").collect()

index_ids, index_meta = [], {}
for r in rep:
    text, brand = rag_text(r)
    index_ids.append(r["candidate_product_id"])
    index_meta[r["candidate_product_id"]] = (text, brand)
n_rep = len(index_ids)

for r in rem_rows:
    cid = f"em_{r['sku_key']}"
    text, brand = rag_text(r)
    index_ids.append(cid)
    index_meta[cid] = (text, brand)

texts = [index_meta[i][0] for i in index_ids]
print(f"index: {n_rep:,} cluster reps + {len(rem_rows):,} unresolved = {len(texts):,} vectors")

chunks = [texts[i:i + EMBED_BATCH] for i in range(0, len(texts), EMBED_BATCH)]
started = time.time()
with ThreadPoolExecutor(max_workers=WORKERS) as pool:
    vecs = [v for part in pool.map(embed_chunk, chunks) for v in part]

V = np.asarray(vecs, dtype=np.float32)
V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-9
print(f"embedded in {(time.time() - started) / 60:.1f} min  |  {embed_usage}")
print(f"matrix: {V.shape}")

# COMMAND ----------

# Cosine similarity in blocks: queries are the unresolved rows, the index is everything.
q_ids = [f"em_{r['sku_key']}" for r in rem_rows]
cand_lists = []
Q = V[n_rep:]

for s0 in range(0, Q.shape[0], 1000):
    S = Q[s0:s0 + 1000] @ V.T
    for qi in range(S.shape[0]):
        S[qi, n_rep + s0 + qi] = -1.0                      # mask self
        idx = np.argpartition(-S[qi], TOP_K)[:TOP_K]
        cands = sorted(((index_ids[j], float(S[qi, j])) for j in idx
                        if S[qi, j] >= SIM_FLOOR), key=lambda t: -t[1])[:MAX_CANDIDATES]
        cand_lists.append(cands)

n_with_cand = sum(1 for c in cand_lists if c)
print(f"queries with at least one candidate: {n_with_cand:,} / {len(cand_lists):,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Tier 3 · LLM adjudication
# MAGIC
# MAGIC Candidate pairs are deduplicated (A-B and B-A are one pair), sorted by similarity and
# MAGIC cut to the call budget, so if there is more work than budget the most promising pairs
# MAGIC are the ones that get spent on. Whatever is cut is simply not written, and the
# MAGIC incremental filter picks it up next run.
# MAGIC
# MAGIC One free gate first: if neither record has a brand, no model can responsibly decide —
# MAGIC `Tomato Sauce 500g` and `Tomato Sauce 500g` from two merchants may be two different
# MAGIC store brands. Those go straight to `uncertain` and cost nothing.

# COMMAND ----------

JUDGE_ENDPOINT = "https://api.deepseek.com/chat/completions"
PRICE_IN_PER_M, PRICE_OUT_PER_M = 1.0, 2.0

judge_usage = Usage(PRICE_IN_PER_M, PRICE_OUT_PER_M)

DEDUP_PROMPT = """You are a product-catalog entity-resolution specialist for an instant-retail
platform. You receive candidate pairs of SKU records from different merchants.
For each pair, decide whether the two records refer to THE SAME real-world
product (same brand, same product, same size/pack — merchant, price and
language may differ).

Rules:
- Same product sold in different languages or with abbreviated names is a
  duplicate ("CC Zero 2L x6" vs "Coca-Cola Zero 2000ml 6-pack" = duplicate).
- Different size or pack count means NOT the same product
  ("500 ml" vs "2 L" = unique), unless one side simply lacks size info.
- Different flavor/variant means NOT the same product (Zero vs Classic).
- If evidence is genuinely insufficient, answer "uncertain" — do not guess.

Respond with a single JSON object exactly in this shape, no other text:

{"items": [
  {"pair_id": "p1", "verdict": "duplicate", "rationale": "same brand, same 2L x6 cola, names in different languages"}
]}

verdict must be one of: "duplicate", "unique", "uncertain".
Keep rationale under 20 words. One item per input pair, same pair_id."""

# Pair assembly: symmetric dedup, then split into gated and judgeable.
pair_map, seen = {}, set()
for qi, cands in enumerate(cand_lists):
    for cid, sim in cands:
        key = frozenset((q_ids[qi], cid))
        if key in seen:
            continue
        seen.add(key)
        pair_map[f"p{len(pair_map)}"] = (q_ids[qi], cid, sim)

gated, judgeable = [], {}
for pid, (a, b, sim) in pair_map.items():
    if not index_meta[a][1] and not index_meta[b][1]:
        gated.append((pid, (a, b, sim)))
    else:
        judgeable[pid] = (a, b, sim)

ordered = sorted(judgeable.items(), key=lambda kv: -kv[1][2])
budget = MAX_LLM_CALLS * PAIRS_PER_CALL
truncated = max(0, len(ordered) - budget)
ordered = ordered[:budget]

print(f"candidate pairs:   {len(pair_map):,}")
print(f"gated (no brand):  {len(gated):,}")
print(f"to judge:          {len(ordered):,}")
if truncated:
    print(f"deferred (budget): {truncated:,}  -> next run")

# COMMAND ----------

def judge_batch(batch):
    payload = {
        "model": JUDGE_MODEL,
        "messages": [
            {"role": "system", "content": DEDUP_PROMPT},
            {"role": "user", "content": json.dumps({"pairs": [
                {"pair_id": pid,
                 "a": {"text": index_meta[a][0]},
                 "b": {"text": index_meta[b][0]}}
                for pid, (a, b, _) in batch]}, ensure_ascii=False)},
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0,
    }
    try:
        out = post_json(JUDGE_ENDPOINT, JUDGE_KEY, payload)
        judge_usage.add(out.get("usage", {}))
        items = json.loads(out["choices"][0]["message"]["content"]).get("items", [])
        return batch, {i.get("pair_id"): i for i in items if isinstance(i, dict)}
    except Exception as e:
        judge_usage.failures += 1
        print(f"  batch failed: {type(e).__name__} {str(e)[:120]}")
        return batch, {}

batches = [ordered[i:i + PAIRS_PER_CALL] for i in range(0, len(ordered), PAIRS_PER_CALL)]
print(f"{len(batches)} batches of up to {PAIRS_PER_CALL} pairs\n")

started = time.time()
with ThreadPoolExecutor(max_workers=WORKERS) as pool:
    outcomes = list(pool.map(judge_batch, batches))
print(f"judged in {(time.time() - started) / 60:.1f} min  |  {judge_usage}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write the verdicts
# MAGIC
# MAGIC Every row records how the decision was reached, not just what it was: the method, the
# MAGIC similarity, the model's verdict and its one-line reason. A duplicate that turns out to
# MAGIC be wrong has to be traceable to the tier and the model version that produced it.
# MAGIC
# MAGIC Rows whose candidates all fell below the floor are written as `no_candidate`, which
# MAGIC means "judged to be a new product". Writing them matters — without a row they would
# MAGIC look unprocessed and be paid for again on every rerun.

# COMMAND ----------

VERDICTS = {"duplicate", "unique", "uncertain"}
now = datetime.now(timezone.utc)
rows, judged = [], set()

def emit(sku, cand, method, sim, verdict, rationale, model, prompt):
    rows.append((sku, cand, method, sim, verdict, rationale, model, prompt, now))
    judged.add(sku)

for pid, (a, b, sim) in gated:
    q, c = (a, b) if a.startswith("em_") else (b, a)
    emit(q[3:], c, "brand_gate", sim, "uncertain", "both records lack brand", None, None)
    if c.startswith("em_"):
        emit(c[3:], q, "brand_gate", sim, "uncertain", "both records lack brand", None, None)

counts = {"duplicate": 0, "unique": 0, "uncertain": 0}
n_failed_batches = 0
for batch, out in outcomes:
    if not out:
        # The whole batch failed. Writing rows here would mark these pairs as adjudicated
        # and the incremental filter would skip them forever — leave them unwritten so
        # the next run picks them up.
        n_failed_batches += 1
        continue
    for pid, (a, b, sim) in batch:
        item = out.get(pid) or {}
        verdict = item.get("verdict")
        if verdict not in VERDICTS:
            verdict, rationale = "uncertain", "no valid answer"
        else:
            rationale = str(item.get("rationale", ""))[:200]
        counts[verdict] += 1
        q, c = (a, b) if a.startswith("em_") else (b, a)
        emit(q[3:], c, "rag_llm", sim, verdict, rationale, JUDGE_VERSION, JUDGE_PROMPT)
        if c.startswith("em_"):
            emit(c[3:], q, "rag_llm", sim, verdict, rationale, JUDGE_VERSION, JUDGE_PROMPT)

n_no_cand = 0
for qi, cands in enumerate(cand_lists):
    sk = q_ids[qi][3:]
    if not cands and sk not in judged:
        n_no_cand += 1
        rows.append((sk, None, "no_candidate", None, None, None, None, None, now))

bc_rows = [(r["sku_key"], r["candidate_product_id"], "barcode_exact", 1.0,
            None, None, None, None, now) for r in bc_hit.collect()]

all_rows = bc_rows + rows
spark.createDataFrame(all_rows, MATCHES_SCHEMA).write.mode("append").saveAsTable(TARGET)

print(f"barcode_exact:  {len(bc_rows):,}")
print(f"rag_llm:        {counts['duplicate'] + counts['unique'] + counts['uncertain']:,}"
      f"  (dup {counts['duplicate']:,} / uniq {counts['unique']:,} / unc {counts['uncertain']:,})")
print(f"brand_gate:     {len(gated):,}")
print(f"no_candidate:   {n_no_cand:,}")
if n_failed_batches:
    print(f"failed batches: {n_failed_batches:,}  ({n_failed_batches * PAIRS_PER_CALL:,} "
          f"pairs unwritten -> next run)")
print(f"\ntotal written:  {len(all_rows):,}")
print(f"cost this run:  ¥{embed_usage.cost + judge_usage.cost:.4f}"
      f"  (embed ¥{embed_usage.cost:.4f} + judge ¥{judge_usage.cost:.4f})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Build the clusters
# MAGIC
# MAGIC The verdicts are pairwise; products are groups. Union-find turns one into the other:
# MAGIC every barcode match and every `duplicate` verdict is an edge, and each connected
# MAGIC component is one product.
# MAGIC
# MAGIC Only the highest-similarity `duplicate` edge per SKU is kept. Transitivity is what
# MAGIC makes this necessary — A=B and B=C silently merges A with C, so a single wrong edge
# MAGIC can chain two unrelated clusters together. Keeping the best edge per row bounds how
# MAGIC far one mistake can propagate.

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

matches = spark.table(TARGET).collect()

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
edges += [(sk, c) for sk, (c, _) in best.items()]

uf = UnionFind()
for a, b in edges:
    uf.union(a, b)

comps = {}
for node in list(uf.parent):
    comps.setdefault(uf.find(node), []).append(node)

assign = {}
for members in comps.values():
    bc = sorted(m for m in members if m.startswith("bc_"))
    skus = sorted(set([m for m in members if not m.startswith(("bc_", "em_"))]
                      + [m[3:] for m in members if m.startswith("em_")]))
    if not skus:
        continue
    cid = bc[0] if bc else f"em_{skus[0]}"
    for s in skus:
        assign[s] = cid

sizes = {}
for cid in assign.values():
    sizes[cid] = sizes.get(cid, 0) + 1
multi = {c: n for c, n in sizes.items() if n > 1}

print(f"SKUs in a cluster:  {len(assign):,}")
print(f"clusters:           {len(sizes):,}")
print(f"  of size 1:        {len(sizes) - len(multi):,}")
print(f"  of size >1:       {len(multi):,}")
print(f"largest cluster:    {max(sizes.values()) if sizes else 0}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Score against ground truth
# MAGIC
# MAGIC 02 kept the Open Food Facts barcode each generated SKU came from, in a table the
# MAGIC governance stages never read. Two SKUs share an `off_code` exactly when they are the
# MAGIC same product, which makes it possible to score this stage rather than eyeball it.
# MAGIC
# MAGIC The metric is pairwise: of the SKU pairs this stage put in one cluster, how many are
# MAGIC truly the same product (precision), and of the pairs that truly are, how many did it
# MAGIC find (recall). Precision is the number to watch — a merged pair that should not have
# MAGIC been merged makes two products disappear into one, and nothing downstream can undo it.

# COMMAND ----------

truth = (spark.table(f"{CATALOG}.ops.feed_ground_truth")
         .select(F.sha1(F.concat_ws(":", "merchant_id", "merchant_sku_id")).alias("sku_key"),
                 "off_code")
         .collect())
truth_of = {r["sku_key"]: r["off_code"] for r in truth}

def pairs_from(groups):
    """groups: dict[sku -> group_id] -> set of unordered sku pairs inside a group"""
    by_group = {}
    for sku, gid in groups.items():
        by_group.setdefault(gid, []).append(sku)
    out = set()
    for members in by_group.values():
        if len(members) < 2 or len(members) > 200:      # skip degenerate giants
            continue
        members = sorted(members)
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                out.add((members[i], members[j]))
    return out

# Every SKU this stage has processed takes part, not just the ones that ended up in a
# cluster. A row judged unique or left without candidates is a cluster of one — scoring
# only the clustered rows would leave every miss outside the denominator and report a
# recall that cannot go down.
all_skus = {r["sku_key"] for r in matches}
full_assign = {s: assign.get(s, f"single_{s}") for s in all_skus}

scored = {s: c for s, c in full_assign.items() if s in truth_of}
pred = pairs_from(scored)
gold = pairs_from({s: truth_of[s] for s in scored})

tp = len(pred & gold)
precision = tp / len(pred) if pred else 0.0
recall = tp / len(gold) if gold else 0.0
f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

print(f"scored SKUs:      {len(scored):,}")
print(f"predicted pairs:  {len(pred):,}")
print(f"true pairs:       {len(gold):,}")
print(f"\nprecision: {precision:.3f}")
print(f"recall:    {recall:.3f}")
print(f"f1:        {f1:.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Where it went wrong
# MAGIC
# MAGIC Both error types are worth reading rather than just counting. A false merge shows
# MAGIC which distinction the model failed to make; a miss shows which duplicates retrieval
# MAGIC never surfaced — those two have completely different fixes, and the counts alone
# MAGIC cannot tell them apart.

# COMMAND ----------

names = {r["sku_key"]: (r["merchant_id"], r["name_clean"])
         for r in base.select("sku_key", "merchant_id", "name_clean").collect()}

def show(pairs, title, limit=15):
    print(f"\n{title} ({len(pairs):,})")
    for a, b in sorted(pairs)[:limit]:
        ma, na = names.get(a, ("?", "?"))
        mb, nb = names.get(b, ("?", "?"))
        print(f"  {ma:16s} {na[:44]:44s}")
        print(f"  {mb:16s} {nb[:44]:44s}")
        print()

show(pred - gold, "false merges — different products put together")
show(gold - pred, "misses — same product left apart")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT method, count(*) AS rows, count(DISTINCT sku_key) AS skus
# MAGIC FROM govpipe.curated.entity_matches
# MAGIC GROUP BY method
# MAGIC ORDER BY rows DESC