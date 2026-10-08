# Databricks notebook source
# MAGIC %md
# MAGIC # 02 · Generate merchant feeds
# MAGIC
# MAGIC Open Food Facts is a single product database with no notion of a merchant,
# MAGIC while the pipeline expects feeds uploaded independently by several merchants.
# MAGIC This step distributes the subset across six simulated merchants, each injecting
# MAGIC a different class of data quality problem.
# MAGIC
# MAGIC Two design points:
# MAGIC - **Noise injection is deterministic.** Every random decision derives from a hash of
# MAGIC   `seed + merchant + barcode + purpose`, so it is independent of execution order —
# MAGIC   any rerun produces byte-identical feeds
# MAGIC - **Ground truth is exported alongside.** The fields OFF has already normalized
# MAGIC   (parsed quantity, hierarchical category tags) are stored separately as the answer
# MAGIC   key for evaluation. Governance stages cannot read it; only evaluation scripts can

# COMMAND ----------

CATALOG = "govpipe"
VOLUME = f"/Volumes/{CATALOG}/raw/landing"
SEED = 20260725

import hashlib, random, re, csv, json, os

# COMMAND ----------

# MAGIC %md
# MAGIC ## Merchant profiles
# MAGIC
# MAGIC Each merchant is dirty in its own way, and every way maps to a capability
# MAGIC the governance pipeline has to demonstrate:
# MAGIC
# MAGIC | Merchant | Format | Noise injected | Capability it exercises |
# MAGIC |---|---|---|---|
# MAGIC | m01_grocermart | CSV | Abbreviated names, quantity appended to name, quantity field dropped | Name cleanup, quantity extraction from the name |
# MAGIC | m02_quickshop | JSONL | Brand dropped 30%, case scrambled | Brand identification from the name alone |
# MAGIC | m03_liquorbarn | CSV | Own category vocabulary, quantity compacted to `6x2000ml` | Category mapping, quantity parsing |
# MAGIC | m04_cornerstore | JSONL | Barcode missing 50%, mistyped 4% | Matching without a reliable barcode |
# MAGIC | m05_megamart | CSV | Relatively clean, 5% duplicate listings | Within-merchant deduplication |
# MAGIC | m06_worldbites | JSONL | Prefers non-English names, localized units | Multilingual name handling |

# COMMAND ----------

MERCHANTS = [
    {"id": "m01_grocermart", "format": "csv", "n_skus": 9000, "price_factor": 1.00,
     "field_map": {"sku_id": "item_no", "name": "item_name", "brand": "brand", "category": "dept",
                   "quantity": "size", "barcode": "upc", "price": "unit_price", "description": "notes"},
     "noise": {"abbrev_name": 0.35, "spec_into_name": 0.40, "drop_quantity": 0.30}},
    {"id": "m02_quickshop", "format": "jsonl", "n_skus": 8000, "price_factor": 1.06,
     "field_map": {"sku_id": "sid", "name": "title", "brand": "brand_name", "category": "cat",
                   "quantity": "pack_size", "barcode": "ean", "price": "price", "description": "desc"},
     "noise": {"drop_brand": 0.30, "case_chaos": 0.50}},
    {"id": "m03_liquorbarn", "format": "csv", "n_skus": 7500, "price_factor": 1.12,
     "field_map": {"sku_id": "sku", "name": "product", "brand": "label", "category": "aisle",
                   "quantity": "vol", "barcode": "barcode", "price": "rrp", "description": "memo"},
     "noise": {"own_category_vocab": 1.0, "compact_spec": 0.80}},
    {"id": "m04_cornerstore", "format": "jsonl", "n_skus": 7000, "price_factor": 0.95,
     "field_map": {"sku_id": "id", "name": "name", "brand": "brand", "category": "category",
                   "quantity": "quantity", "barcode": "code", "price": "amount", "description": "description"},
     "noise": {"barcode_drop": 0.50, "barcode_typo": 0.04, "case_chaos": 0.25}},
    {"id": "m05_megamart", "format": "csv", "n_skus": 8000, "price_factor": 1.03,
     "field_map": {"sku_id": "sku_code", "name": "display_name", "brand": "brand_name",
                   "category": "category_name", "quantity": "net_content", "barcode": "gtin",
                   "price": "retail_price", "description": "long_desc"},
     "noise": {"dup_listing": 0.05}},
    {"id": "m06_worldbites", "format": "jsonl", "n_skus": 6500, "price_factor": 1.08,
     "field_map": {"sku_id": "ref", "name": "nom", "brand": "marque", "category": "rayon",
                   "quantity": "contenance", "barcode": "cab", "price": "prix", "description": "infos"},
     "noise": {"prefer_foreign_name": 0.70, "localize_unit": 0.40}},
]

OWN_VOCAB = {
    "en:beverages": "Drinks & Bar", "en:alcoholic-beverages": "Booze Corner",
    "en:snacks": "Munchies", "en:sweet-snacks": "Sweet Treats", "en:dairies": "Milk & Co",
    "en:plant-based-foods-and-beverages": "Green Shelf",
    "en:meats-and-their-products": "Butcher", "en:frozen-foods": "Freezer",
    "en:condiments": "Sauce Rack", "en:breakfasts": "Morning Goods",
    "fallback": "General Aisle",
}
STRATIFY = {"truth_ratio": 0.7}
FLOORS = {"quantity_truth_products": 8000, "category_truth_products": 10000}

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read the subset, pick a primary-language name
# MAGIC
# MAGIC Product names in OFF are an array of language-tagged structs, so a primary language
# MAGIC has to be resolved first: main → the record's own language → English → first available.
# MAGIC Profiling measured this chain to cover 94% of rows.

# COMMAND ----------

import pyspark.sql.functions as F

def _first_text(lang_pred: str) -> str:
    inner = ("product_name" if lang_pred is None
             else f"filter(product_name, x -> x.lang {lang_pred})")
    return f"try_element_at(transform({inner}, x -> x.text), 1)"

NAME_EXPR = f"""
    coalesce(
        {_first_text("= 'main'")},
        {_first_text("= lang")},
        {_first_text("= 'en'")},
        {_first_text(None)})
"""
off = spark.read.parquet(f"{VOLUME}/off_subset.parquet")
rows_df = off.selectExpr(
    "code",
    f"{NAME_EXPR} as name_main",
    f"{NAME_EXPR.replace('product_name', 'generic_name')} as desc_main",
    "brands", "brands_tags", "categories", "categories_tags",
    "quantity", "product_quantity", "product_quantity_unit",
    "filter(transform(product_name, x -> concat(x.lang, '|', x.text)), "
    "  x -> x not like 'main|%' and x is not null) as name_langs",
)
rows = [r.asDict() for r in rows_df.collect()]
print("subset rows:", len(rows))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Noise rules
# MAGIC
# MAGIC Each rule simulates one class of real merchant data problem.
# MAGIC All randomness is hash-derived, hence reproducible.

# COMMAND ----------

VOWELS = "aeiouAEIOU"
UNIT_RE = re.compile(r"^\s*(\d+(?:[.,]\d+)?)\s*(l|cl|dl)\s*$", re.I)

def rng_for(merchant, code, purpose):
    h = hashlib.sha256(f"{SEED}:{merchant}:{code}:{purpose}".encode()).digest()
    return random.Random(int.from_bytes(h[:8], "big"))

def hit(merchant, code, purpose, p):
    return rng_for(merchant, code, purpose).random() < p

def abbrev_name(name, rng):
    out = []
    for w in name.split():
        mode = rng.random()
        if len(w) > 6 and mode < 0.4:
            out.append(w[:4])                                   # truncate
        elif len(w) > 4 and mode < 0.75:
            out.append(w[0] + "".join(c for c in w[1:] if c not in VOWELS))   # drop vowels
        else:
            out.append(w)
    return " ".join(out)

def case_chaos(name, rng):
    return " ".join(w.upper() if (r := rng.random()) < 0.3 else w.lower() if r < 0.6 else w
                    for w in name.split())

def compact_spec(q, rng):
    s = re.sub(r"\s*[x×]\s*", "x", q.strip())
    s = re.sub(r"(\d)\s+([a-zA-Z])", r"\1\2", s)
    m = UNIT_RE.match(q)
    if m and rng.random() < 0.5:                                # "2 l" -> "2000ml"
        val = float(m.group(1).replace(",", "."))
        return f"{int(val * {'l': 1000, 'dl': 100, 'cl': 10}[m.group(2).lower()])}ml"
    return s

def localize_unit(q, rng):
    reps = [(r"\bg\b", "gr"), (r"\bl\b", "litre"), (r"\bml\b", "millilitre")]
    pat, rep = reps[rng.randrange(len(reps))]
    return re.sub(pat, rep, q)

def barcode_typo(code, rng):
    i = rng.randrange(len(code))
    return code[:i] + str((int(code[i]) + rng.randrange(1, 9)) % 10) + code[i + 1:]

print("noise rules ready")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stratified sampling
# MAGIC
# MAGIC 70% is drawn from rows carrying at least one ground-truth field — this is what keeps
# MAGIC the evaluation set large enough. The other 30% comes from rows with none; those are
# MAGIC the test material for the quarantine path and the human review queue. Without the
# MAGIC stratification, the rows evaluation depends on get diluted and accuracy figures
# MAGIC lose statistical meaning.
# MAGIC
# MAGIC The two pools will not add up to the subset size. `truth_pool` additionally requires
# MAGIC a non-empty name, so rows that have ground truth but no name resolvable through the
# MAGIC language fallback chain fall out of both pools. Dropping them is deliberate: a
# MAGIC nameless product gets quarantined by the very first governance stage, so it would
# MAGIC occupy a slot in the evaluation set while producing no signal to evaluate. When the
# MAGIC quarantine path itself needs exercising, `missing_pool` supplies plenty of material.
# MAGIC

# COMMAND ----------

def has_truth(r):
    q = r["product_quantity"]
    return bool(q and str(q).strip()) or bool(r["categories_tags"])

truth_pool = [r for r in rows if has_truth(r) and r["name_main"] and r["name_main"].strip()]
missing_pool = [r for r in rows if not has_truth(r)]
print(f"truth_pool={len(truth_pool):,}  missing_pool={len(missing_pool):,}")

def pick(pool, k, merchant, tier):
    ranked = sorted(pool, key=lambda r: hashlib.sha256(
        f"{SEED}:{merchant}:{tier}:{r['code']}".encode()).digest())
    return ranked[:k]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate
# MAGIC
# MAGIC Apply the noise rules merchant by merchant, write the files to the volume,
# MAGIC and collect ground truth along the way.

# COMMAND ----------

def synth_price(r, factor, rng):
    tags = r["categories_tags"] or []
    base = (1.5 + (int(hashlib.sha256(tags[0].encode()).hexdigest()[:8], 16) % 1050) / 100
            if tags else 3.5)
    return round(base * factor * rng.uniform(0.9, 1.1), 2)

def build_record(r, m):
    mid, code, noise = m["id"], r["code"], m.get("noise", {})
    name, brand, category = r["name_main"], r["brands"], r["categories"]
    quantity, barcode, desc = r["quantity"], code, r["desc_main"]

    if name and noise.get("prefer_foreign_name") and hit(mid, code, "foreign", noise["prefer_foreign_name"]):
        foreign = [t for t in (r["name_langs"] or []) if not t.startswith("en|")]
        if foreign:
            name = foreign[0].split("|", 1)[1]
    if name and noise.get("abbrev_name") and hit(mid, code, "abbrev", noise["abbrev_name"]):
        name = abbrev_name(name, rng_for(mid, code, "abbrev_r"))
    if name and quantity and noise.get("spec_into_name") and hit(mid, code, "specname", noise["spec_into_name"]):
        name = f"{name} {compact_spec(quantity, rng_for(mid, code, 'specname_r'))}"
    if name and noise.get("case_chaos") and hit(mid, code, "case", noise["case_chaos"]):
        name = case_chaos(name, rng_for(mid, code, "case_r"))
    if noise.get("drop_brand") and hit(mid, code, "dropbrand", noise["drop_brand"]):
        brand = None
    if noise.get("drop_quantity") and hit(mid, code, "dropqty", noise["drop_quantity"]):
        quantity = None
    if quantity and noise.get("compact_spec") and hit(mid, code, "compact", noise["compact_spec"]):
        quantity = compact_spec(quantity, rng_for(mid, code, "compact_r"))
    if quantity and noise.get("localize_unit") and hit(mid, code, "localize", noise["localize_unit"]):
        quantity = localize_unit(quantity, rng_for(mid, code, "localize_r"))
    if noise.get("own_category_vocab") and hit(mid, code, "vocab", noise["own_category_vocab"]):
        tags = r["categories_tags"] or []
        category = OWN_VOCAB.get(tags[0], OWN_VOCAB["fallback"]) if tags else OWN_VOCAB["fallback"]
    if noise.get("barcode_drop") and hit(mid, code, "bcdrop", noise["barcode_drop"]):
        barcode = None
    elif noise.get("barcode_typo") and hit(mid, code, "bctypo", noise["barcode_typo"]):
        barcode = barcode_typo(barcode, rng_for(mid, code, "bctypo_r"))

    # 12-digit SKU id: at 8 digits the birthday-collision expectation over 9,000
    # products is ~0.4, and one collision did show up in practice
    sku_id = f"{int(hashlib.sha256(f'{mid}:{code}'.encode()).hexdigest()[:12], 16) % 10**12:012d}"
    return {"sku_id": sku_id, "name": name, "brand": brand, "category": category,
            "quantity": quantity, "barcode": barcode,
            "price": synth_price(r, m.get("price_factor", 1.0), rng_for(mid, code, "price")),
            "description": desc}

os.makedirs(f"{VOLUME}/feeds", exist_ok=True)
os.makedirs(f"{VOLUME}/eval", exist_ok=True)
gt_rows, stats = [], {}

for m in MERCHANTS:
    mid, n = m["id"], m["n_skus"]
    n_truth = round(n * STRATIFY["truth_ratio"])
    chosen = ([(r, "truth") for r in pick(truth_pool, n_truth, mid, "truth")]
              + [(r, "missing") for r in pick(missing_pool, n - n_truth, mid, "missing")])
    records = []
    for r, tier in chosen:
        rec = build_record(r, m)
        emits = [(rec, False)]
        p_dup = m.get("noise", {}).get("dup_listing", 0)
        if p_dup and hit(mid, r["code"], "dup", p_dup):
            dup = dict(rec, sku_id=rec["sku_id"] + "B")
            if dup["name"]:
                dup["name"] = case_chaos(dup["name"], rng_for(mid, r["code"], "dup_case"))
            dup["price"] = round(rec["price"] * rng_for(mid, r["code"], "dup_price").uniform(0.97, 1.03), 2)
            emits.append((dup, True))
        for r2, is_dup in emits:
            records.append(r2)
            gt_rows.append({"merchant_id": mid, "merchant_sku_id": r2["sku_id"],
                            "off_code": r["code"], "tier": tier, "is_dup_listing": is_dup,
                            "true_name": r["name_main"], "true_brands": r["brands"],
                            "true_brands_tags": r["brands_tags"],
                            "true_categories_tags": r["categories_tags"],
                            "true_quantity_value": r["product_quantity"],
                            "true_quantity_unit": r["product_quantity_unit"]})

    ids = [r["sku_id"] for r in records]
    assert len(ids) == len(set(ids)), f"{mid}: sku_id collision"
    records.sort(key=lambda r: r["sku_id"])
    fmap = m["field_map"]
    mapped = [{fmap[k]: v for k, v in r.items()} for r in records]
    path = f"{VOLUME}/feeds/{mid}.{m['format']}"
    if m["format"] == "csv":
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(fmap.values()))
            w.writeheader(); w.writerows(mapped)
    else:
        with open(path, "w") as f:
            for r in mapped:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    stats[mid] = len(records)
    print(f"{mid:16s} {len(records):>6,} rows -> {path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Persist ground truth, then check the floors
# MAGIC
# MAGIC Ground truth lands in a table only evaluation reads — governance stages must never
# MAGIC touch it. The floor check guards against the evaluation set silently shrinking
# MAGIC when the subset is swapped out later.

# COMMAND ----------

gt_df = spark.createDataFrame(gt_rows)
gt_df.write.mode("overwrite").saveAsTable(f"{CATALOG}.ops.feed_ground_truth")

q_products = len({r["off_code"] for r in gt_rows
                  if r["true_quantity_value"] and str(r["true_quantity_value"]).strip()})
c_products = len({r["off_code"] for r in gt_rows if r["true_categories_tags"]})
print(f"ground truth rows: {len(gt_rows):,}")
print(f"products with quantity truth: {q_products:,} (floor {FLOORS['quantity_truth_products']:,})")
print(f"products with category truth: {c_products:,} (floor {FLOORS['category_truth_products']:,})")
assert q_products >= FLOORS["quantity_truth_products"], \
    "not enough quantity ground truth — the evaluation set would starve"
assert c_products >= FLOORS["category_truth_products"], "not enough category ground truth"
print("\nfloor checks passed. total feed rows:", sum(stats.values()))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Spot-check the injected noise

# COMMAND ----------

def read_csv_feed(mid):
    with open(f"{VOLUME}/feeds/{mid}.csv", newline="") as f:
        return list(csv.DictReader(f))

def read_jsonl_feed(mid):
    with open(f"{VOLUME}/feeds/{mid}.jsonl") as f:
        return [json.loads(line) for line in f]

# m04 · barcode dropped at the configured rate
m04 = read_jsonl_feed("m04_cornerstore")
bc_null = sum(1 for r in m04 if r["code"] is None) / len(m04)
assert 0.47 <= bc_null <= 0.53, f"m04 barcode drop rate off target: {bc_null:.1%}"

# m05 · duplicate listings, exact — the count is hash-determined, not sampled
m05_cfg = next(m for m in MERCHANTS if m["id"] == "m05_megamart")
expected_dups = stats["m05_megamart"] - m05_cfg["n_skus"]
n_dups = sum(1 for r in read_csv_feed("m05_megamart") if r["sku_code"].endswith("B"))
assert n_dups == expected_dups, f"m05 duplicate listings: {n_dups}, expected {expected_dups}"

# m02 · brand missing well above the source rate, since 30% is dropped on top of it
m02 = read_jsonl_feed("m02_quickshop")
brand_null = sum(1 for r in m02 if not r["brand_name"]) / len(m02)
assert brand_null > 0.45, f"m02 brand missing rate too low: {brand_null:.1%}"

# m01 · quantity actually reaches the name, and accented characters survive the write
m01 = read_csv_feed("m01_grocermart")
in_name = sum(1 for r in m01 if r["item_name"]
              and re.search(r"\d+\s*(ml|l|cl|dl|kg|g|oz)\b", r["item_name"]))
assert in_name > len(m01) * 0.08, f"quantity rarely folded into m01 names: {in_name}"
assert any(ord(c) > 127 for r in m01 for c in (r["item_name"] or "")), \
    "no non-ASCII characters in m01 names — the encoding is broken"

print(f"m04 barcode missing    {bc_null:>6.1%}   (configured 50%)")
print(f"m05 duplicate rows     {n_dups:>6,}   (exact match required)")
print(f"m02 brand missing      {brand_null:>6.1%}   (36.7% already absent in source, 30% dropped on top)")
print(f"m01 quantity in name   {in_name / len(m01):>6.1%}")
print("\nall noise assertions passed")

# COMMAND ----------

for r in read_csv_feed("m01_grocermart")[:6]:
    print(f"  {r['item_name'][:52]:<52} | size={str(r['size'] or ''):<10} | brand={r['brand'] or ''}")