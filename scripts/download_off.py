"""OFF 子集下载脚本（D2）。

数据源：Hugging Face 镜像 openfoodfacts/product-database 的 food.parquet（~7.7GB）。

两条路的实测结论（2026-07-21，本地代理环境）：
- duckdb httpfs 远程扫描 + 列裁剪：❌ 远程 range 请求过代理后有效吞吐 ~8KB/s，
  3.5 小时无产出。散碎 range 读在代理链路上不可用。
- aria2c 多连接整文件下载（~2.5-5MB/s）+ 本地 duckdb 过滤：✅ 稳定路径，采用。

三个模式：
  --check   探测：远程 schema 列数、总行数、抽样参数预估（只读 footer，代理下可行）
  --fetch   下载：aria2c 8 连接断点续传，落 data/food_full.parquet
  --sample  抽样：本地过滤 hash(code) % mod == 0，落 data/off_subset.parquet

抽样用 hash 取模而非 USING SAMPLE：结果可复现（同 mod 永远同一批行），
重跑、他人复现、脏化脚本引用同一子集都不依赖随机种子。

用法：
  python scripts/download_off.py --check
  python scripts/download_off.py --fetch
  python scripts/download_off.py --sample --mod 100   # ~1% ≈ 4.6 万条
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

import duckdb

# revision 必须 pin 死：该数据集每日重导出两次，main 指针下的文件天天换，
# 不 pin 则断点续传跨更新点必然长度不匹配，且子集不可复现（2026-07-22 实测踩坑）
REVISION = "49219b1a55a0ef471daedfbee223f31a32f97dbc"  # Database updated 2026-07-20
REMOTE = "hf://datasets/openfoodfacts/product-database@" + REVISION + "/food.parquet"
URL = f"https://huggingface.co/datasets/openfoodfacts/product-database/resolve/{REVISION}/food.parquet"
ROOT = Path(__file__).resolve().parent.parent
FULL_DEFAULT = ROOT / "data" / "food_full.parquet"
OUT_DEFAULT = ROOT / "data" / "off_subset.parquet"

# 只留这些列：§1.1 七项检查 + raw 表字段映射所需。
# ingredients_text 故意不留（体积大，且不在七项检查内）；
# 若 D3 决定拼接 description，从 food_full.parquet 再取即可。
COLUMNS = [
    "code",                        # barcode（检查6）
    "product_name",                # 多语言 struct 列表（检查1、2）
    "generic_name",                # raw_description 来源（多语言）
    "brands",                      # 逗号多值（检查3）
    "brands_tags",                 # 规范化品牌 tag
    "owner",                       # parquet 镜像无 brand_owner，owner 为最接近字段
    "categories",                  # 自由文本（检查5）
    "categories_tags",             # 层级展开 tag 列表 = 品类真值（CSV 侧叫 categories_hierarchy）
    "quantity",                    # 规格自由文本（检查4）
    "product_quantity",            # 官方解析值 = unit_normalize 真值
    "product_quantity_unit",       # 官方解析单位
    "labels_tags",
    "countries_tags",
    "lang",                        # 主语言（检查2）
    "completeness",                # 抽样分层用
    "data_quality_warnings_tags",  # 检查6 条码质量线索在这里（如 conflict-with-upc-12）
    "data_quality_errors_tags",
    "obsolete",
    "popularity_key",              # feed 合成时按热门度分层备用
    "scans_n",
]


def connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("SET enable_progress_bar=false")
    return con


def check(con: duckdb.DuckDBPyConnection, mod: int) -> None:
    names = {r[0] for r in con.execute(f"DESCRIBE SELECT * FROM read_parquet('{REMOTE}')").fetchall()}
    total = con.execute(f"SELECT count(*) FROM read_parquet('{REMOTE}')").fetchone()[0]
    print(f"remote columns : {len(names)}")
    print(f"total rows     : {total:,}")
    print(f"mod={mod} -> est. sample rows: ~{total // mod:,}")
    missing = [c for c in COLUMNS if c not in names]
    if missing:
        print(f"!! COLUMNS 里以下列远端不存在: {missing}", file=sys.stderr)
        sys.exit(1)
    print("all requested columns present")


def fetch(full: Path) -> None:
    full.parent.mkdir(parents=True, exist_ok=True)
    # --max-tries=0：代理链路会偶发 SSL 握手失败与 Xet CDN 分段 403，
    # 都是瞬时错误，无限重试 + 断点续传比整体失败重跑便宜得多
    cmd = [
        "aria2c", "-x", "8", "-s", "8", "-k", "4M", "-c",
        "--max-tries=0", "--retry-wait=3", "--timeout=60",
        "--console-log-level=warn", "--summary-interval=120",
        "-d", str(full.parent), "-o", full.name, URL,
    ]
    # 代理从环境变量继承（aria2 认 all_proxy/http_proxy）；代理配置不进代码库
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        sys.exit(f"aria2c 退出码 {proc.returncode}（-c 断点续传，重跑即续传）")
    print(f"fetched {full} ({full.stat().st_size / 1e9:.2f} GB)")


def sample(con: duckdb.DuckDBPyConnection, mod: int, full: Path, out: Path) -> None:
    if not full.exists():
        sys.exit(f"{full} 不存在，先跑 --fetch")
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = ", ".join(COLUMNS)
    t0 = time.time()
    con.execute(
        f"""
        COPY (
            SELECT {cols}
            FROM read_parquet('{full}')
            WHERE code IS NOT NULL AND hash(code) % {mod} = 0
        ) TO '{out}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )
    n = con.execute(f"SELECT count(*) FROM read_parquet('{out}')").fetchone()[0]
    print(f"wrote {n:,} rows -> {out} ({out.stat().st_size / 1e6:.1f} MB) in {time.time() - t0:.0f}s")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="远程探测，不落盘")
    ap.add_argument("--fetch", action="store_true", help="aria2c 下载整文件")
    ap.add_argument("--sample", action="store_true", help="本地抽样并写子集 parquet")
    ap.add_argument("--mod", type=int, default=100, help="hash 取模基数，100≈1%%")
    ap.add_argument("--full", type=Path, default=FULL_DEFAULT)
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    args = ap.parse_args()
    if not (args.check or args.fetch or args.sample):
        ap.error("需要 --check / --fetch / --sample 至少其一")
    if args.check:
        check(connect(), args.mod)
    if args.fetch:
        fetch(args.full)
    if args.sample:
        sample(connect(), args.mod, args.full, args.out)


if __name__ == "__main__":
    main()
