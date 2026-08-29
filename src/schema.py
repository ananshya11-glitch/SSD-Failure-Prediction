"""
SMART column schema.

ASSUMED, NOT VERIFIED. The real Alibaba parquet has 105 columns:
3 metadata (disk_id, ds, model) + 102 SMART = 51 SMART IDs x {n_i, r_i}.
The IDs below are a plausible SSD-relevant set chosen to hit that count.

TO REPLACE WITH THE REAL SCHEMA:
    import polars as pl
    print(pl.scan_parquet(PARQUET).collect_schema().names())
then overwrite SMART_IDS (or SMART_COLS directly) here. Nothing else in
the codebase should hardcode column names.
"""

from .config import META_COLS

# 51 SMART IDs. Standard ATA attributes plus the vendor-specific 17x/18x
# block that SSDs use for wear, program/erase failures and reserved blocks.
SMART_IDS = [
    1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13,
    170, 171, 172, 173, 174, 175, 176, 177, 178, 179,
    180, 181, 182, 183, 184,
    187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199,
    200, 201, 202, 206,
    225, 226, 227, 228,
    232, 233, 234,
]

NORM_COLS = [f"n_{i}" for i in SMART_IDS]
RAW_COLS = [f"r_{i}" for i in SMART_IDS]

# Interleaved n_i, r_i — matches the readme's field ordering.
SMART_COLS = [c for i in SMART_IDS for c in (f"n_{i}", f"r_{i}")]

ALL_COLS = META_COLS + SMART_COLS

assert len(SMART_IDS) == 51, len(SMART_IDS)
assert len(ALL_COLS) == 105, len(ALL_COLS)


# ---------------------------------------------------------------
# Per-vendor attribute availability
#
# Vendors populate different SMART ID subsets. This divergence IS the
# covariate shift the paper measures, so the synthetic generator must
# reproduce it. Values below are invented; replace from
# reports/attribute_availability.csv once profiling has run.
# ---------------------------------------------------------------

_COMMON = [1, 5, 9, 12, 187, 190, 194, 195, 197, 198, 199, 241, 242]
_COMMON = [i for i in _COMMON if i in SMART_IDS]

VENDOR_SMART_IDS = {
    "A": _COMMON + [3, 4, 7, 170, 171, 172, 173, 174, 175, 183, 184,
                    188, 189, 192, 193, 196, 200, 225, 226, 227, 228],
    "B": _COMMON + [2, 8, 10, 11, 13, 176, 177, 178, 179, 180, 181,
                    182, 191, 201, 202, 206, 232, 233, 234],
    "C": _COMMON + [3, 4, 170, 171, 173, 177, 179, 181, 184, 188,
                    192, 202, 225, 232, 233, 234],
}

VENDOR_SMART_IDS = {
    v: sorted(set(i for i in ids if i in SMART_IDS))
    for v, ids in VENDOR_SMART_IDS.items()
}


def vendor_cols(vendor: str) -> list[str]:
    """SMART columns a vendor populates. Others are NaN throughout."""
    ids = VENDOR_SMART_IDS[vendor]
    return [c for i in ids for c in (f"n_{i}", f"r_{i}")]
