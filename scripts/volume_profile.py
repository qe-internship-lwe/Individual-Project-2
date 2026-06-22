"""
volume_profile.py
=================

Average intraday volume profile per qcode.

For each (qcode, day) the front-month contract's 5-minute bins are turned into a
volume *proportion* (bin volume / that day's total volume). Those daily profiles
are then averaged bin-by-bin across all trading days for the qcode, so the result
for each qcode sums to ~1.0 across the trading day.

Note: ``BINNED_DATA`` already contains exactly one security per (qcode, date) -
the front-month contract - so no contract selection is needed.

Outputs
-------
* data/processed/volume_profile_by_qcode.csv  - tidy (qcode, bin, mean_proportion)
* reports/volume_profile_by_qcode.png         - faceted grid, one panel per qcode
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl

RAW = Path("data/raw/binned_data.csv")
OUT_CSV = Path("data/processed/volume_profile_by_qcode.csv")
OUT_PNG = Path("reports/volume_profile_by_qcode.png")
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
OUT_PNG.parent.mkdir(parents=True, exist_ok=True)


def compute_profiles() -> pl.DataFrame:
    """Return tidy (qcode, bin_start_time, minutes, mean_proportion)."""
    lf = pl.scan_csv(RAW).select(
        "qcode", "publication_date", "bin_start_time", "volume"
    )

    # daily total volume per (qcode, day)
    day_tot = lf.group_by("qcode", "publication_date").agg(
        pl.col("volume").sum().alias("day_volume")
    )

    # number of trading days per qcode (denominator for the average)
    n_days = lf.select("qcode", "publication_date").unique().group_by("qcode").agg(
        pl.len().alias("n_days")
    )

    # per-bin daily proportion, then average across days (missing bins -> 0,
    # so we sum proportions and divide by the qcode's total trading days).
    prof = (
        lf.join(day_tot, on=["qcode", "publication_date"])
        .filter(pl.col("day_volume") > 0)
        .with_columns(
            (pl.col("volume") / pl.col("day_volume")).alias("proportion")
        )
        .group_by("qcode", "bin_start_time")
        .agg(pl.col("proportion").sum().alias("sum_proportion"))
        .join(n_days, on="qcode")
        .with_columns(
            (pl.col("sum_proportion") / pl.col("n_days")).alias("mean_proportion")
        )
    )

    # numeric minutes-since-midnight for plotting/sorting
    prof = prof.with_columns(
        (
            pl.col("bin_start_time").str.slice(0, 2).cast(pl.Int32) * 60
            + pl.col("bin_start_time").str.slice(3, 2).cast(pl.Int32)
        ).alias("minutes")
    )

    return prof.select(
        "qcode", "bin_start_time", "minutes", "mean_proportion"
    ).sort("qcode", "minutes").collect()


def plot_profiles(prof: pl.DataFrame, desc: dict[str, str]) -> None:
    qcodes = prof["qcode"].unique().sort().to_list()
    n = len(qcodes)
    ncols = 5
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(ncols * 3.4, nrows * 2.3), squeeze=False
    )

    for i, q in enumerate(qcodes):
        ax = axes[i // ncols][i % ncols]
        sub = prof.filter(pl.col("qcode") == q).sort("minutes")
        ax.bar(
            sub["minutes"], sub["mean_proportion"] * 100,
            width=5, align="edge", color="#1f5fa8", edgecolor="none",
        )
        title = f"{q}" + (f"  {desc[q]}" if q in desc else "")
        ax.set_title(title[:34], fontsize=7)
        ax.tick_params(labelsize=6)
        # x ticks every 3 hours
        lo, hi = sub["minutes"].min(), sub["minutes"].max()
        ticks = list(range(((lo // 180) + 1) * 180, hi + 1, 180))
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t // 60:02d}:{t % 60:02d}" for t in ticks])
        ax.margins(x=0.01)
        ax.grid(True, alpha=0.2, lw=0.4)

    # hide unused axes
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.suptitle(
        "Average intraday volume profile by qcode\n"
        "(% of front-month daily volume per 5-min bin, averaged over all days)",
        fontsize=11,
    )
    fig.supxlabel("Time of day (bin start, exchange-local)", fontsize=9)
    fig.supylabel("Mean share of daily volume (%)", fontsize=9)
    fig.tight_layout(rect=(0.01, 0.01, 1, 0.96))
    fig.savefig(OUT_PNG, dpi=130)
    print(f"saved {OUT_PNG}")


def main() -> None:
    prof = compute_profiles()
    prof.write_csv(OUT_CSV)
    print(f"saved {OUT_CSV}  ({prof.height} rows, {prof['qcode'].n_unique()} qcodes)")

    # qcode -> short description for panel titles
    qm = pl.read_csv("data/raw/qcode_mapping.csv")
    desc = dict(zip(qm["qcode"].to_list(), qm["description"].to_list()))

    # sanity: each qcode's profile should sum to ~1
    chk = prof.group_by("qcode").agg(pl.col("mean_proportion").sum().alias("s"))
    print("profile sums (should be ~1.0): "
          f"min={chk['s'].min():.3f} max={chk['s'].max():.3f}")

    plot_profiles(prof, desc)


if __name__ == "__main__":
    main()
