# `summarise_fills` — metric reference

`summarise_fills` collapses the per-bin output of `run_strategy` / `run_trade_list`
into **one row per order** (`order_id`). This document explains every column it
produces, grouped by purpose.

Throughout, `sign = +1` for buys and `sign = -1` for sells.

---

## Identifiers (carried through, one value per order)

| Column | Meaning |
|---|---|
| `order_id` | The grouping key (the trade-list row index). |
| `trade_list` | The bucket (`small_buys`, `small_sells`, …) — carried via `extra_cols` if present. |
| `security`, `date`, `qcode` | Which contract, day, and curve. |
| `side` | `buy` / `sell` — drives the sign in slippage / IS. |
| `order_quantity` | **Total** quantity to execute over the day (the order size passed in). |

---

## Counts (bins)

| Column | Formula | Meaning |
|---|---|---|
| `n_bins` | `len()` | Total bins in the day. |
| `n_fillable` | `sum(data_available)` | Bins with all four market fields present (could fill). |
| `n_filled` | `sum(filled)` | Bins that actually filled (`q_filled > 0`). |

The gap `n_bins - n_fillable` is how many bins were structurally dead
(zero-volume / missing-data) — exactly the bins naive TWAP wastes allocation on.

---

## Quantities & volume

| Column | Formula | Meaning |
|---|---|---|
| `total_volume` | `sum(volume)` | Total market volume across the day's bins. |
| `total_requested` | `sum(q_requested)` | What the strategy *asked* to trade (for TWAP this ≈ `order_quantity`). |
| `total_filled` | `sum(q_filled)` | What actually filled (after the 2×volume cap). |
| `unfilled_qty` | `order_quantity - total_filled` | Leftover vs the **order** (note: vs `order_quantity`, not `total_requested`). |

---

## Headline performance metrics

| Column | Formula | Meaning |
|---|---|---|
| **`fill_rate`** | `total_filled / order_quantity` | Fraction of the order that got done. `< 1` means the cap / illiquidity blocked you. |
| **`participation_overall`** | `total_filled / total_volume` | Your share of total market volume — how aggressive you were. |
| **`avg_fill_price`** | `filled_notional / total_filled` | Notional-weighted average price you actually transacted at. |

---

## Prices / benchmarks

| Column | Meaning |
|---|---|
| `filled_notional` | `Σ q_filled × fill_price` — total currency transacted (cost for buys / proceeds for sells). |
| `arrival_price` | **Open of the first bin** — the pre-trade decision / benchmark price. (The bin VWAP is an average *over* the bin, already polluted by intra-bin trading, so the open is the cleaner arrival mark. Falls back to the first bin VWAP only if no `open` column is carried through.) |
| `terminal_price` | **Last** non-null bin VWAP — used to mark the unfilled remainder. |

---

## Cost & slippage ("how good was execution")

| Column | Formula | Meaning |
|---|---|---|
| `total_cost` | `Σ slippage_factor × spread × q_filled` | Spread cost paid vs each bin's **own VWAP** (always ≥ 0). Pure execution friction — excludes market drift. |
| **`exec_slippage_bps`** | `sign × (avg_fill_price - arrival_price) / arrival_price × 1e4` | Filled-only slippage vs the arrival price, in bps. **Signed so positive = worse** (bought above / sold below arrival). Includes both spread *and* intraday price drift. |

---

## Implementation shortfall (all-in, includes opportunity cost)

| Column | Formula | Meaning |
|---|---|---|
| `is_currency` | `sign × (realised - paper)` | Total shortfall in currency. |
| **`is_bps`** | `is_currency / (arrival_price × order_quantity) × 1e4` | Same, in bps of the order's arrival-price value. |

where

```
paper    = arrival_price × order_quantity                     # cost of an instant fill at arrival
realised = filled_notional + unfilled_qty × terminal_price    # what you paid + unfilled marked at close
```

So `is_bps` charges you for **both** what you executed *and* the opportunity cost
of what you failed to fill (marked at the terminal price). **Positive =
underperformance** vs filling the whole order instantly at the arrival price.

---

## How the three "quality" numbers differ (important)

- **`total_cost`** — friction only (spread vs each bin's VWAP). Always ≥ 0.
  *"What the spread cost me."*
- **`exec_slippage_bps`** — friction **+ price drift over the day**, but only on
  the part that filled. Can be negative if the market moved in your favour.
- **`is_bps`** — exec slippage **+ opportunity cost of the unfilled quantity**.
  This is the one that punishes a low `fill_rate`, so it's the right top-line
  score when comparing strategies — TWAP's under-filling on illiquid / short days
  shows up *here* even though `total_cost` looks fine.

**Quick read of an order:** `fill_rate` tells you *how much* got done, `is_bps`
tells you *how expensively* (all-in), and `total_cost` vs `exec_slippage_bps`
tells you how much of the damage was spread vs market movement.
