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
| `total_realised_impact` | `Σ x · s · q_filled` (= `total_cost`) | **Total realised market impact** ($) — the cost from the fill model's `x × s` term summed over the order's filled bins. Identical to `total_cost`; surfaced under this name, and it is the `realised_impact` component of the IS decomposition once put in bps. |
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

---

## Decomposing `is_bps` (where the shortfall comes from)

`summarise_fills` also splits `is_bps` into three **additive** components — they
sum to `is_bps` exactly, for every order:

| Column | Currency formula (÷ `paper` × 1e4 for bps) | Meaning |
|---|---|---|
| `is_slippage_bps` (a.k.a. **`realised_impact`**) | `total_cost` | **Total realised impact** = spread cost (`x·s·q_filled`) paid on the filled lots; always ≥ 0. Surfaced in `$` as `total_realised_impact`; `decompose_is_stats` labels this component `realised_impact`. |
| `is_drift_bps` | `sign × (Σ q_filled×vwap − arrival_price × total_filled)` | **Realised price drift** from the arrival price to the bin VWAPs you actually filled at — the *timing* of the filled portion. Signed; negative if the market moved in your favour. |
| `is_opportunity_bps` | `sign × unfilled_qty × (terminal_price − arrival_price)` | **Opportunity cost** of the unfilled remainder, marked at the terminal price vs arrival. Exactly `0` when `fill_rate = 1`. |

```
is_slippage_bps + is_drift_bps + is_opportunity_bps  ==  is_bps
```

Relationship to the other columns: `exec_slippage_bps` ≈ slippage + drift, but it
is measured *per filled unit* (÷ filled notional), whereas these components are
÷ the whole order's `paper` notional so they add up to `is_bps`. The opportunity
term is exactly the part `exec_slippage_bps` ignores (it looks only at the filled
portion).

**Aggregating — `decompose_is(summary, by=None)`:** averages the components across
orders (optionally grouped by `qcode`, `trade_list`, …) alongside `n_orders` and
`fill_rate`. Because the mean is linear, the component means still sum to the mean
`is_bps`, so you can read off "of the −11 bps mean, X bps was slippage, Y drift,
Z opportunity." Orders with an undefined `is_bps` (e.g. a non-trading day with no
arrival price) are dropped.

---

## Order impact — the market-impact shortfall (no drift)

A separate, drift-free score: **what execution actually cost** (spread paid on the
filled lots) **plus** the cost of cleaning up the shortfall by crossing the spread
at the close. It deliberately excludes the price-drift term of `is_bps` — we are
not trying to forecast drift, so it is not charged here.

### Per-order columns (added by `summarise_fills`)

| Column | Formula | Meaning |
|---|---|---|
| `total_realised_impact` | `Σ x·s·q_filled` | Realised market impact ($) — spread cost on the filled lots (same as `total_cost`). |
| `terminal_spread` | last quoting bin's `twa_ask − twa_bid` | Closing spread, used to price the clean-up. |
| `opportunity_impact` | `0.5 · terminal_spread · unfilled_qty` | **Opportunity cost** ($): cross **half** the closing spread (`CROSS_SPREAD_FRACTION = 0.5`) on the unfilled remainder — the cost of buying the shortfall at end of day. `0` when filled 100%. |
| `impact_notional` | `Σ VWAP_bin·q_filled + VWAP_last·unfilled_qty` | Denominator: filled lots at their bin VWAP plus the unfilled remainder marked at the terminal VWAP. |
| `order_impact` | `total_realised_impact + opportunity_impact` | Total order impact ($). |

So the headline, in bps:

```
Total Order Impact (bps) = ( Σ x·s·q_filled + 0.5·s_last·q_unfilled )
                         / ( Σ VWAP_bin·q_filled + VWAP_last·q_unfilled ) × 1e4
```

### Aggregating — `asset_impact(summary)` and `order_impact_stats(summary, label)`

- **`asset_impact`** — one row per **qcode**. Sums `total_realised_impact`,
  `opportunity_impact` and `impact_notional` across that asset's orders, then forms
  **notional-weighted** bps (`Σ impact / Σ notional × 1e4`). `realised_bps` and
  `opportunity_bps` share the one denominator, so they add to `total_impact_bps`
  exactly. Orders with null/non-positive `impact_notional` (a fully dead day — no
  terminal price to mark against, same convention as `is_bps`) are dropped.
- **`order_impact_stats`** — one row per **strategy** (stack several to compare).
  The `*_$` columns are summed over all orders; the bps columns are the **mean
  across assets** of the `asset_impact` bps (reconciling assets by a simple mean,
  for now). Columns: `fill_rate`, `realised_impact_$`, `opportunity_impact_$`,
  `total_impact_$`, `mean_$_per_order`, `realised_bps`, `opportunity_bps`,
  `total_impact_bps`.

This is distinct from `decompose_is` / `is_bps`, which is the full drift-inclusive
shortfall vs the arrival price and is still recorded unchanged.
