# Trade Execution

A back-testing harness for **execution strategies** on front-month futures. It
takes a list of orders, simulates how each one would have filled if executed
across a trading day, and reports how good that execution was (fill rate, cost,
implementation shortfall).

The goal is to have a common framework where different execution strategies can
be plugged in and measured on the same orders and the same market data. They fall
into three groups:

- **Deployable** (no future information): **TWAP**; a historical-volume VWAP
  (`vwap_static`); a cost-minimising liquidity/spread allocator (`liq_spr_static`);
  and three **adaptive** schedules that react to volume as the day unfolds — an AR(1)
  volume-surprise tilt (`vwap_adaptive`), a K-factor Kalman filter on the latent
  daily volume factor (`vwap_factor`), and a per-pair OLS forecast
  (`adaptive_volume_ols`).
- **Lookahead benchmarks** (allowed to see the realised day — *not* tradeable):
  `omniscient_vwap`, `omniscient_liq_spr`, and the full `omniscient` — the bars the
  deployable strategies are measured against.
- A realised-volume `vwap_schedule` is also kept as a hindsight reference.

Per-bin spread/liquidity profiles come from either `build_liq_spread_curves`
(static, all-history) or `build_rolling_curves` (trailing-window, lookahead-free).

The reusable logic lives in [`execution.py`](execution.py); the
[`trade_execution.ipynb`](trade_execution.ipynb) notebook is the driver that
loads data, runs strategies, and inspects the results.

---

## The big picture

```
 orders (trade_list)          market data (binned_data)
        │                              │
        └──────────────┬───────────────┘
                       ▼
                 a strategy          ← decides how much to trade in each 5-min bin
                       ▼
                 fill model          ← simulates what actually fills, and at what price
                       ▼
              per-bin fill rows
                       ▼
              summarise_fills        ← one row per order: fill rate, cost, shortfall
```

The day is split into **5-minute bins**. A strategy decides *how much of the
order to send in each bin*; the fill model then decides *how much of that
actually trades and at what price*, given the bin's liquidity. Collapsing all
the bins back to one row per order gives the execution-quality scorecard.

---

## Inputs

### Orders — the trade list

Each order is one **(security, date, side, quantity)** to execute over a single
day. The orders come from a `TRADE_LIST` table that holds **six orders per
security-day**, one for each combination of size bucket × side:

| | buy | sell |
|---|---|---|
| **small** | `small_buys` | `small_sells` |
| **medium** | `medium_buys` | `medium_sells` |
| **large** | `large_buys` | `large_sells` |

You typically back-test **one bucket at a time** (e.g. all `medium_buys`), so
results are comparable within a size/side regime.

### Market data — the bins

For each instrument and day, the market is described in 5-minute bins. Each bin
carries the microstructure the fill model needs:

- **volume** — contracts traded in the bin (the liquidity available),
- **VWAP** — volume-weighted average price (the reference fill price),
- **TWA ask / TWA bid** — time-weighted average quotes (their difference is the
  spread),
- **open** — the bin's opening price (used as the pre-trade benchmark).

Some bins are structurally dead — zero volume or missing data. Nothing can fill
in those bins, which is exactly the friction a good strategy has to work around.

---

## The fill model

For a single bin, given how much the strategy asked to trade:

1. **You can't take the whole bin.** Fills are capped at **twice the bin's
   volume** — you can trade at most `2 × volume` no matter how much you ask for.
   Anything above the cap is left unfilled.
2. **Trading moves the price against you.** The more of the bin's volume you
   take (your *participation rate*), the worse your price. This is modelled as a
   **square-root slippage** function: small participation barely moves you off
   VWAP; pushing all the way to the cap costs you roughly **half the spread**.
3. **Direction sets the sign.** Buys pay *up* from VWAP (VWAP + slippage);
   sells get hit *down* (VWAP − slippage).
4. **No data, no fill.** If a bin is missing any of volume / VWAP / ask / bid,
   or has zero volume, it produces no fill and the requested quantity is left
   over.

Unfilled quantity is **not** automatically retried — it's reported so a strategy
can decide whether to re-queue it. The baseline TWAP does not, so under-filling
in illiquid names shows up directly in its score.

---

## Strategies

A strategy answers one question: **given a day's bins and a total quantity, how
much should be requested in each bin?** Everything else (the fill model, the
metrics) is shared, so swapping strategies is the only variable when comparing.

All strategies round their per-bin requests to **whole lots** — you can't trade
fractional contracts.

Strategies fall into two families that differ in how they treat an unfilled
slice:

- **Static** schedules (`twap`, `vwap_static`, `liq_spr_static`, and the
  `omniscient*` benchmarks) commit a fixed plan up front that sums exactly to the
  order quantity. If a bin under-fills (the `2 × volume` cap or a dead bin), that
  quantity is simply lost — they do **not** chase it.
- **Dynamic** schedules (`vwap_adaptive`, `vwap_factor`, `adaptive_volume_ols`)
  re-plan as the day unfolds and **carry forward**: each
  bin requests a *proportion of the lots still to fill*, and `remaining` is
  decremented by what **actually filled** (not what was requested). A slice the
  cap or a dead bin failed to fill rolls into the next bin instead of being lost,
  so the requested total can exceed the order quantity while the realised fills
  target the full order. (When you add a strategy, decide which family it belongs
  to before wiring this in.)

### TWAP (Time-Weighted Average Price) — the baseline

Splits the order **evenly across every bin** in the day — each bin requests
`quantity / number_of_bins`, regardless of that bin's liquidity.

This is deliberately naive. It allocates to dead bins too, so that quantity
simply never fills and is lost. That under-filling on illiquid or short days is
the weakness a smarter, liquidity-aware schedule should improve on — and the
reason TWAP is the baseline everything else is measured against.

### VWAP (Volume-Weighted Average Price) — the volume-aware baseline

Splits the order across bins **in proportion to each bin's share of the day's
traded volume**: a bin that carried 3% of the day's volume is asked to trade 3%
of the order (`quantity × volume / total_volume`). This tracks the intraday
volume profile — heavy at the open/close, thin midday — instead of treating
every bin equally.

The key difference from TWAP: it allocates **nothing** to zero-volume bins, so
it stops wasting quantity on bins that can't fill. On illiquid names that lifts
the fill rate substantially (in the demo, the illiquid contract goes from ~78%
filled under TWAP to ~100% under VWAP).

A full fill is **not** guaranteed, though — VWAP only spreads the order over bins
that actually traded, so it still under-fills when the order is larger than the
day can absorb (`quantity > 2 × total daily volume`, since each bin caps at
`2 × volume`), on zero-volume days (it falls back to an even split that fills
nothing), or in the rare bins that have volume but missing price data. The demo
orders are sized well within each instrument's daily liquidity, which is why they
happen to fill completely.

Because the schedule is built from the `volume` of the bins for one
`(security, date)` — and the data carries exactly one front-month contract per
`(qcode, date)` — **each instrument gets its own schedule from its own volume
curve**, and the same instrument gets a different schedule on different days.

This is a *realised*-volume VWAP — it uses the actual volume of the day being
executed (**perfect hindsight** of the profile), so it is a reference point, not a
deployable strategy. The deployable version is **`vwap_static`** below, which
weights by the *historical-average* volume curve instead.

### VWAP (static) — historical-volume, deployable

The same volume-proportional split as VWAP, but weighted by the
**historical-average** volume per `(security, bin)` — the mean volume of each
5-minute bin across all days in the data — instead of the day being executed. That
removes the perfect-hindsight assumption, so `vwap_static` is the deployable VWAP
baseline (`vwap_schedule` stays only as the hindsight reference). The static
per-instrument curves are precomputed once by `build_liq_spread_curves`.

### liq_spr_static — cost-minimising liquidity/spread allocation

Chooses the per-bin quantities that **minimise the fill model's total slippage
cost**, using static historical **spread** and **liquidity** curves per
`(security, bin)` (the same curves as `vwap_static`). The spread curve is the
average `twa_ask − twa_bid` for each bin across all days; the liquidity curve is the
average volume.

Because the slippage cost is convex in the quantity sent to a bin, the optimum sets
the **marginal cost equal across all bins** — a single Lagrange multiplier `μ`,
solved by bisection so the per-bin quantities sum to the order. Intuitively it pours
quantity into tight-spread, liquid bins and starves wide-spread, thin ones; when
every bin has the same spread it reduces *exactly* to volume-weighting (VWAP). It
uses no information from the day being traded, so it is fully deployable. The
derivation is in [`execution.py`](execution.py).

### Curves: static vs trailing (rolling)

`vwap_static` and `liq_spr_static` take a `curves` argument that can come from
either builder, and any strategy accepts either (same
`{(qcode, date): {bin: (spread, volume)}}` shape):

- **`build_liq_spread_curves`** — averages each bin over **all** days. Simple, but
  its "no same-day lookahead" still leaks the future profile (it includes days
  *after* the order).
- **`build_rolling_curves`** — estimates each `(qcode, bin)` from only that qcode's
  **previous N trading days** (default 22), strictly before the order date — the
  genuinely lookahead-free version used in the full back-test. A qcode's first day
  has no history, so those orders fall back to TWAP.

### Adaptive VWAP — `vwap_adaptive`

Works in **deseasonalised log space**, anchored on the trailing-window **mean of
log volume** `L_k` — so the baseline `exp(L_k)` is the *geometric* mean volume (the
mean of the logs, **not** the log of the arithmetic mean), the same baseline
convention as `vwap_factor`. As the day unfolds it **tilts the remaining schedule
toward bins that recent volume suggests will be busier**, exploiting the one
genuinely predictable intraday signal we found: **volume clustering** — the
deseasonalised log-volume residual is persistent (AR(1) ρ ≈ 0.5). It uses no price
information; it is strictly causal (only volume from *completed* bins), rounds to
whole lots, and **carries forward** unfilled lots (see the static/dynamic note).

Causal receding-horizon form: at each bin it measures the recent *surprise* — the
**mean of the per-bin log-deviations** `log(volume_j) − L_j` over the last `recent`
traded bins — and re-forecasts **every** remaining bin with that surprise decaying
by horizon, exponentiated back onto the geometric baseline:
`v̂_f = exp(L_f) · exp(ρ^h · surprise)`. Far bins (`ρ^h → 0`) revert to `exp(L_f)`,
so it degrades to the static geometric profile when there's no signal. `ρ` is a
**fixed structural constant** (default 0.5, the measured pooled value), not fitted
to the executed day. (Note: plain `vwap_static` still uses the **arithmetic** mean
`V_k` — correct for a volume-proportional VWAP; only the adaptive/factor log-space
forecasts use the geometric baseline.)

### Factor-Kalman VWAP — `vwap_factor`

The same idea as `vwap_adaptive` — forecast the rest of the day's volume from what's
traded so far — but driven by a **K-factor state-space model** instead of a single
AR(1) surprise. The day's intraday log-volume deviations are modelled as
`M_k = λ_k·F + u_k`, where `F` is a small set of **latent daily volume factors**
(1 = the dominant "busy-day" level factor) with a standard-normal prior, `λ_k` the
per-bin loadings, and `u_k` idiosyncratic noise. The loadings `λ` and idiosyncratic
variances `σ_u²` are estimated **once per order** by maximum-likelihood factor
analysis on the trailing window (`build_factor_curves`, lookahead-free — day `t`
uses days `t-LOOKBACK_DAYS … t-1`, never `t` itself).

As each bin completes, a **sequential Kalman update** revises the belief `(μ, P)`
about `F` from that bin's log-volume surprise, then the remaining bins are forecast
as `volume_hat_m = exp(L_m + λ_m·μ)`. Each forecast is then **confidence-adjusted**
by its predictive log-volume variance `r_m² = λ_mᵀ P λ_m + σ²_{u,m}` —
`volume_tilde_m = volume_hat_m · exp(−r_m²/4)` (`certainty_equivalent_volume`) —
because execution cost is convex in volume (√-impact), so uncertain bins are trusted
less; the factor-uncertainty term shrinks as the day is observed (`P` falls), so the
penalty relaxes. The schedule participates in proportion to `volume_tilde` over the
lots still to fill (toggle with `ce_adjust`). It's strictly causal (bin `k` uses only
volume `< k`), carries forward, and rounds to whole lots. For one factor the update
is exactly the scalar Kalman recursion; for K>1 the forecast is invariant to factor
rotation, so loadings need no rotation. **`N_FACTORS`** and **`LOOKBACK_DAYS`** are
notebook parameters (the latter also sets the minimum history, so orders in a
qcode's first `LOOKBACK_DAYS` trading days are dropped). Falls back to TWAP when an
order has no factor model.

### OLS Adaptive VWAP — `adaptive_volume_ols`

A lighter-weight cousin of `vwap_factor`: same log-deviation set-up, but the forecast
comes from a **per-pair OLS** rather than a factor Kalman filter. For each future bin
`m` it uses the backward regression of `M_m` on the **most recent observed** bin `M_k`
(the Markov / "Option A" assumption `M_m | M_1..M_k = M_m | M_k`), with the slope and
variances read from the trailing-window cross-moments (`build_ols_curves`):

- slope `β_{m|k} = ρ·σ_m/σ_k` → forecast `M̂_m = β_{m|k}·M_k`;
- prediction variance `r_m² = σ²_{m|k} + Var(β̂)·M_k²` (residual `σ²_{m|k}=σ_m²(1−ρ²)`
  **plus** a slope-estimation term, so a large/extrapolated surprise is trusted less).

Each remaining bin is forecast `volume_hat_m = exp(L_m + M̂_m)`, **confidence-adjusted**
`volume_tilde_m = volume_hat_m · exp(−r_m²/4)` (`certainty_equivalent_volume`, toggle
`ce_adjust`), and the schedule participates in proportion to `volume_tilde` over the
lots still to fill. Conditions on the latest *completed* traded bin (re-planned every
bin), so it's strictly causal; before anything trades it uses the unconditional
baseline (`M̂=0`, `r²=σ_m²`). Dynamic, carries forward, whole lots. Uses the global
`LOOKBACK_DAYS` window; falls back to TWAP with no model. Being Markov it conditions on
only the latest bin, so it tends to under-perform `vwap_factor` where the level factor
dominates (long-lag volume correlations don't decay), but it's a simpler, transparent
baseline.

---

## Lookahead benchmarks (not deployable)

These are allowed to see the **realised day** and trade against it — the unbeatable
bars the deployable strategies are measured against, *not* strategies you could run
live. All three respect the `2 × volume` fill cap, so they fill 100% whenever the
order fits the day's capacity.

### omniscient_vwap — liquidity-only benchmark

Tracks the **realised** volume profile (perfect foresight of *volume* only): trades
each bin in proportion to that day's actual volume. The classic VWAP benchmark
execution is normally scored against — like `vwap_static`, but on the real day's
volume instead of the historical curve.

### omniscient_liq_spr — liquidity + spread benchmark

Minimises the fill model's **spread cost** on the realised spread and volume (no
price drift) — the cost-min `liq_spr` allocation with perfect foresight of the day's
spread/liquidity.

### omniscient — full lower bound (liquidity + spread + drift)

The strongest: also uses realised **price drift**, so it minimises the *actual*
implementation shortfall (buy where the price turns out lowest, sell where highest).
It exploits everything possible — the theoretical floor. Because drift isn't
realistically predictable, this is an *unattainable* bound; the two above are the
*achievable* benchmarks.

> More strategies will be documented here as they are added.

---

## Outputs — what gets measured

Each order collapses to a single row of execution metrics. At a glance:

- **`fill_rate`** — fraction of the order that actually got done. Below 1 means
  the cap / illiquidity blocked you.
- **`participation_overall`** — your share of total market volume over the day
  (how aggressive the execution was).
- **`avg_fill_price`** — the volume-weighted price you actually transacted at.
- **`total_cost`** / **`total_realised_impact`** — total realised market impact ($), the `Σ x·s·q_filled` cost from the fill model; always ≥ 0 (two names for the same quantity).
- **`exec_slippage_bps`** — slippage of the filled portion vs the pre-trade
  benchmark price, in basis points, **signed so positive = worse**. Includes
  both spread and intraday price drift.
- **`is_bps`** — **implementation shortfall**: the all-in score. Like
  `exec_slippage_bps` but *also* charges opportunity cost on whatever didn't
  fill. This is the metric that punishes a low fill rate, so it's the natural
  top-line number when comparing strategies.

The benchmark (the "arrival price" these are measured against) is the **open of
the first bin** — the decision price before any trading happens. The unfilled
remainder's opportunity cost is marked at the **terminal price** — the last bin's
VWAP.

`is_bps` also comes pre-split into three additive pieces — `is_slippage_bps`
(total **realised impact**, `Σ x·s·q_filled`, in bps), `is_drift_bps` (realised
price drift on the filled part), and `is_opportunity_bps` (the unfilled remainder
vs arrival) — which sum to `is_bps`. (`decompose_is_stats` labels the first piece
`realised_impact`.)
`decompose_is(summary)` averages them across orders so you can see where the
shortfall actually came from (and `decompose_is_stats` gives per-component
mean / median / variance, per strategy).

There is also a separate, **drift-free** score — the **order-impact** (market-impact
shortfall): the realised spread cost on the filled lots **plus** an opportunity
cost that crosses **half the closing spread** on the shortfall
(`0.5 · terminal_spread · unfilled`), i.e. the cost of cleaning up the unfilled
lots at the close. It excludes price drift on purpose (we aren't forecasting
drift). `asset_impact(summary, by="qcode")` aggregates it per **qcode** as
notional-weighted bps (`Σ impact / Σ notional × 1e4`, with realised + opportunity
adding to the total) — pass `by="asset_class"` to pool bond/commodity/equity
futures instead — and `order_impact_stats(summary, label)` gives a one-row-per-strategy
summary — `$` totals plus bps both **averaged across assets** (`realised_bps` /
`opportunity_bps` / `total_impact_bps`, each qcode equal-weighted) and
**notional-weighted** across all orders (the `*_bps_nw` columns, which track the
`$` totals). The two can differ markedly when opportunity concentrates in
illiquid, low-fill names.

> **Units — read before pooling `$` across assets.** The raw `$` columns from
> `summarise_fills` are in **quote-point × contract** units *and* in each instrument's
> **native currency** — and the universe spans 50 contracts with different multipliers
> and 7 quote currencies (`docs/qcode_mapping.md`). So any metric that *sums* `$` across
> instruments — `asset_impact(by="tier"/"asset_class")`, the `*_bps_nw` / `$` columns of
> `order_impact_stats` — is meaningless until the pieces share a unit. Put every summary
> into common USD notional with two per-order constant scales, **in order**:
> 1. **`apply_multiplier(summary, mults)`** — ×contract multiplier
>    (`data/raw/contract_multipliers.csv`, e.g. ES = `$50`/pt, bond futures = `1000`):
>    quote-points → native currency.
> 2. **`to_usd(summary, fx, ccy_map)`** — ×daily FX rate (`data/raw/fx_rates.csv`,
>    as-of by date; USD passes through at 1.0): native currency → USD.
>
> Both scale numerator and denominator together, so all **bps ratios** (per-order,
> per-`qcode`, and the paired t-tests) are **unchanged**; only cross-asset sums move —
> a high-multiplier / stronger-currency contract now carries proportionally more
> notional weight in a pool. The back-test notebook applies both to every summary
> before building the tier tables. (`by="qcode"` is single-unit, so pooling there is
> already valid — the scales only re-weight *across* qcodes.)

Finally there's a **shape** diagnostic — **participation dispersion** — for how
flat a schedule's intraday participation is: the volume-weighted volatility
`σ_p = sqrt(Σ v_k (p_k − P)²)` of per-bin participation `p_k = q_filled_k/volume_k`
around the overall rate `P`, and its coefficient of variation `CV_p = σ_p/P`. A
perfect volume-tracker has `p_k = P` everywhere so `σ_p ≈ 0` — `omniscient_vwap`
is the sanity check. `participation_dispersion(fills)` gives it per order and
`participation_stats(disp, label)` the mean/median CV per strategy.

**Pairwise significance tests.** To ask whether two strategies genuinely differ
(not just on a pooled average), `paired_impact_ttest(summaries, tiers, bucket=…)`
and `paired_participation_ttest(fills, tiers, bucket=…)` run a **paired t-test**
(`scipy.stats.ttest_rel`) for every strategy pair within each impact tier: orders
are matched on `order_id` (all strategies share the trade list), and the per-order
metric is compared — the per-order **total-impact bps**
(`order_impact / impact_notional × 1e4`, the pairable analogue of the pooled
`asset_impact` bps; `metric=` also allows `realised_bps` / `opportunity_bps`) or the
per-order **participation CV**. Each draws one `N×N` matrix per tier
(`N = len(summaries)`, never hardcoded), colour-coded by p-value (green `p<0.01`,
orange `0.01≤p<0.05`, grey = not significant) and annotated with `t` / `p` / paired
`n`; `t>0` means the row strategy's metric is higher than the column's. Both are thin
wrappers over the general `paired_ttest_panels(by_strat, strats, groups, …)`
(matplotlib/scipy are imported lazily, so they're only needed when these are called).

> **Full column-by-column reference:** see
> [`docs/execution_metrics.md`](docs/execution_metrics.md). It documents every
> field `summarise_fills` produces and, importantly, explains how the three
> "quality" numbers (`total_cost`, `exec_slippage_bps`, `is_bps`) differ.

*(Cross-strategy **ranking** is still out of scope — we're building strategies, not
yet crowning one — but the paired t-tests above are the first cut at pairwise
comparison: which differences are real, tier by tier.)*

---

## Where things live

| File | Role |
|---|---|
| [`execution.py`](execution.py) | All reusable logic — fill model, strategies, runners, metrics. |
| [`trade_execution.ipynb`](trade_execution.ipynb) | Driver: loads data, runs strategies, inspects results. |
| [`docs/execution_metrics.md`](docs/execution_metrics.md) | Detailed reference for every output metric. |
| [`docs/qcode_mapping.md`](docs/qcode_mapping.md) | Reference for the instrument dimension table. |

---

## Keeping this in sync

This README is the **high-level overview**. When the execution logic changes,
update it alongside the docstrings — in particular:

- a **new strategy** → add it under [Strategies](#strategies),
- a **change to the fill model** → update [The fill model](#the-fill-model),
- a **new or changed metric** → update both [Outputs](#outputs--what-gets-measured)
  here and the detailed table in [`docs/execution_metrics.md`](docs/execution_metrics.md).
