"""
execution.py
============

Helper functions for simulating order execution against the 5-minute
``BINNED_DATA`` table (OHLC + microstructure bins for front-month futures).

The functions here are intended to be imported from ``trade_execution.ipynb``,
which carries the bulk of the schedule-building and back-testing logic::

    from execution import (
        simulate_bin_fill,        # scalar reference model (one bin)
        twap_schedule,            # baseline strategy: even split across bins
        vwap_schedule,            # baseline strategy: split by each bin's volume share
        run_strategy,             # run one order (security, date, side, qty)
        run_trade_list,           # run many orders (the trade list)
        summarise_fills,          # order-level metrics: fill rate, IS, cost
    )

The layering is:

* :func:`simulate_bin_fill` - scalar, single-bin reference implementation of
  the fill model (no dependencies beyond the stdlib).
* a *strategy* (e.g. :func:`twap_schedule`, :func:`vwap_schedule`) - maps a
  day's bins + a total quantity to a per-bin requested quantity.
* :func:`run_strategy` - applies a strategy to one (security, date) order and
  simulates the fills **vectorised** (polars), returning one row per bin.
* :func:`run_trade_list` - loops :func:`run_strategy` over many orders.
* :func:`summarise_fills` - collapses the per-bin output to one row per order
  with fill rate, average fill price, implementation shortfall, and cost.

Execution model (per bin)
-------------------------
For a single 5-minute bin, given the quantity your schedule submits
(``q_requested``) and the bin's market data:

    q_filled  = min(q_requested, 2 * VOLUME)        # you can take at most 2x the bin volume
    p         = q_filled / VOLUME                    # realised participation rate, in [0, 2]
    x         = 0.1 + 0.2828 * sqrt(p)               # slippage factor (half-spread multiplier)
    s         = TWA_ASK - TWA_BID                     # bin spread

    buy : fill_price = VWAP + x * s
    sell: fill_price = VWAP - x * s

Notes / conventions implemented below:

* The cap is ``2 * VOLUME``; any unfilled portion (``q_requested - q_filled``)
  is **not** carried forward automatically - the caller decides whether/when to
  retry it. It is returned as ``q_unfilled`` so the schedule can re-queue it.
* If any of ``VOLUME``, ``VWAP``, ``TWA_ASK``, ``TWA_BID`` is missing
  (``None`` or ``NaN``), **no fill** is generated for that bin, regardless of
  how much was requested.
* If ``VOLUME == 0`` the cap is 0, so nothing can fill (``p`` is undefined and
  reported as 0.0, ``fill_price`` is ``NaN``). This is the genuinely-illiquid
  / no-trade bin case seen in the data.
* At full participation ``p = 1`` you pay ``x = 0.383`` of the spread; at the
  ``p = 2`` cap you pay ``x = 0.4999`` (~half the spread - the coefficient
  ``0.2828`` is a rounded ``0.4 / sqrt(2)``, so it lands just under 0.5).
"""

from __future__ import annotations

import math
from typing import Callable, NamedTuple, Optional, Sequence

import numpy as np
import polars as pl

__all__ = [
    "BinFill",
    "simulate_bin_fill",
    "twap_schedule",
    "vwap_schedule",
    "build_liq_spread_curves",
    "vwap_static",
    "liq_spr_static",
    "omniscient_vwap",
    "omniscient_liq_spr",
    "omniscient",
    "run_strategy",
    "run_trade_list",
    "summarise_fills",
    "decompose_is",
    "decompose_is_stats",
    "SLIPPAGE_BASE",
    "SLIPPAGE_COEF",
    "FILL_CAP_MULTIPLE",
    # column-name constants (the expected BINNED_DATA schema)
    "SECURITY_COL",
    "DATE_COL",
    "QCODE_COL",
    "BIN_TIME_COL",
    "VOLUME_COL",
    "VWAP_COL",
    "OPEN_COL",
    "ASK_COL",
    "BID_COL",
]

# --- model constants (kept named so they are documented and easy to tweak) ---
SLIPPAGE_BASE: float = 0.1       # x = SLIPPAGE_BASE + SLIPPAGE_COEF * sqrt(p)
SLIPPAGE_COEF: float = 0.2828    # ~= 0.4 / sqrt(2): gives x = 0.5 at the p = 2 cap
FILL_CAP_MULTIPLE: float = 2.0   # q_filled <= FILL_CAP_MULTIPLE * VOLUME


class BinFill(NamedTuple):
    """Result of simulating a single bin fill.

    Attributes
    ----------
    side : str
        ``"buy"`` or ``"sell"`` (normalised to lowercase).
    q_requested : float
        Quantity the schedule asked to trade in this bin.
    q_filled : float
        Quantity actually filled, ``min(q_requested, 2 * VOLUME)`` (0 if the
        bin has no usable market data or zero volume).
    q_unfilled : float
        ``q_requested - q_filled`` - the leftover the caller may re-queue.
    participation_rate : float
        ``p = q_filled / VOLUME`` (0.0 when nothing filled).
    slippage_factor : float
        ``x = 0.1 + 0.2828 * sqrt(p)`` (``NaN`` when nothing filled).
    spread : float
        Bin spread ``s = TWA_ASK - TWA_BID`` (``NaN`` when data is missing).
    fill_price : float
        Execution price, ``VWAP +/- x * s`` (``NaN`` when nothing filled).
    notional : float
        ``q_filled * fill_price`` - unsigned traded value (0.0 when nothing
        filled). The caller applies the sign convention for P&L.
    filled : bool
        ``True`` iff ``q_filled > 0``.
    data_available : bool
        ``True`` iff all four required market-data fields were present.
    """

    side: str
    q_requested: float
    q_filled: float
    q_unfilled: float
    participation_rate: float
    slippage_factor: float
    spread: float
    fill_price: float
    notional: float
    filled: bool
    data_available: bool


def _is_missing(value: Optional[float]) -> bool:
    """True if ``value`` is ``None`` or ``NaN`` (treated as "no market data")."""
    if value is None:
        return True
    try:
        return math.isnan(value)
    except TypeError:
        return False


def simulate_bin_fill(
    q_requested: float,
    side: str,
    *,
    volume: Optional[float],
    vwap: Optional[float],
    twa_ask: Optional[float],
    twa_bid: Optional[float],
) -> BinFill:
    """Simulate filling an order against a single 5-minute bin.

    Implements the execution model described in the module docstring: the fill
    is capped at twice the bin's traded volume, the realised participation rate
    drives a square-root slippage factor, and the fill price is VWAP plus
    (buy) or minus (sell) that fraction of the bin spread.

    Parameters
    ----------
    q_requested : float
        Quantity your schedule wants to trade in this bin. Must be >= 0;
        a value <= 0 yields a zero fill. Sign is *not* used to infer
        direction - pass the magnitude and set ``side`` instead.
    side : str
        Trade direction: ``"buy"`` or ``"sell"`` (case-insensitive). Buys pay
        up (``+x*s``), sells get hit down (``-x*s``).
    volume : float or None
        ``VOLUME`` of the bin (number of contracts traded). Keyword-only.
    vwap : float or None
        ``VWAP`` of the bin - the reference fill price. Keyword-only.
    twa_ask : float or None
        ``TWA_ASK`` - time-weighted average ask, for the spread. Keyword-only.
    twa_bid : float or None
        ``TWA_BID`` - time-weighted average bid, for the spread. Keyword-only.

    Returns
    -------
    BinFill
        A named tuple with the fill quantity, leftover, participation rate,
        slippage factor, spread, fill price, notional, and status flags. See
        :class:`BinFill` for field-by-field documentation.

    Raises
    ------
    ValueError
        If ``side`` is not ``"buy"`` or ``"sell"``.

    Notes
    -----
    * If any of ``volume``, ``vwap``, ``twa_ask`` or ``twa_bid`` is ``None`` /
      ``NaN``, the bin produces **no fill** (``q_filled == 0``,
      ``fill_price == NaN``) and ``data_available`` is ``False``.
    * ``volume == 0`` also produces no fill (the cap ``2 * volume`` is 0).
    * To model a TWAP/VWAP schedule, call this once per bin in the execution
      window and feed any ``q_unfilled`` into your own retry logic.

    Examples
    --------
    A liquid buy bin, requesting less than 2x volume (spread = 0.25)::

        >>> f = simulate_bin_fill(
        ...     1000, "buy",
        ...     volume=5000, vwap=100.0, twa_ask=100.25, twa_bid=100.0)
        >>> f.q_filled
        1000.0
        >>> round(f.participation_rate, 3)          # 1000 / 5000
        0.2
        >>> round(f.slippage_factor, 4)             # 0.1 + 0.2828*sqrt(0.2)
        0.2265
        >>> round(f.fill_price, 5)                  # 100 + 0.2265 * 0.25
        100.05662

    Requesting more than the 2x cap - fill is capped, remainder returned::

        >>> f = simulate_bin_fill(
        ...     20000, "sell",
        ...     volume=5000, vwap=100.0, twa_ask=100.25, twa_bid=100.0)
        >>> f.q_filled, f.q_unfilled                # capped at 2 * 5000
        (10000.0, 10000.0)
        >>> round(f.slippage_factor, 4)             # p = 2 -> x ~= 0.5
        0.4999
        >>> round(f.fill_price, 5)                  # 100 - 0.4999 * 0.25
        99.87502

    A bin with missing data produces no fill::

        >>> f = simulate_bin_fill(
        ...     1000, "buy",
        ...     volume=None, vwap=100.0, twa_ask=100.1, twa_bid=99.9)
        >>> f.filled, f.q_unfilled
        (False, 1000.0)
    """
    side_norm = side.strip().lower()
    if side_norm not in ("buy", "sell"):
        raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")

    q_requested = max(float(q_requested), 0.0)

    # No usable market data -> no fill, full quantity left to re-queue.
    if _is_missing(volume) or _is_missing(vwap) or _is_missing(twa_ask) or _is_missing(twa_bid):
        return BinFill(
            side=side_norm,
            q_requested=q_requested,
            q_filled=0.0,
            q_unfilled=q_requested,
            participation_rate=0.0,
            slippage_factor=math.nan,
            spread=math.nan,
            fill_price=math.nan,
            notional=0.0,
            filled=False,
            data_available=False,
        )

    spread = twa_ask - twa_bid
    cap = FILL_CAP_MULTIPLE * volume
    q_filled = min(q_requested, cap)

    # Zero volume (cap == 0) or nothing requested -> data present but no fill.
    if q_filled <= 0:
        return BinFill(
            side=side_norm,
            q_requested=q_requested,
            q_filled=0.0,
            q_unfilled=q_requested,
            participation_rate=0.0,
            slippage_factor=math.nan,
            spread=spread,
            fill_price=math.nan,
            notional=0.0,
            filled=False,
            data_available=True,
        )

    p = q_filled / volume
    x = SLIPPAGE_BASE + SLIPPAGE_COEF * math.sqrt(p)
    fill_price = vwap + x * spread if side_norm == "buy" else vwap - x * spread

    return BinFill(
        side=side_norm,
        q_requested=q_requested,
        q_filled=q_filled,
        q_unfilled=q_requested - q_filled,
        participation_rate=p,
        slippage_factor=x,
        spread=spread,
        fill_price=fill_price,
        notional=q_filled * fill_price,
        filled=True,
        data_available=True,
    )


# ---------------------------------------------------------------------------
# Column names of the BINNED_DATA schema the strategy/runner functions expect.
# Override-free defaults; rename your columns to match before calling, or edit
# these if your normalised schema differs.
# ---------------------------------------------------------------------------
SECURITY_COL = "security"
DATE_COL = "publication_date"
QCODE_COL = "qcode"
BIN_TIME_COL = "bin_start_time"
VOLUME_COL = "volume"
VWAP_COL = "vwap"
OPEN_COL = "open"
ASK_COL = "twa_ask"
BID_COL = "twa_bid"

# A strategy maps (day's bins, total quantity) -> per-bin requested quantity.
Strategy = Callable[..., Sequence[float]]


def twap_schedule(bins: pl.DataFrame, quantity: float, **_: object) -> pl.Series:
    """Baseline TWAP schedule: split ``quantity`` evenly across every bin in
    **whole lots**.

    Time-Weighted Average Price. We cannot trade fractional lots, only round
    lots, so the total order quantity ``X`` (rounded to an integer number of
    lots) is split across the ``N`` bins of the (security, date) using a
    two-layer integer decomposition:

    * **Floor quotient (base layer).** ``B = X // N`` is the minimum integer
      number of lots every single bin is guaranteed to execute.
    * **Modulo remainder (remainder layer).** ``R = X % N`` is the leftover
      lots that do not divide evenly. By construction ``0 <= R < N``, so the
      remainder is dealt out as one extra lot to the first ``R`` bins (one
      lot each, never more). The earliest bins are favoured so the residual
      is worked off promptly rather than left to the end of the day.

    The result is ``R`` bins of ``B + 1`` lots followed by ``N - R`` bins of
    ``B`` lots, which sums to ``R*(B+1) + (N-R)*B = N*B + R = X`` exactly - the
    order is fully scheduled with no fractional lots anywhere.

    This is still the naive baseline on purpose:

    * It allocates to **every** bin in the day, including bins that cannot fill
      (zero-volume / missing-data bins). Lots assigned to those bins simply do
      not fill and are lost (no auto-retry), so the realised fill rate will be
      below 100% in illiquid names - which is exactly what we want to measure a
      smarter strategy against.

    Parameters
    ----------
    bins : pl.DataFrame
        The bins for one (security, date). Only the row count is used; the
        caller (:func:`run_strategy`) is responsible for sorting by time.
    quantity : float
        Total quantity to execute over the day (a magnitude in lots; ``side``
        is handled by the runner). Rounded to the nearest whole lot before
        splitting, since only round lots can be traded.
    **_ :
        Ignored. Present so strategies share a uniform calling convention.

    Returns
    -------
    pl.Series
        Integer-valued float series of length ``len(bins)`` summing to the
        rounded ``quantity``; each element is either ``B`` or ``B + 1`` lots.

    Examples
    --------
    >>> import polars as pl
    >>> b = pl.DataFrame({"bin_start_time": ["09:00", "09:05", "09:10", "09:15"]})
    >>> list(twap_schedule(b, 1000))          # 1000 / 4 divides evenly
    [250.0, 250.0, 250.0, 250.0]
    >>> list(twap_schedule(b, 1002))          # B=250, R=2 -> first 2 bins get +1
    [251.0, 251.0, 250.0, 250.0]
    """
    n = bins.height
    if n == 0:
        return pl.Series("q_requested", [], dtype=pl.Float64)
    total_lots = int(round(float(quantity)))      # only whole lots can be traded
    base = total_lots // n                         # floor quotient (base layer)
    remainder = total_lots % n                     # modulo remainder (0 <= R < n)
    per_bin = [base + 1 if i < remainder else base for i in range(n)]
    return pl.Series("q_requested", per_bin, dtype=pl.Float64)


def vwap_schedule(bins: pl.DataFrame, quantity: float, **_: object) -> pl.Series:
    """Baseline VWAP schedule: split ``quantity`` by each bin's volume share.

    Volume-Weighted Average Price. Each bin is requested in proportion to the
    fraction of the **day's** total traded volume that occurred in that bin, but
    in **whole lots** only (we cannot trade fractional lots). The ideal
    continuous target for bin ``k`` is::

        q_k* = X * volume[k] / sum(volume)

    where ``X`` is the order quantity rounded to whole lots. Naively rounding
    each ``q_k*`` independently would not sum back to ``X``. Instead we use
    **cumulative rounding** (a.k.a. the largest-remainder method written as a
    running total), which is self-correcting and guarantees the integer vector
    sums exactly to ``X``:

    1. Cumulative ideal float target up to bin ``k``: ``Q_k* = sum_{i<=k} q_i*``.
    2. Cumulative integer goal: ``Qbar_k = round(Q_k*)``.
    3. Requested integer lots for bin ``k``: ``q_k = Qbar_k - Qbar_{k-1}``
       (with ``Qbar_0 = 0``).

    Because ``Q_N* = X`` exactly and ``X`` is an integer, ``Qbar_N = X``, so the
    requested vector telescopes to ``X``. Any fractional dust a bin would have
    carried is pushed onto the next bin's cumulative goal rather than dropped.

    How it differs from :func:`twap_schedule`:

    * It allocates **nothing** to zero-volume bins (their share is 0), so it
      stops wasting quantity on bins that cannot fill - the main weakness of
      naive TWAP. It still requests proportionally more in heavy bins, which can
      bump into the ``2 * volume`` fill cap, but those are exactly the liquid
      bins where the cap is least binding.
    * This is a *realised*-volume VWAP: it uses the actual ``volume`` of the day
      being executed (perfect hindsight of the profile). A forecast-based VWAP
      (using the historical average profile per qcode) would be the next step.

    Parameters
    ----------
    bins : pl.DataFrame
        The bins for one (security, date); must contain the ``volume`` column.
        Only ``volume`` is used; the caller (:func:`run_strategy`) sorts by time.
    quantity : float
        Total quantity to execute over the day (a magnitude in lots; ``side`` is
        handled by the runner). Rounded to the nearest whole lot before
        splitting, since only round lots can be traded.
    **_ :
        Ignored. Present so strategies share a uniform calling convention.

    Returns
    -------
    pl.Series
        Integer-valued float series of length ``len(bins)`` summing exactly to
        the rounded ``quantity``. When the day has **no** traded volume at all
        (``sum(volume) <= 0``, e.g. a fully dead day), the proportions are
        undefined and it falls back to the integer-lot TWAP split so the order
        is still fully requested (nothing fills either way, since every bin is a
        no-trade bin).

    Examples
    --------
    >>> import polars as pl
    >>> b = pl.DataFrame({
    ...     "bin_start_time": ["09:00", "09:05", "09:10"],
    ...     "volume": [100, 0, 50],          # middle bin is a no-trade bin
    ... })
    >>> [round(x, 2) for x in vwap_schedule(b, 300)]   # shares 2/3, 0, 1/3
    [200.0, 0.0, 100.0]
    >>> list(vwap_schedule(b, 301))          # cumulative rounding keeps the sum
    [201.0, 0.0, 100.0]
    """
    n = bins.height
    if n == 0:
        return pl.Series("q_requested", [], dtype=pl.Float64)

    total_lots = int(round(float(quantity)))      # only whole lots can be traded
    vol = bins.get_column(VOLUME_COL).cast(pl.Float64).fill_null(0.0).fill_nan(0.0)
    total = vol.sum()
    if total is None or total <= 0:
        # No traded volume anywhere -> fall back to the integer-lot even split.
        return twap_schedule(bins, total_lots)

    # Cumulative rounding: round the running ideal target, then difference it so
    # the per-bin integer lots telescope back to exactly ``total_lots``.
    cum_ideal = (vol / total * total_lots).cum_sum()
    cum_goal = cum_ideal.round(0)
    per_bin = cum_goal.diff().fill_null(cum_goal.head(1))
    return per_bin.cast(pl.Float64).rename("q_requested")


# --- liq_spr_static: cost-minimising allocation over static historical curves ---
#
# The fill model charges, per bin, ``cost = slippage_factor * spread * q`` with
# ``slippage_factor = SLIPPAGE_BASE + SLIPPAGE_COEF * sqrt(q / V)``. Expanding:
#
#     C_k(q_k) = SLIPPAGE_BASE * s_k * q_k
#              + SLIPPAGE_COEF * s_k * q_k**1.5 / sqrt(V_k)
#
# We minimise total cost sum_k C_k(q_k) subject to sum_k q_k = X, q_k >= 0. Each
# C_k is convex (q**1.5 is convex), so the equal-marginal-cost (KKT) point is the
# unique global optimum. The marginal cost is
#
#     dC_k/dq_k = SLIPPAGE_BASE * s_k + MARG_COEF * s_k * sqrt(q_k / V_k) = mu
#
# with MARG_COEF = 1.5 * SLIPPAGE_COEF. Inverting for a common multiplier ``mu``:
#
#     q_k*(mu) = V_k * max(0, mu / (MARG_COEF * s_k) - THRESH_RATIO)**2
#
# where THRESH_RATIO = SLIPPAGE_BASE / MARG_COEF. The ``max(0, .)`` is the
# non-negativity clamp: a bin trades only once mu exceeds its first-lot marginal
# cost SLIPPAGE_BASE * s_k; below that, q_k = 0 (squaring a negative bracket
# would wrongly load quantity into expensive bins). Note that with equal spreads
# the bracket is constant across bins and q_k* proportional to V_k -> plain VWAP,
# so this is a spread-aware generalisation of VWAP.
MARG_COEF: float = 1.5 * SLIPPAGE_COEF          # 0.4242: slope of the sqrt term
THRESH_RATIO: float = SLIPPAGE_BASE / MARG_COEF  # 0.2357: SLIPPAGE_BASE / MARG_COEF


def build_liq_spread_curves(bins: pl.DataFrame) -> dict:
    """Precompute the static per-(security, bin) spread and liquidity curves.

    For each ``(security, bin_start_time)`` this averages, over **all days** in
    ``bins``:

    * ``spread_curve`` ``s_k`` = mean of ``twa_ask - twa_bid``;
    * ``liq_curve``    ``V_k`` = mean of ``volume``.

    Nulls are skipped by the mean (a bin missing prices on some days still gets a
    curve from the days it has). These are *static* (historical) profiles - they
    do not use the day being executed, so :func:`liq_spr_static` is lookahead-free
    (unlike the realised-volume :func:`vwap_schedule`).

    Parameters
    ----------
    bins : pl.DataFrame
        Bins to estimate the curves from (typically the full ``BINNED_DATA``),
        with the ``security``, ``bin_start_time``, ``volume``, ``twa_ask`` and
        ``twa_bid`` columns.

    Returns
    -------
    dict
        Nested lookup ``{security: {bin_start_time: (s_k, V_k)}}`` for O(1) access
        per bin inside the strategy. Pass it as
        ``strategy_params={"curves": curves}`` to :func:`run_strategy` /
        :func:`run_trade_list`.
    """
    g = (
        bins.select(SECURITY_COL, BIN_TIME_COL, VOLUME_COL, ASK_COL, BID_COL)
        .with_columns((pl.col(ASK_COL) - pl.col(BID_COL)).alias("_spread"))
        .group_by(SECURITY_COL, BIN_TIME_COL)
        .agg(
            pl.col("_spread").mean().alias("spread_curve"),
            pl.col(VOLUME_COL).cast(pl.Float64).mean().alias("liq_curve"),
        )
    )
    curves: dict = {}
    for sec, bt, s_k, v_k in g.iter_rows():
        curves.setdefault(sec, {})[bt] = (s_k, v_k)
    return curves


def vwap_static(bins: pl.DataFrame, quantity: float, *,
                curves: dict, **_: object) -> pl.Series:
    """VWAP weighted by the **static historical** volume curve (no lookahead).

    Like :func:`vwap_schedule`, this splits the order in proportion to each bin's
    volume share - but it uses the average per-(security, bin) volume ``V_k`` from
    :func:`build_liq_spread_curves` (the mean over all days) instead of the day's
    *realised* volume. That removes the perfect-hindsight assumption baked into
    :func:`vwap_schedule`, so this is the deployable VWAP baseline and the fair
    head-to-head for :func:`liq_spr_static` (both run off the same static
    ``curves``)::

        q_requested[k] = quantity * V_k / sum(V_k)

    Quantities are rounded to whole lots by the same cumulative-rounding scheme
    as :func:`vwap_schedule`, so they sum exactly to ``round(quantity)``.

    Parameters
    ----------
    bins : pl.DataFrame
        Bins for one (security, date); must carry ``security`` and
        ``bin_start_time``. Aligned to the caller's (time-sorted) row order.
    quantity : float
        Total lots to execute (rounded to whole lots).
    curves : dict
        ``{security: {bin_start_time: (s_k, V_k)}}`` from
        :func:`build_liq_spread_curves`, passed via ``strategy_params``. Only the
        ``V_k`` (volume) component is used.
    **_ :
        Ignored. Present so strategies share a uniform calling convention.

    Returns
    -------
    pl.Series
        Integer-valued float series of length ``len(bins)`` summing to
        ``round(quantity)``. When no bin has a usable historical volume, falls
        back to the integer-lot TWAP split.
    """
    n = bins.height
    if n == 0:
        return pl.Series("q_requested", [], dtype=pl.Float64)

    total_lots = int(round(float(quantity)))
    if total_lots <= 0:
        return pl.Series("q_requested", [0.0] * n, dtype=pl.Float64)

    per_bin = curves.get(bins.get_column(SECURITY_COL)[0], {})
    times = bins.get_column(BIN_TIME_COL).to_list()
    vol = pl.Series(
        [per_bin.get(t, (None, 0.0))[1] or 0.0 for t in times], dtype=pl.Float64
    ).fill_nan(0.0).clip(lower_bound=0.0)

    total = vol.sum()
    if total is None or total <= 0:
        # No historical volume for this security/day -> even-split fallback.
        return twap_schedule(bins, total_lots)

    # Cumulative rounding (same as vwap_schedule): the per-bin integer lots
    # telescope back to exactly ``total_lots``.
    cum_goal = (vol / total * total_lots).cum_sum().round(0)
    per_bin_lots = cum_goal.diff().fill_null(cum_goal.head(1))
    return per_bin_lots.cast(pl.Float64).rename("q_requested")


def _solve_mu(total_lots: int, s: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Continuous cost-optimal per-bin quantities summing to ``total_lots``.

    ``s`` and ``v`` are the per-bin spread/liquidity for the **active** bins only
    (finite ``s > 0`` and ``v > 0``). Solves ``sum_k q_k*(mu) = total_lots`` for
    the Lagrange multiplier ``mu`` by bisection - ``q_k*(mu)`` is non-decreasing
    in ``mu``, so the total is monotone and bisection is exact to tolerance.
    Returns the continuous ``q_k*`` (not yet rounded) for those active bins.
    """
    def q_of(mu: float) -> np.ndarray:
        bracket = np.maximum(0.0, mu / (MARG_COEF * s) - THRESH_RATIO)
        return v * bracket * bracket

    # Upper bound: grow mu until the achievable total reaches the target.
    mu_hi = max(SLIPPAGE_BASE * float(s.max()) * 2.0, 1e-9)
    for _ in range(200):
        if q_of(mu_hi).sum() >= total_lots:
            break
        mu_hi *= 2.0
    mu_lo = 0.0
    for _ in range(100):
        mid = 0.5 * (mu_lo + mu_hi)
        if q_of(mid).sum() < total_lots:
            mu_lo = mid
        else:
            mu_hi = mid
    return q_of(mu_hi)


def _largest_remainder_round(q: np.ndarray, total_lots: int) -> np.ndarray:
    """Round a continuous allocation to whole lots summing exactly to ``total``.

    Floors every bin, then deals the leftover ``total - sum(floor)`` lots one each
    to the bins with the largest fractional remainders (the standard
    largest-remainder method). Zero-allocation bins keep their integer 0.
    """
    floor = np.floor(q)
    frac = q - floor
    out = floor.astype(np.int64)
    remainder = int(total_lots - out.sum())
    if remainder > 0:
        # earliest bins break ties (stable sort) so the result is deterministic
        order = np.argsort(-frac, kind="stable")
        out[order[:remainder]] += 1
    elif remainder < 0:
        # defensive: overshoot shouldn't happen, but peel lots off the smallest
        # fractions among bins that still hold >= 1 lot.
        eligible = np.where(out > 0)[0]
        order = eligible[np.argsort(frac[eligible], kind="stable")]
        out[order[: -remainder]] -= 1
    return out


def liq_spr_static(bins: pl.DataFrame, quantity: float, *,
                   curves: dict, **_: object) -> pl.Series:
    """Cost-minimising schedule over static historical spread/liquidity curves.

    Chooses the per-bin lots ``q_k`` that **minimise the fill model's total
    slippage cost** for this order, using the static per-(security, bin) curves
    from :func:`build_liq_spread_curves` (so it uses no information from the day
    being executed). With equal spreads across bins this reduces to VWAP; the
    spread term tilts allocation toward tight-spread, liquid bins.

    The continuous optimum is ``q_k*(mu) = V_k * max(0, mu/(MARG_COEF*s_k) -
    THRESH_RATIO)**2`` with ``mu`` solved so the bins sum to ``round(quantity)``
    (see module notes above :func:`build_liq_spread_curves`). The result is then
    rounded to whole lots by the largest-remainder method, summing exactly to the
    order. No ``2*V`` fill cap is imposed (pure cost minimisation); the convex
    cost already discourages over-concentration.

    Parameters
    ----------
    bins : pl.DataFrame
        Bins for one (security, date); must carry ``security`` and
        ``bin_start_time``. Aligned to the caller's (time-sorted) row order.
    quantity : float
        Total lots to execute (rounded to whole lots).
    curves : dict
        ``{security: {bin_start_time: (s_k, V_k)}}`` from
        :func:`build_liq_spread_curves`, passed via ``strategy_params``.
    **_ :
        Ignored. Present so strategies share a uniform calling convention.

    Returns
    -------
    pl.Series
        Integer-valued float series of length ``len(bins)`` summing to
        ``round(quantity)``. Bins with no historical liquidity/spread get 0; if
        **no** bin has a usable curve, falls back to the integer-lot TWAP split.
    """
    n = bins.height
    if n == 0:
        return pl.Series("q_requested", [], dtype=pl.Float64)

    total_lots = int(round(float(quantity)))
    if total_lots <= 0:
        return pl.Series("q_requested", [0.0] * n, dtype=pl.Float64)

    security = bins.get_column(SECURITY_COL)[0]
    per_bin = curves.get(security, {})
    times = bins.get_column(BIN_TIME_COL).to_list()
    s = np.array([per_bin.get(t, (np.nan, 0.0))[0] for t in times], dtype=float)
    v = np.array([per_bin.get(t, (np.nan, 0.0))[1] for t in times], dtype=float)

    active = np.isfinite(s) & (s > 0) & np.isfinite(v) & (v > 0)
    if not active.any():
        # No historical curve for this security/day -> safe even-split fallback.
        return twap_schedule(bins, total_lots)

    q_cont = _solve_mu(total_lots, s[active], v[active])
    q_full = np.zeros(n, dtype=float)
    q_full[active] = q_cont
    q_int = _largest_remainder_round(q_full, total_lots)
    return pl.Series("q_requested", q_int.astype(float), dtype=pl.Float64)


def _solve_omniscient_q(total_lots: int, base: np.ndarray, s: np.ndarray,
                        v: np.ndarray) -> np.ndarray:
    """Continuous cost-optimal per-bin quantities for the omniscient strategy.

    Solves ``sum_k q_k(mu) = total_lots`` by bisection, where each bin is both
    floored at 0 and **capped at the fill limit** ``2 * v_k``::

        q_k(mu) = min( v_k * max(0, (mu - base_k - SLIPPAGE_BASE*s_k)
                                     / (MARG_COEF*s_k))**2 , 2*v_k )

    ``base_k = sign * vwap_k`` carries the (signed) price term. ``q_k(mu)`` is
    non-decreasing in ``mu``, so the total is monotone and bisection is exact to
    tolerance. Caller guarantees ``total_lots`` is below the total cap capacity.
    """
    cap = FILL_CAP_MULTIPLE * v

    def q_of(mu: float) -> np.ndarray:
        bracket = np.maximum(0.0, (mu - base - SLIPPAGE_BASE * s) / (MARG_COEF * s))
        return np.minimum(v * bracket * bracket, cap)

    mu_lo = float(base.min())                     # Q(mu_lo) == 0
    step = max(1.0, float(np.abs(base).max()) * 0.5 + float(s.max()))
    mu_hi = mu_lo + step
    for _ in range(200):
        if q_of(mu_hi).sum() >= total_lots:
            break
        mu_hi += step
        step *= 2.0
    for _ in range(100):
        mid = 0.5 * (mu_lo + mu_hi)
        if q_of(mid).sum() < total_lots:
            mu_lo = mid
        else:
            mu_hi = mid
    return q_of(mu_hi)


def _omniscient_day(bins: pl.DataFrame):
    """Pull the realised per-bin arrays and the fillable mask for one day.

    Returns ``(vwap, spread, vol, active)`` as numpy arrays, where ``active`` marks
    bins that can actually fill (full market data and positive volume).
    """
    vwap = bins.get_column(VWAP_COL).cast(pl.Float64).to_numpy()
    ask = bins.get_column(ASK_COL).cast(pl.Float64).to_numpy()
    bid = bins.get_column(BID_COL).cast(pl.Float64).to_numpy()
    vol = (bins.get_column(VOLUME_COL).cast(pl.Float64)
           .fill_null(0.0).fill_nan(0.0).to_numpy())
    active = (np.isfinite(vwap) & np.isfinite(ask) & np.isfinite(bid) & (vol > 0))
    return vwap, ask - bid, vol, active


def _omniscient_cost_schedule(bins: pl.DataFrame, total_lots: int, sign: float,
                              with_drift: bool) -> pl.Series:
    """Capped cost-minimising lookahead schedule (shared by the omniscient family).

    Minimises ``sum_k [ base_k * q_k + slip_cost_k(q_k) ]`` over the day's realised
    spread/liquidity, with ``0 <= q_k <= 2 * V_k`` and ``sum_k q_k = total_lots``,
    where ``slip_cost_k`` is the fill model's spread cost. The linear term carries
    the (signed) price/drift coefficient: ``base_k = sign * vwap_k`` when
    ``with_drift`` (the full :func:`omniscient`), or ``0`` when not (the no-drift
    :func:`omniscient_liq_spr`, which then ignores ``sign``). Respecting the cap
    means it fills 100% whenever ``total_lots <= sum_k 2*V_k``; otherwise it maxes
    out every fillable bin.
    """
    n = bins.height
    vwap, spread, vol, active = _omniscient_day(bins)
    if not active.any():
        return twap_schedule(bins, total_lots)

    v = vol[active]
    capacity = float((FILL_CAP_MULTIPLE * v).sum())
    q_full = np.zeros(n, dtype=float)
    if total_lots >= capacity:
        q_full[active] = FILL_CAP_MULTIPLE * v          # max fill; cannot do 100%
    else:
        s_eff = np.maximum(spread[active], 1e-9)        # guard zero/neg spreads
        base = sign * vwap[active] if with_drift else np.zeros(int(active.sum()))
        q_cont = _solve_omniscient_q(total_lots, base, s_eff, v)
        q_full[active] = _largest_remainder_round(q_cont, total_lots)
    return pl.Series("q_requested", q_full, dtype=pl.Float64)


def omniscient_vwap(bins: pl.DataFrame, quantity: float, **_: object) -> pl.Series:
    """Lookahead VWAP benchmark: track the **realised** volume profile (liquidity only).

    The classic VWAP benchmark - the one execution is normally measured against.
    With perfect foresight of the day's **volume** (but using *no* spread or price
    information), it requests each fillable bin in proportion to that bin's realised
    share of the day's volume::

        q_requested[k] = quantity * V_k / sum(V_k)     (over fillable bins)

    rounded to whole lots. Allocating only to fillable bins (data present, volume
    > 0) and never above the ``2 * V_k`` cap, it **fills 100%** whenever the order
    fits the day's capacity (``quantity <= sum_k 2*V_k``); otherwise it maxes out
    every bin. Unlike :func:`vwap_static` (historical curve, deployable), this uses
    the actual day's volume, so it is a benchmark, not a deployable strategy. It is
    the liquidity-only member of the omniscient family - it deliberately does **not**
    optimise spread or price drift (see :func:`omniscient_liq_spr` and
    :func:`omniscient`).

    Parameters
    ----------
    bins : pl.DataFrame
        Bins for one (security, date) with ``volume``, ``vwap``, ``twa_ask``,
        ``twa_bid``. Aligned to the caller's (time-sorted) row order.
    quantity : float
        Total lots to execute (rounded to whole lots).
    **_ :
        Ignored. Present so strategies share a uniform calling convention.

    Returns
    -------
    pl.Series
        Integer-valued float series of length ``len(bins)``. Sums to
        ``round(quantity)`` when the order fits the day's capacity; otherwise sums
        to capacity. Falls back to the integer-lot TWAP split if nothing can fill.
    """
    n = bins.height
    if n == 0:
        return pl.Series("q_requested", [], dtype=pl.Float64)
    total_lots = int(round(float(quantity)))
    if total_lots <= 0:
        return pl.Series("q_requested", [0.0] * n, dtype=pl.Float64)

    _vwap, _spread, vol, active = _omniscient_day(bins)
    if not active.any():
        return twap_schedule(bins, total_lots)

    v = vol[active]
    capacity = float((FILL_CAP_MULTIPLE * v).sum())
    q_full = np.zeros(n, dtype=float)
    if total_lots >= capacity:
        q_full[active] = FILL_CAP_MULTIPLE * v
    else:
        # proportional to realised volume; cumulative rounding keeps the sum exact
        cum_goal = np.round(np.cumsum(total_lots * v / v.sum()))
        per_bin = np.diff(np.concatenate(([0.0], cum_goal)))
        q_full[active] = per_bin
    return pl.Series("q_requested", q_full, dtype=pl.Float64)


def omniscient_liq_spr(bins: pl.DataFrame, quantity: float, **_: object) -> pl.Series:
    """Lookahead liquidity+spread benchmark: minimise slippage cost, **no drift**.

    The middle member of the omniscient family. With perfect foresight of the day's
    realised **spread and liquidity** (but **not** using the price/drift signal), it
    chooses the per-bin lots that minimise the fill model's spread cost
    ``sum_k [ SLIPPAGE_BASE*s_k*q_k + SLIPPAGE_COEF*s_k*q_k**1.5 / sqrt(V_k) ]``
    subject to ``0 <= q_k <= 2*V_k`` and the bins summing to the order. It is the
    same cost minimisation as :func:`liq_spr_static`, but on the *actual* day's
    spread/liquidity (lookahead) and respecting the fill cap, so it fills 100%
    whenever the order fits capacity.

    Because the spread cost is the same whether buying or selling, this is
    side-independent. It isolates the part of the omniscient edge that comes from
    spread/liquidity timing - which is far more predictable than the drift term that
    :func:`omniscient` additionally exploits.

    Parameters
    ----------
    bins : pl.DataFrame
        Bins for one (security, date) with ``volume``, ``vwap``, ``twa_ask``,
        ``twa_bid``. Aligned to the caller's (time-sorted) row order.
    quantity : float
        Total lots to execute (rounded to whole lots).
    **_ :
        Ignored. Present so strategies share a uniform calling convention.

    Returns
    -------
    pl.Series
        Integer-valued float series of length ``len(bins)`` (see
        :func:`_omniscient_cost_schedule`).
    """
    n = bins.height
    if n == 0:
        return pl.Series("q_requested", [], dtype=pl.Float64)
    total_lots = int(round(float(quantity)))
    if total_lots <= 0:
        return pl.Series("q_requested", [0.0] * n, dtype=pl.Float64)
    return _omniscient_cost_schedule(bins, total_lots, sign=1.0, with_drift=False)


def omniscient(bins: pl.DataFrame, quantity: float, *,
               side: str = "buy", **_: object) -> pl.Series:
    """Full lookahead lower bound: minimise realised cost using spread, liquidity AND drift.

    The strongest member of the omniscient family: it sees the **whole day** for this
    (security, date) - realised per-bin price (``vwap``), spread (``twa_ask -
    twa_bid``) and liquidity (``volume``) - and chooses the per-bin lots that
    **minimise the actual implementation shortfall**. Since the arrival benchmark
    ``arrival_price * quantity`` is fixed, that is the same as minimising the
    transacted notional::

        buy : minimise  sum_k q_k * (vwap_k + slip_k * s_k)   -> trade where price is LOW
        sell: maximise  sum_k q_k * (vwap_k - slip_k * s_k)   -> trade where price is HIGH

    with ``slip_k = SLIPPAGE_BASE + SLIPPAGE_COEF * sqrt(q_k / V_k)``. Convex, so the
    optimum equalises the signed marginal cost across traded bins, each floored at 0
    and **capped at** ``2 * V_k`` (so it fills 100% when the order fits capacity,
    else maxes every bin). See :func:`_omniscient_cost_schedule` / ``_solve_omniscient_q``.

    It exploits everything a schedule could - spread, liquidity **and** intraday
    price drift - so it is the unbeatable reference, not a deployable strategy. The
    drift edge in particular is not realistically predictable; the no-drift
    :func:`omniscient_liq_spr` and the liquidity-only :func:`omniscient_vwap` are the
    achievable benchmarks.

    Parameters
    ----------
    bins : pl.DataFrame
        Bins for one (security, date) with ``volume``, ``vwap``, ``twa_ask`` and
        ``twa_bid``. Aligned to the caller's (time-sorted) row order.
    quantity : float
        Total lots to execute (rounded to whole lots).
    side : str, default ``"buy"``
        ``"buy"`` or ``"sell"``; forwarded by :func:`run_strategy`. Sets whether
        low-price bins (buy) or high-price bins (sell) are favoured.
    **_ :
        Ignored. Present so strategies share a uniform calling convention.

    Returns
    -------
    pl.Series
        Integer-valued float series of length ``len(bins)`` (see
        :func:`_omniscient_cost_schedule`).
    """
    n = bins.height
    if n == 0:
        return pl.Series("q_requested", [], dtype=pl.Float64)
    total_lots = int(round(float(quantity)))
    if total_lots <= 0:
        return pl.Series("q_requested", [0.0] * n, dtype=pl.Float64)
    sign = 1.0 if str(side).strip().lower() == "buy" else -1.0
    return _omniscient_cost_schedule(bins, total_lots, sign=sign, with_drift=True)


def _simulate_fills_df(df: pl.DataFrame, *, side_col: str = "side",
                       qty_col: str = "q_requested") -> pl.DataFrame:
    """Vectorised counterpart of :func:`simulate_bin_fill` over a bins frame.

    ``df`` must contain the market-data columns (``VOLUME_COL``, ``VWAP_COL``,
    ``ASK_COL``, ``BID_COL``), a per-bin requested-quantity column (``qty_col``)
    and a side column (``side_col``, values ``"buy"``/``"sell"``). Adds the
    fill columns and returns the augmented frame. Logic mirrors the scalar
    function exactly, including the missing-data and zero-volume no-fill rules.
    """
    vol, vwap, ask, bid = VOLUME_COL, VWAP_COL, ASK_COL, BID_COL

    df = df.with_columns(
        (pl.col(ask) - pl.col(bid)).alias("spread"),
        (
            pl.col(vol).is_not_null()
            & pl.col(vwap).is_not_null()
            & pl.col(ask).is_not_null()
            & pl.col(bid).is_not_null()
        ).alias("data_available"),
    )
    # q_filled = min(requested, 2*volume), but only when data present & volume>0
    df = df.with_columns(
        pl.when(pl.col("data_available") & (pl.col(vol) > 0))
        .then(pl.min_horizontal(pl.col(qty_col), FILL_CAP_MULTIPLE * pl.col(vol)))
        .otherwise(0.0)
        .alias("q_filled")
    )
    df = df.with_columns(
        (pl.col(qty_col) - pl.col("q_filled")).alias("q_unfilled"),
        (pl.col("q_filled") > 0).alias("filled"),
        pl.when(pl.col("q_filled") > 0)
        .then(pl.col("q_filled") / pl.col(vol))
        .otherwise(0.0)
        .alias("participation_rate"),
    )
    df = df.with_columns(
        pl.when(pl.col("filled"))
        .then(SLIPPAGE_BASE + SLIPPAGE_COEF * pl.col("participation_rate").sqrt())
        .otherwise(None)
        .alias("slippage_factor"),
        pl.when(pl.col(side_col).str.to_lowercase() == "buy")
        .then(1.0)
        .otherwise(-1.0)
        .alias("_sign"),
    )
    df = df.with_columns(
        pl.when(pl.col("filled"))
        .then(pl.col(vwap) + pl.col("_sign") * pl.col("slippage_factor") * pl.col("spread"))
        .otherwise(None)
        .alias("fill_price"),
    )
    df = df.with_columns(
        pl.when(pl.col("filled"))
        .then(pl.col("q_filled") * pl.col("fill_price"))
        .otherwise(0.0)
        .alias("notional"),
        # per-bin execution cost = spread cost paid vs the bin VWAP (>= 0)
        pl.when(pl.col("filled"))
        .then(pl.col("slippage_factor") * pl.col("spread") * pl.col("q_filled"))
        .otherwise(0.0)
        .alias("cost"),
    )
    return df.drop("_sign")


# tidy per-bin output column order
_FILL_COLS = [
    "order_id", SECURITY_COL, DATE_COL, QCODE_COL, "side", "order_quantity",
    BIN_TIME_COL, VOLUME_COL, OPEN_COL, VWAP_COL, BID_COL, ASK_COL, "spread",
    "q_requested", "q_filled", "q_unfilled", "participation_rate",
    "slippage_factor", "fill_price", "notional", "cost", "filled",
    "data_available",
]


def run_strategy(
    bins: pl.DataFrame,
    side: str,
    quantity: float,
    *,
    strategy: Strategy = twap_schedule,
    strategy_params: Optional[dict] = None,
    order_id: object = 0,
) -> pl.DataFrame:
    """Run one order (one security-day) through a strategy and simulate fills.

    The order is identified by the supplied ``bins`` (which already pin down a
    single security and date) plus ``side`` and ``quantity``. The ``strategy``
    decides how much to request in each bin; this function then applies the
    fill model (vectorised) and returns one row per bin.

    Parameters
    ----------
    bins : pl.DataFrame
        Bins for exactly one (security, date), with at least the columns
        ``volume``, ``vwap``, ``twa_ask``, ``twa_bid``, ``bin_start_time``,
        ``security`` and ``publication_date`` (``qcode`` is carried through if
        present). Need not be pre-sorted - it is sorted by ``bin_start_time``
        here. Pass a slice like
        ``binned.filter((pl.col("security")==s) & (pl.col("publication_date")==d))``.
    side : str
        ``"buy"`` or ``"sell"`` (case-insensitive).
    quantity : float
        Total quantity to execute over the day (magnitude).
    strategy : callable, default :func:`twap_schedule`
        ``strategy(bins, quantity, **strategy_params) -> Sequence[float]`` of
        length ``len(bins)`` giving the requested quantity per bin (aligned to
        the time-sorted bins).
    strategy_params : dict, optional
        Extra keyword arguments forwarded to ``strategy``.
    order_id : hashable, default 0
        Identifier stamped on every row; set by :func:`run_trade_list` to a
        unique value per order so :func:`summarise_fills` can group on it.

    Returns
    -------
    pl.DataFrame
        One row per bin with the requested/filled quantities, participation
        rate, slippage factor, spread, fill price, notional, per-bin cost and
        status flags - everything needed to compute fill rate and
        implementation shortfall downstream (see :func:`summarise_fills`).

    Raises
    ------
    ValueError
        If ``bins`` is empty or ``side`` is not ``"buy"``/``"sell"``.

    Examples
    --------
    >>> import polars as pl
    >>> bins = pl.DataFrame({
    ...     "security": ["GC2025V Comdty"] * 3,
    ...     "publication_date": ["2025-08-21"] * 3,
    ...     "qcode": ["GC"] * 3,
    ...     "bin_start_time": ["12:30", "12:35", "12:40"],
    ...     "volume": [100, 0, 50],          # middle bin is a no-trade bin
    ...     "vwap": [2000.0, None, 2001.0],
    ...     "twa_ask": [2000.5, None, 2001.5],
    ...     "twa_bid": [1999.5, None, 2000.5],
    ... })
    >>> out = run_strategy(bins, "buy", 300)        # 100 per bin
    >>> out.select("bin_start_time", "q_requested", "q_filled", "filled").to_dicts()
    [{'bin_start_time': '12:30', 'q_requested': 100.0, 'q_filled': 100.0, 'filled': True}, {'bin_start_time': '12:35', 'q_requested': 100.0, 'q_filled': 0.0, 'filled': False}, {'bin_start_time': '12:40', 'q_requested': 100.0, 'q_filled': 100.0, 'filled': True}]
    """
    side_norm = str(side).strip().lower()
    if side_norm not in ("buy", "sell"):
        raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")
    if bins.height == 0:
        raise ValueError("`bins` is empty - no bins for this (security, date).")

    b = bins.sort(BIN_TIME_COL)
    # Forward `side` so side-aware strategies (e.g. omniscient) can use it; all
    # other strategies absorb it via **_. An explicit side in strategy_params wins.
    params = dict(strategy_params or {})
    params.setdefault("side", side_norm)
    requested = strategy(b, quantity, **params)
    req_series = requested if isinstance(requested, pl.Series) else pl.Series(requested)
    if req_series.len() != b.height:
        raise ValueError(
            f"strategy returned {req_series.len()} quantities for {b.height} bins"
        )

    b = b.with_columns(
        req_series.cast(pl.Float64).alias("q_requested"),
        pl.lit(side_norm).alias("side"),
        pl.lit(float(quantity)).alias("order_quantity"),
        pl.lit(order_id).alias("order_id"),
    )
    b = _simulate_fills_df(b)
    return b.select([c for c in _FILL_COLS if c in b.columns])


def run_trade_list(
    bins: pl.DataFrame,
    orders: pl.DataFrame,
    *,
    trade_list: Optional[str] = None,
    trade_list_col: str = "trade_list",
    strategy: Strategy = twap_schedule,
    strategy_params: Optional[dict] = None,
    security_col: str = "security",
    date_col: str = "date",
    side_col: str = "side",
    quantity_col: str = "quantity",
    carry_cols: Optional[Sequence[str]] = None,
    verbose: bool = True,
) -> pl.DataFrame:
    """Run a trade list, one :func:`run_strategy` call per order.

    The ``TRADE_LIST`` table holds six rows per (security, date) - one per
    size-bucket x side: ``small_buys``, ``small_sells``, ``medium_buys``,
    ``medium_sells``, ``large_buys``, ``large_sells``. Pass ``trade_list`` to
    run **just one** of those buckets (the usual case - you back-test one
    schedule at a time); leave it ``None`` to run every order in ``orders``.

    Parameters
    ----------
    bins : pl.DataFrame
        All bins to execute against (e.g. the full ``BINNED_DATA``, or a subset
        for the instruments/dates of interest). Partitioned internally by
        (security, date) for lookup, so for large inputs this holds the data
        grouped in memory - filter it down first if memory is tight.
    orders : pl.DataFrame
        One row per order. By default expects the ``TRADE_LIST`` schema columns
        ``security``, ``date``, ``trade_list``, ``side``, ``quantity`` (override
        via the ``*_col`` args). Each surviving row becomes one ``order_id``.
    trade_list : str, optional
        Which of the six buckets to run, e.g. ``"small_buys"``. When given,
        ``orders`` is first filtered to ``orders[trade_list_col] == trade_list``
        (and an error is raised if that value is absent). ``None`` runs all
        rows of ``orders`` as supplied.
    trade_list_col : str, default ``"trade_list"``
        Column in ``orders`` holding the bucket label. If present, it is also
        auto-added to ``carry_cols`` so every fill row is tagged with its
        bucket.
    strategy, strategy_params :
        Forwarded to :func:`run_strategy`.
    security_col, date_col, side_col, quantity_col : str
        Column names in ``orders``. Note ``orders`` uses ``date`` while
        ``bins`` uses ``publication_date``; both hold the same ``YYYY-MM-DD``
        strings and are matched on value.
    carry_cols : sequence of str, optional
        Extra ``orders`` columns to copy onto every fill row.
    verbose : bool, default True
        Print a one-line note if some orders had no matching bins.

    Returns
    -------
    pl.DataFrame
        All per-bin fill rows concatenated, with a unique ``order_id`` per
        order (plus the bucket label and any ``carry_cols``). Feed straight
        into :func:`summarise_fills`. Orders with no matching bins are skipped.

    Raises
    ------
    ValueError
        If ``trade_list`` is given but not found in ``orders[trade_list_col]``.

    Examples
    --------
    >>> fills = run_trade_list(binned, trade_list_df, trade_list="small_buys")  # doctest: +SKIP
    >>> summary = summarise_fills(fills)                                        # doctest: +SKIP
    """
    if trade_list is not None:
        if trade_list_col not in orders.columns:
            raise ValueError(
                f"orders has no {trade_list_col!r} column to select a bucket from"
            )
        available = orders[trade_list_col].unique().to_list()
        if trade_list not in available:
            raise ValueError(
                f"trade_list {trade_list!r} not found in column {trade_list_col!r}; "
                f"available: {sorted(v for v in available if v is not None)}"
            )
        orders = orders.filter(pl.col(trade_list_col) == trade_list)

    carry_cols = list(carry_cols or [])
    # tag every fill with its bucket label when the column is available
    if trade_list_col in orders.columns and trade_list_col not in carry_cols:
        carry_cols.append(trade_list_col)

    index = bins.partition_by(SECURITY_COL, DATE_COL, as_dict=True)

    frames = []
    missing = 0
    for i, row in enumerate(orders.iter_rows(named=True)):
        day = index.get((row[security_col], row[date_col]))
        if day is None or day.height == 0:
            missing += 1
            continue
        f = run_strategy(
            day, row[side_col], row[quantity_col],
            strategy=strategy, strategy_params=strategy_params, order_id=i,
        )
        if carry_cols:
            f = f.with_columns([pl.lit(row[c]).alias(c) for c in carry_cols])
        frames.append(f)

    if verbose and missing:
        print(f"[run_trade_list] {missing} of {orders.height} orders had no "
              f"matching bins and were skipped.")
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="vertical_relaxed")


def summarise_fills(
    fills: pl.DataFrame,
    *,
    by: str = "order_id",
    extra_cols: Sequence[str] = ("trade_list",),
) -> pl.DataFrame:
    """Collapse per-bin fills to one row per order with execution metrics.

    Parameters
    ----------
    fills : pl.DataFrame
        Per-bin output of :func:`run_strategy` / :func:`run_trade_list`.
    by : str, default ``"order_id"``
        Grouping key identifying an order.
    extra_cols : sequence of str, default ``("trade_list",)``
        Identifier columns to carry through unchanged (taken as ``first()`` per
        order). Names not present in ``fills`` are silently ignored, so the
        default safely picks up the bucket label when it exists.

    Returns
    -------
    pl.DataFrame
        One row per order with, among others:

        ``fill_rate``
            ``total_filled / order_quantity``.
        ``avg_fill_price``
            Notional-weighted average execution price.
        ``arrival_price`` / ``terminal_price``
            Benchmarks. ``arrival_price`` is the **open of the first bin** - the
            pre-trade decision price, before any execution moves the bin VWAP
            (falls back to the first bin VWAP only if no ``open`` column was
            carried through). ``terminal_price`` is the last non-null bin VWAP,
            used to mark the unfilled remainder.
        ``participation_overall``
            ``total_filled / total_volume`` over the window.
        ``total_cost``
            Sum of per-bin spread cost paid (currency).
        ``exec_slippage_bps``
            Filled-only slippage vs arrival price, in bps, signed so that
            **positive = worse** (paid above arrival on a buy / sold below on a
            sell): ``sign * (avg_fill_price - arrival) / arrival * 1e4``.
        ``is_bps`` / ``is_currency``
            Implementation shortfall vs arrival price including opportunity
            cost on the unfilled quantity marked at ``terminal_price``::

                realised = filled_notional + (order_quantity - total_filled) * terminal_price
                is_currency = sign * (realised - arrival_price * order_quantity)
                is_bps      = is_currency / (arrival_price * order_quantity) * 1e4

            Positive = underperformance vs an immediate arrival-price fill.
        ``is_slippage_bps`` / ``is_drift_bps`` / ``is_opportunity_bps``
            Additive decomposition of ``is_bps`` (the three sum to it exactly):
            **slippage** = spread cost paid on the filled lots (>= 0); **drift** =
            realised price move from the arrival price to the bin VWAPs actually
            filled at (timing of the filled portion); **opportunity** = the
            unfilled remainder marked at ``terminal_price`` vs arrival (0 when the
            order fills 100%). See :func:`decompose_is` to aggregate them.

    Notes
    -----
    Implementation shortfall here uses the **open of the first bin** as the
    arrival/decision price (it precedes any trading within the bin, unlike the
    bin VWAP) and the last traded bin VWAP as the terminal price for the
    unfilled remainder. Swap in your own benchmarks if your decision time
    differs from the first bin.
    """
    has_qcode = QCODE_COL in fills.columns
    # arrival = open of the first bin (pre-trade decision price); fall back to
    # the first bin VWAP only if no `open` column was carried through.
    arrival_col = OPEN_COL if OPEN_COL in fills.columns else VWAP_COL
    carried = [pl.col(c).first().alias(c) for c in extra_cols if c in fills.columns]
    f = fills.sort([by, BIN_TIME_COL])

    g = f.group_by(by, maintain_order=True).agg(
        *carried,
        security=pl.col(SECURITY_COL).first(),
        date=pl.col(DATE_COL).first(),
        qcode=(pl.col(QCODE_COL).first() if has_qcode else pl.lit(None)),
        side=pl.col("side").first(),
        order_quantity=pl.col("order_quantity").first(),
        n_bins=pl.len(),
        n_fillable=pl.col("data_available").sum(),
        n_filled=pl.col("filled").sum(),
        total_volume=pl.col(VOLUME_COL).sum(),
        total_requested=pl.col("q_requested").sum(),
        total_filled=pl.col("q_filled").sum(),
        filled_notional=pl.col("notional").sum(),
        # notional of the filled lots at the *bin VWAP* (i.e. before spread cost) -
        # used to split IS into a price-drift term vs the spread-cost term.
        filled_vwap_notional=(pl.col("q_filled") * pl.col(VWAP_COL)).sum(),
        total_cost=pl.col("cost").sum(),
        arrival_price=pl.col(arrival_col).filter(pl.col(arrival_col).is_not_null()).first(),
        terminal_price=pl.col(VWAP_COL).filter(pl.col(VWAP_COL).is_not_null()).last(),
    )

    sign = pl.when(pl.col("side") == "buy").then(1.0).otherwise(-1.0)
    g = g.with_columns(
        (pl.col("total_filled") / pl.col("order_quantity")).alias("fill_rate"),
        pl.when(pl.col("total_filled") > 0)
        .then(pl.col("filled_notional") / pl.col("total_filled"))
        .otherwise(None)
        .alias("avg_fill_price"),
        (pl.col("order_quantity") - pl.col("total_filled")).alias("unfilled_qty"),
        pl.when(pl.col("total_volume") > 0)
        .then(pl.col("total_filled") / pl.col("total_volume"))
        .otherwise(None)
        .alias("participation_overall"),
    )
    realised = pl.col("filled_notional") + pl.col("unfilled_qty") * pl.col("terminal_price")
    paper = pl.col("arrival_price") * pl.col("order_quantity")
    g = g.with_columns(
        (sign * (pl.col("avg_fill_price") - pl.col("arrival_price"))
         / pl.col("arrival_price") * 1e4).alias("exec_slippage_bps"),
        (sign * (realised - paper)).alias("is_currency"),
    )
    # --- IS decomposition (the three pieces sum to is_currency / is_bps) ---
    #   slippage   = spread cost paid on the filled lots             (>= 0)
    #   drift      = price move arrival -> filled bin VWAPs (timing on the filled part)
    #   opportunity= unfilled remainder marked at terminal vs arrival (0 if 100% filled)
    slippage_cur = pl.col("total_cost")
    drift_cur = sign * (pl.col("filled_vwap_notional")
                        - pl.col("arrival_price") * pl.col("total_filled"))
    opp_cur = sign * pl.col("unfilled_qty") * (pl.col("terminal_price")
                                               - pl.col("arrival_price"))
    g = g.with_columns(
        pl.when(paper != 0).then(pl.col("is_currency") / paper * 1e4)
        .otherwise(None).alias("is_bps"),
        pl.when(paper != 0).then(slippage_cur / paper * 1e4)
        .otherwise(None).alias("is_slippage_bps"),
        pl.when(paper != 0).then(drift_cur / paper * 1e4)
        .otherwise(None).alias("is_drift_bps"),
        pl.when(paper != 0).then(opp_cur / paper * 1e4)
        .otherwise(None).alias("is_opportunity_bps"),
    )
    return g


def decompose_is(summary: pl.DataFrame, by: Optional[str] = None) -> pl.DataFrame:
    """Aggregate the implementation-shortfall decomposition across orders.

    Averages the per-order IS components from :func:`summarise_fills` so you can
    see, in basis points, where the shortfall comes from:

    * ``is_slippage_bps``    - spread cost paid on the filled lots (always >= 0);
    * ``is_drift_bps``       - realised price drift between the arrival price and
      the bin VWAPs you actually filled at (the *timing* of the filled portion);
    * ``is_opportunity_bps`` - cost of the unfilled remainder, marked at the
      terminal price vs arrival (zero when the order fills 100%).

    The three sum to ``is_bps`` for every order, and because the mean is linear
    they also sum to the mean ``is_bps`` shown here. Orders with an undefined
    ``is_bps`` (e.g. a non-trading day with no arrival price) are dropped.

    Parameters
    ----------
    summary : pl.DataFrame
        Output of :func:`summarise_fills`.
    by : str, optional
        Column to group by (e.g. ``"qcode"`` or ``"trade_list"``). ``None``
        returns a single overall row.

    Returns
    -------
    pl.DataFrame
        ``n_orders``, ``fill_rate`` and the mean of each component plus
        ``is_bps`` (one row, or one row per group).
    """
    comps = ["is_slippage_bps", "is_drift_bps", "is_opportunity_bps", "is_bps"]
    s = summary.filter(pl.col("is_bps").is_not_null() & pl.col("is_bps").is_not_nan())
    aggs = ([pl.len().alias("n_orders"), pl.col("fill_rate").mean().alias("fill_rate")]
            + [pl.col(c).mean().alias(c) for c in comps])
    if by is None:
        return s.select(aggs)
    return s.group_by(by, maintain_order=True).agg(aggs).sort(by)


def decompose_is_stats(summary: pl.DataFrame,
                       label: Optional[str] = None) -> pl.DataFrame:
    """Per-component distribution stats of the IS decomposition (one strategy).

    Returns one row per IS component - ``slippage``, ``drift``, ``opportunity``
    and the ``total`` (``is_bps``) - with the **mean**, **median** and (sample,
    ddof=1) **variance** in bps across orders. Orders with an undefined ``is_bps``
    are dropped. Pass ``label`` (e.g. the strategy name) to tag every row, so the
    results for several strategies can be ``pl.concat``-ed into one table.

    Parameters
    ----------
    summary : pl.DataFrame
        Output of :func:`summarise_fills` (must carry the ``is_*_bps`` columns).
    label : str, optional
        Strategy name stamped into a leading ``strategy`` column when given.

    Returns
    -------
    pl.DataFrame
        Columns ``[strategy?, component, mean, median, variance]`` - four rows
        (slippage, drift, opportunity, total).
    """
    comps = [("slippage", "is_slippage_bps"), ("drift", "is_drift_bps"),
             ("opportunity", "is_opportunity_bps"), ("total", "is_bps")]
    s = summary.filter(pl.col("is_bps").is_not_null() & pl.col("is_bps").is_not_nan())
    recs = []
    for name, col in comps:
        c = s.get_column(col)
        rec = {"component": name, "mean": c.mean(),
               "median": c.median(), "variance": c.var()}
        if label is not None:
            rec = {"strategy": label, **rec}
        recs.append(rec)
    return pl.DataFrame(recs)
