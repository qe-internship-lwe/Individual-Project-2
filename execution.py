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
    "build_rolling_curves",
    "vwap_static",
    "liq_spr_static",
    "vwap_adaptive",
    "omniscient_vwap",
    "omniscient_liq_spr",
    "omniscient",
    "run_strategy",
    "run_trade_list",
    "summarise_fills",
    "decompose_is",
    "decompose_is_stats",
    "asset_impact",
    "order_impact_stats",
    "participation_dispersion",
    "participation_stats",
    "build_factor_curves",
    "build_spread_factor_curves",
    "certainty_equivalent_volume",
    "vwap_factor",
    "build_ols_curves",
    "adaptive_volume_ols",
    "spread_factor_vol_ols",
    "spread_factor_vol_factor",
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
CROSS_SPREAD_FRACTION: float = 0.5  # opportunity cost: cross half the terminal spread on the shortfall


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
        Nested lookup ``{security: {bin_start_time: (s_k, V_k, L_k)}}`` for O(1)
        access per bin inside the strategy, where ``L_k`` is the mean of log volume
        (``exp(L_k)`` = geometric-mean volume, used by :func:`vwap_adaptive`). Pass
        it as ``strategy_params={"curves": curves}`` to :func:`run_strategy` /
        :func:`run_trade_list`.
    """
    g = (
        bins.select(SECURITY_COL, BIN_TIME_COL, VOLUME_COL, ASK_COL, BID_COL)
        .with_columns((pl.col(ASK_COL) - pl.col(BID_COL)).alias("_spread"))
        .group_by(SECURITY_COL, BIN_TIME_COL)
        .agg(
            pl.col("_spread").mean().alias("spread_curve"),
            pl.col(VOLUME_COL).cast(pl.Float64).mean().alias("liq_curve"),
            # mean of logs (-> geometric-mean volume exp(L_k)); used by vwap_adaptive
            pl.when(pl.col(VOLUME_COL) > 0).then(pl.col(VOLUME_COL).cast(pl.Float64).log())
            .otherwise(None).mean().alias("logvol_curve"),
        )
    )
    curves: dict = {}
    for sec, bt, s_k, v_k, l_k in g.iter_rows():
        curves.setdefault(sec, {})[bt] = (s_k, v_k, l_k)
    return curves


def build_rolling_curves(
    history: pl.DataFrame,
    needed: pl.DataFrame,
    *,
    window_days: int = 22,
) -> dict:
    """Per-order spread/liquidity curves from the **trailing** ``window_days``.

    Lookahead-free replacement for :func:`build_liq_spread_curves`. For every
    ``(qcode, date)`` order key in ``needed`` this estimates the per-bin spread
    ``s_k`` and liquidity ``V_k`` from **only the qcode's previous
    ``window_days`` trading days, strictly before that date** - a sliding window
    that uses no information from the execution day or any future day. (The
    static builder averages over *all* days, including days after the order, so
    its "no same-day lookahead" claim still leaks the future profile.)

    The profile is built at the **qcode** level (all that qcode's contract
    months pooled per day), because the intraday shape is a qcode/exchange
    property and a single contract month rarely has 22 prior days of its own.
    Per ``(qcode, date_of_day, bin_start_time)`` the daily inputs are the
    **summed** volume across that qcode's securities and the **volume-weighted**
    mean spread; the trailing window then takes, per bin:

    * ``V_k`` = mean daily volume over the window (absent days count as 0
      volume, so a bin that seldom trades gets a correspondingly low weight);
    * ``s_k`` = mean spread over the window days on which the bin actually
      quoted (missing days skipped).

    Fewer than ``window_days`` available -> uses whatever prior days exist. On
    the **first** day a qcode is traded there is no prior data, so that key is
    simply omitted from the result; :func:`vwap_static` / :func:`liq_spr_static`
    then find no curve and fall back to the even-split :func:`twap_schedule`.

    Parameters
    ----------
    history : pl.DataFrame
        Full bin history to estimate from (e.g. the whole ``BINNED_DATA``), with
        ``qcode``, ``publication_date``, ``bin_start_time``, ``volume``,
        ``twa_ask``, ``twa_bid``.
    needed : pl.DataFrame
        The order keys to emit curves for; must have a ``qcode`` column and a
        date column named either ``date`` (the ``TRADE_LIST`` convention) or
        ``publication_date``. Only these ``(qcode, date)`` pairs are returned,
        which keeps the result small even though the window is computed over the
        whole history.
    window_days : int, default 22
        Length of the trailing trading-day window (per qcode).

    Returns
    -------
    dict
        ``{(qcode, date): {bin_start_time: (s_k, V_k, L_k)}}`` for the keys in
        ``needed`` that have at least one prior trading day, where ``L_k`` is the
        mean of log volume over the window's traded days (``exp(L_k)`` =
        geometric-mean volume, used by :func:`vwap_adaptive`; ``V_k`` is the
        arithmetic mean used by :func:`vwap_static` / :func:`liq_spr_static`). Pass
        as ``strategy_params={"curves": curves}``; strategies look up the rolling
        ``(qcode, date)`` key automatically.
    """
    date_col = "date" if "date" in needed.columns else DATE_COL
    want: dict = {}
    for qc, dt in needed.select(QCODE_COL, date_col).unique().iter_rows():
        want.setdefault(qc, set()).add(dt)

    h = history.select(QCODE_COL, DATE_COL, BIN_TIME_COL, VOLUME_COL, ASK_COL, BID_COL)
    h = h.filter(pl.col(QCODE_COL).is_in(list(want.keys())))
    h = h.with_columns(
        pl.col(VOLUME_COL).cast(pl.Float64).fill_null(0.0).alias("_v"),
        (pl.col(ASK_COL) - pl.col(BID_COL)).alias("_spr"),
    )
    # Daily per-(qcode, date, bin): summed volume + volume-weighted mean spread.
    daily = h.group_by(QCODE_COL, DATE_COL, BIN_TIME_COL).agg(
        pl.col("_v").sum().alias("vol"),
        pl.when(pl.col("_v").sum() > 0)
        .then((pl.col("_spr") * pl.col("_v")).sum() / pl.col("_v").sum())
        .otherwise(pl.col("_spr").mean())
        .alias("spr"),
    )

    curves: dict = {}
    for qc, want_dates in want.items():
        d = daily.filter(pl.col(QCODE_COL) == qc)
        if d.height == 0:
            continue
        # Dense (date x bin) grid so absent bins count as zero-volume days and
        # the trailing window is measured in the qcode's own trading days.
        dates = d.select(DATE_COL).unique()
        binsa = d.select(BIN_TIME_COL).unique()
        grid = dates.join(binsa, how="cross").join(
            d, on=[DATE_COL, BIN_TIME_COL], how="left"
        )
        grid = grid.with_columns(
            pl.col("vol").fill_null(0.0),
            pl.col("spr").is_not_null().cast(pl.Float64).alias("_pres"),
            pl.col("spr").fill_null(0.0).alias("_spr0"),
        ).with_columns(
            # log-volume on traded days only (mean of these = geometric-mean baseline)
            (pl.col("vol") > 0).cast(pl.Float64).alias("_vpres"),
            pl.when(pl.col("vol") > 0).then(pl.col("vol").log()).otherwise(0.0).alias("_lv"),
        ).sort(BIN_TIME_COL, DATE_COL)
        # Trailing window of `window_days`, then shift(1) to exclude the day itself.
        grid = grid.with_columns(
            pl.col("vol").rolling_mean(window_days, min_samples=1).over(BIN_TIME_COL).alias("_vinc"),
            pl.col("_spr0").rolling_sum(window_days, min_samples=1).over(BIN_TIME_COL).alias("_ssum"),
            pl.col("_pres").rolling_sum(window_days, min_samples=1).over(BIN_TIME_COL).alias("_psum"),
            pl.col("_lv").rolling_sum(window_days, min_samples=1).over(BIN_TIME_COL).alias("_lvsum"),
            pl.col("_vpres").rolling_sum(window_days, min_samples=1).over(BIN_TIME_COL).alias("_vpsum"),
        ).with_columns(
            pl.col("_vinc").shift(1).over(BIN_TIME_COL).alias("V_k"),
            (pl.col("_ssum") / pl.col("_psum")).shift(1).over(BIN_TIME_COL).alias("s_k"),
            # L_k = mean of log(volume) over the window's traded days (null if none)
            pl.when(pl.col("_vpsum") > 0).then(pl.col("_lvsum") / pl.col("_vpsum"))
            .otherwise(None).shift(1).over(BIN_TIME_COL).alias("L_k"),
        )
        # Emit only the needed dates that have prior history (V_k not null).
        out = grid.filter(
            pl.col(DATE_COL).is_in(list(want_dates)) & pl.col("V_k").is_not_null()
        ).select(DATE_COL, BIN_TIME_COL, "s_k", "V_k", "L_k")
        for dt, bt, s_k, v_k, l_k in out.iter_rows():
            curves.setdefault((qc, dt), {})[bt] = (s_k, v_k, l_k)
    return curves


def _per_bin_curve(bins: pl.DataFrame, curves: dict) -> dict:
    """Look up the per-bin ``{bin_start_time: (s_k, V_k, L_k)}`` curve for these bins.

    Supports both curve flavours transparently: the trailing
    :func:`build_rolling_curves` keyed by ``(qcode, date)`` (preferred - no
    lookahead) and the static :func:`build_liq_spread_curves` keyed by
    ``security``. Returns ``{}`` when no curve exists (e.g. a qcode's first day),
    which makes the caller fall back to :func:`twap_schedule`.
    """
    if QCODE_COL in bins.columns and DATE_COL in bins.columns:
        key = (bins.get_column(QCODE_COL)[0], bins.get_column(DATE_COL)[0])
        hit = curves.get(key)
        if hit is not None:
            return hit
    return curves.get(bins.get_column(SECURITY_COL)[0], {})


def _fill_capacity(bins: pl.DataFrame):
    """Per-bin ``(fillable, cap)`` exactly as the fill model sees it.

    ``fillable`` is True iff ``volume``, ``vwap``, ``twa_ask``, ``twa_bid`` are all
    present and ``volume > 0`` (the only bins that can fill); ``cap = 2 * volume``
    is the most that can fill there. Used by the **dynamic** strategies to carry
    unfilled lots forward (track lots-still-to-fill rather than lots-requested).
    """
    vol = (bins.get_column(VOLUME_COL).cast(pl.Float64).fill_null(0.0).fill_nan(0.0).to_numpy())
    vwap = bins.get_column(VWAP_COL).cast(pl.Float64).to_numpy()
    ask = bins.get_column(ASK_COL).cast(pl.Float64).to_numpy()
    bid = bins.get_column(BID_COL).cast(pl.Float64).to_numpy()
    fillable = (vol > 0) & np.isfinite(vwap) & np.isfinite(ask) & np.isfinite(bid)
    return fillable, FILL_CAP_MULTIPLE * vol


def _carry_forward_exec(weights: np.ndarray, total: int, fillable: np.ndarray,
                        cap: np.ndarray, last_idx: int) -> np.ndarray:
    """Execute fraction-of-remaining ``weights`` against lots-still-to-FILL.

    At each bin request ``round(weights_k * remaining)`` lots (the last allocatable
    bin ``last_idx`` sweeps up all that's left), then decrement ``remaining`` by the
    **actual** causal fill ``min(request, cap_k)`` (0 in a non-fillable bin) - so a
    slice that doesn't fill rolls forward into later bins. Whole lots throughout;
    no lookahead (bin k uses only its own ``remaining``, which reflects fills < k).
    """
    n = len(weights)
    sched = np.zeros(n, dtype=float)
    remaining = float(total)
    for k in range(n):
        if remaining <= 0.5:
            break
        q = remaining if k == last_idx else min(float(round(weights[k] * remaining)), remaining)
        sched[k] = q
        if fillable[k]:
            remaining -= min(q, cap[k])
    return sched


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

    per_bin = _per_bin_curve(bins, curves)
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

    per_bin = _per_bin_curve(bins, curves)
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


def vwap_adaptive(bins: pl.DataFrame, quantity: float, *,
                  curves: dict, rho: object = 0.5, recent: int = 3,
                  clip_s: float = 1.5, **_: object) -> pl.Series:
    """Adaptive (receding-horizon) VWAP that reacts to *realised* volume - **no lookahead**.

    Works entirely in **deseasonalised log space**, anchored on the trailing-window
    **mean of log volume** ``L_k`` (so the baseline ``exp(L_k)`` is the *geometric*
    mean volume, the mean of the logs - **not** the log of the arithmetic mean, and
    the same baseline convention as :func:`vwap_factor`). ``L_k`` comes from
    :func:`build_rolling_curves` (lookahead-free); as the day unfolds the schedule
    tilts toward bins it now expects to be busier via the short-horizon **volume
    clustering** in the residual.

    At the start of bin ``k`` (using only bins ``< k``):

    1. **Surprise** = the **mean** of the per-bin log-deviations over the last
       ``recent`` *traded* bins (the residual in log space)::

           s_k = mean_{j in [k-recent, k), vol_j > 0} ( log(vol_j) - L_j )

       (``s_0 = 0``; clipped to ``+/- clip_s`` for safety).
    2. **Forecast** every remaining bin, surprise decaying by horizon, exponentiated
       back to the volume scale on the geometric baseline::

           v_hat[f] = exp(L_f) * exp(rho ** (f - k + 1) * s_k),   f >= k

       Far bins (``rho**h -> 0``) revert to ``exp(L_f)`` - the static geometric
       profile - so this degrades gracefully when there is no signal (and at the
       open). (A 2-tuple curve without ``L_k`` falls back to ``log(V_k)``.)
    3. **Allocate** in proportion to ``v_hat`` over the remaining bins. The
       weights ``a_k = v_hat_k / sum_{f>=k} v_hat_f`` are the *proportion of
       lots-still-to-fill* to request in bin ``k``; bin ``k``'s request is
       ``round(a_k * remaining)`` whole lots, and ``remaining`` is then decremented
       by the **actually filled** lots (``min(request, 2*volume)`` on fillable
       bins) - so a slice the ``2*volume`` cap or a dead bin failed to fill rolls
       **forward** into the next bin rather than being lost (carry-forward).

    This is a **dynamic** strategy: the schedule is driven by realised fills, so
    carry-forward applies. The hazard weights ``a_k`` are the causal telescoping
    form and depend only on ``s_k`` (hence on ``volume[< k]``): bin ``k``'s
    *weight* is invariant to any change in ``volume[k:]`` (see the causality
    self-test in the notebook). Whole-lot integrality comes from rounding each
    committed slice (not a global largest-remainder pass, which could let a later
    bin flip an earlier bin's lots). The last active bin receives all lots still
    outstanding so the order is fully scheduled.

    Parameters
    ----------
    bins : pl.DataFrame
        Bins for one (security, date); needs ``security``, ``bin_start_time``,
        ``volume`` and (for the rolling key) ``qcode`` + ``publication_date``.
        Time-sorted by the caller.
    quantity : float
        Total lots to execute (rounded to whole lots).
    curves : dict
        ``{(qcode, date): {bin_start_time: (s_k, V_k, L_k)}}`` from
        :func:`build_rolling_curves` (only the mean-of-log ``L_k`` is used). The
        security-keyed :func:`build_liq_spread_curves` form also works via the
        shared lookup, but then the baseline is static rather than trailing.
    rho : float or dict, default 0.5
        AR(1) persistence of the deseasonalised residual. A float applies to all;
        a dict is looked up by ``qcode`` (then ``(qcode, date)``), falling back to
        0.5. ``rho`` is a fixed structural constant (not fitted to the executed
        day), so it introduces no lookahead.
    recent : int, default 3
        Number of trailing bins averaged into the surprise ``s_k`` (only traded
        bins, ``vol_j > 0``, contribute).
    clip_s : float, default 1.5
        Symmetric clip on ``s_k`` (in log units) to bound the tilt.
    **_ :
        Ignored (uniform calling convention; absorbs ``side``).

    Returns
    -------
    pl.Series
        Integer-valued float series of length ``len(bins)``. Whole lots; because
        unfilled lots **carry forward** and get re-requested, the requested total
        is ``>= round(quantity)`` (it converges so realised fills target the full
        order). Falls back to :func:`twap_schedule` when no trailing profile exists
        (e.g. a qcode's first day).
    """
    n = bins.height
    if n == 0:
        return pl.Series("q_requested", [], dtype=pl.Float64)
    total_lots = int(round(float(quantity)))
    if total_lots <= 0:
        return pl.Series("q_requested", [0.0] * n, dtype=pl.Float64)

    per_bin = _per_bin_curve(bins, curves)
    times = bins.get_column(BIN_TIME_COL).to_list()
    trips = [per_bin.get(t) for t in times]
    Vk = np.array([(c[1] if c is not None else 0.0) for c in trips], dtype=float)
    # mean-of-log baseline L_k (rolling curves provide it as the 3rd element); for an
    # older 2-tuple curve fall back to log of the arithmetic mean so it still runs.
    Lk = np.array([(c[2] if (c is not None and len(c) >= 3) else np.nan) for c in trips],
                  dtype=float)
    Vk = np.clip(np.nan_to_num(Vk, nan=0.0), 0.0, None)
    Lbase = np.where(np.isfinite(Lk), Lk,
                     np.where(Vk > 0, np.log(np.where(Vk > 0, Vk, 1.0)), np.nan))
    active = np.isfinite(Lbase)
    if not active.any():
        # No trailing profile (e.g. qcode's first day) -> even-split TWAP.
        return twap_schedule(bins, total_lots)
    # baseline volume = GEOMETRIC mean exp(L_k) (mean of logs, not log of the mean)
    base = np.where(active, np.exp(np.clip(Lbase, -50.0, 50.0)), 0.0)

    # resolve rho (fixed structural constant; never the executed day's data)
    if isinstance(rho, dict):
        qc = bins.get_column(QCODE_COL)[0] if QCODE_COL in bins.columns else None
        dt = bins.get_column(DATE_COL)[0] if DATE_COL in bins.columns else None
        r = rho.get((qc, dt), rho.get(qc, 0.5))
    else:
        r = float(rho)
    r = min(max(float(r), 0.0), 0.99)

    # realised volume, used STRICTLY causally (bin k sees only volume[:k])
    vr = bins.get_column(VOLUME_COL).cast(pl.Float64).fill_null(0.0).fill_nan(0.0).to_numpy()

    # surprise s_k = MEAN of the per-bin log-deviations log(vol_j) - L_j over the last
    # `recent` traded bins (< k) - the deseasonalised residual in log space, matching
    # the geometric baseline (not a pooled log-of-summed-volume ratio).
    with np.errstate(divide="ignore", invalid="ignore"):
        dev = np.where((vr > 0) & active, np.log(np.where(vr > 0, vr, 1.0)) - Lbase, np.nan)
    k_idx = np.arange(n)
    lo = np.maximum(0, k_idx - int(recent))
    valid_dev = np.isfinite(dev).astype(float)
    devf = np.where(np.isfinite(dev), dev, 0.0)
    cum_d = np.concatenate([[0.0], np.cumsum(devf)])
    cum_n = np.concatenate([[0.0], np.cumsum(valid_dev)])
    num = cum_d[k_idx] - cum_d[lo]
    den = cum_n[k_idx] - cum_n[lo]
    s = np.where(den > 0, num / np.where(den > 0, den, 1.0), 0.0)
    s[0] = 0.0                                          # no history at the open
    s = np.clip(s, -clip_s, clip_s)

    # v_hat[k, f] = exp(L_f) * exp(rho**(f-k+1) * s[k]) for f >= k  (lower triangle = 0)
    D = k_idx[None, :] - k_idx[:, None]                 # D[k, f] = f - k
    valid = D >= 0
    # exponent >= 1 only on the valid (upper-triangle) entries; clamp elsewhere so
    # r ** (neg) is never evaluated (avoids 0**neg warnings); masked out anyway.
    decay = np.where(valid, r ** np.where(valid, D + 1, 0), 0.0)
    vhat = base[None, :] * np.exp(decay * s[:, None]) * valid
    denom = vhat.sum(axis=1)
    a = np.where(denom > 0, np.diag(vhat) / np.where(denom > 0, denom, 1.0), 0.0)
    a = np.clip(a, 0.0, 1.0)

    # --- causal carry-forward execution (DYNAMIC strategy) ---
    # a_k is the fraction of the *remaining* lots to execute at bin k. Run it
    # against lots-still-to-FILL: an unfilled slice (dataless bin / 2*V cap) rolls
    # into later bins rather than being lost, so the order actually completes (up to
    # capacity). The last allocatable bin sweeps up the residual. Still causal -
    # bin k only uses `remaining`, which reflects fills strictly before k.
    fillable, cap = _fill_capacity(bins)
    last_idx = int(np.max(np.where(a > 0))) if np.any(a > 0) else (n - 1)
    sched = _carry_forward_exec(a, total_lots, fillable, cap, last_idx)
    return pl.Series("q_requested", sched, dtype=pl.Float64)


def _build_window_models(history: pl.DataFrame, needed: pl.DataFrame, *,
                         window_days: int, fit_fn, value: str = "volume") -> dict:
    """Trailing-window driver shared by the per-order volume/spread models (lookahead-free).

    For each ``(qcode, date)`` in ``needed`` it slices that qcode's previous
    ``window_days`` trading days (the dense (date x bin) grid, **strictly before**
    the date - the day itself is never in the window) and calls
    ``fit_fn(window_mat, bintimes)``; non-``None`` results are keyed by
    ``(qcode, date)``. Pooled at the qcode level (intraday shape is a qcode property).

    ``value`` selects what the ``(D_days, N_bins)`` matrix holds, so the same MLE /
    OLS ``fit_fn`` can model either quantity's intraday **log-deviations**:

    * ``"volume"`` (default) - the qcode's **summed** volume per ``(date, bin)``;
    * ``"spread"``           - the **volume-weighted mean** spread ``twa_ask - twa_bid``
      per ``(date, bin)`` (same daily aggregation as :func:`build_rolling_curves`).

    Both feed :func:`_fit_factor_model` / :func:`_fit_ols_model` unchanged: those treat
    a matrix entry ``> 0`` as an observed value to log-deseasonalise and ``<= 0`` /
    missing as a no-observation day (imputed at the bin's log-mean).
    """
    date_col = "date" if "date" in needed.columns else DATE_COL
    want: dict = {}
    for qc, dt in needed.select(QCODE_COL, date_col).unique().iter_rows():
        want.setdefault(qc, set()).add(dt)

    if value == "spread":
        h = (history.select(QCODE_COL, DATE_COL, BIN_TIME_COL, VOLUME_COL, ASK_COL, BID_COL)
             .filter(pl.col(QCODE_COL).is_in(list(want.keys())))
             .with_columns(pl.col(VOLUME_COL).cast(pl.Float64).fill_null(0.0).alias("_v"),
                           (pl.col(ASK_COL) - pl.col(BID_COL)).alias("_spr")))
        # volume-weighted mean spread per (qcode, date, bin) (fall back to plain mean
        # on zero-volume days); non-positive/absent spreads are dropped downstream.
        daily = h.group_by(QCODE_COL, DATE_COL, BIN_TIME_COL).agg(
            pl.when(pl.col("_v").sum() > 0)
            .then((pl.col("_spr") * pl.col("_v")).sum() / pl.col("_v").sum())
            .otherwise(pl.col("_spr").mean())
            .alias("vol"))
    else:
        h = (history.select(QCODE_COL, DATE_COL, BIN_TIME_COL, VOLUME_COL)
             .filter(pl.col(QCODE_COL).is_in(list(want.keys()))))
        daily = h.group_by(QCODE_COL, DATE_COL, BIN_TIME_COL).agg(
            pl.col(VOLUME_COL).cast(pl.Float64).fill_null(0.0).sum().alias("vol"))

    curves: dict = {}
    for qc, want_dates in want.items():
        d = daily.filter(pl.col(QCODE_COL) == qc)
        if d.height == 0:
            continue
        piv = (d.pivot(on=BIN_TIME_COL, index=DATE_COL, values="vol",
                       aggregate_function="sum").sort(DATE_COL))
        dts = piv.get_column(DATE_COL).to_list()
        bintimes = [c for c in piv.columns if c != DATE_COL]
        vol = np.nan_to_num(piv.drop(DATE_COL).to_numpy().astype(float), nan=0.0)
        pos_of = {dt: i for i, dt in enumerate(dts)}
        for t in want_dates:
            i = pos_of.get(t)
            if i is None:
                continue
            w = vol[max(0, i - window_days):i, :]          # days t-window_days .. t-1
            model = fit_fn(w, bintimes)
            if model is not None:
                curves[(qc, t)] = model
    return curves


def _fit_factor_model(window_vol: np.ndarray, bintimes: Sequence,
                      n_factors: int) -> Optional[dict]:
    """Fit an MLE ``n_factors``-factor model of intraday **log-volume deviations**.

    ``window_vol`` is the trailing window's daily volume per bin, shape
    ``(D_days, N_bins)`` (``0`` = no trade / missing). Per bin ``k`` we form the
    seasonal log-mean ``L_k = mean_d log(vol_{d,k})`` over days that traded, and the
    deviation ``M_{d,k} = log(vol_{d,k}) - L_k`` (imputed ``0`` - i.e. *at the mean*
    - on no-trade days). The cross-sectional covariance of ``M`` is modelled as
    ``Sigma ~ lambda lambda^T + diag(sigma_u^2)`` and fit by **maximum-likelihood
    factor analysis** (the estimator that matches this exact structure; rotation is
    irrelevant here - the downstream forecast ``lambda_m^T mu`` is invariant to any
    orthogonal rotation of the factors). The loadings ``lambda`` and idiosyncratic
    variances ``sigma_u^2`` are what :func:`vwap_factor` filters on.

    Returns ``{"times", "L", "lam" (N,K), "su2" (N), "active", "n_factors"}`` aligned
    to ``bintimes``, or ``None`` if there is too little usable history. Bins that
    seldom trade (``< 2`` days) are marked inactive (``lam = 0``, ``su2 = NaN``).
    """
    from sklearn.decomposition import FactorAnalysis      # lazy: only this path needs sklearn

    pos = window_vol > 0
    cnt = pos.sum(axis=0)
    active = cnt >= 2                                      # need >= 2 obs for a deviation
    na = int(active.sum())
    if na < 2 or window_vol.shape[0] < 3:                  # too few bins / days to fit
        return None
    logv = np.zeros_like(window_vol, dtype=float)
    np.log(window_vol, out=logv, where=pos)               # log only where vol > 0
    L = np.full(window_vol.shape[1], -np.inf)
    L[active] = logv[:, active].sum(axis=0) / cnt[active]  # seasonal log-mean per active bin
    M = np.where(pos, logv - np.where(np.isfinite(L), L, 0.0)[None, :], 0.0)[:, active]
    K = max(1, min(int(n_factors), na, window_vol.shape[0] - 1))
    try:
        fa = FactorAnalysis(n_components=K, max_iter=2000, random_state=0).fit(M)
        lam_a = fa.components_.T                           # (na, K): Cov ~ lam lam^T + diag(su2)
        su2_a = np.maximum(fa.noise_variance_, 1e-10)
    except Exception:
        return None
    N = window_vol.shape[1]
    lam = np.zeros((N, K), dtype=float)
    su2 = np.full(N, np.nan)
    lam[active] = lam_a
    su2[active] = su2_a
    return {"times": list(bintimes), "L": L, "lam": lam, "su2": su2,
            "active": active, "n_factors": K}


def _fit_ols_model(window_vol: np.ndarray, bintimes: Sequence) -> Optional[dict]:
    """Per-pair OLS sufficient stats of intraday **log-volume deviations**.

    Builds ``M_{d,k} = log(vol_{d,k}) - L_k`` (mean-of-logs baseline ``L_k``, imputed
    ``0`` on no-trade days, like :func:`_fit_factor_model`) over the window's active
    bins, and stores just the **cross-moment matrix** ``C = M^T M`` (via ``M``) and the
    day count ``D``. From these :func:`adaptive_volume_ols` derives, for any pair
    ``(k, m)``, the no-intercept regression of ``M_m`` on ``M_k``::

        beta_{m|k}      = C[k, m] / C[k, k]                          (= rho * sigma_m / sigma_k)
        sigma2_{m|k}    = (C[m, m] - C[k, m]^2 / C[k, k]) / (D - 1)  (= sigma_m^2 (1 - rho^2))
        Var(beta_{m|k}) = sigma2_{m|k} / C[k, k]
        pred var r2     = sigma2_{m|k} + Var(beta_{m|k}) * M_k^2

    Returns ``{"times", "L", "active", "M" (D, na), "D"}`` (``M`` kept rather than
    ``C`` - smaller when ``D < N`` - and ``C`` is reformed in the strategy), or
    ``None`` if there is too little usable history.
    """
    pos = window_vol > 0
    cnt = pos.sum(axis=0)
    active = cnt >= 2
    na = int(active.sum())
    D = window_vol.shape[0]
    if na < 2 or D < 3:
        return None
    logv = np.zeros_like(window_vol, dtype=float)
    np.log(window_vol, out=logv, where=pos)
    L = np.full(window_vol.shape[1], -np.inf)
    L[active] = logv[:, active].sum(axis=0) / cnt[active]
    M = np.where(pos, logv - np.where(np.isfinite(L), L, 0.0)[None, :], 0.0)[:, active]
    return {"times": list(bintimes), "L": L, "active": active, "M": M, "D": int(D)}


def build_factor_curves(history: pl.DataFrame, needed: pl.DataFrame, *,
                        window_days: int = 22, n_factors: int = 1) -> dict:
    """Per-order **factor model** of intraday log-volume deviations (lookahead-free).

    For every ``(qcode, date)`` order key in ``needed`` this fits an
    ``n_factors``-factor model (see :func:`_fit_factor_model`) on **only** that
    qcode's previous ``window_days`` trading days, **strictly before** the order
    date (day ``t`` uses days ``t-window_days .. t-1`` - the same day is never in the
    window). The decomposition is done **once per order** and consumed live by
    :func:`vwap_factor`.

    The profile is built at the **qcode** level (the intraday shape is a
    qcode/exchange property), pooling that qcode's contract months' volume per
    ``(date, bin_start_time)``.

    Parameters
    ----------
    history : pl.DataFrame
        Full bin history (``qcode``, ``publication_date``, ``bin_start_time``,
        ``volume``).
    needed : pl.DataFrame
        Order keys to emit models for; needs ``qcode`` and a date column
        (``date`` or ``publication_date``).
    window_days : int, default 22
        Trailing trading-day window (per qcode), excluding the order day.
    n_factors : int, default 1
        Number of latent factors ``K`` (the parameter to vary in the notebook). The
        actual ``K`` is capped at the usable bin/day count per order.

    Returns
    -------
    dict
        ``{(qcode, date): factor_model_dict}`` for keys with enough prior history;
        keys without it are omitted, so :func:`vwap_factor` falls back to TWAP.
    """
    return _build_window_models(history, needed, window_days=window_days,
                                fit_fn=lambda w, bt: _fit_factor_model(w, bt, n_factors))


def build_spread_factor_curves(history: pl.DataFrame, needed: pl.DataFrame, *,
                               window_days: int = 22, n_factors: int = 1) -> dict:
    """Per-order **factor model** of intraday log-**spread** deviations (lookahead-free).

    The spread analogue of :func:`build_factor_curves`: for every ``(qcode, date)``
    order key in ``needed`` it fits an ``n_factors``-factor model (see
    :func:`_fit_factor_model`) to the intraday **log-deviations of the bid-ask spread**
    ``s = twa_ask - twa_bid`` over **only** that qcode's previous ``window_days``
    trading days, strictly before the order date (the same day is never in the window).
    The daily per-bin spread is the volume-weighted mean across the qcode's contract
    months (as in :func:`build_rolling_curves`); bins that seldom quote a positive
    spread are marked inactive.

    Same estimator and return shape as :func:`build_factor_curves` - only the modelled
    quantity differs - so :func:`spread_factor_vol_ols` / :func:`spread_factor_vol_factor`
    run the identical sequential Kalman update on **spread** surprises to forecast each
    remaining bin's spread ``s_hat_m = exp(L_m + lambda_m . mu)``. Intraday spread has a
    much stronger single-factor structure than volume (the U-shaped liquidity curve is
    very stable day to day), so ``n_factors = 1`` already captures most instruments.

    Parameters
    ----------
    history : pl.DataFrame
        Full bin history (``qcode``, ``publication_date``, ``bin_start_time``,
        ``volume``, ``twa_ask``, ``twa_bid``).
    needed : pl.DataFrame
        Order keys to emit models for; needs ``qcode`` and a date column
        (``date`` or ``publication_date``).
    window_days : int, default 22
        Trailing trading-day window (per qcode), excluding the order day.
    n_factors : int, default 1
        Number of latent **spread** factors ``K`` (capped at the usable bin/day count).

    Returns
    -------
    dict
        ``{(qcode, date): factor_model_dict}`` of log-spread models for keys with enough
        prior history; keys without it are omitted, so the consuming strategies fall
        back to TWAP.
    """
    return _build_window_models(history, needed, window_days=window_days, value="spread",
                                fit_fn=lambda w, bt: _fit_factor_model(w, bt, n_factors))


def build_ols_curves(history: pl.DataFrame, needed: pl.DataFrame, *,
                     window_days: int = 22) -> dict:
    """Per-order **pairwise-OLS** model of intraday log-volume deviations (lookahead-free).

    For every ``(qcode, date)`` order key in ``needed`` this stores the cross-moment
    sufficient statistics (see :func:`_fit_ols_model`) of the deseasonalised
    log-volume deviations over **only** that qcode's previous ``window_days`` trading
    days, strictly before the order date. :func:`adaptive_volume_ols` then forms, at
    each step, the backward regression of every future bin ``m`` on the most recent
    observed bin ``k`` and forecasts ``M_m | M_k`` (the Markov / "Option A" form).

    Parameters
    ----------
    history : pl.DataFrame
        Full bin history (``qcode``, ``publication_date``, ``bin_start_time``,
        ``volume``).
    needed : pl.DataFrame
        Order keys to emit models for; needs ``qcode`` and a date column.
    window_days : int, default 22
        Trailing trading-day window (per qcode), excluding the order day.

    Returns
    -------
    dict
        ``{(qcode, date): ols_model_dict}`` for keys with enough prior history; keys
        without it are omitted, so :func:`adaptive_volume_ols` falls back to TWAP.
    """
    return _build_window_models(history, needed, window_days=window_days,
                                fit_fn=_fit_ols_model)


def certainty_equivalent_volume(v_hat: object, log_var: object, *,
                                coef: float = 0.25) -> np.ndarray:
    r"""Confidence-adjust a volume forecast by its **log-volume forecast variance**.

    Under square-root market impact the execution cost is convex in volume
    (``cost ~ V^{-1/2}``), so feeding a *point* volume forecast into a schedule is
    optimistic - a bin whose volume is uncertain should be trusted less. Taking the
    expectation over a log-normal volume ``V = exp(m + r * eps)`` and matching the
    certainty-equivalent ``E[V^{-1/2}]`` gives::

        V_tilde_k = V_hat_k * exp(-coef * r_k^2),     coef = 1/4

    where ``V_hat_k = exp(m_k)`` is the point (median) forecast and ``r_k^2`` is the
    forecast **variance of log-volume** for bin ``k``. ``coef = 1/4`` is exact for the
    ``V^{-1/2}`` cost here (derived from the log-normal MGF ``E[V^{-1/2}] =
    exp(-m/2 + r^2/8)``); it is exposed for other impact curves. High certainty
    (``r_k^2 -> 0``) leaves the forecast unchanged; high uncertainty shrinks it.

    Reusable across any model that produces a per-bin log-volume forecast variance
    (e.g. :func:`vwap_factor`, where ``r_k^2`` is the predictive diagonal
    ``lambda_k^T P lambda_k + sigma_{u,k}^2``).

    Parameters
    ----------
    v_hat : array-like
        Point volume forecast(s) ``V_hat_k`` (``>= 0``).
    log_var : array-like
        Forecast variance of **log**-volume ``r_k^2`` (``>= 0``), same shape.
    coef : float, default 0.25
        Penalty coefficient (``1/4`` for square-root impact).

    Returns
    -------
    np.ndarray
        Penalised ("certainty-equivalent") volume ``V_tilde_k``.

    Examples
    --------
    >>> round(float(certainty_equivalent_volume(100.0, 0.0)), 6)    # no uncertainty
    100.0
    >>> round(float(certainty_equivalent_volume(100.0, 4.0)), 6)    # 100 * exp(-1)
    36.787944
    """
    return np.asarray(v_hat, dtype=float) * np.exp(-coef * np.asarray(log_var, dtype=float))


def vwap_factor(bins: pl.DataFrame, quantity: float, *,
                curves: dict, ce_adjust: bool = True, **_: object) -> pl.Series:
    """Dynamic K-factor VWAP via a sequential Kalman update on log-volume surprises.

    Treats the day's latent volume **factor(s)** ``F`` as a state with a
    standard-normal prior and sequentially updates the belief ``(mu, P)`` as each
    bin's volume is realised, then re-forecasts the *remaining* bins and participates
    in proportion to the forecast - the factor-model analogue of
    :func:`vwap_adaptive` (which uses a single AR(1) surprise). The loadings
    ``lambda_k`` and idiosyncratic variances ``sigma_{u,k}^2`` come from
    :func:`build_factor_curves` (trailing ``window_days``, lookahead-free).

    With per-bin log-volume deviation ``M_k = log(vol_k) - L_k``, at the start of bin
    ``k`` the belief reflects bins ``< k`` only (causal). For each just-completed bin
    the standard measurement update is applied (``H = lambda_k``, ``R =
    sigma_{u,k}^2``)::

        y_k = M_k - lambda_k . mu                 # surprise
        S_k = lambda_k . P lambda_k + sigma_u^2   # innovation variance (scalar)
        Kg  = P lambda_k / S_k                    # Kalman gain (K-vector)
        mu  = mu + Kg * y_k
        P   = (I - Kg lambda_k^T) P

    The remaining bins are forecast as ``vol_hat_m = exp(L_m + lambda_m . mu)`` and
    (when ``ce_adjust``) **confidence-adjusted** by their predictive log-volume
    variance via :func:`certainty_equivalent_volume`::

        r_m^2       = lambda_m^T P lambda_m + sigma_{u,m}^2    # predictive log-var (current P)
        vol_tilde_m = vol_hat_m * exp(-r_m^2 / 4)             # certainty-equivalent volume

    so bins whose volume is less certain are trusted less (square-root impact makes
    cost convex in volume). The factor-uncertainty term ``lambda_m^T P lambda_m``
    shrinks as bins are observed (``P`` falls), so the penalty relaxes through the
    day. Bin ``k`` then requests its share ``vol_tilde_k / sum_{m>=k} vol_tilde_m``
    of the lots **still to fill**. For ``K = 1`` the update is exactly the scalar
    spec; for ``K > 1`` the forecast is invariant to factor rotation, so no rotation
    handling is needed.

    This is a **dynamic** strategy: it **carries forward** unfilled lots (``remaining``
    is decremented by the actual fill, an unfilled slice rolls into later bins, and
    the last active bin sweeps up the residual). Whole lots throughout; strictly
    causal (bin ``k`` uses only volume ``< k``). Falls back to :func:`twap_schedule`
    when no factor model exists for the (qcode, date) or nothing is tradeable.

    Parameters
    ----------
    bins : pl.DataFrame
        Bins for one (security, date) with ``qcode``, ``publication_date``,
        ``bin_start_time`` and ``volume``. Time-sorted by the caller.
    quantity : float
        Total lots (rounded to whole lots).
    curves : dict
        ``{(qcode, date): factor_model}`` from :func:`build_factor_curves`, via
        ``strategy_params``. The number of factors is fixed at build time.
    ce_adjust : bool, default True
        Apply the certainty-equivalent volume penalty ``exp(-r_m^2 / 4)`` (see
        :func:`certainty_equivalent_volume`) before forming the schedule. ``False``
        allocates on the raw point forecast.

    Returns
    -------
    pl.Series
        Integer-valued float series of length ``len(bins)``; whole lots, with
        unfilled lots carried forward (requested total ``>= round(quantity)``).
    """
    n = bins.height
    if n == 0:
        return pl.Series("q_requested", [], dtype=pl.Float64)
    total_lots = int(round(float(quantity)))
    if total_lots <= 0:
        return pl.Series("q_requested", [0.0] * n, dtype=pl.Float64)

    model = None
    if QCODE_COL in bins.columns and DATE_COL in bins.columns:
        model = curves.get((bins.get_column(QCODE_COL)[0], bins.get_column(DATE_COL)[0]))
    if model is None:
        return twap_schedule(bins, total_lots)

    idx = {t: j for j, t in enumerate(model["times"])}
    sel = [idx.get(t, -1) for t in bins.get_column(BIN_TIME_COL).to_list()]
    Kf = int(model["n_factors"])
    L = np.array([model["L"][j] if j >= 0 else -np.inf for j in sel], dtype=float)
    lam = np.array([model["lam"][j] if j >= 0 else np.zeros(Kf) for j in sel], dtype=float)
    su2 = np.array([model["su2"][j] if j >= 0 else np.nan for j in sel], dtype=float)
    active = np.isfinite(su2) & (su2 > 0) & np.isfinite(L)
    if not active.any():
        return twap_schedule(bins, total_lots)

    base = np.where(active, np.exp(np.clip(L, -50.0, 50.0)), 0.0)   # seasonal volume exp(L_k)
    vr = (bins.get_column(VOLUME_COL).cast(pl.Float64)
          .fill_null(0.0).fill_nan(0.0).to_numpy())
    fillable, cap = _fill_capacity(bins)
    last_idx = int(np.max(np.where(active)))

    mu = np.zeros(Kf)
    P = np.eye(Kf)
    sched = np.zeros(n, dtype=float)
    remaining = float(total_lots)
    for k in range(n):
        # sequential measurement update with the just-completed bin (k-1): causal
        if k >= 1 and active[k - 1] and vr[k - 1] > 0:
            y = (math.log(vr[k - 1]) - L[k - 1]) - float(lam[k - 1] @ mu)
            Pl = P @ lam[k - 1]
            S = float(lam[k - 1] @ Pl + su2[k - 1])
            if S > 0:
                Kg = Pl / S
                mu = mu + Kg * y
                P = P - np.outer(Kg, Pl)
        if remaining <= 0.5:
            break
        if k == last_idx:
            q = remaining                                  # sweep up the residual
        elif not active[k]:
            q = 0.0
        else:
            vhat = base * np.exp(np.clip(lam @ mu, -50.0, 50.0))   # point volume forecast
            if ce_adjust:
                # certainty-equivalent penalty using the PREDICTIVE log-volume
                # variance r2 = lam_m^T P lam_m + su2 (current P -> shrinks over the day)
                r2 = ((lam @ P) * lam).sum(axis=1) + np.nan_to_num(su2, nan=0.0)
                vhat = certainty_equivalent_volume(vhat, r2)
            fcast = np.where(active, vhat, 0.0)
            denom = float(fcast[k:].sum())
            w = fcast[k] / denom if denom > 0 else 0.0
            q = min(float(round(w * remaining)), remaining)
        sched[k] = q
        if fillable[k]:
            remaining -= min(q, cap[k])
    return pl.Series("q_requested", sched, dtype=pl.Float64)


def adaptive_volume_ols(bins: pl.DataFrame, quantity: float, *,
                        curves: dict, ce_adjust: bool = True, **_: object) -> pl.Series:
    """Dynamic VWAP via per-pair **OLS** forecasts of log-volume (Markov / Option A).

    A lighter-weight cousin of :func:`vwap_factor`: instead of a factor Kalman filter
    it uses, for each future bin ``m``, the **backward linear regression** of ``M_m``
    on the most recently observed bin's deviation ``M_k`` (the Markov assumption
    ``M_m | M_1..M_k = M_m | M_k``), estimated on the trailing window by
    :func:`build_ols_curves`. With cross-moments ``C = M^T M`` and ``D`` window days::

        beta_{m|k}   = C[k, m] / C[k, k]                            # = rho * sigma_m/sigma_k
        sigma2_{m|k} = (C[m, m] - C[k, m]^2 / C[k, k]) / (D - 1)    # = sigma_m^2 (1 - rho^2)
        M_m | M_k ~ N( beta_{m|k} M_k ,  sigma2_{m|k} + (sigma2_{m|k}/C[k,k]) * M_k^2 )

    The prediction variance ``r2`` adds the slope-estimation term
    ``Var(beta) * M_k^2`` (so a large/extrapolated surprise is trusted less). Each
    remaining bin is forecast ``vol_hat_m = exp(L_m + beta_{m|k} M_k)``, then (when
    ``ce_adjust``) shrunk by the convexity penalty
    ``vol_tilde_m = vol_hat_m * exp(-r2_m / 4)`` via
    :func:`certainty_equivalent_volume`, and the schedule participates in proportion
    to ``vol_tilde`` over the lots **still to fill**.

    Conditions on the **latest completed traded bin** (re-planned every bin), so it is
    strictly causal; before any bin has traded it uses the unconditional baseline
    (``M_hat = 0``, ``r2 = sigma_m^2``). **Dynamic** strategy - carries forward unfilled
    lots, whole lots throughout. Falls back to :func:`twap_schedule` when there is no
    OLS model for the (qcode, date) or nothing is tradeable.

    Parameters
    ----------
    bins : pl.DataFrame
        Bins for one (security, date) with ``qcode``, ``publication_date``,
        ``bin_start_time`` and ``volume``. Time-sorted by the caller.
    quantity : float
        Total lots (rounded to whole lots).
    curves : dict
        ``{(qcode, date): ols_model}`` from :func:`build_ols_curves`, via
        ``strategy_params``.
    ce_adjust : bool, default True
        Apply the certainty-equivalent volume penalty ``exp(-r2_m / 4)`` before
        scheduling (see :func:`certainty_equivalent_volume`).

    Returns
    -------
    pl.Series
        Integer-valued float series of length ``len(bins)``; whole lots, unfilled
        lots carried forward.
    """
    n = bins.height
    if n == 0:
        return pl.Series("q_requested", [], dtype=pl.Float64)
    total_lots = int(round(float(quantity)))
    if total_lots <= 0:
        return pl.Series("q_requested", [0.0] * n, dtype=pl.Float64)

    model = None
    if QCODE_COL in bins.columns and DATE_COL in bins.columns:
        model = curves.get((bins.get_column(QCODE_COL)[0], bins.get_column(DATE_COL)[0]))
    if model is None:
        return twap_schedule(bins, total_lots)

    C = model["M"].T @ model["M"]                          # (na, na) cross-moments M^T M
    diagC = np.diag(C).copy()
    D = int(model["D"])
    active_full = np.where(model["active"])[0]             # model-bin indices, in C-column order
    Lc = model["L"][active_full]                           # mean-of-log baseline per C column
    colpos = {int(fi): a for a, fi in enumerate(active_full)}

    idx = {t: j for j, t in enumerate(model["times"])}
    sel = [idx.get(t, -1) for t in bins.get_column(BIN_TIME_COL).to_list()]
    L = np.array([model["L"][j] if j >= 0 else -np.inf for j in sel], dtype=float)
    col = np.array([colpos.get(j, -1) if j >= 0 else -1 for j in sel], dtype=int)  # day-bin -> C column
    active = (col >= 0) & np.isfinite(L)
    if not active.any():
        return twap_schedule(bins, total_lots)

    vr = (bins.get_column(VOLUME_COL).cast(pl.Float64)
          .fill_null(0.0).fill_nan(0.0).to_numpy())
    fillable, cap = _fill_capacity(bins)
    last_idx = int(np.max(np.where(active)))

    sched = np.zeros(n, dtype=float)
    remaining = float(total_lots)
    j_col = -1                                             # C-column of the latest traded bin
    m_obs = 0.0                                            # its realised deviation M_k
    for k in range(n):
        # condition on the just-completed bin (k-1): causal Markov update
        if k >= 1 and active[k - 1] and vr[k - 1] > 0:
            j_col = int(col[k - 1])
            m_obs = math.log(vr[k - 1]) - L[k - 1]
        if remaining <= 0.5:
            break
        if k == last_idx:
            q = remaining
        elif not active[k]:
            q = 0.0
        else:
            if j_col >= 0 and diagC[j_col] > 0:
                ck = diagC[j_col]
                beta = C[j_col, :] / ck                    # (na,)
                mhat = beta * m_obs
                sigma2 = np.maximum(diagC - C[j_col, :] ** 2 / ck, 0.0) / max(D - 1, 1)
                r2 = sigma2 + (sigma2 / ck) * m_obs ** 2
            else:                                          # no observation yet -> unconditional
                mhat = np.zeros(C.shape[0])
                r2 = diagC / max(D - 1, 1)
            vt = np.exp(np.clip(Lc + mhat, -50.0, 50.0))   # vol_hat_m = exp(L_m + beta*M_k)
            if ce_adjust:
                vt = certainty_equivalent_volume(vt, r2)
            fcast = np.zeros(n)
            fcast[active] = vt[col[active]]
            denom = float(fcast[k:].sum())
            w = fcast[k] / denom if denom > 0 else 0.0
            q = min(float(round(w * remaining)), remaining)
        sched[k] = q
        if fillable[k]:
            remaining -= min(q, cap[k])
    return pl.Series("q_requested", sched, dtype=pl.Float64)


def _align_factor_model(model: dict, times: Sequence):
    """Align a factor model's per-bin arrays to this day's bin order.

    Returns ``(K, L, lam (n,K), su2 (n), active (n))`` where ``active`` marks bins the
    model can forecast (finite baseline and positive idiosyncratic variance). Bins with
    no model entry get ``L=-inf``, ``lam=0``, ``su2=NaN`` (inactive). Shared by the
    spread Kalman (always) and the volume Kalman (factor variant).
    """
    idx = {t: j for j, t in enumerate(model["times"])}
    sel = [idx.get(t, -1) for t in times]
    K = int(model["n_factors"])
    L = np.array([model["L"][j] if j >= 0 else -np.inf for j in sel], dtype=float)
    lam = np.array([model["lam"][j] if j >= 0 else np.zeros(K) for j in sel], dtype=float)
    su2 = np.array([model["su2"][j] if j >= 0 else np.nan for j in sel], dtype=float)
    active = np.isfinite(su2) & (su2 > 0) & np.isfinite(L)
    return K, L, lam, su2, active


def _align_ols_model(model: dict, times: Sequence):
    """Align a pairwise-OLS model to this day's bin order (see :func:`adaptive_volume_ols`).

    Returns ``(C, diagC, D, Lc, L (n), col (n), active (n))`` where ``C = M^T M`` is the
    cross-moment matrix over the window's active bins, ``Lc`` their log-mean baselines,
    ``L`` the per-day-bin baseline and ``col`` maps each day bin to its column in ``C``
    (``-1`` if absent). ``active`` marks bins present in ``C`` with a finite baseline.
    """
    C = model["M"].T @ model["M"]
    diagC = np.diag(C).copy()
    D = int(model["D"])
    active_full = np.where(model["active"])[0]
    Lc = model["L"][active_full]
    colpos = {int(fi): a for a, fi in enumerate(active_full)}
    idx = {t: j for j, t in enumerate(model["times"])}
    sel = [idx.get(t, -1) for t in times]
    L = np.array([model["L"][j] if j >= 0 else -np.inf for j in sel], dtype=float)
    col = np.array([colpos.get(j, -1) if j >= 0 else -1 for j in sel], dtype=int)
    active = (col >= 0) & np.isfinite(L)
    return C, diagC, D, Lc, L, col, active


def _dyn_spread_vol_schedule(bins: pl.DataFrame, total_lots: int, spread_model: dict,
                             vol_model: dict, *, vol_kind: str, ce_adjust: bool) -> np.ndarray:
    """Shared engine for the dynamic spread x volume cost-minimising strategies.

    Runs, in lockstep down the day, a sequential Kalman filter on **log-spread**
    surprises (from ``spread_model``, always a factor model) and a **log-volume**
    forecaster (``vol_kind="factor"`` -> a second Kalman filter; ``"ols"`` -> the Markov
    pairwise-OLS of :func:`adaptive_volume_ols`). At each bin ``k`` it re-forecasts every
    remaining bin's spread ``s_hat_m`` and volume ``V_hat_m``, feeds them into the fill
    model's convex cost optimiser (:func:`_solve_mu`: ``q_m* = V_m (max(0, mu/(MARG_COEF
    s_m) - THRESH_RATIO))^2``) over the lots **still to fill**, and executes bin ``k``'s
    optimal share - the dynamic, spread-aware generalisation of :func:`liq_spr_static`.

    Both forecasts are strictly causal (bin ``k`` uses only bins ``< k``). When
    ``ce_adjust`` the point forecasts are replaced by their lognormal certainty
    equivalents before optimising:

    * **volume** enters the cost through ``V^{-1/2}`` (convex), so an uncertain bin is
      *shrunk*: ``V_tilde_m = V_hat_m * exp(-r2_v,m / 4)`` (as in :func:`vwap_factor`);
    * **spread** enters the cost **linearly**, so expected-cost minimisation uses the
      *mean* spread ``E[s] = exp(mu + r2/2)``, i.e. an uncertain bin is *inflated*:
      ``s_tilde_m = s_hat_m * exp(+r2_s,m / 2)`` - so the optimiser steers away from bins
      whose tight spread it is unsure about. (Log-spread variance is small - spread is
      far more stable intraday than volume - so this term is mild in practice.)

    Returns the per-bin whole-lot request array (unfilled lots carried forward; the last
    active bin sweeps the residual). Falls back to :func:`twap_schedule` if no bin has
    both a spread and a volume forecast.
    """
    n = bins.height
    times = bins.get_column(BIN_TIME_COL).to_list()

    # spread: always a factor model; volume: factor Kalman or pairwise-OLS
    Ks, Ls, lam_s, su2_s, active_s = _align_factor_model(spread_model, times)
    if vol_kind == "factor":
        Kv, Lv, lam_v, su2_v, active_v = _align_factor_model(vol_model, times)
    else:
        C, diagC, D, Lc, Lv, col, active_v = _align_ols_model(vol_model, times)

    active = active_s & active_v
    if not active.any():
        return twap_schedule(bins, total_lots).to_numpy()

    vr = (bins.get_column(VOLUME_COL).cast(pl.Float64).fill_null(0.0).fill_nan(0.0).to_numpy())
    sr = (bins.get_column(ASK_COL).cast(pl.Float64).to_numpy()
          - bins.get_column(BID_COL).cast(pl.Float64).to_numpy())     # realised spread
    fillable, cap = _fill_capacity(bins)
    last_idx = int(np.max(np.where(active)))
    idxrange = np.arange(n)

    mu_s = np.zeros(Ks); P_s = np.eye(Ks)
    base_s = np.where(active_s, np.exp(np.clip(Ls, -50.0, 50.0)), 0.0)
    if vol_kind == "factor":
        mu_v = np.zeros(Kv); P_v = np.eye(Kv)
        base_v = np.where(active_v, np.exp(np.clip(Lv, -50.0, 50.0)), 0.0)
    else:
        j_col = -1; m_obs = 0.0

    sched = np.zeros(n, dtype=float)
    remaining = float(total_lots)
    for k in range(n):
        # --- causal measurement updates from the just-completed bin (k-1) ---
        if k >= 1:
            if active_s[k - 1] and sr[k - 1] > 0:                     # spread factor update
                y = (math.log(sr[k - 1]) - Ls[k - 1]) - float(lam_s[k - 1] @ mu_s)
                Pl = P_s @ lam_s[k - 1]
                S = float(lam_s[k - 1] @ Pl + su2_s[k - 1])
                if S > 0:
                    Kg = Pl / S
                    mu_s = mu_s + Kg * y
                    P_s = P_s - np.outer(Kg, Pl)
            if vol_kind == "factor":                                  # volume factor update
                if active_v[k - 1] and vr[k - 1] > 0:
                    y = (math.log(vr[k - 1]) - Lv[k - 1]) - float(lam_v[k - 1] @ mu_v)
                    Pl = P_v @ lam_v[k - 1]
                    S = float(lam_v[k - 1] @ Pl + su2_v[k - 1])
                    if S > 0:
                        Kg = Pl / S
                        mu_v = mu_v + Kg * y
                        P_v = P_v - np.outer(Kg, Pl)
            elif active_v[k - 1] and vr[k - 1] > 0:                   # volume OLS conditioning bin
                j_col = int(col[k - 1])
                m_obs = math.log(vr[k - 1]) - Lv[k - 1]
        if remaining <= 0.5:
            break
        if k == last_idx:
            q = remaining                                            # sweep up the residual
        elif not active[k]:
            q = 0.0
        else:
            # spread forecast + predictive log-variance for every bin (current belief)
            shat = base_s * np.exp(np.clip(lam_s @ mu_s, -50.0, 50.0))
            r2s = ((lam_s @ P_s) * lam_s).sum(axis=1) + np.nan_to_num(su2_s, nan=0.0)
            # volume forecast + predictive log-variance
            if vol_kind == "factor":
                vhat = base_v * np.exp(np.clip(lam_v @ mu_v, -50.0, 50.0))
                r2v = ((lam_v @ P_v) * lam_v).sum(axis=1) + np.nan_to_num(su2_v, nan=0.0)
            else:
                if j_col >= 0 and diagC[j_col] > 0:
                    ck = diagC[j_col]
                    mhat = (C[j_col, :] / ck) * m_obs
                    sig2 = np.maximum(diagC - C[j_col, :] ** 2 / ck, 0.0) / max(D - 1, 1)
                    r2c = sig2 + (sig2 / ck) * m_obs ** 2
                else:                                                # no observation yet
                    mhat = np.zeros(C.shape[0])
                    r2c = diagC / max(D - 1, 1)
                vt = np.exp(np.clip(Lc + mhat, -50.0, 50.0))
                vhat = np.zeros(n); r2v = np.zeros(n)
                vhat[active_v] = vt[col[active_v]]
                r2v[active_v] = r2c[col[active_v]]
            # certainty-equivalent forecasts (lognormal): shrink uncertain volume
            # (convex V^-1/2 cost), inflate uncertain spread (linear-in-s expected cost)
            if ce_adjust:
                vopt = certainty_equivalent_volume(vhat, r2v)
                sopt = shat * np.exp(0.5 * np.clip(r2s, 0.0, 100.0))
            else:
                vopt, sopt = vhat, shat
            # cost-optimal split of the lots STILL TO FILL over remaining active bins
            arem = active & (idxrange >= k)
            q_star = np.zeros(n)
            q_star[arem] = _solve_mu(remaining, sopt[arem], vopt[arem])
            q = min(float(round(q_star[k])), remaining)
        sched[k] = q
        if fillable[k]:
            remaining -= min(q, cap[k])
    return sched


def _dyn_spread_vol(bins: pl.DataFrame, quantity: float, *, spread_curves: dict,
                    vol_curves: dict, vol_kind: str, ce_adjust: bool) -> pl.Series:
    """Entry point shared by the dynamic spread x volume strategies (see wrappers)."""
    n = bins.height
    if n == 0:
        return pl.Series("q_requested", [], dtype=pl.Float64)
    total_lots = int(round(float(quantity)))
    if total_lots <= 0:
        return pl.Series("q_requested", [0.0] * n, dtype=pl.Float64)

    key = None
    if QCODE_COL in bins.columns and DATE_COL in bins.columns:
        key = (bins.get_column(QCODE_COL)[0], bins.get_column(DATE_COL)[0])
    spread_model = spread_curves.get(key) if key is not None else None
    vol_model = vol_curves.get(key) if key is not None else None
    if spread_model is None or vol_model is None:
        return twap_schedule(bins, total_lots)

    sched = _dyn_spread_vol_schedule(bins, total_lots, spread_model, vol_model,
                                     vol_kind=vol_kind, ce_adjust=ce_adjust)
    return pl.Series("q_requested", sched, dtype=pl.Float64)


def spread_factor_vol_ols(bins: pl.DataFrame, quantity: float, *, spread_curves: dict,
                          vol_curves: dict, ce_adjust: bool = True, **_: object) -> pl.Series:
    """Dynamic cost-minimiser: **spread = factor Kalman**, **volume = pairwise OLS**.

    Forecasts each remaining bin's **spread** with the sequential factor-Kalman update
    on log-spread surprises (from :func:`build_spread_factor_curves`) and its **volume**
    with the Markov pairwise-OLS regression on the latest observed bin (from
    :func:`build_ols_curves`, as in :func:`adaptive_volume_ols`), then schedules the lots
    still to fill by **minimising the fill model's slippage cost** over both forecasts
    (:func:`_solve_mu` - the same convex optimiser as :func:`liq_spr_static`, but on live,
    per-bin-updated forecasts instead of static curves). Whereas :func:`vwap_factor` /
    :func:`adaptive_volume_ols` only track volume (spread-blind, VWAP-style), this tilts
    the schedule toward bins expected to be both **liquid and tight-spread**.

    **Dynamic**: strictly causal, carries unfilled lots forward, whole lots throughout;
    the last active bin sweeps the residual. Falls back to :func:`twap_schedule` when no
    ``(qcode, date)`` spread **or** volume model exists.

    Parameters
    ----------
    bins : pl.DataFrame
        Bins for one (security, date) with ``qcode``, ``publication_date``,
        ``bin_start_time``, ``volume``, ``vwap``, ``twa_ask``, ``twa_bid``.
    quantity : float
        Total lots (rounded to whole lots).
    spread_curves : dict
        ``{(qcode, date): factor_model}`` from :func:`build_spread_factor_curves`.
    vol_curves : dict
        ``{(qcode, date): ols_model}`` from :func:`build_ols_curves`.
    ce_adjust : bool, default True
        Apply the lognormal certainty-equivalent forecasts before optimising (shrink
        uncertain volume ``exp(-r2/4)``, inflate uncertain spread ``exp(+r2/2)``); see
        :func:`_dyn_spread_vol_schedule`.

    Returns
    -------
    pl.Series
        Integer-valued float series of length ``len(bins)``; unfilled lots carried
        forward (requested total ``>= round(quantity)``).
    """
    return _dyn_spread_vol(bins, quantity, spread_curves=spread_curves,
                           vol_curves=vol_curves, vol_kind="ols", ce_adjust=ce_adjust)


def spread_factor_vol_factor(bins: pl.DataFrame, quantity: float, *, spread_curves: dict,
                             vol_curves: dict, ce_adjust: bool = True, **_: object) -> pl.Series:
    """Dynamic cost-minimiser: **spread = factor Kalman**, **volume = factor Kalman**.

    Identical to :func:`spread_factor_vol_ols` except the **volume** forecast is a second
    sequential factor-Kalman update on log-volume surprises (from
    :func:`build_factor_curves`, exactly the :func:`vwap_factor` filter) rather than the
    Markov OLS. Both the spread and the volume beliefs update on each bin's surprise and
    feed the fill model's convex cost optimiser (:func:`_solve_mu`) over the lots still to
    fill, so the schedule concentrates in bins expected to be liquid **and** tight-spread.

    **Dynamic**: strictly causal, carries unfilled lots forward, whole lots throughout;
    the last active bin sweeps the residual. Falls back to :func:`twap_schedule` when no
    ``(qcode, date)`` spread **or** volume model exists.

    Parameters
    ----------
    bins : pl.DataFrame
        Bins for one (security, date); see :func:`spread_factor_vol_ols`.
    quantity : float
        Total lots (rounded to whole lots).
    spread_curves : dict
        ``{(qcode, date): factor_model}`` from :func:`build_spread_factor_curves`.
    vol_curves : dict
        ``{(qcode, date): factor_model}`` from :func:`build_factor_curves`.
    ce_adjust : bool, default True
        Apply the lognormal certainty-equivalent forecasts before optimising (see
        :func:`spread_factor_vol_ols`).

    Returns
    -------
    pl.Series
        Integer-valued float series of length ``len(bins)``; unfilled lots carried
        forward.
    """
    return _dyn_spread_vol(bins, quantity, spread_curves=spread_curves,
                           vol_curves=vol_curves, vol_kind="factor", ce_adjust=ce_adjust)


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
        # spread of the last bin that quoted - the cost of crossing at end of day
        # to clean up the unfilled shortfall (used by the order-impact metric).
        terminal_spread=pl.col("spread").filter(pl.col("spread").is_not_null()).last(),
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
        # total realised impact ($) = sum over bins of x * s * q_filled (= total_cost).
        pl.col("total_cost").alias("total_realised_impact"),
        pl.when(paper != 0).then(pl.col("is_currency") / paper * 1e4)
        .otherwise(None).alias("is_bps"),
        pl.when(paper != 0).then(slippage_cur / paper * 1e4)
        .otherwise(None).alias("is_slippage_bps"),
        pl.when(paper != 0).then(drift_cur / paper * 1e4)
        .otherwise(None).alias("is_drift_bps"),
        pl.when(paper != 0).then(opp_cur / paper * 1e4)
        .otherwise(None).alias("is_opportunity_bps"),
    )
    # --- Order-impact metric (market-impact IS: realised cost + cross-the-spread
    #     opportunity, NO drift term). Per-order $ pieces; bps are best formed at
    #     the asset level (notional-weighted) via :func:`asset_impact`. ---
    #   realised   = total_realised_impact = Sum_bins x*s*q_filled  (= total_cost)
    #   opportunity= CROSS_SPREAD_FRACTION * terminal_spread * unfilled_qty
    #                (cross half the closing spread on the shortfall)
    #   notional   = Sum_bins VWAP_bin*q_filled + VWAP_last*unfilled_qty
    opp_impact_cur = pl.when(pl.col("unfilled_qty") > 0).then(
        CROSS_SPREAD_FRACTION * pl.col("terminal_spread") * pl.col("unfilled_qty")
    ).otherwise(0.0)
    impact_notional = pl.col("filled_vwap_notional") + pl.when(pl.col("unfilled_qty") > 0).then(
        pl.col("terminal_price") * pl.col("unfilled_qty")
    ).otherwise(0.0)
    g = g.with_columns(
        opp_impact_cur.alias("opportunity_impact"),
        impact_notional.alias("impact_notional"),
    )
    g = g.with_columns(
        (pl.col("total_realised_impact") + pl.col("opportunity_impact")).alias("order_impact"),
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

    Returns one row per IS component - ``realised_impact`` (the summed x*s*q_filled,
    i.e. ``total_realised_impact`` in bps), ``drift``, ``opportunity``
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
        Columns ``[strategy?, component, mean, median, variance, fill_rate]`` -
        four rows (realised_impact, drift, opportunity, total). ``fill_rate`` is the
        mean fill rate across the same orders (constant within a strategy); it puts
        the ``opportunity`` component in context, since opportunity cost is non-zero
        only when ``fill_rate < 1``.
    """
    comps = [("realised_impact", "is_slippage_bps"), ("drift", "is_drift_bps"),
             ("opportunity", "is_opportunity_bps"), ("total", "is_bps")]
    s = summary.filter(pl.col("is_bps").is_not_null() & pl.col("is_bps").is_not_nan())
    fill = float(s.get_column("fill_rate").mean()) if "fill_rate" in s.columns else None
    recs = []
    for name, col in comps:
        c = s.get_column(col)
        rec = {"component": name, "mean": c.mean(),
               "median": c.median(), "variance": c.var(), "fill_rate": fill}
        if label is not None:
            rec = {"strategy": label, **rec}
        recs.append(rec)
    return pl.DataFrame(recs)


# monetary (native-currency) columns of a summarise_fills row that must be scaled
# TOGETHER by the same FX rate so their ratios (the bps metrics) stay invariant while
# their cross-asset SUMS become currency-consistent. Deliberately excludes the IS
# columns (is_currency / total_cost) and the per-unit prices - the order-impact family
# is what asset_impact / order_impact_stats pool across assets.
_USD_MONETARY_COLS = ("total_realised_impact", "opportunity_impact",
                      "impact_notional", "order_impact")


def apply_multiplier(summary: pl.DataFrame, multipliers: pl.DataFrame, *,
                     monetary_cols: Sequence[str] = _USD_MONETARY_COLS,
                     qcode_col: str = "qcode", mult_col: str = "mult") -> pl.DataFrame:
    """Scale an order summary's ``$`` columns by the per-contract **contract multiplier**.

    The fill model works in **quote points x contract counts**, so the raw ``$``
    columns from :func:`summarise_fills` (``total_realised_impact``,
    ``impact_notional``, ...) are in *point-contracts*, not real currency. Multiplying
    by the contract multiplier (currency per point per contract - e.g. ES = ``$50`` /
    point, a bond future = ``1000``) turns them into genuine native-currency notional.

    Like :func:`to_usd` this is a per-order **constant** scale (one multiplier per
    ``qcode``), so every ratio built from these columns - per-order and per-``qcode``
    bps, the paired t-tests - is unchanged. Only cross-asset ``$`` sums shift: a
    high-multiplier contract now contributes proportionally more notional to a pooled
    group. Apply this **before** :func:`to_usd` (points -> native currency -> USD).

    Parameters
    ----------
    summary : pl.DataFrame
        Output of :func:`summarise_fills` (needs ``qcode_col``).
    multipliers : pl.DataFrame
        ``qcode -> mult`` lookup (e.g. ``contract_multipliers.csv``); must contain
        ``qcode_col`` and ``mult_col``.
    monetary_cols : sequence of str, default the order-impact ``$`` family
        Columns to scale. Missing columns are skipped.
    qcode_col, mult_col : str
        Column names for the join key and the multiplier value.

    Returns
    -------
    pl.DataFrame
        ``summary`` with the ``monetary_cols`` scaled to native-currency notional.
    """
    mm = multipliers.select(pl.col(qcode_col), pl.col(mult_col).cast(pl.Float64).alias("_mult"))
    s = (summary.join(mm, on=qcode_col, how="left")
                .with_columns(pl.col("_mult").fill_null(1.0)))
    s = s.with_columns([(pl.col(c) * pl.col("_mult")).alias(c)
                        for c in monetary_cols if c in s.columns])
    return s.drop("_mult")


def to_usd(summary: pl.DataFrame, fx: pl.DataFrame, ccy_map: pl.DataFrame, *,
           monetary_cols: Sequence[str] = _USD_MONETARY_COLS,
           date_col: str = "date", qcode_col: str = "qcode") -> pl.DataFrame:
    """Convert an order summary's native-currency ``$`` columns to USD.

    The dataset spans several quote currencies (see ``docs/qcode_mapping.md``), so any
    metric that **sums** ``$`` across instruments - :func:`asset_impact` pooled by tier
    / asset class, the ``*_bps_nw`` and ``$`` columns of :func:`order_impact_stats` -
    is meaningless until the pieces share a currency. This joins each order's quote
    currency (``ccy_map``: ``qcode -> currency``) and the daily FX rate (``fx``:
    columns ``currency``, ``date``, ``usd_per_unit``) and multiplies every column in
    ``monetary_cols`` by ``usd_per_unit`` (USD instruments pass through at 1.0). The FX
    match is an **as-of backward** join on ``date`` within currency, so weekends /
    holidays take the most recent prior rate.

    Because all ``monetary_cols`` are scaled by the **same** per-order factor, every
    ratio built from them (per-order and per-qcode bps, the paired t-tests) is
    unchanged - only the cross-asset sums are corrected. Run this on each summary
    **before** any grouped/pooled aggregation.

    Parameters
    ----------
    summary : pl.DataFrame
        Output of :func:`summarise_fills` (needs ``qcode_col`` and ``date_col``).
    fx : pl.DataFrame
        Daily FX table with columns ``currency``, ``date`` and ``usd_per_unit`` (USD
        per one unit of the local currency). USD need not be present - it passes
        through at 1.0.
    ccy_map : pl.DataFrame
        ``qcode -> currency`` lookup (e.g. ``qcode_mapping.csv``); must contain
        ``qcode`` and ``currency``.
    monetary_cols : sequence of str, default the order-impact ``$`` family
        Columns to convert (scaled together). Missing columns are skipped.
    date_col, qcode_col : str
        Column names in ``summary``.

    Returns
    -------
    pl.DataFrame
        ``summary`` with the ``monetary_cols`` expressed in USD and an added
        ``currency`` column. Row order is not preserved (sorted for the as-of join).
    """
    fx2 = (fx.select("currency",
                     pl.col(date_col if date_col in fx.columns else "date")
                     .cast(pl.Utf8).alias("_fxdate"),
                     pl.col("usd_per_unit").cast(pl.Float64))
             .sort(["currency", "_fxdate"]))
    ccy = ccy_map.select(pl.col("qcode").alias(qcode_col), "currency")
    s = (summary.join(ccy, on=qcode_col, how="left")
                .with_columns(pl.col("currency").fill_null("USD"),
                              pl.col(date_col).cast(pl.Utf8).alias("_sdate"))
                .sort(["currency", "_sdate"]))
    s = s.join_asof(fx2, left_on="_sdate", right_on="_fxdate", by="currency",
                    strategy="backward", check_sortedness=False)
    # USD (never in fx) and any pre-history gap fall back to 1.0
    s = s.with_columns(pl.col("usd_per_unit").fill_null(1.0))
    s = s.with_columns([(pl.col(c) * pl.col("usd_per_unit")).alias(c)
                        for c in monetary_cols if c in s.columns])
    return s.drop(["_sdate", "_fxdate", "usd_per_unit"], strict=False)


def asset_impact(summary: pl.DataFrame, *, by: str = "qcode") -> pl.DataFrame:
    """Grouped **order-impact** metric, notional-weighted across orders.

    .. warning::
        The ``$`` pieces are **summed** within each group, so ``summary`` must be in a
        **single currency** for any group that spans instruments. Grouping by ``qcode``
        is always safe (one currency per qcode); for ``by="tier"`` / ``"asset_class"``
        run :func:`to_usd` on the summary first, or the pooled bps mixes currencies.

    The "market-impact implementation shortfall": realised spread cost on the
    filled lots **plus** a cross-the-spread opportunity cost on the unfilled
    shortfall, with **no drift term** (unlike :func:`decompose_is`). Within each
    group (``by``, default per ``qcode``; pass e.g. ``"asset_class"`` to pool by
    asset class) it **sums the dollar pieces** from :func:`summarise_fills` and
    divides once - a genuine notional-weighted ratio, **not** an average of the
    per-qcode bps::

        realised_$       = Sum_orders Sum_bins  x * s * q_filled        (= total_realised_impact)
        opportunity_$    = Sum_orders  CROSS_SPREAD_FRACTION * s_last * q_unfilled
        impact_notional  = Sum_orders [ Sum_bins VWAP_bin*q_filled + VWAP_last*q_unfilled ]

        realised_bps     = realised_$    / impact_notional * 1e4
        opportunity_bps  = opportunity_$ / impact_notional * 1e4
        total_impact_bps = realised_bps + opportunity_bps        (shared denominator -> additive)

    Orders with a null / non-positive ``impact_notional`` (a fully dead day -
    nothing traded and nothing to mark) are dropped.

    Parameters
    ----------
    summary : pl.DataFrame
        Output of :func:`summarise_fills` (needs ``total_realised_impact``,
        ``opportunity_impact``, ``impact_notional``, ``fill_rate`` and the ``by``
        column).
    by : str, default ``"qcode"``
        Grouping column - ``"qcode"`` for per-asset, ``"asset_class"`` (after a join)
        to pool bond / commodity / equity futures, etc.

    Returns
    -------
    pl.DataFrame
        One row per ``by`` group with ``n_orders``, ``fill_rate`` (mean), the three
        ``*_$`` totals, ``impact_notional`` and ``realised_bps`` /
        ``opportunity_bps`` / ``total_impact_bps``.
    """
    s = summary.filter(
        pl.col("impact_notional").is_not_null()
        & pl.col("impact_notional").is_not_nan()
        & (pl.col("impact_notional") > 0)
    )
    g = s.group_by(by, maintain_order=True).agg(
        n_orders=pl.len(),
        fill_rate=pl.col("fill_rate").mean(),
        realised_impact=pl.col("total_realised_impact").sum(),
        opportunity_impact=pl.col("opportunity_impact").sum(),
        impact_notional=pl.col("impact_notional").sum(),
    )
    g = g.with_columns(
        (pl.col("realised_impact") + pl.col("opportunity_impact")).alias("total_impact"),
        (pl.col("realised_impact") / pl.col("impact_notional") * 1e4).alias("realised_bps"),
        (pl.col("opportunity_impact") / pl.col("impact_notional") * 1e4).alias("opportunity_bps"),
    )
    return g.with_columns(
        (pl.col("realised_bps") + pl.col("opportunity_bps")).alias("total_impact_bps"),
    )


def order_impact_stats(summary: pl.DataFrame,
                       label: Optional[str] = None) -> pl.DataFrame:
    """One-row strategy summary of the order-impact metric (stack several to compare).

    The ``$`` columns are summed over **all** orders. The bps come two ways:

    * ``*_bps`` - **mean across assets** of the per-qcode notional-weighted bps
      from :func:`asset_impact` (each asset counts equally - the "reconcile by a
      simple mean" view; sensitive to illiquid, low-fill names).
    * ``*_bps_nw`` - **notional-weighted across everything**: pool every order's
      ``$`` and ``impact_notional`` and divide once
      (``Sum impact / Sum notional * 1e4``). This tracks the ``$`` columns
      directly (dominated by the liquid, high-notional names).

    Either way realised + opportunity add to the total (shared denominator within
    each, and the mean / pooled sum are both linear).

    Parameters
    ----------
    summary : pl.DataFrame
        Output of :func:`summarise_fills`.
    label : str, optional
        Strategy name stamped into a leading ``strategy`` column when given.

    Returns
    -------
    pl.DataFrame
        Single row: ``[strategy?, fill_rate, realised_impact_$, opportunity_impact_$,
        total_impact_$, mean_$_per_order, realised_bps, opportunity_bps,
        total_impact_bps, realised_bps_nw, opportunity_bps_nw, total_impact_bps_nw]``.
    """
    per_asset = asset_impact(summary)
    valid = summary.filter(
        pl.col("impact_notional").is_not_null()
        & pl.col("impact_notional").is_not_nan()
        & (pl.col("impact_notional") > 0)
    )
    n = valid.height
    realised = float(valid.get_column("total_realised_impact").sum()) if n else 0.0
    opp = float(valid.get_column("opportunity_impact").sum()) if n else 0.0
    notional = float(valid.get_column("impact_notional").sum()) if n else 0.0
    has = per_asset.height > 0
    scale = 1e4 / notional if notional > 0 else None
    rec = {
        "fill_rate": float(summary.get_column("fill_rate").mean()),
        "realised_impact_$": realised,
        "opportunity_impact_$": opp,
        "total_impact_$": realised + opp,
        "mean_$_per_order": (realised + opp) / n if n else None,
        # mean across assets (each qcode equal weight)
        "realised_bps": float(per_asset.get_column("realised_bps").mean()) if has else None,
        "opportunity_bps": float(per_asset.get_column("opportunity_bps").mean()) if has else None,
        "total_impact_bps": float(per_asset.get_column("total_impact_bps").mean()) if has else None,
        # notional-weighted across all orders (Sum impact / Sum notional)
        "realised_bps_nw": realised * scale if scale is not None else None,
        "opportunity_bps_nw": opp * scale if scale is not None else None,
        "total_impact_bps_nw": (realised + opp) * scale if scale is not None else None,
    }
    if label is not None:
        rec = {"strategy": label, **rec}
    return pl.DataFrame([rec])


def participation_dispersion(fills: pl.DataFrame, *, by: str = "order_id",
                             extra_cols: Sequence[str] = ("trade_list",)) -> pl.DataFrame:
    r"""Per-order **weighted participation volatility** ``sigma_p`` and its CV.

    Measures how far a strategy's intraday participation strays from a flat,
    volume-matching profile. For one order, with per-bin market-volume share
    ``v_k = vol_k / sum(vol)``, actual participation ``p_k = q_filled_k / vol_k``,
    and overall (target) participation ``P = sum(q_filled) / sum(vol)``::

        sigma_p = sqrt( sum_k  v_k * (p_k - P)^2 )        # volume-weighted std of p_k
        CV_p    = sigma_p / P

    ``P`` is exactly the volume-weighted mean of ``p_k`` (``sum_k v_k p_k =
    sum q_filled / sum vol``), so ``sigma_p`` is the volume-weighted standard
    deviation of the per-bin participation and ``CV_p`` its coefficient of
    variation. A schedule that perfectly tracks volume has ``p_k = P`` in every
    bin, giving ``sigma_p = 0`` - so **omniscient_vwap is ~0** (only whole-lot
    rounding and the ``2*volume`` cap perturb it), which is the sanity check.

    Bins with no market volume (``vol_k = 0``) get ``v_k = 0`` and ``p_k = 0``, so
    they drop out of both the weight and the sum.

    Parameters
    ----------
    fills : pl.DataFrame
        Per-bin output of :func:`run_strategy` / :func:`run_trade_list` (needs the
        grouping key, ``volume`` and ``q_filled``).
    by : str, default ``"order_id"``
        Order grouping key.
    extra_cols : sequence of str, default ``("trade_list",)``
        Identifier columns carried through (``first()`` per order) when present.

    Returns
    -------
    pl.DataFrame
        One row per order: identifiers, ``n_bins``, ``total_volume``,
        ``total_filled``, ``target_participation`` (``P``), ``participation_var``
        (``sigma_p^2``), ``participation_vol`` (``sigma_p``) and
        ``participation_cv`` (``CV_p``). ``P``/var/vol/cv are null on a day with no
        traded volume.

    Examples
    --------
    Flat participation (p_k = P everywhere) gives zero volatility and CV:

    >>> import polars as pl
    >>> f = pl.DataFrame({"order_id": [0, 0], "volume": [100.0, 300.0],
    ...                   "q_filled": [10.0, 30.0]})
    >>> d = participation_dispersion(f)
    >>> round(d["participation_vol"][0], 12), round(d["participation_cv"][0], 12)
    (0.0, 0.0)
    """
    has_qcode = QCODE_COL in fills.columns
    carried = [pl.col(c).first().alias(c) for c in extra_cols if c in fills.columns]
    ident = [pl.col(c).first().alias(c) for c in (SECURITY_COL, DATE_COL, "side")
             if c in fills.columns]
    vol = pl.col(VOLUME_COL).cast(pl.Float64).fill_null(0.0)
    fq = pl.col("q_filled").cast(pl.Float64).fill_null(0.0)
    pk = pl.when(vol > 0).then(fq / vol).otherwise(0.0)        # per-bin participation p_k
    tot = vol.sum()
    P = fq.sum() / tot                                         # overall participation = weighted mean of p_k
    wvar = (vol * (pk - P).pow(2)).sum() / tot                 # sum_k v_k (p_k - P)^2

    g = fills.group_by(by, maintain_order=True).agg(
        *carried, *ident,
        qcode=(pl.col(QCODE_COL).first() if has_qcode else pl.lit(None)),
        n_bins=pl.len(),
        total_volume=vol.sum(),
        total_filled=fq.sum(),
        target_participation=pl.when(tot > 0).then(P).otherwise(None),
        participation_var=pl.when(tot > 0).then(wvar).otherwise(None),
    )
    g = g.with_columns(pl.col("participation_var").sqrt().alias("participation_vol"))
    return g.with_columns(
        pl.when(pl.col("target_participation") > 0)
        .then(pl.col("participation_vol") / pl.col("target_participation"))
        .otherwise(None).alias("participation_cv")
    )


def participation_stats(disp: pl.DataFrame,
                        label: Optional[str] = None) -> pl.DataFrame:
    """One-row strategy summary of participation dispersion (stack to compare).

    Averages the per-order :func:`participation_dispersion` outputs - the headline
    is ``mean_cv`` (mean coefficient of variation across orders). Orders with an
    undefined CV (no traded volume / ``P = 0``) are dropped.

    Parameters
    ----------
    disp : pl.DataFrame
        Output of :func:`participation_dispersion`.
    label : str, optional
        Strategy name stamped into a leading ``strategy`` column when given.

    Returns
    -------
    pl.DataFrame
        Single row: ``[strategy?, n_orders, mean_target_participation,
        mean_participation_vol, mean_participation_var, mean_cv, median_cv]``.
    """
    d = disp.filter(
        pl.col("participation_cv").is_not_null()
        & pl.col("participation_cv").is_not_nan()
    )
    has = d.height > 0
    rec = {
        "n_orders": d.height,
        "mean_target_participation": float(d["target_participation"].mean()) if has else None,
        "mean_participation_vol": float(d["participation_vol"].mean()) if has else None,
        "mean_participation_var": float(d["participation_var"].mean()) if has else None,
        "mean_cv": float(d["participation_cv"].mean()) if has else None,
        "median_cv": float(d["participation_cv"].median()) if has else None,
    }
    if label is not None:
        rec = {"strategy": label, **rec}
    return pl.DataFrame([rec])


# ---------------------------------------------------------------------------
# Pairwise significance tests (paired t-tests, one matrix per group)
# ---------------------------------------------------------------------------

# per-order impact-bps numerator (divided by impact_notional * 1e4) for each metric;
# mirrors the METRIC options of :func:`asset_impact` / the notebook tier tables.
_IMPACT_NUMERATOR = {
    "total_impact_bps": "order_impact",
    "realised_bps": "total_realised_impact",
    "opportunity_bps": "opportunity_impact",
}


def _tier_lookup(tiers: dict) -> pl.DataFrame:
    """Long-form ``qcode -> _group`` lookup from a ``{group_label: [qcodes]}`` map."""
    return pl.DataFrame({
        "qcode": [q for qs in tiers.values() for q in qs],
        "_group": [g for g, qs in tiers.items() for _ in qs],
    })


def paired_ttest_panels(by_strat: dict, strats: Sequence[str], groups: Sequence,
                        *, title: str = "", value_label: str = "value",
                        ncol: int = 2, show: bool = True):
    """Grid of per-group ``N x N`` paired-t-test matrices comparing strategies pairwise.

    For each ``group`` and each ordered strategy pair ``(i, j)``, the orders present in
    **both** strategies (matched on ``order_id``) are paired and a paired t-test
    (``scipy.stats.ttest_rel``) is run on the per-order ``val`` - H0: the mean pairwise
    difference is zero. ``t > 0`` means strategy ``i``'s metric is higher than strategy
    ``j``'s. Every matrix cell is coloured by the p-value (green ``p<0.01``, orange
    ``0.01<=p<0.05``, grey = not significant) and annotated with ``t``, ``p`` and the
    paired sample size ``n``; the diagonal is blank. ``matplotlib`` and ``scipy`` are
    imported lazily, so they are only required when this function is actually called.

    Parameters
    ----------
    by_strat : dict[str, pl.DataFrame]
        ``{strategy_name -> frame}`` where each frame has columns ``order_id``,
        ``_group`` (the panel key) and ``val`` (the per-order metric under test).
    strats : sequence of str
        Strategy names, in the row/column order of every matrix. ``N = len(strats)`` is
        read here, never hardcoded, so extra strategies just enlarge the grid.
    groups : sequence
        Panel keys to draw - one ``N x N`` matrix each (e.g. the impact tiers).
    title : str, optional
        Figure suptitle.
    value_label : str, optional
        Name of the per-order metric, used in the x-axis caption.
    ncol : int, default 2
        Number of panel columns in the subplot grid.
    show : bool, default True
        Call ``plt.show()`` when done (notebook convenience).

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.patches import Patch
    from scipy import stats

    strats = list(strats)
    groups = list(groups)
    N = len(strats)
    cmap = ListedColormap(["#d9d9d9", "#fdae61", "#1a9850"])   # n.s. / p<0.05 / p<0.01
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
    nrow = int(np.ceil(len(groups) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(1.25 * N * ncol, 1.15 * N * nrow),
                             squeeze=False)
    axes = axes.reshape(-1)
    for ax, grp in zip(axes, groups):
        # per-strategy {order_id -> val} restricted to this group
        cols = {}
        for nm in strats:
            d = by_strat[nm].filter(pl.col("_group") == grp)
            cols[nm] = dict(zip(d["order_id"].to_list(), d["val"].to_list()))
        tvals = np.full((N, N), np.nan)
        pvals = np.full((N, N), np.nan)
        code = np.full((N, N), np.nan)
        nobs = np.zeros((N, N), int)
        for i, a in enumerate(strats):
            for j, b in enumerate(strats):
                if i == j:
                    continue
                ca, cb = cols[a], cols[b]
                keys = ca.keys() & cb.keys()
                xa = np.fromiter((ca[k] for k in keys), float, len(keys))
                xb = np.fromiter((cb[k] for k in keys), float, len(keys))
                m = np.isfinite(xa) & np.isfinite(xb)
                xa, xb = xa[m], xb[m]
                nobs[i, j] = xa.size
                if xa.size < 2 or np.allclose(xa, xb):
                    continue
                t, p = stats.ttest_rel(xa, xb)
                if not np.isfinite(t):
                    continue
                tvals[i, j], pvals[i, j] = t, p
                code[i, j] = 2 if p < 0.01 else (1 if p < 0.05 else 0)
        ax.imshow(np.ma.masked_invalid(code), cmap=cmap, norm=norm, aspect="equal")
        ax.set_xticks(range(N)); ax.set_yticks(range(N))
        ax.set_xticklabels(strats, rotation=45, ha="right", fontsize=6)
        ax.set_yticklabels(strats, fontsize=6)
        ax.set_title(f"{grp}", fontsize=9)
        for i in range(N):
            for j in range(N):
                if i == j:
                    ax.text(j, i, "--", ha="center", va="center", fontsize=7, color="grey")
                elif np.isfinite(tvals[i, j]):
                    ax.text(j, i, f"t={tvals[i, j]:.2f}\np={pvals[i, j]:.2g}\nn={nobs[i, j]}",
                            ha="center", va="center", fontsize=5.0)
                else:
                    ax.text(j, i, "n/a", ha="center", va="center", fontsize=5.5, color="grey")
    for ax in axes[len(groups):]:
        ax.set_visible(False)
    handles = [Patch(color="#1a9850", label="p < 0.01"),
               Patch(color="#fdae61", label="0.01 <= p < 0.05"),
               Patch(color="#d9d9d9", label="p >= 0.05  (not significant)")]
    fig.suptitle(title, fontsize=12, y=0.995)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.955),
               ncol=3, fontsize=9, framealpha=0.9)
    fig.supxlabel(f"strategy B (columns) -- entry (row i, col j): paired t-test of "
                  f"{value_label}; t>0 => row i higher than col j", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    if show:
        plt.show()
    return fig


def paired_impact_ttest(summaries: Sequence, tiers: dict, *, bucket: str = "",
                        metric: str = "total_impact_bps", show: bool = True):
    """Pairwise paired t-tests of the per-order market-impact bps, one matrix per tier.

    Thin wrapper over :func:`paired_ttest_panels`. For each strategy the per-order
    impact bps is ``<numerator> / impact_notional * 1e4`` over the valid orders
    (positive ``impact_notional``); ``metric`` picks the numerator
    (``total_impact_bps`` -> ``order_impact``; ``realised_bps`` ->
    ``total_realised_impact``; ``opportunity_bps`` -> ``opportunity_impact``). This is
    the **per-order** analogue of the pooled, notional-weighted tier bps from
    :func:`asset_impact` - the pooled figure is one number per tier and so cannot be
    paired, hence the per-order ratio is tested instead.

    Parameters
    ----------
    summaries : sequence of (str, pl.DataFrame)
        ``(strategy_name, summarise_fills output)`` pairs; all strategies must share the
        ``order_id`` space (i.e. be run from the same trade list).
    tiers : dict[str, sequence[str]]
        ``{tier_label: [qcodes]}`` mapping (e.g. the notebook's ``IMPACT_TIERS``).
    bucket : str, optional
        Trade-list label, shown in the title.
    metric : str, default ``"total_impact_bps"``
        Which impact component to test (see above).
    show : bool, default True

    Returns
    -------
    matplotlib.figure.Figure
    """
    num = _IMPACT_NUMERATOR[metric]
    lut = _tier_lookup(tiers)
    by_strat = {
        nm: (s.filter(pl.col("impact_notional").is_not_null()
                      & pl.col("impact_notional").is_not_nan()
                      & (pl.col("impact_notional") > 0))
              .join(lut, on="qcode", how="inner")
              .select("order_id", "_group",
                      (pl.col(num) / pl.col("impact_notional") * 1e4).alias("val")))
        for nm, s in summaries
    }
    return paired_ttest_panels(
        by_strat, [nm for nm, _ in summaries], list(tiers.keys()),
        title=f"Paired t-test: per-order {metric}, strategy x strategy by tier - {bucket}",
        value_label=metric, show=show)


def paired_participation_ttest(fills: Sequence, tiers: dict, *, bucket: str = "",
                               show: bool = True):
    """Pairwise paired t-tests of the per-order participation CV, one matrix per tier.

    Thin wrapper over :func:`paired_ttest_panels`. For each strategy the per-order
    metric is the participation CV from :func:`participation_dispersion` (orders with an
    undefined CV dropped), tagged with its ``tiers`` bucket and paired across strategies
    on ``order_id``.

    Parameters
    ----------
    fills : sequence of (str, pl.DataFrame)
        ``(strategy_name, per-bin fills)`` pairs (the ``*_fills_full`` frames).
    tiers : dict[str, sequence[str]]
        ``{tier_label: [qcodes]}`` mapping.
    bucket : str, optional
    show : bool, default True

    Returns
    -------
    matplotlib.figure.Figure
    """
    lut = _tier_lookup(tiers)
    by_strat = {
        nm: (participation_dispersion(f)
             .filter(pl.col("participation_cv").is_not_null()
                     & pl.col("participation_cv").is_not_nan())
             .join(lut, on="qcode", how="inner")
             .select("order_id", "_group", pl.col("participation_cv").alias("val")))
        for nm, f in fills
    }
    return paired_ttest_panels(
        by_strat, [nm for nm, _ in fills], list(tiers.keys()),
        title=f"Paired t-test: per-order participation CV, strategy x strategy by tier - {bucket}",
        value_label="participation CV", show=show)
