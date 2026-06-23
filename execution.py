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

import polars as pl

__all__ = [
    "BinFill",
    "simulate_bin_fill",
    "twap_schedule",
    "vwap_schedule",
    "run_strategy",
    "run_trade_list",
    "summarise_fills",
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
    """Baseline TWAP schedule: split ``quantity`` evenly across every bin.

    Time-Weighted Average Price. The total order quantity is divided equally
    over all bins present for the (security, date), so each bin requests
    ``quantity / n_bins`` regardless of that bin's volume or liquidity.

    This is the naive baseline on purpose:

    * It allocates to **every** bin in the day, including bins that cannot fill
      (zero-volume / missing-data bins). Quantity assigned to those bins simply
      does not fill and is lost (no auto-retry), so the realised fill rate will
      be below 100% in illiquid names - which is exactly what we want to
      measure a smarter strategy against.
    * It does not round to integer lots; requested quantities may be fractional.

    Parameters
    ----------
    bins : pl.DataFrame
        The bins for one (security, date). Only the row count is used; the
        caller (:func:`run_strategy`) is responsible for sorting by time.
    quantity : float
        Total quantity to execute over the day (a magnitude; ``side`` is
        handled by the runner).
    **_ :
        Ignored. Present so strategies share a uniform calling convention.

    Returns
    -------
    pl.Series
        Float series of length ``len(bins)`` summing to ``quantity``; every
        element equals ``quantity / len(bins)``.

    Examples
    --------
    >>> import polars as pl
    >>> b = pl.DataFrame({"bin_start_time": ["09:00", "09:05", "09:10", "09:15"]})
    >>> list(twap_schedule(b, 1000))
    [250.0, 250.0, 250.0, 250.0]
    """
    n = bins.height
    if n == 0:
        return pl.Series("q_requested", [], dtype=pl.Float64)
    per_bin = float(quantity) / n
    return pl.Series("q_requested", [per_bin] * n, dtype=pl.Float64)


def vwap_schedule(bins: pl.DataFrame, quantity: float, **_: object) -> pl.Series:
    """Baseline VWAP schedule: split ``quantity`` by each bin's volume share.

    Volume-Weighted Average Price. Each bin is requested in proportion to the
    fraction of the **day's** total traded volume that occurred in that bin::

        q_requested[i] = quantity * volume[i] / sum(volume)

    So if a bin carried 3% of the day's volume it is asked to trade 3% of the
    order. This tracks the realised intraday volume profile (the U-shape seen in
    ``test_snowflake.ipynb``): heavy bins at the open/close get more, thin
    midday bins get less.

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
        Total quantity to execute over the day (a magnitude; ``side`` is
        handled by the runner).
    **_ :
        Ignored. Present so strategies share a uniform calling convention.

    Returns
    -------
    pl.Series
        Float series of length ``len(bins)`` summing to ``quantity``. When the
        day has **no** traded volume at all (``sum(volume) <= 0``, e.g. a fully
        dead day), the proportions are undefined and it falls back to an even
        TWAP split so the order is still fully requested (nothing fills either
        way, since every bin is a no-trade bin).

    Examples
    --------
    >>> import polars as pl
    >>> b = pl.DataFrame({
    ...     "bin_start_time": ["09:00", "09:05", "09:10"],
    ...     "volume": [100, 0, 50],          # middle bin is a no-trade bin
    ... })
    >>> [round(x, 2) for x in vwap_schedule(b, 300)]   # shares 2/3, 0, 1/3
    [200.0, 0.0, 100.0]
    """
    n = bins.height
    if n == 0:
        return pl.Series("q_requested", [], dtype=pl.Float64)

    vol = bins.get_column(VOLUME_COL).cast(pl.Float64).fill_null(0.0).fill_nan(0.0)
    total = vol.sum()
    if total is None or total <= 0:
        # No traded volume anywhere -> fall back to an even split.
        per_bin = float(quantity) / n
        return pl.Series("q_requested", [per_bin] * n, dtype=pl.Float64)

    return (vol / total * float(quantity)).rename("q_requested")


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
    requested = strategy(b, quantity, **(strategy_params or {}))
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
    g = g.with_columns(
        pl.when(paper != 0)
        .then(pl.col("is_currency") / paper * 1e4)
        .otherwise(None)
        .alias("is_bps"),
    )
    return g
