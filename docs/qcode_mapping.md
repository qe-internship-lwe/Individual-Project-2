# `QCODE_MAPPING`

`QCODE_MAPPING` is the **reference (dimension) table** for the 50 instruments in
this dataset — one row per `qcode`, holding the static descriptive attributes of
each futures *curve*. It's the lookup you join onto the other three tables
(`BINNED_DATA`, `SECURITY_META`, `TRADE_LIST`) on `qcode` to attach
human-readable names, exchange, currency, asset class, etc. It carries **no
prices or dates** — purely "what is this instrument."

**Grain:** 50 rows, one per `qcode` (the primary key).

## Columns

| Column | What it is |
|---|---|
| `qcode` | Internal curve identifier / primary key (e.g. `SI`, `TY`, `ES`). This is the key used across all tables. |
| `bbg_code` | Bloomberg root ticker (e.g. `SI`, `WN`, `UXY`). 49 unique — there's **one collision**: `LO` (Cocoa #7) and `OX` (Swedish OMX) both have root `QC`. |
| `yellow_key` | Bloomberg "yellow key" sector — `Comdty` or `Index`. Combined with `bbg_code` it forms the full Bloomberg ticker (e.g. `TY Comdty`), and it disambiguates the `QC` collision above. |
| `description` | Human-readable name + exchange, e.g. *"Treasury Note, U.S., 10-year (CBOT)"*. |
| `exchange` | Listing exchange code — 13 of them: `CBT` (CBOT), `CME`, `CMX` (COMEX), `NYM` (NYMEX), `NYB` (ICE US), `ICE` (ICE Europe), `EUX` (Eurex), `EDX`, `EOE` (Euronext Amsterdam), `SSE`, `OSE` (Osaka), `SFE` (Sydney), `SGX`. |
| `currency` | Quote currency — `USD`, `EUR`, `GBP`, `JPY`, `SEK`, `SGD`, `AUD`. |
| `asset_class` | `BondFuture` (12), `CommodityFuture` (28), or `EquityFuture` (10). |
| `delivery` | Settlement type — `Phys` (physical) or `Cash`. |
| `is_convention_buy_near` | 0/1 flag for the **roll/quote convention** — whether the standard direction for this curve is to buy the near contract (vs the deferred). Relevant for signing calendar-spread / roll trades; all 50 here are `1.0`. |

## How it's used

- `qcode` is the **join key everywhere** (to `BINNED_DATA`, `SECURITY_META`, `TRADE_LIST`).
- `description` / `exchange` / `asset_class` are what you'd use to **label and group**
  instruments (e.g. the trading-hour groupings: US rates, US grains, EUREX rates,
  Asia-Pacific equity index, …).
- `bbg_code` + `yellow_key` map back to **Bloomberg tickers**.

## Caveats

- `is_convention_buy_near` is constant (`1.0`) across all 50 rows in the extracted
  CSV, so it carries no discriminating information here — fine to ignore unless a
  broader universe is re-pulled.
- `bbg_code` is **not** unique on its own (the `QC` collision); use `qcode` as the
  key, or `bbg_code` + `yellow_key` if you must map by Bloomberg root.

## Distinct-value summary

| Field | # distinct | Values |
|---|---|---|
| `qcode` | 50 | (primary key) |
| `bbg_code` | 49 | one collision: `QC` → {`LO`, `OX`} |
| `yellow_key` | 2 | `Comdty`, `Index` |
| `exchange` | 13 | `CBT`, `CME`, `CMX`, `EDX`, `EOE`, `EUX`, `ICE`, `NYB`, `NYM`, `OSE`, `SFE`, `SGX`, `SSE` |
| `currency` | 7 | `AUD`, `EUR`, `GBP`, `JPY`, `SEK`, `SGD`, `USD` |
| `asset_class` | 3 | `BondFuture`, `CommodityFuture`, `EquityFuture` |
| `delivery` | 2 | `Cash`, `Phys` |
| `is_convention_buy_near` | 1 | `1.0` (constant) |
