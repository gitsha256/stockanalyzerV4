"""
NSE Stock Screener — Intraday & Weekly Swing Picks
====================================================
Reads your snapshot CSV (same schema as StockAnalyzerV4 output)
and produces two ranked lists:
  1. Intraday candidates
  2. Weekly swing candidates

Column schema reference (abbreviated → full):
  stge  : Weinstein stage string  e.g. "Stage 2 (Uptrend)"
  g200  : bool  price > SMA 200
  g050  : bool  price > SMA 50
  g020  : bool  price > SMA 20
  rsi   : float daily RSI (14)
  wrsi  : float weekly RSI (14)
  adx   : float ADX trend strength
  rvol  : float relative volume (vs 20d avg)
  ascr  : float activity score = Price × Vol / 10M
  chan  : float daily % change open→close
  tren  : str   "Uptrend" / "Downtrend" / "Sideways"
  tstr  : str   "Strong" / "Moderate" / "Weak"
  vola  : float annualised volatility (21d)
  zone  : str   "Premium" / "Near Premium" / "Equilibrium" / "Near Discount" / "Discount"
  delt  : float % distance from 52W high  (0 = AT high, positive = below high)
  DlPer : float delivery % of traded volume
  bbup  : bool  close > upper Bollinger Band
  bbsq  : bool  BB squeeze (volatility at 300-period low)
  mpat  : str   main chart pattern name
  pcon  : int   pattern confidence 0–99
  ws30  : float weekly SMA 30
  vtrd  : str   volume trend "Increasing" / "Decreasing"
  sect  : str   sector name
  clos  : float last traded price
  arnk  : int   activity rank (lower = more liquid)
"""

import pandas as pd
import numpy as np
import sys
import os

# ─────────────────────────────────────────────
# CONFIG — tweak these without touching logic
# ─────────────────────────────────────────────
INTRADAY_CFG = {
    # Hard filters (all must pass)
    "stage":          "Stage 2",       # must contain this string
    "rsi_min":        45.0,
    "rsi_max":        72.0,
    "adx_min":        18.0,
    "rvol_min":       0.8,             # relative volume floor
    "ascr_min":       20.0,            # min activity score (liquidity)
    "trend_allow":    ["Uptrend", "Sideways"],  # allowed tren values
    "above_sma200":   True,            # g200 must be True

    # Scoring weights  (higher = more important)
    "w_rvol":         25.0,   # volume surge is the #1 intraday signal
    "w_adx":          1.5,    # trend strength
    "w_rsi":          0.5,    # momentum (mild weight — avoids chasing overbought)
    "w_ascr_log":     5.0,    # log(ascr) — liquidity, log-scaled to avoid mega-caps dominating
    "w_positive_day": 10.0,   # bonus if chan > 0 (positive day)
    "w_uptrend":      8.0,    # bonus if tren == "Uptrend" (vs Sideways)
    "w_strong_trend": 8.0,    # bonus if tstr == "Strong"
    "w_bbbreakout":   5.0,    # bonus if BB breakout up active

    "top_n":          10,
}

SWING_CFG = {
    # Hard filters
    "stage":          "Stage 2",
    "wrsi_min":       50.0,
    "wrsi_max":       75.0,            # avoid overbought weekly (>75 = exhaustion risk)
    "adx_min":        18.0,
    "dlper_min":      35.0,            # delivery % floor — filters speculative churn
    "delt_max":       25.0,            # within 25% of 52W high (delt is distance below high)
    "above_sma200":   True,
    "above_sma050":   True,
    "trend_require":  "Uptrend",       # swing needs confirmed uptrend (stricter than intraday)

    # Scoring weights
    "w_wrsi":         1.0,    # weekly RSI momentum
    "w_adx":          1.0,    # trend strength
    "w_dlper":        0.3,    # delivery % (conviction)
    "w_delt_prox":    0.5,    # proximity to 52W high: (25 - delt) → higher = closer to high
    "w_pcon":         0.3,    # pattern confidence (0–99)
    "w_strong_trend": 15.0,   # strong trend bonus
    "w_bbsqueeze":    10.0,   # BB squeeze bonus (compressed vol = breakout potential)
    "w_premium_zone": 8.0,    # bonus if zone is Premium or Near Premium
    "w_ascr_log":     2.0,    # liquidity (log-scaled)

    "top_n":          10,
}

OUTPUT_COLS_INTRADAY = [
    "symb", "clos", "chan", "rvol", "arnk", "ascr",
    "rsi", "adx", "tren", "tstr", "vola",
    "zone", "delt", "bbup", "bbsq", "mpat", "pcon",
    "sect", "score"
]

OUTPUT_COLS_SWING = [
    "symb", "clos", "chan", "wrsi", "ws30", "DlPer",
    "delt", "adx", "tstr", "vola",
    "zone", "mpat", "pcon", "xpat",
    "sect", "score"
]


# ─────────────────────────────────────────────
# LOAD & VALIDATE
# ─────────────────────────────────────────────
def load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        sys.exit(f"[ERROR] File not found: {path}")
    df = pd.read_csv(path)
    required = {"stge", "g200", "g050", "rsi", "wrsi", "adx", "rvol",
                "ascr", "chan", "tren", "tstr", "vola", "zone", "delt",
                "DlPer", "bbup", "bbsq", "mpat", "pcon", "ws30", "clos", "sect"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"[ERROR] Missing columns in CSV: {missing}")
    # fill optional cols that may be NaN
    df["pcon"] = pd.to_numeric(df["pcon"], errors="coerce").fillna(0)
    df["xpat"] = df["xpat"].fillna("")
    df["mpat"] = df["mpat"].fillna("No Pattern")
    return df


# ─────────────────────────────────────────────
# INTRADAY FILTER + SCORE
# ─────────────────────────────────────────────
def screen_intraday(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, int]:
    """
    LOGIC EXPLAINED
    ───────────────
    Hard filters (binary pass/fail):
      1. Stage 2 (Uptrend) — Weinstein stage ensures we are in a rising phase,
         not a basing or declining stock.
      2. Above SMA 200 — primary bull market filter; below this = risky long.
      3. RSI 45–72 — momentum must be present (>45) but not overbought (>72
         risks reversal intraday). Range avoids both sluggish and exhausted stocks.
      4. ADX ≥ 18 — trend must have some strength; ADX < 18 = choppy/ranging,
         directional trades fail on ranging stocks intraday.
      5. RVol ≥ 0.8 — relative volume floor; below this the stock is too thin
         for reliable intraday price action.
      6. Activity score ≥ 20 — ensures enough ₹ liquidity (Price × Vol / 10M)
         for smooth entries and exits.
      7. Trend in [Uptrend, Sideways] — avoids confirmed downtrends.

    Scoring (additive, higher = better intraday candidate):
      • RVol × 25     → volume surge is the single most predictive intraday signal
      • ADX × 1.5     → stronger trend = cleaner directional move
      • RSI × 0.5     → mild momentum nudge without overweighting
      • log(ascr) × 5 → log-scaled liquidity; prevents large-caps from
                        dominating purely due to high absolute activity score
      • +10 if chan > 0    → day is already positive = momentum confirmation
      • +8  if Uptrend     → confirmed trend > sideways
      • +8  if tstr=Strong → strong trend confirmation
      • +5  if bbup=True   → price above upper BB = breakout momentum
    """
    c = cfg
    mask = (
        df["stge"].str.contains(c["stage"], na=False) &
        (df["g200"] == c["above_sma200"]) &
        df["rsi"].between(c["rsi_min"], c["rsi_max"]) &
        (df["adx"] >= c["adx_min"]) &
        (df["rvol"] >= c["rvol_min"]) &
        (df["ascr"] >= c["ascr_min"]) &
        df["tren"].isin(c["trend_allow"])
    )
    pool = df[mask].copy()
    pool_size = len(pool)

    pool["score"] = (
        pool["rvol"]                              * c["w_rvol"] +
        pool["adx"]                               * c["w_adx"] +
        pool["rsi"]                               * c["w_rsi"] +
        np.log1p(pool["ascr"])                    * c["w_ascr_log"] +
        (pool["chan"] > 0).astype(int)            * c["w_positive_day"] +
        (pool["tren"] == "Uptrend").astype(int)   * c["w_uptrend"] +
        (pool["tstr"] == "Strong").astype(int)    * c["w_strong_trend"] +
        (pool["bbup"] == True).astype(int)        * c["w_bbbreakout"]
    )

    # round score for readability
    pool["score"] = pool["score"].round(2)

    # keep only output columns that actually exist
    cols = [c for c in OUTPUT_COLS_INTRADAY if c in pool.columns]
    return pool.sort_values("score", ascending=False).head(c["top_n"])[cols], pool_size


# ─────────────────────────────────────────────
# SWING FILTER + SCORE
# ─────────────────────────────────────────────
def screen_swing(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, int]:
    """
    LOGIC EXPLAINED
    ───────────────
    Hard filters:
      1. Stage 2 — same as intraday; only rising stocks.
      2. Above SMA 200 + SMA 50 — for a weekly swing, we want confirmed
         multi-timeframe alignment. Above 50 ensures the intermediate trend
         is also bullish (intraday only required 200).
      3. Weekly RSI 50–75 — weekly RSI > 50 confirms the larger trend is up.
         Cap at 75 because weekly overbought readings can persist but raise
         the risk of mean reversion killing the swing before target.
      4. ADX ≥ 18 — trend quality filter.
      5. Delivery% ≥ 35 — delivery is the NSE-specific filter for genuine
         accumulation. Low delivery = speculative intraday churn, not the
         institutional buying that drives multi-day swings.
      6. delt ≤ 25 — delta is the % distance BELOW the 52W high.
         delt = 0 means AT the 52W high. We want stocks within 25% of their
         highs; stocks more than 25% below their high have more overhead supply
         (prior buyers waiting to sell) that can cap a swing move.
      7. tren == Uptrend — swing trades need a confirmed trend, not just sideways.

    Scoring:
      • wrsi × 1.0           → weekly momentum
      • adx × 1.0            → trend strength
      • DlPer × 0.3          → delivery conviction
      • (25 − delt) × 0.5    → proximity to 52W high (0 delt = max score here)
      • pcon × 0.3           → chart pattern confidence
      • +15 if tstr=Strong   → strongest bonus; multi-day swings live or die
                               on trend quality
      • +10 if bbsq=True     → BB squeeze means volatility has compressed;
                               expansion (= the swing move) is likely imminent
      • +8 if Premium/Near Premium zone → price in upper half of 52W range
      • log(ascr) × 2        → mild liquidity weight (log-scaled)
    """
    c = cfg
    mask = (
        df["stge"].str.contains(c["stage"], na=False) &
        (df["g200"] == c["above_sma200"]) &
        (df["g050"] == c["above_sma050"]) &
        df["wrsi"].between(c["wrsi_min"], c["wrsi_max"]) &
        (df["adx"] >= c["adx_min"]) &
        (df["DlPer"] >= c["dlper_min"]) &
        (df["delt"] <= c["delt_max"]) &
        (df["tren"] == c["trend_require"])
    )
    pool = df[mask].copy()
    pool_size = len(pool)

    pool["score"] = (
        pool["wrsi"]                                              * c["w_wrsi"] +
        pool["adx"]                                               * c["w_adx"] +
        pool["DlPer"]                                             * c["w_dlper"] +
        (c["delt_max"] - pool["delt"]).clip(lower=0)              * c["w_delt_prox"] +
        pool["pcon"]                                              * c["w_pcon"] +
        (pool["tstr"] == "Strong").astype(int)                    * c["w_strong_trend"] +
        (pool["bbsq"] == True).astype(int)                        * c["w_bbsqueeze"] +
        pool["zone"].isin(["Premium", "Near Premium"]).astype(int)* c["w_premium_zone"] +
        np.log1p(pool["ascr"])                                    * c["w_ascr_log"]
    )

    pool["score"] = pool["score"].round(2)

    cols = [c for c in OUTPUT_COLS_SWING if c in pool.columns]
    return pool.sort_values("score", ascending=False).head(cfg["top_n"])[cols], pool_size


# ─────────────────────────────────────────────
# DISPLAY HELPERS
# ─────────────────────────────────────────────
def print_section(title: str, df: pd.DataFrame, pool_size: int):
    sep = "═" * 90
    print(f"\n{sep}")
    print(f"  {title}  (pool after filters: {pool_size} stocks, showing top {len(df)})")
    print(sep)
    if df.empty:
        print("  No candidates matched the criteria. Try relaxing the filters in CFG.")
        return
    # format score column prominently
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", "{:.2f}".format)
    print(df.to_string(index=False))


def print_logic_summary():
    print("""
╔══════════════════════════════════════════════════════════════════╗
║          SCREENING LOGIC — SHORT SUMMARY                        ║
╠══════════════════════════════════════════════════════════════════╣
║  INTRADAY                                                        ║
║  Hard: Stage2 | g200 | RSI 45-72 | ADX≥18 | RVol≥0.8           ║
║        ascr≥20 | Trend∈{Uptrend,Sideways}                       ║
║  Score: RVol(×25) + ADX(×1.5) + RSI(×0.5) + logAscr(×5)        ║
║         + positive day(+10) + Uptrend(+8) + Strong(+8)          ║
║         + BBbreakout(+5)                                         ║
╠══════════════════════════════════════════════════════════════════╣
║  SWING (weekly)                                                  ║
║  Hard: Stage2 | g200 | g050 | wRSI 50-75 | ADX≥18              ║
║        DlPer≥35 | delt≤25 | Uptrend                             ║
║  Score: wRSI(×1) + ADX(×1) + DlPer(×0.3) + proximity(×0.5)     ║
║         + pcon(×0.3) + Strong(+15) + BBsqueeze(+10)             ║
║         + Premium zone(+8) + logAscr(×2)                        ║
╚══════════════════════════════════════════════════════════════════╝
    """)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    # ── accept CSV path as CLI arg or fall back to default ──
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        # default: look for the most recent snapshot in current dir
        csv_files = [f for f in os.listdir(".") if f.endswith("snapshot.csv")]
        if csv_files:
            csv_path = sorted(csv_files)[-1]   # latest alphabetically
            print(f"[INFO] Auto-detected CSV: {csv_path}")
        else:
            sys.exit("[ERROR] No CSV path provided and no *snapshot.csv found in current dir.\n"
                     "Usage: python screen_stocks.py <path_to_snapshot.csv>")

    df = load_csv(csv_path)
    print(f"[INFO] Loaded {len(df)} rows from {csv_path}")

    print_logic_summary()

    # ── run screens ──
    intraday_results, intraday_pool_size = screen_intraday(df, INTRADAY_CFG)
    swing_results, swing_pool_size    = screen_swing(df, SWING_CFG)

    # ── dual-timeframe confluence (appears in both lists) ──
    both = set(intraday_results["symb"]) & set(swing_results["symb"])

    print_section("INTRADAY CANDIDATES", intraday_results, intraday_pool_size)
    print_section("WEEKLY SWING CANDIDATES", swing_results, swing_pool_size)

    if both:
        print(f"\n{'─'*60}")
        print(f"  ⭐ DUAL-TIMEFRAME CONFLUENCE (in both lists): {', '.join(sorted(both))}")
        print(f"  These have alignment across intraday + weekly — higher conviction.")
        print(f"{'─'*60}")

    # ── optional CSV export ──

    out_all      = "all_picks.csv"


    # Create a consolidated file to avoid confusion
    intraday_results_tagged = intraday_results.assign(screener_type="Intraday")
    swing_results_tagged = swing_results.assign(screener_type="Swing")
    all_picks = pd.concat([intraday_results_tagged, swing_results_tagged], ignore_index=True)
    all_picks.to_csv(out_all, index=False)

    print(f"\n[INFO] Results saved at {out_all}")


if __name__ == "__main__":
    main()
