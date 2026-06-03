import pandas as pd
import numpy as np
import sys
import os
import re
from datetime import datetime

# ─────────────────────────────────────────────────────────────
# CONFIG — tweak weights/thresholds here without touching logic
# ─────────────────────────────────────────────────────────────

INTRADAY_CFG = {
    # ── Hard filters (ALL must pass) ──
    "stage":            "Stage 2",          # stge must contain this
    "rsi_min":          45.0,
    "rsi_max":          72.0,
    "adx_min":          18.0,
    "rvol_min":         0.8,                # relative volume floor
    "ascr_min":         20.0,               # ₹ liquidity floor
    "trend_allow":      ["Uptrend", "Sideways"],
    "above_sma200":     True,

    # ── Base scoring weights ──
    "w_rvol":           25.0,   # volume surge — #1 intraday signal
    "w_adx":            1.5,    # trend strength
    "w_rsi":            0.5,    # momentum (mild — avoids chasing overbought)
    "w_ascr_log":       5.0,    # log(ascr) liquidity — log-scaled prevents mega-caps dominating
    "w_positive_day":   10.0,   # bonus: chan > 0
    "w_uptrend":        8.0,    # bonus: tren == Uptrend
    "w_strong_trend":   8.0,    # bonus: tstr == Strong
    "w_bbbreakout":     5.0,    # bonus: bbup == True

    # ── New indicator weights (all pre-computed in snapshot) ──
    "w_supertrend":     10.0,   # bonus: SUPERTd == +1  (uptrend confirmed)
    "w_cmf":            8.0,    # CMF × weight  (negative CMF penalises score)
    "w_squeeze_on":     6.0,    # bonus: SQZ_ON == 1  (coiling before breakout)
    "w_stoch_setup":    5.0,    # bonus: STOCHk < 80 AND %k > %d  (bullish, not OB)
    "w_efi_positive":   3.0,    # bonus: EFI_13 > 0  (buying force present)

    "top_n":            10,
}

SWING_CFG = {
    # ── Hard filters ──
    "stage":            "Stage 2",
    "wrsi_min":         50.0,
    "wrsi_max":         75.0,               # cap: >75 = weekly exhaustion risk
    "adx_min":          18.0,
    "dlper_min":        35.0,               # delivery % — filters speculative churn
    "delt_max":         25.0,               # within 25% of 52W high
    "above_sma200":     True,
    "above_sma050":     True,
    "trend_require":    "Uptrend",

    # ── Base scoring weights ──
    "w_wrsi":           1.0,    # weekly RSI momentum
    "w_adx":            1.0,    # trend strength
    "w_dlper":          0.3,    # delivery conviction
    "w_delt_prox":      0.5,    # proximity to 52W high: (25−delt) → higher = closer
    "w_pcon":           0.3,    # chart pattern confidence
    "w_strong_trend":   15.0,   # largest bonus — swing lives/dies on trend quality
    "w_bbsqueeze":      10.0,   # BB squeeze: compressed vol = breakout potential
    "w_premium_zone":   8.0,    # bonus: zone Premium or Near Premium
    "w_ascr_log":       2.0,    # liquidity (log-scaled)

    # ── New indicator weights ──
    "w_supertrend":     12.0,   # bonus: SUPERTd == +1  (multi-day confirmation)
    "w_cmf":            10.0,   # CMF × weight  (institutional accumulation)
    "w_squeeze_on":     8.0,    # bonus: SQZ_ON == 1  (imminent swing move)
    "w_willr_pullback": 6.0,    # bonus: WILLR < -60  (pulled back, entry timing)
    "w_ema21_support":  5.0,    # bonus: clos > EMA_21  (dynamic support intact)
    "w_stoch_setup":    4.0,    # bonus: STOCHk < 70 AND %k > %d

    "top_n":            10,
}

# ── Output column lists (missing cols auto-skipped) ──
OUTPUT_COLS_INTRADAY = [
    "symb", "clos", "chan", "rvol", "arnk", "ascr",
    "rsi", "adx", "tren", "tstr", "vola",
    "zone", "delt", "bbup", "bbsq",
    "SUPERTd_7_3.0", "CMF_20", "SQZ_ON", "STOCHk_14_3_3", "EFI_13",
    "mpat", "pcon", "sect", "score",
]

OUTPUT_COLS_SWING = [
    "symb", "clos", "chan", "wrsi", "ws30", "DlPer",
    "delt", "adx", "tstr", "vola", "zone",
    "SUPERTd_7_3.0", "CMF_20", "SQZ_ON", "WILLR_14", "EMA_21",
    "STOCHk_14_3_3",
    "mpat", "pcon", "xpat", "sect", "score",
]


# ─────────────────────────────────────────────
# LOAD & VALIDATE
# ─────────────────────────────────────────────
def load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        sys.exit(f"[ERROR] File not found: {path}")
    df = pd.read_csv(path)

    # core columns that must exist
    required = {
        "stge", "g200", "g050", "rsi", "wrsi", "adx", "rvol",
        "ascr", "chan", "tren", "tstr", "vola", "zone", "delt",
        "DlPer", "bbup", "bbsq", "mpat", "pcon", "ws30", "clos", "sect",
    }
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"[ERROR] Missing required columns: {missing}")

    # new indicator columns — warn if absent but don't abort
    new_cols = [
        "CMF_20", "SUPERTd_7_3.0", "STOCHk_14_3_3", "STOCHd_14_3_3",
        "EMA_21", "SQZ_ON", "SQZ_OFF", "WILLR_14", "EFI_13", "RSI_2",
    ]
    missing_new = [c for c in new_cols if c not in df.columns]
    if missing_new:
        print(f"[WARN] New indicator columns not found (scoring will skip them): {missing_new}")

    # ── type coercions ──
    bool_cols = ["g200", "g050", "g020", "bbup", "bbsq"]
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].map(
                lambda x: True if str(x).strip().lower() in ("true", "1", "yes") else False
            )

    int_cols = ["SQZ_ON", "SQZ_OFF", "SQZ_NO"]
    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    float_cols = [
        "CMF_20", "SUPERTd_7_3.0", "STOCHk_14_3_3", "STOCHd_14_3_3",
        "EMA_21", "WILLR_14", "EFI_13", "RSI_2",
    ]
    for col in float_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # fill optional / nullable cols
    df["pcon"]  = pd.to_numeric(df["pcon"], errors="coerce").fillna(0)
    df["xpat"]  = df.get("xpat", pd.Series([""] * len(df))).fillna("")
    df["mpat"]  = df["mpat"].fillna("No Pattern")

    return df


def parse_date_from_filename(path: str) -> str:
    """Extract display date from filename like 29-05-26snapshot.csv → 29 May 2026"""
    name = os.path.basename(path)
    m = re.match(r"(\d{2})-(\d{2})-(\d{2})snapshot", name)
    if m:
        dd, mm, yy = m.groups()
        try:
            dt = datetime.strptime(f"{dd}-{mm}-20{yy}", "%d-%m-%Y")
            return dt.strftime("%d %b %Y")
        except ValueError:
            pass
    return name


# ─────────────────────────────────────────────
# INTRADAY FILTER + SCORE
# ─────────────────────────────────────────────
def screen_intraday(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, int]:
    """
    Hard filters:
      Stage 2 | g200 | RSI 45-72 | ADX≥18 | RVol≥0.8 | ascr≥20 | Trend∈{Up,Sideways}

    Base score:
      RVol×25  ADX×1.5  RSI×0.5  logAscr×5
      +10 positive day  +8 Uptrend  +8 Strong  +5 BB breakout

    New indicator score:
      +10  SUPERTd == +1          (trend direction confirmed)
      CMF×8                       (money flow; negative penalises)
      +6   SQZ_ON                 (coiling = breakout imminent)
      +5   STOCHk<80 AND k>d      (bullish setup, not overbought)
      +3   EFI_13 > 0             (buying force present)
    """
    c = cfg
    mask = (
        df["stge"].str.contains(c["stage"], na=False) &
        (df["g200"] == True) &
        df["rsi"].between(c["rsi_min"], c["rsi_max"]) &
        (df["adx"]  >= c["adx_min"]) &
        (df["rvol"] >= c["rvol_min"]) &
        (df["ascr"] >= c["ascr_min"]) &
        df["tren"].isin(c["trend_allow"])
    )
    pool = df[mask].copy()
    pool_size = len(pool)

    # ── base score ──
    pool["score"] = (
        pool["rvol"]                             * c["w_rvol"] +
        pool["adx"]                              * c["w_adx"] +
        pool["rsi"]                              * c["w_rsi"] +
        np.log1p(pool["ascr"])                   * c["w_ascr_log"] +
        (pool["chan"] > 0).astype(int)           * c["w_positive_day"] +
        (pool["tren"] == "Uptrend").astype(int)  * c["w_uptrend"] +
        (pool["tstr"] == "Strong").astype(int)   * c["w_strong_trend"] +
        (pool["bbup"] == True).astype(int)       * c["w_bbbreakout"]
    )

    # ── new indicator scores ──
    if "SUPERTd_7_3.0" in pool.columns:
        pool["score"] += (pool["SUPERTd_7_3.0"] == 1).astype(int) * c["w_supertrend"]

    if "CMF_20" in pool.columns:
        # clip to ±1 just in case; negative CMF subtracts from score
        pool["score"] += pool["CMF_20"].clip(-1, 1) * c["w_cmf"]

    if "SQZ_ON" in pool.columns:
        pool["score"] += pool["SQZ_ON"] * c["w_squeeze_on"]

    if "STOCHk_14_3_3" in pool.columns and "STOCHd_14_3_3" in pool.columns:
        stoch_ok = (
            (pool["STOCHk_14_3_3"] < 80) &
            (pool["STOCHk_14_3_3"] > pool["STOCHd_14_3_3"])
        )
        pool["score"] += stoch_ok.astype(int) * c["w_stoch_setup"]

    if "EFI_13" in pool.columns:
        pool["score"] += (pool["EFI_13"] > 0).astype(int) * c["w_efi_positive"]

    pool["score"] = pool["score"].round(2)
    cols = [col for col in OUTPUT_COLS_INTRADAY if col in pool.columns]
    return pool.sort_values("score", ascending=False).head(c["top_n"])[cols], pool_size


# ─────────────────────────────────────────────
# SWING FILTER + SCORE
# ─────────────────────────────────────────────
def screen_swing(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, int]:
    """
    Hard filters:
      Stage 2 | g200 | g050 | wRSI 50-75 | ADX≥18
      DlPer≥35 | delt≤25 | tren==Uptrend

    Base score:
      wRSI×1  ADX×1  DlPer×0.3  proximity×0.5  pcon×0.3
      +15 Strong  +10 BBsqueeze  +8 Premium zone  logAscr×2

    New indicator score:
      +12  SUPERTd == +1          (multi-day trend confirmation)
      CMF×10                      (institutional accumulation; negative penalises)
      +8   SQZ_ON                 (compressed vol = imminent swing)
      +6   WILLR < -60            (pulled back to entry zone)
      +5   clos > EMA_21          (dynamic support intact)
      +4   STOCHk<70 AND k>d      (bullish momentum setup)
    """
    c = cfg
    mask = (
        df["stge"].str.contains(c["stage"], na=False) &
        (df["g200"] == True) &
        (df["g050"] == True) &
        df["wrsi"].between(c["wrsi_min"], c["wrsi_max"]) &
        (df["adx"]   >= c["adx_min"]) &
        (df["DlPer"] >= c["dlper_min"]) &
        (df["delt"]  <= c["delt_max"]) &
        (df["tren"]  == c["trend_require"])
    )
    pool = df[mask].copy()
    pool_size = len(pool)

    # ── base score ──
    pool["score"] = (
        pool["wrsi"]                                               * c["w_wrsi"] +
        pool["adx"]                                                * c["w_adx"] +
        pool["DlPer"]                                              * c["w_dlper"] +
        (c["delt_max"] - pool["delt"]).clip(lower=0)               * c["w_delt_prox"] +
        pool["pcon"]                                               * c["w_pcon"] +
        (pool["tstr"] == "Strong").astype(int)                     * c["w_strong_trend"] +
        (pool["bbsq"] == True).astype(int)                         * c["w_bbsqueeze"] +
        pool["zone"].isin(["Premium", "Near Premium"]).astype(int) * c["w_premium_zone"] +
        np.log1p(pool["ascr"])                                     * c["w_ascr_log"]
    )

    # ── new indicator scores ──
    if "SUPERTd_7_3.0" in pool.columns:
        pool["score"] += (pool["SUPERTd_7_3.0"] == 1).astype(int) * c["w_supertrend"]

    if "CMF_20" in pool.columns:
        pool["score"] += pool["CMF_20"].clip(-1, 1) * c["w_cmf"]

    if "SQZ_ON" in pool.columns:
        pool["score"] += pool["SQZ_ON"] * c["w_squeeze_on"]

    if "WILLR_14" in pool.columns:
        pool["score"] += (pool["WILLR_14"] < -60).astype(int) * c["w_willr_pullback"]

    if "EMA_21" in pool.columns:
        pool["score"] += (pool["clos"] > pool["EMA_21"]).astype(int) * c["w_ema21_support"]

    if "STOCHk_14_3_3" in pool.columns and "STOCHd_14_3_3" in pool.columns:
        stoch_ok = (
            (pool["STOCHk_14_3_3"] < 70) &
            (pool["STOCHk_14_3_3"] > pool["STOCHd_14_3_3"])
        )
        pool["score"] += stoch_ok.astype(int) * c["w_stoch_setup"]

    pool["score"] = pool["score"].round(2)
    cols = [col for col in OUTPUT_COLS_SWING if col in pool.columns]
    return pool.sort_values("score", ascending=False).head(cfg["top_n"])[cols], pool_size


# ─────────────────────────────────────────────
# DISPLAY HELPERS
# ─────────────────────────────────────────────
def print_section(title: str, df: pd.DataFrame, pool_size: int):
    sep = "═" * 110
    print(f"\n{sep}")
    print(f"  {title}  (pool after filters: {pool_size} stocks, showing top {len(df)})")
    print(sep)
    if df.empty:
        print("  No candidates matched the criteria. Try relaxing the filters in CFG.")
        return
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 220)
    pd.set_option("display.float_format", "{:.2f}".format)
    print(df.to_string(index=False))


def print_logic_summary():
    print("""
╔══════════════════════════════════════════════════════════════════════════╗
║              SCREENING LOGIC — SUMMARY                                  ║
╠══════════════════════════════════════════════════════════════════════════╣
║  INTRADAY                                                                ║
║  Hard : Stage2 | g200 | RSI 45-72 | ADX≥18 | RVol≥0.8                  ║
║         ascr≥20 | Trend∈{Uptrend,Sideways}                              ║
║  Base : RVol(×25) ADX(×1.5) RSI(×0.5) logAscr(×5)                      ║
║         +10 posDay  +8 Uptrend  +8 Strong  +5 BBbreakout                ║
║  New  : +10 SUPERTd=+1  CMF×8  +6 SQZ_ON  +5 Stoch setup  +3 EFI>0    ║
╠══════════════════════════════════════════════════════════════════════════╣
║  SWING (weekly)                                                          ║
║  Hard : Stage2 | g200 | g050 | wRSI 50-75 | ADX≥18                     ║
║         DlPer≥35 | delt≤25 | Uptrend                                    ║
║  Base : wRSI(×1) ADX(×1) DlPer(×0.3) proximity(×0.5) pcon(×0.3)        ║
║         +15 Strong  +10 BBsqueeze  +8 PremZone  logAscr(×2)            ║
║  New  : +12 SUPERTd=+1  CMF×10  +8 SQZ_ON  +6 WILLR<-60               ║
║         +5 clos>EMA21  +4 Stoch setup                                   ║
╚══════════════════════════════════════════════════════════════════════════╝
    """)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    # ── resolve CSV path ──
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        # Sort files by modification time so the most recently created file is truly last
        snapshot_files = sorted([f for f in os.listdir(".") if f.endswith("snapshot.csv") and not f.endswith("snapshot_all.csv")], key=os.path.getmtime)
        snapshot_all_files = sorted([f for f in os.listdir(".") if f.endswith("snapshot_all.csv")], key=os.path.getmtime)

        if snapshot_files and snapshot_all_files:
            print("Choose snapshot source:")
            print("1. snapshot.csv")
            print("2. snapshot_all.csv")
            choice = input("Enter choice [default 1]: ").strip()
            if choice == '2':
                csv_path = snapshot_all_files[-1]
            else:
                csv_path = snapshot_files[-1]
            print(f"[INFO] Using CSV: {csv_path}")
        elif snapshot_files:
            csv_path = snapshot_files[-1]
            print(f"[INFO] Auto-detected CSV: {csv_path}")
        elif snapshot_all_files:
            csv_path = snapshot_all_files[-1]
            print(f"[INFO] Auto-detected CSV: {csv_path}")
        else:
            sys.exit(
                "[ERROR] No CSV path provided and no snapshot file found in current dir.\n"
                "Usage: python screen_stocks.py <path_to/DD-MM-YYsnapshot.csv>"
            )

    snap_date = parse_date_from_filename(csv_path)

    df = load_csv(csv_path)
    print(f"[INFO] Loaded {len(df)} stocks — snapshot date: {snap_date}")

    print_logic_summary()

    # ── run screens ──
    intraday_results, intraday_pool = screen_intraday(df, INTRADAY_CFG)
    swing_results,   swing_pool    = screen_swing(df, SWING_CFG)

    print_section(f"INTRADAY CANDIDATES  [{snap_date}]",  intraday_results, intraday_pool)
    print_section(f"WEEKLY SWING CANDIDATES  [{snap_date}]", swing_results, swing_pool)

    # ── dual-timeframe confluence ──
    both = set(intraday_results["symb"]) & set(swing_results["symb"])
    if both:
        print(f"\n{'─'*70}")
        print(f"  ⭐ DUAL-TIMEFRAME CONFLUENCE: {', '.join(sorted(both))}")
        print(f"  Appear in BOTH lists — highest conviction picks.")
        print(f"{'─'*70}")

    # ── export ──
    out_dir  = os.path.dirname(os.path.abspath(csv_path)) or "."
    suffix = "_all" if os.path.basename(csv_path).endswith("snapshot_all.csv") else ""
    out_file = os.path.join(out_dir, f"picks_{snap_date.replace(' ', '_')}{suffix}.csv")
    intraday_tagged = intraday_results.assign(screener_type="Intraday")
    swing_tagged    = swing_results.assign(screener_type="Swing")
    pd.concat([intraday_tagged, swing_tagged], ignore_index=True).to_csv(out_file, index=False)
    print(f"\n[INFO] Results saved → {out_file}")


if __name__ == "__main__":
    
  main()