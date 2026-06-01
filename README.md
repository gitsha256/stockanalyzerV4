# NSE Stock Analyzer V4

A comprehensive technical analysis suite for the National Stock Exchange (NSE). This project fetches market data, detects price patterns, calculates indicators, and filters for high-conviction intraday and swing trading opportunities.
# run requirements
pip install -r requirements.txt
## 🚀 Project Components

| Script | Description |
| :--- | :--- |
| `analyzer.py` | The main engine. Fetches data from `symbols.csv`, handles stock splits, and generates technical snapshots. |
| `formatter.py` | Converts CSV snapshots to color-coded Excel (.xlsx) files with directional highlighting. |
| `analyzerall.py` | Full-market version of the analyzer. Processes a larger set of symbols from `symbolsall.csv`. |
| `screen_stocks.py` | A heavy-duty ranking tool that reads snapshots and produces a scored list of Intraday and Swing picks. |
| `sma_filter.py` | A utility to find stocks where multiple Simple Moving Averages (20, 50, 100, 200) are converging (confluence). |
| `montecarlo.py` | Options-specific risk management tool providing position sizing and Monte Carlo price simulations for Nifty. |

> Before running `montecarlo.py`, create and activate a Python virtual environment to keep dependencies isolated.
> ```powershell
> python -m venv venv
> venv\Scripts\Activate.ps1
> ```

## 📊 Abbreviation Dictionary

The analysis outputs (e.g., `26-08-25snapshot_all.csv`) use the following column headers:

### Core Data
- **symb**: Stock Symbol.
- **clos**: Last Traded Price (Close).
- **open**: Market Opening Price.
- **high / low**: Daily High and Low prices.
- **chan**: Daily percentage change from Open to Close.
- **volu**: Total Traded Volume.
- **DlPer**: Delivery Percentage (Genuine accumulation indicator).
- **date**: The trading date of the record.
- **sect**: The industry sector the stock belongs to.
- **ascr / arnk**: Activity Score (Liquidity in ₹) and its Rank across the market.

### Technical Indicators
- **rsi / wrsi**: Daily and Weekly Relative Strength Index (Momentum).
- **adx**: Average Directional Index (Trend Strength).
- **obv**: On-Balance Volume (Cumulative volume flow).
- **s020 / s050 / s100 / s200**: Simple Moving Averages (20, 50, 100, 200 days).
- **g200 / g050 / g020**: Boolean (True/False) - is price above the 200, 50, or 20 SMA?
- **ws30**: Weekly 30-period SMA (Weinstein's preferred indicator).
- **vola**: Annualized Volatility (based on 21-day standard deviation).
- **bbup / bbdn**: Boolean - is price breaking out above the upper or below the lower Bollinger Band?
- **bbbw**: Bollinger Bandwidth (Volatility measurement).
- **bbsq**: Bollinger Squeeze (Boolean) - true if volatility is at a 300-day relative low.
- **bbdn**: Boolean - price closing below the lower Bollinger Band.
- **CMF_20**: 20-day Chaikin Money Flow (volume-weighted accumulation/distribution).
- **SUPERT_7_3.0 / SUPERTd_7_3.0**: Supertrend indicator values and trend direction.
- **STOCHk_14_3_3 / STOCHd_14_3_3**: Stochastic oscillator %K and %D values.
- **EMA_21**: 21-day exponential moving average.
- **SQZ_ON / SQZ_OFF / SQZ_NO**: Squeeze state flags for volatility compression/breakout.
- **WILLR_14**: Williams %R momentum oscillator.
- **EFI_13**: Elder Force Index (volume-based momentum).
- **RSI_2**: Short-term 2-period Relative Strength Index.
- **cand / mcdl**: Primary detected Candlestick pattern and list of additional minor patterns.

### Trend & Stage Analysis
- **stge**: Weinstein Stage (Stage 1: Base, Stage 2: Uptrend, Stage 3: Top, Stage 4: Downtrend).
- **tren**: Trend Direction (Uptrend, Sideways, Downtrend).
- **tstr**: Trend Strength (Strong, Moderate, Weak).
- **zone**: Price Location (Premium, Near Premium, Equilibrium, Near Discount, Discount).
- **vrnk / rrnk**: Volume Rank and Relative Volume Rank.
- **delt**: Percentage distance currently below the 52-week high.
- **h52h / l52l**: 52-Week High and Low prices.
- **shgh / slw**: Most recent Swing High and Swing Low prices.
- **eqb**: Equilibrium price (Midpoint between Swing High and Swing Low).

### Volume & Activity
- **rvol**: Relative Volume (Current volume vs. 20-day average).
- **vspk**: Volume Spike (Boolean - true if volume is > 2x average).
- **vtrd**: Volume Trend (Increasing or Decreasing).
- **ascr**: Activity Score (Price × Volume / 10 Million) - measures ₹ liquidity.
- **arnk**: Activity Rank (Ranked liquidity across the processed list).

### Chart Patterns
- **mpat**: Main chart pattern detected (e.g., Cup and Handle, Double Bottom).
- **pcon**: Pattern Confidence score (0–99).
- **patt**: String containing all detected patterns and their scores.
- **xpat**: Miscellaneous patterns detected in the background.
- **ppnt**: Pivot points (Date@Price) defining the detected pattern structure.
- **psta / pend**: The start and end dates of the detected pattern.

## 🛠️ Usage Workflow

### 📡 Operating the Analyzer (`analyzer.py` / `analyzerall.py`)
The analyzer is your database manager. You should generally follow the menus in numerical order:

1.  **Menu 1 - Fetch**: Used for initial setup or downloading historical blocks.
    *   Input a start/end date or a "years back" value (e.g., 3.0) to build your `raw_data.csv`.
2.  **Menu 2 - Update**: Your daily maintenance tool.
    *   It checks the last date in your CSV and only fetches the missing data until today.
3.  **Menu 3 - Adjust**: **Essential for technical accuracy.**
    *   This scans for price gaps caused by stock splits or bonuses and mathematically adjusts historical prices. Without this, your SMAs and RSI will be broken. Generates `data.csv`.
4.  **Menu 4 - Analyze**: The signal generator.
    *   Enter a date range (or press Enter for the latest).
    *   Choose whether to run CPU-intensive Pattern Recognition (Chart/Candle).
    *   Outputs the final `snapshot.csv` (or `snapshot_all.csv`).

### 🔍 Screening and Filtering
Once the snapshots are generated, use the secondary tools:

1.  **Run Screener**: To get the top-ranked Intraday and Swing picks:
    ```bash
    python screen_stocks.py 26-08-25snapshot_all.csv

"""
NSE Stock Screener — Intraday & Weekly Swing Picks
====================================================
Reads your snapshot CSV (same schema as StockAnalyzerV4 output)
and produces two ranked lists:
  1. Intraday candidates
  2. Weekly swing candidates

Filename pattern  : DD-MM-YYsnapshot.csv  (date auto-parsed for display)
Run               : python screen_stocks.py [path/to/snapshot.csv]
Auto-detect       : drops latest *snapshot.csv in current dir if no arg given

Column schema (key columns):
  stge            : Weinstein stage  e.g. "Stage 2 (Uptrend)"
  g200/g050/g020  : bool  price > SMA 200/50/20
  rsi             : float daily RSI-14
  wrsi            : float weekly RSI-14
  adx             : float ADX trend strength
  rvol            : float relative volume (vs 20d avg)
  ascr            : float activity score = Price×Vol/10M
  chan            : float daily % change open→close
  tren            : "Uptrend" / "Downtrend" / "Sideways"
  tstr            : "Strong" / "Moderate" / "Weak"
  vola            : float annualised volatility (21d)
  zone            : "Premium" / "Near Premium" / "Equilibrium" / "Near Discount" / "Discount"
  delt            : float % distance from 52W high  (0 = AT high)
  DlPer           : float delivery % of traded volume
  bbup            : bool  close > upper Bollinger Band
  bbsq            : bool  BB squeeze (300-period volatility low)
  mpat            : str   main chart pattern name
  pcon            : int   pattern confidence 0–99

  ── pandas_ta indicators (pre-computed in snapshot) ──
  CMF_20          : Chaikin Money Flow  (-1 to +1)
  SUPERTd_7_3.0   : Supertrend direction  +1=up / -1=down
  STOCHk_14_3_3   : Stochastic %K  (0–100)
  STOCHd_14_3_3   : Stochastic %D  (0–100)
  EMA_21          : EMA 21-period value
  SQZ_ON          : BB+KC squeeze active  1/0
  SQZ_OFF         : Squeeze just released 1/0
  WILLR_14        : Williams %R  (-100 to 0)
  EFI_13          : Elder Force Index  (large raw values)
  RSI_2           : RSI 2-period  (0–100)
"""
    ```
2.  **SMA Filter**: Run `sma_filter.py` to find "Tightening" setups where multiple SMAs are converging.

### 🟢 Excel Formatting (`formatter.py`)
The formatter is designed to turn raw CSV data into a visually intuitive heat-map of market signals. It processes the snapshot files and applies conditional formatting based on technical consensus.

**Usage:**
1. **Manual**: `python formatter.py path/to/your_snapshot.csv`
2. **Automatic**: `python formatter.py` (It will auto-detect the latest `*snapshot.csv` in your directory).

**Key Features:**
- **Automated Logic**:
    - **Booleans**: Automatically highlights signals like `bbup` (Bollinger Breakout) or `vspk` (Volume Spike).
    - **Trend Alignment**: Color codes `tren` (Direction) and `tstr` (Strength) to help you spot strong uptrends instantly.
    - **Range Analysis**: Validates `rsi`, `wrsi`, and `Stochastics` against ideal entry/exit zones.
    - **Contrarian Highlighting**: Specifically flags `zone` (Discount/Premium) and `delt` (52W High distance) to identify mean-reversion or breakout setups.
- **Multi-Sheet Architecture**:
    - **Main Analysis**: The primary dashboard with frozen headers and auto-adjusted column widths.
    - **Chart Patterns**: Separates verbose pattern data (`psta`, `pend`, `ppnt`) to keep the main view clean.
    - **Legend**: Includes an embedded guide explaining every color rule and condition.

**Visual Standards:**
- **Light Green (#C6EFCE)**: Bullish / Signal Active.
- **Light Red (#FFC7CE)**: Bearish / Signal Inactive.
- **Dark Fills with White Text**: Highlights extreme conditions (e.g., very close to or far from 52W Highs).

## 📋 Requirements

- Python 3.12+
- `pandas`, `numpy`, `nselib`, `pandas-ta`, `scipy`, `plotly`, `streamlit`, `yfinance`, `requests`, `openpyxl`

---
*Disclaimer: This tool is for educational and analytical purposes only. Trading involves significant risk.*