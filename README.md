# NSE Stock Analyzer V4

A comprehensive technical analysis suite for the National Stock Exchange (NSE). This project fetches market data, detects price patterns, calculates indicators, and filters for high-conviction intraday and swing trading opportunities.

## 🚀 Project Components

| Script | Description |
| :--- | :--- |
| `analyzer.py` | The main engine. Fetches data from `symbols.csv`, handles stock splits, and generates technical snapshots. |
| `analyzerall.py` | Full-market version of the analyzer. Processes a larger set of symbols from `symbolsall.csv`. |
| `screen_stocks.py` | A heavy-duty ranking tool that reads snapshots and produces a scored list of Intraday and Swing picks. |
| `sma_filter.py` | A utility to find stocks where multiple Simple Moving Averages (20, 50, 100, 200) are converging (confluence). |
| `options.py` | Options-specific risk management tool providing position sizing and Monte Carlo price simulations for Nifty. |

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
- **bbup / bbdn**: Boolean - is price breaking out above the Upper or below the Lower Bollinger Band?
- **bbbw**: Bollinger Bandwidth (Volatility measurement).
- **bbsq**: Bollinger Squeeze (Boolean) - true if volatility is at a 300-day relative low.
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
    ```
2.  **SMA Filter**: Run `sma_filter.py` to find "Tightening" setups where multiple SMAs are converging.

## 📋 Requirements

- Python 3.12+
- `pandas`, `numpy`, `nselib`, `ta`, `scipy`, `tabox`

---
*Disclaimer: This tool is for educational and analytical purposes only. Trading involves significant risk.*