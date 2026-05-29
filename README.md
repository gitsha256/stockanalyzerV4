# StockAnalyzer V4

StockAnalyzer V4 is a sophisticated technical analysis suite designed for the Indian National Stock Exchange (NSE). It automates the end-to-end workflow of fetching market data, adjusting for corporate actions like stock splits, detecting complex chart patterns, and calculating a wide array of technical indicators.

## Core Features

- **Data Management**:
  - **Automated Fetching**: Retrieves daily bhavcopy and delivery data directly from NSE via `nselib`.
  - **Split Adjustment**: Intelligent detection of stock splits (price drops >30%) and automatic retrospective adjustment of OHLC data.
  - **Holiday Awareness**: Supports custom holiday lists to skip non-trading days during data fetching.

- **Advanced Chart Pattern Recognition**:
  - **Multi-Timeframe Analysis**: Uses daily data for momentum patterns and weekly-resampled data for structural reversals to reduce noise.
  - **Structural Patterns**: Head and Shoulders, Double/Triple Tops & Bottoms, Ascending/Descending Channels, Wedges, and Diamonds.
  - **Momentum Patterns**: Bull/Bear Flags, Pennants, and Cup and Handle.
  - **Mathematical Fitting**: Rounding Bottoms and Tops detected using parabolic (quadratic) regression.
  - **Pattern Traceability**: Provides anchor points (date and price) for detected patterns.

- **Technical Analysis**:
  - **Indicators**: RSI, ADX (Trend Strength), OBV (Volume Flow), Bollinger Bands.
  - **Moving Averages**: SMA 20, 50, 100, and 200.
  - **Price Zones**: Categorizes current price into Equilibrium, Discount, or Premium zones based on 52-week swing highs and lows.
  - **Volume Analysis**: Relative volume calculation and volume spike detection.
  - **Candlestick Patterns**: Full integration with TA-Lib (via `tabox`) for detecting dozens of candle formations.

## Installation

### Prerequisites
- Python 3.8+
- A `symbols.csv` file in the root directory.

### Dependencies
Install the required packages using pip:
```bash
pip install pandas numpy nselib tqdm scipy ta tabox
```

## Usage

### 1. Market Analysis (`analyzer.py`)
Run the main script to interact with the CLI:
```bash
python analyzer.py
```
**Operation Modes:**
- **Fetch**: Download historical data for a custom date range or a specific number of years.
- **Update**: Sync your local `raw_data.csv` with the latest market closes.
- **Adjust**: Process raw data to handle stock splits and save to `data.csv`.
- **Analyze**: Generate a detailed technical snapshot. You can analyze the latest date or a specific date range.

## Column Abbreviations Guide (`snapshot.csv`)

The analysis output uses abbreviated headers to keep the CSV file concise. Below is the mapping and description:

| Abbr. | Full Name | Description |
| :--- | :--- | :--- |
| **date** | Analysis Date | The date on which the snapshot was generated. |
| **symb** | Symbol | NSE Stock ticker symbol. |
| **clos** | Close | The last traded price (LTP). |
| **stge** | Stage | Weinstein Market Stage (Stage 1: Base, 2: Uptrend, 3: Peak, 4: Downtrend). |
| **wrsi** | Weekly RSI | 14-period Relative Strength Index on a weekly timeframe. |
| **ws30** | Weekly SMA 30 | The 30-week Simple Moving Average (Core for Swing Trading). |
| **volu** | Volume | Total traded quantity for the day. |
| **DlPer** | Delivery % | Percentage of total volume that was delivered. |
| **rvol** | Relative Volume | Current volume divided by its 20-day average. |
| **vspk** | Volume Spike | True if `rvol` is greater than 2.0. |
| **ascr** | Activity Score | Liquidity indicator (`Price * Volume / 10,000,000`). |
| **arnk** | Activity Rank | Ranking based on Activity Score across all symbols. |
| **chan** | % Change | Daily price change percentage (Open to Close). |
| **g200** | > SMA 200 | True if price is above the 200-day Simple Moving Average. |
| **zone** | Price Zone | Categorization (Discount, Equilibrium, Premium) based on 52W Swings. |
| **rsi** | Daily RSI | 14-period Relative Strength Index on a daily timeframe. |
| **delt** | Delta 52H | Percentage distance from the 52-week high. |
| **cand** | Candle Pattern | The most significant daily candlestick pattern detected. |
| **bbup** | BB Breakout Up | True if price closed above the Upper Bollinger Band. |
| **bbdn** | BB Breakout Down | True if price closed below the Lower Bollinger Band. |
| **vtrd** | Volume Trend | Indicates if volume is Increasing or Decreasing vs 10-day average. |
| **bbbw** | BB Bandwidth | Width of Bollinger Bands (Measure of volatility). |
| **bbsq** | BB Squeeze | True if volatility is at a 300-period relative low. |
| **mcdl** | Misc Candles | Other candlestick patterns detected on the same day. |
| **g050** | > SMA 50 | True if price is above the 50-day Simple Moving Average. |
| **g020** | > SMA 20 | True if price is above the 20-day Simple Moving Average. |
| **adx** | ADX | Average Directional Index (Trend strength 0-100). |
| **shgh** | Swing High | The highest price point of the current 52-week swing. |
| **slw** | Swing Low | The lowest price point of the current 52-week swing. |
| **eqb** | Equilibrium | The midpoint between the 52-week swing high and low. |
| **s020-s200**| SMAs | Daily Simple Moving Averages for 20, 50, 100, and 200 periods. |
| **h52h** | 52W High | The highest price reached in the last 252 trading days. |
| **l52l** | 52W Low | The lowest price reached in the last 252 trading days. |
| **vrnk** | Volume Rank | Ranking of the symbol by total traded volume. |
| **rrnk** | R-Vol Rank | Ranking of the symbol by Relative Volume. |
| **tren** | Trend | Primary trend direction (Uptrend, Downtrend, Sideways). |
| **tstr** | Trend Strength | Strength of the trend (Strong, Moderate, Weak). |
| **vola** | Volatility % | Annualized volatility based on the last 21 trading days. |
| **mpat** | Main Pattern | The chart pattern with the highest confidence score. |
| **pcon** | Pattern Conf. | Confidence score for the main pattern (0-99). |
| **psta** | Pattern Start | The date when the detected pattern began forming. |
| **pend** | Pattern End | The date when the detected pattern was completed. |
| **ppnt** | Pattern Points | Coordinates (Date@Price) of the pattern's pivot points. |
| **xpat** | Misc Patterns | Secondary chart patterns detected in the same window. |
| **patt** | All Patterns | Summary string of all detected chart patterns and their scores. |
| **obv** | OBV | On-Balance Volume (Cumulative volume flow). |
| **sect** | Sector | Industry sector classification from `symbols.csv`. |

### 2. SMA Confluence Filter (`sma_filter.py`)
Identify stocks where short-term and medium-term moving averages are converging toward the long-term 200 SMA (a sign of potential consolidation or trend change).
```bash
python sma_filter.py
```
Enter a threshold (e.g., `5` for 5%) to find stocks where SMA 20, 50, and 100 are all within that percentage of the SMA 200.

## Configuration

The `CONFIG` dictionary in `analyzer.py` allows you to fine-tune the engine:
- `MAX_WORKERS`: Adjust threading for faster data fetching.
- `PATTERN_MAX_AGE_DAYS`: Lookback window for chart pattern detection (default 124 days).
- `PATTERN_DAILY_PIVOT_ORDER`: Sensitivity of pivot point detection.

## File Structure

| File | Description |
| :--- | :--- |
| `symbols.csv` | Input: List of symbols, sectors, and holidays. |
| `raw_data.csv` | Storage: Unadjusted historical OHLCV data. |
| `data.csv` | Storage: Split-adjusted data used for analysis. |
| `snapshot.csv` | Output: The final technical analysis report. |
| `workflow.log` | Diagnostics: Detailed execution logs. |

## Disclaimer
*This tool is for educational and research purposes only. Trading stocks involves significant risk. Always perform your own due diligence before making investment decisions.*