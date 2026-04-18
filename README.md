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
