import pandas as pd
import os
import sys
import glob
from openpyxl import load_workbook
from openpyxl.styles import PatternFill, Font, Alignment
from openpyxl.utils import get_column_letter

# --- COLOR CONSTANTS ---
GREEN_FILL = PatternFill(start_color='C6EFCE', end_color='C6EFCE', fill_type='solid')
RED_FILL = PatternFill(start_color='FFC7CE', end_color='FFC7CE', fill_type='solid')
DARK_GREEN_FILL = PatternFill(start_color='375623', end_color='375623', fill_type='solid')
DARK_RED_FILL = PatternFill(start_color='9C0006', end_color='9C0006', fill_type='solid')
WHITE_FONT = Font(color='FFFFFF')

def get_style(col_name, val):
    """Returns (fill, font) based on the specific column rules."""
    try:
        # ── BOOLEAN COLUMNS ──
        if col_name in ['g200', 'g050', 'g020', 'bbup', 'bbsq', 'vspk']:
            if str(val).strip().lower() in ['true', '1', '1.0']: return GREEN_FILL, None
            if str(val).strip().lower() in ['false', '0', '0.0']: return RED_FILL, None

        # ── DIRECTIONAL COLUMNS ──
        if col_name == 'chan':
            if val > 0: return GREEN_FILL, None
            if val < 0: return RED_FILL, None
        
        if col_name == 'CMF_20':
            if val > 0.05: return GREEN_FILL, None
            if val < -0.05: return RED_FILL, None
            
        if col_name == 'SUPERTd_7_3.0':
            if val == 1.0: return GREEN_FILL, None
            if val == -1.0: return RED_FILL, None
            
        if col_name == 'SQZ_ON' and val == 1:
            return GREEN_FILL, None
            
        if col_name == 'EFI_13':
            if val > 0: return GREEN_FILL, None
            if val < 0: return RED_FILL, None
            
        if col_name == 'tren':
            if val == "Uptrend": return GREEN_FILL, None
            if val == "Downtrend": return RED_FILL, None
            
        if col_name == 'tstr':
            if val == "Strong": return GREEN_FILL, None
            if val == "Weak": return RED_FILL, None
            
        if col_name == 'vtrd':
            if val == "Increasing": return GREEN_FILL, None
            if val == "Decreasing": return RED_FILL, None
            
        if col_name == 'stge':
            if "Stage 2" in str(val): return GREEN_FILL, None
            if "Stage 4" in str(val): return RED_FILL, None

        # ── RANGE COLUMNS ──
        if col_name == 'rsi':
            if 45 <= val <= 60: return GREEN_FILL, None
            if val < 15 or val > 70: return RED_FILL, None
            
        if col_name == 'wrsi':
            if 50 <= val <= 75: return GREEN_FILL, None
            if val < 40 or val > 80: return RED_FILL, None
            
        if col_name == 'adx':
            if val >= 25: return GREEN_FILL, None
            if val < 18: return RED_FILL, None
            
        if col_name == 'rvol':
            if val >= 1.5: return GREEN_FILL, None
            if val < 0.8: return RED_FILL, None
            
        if col_name == 'STOCHk_14_3_3':
            if val < 20: return GREEN_FILL, None
            if val > 70: return RED_FILL, None
            
        if col_name == 'WILLR_14':
            if val < -80: return GREEN_FILL, None
            if val > -20: return RED_FILL, None
            
        if col_name == 'RSI_2':
            if val < 10: return GREEN_FILL, None
            if val > 90: return RED_FILL, None
            
        if col_name == 'DlPer':
            if val >= 50: return GREEN_FILL, None
            if val < 25: return RED_FILL, None
            
        if col_name == 'delt':
            if val <= 5: return DARK_RED_FILL, WHITE_FONT
            if val >= 10: return DARK_GREEN_FILL, WHITE_FONT
            
        if col_name == 'zone':
            if val in ["Premium", "Near Premium"]: return RED_FILL, None
            if val in ["Discount", "Near Discount"]: return GREEN_FILL, None

    except:
        pass
    return None, None

def colorize_snapshot(csv_path):
    if not os.path.exists(csv_path):
        print(f"Error: File {csv_path} not found.")
        return

    xlsx_path = csv_path.replace('.csv', '.xlsx')
    print(f"Processing {csv_path}...")

    # Load Data
    df = pd.read_csv(csv_path)
    
    # Split Patterns to second sheet
    pattern_cols = ['psta', 'pend', 'ppnt']
    # Check if cols exist before splitting
    actual_pattern_cols = [c for c in pattern_cols if c in df.columns]
    
    df_main = df.drop(columns=actual_pattern_cols)
    df_patterns = df[['symb'] + actual_pattern_cols].copy() if actual_pattern_cols else pd.DataFrame()

    # Create Excel Writer
    with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
        df_main.to_excel(writer, sheet_name='Main Analysis', index=False)
        if not df_patterns.empty:
            df_patterns.to_excel(writer, sheet_name='Chart Patterns', index=False)
        
        # Legend Sheet Data
        legend_data = [
            ["Column", "Condition", "Color", "Meaning"],
            ["Booleans", "True", "Light Green", "Bullish / Signal Active"],
            ["Booleans", "False", "Light Red", "Bearish / Signal Inactive"],
            ["chan", "> 0", "Light Green", "Price Gain"],
            ["CMF_20", "> 0.05", "Light Green", "Money Inflow"],
            ["SUPERTd", "1.0", "Light Green", "Uptrend Confirm"],
            ["SQZ_ON", "1", "Light Green", "Coiling / Squeeze"],
            ["tren", "Uptrend", "Light Green", "Bullish Trend"],
            ["rsi", "45-72", "Light Green", "Ideal Momentum"],
            ["adx", ">= 25", "Light Green", "Strong Trend"],
            ["rvol", ">= 2.0", "Light Green", "Volume Surge"],
            ["delt", "<= 5%", "Dark Green / White Font", "Near 52W High"],
            ["delt", ">= 30%", "Dark Red / White Font", "Far from High"],
            ["zone", "Premium", "Light Green", "Over Upper Midpoint"],
            ["zone", "Discount", "Light Red", "Below Lower Midpoint"]
        ]
        pd.DataFrame(legend_data).to_excel(writer, sheet_name='Legend', index=False, header=False)

    # Re-open with openpyxl for styling
    wb = load_workbook(xlsx_path)
    ws = wb['Main Analysis']

    # 1. Freeze Top Row
    ws.freeze_panes = 'A2'

    # 2. Apply Colors and Auto-Fit
    headers = [cell.value for cell in ws[1]]
    
    for row_idx, row in enumerate(ws.iter_rows(min_row=2, max_row=ws.max_row), start=2):
        for col_idx, cell in enumerate(row):
            col_name = headers[col_idx]
            fill, font = get_style(col_name, cell.value)
            if fill: cell.fill = fill
            if font: cell.font = font

    # Auto-fit Column Widths
    for i, column_cells in enumerate(ws.columns, start=1):
        max_length = 0
        column = get_column_letter(i)
        for cell in column_cells:
            try:
                if len(str(cell.value)) > max_length:
                    max_length = len(str(cell.value))
            except:
                pass
        adjusted_width = (max_length + 2)
        ws.column_dimensions[column].width = min(adjusted_width, 40) # Cap width

    # Style Legend Sheet
    if 'Legend' in wb.sheetnames:
        lws = wb['Legend']
        for row in lws.iter_rows():
            for cell in row:
                cell.alignment = Alignment(horizontal='left')
        lws.column_dimensions['A'].width = 20
        lws.column_dimensions['B'].width = 30
        lws.column_dimensions['C'].width = 25
        lws.column_dimensions['D'].width = 40

    wb.save(xlsx_path)
    print(f"Done! Excel file saved: {xlsx_path}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        target = sys.argv[1]
    else:
        # Auto-detect latest snapshot in current directory
        # Prioritize files NOT ending in _all if mixed, otherwise just take latest
        files = sorted(glob.glob("*snapshot.csv"))
        if not files:
            files = sorted(glob.glob("*snapshot_all.csv"))
            
        if not files:
            print("No snapshot CSV found in current directory.")
            sys.exit(1)
        target = files[-1]

    colorize_snapshot(target)
