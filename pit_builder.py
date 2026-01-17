import os
import re
import zipfile
import io
import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from datetime import datetime
from opus import prepare_macro_features

# Configuration
VINTAGE_DIR = 'data/vintages'
OUTPUT_FILE = 'PIT_Macro_Features.csv'
FRED_BASE_URL = 'https://www.stlouisfed.org/research/economists/mccracken/fred-databases'

def update_local_vintages():
    """
    Scrapes FRED-MD website for vintage CSVs and ZIP archives.
    Statically filters for Monthly (MD) data, ignoring Quarterly (QD).
    """
    print(">>> Starting Vintage Update...")
    if not os.path.exists(VINTAGE_DIR):
        os.makedirs(VINTAGE_DIR)

    try:
        response = requests.get(FRED_BASE_URL, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
    except Exception as e:
        print(f"Warning: Failed to reach FRED website: {e}. Proceeding with local data.")
        return

    links = soup.find_all('a', href=True)
    
    # Phase 1: Historical Archives (ZIPs)
    zip_links = [l['href'] for l in links if 'zip' in l['href'].lower()]
    for link in zip_links:
        # Selection: Historical Vintages AND FRED-MD
        # Exclusion: FRED-QD or Quarterly
        if ('historical' in link.lower() and 'fred-md' in link.lower()) and \
           ('qd' not in link.lower() and 'quarterly' not in link.lower()):
            
            full_url = link if link.startswith('http') else 'https://www.stlouisfed.org' + link
            print(f"Checking Archive: {full_url}")
            
            try:
                z_resp = requests.get(full_url)
                with zipfile.ZipFile(io.BytesIO(z_resp.content)) as z:
                    for filename in z.namelist():
                        # Match YYYY-MM.csv, no 'qd'
                        if filename.endswith('.csv') and \
                           re.match(r'^\d{4}-\d{2}\.csv$', os.path.basename(filename)) and \
                           'qd' not in filename.lower():
                            
                            target_path = os.path.join(VINTAGE_DIR, os.path.basename(filename))
                            if not os.path.exists(target_path):
                                with open(target_path, 'wb') as f:
                                    f.write(z.read(filename))
                                print(f"  Extracted: {os.path.basename(filename)}")
            except Exception as e:
                print(f"  Error processing zip {link}: {e}")

    # Phase 2: Recent Monthly Files (CSVs)
    csv_links = [l['href'] for l in links if l['href'].lower().endswith('.csv') or '.csv?' in l['href'].lower()]
    for link in csv_links:
        full_url = link if link.startswith('http') else 'https://www.stlouisfed.org' + link
        
        # Strip query parameters for filename check
        clean_url = link.split('?')[0]
        filename = os.path.basename(clean_url)
        
        # Match YYYY-MM.csv, YYYY-MM-md.csv, or FRED-MD_YYYYmMM.csv
        is_monthly = (re.match(r'^\d{4}-\d{2}(-md)?\.csv$', filename) or 
                    re.match(r'^FRED-MD_\d{4}m\d{2}\.csv$', filename))
        
        if is_monthly and \
           'qd' not in filename.lower() and \
           'quarterly' not in clean_url.lower():
            
            # Map filename to a standard pattern if it has -md suffix
            target_filename = filename.replace('-md.csv', '.csv')
            target_path = os.path.join(VINTAGE_DIR, target_filename)
            
            if not os.path.exists(target_path):
                print(f"Downloading: {filename} -> {target_filename}")
                try:
                    c_resp = requests.get(full_url, timeout=30)
                    c_resp.raise_for_status()
                    with open(target_path, 'wb') as f:
                        f.write(c_resp.content)
                except Exception as e:
                    print(f"  Error downloading {filename}: {e}")

    print(">>> Vintage Update Complete.")


def build_pit_matrix():
    """
    Diagonalizes the vintages into a single PIT state matrix.
    """
    print(">>> Building PIT Matrix...")
    vintage_files = sorted([f for f in os.listdir(VINTAGE_DIR) if f.endswith('.csv')])
    
    # regex for YYYY-MM.csv or FRED-MD_YYYYmMM.csv
    re_pattern = re.compile(r'^(\d{4}-\d{2}|FRED-MD_\d{4}m\d{2})\.csv$')
    vintage_files = [f for f in vintage_files if re_pattern.match(f) and 'qd' not in f.lower()]
    
    if not vintage_files:
        print("Error: No valid vintage files found in data/vintages/")
        return

    # Incremental Update: Load existing data if available
    if os.path.exists(OUTPUT_FILE):
        existing_pit = pd.read_csv(OUTPUT_FILE, index_col=0, parse_dates=True)
        print(f"  Found existing PIT matrix with {len(existing_pit)} rows.")
        
        # Check if we need base history (data before the first vintage timestamp)
        date_match = re.search(r'(\d{4})[-m](\d{2})', vintage_files[0])
        first_v_date = pd.to_datetime(f"{date_match.group(1)}-{date_match.group(2)}-01")
        
        if not existing_pit.empty and existing_pit.index.min() >= first_v_date:
            print(f"  Base history (pre-{first_v_date.date()}) missing. Forcing rebuild to seed history.")
            existing_pit = pd.DataFrame()
            existing_vintages = set()
        else:
            existing_vintages = set(existing_pit.index.strftime('%Y-%m'))
    else:
        existing_pit = pd.DataFrame()
        existing_vintages = set()

    pit_rows = []
    
    for idx, v_file in enumerate(vintage_files):
        # Extract date from filename
        date_match = re.search(r'(\d{4})[-m](\d{2})', v_file)
        if not date_match:
            continue
        v_date_str = f"{date_match.group(1)}-{date_match.group(2)}"
        v_date = pd.to_datetime(v_date_str + '-01')
        
        # Skip if already in PIT matrix
        if v_date_str in existing_vintages:
            continue
            
        v_path = os.path.join(VINTAGE_DIR, v_file)
        
        try:
            # Read CSV
            df_raw = pd.read_csv(v_path)
            
            # Transformation Handling
            # Some files might have leading junk or different structure.
            # Usually sasdate is in the second row (index 0 after header)
            
            # Find sasdate column
            if 'sasdate' not in df_raw.columns:
                # Try reading with header=None and finding sasdate
                df_raw_no_header = pd.read_csv(v_path, header=None)
                # Look for 'sasdate' in any of the first few rows
                found = False
                for r_idx in range(3):
                    if r_idx < len(df_raw_no_header) and 'sasdate' in df_raw_no_header.iloc[r_idx].values:
                        df_raw = pd.read_csv(v_path, header=r_idx)
                        found = True
                        break
                if not found:
                    print(f"  Warning: Could not find 'sasdate' header in {v_file}, skipping.")
                    continue

            # Detect transform row
            transform_row = df_raw.iloc[0]
            is_transform = all(pd.to_numeric(transform_row.iloc[1:], errors='coerce').fillna(0).between(1, 7))
            
            if is_transform:
                df_vintage = df_raw.iloc[1:].copy()
            else:
                df_vintage = df_raw.copy()
            
            # Standardize index
            df_vintage['sasdate'] = pd.to_datetime(df_vintage['sasdate'], utc=True, errors='coerce').dt.tz_localize(None)
            df_vintage = df_vintage.dropna(subset=['sasdate'])
            df_vintage = df_vintage.set_index('sasdate')
                
            for col in df_vintage.columns:
                df_vintage[col] = pd.to_numeric(df_vintage[col], errors='coerce')

            processed_data = clean_vintage_data(df_vintage)
            
            # Call feature engineering (Isolated to this vintage)
            features = prepare_macro_features(processed_data)
            
            if features.empty:
                continue
            
            # BASE HISTORY SEEDING: If this is the oldest vintage and we have no existing matrix,
            # take ALL historical rows before this vintage date.
            if idx == 0 and existing_pit.empty:
                base_history = features[features.index < v_date].copy()
                if not base_history.empty:
                    pit_rows.append(base_history)
                    print(f"  Seed: Populated Base History with {len(base_history)} rows (pre-{v_date.date()})")

            # EXTRACTION: The Nowcast (Last valid row)
            nowcast = features.iloc[-1:].copy()
            
            # Assign Vintage Date as the index
            nowcast.index = [v_date]
            
            pit_rows.append(nowcast)
            print(f"  Processed {v_file} -> {len(features)} rows, Nowcast at {v_date.date()}")
            
        except Exception as e:
            print(f"  Error processing {v_file}: {e}")

    if not pit_rows and existing_pit.empty:
        print("Error: No data extracted and no existing matrix.")
        return

    # Harmonizer: Union-based alignment
    print(">>> Harmonizing and saving...")
    if pit_rows:
        new_pit = pd.concat(pit_rows, axis=0, sort=True)
        final_pit = pd.concat([existing_pit, new_pit], axis=0, sort=True)
        # Ensure index is datetime and sorted
        final_pit.index = pd.to_datetime(final_pit.index)
        # Drop duplicates but keep the last (most recent vintage info)
        final_pit = final_pit[~final_pit.index.duplicated(keep='last')].sort_index()
    else:
        final_pit = existing_pit
    
    # Save to CSV
    final_pit.to_csv(OUTPUT_FILE)
    print(f">>> PIT Matrix updated: {OUTPUT_FILE} ({len(final_pit)} total rows, {len(final_pit.columns)} features)")


def clean_vintage_data(df):
    """
    Simulates opus.py load_fred_md_data logic for a single vintage.
    """
    mapping = {
        'PAYEMS': 'PAYEMS',     # Labor
        'UNRATE': 'UNRATE',     # Labor
        'INDPRO': 'INDPRO',     # Output
        'CUMFNS': 'CAPACITY',   # Output
        'CPIAUCSL': 'CPI',      # Prices
        'WPSFD49207': 'PPI',    # Prices
        'PCEPI': 'PCE',         # Prices
        'FEDFUNDS': 'FEDFUNDS', # Rates
        'GS10': 'GS10',         # Rates
        'HOUST': 'HOUST',       # Housing
        'M2SL': 'M2'            # Money
    }
    
    data = pd.DataFrame(index=df.index)
    for fred_col, target_col in mapping.items():
        if fred_col in df.columns:
            data[target_col] = df[fred_col]
    
    # Derived Financial Variables
    if 'GS10' in data.columns and 'FEDFUNDS' in data.columns:
        data['SPREAD'] = data['GS10'] - data['FEDFUNDS']
    
    if 'BAA' in df.columns and 'AAA' in df.columns:
        data['BAA_AAA'] = df['BAA'] - df['AAA']
        
    log_vars = ['PAYEMS', 'INDPRO', 'CPI', 'PPI', 'PCE', 'HOUST', 'M2']
    for col in log_vars:
        if col in data.columns:
            data[col] = np.log(data[col].replace(0, np.nan))
    
    return data.replace([np.inf, -np.inf], np.nan).dropna()


if __name__ == "__main__":
    update_local_vintages()
    build_pit_matrix()
