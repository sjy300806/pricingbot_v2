"""Build conservative point-in-time SEC Company Facts features.

Uses only separately reported ~quarter-length flow observations. Ratios require
matching fiscal periods, rather than independently choosing the latest metric.
Missing or unmatched facts remain NaN. No financial values are fabricated.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / 'data' / 'sec_raw'
DATA = ROOT / 'data'
TAGS = {
    'revenue': ['RevenueFromContractWithCustomerExcludingAssessedTax', 'Revenues',
                'SalesRevenueNet', 'RevenueFromContractWithCustomerIncludingAssessedTax'],
    'net_income': ['NetIncomeLoss'],
    'operating_cashflow': ['NetCashProvidedByUsedInOperatingActivities'],
    'assets': ['Assets'],
    'liabilities': ['Liabilities'],
}
FLOWS = {'revenue', 'net_income', 'operating_cashflow'}
COLUMNS = ['start', 'period_end', 'filed', 'value', 'accn', 'tag']


def fact_rows(obj, key):
    gaap = obj.get('facts', {}).get('us-gaap', {})
    # Use one consistent revenue concept per company. If the preferred tag has
    # no valid quarter-length observations, try the next tag.
    for tag in TAGS[key]:
        rows = []
        for rec in gaap.get(tag, {}).get('units', {}).get('USD', []):
            if rec.get('form') not in ('10-Q', '10-K') or not rec.get('filed') or not rec.get('end'):
                continue
            start = pd.to_datetime(rec.get('start'), errors='coerce')
            end = pd.to_datetime(rec['end'], errors='coerce')
            filed = pd.to_datetime(rec['filed'], errors='coerce')
            if pd.isna(end) or pd.isna(filed):
                continue
            if key in FLOWS:
                if pd.isna(start) or not 70 <= (end - start).days + 1 <= 110:
                    continue
            elif pd.notna(start):
                continue
            try:
                value = float(rec['val'])
            except (KeyError, TypeError, ValueError):
                continue
            if not np.isfinite(value):
                continue
            rows.append({'start': start, 'period_end': end, 'filed': filed,
                         'value': value, 'accn': rec.get('accn', ''), 'tag': tag})
        if rows:
            return pd.DataFrame(rows, columns=COLUMNS)
    return pd.DataFrame(columns=COLUMNS)


def visible(frame, cutoff, flow=False):
    """Latest visible version for each exact observation period."""
    d = frame.loc[frame.filed <= cutoff].copy()
    if d.empty:
        return d
    keys = ['start', 'period_end'] if flow else ['period_end']
    return (d.sort_values(['filed', 'accn'])
             .drop_duplicates(keys, keep='last')
             .sort_values('period_end'))


def find_matching_flow(frame, period_start, period_end):
    """Match both start and end, to avoid quarter/YTD mixups."""
    d = frame.loc[(frame.start == period_start) & (frame.period_end == period_end)]
    return float(d.iloc[-1].value) if not d.empty else np.nan


def safe_ratio(a, b):
    if pd.isna(a) or pd.isna(b) or not np.isfinite(a) or not np.isfinite(b) or abs(b) < 1e-9:
        return np.nan
    return float(a / b)


def features_one(obj, symbol):
    frames = {key: fact_rows(obj, key) for key in TAGS}
    filed_dates = sorted(set(pd.concat([f.filed for f in frames.values() if not f.empty]).dropna())) if any(not f.empty for f in frames.values()) else []
    records = []
    for filed in filed_dates:
        view = {key: visible(frame, filed, key in FLOWS) for key, frame in frames.items()}
        rev = view['revenue']
        latest_rev = rev.iloc[-1] if not rev.empty else None
        revenue = float(latest_rev.value) if latest_rev is not None else np.nan
        period_start = latest_rev.start if latest_rev is not None else pd.NaT
        period_end = latest_rev.period_end if latest_rev is not None else pd.NaT
        income = find_matching_flow(view['net_income'], period_start, period_end) if latest_rev is not None else np.nan
        cashflow = find_matching_flow(view['operating_cashflow'], period_start, period_end) if latest_rev is not None else np.nan

        yoy = np.nan
        if latest_rev is not None:
            prior = rev.loc[
                (rev.period_end >= period_end - pd.Timedelta(days=380)) &
                (rev.period_end <= period_end - pd.Timedelta(days=350)) &
                ((rev.period_end - rev.start).dt.days - (period_end - period_start).days).abs().le(7)
            ]
            if not prior.empty:
                prior_val = float(prior.iloc[-1].value)
                yoy = safe_ratio(revenue, prior_val) - 1 if prior_val > 0 else np.nan

        # Balance-sheet ratio only if assets and liabilities refer to SAME date.
        assets_frame, liabilities_frame = view['assets'], view['liabilities']
        assets = float(assets_frame.iloc[-1].value) if not assets_frame.empty else np.nan
        asset_end = assets_frame.iloc[-1].period_end if not assets_frame.empty else pd.NaT
        matching_liab = liabilities_frame.loc[liabilities_frame.period_end == asset_end]
        liabilities = float(matching_liab.iloc[-1].value) if not matching_liab.empty else np.nan

        records.append({
            'symbol': symbol,
            'filed_date': filed,
            'available_date': filed + pd.Timedelta(days=1),
            'revenue_growth_yoy': yoy,
            'net_margin': safe_ratio(income, revenue),
            'debt_to_assets': safe_ratio(liabilities, assets),  # legacy name; actually liabilities/assets
            'operating_cashflow_margin': safe_ratio(cashflow, revenue),
            'reported_revenue': revenue,
            'reported_assets': assets,
            'revenue_period_end': period_end,
            'assets_period_end': asset_end,
        })
    return records


def main():
    manifest_path = CACHE / 'manifest.csv'
    if not manifest_path.exists():
        raise SystemExit('Run collect_fundamentals.py first.')
    manifest = pd.read_csv(manifest_path)
    all_rows = []
    for row in manifest.itertuples(index=False):
        obj = json.loads((CACHE / row.raw_file).read_text(encoding='utf-8'))
        result = features_one(obj, row.symbol)
        all_rows.extend(result)
        print(f'{row.symbol}: {len(result)} historical update dates')
    if not all_rows:
        raise SystemExit('No SEC facts found.')
    output = pd.DataFrame(all_rows).sort_values(['symbol', 'available_date'])
    path = DATA / 'fundamental_features.csv'
    output.to_csv(path, index=False)
    print(f'Saved {len(output)} rows -> {path}')
    print('Missing ratios are intentional where matched fiscal periods are unavailable.')
    print('Note: debt_to_assets retains legacy column name but represents liabilities/assets.')


if __name__ == '__main__':
    main()
