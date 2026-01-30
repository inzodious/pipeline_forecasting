"""
Data Exploration: pipeline insights for assumptions and model validation.

Produces a single .xlsx in assumptions_log with three sheets.
Requires: pandas, openpyxl (pip install openpyxl).

Sheets:
1. Creation_Volume: deal creation by month/segment with skipper-adjusted volume.
2. Deal_Size: volume and revenue by month/segment (deal size context).
3. Win_Rates_By_Stage: for each month/segment/stage, how many deals (and revenue) touched that stage and went on to win.

Won = Closed Won or Verbal. Skipper weight = 0.5 for deals whose first stage is Closed Lost.
"""
import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

SKIPPER_WEIGHT = 0.5

def _project_root():
    d = os.path.dirname(os.path.abspath(__file__))
    if os.path.basename(d) == 'draft':
        d = os.path.dirname(d)
    if os.path.basename(d) == 'scripts':
        d = os.path.dirname(d)
    return d

_ROOT = _project_root()
DATA_PATH = os.path.join(_ROOT, 'data', 'fact_snapshots.csv')
ASSUMPTIONS_DIR = os.path.join(_ROOT, 'assumptions_log')
os.makedirs(ASSUMPTIONS_DIR, exist_ok=True)

OUTPUT_XLSX = os.path.join(ASSUMPTIONS_DIR, 'data_exploration.xlsx')


def load_snapshots():
    df = pd.read_csv(DATA_PATH, parse_dates=['date_created', 'date_closed', 'date_snapshot'])
    df = df.sort_values(['deal_id', 'date_snapshot'])
    df['is_closed_won'] = df['stage'].isin(['Closed Won', 'Verbal'])
    df['is_closed_lost'] = df['stage'] == 'Closed Lost'
    df['is_closed'] = df['is_closed_won'] | df['is_closed_lost']
    return df


def build_deal_facts(df):
    first = df.groupby('deal_id').first().reset_index()
    closed = df[df['is_closed']].groupby('deal_id').first().reset_index()[
        ['deal_id', 'date_closed', 'stage', 'net_revenue']
    ]
    closed = closed.rename(columns={
        'stage': 'final_stage',
        'date_closed': 'final_date_closed',
        'net_revenue': 'final_net_revenue',
    })
    deals = first.merge(closed, on='deal_id', how='left')
    deals['created_month'] = deals['date_created'].dt.to_period('M')
    deals['stage'] = deals['final_stage'].fillna('Open')
    deals['date_closed'] = deals['final_date_closed']
    deals['net_revenue'] = deals['final_net_revenue'].fillna(deals['net_revenue'])
    deals['won'] = deals['stage'].isin(['Closed Won', 'Verbal'])

    first_stage = df.groupby('deal_id')['stage'].first()
    skippers = first_stage[first_stage == 'Closed Lost'].index
    deals['volume_weight'] = 1.0
    deals.loc[deals['deal_id'].isin(skippers), 'volume_weight'] = SKIPPER_WEIGHT
    return deals


def build_creation_volume(deals):
    """Month | market_segment | total_volume | total_revenue | adj_volume (skipper adjusted)."""
    g = deals.groupby([deals['created_month'].astype(str), 'market_segment']).agg(
        total_volume=('deal_id', 'count'),
        total_revenue=('net_revenue', 'sum'),
        adj_volume=('volume_weight', 'sum'),
    ).reset_index()
    g = g.rename(columns={'created_month': 'month'})
    return g[['month', 'market_segment', 'total_volume', 'total_revenue', 'adj_volume']]


def build_deal_size(deals):
    """Month | market_segment | total_volume | total_revenue."""
    g = deals.groupby([deals['created_month'].astype(str), 'market_segment']).agg(
        total_volume=('deal_id', 'count'),
        total_revenue=('net_revenue', 'sum'),
    ).reset_index()
    g = g.rename(columns={'created_month': 'month'})
    return g[['month', 'market_segment', 'total_volume', 'total_revenue']]


def build_win_rates_by_stage(snapshots, deals):
    """
    For each (month, market_segment, stage): deals that touched that stage in that month;
    total_volume, total_won_volume, total_revenue, total_won_revenue, win_rate % by count and revenue.
    """
    snapshots = snapshots.copy()
    snapshots['month'] = snapshots['date_snapshot'].dt.to_period('M').astype(str)

    # One row per (deal_id, month, market_segment, stage) — each deal counted once per stage-month it touched
    touch = snapshots[['deal_id', 'month', 'market_segment', 'stage']].drop_duplicates()

    deal_outcomes = deals[['deal_id', 'won', 'net_revenue']].drop_duplicates(subset='deal_id')
    touch = touch.merge(deal_outcomes, on='deal_id', how='left')
    touch['won_revenue'] = touch['net_revenue'].where(touch['won'].fillna(False), 0)

    agg = touch.groupby(['month', 'market_segment', 'stage']).agg(
        total_volume=('deal_id', 'count'),
        total_won_volume=('won', 'sum'),
        total_revenue=('net_revenue', 'sum'),
        total_won_revenue=('won_revenue', 'sum'),
    ).reset_index()
    agg['total_won_revenue'] = agg['total_won_revenue'].fillna(0)
    agg['win_rate_pct_count'] = np.where(agg['total_volume'] > 0, 100 * agg['total_won_volume'] / agg['total_volume'], np.nan)
    agg['win_rate_pct_revenue'] = np.where(agg['total_revenue'] > 0, 100 * agg['total_won_revenue'] / agg['total_revenue'], np.nan)
    return agg


def main():
    print("Loading snapshots and building deal facts...")
    snapshots = load_snapshots()
    deals = build_deal_facts(snapshots)

    print("Building Creation Volume by month / market segment...")
    creation_volume = build_creation_volume(deals)

    print("Building Deal Size by month / market segment...")
    deal_size = build_deal_size(deals)

    print("Building Win Rates by Stage (month / market segment / stage)...")
    win_rates = build_win_rates_by_stage(snapshots, deals)

    print(f"Writing {OUTPUT_XLSX}...")
    with pd.ExcelWriter(OUTPUT_XLSX, engine='openpyxl') as writer:
        creation_volume.to_excel(writer, sheet_name='Creation_Volume', index=False)
        deal_size.to_excel(writer, sheet_name='Deal_Size', index=False)
        win_rates.to_excel(writer, sheet_name='Win_Rates_By_Stage', index=False)

    print(f"Done. Output: {OUTPUT_XLSX}")
    print("  Sheets: Creation_Volume, Deal_Size, Win_Rates_By_Stage")


if __name__ == '__main__':
    main()
