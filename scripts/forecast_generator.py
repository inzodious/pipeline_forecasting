import pandas as pd
import numpy as np
import os
import random
from datetime import datetime, timedelta
from collections import defaultdict

# ==========================================
# CONFIGURATION & ASSUMPTIONS
# ==========================================
"""
PHILOSOPHY:
- All baselines (win rates, velocity, deal sizes, DSO) are DERIVED from data
- Assumptions are MULTIPLIERS applied on top of data-derived baselines
- No hardcoded floors/ceilings that override actual patterns
"""

ASSUMPTIONS = {
    # Growth Multipliers (applied to data-derived baselines)
    "volume_growth_multiplier": 1.10,       # 10% increase in deal creation volume
    "win_rate_uplift_multiplier": 1.05,     # 5% increase in win efficiency
    "deal_size_inflation": 1.03,            # 3% increase in average deal value
    
    # Management Override (set to 0 to disable)
    "target_annual_revenue": 0,
    
    # Simulation Parameters
    "num_simulations": 500,
    
    # Smoothing (optional - set to 1 to disable)
    "velocity_smoothing_window": 3,
    "confidence_interval_clip": 0.95,
    
    # Stage Classifications
    "won_stages": ["Closed Won", "Verbal"],
    "lost_stage_patterns": ["Closed Lost", "Declined to Bid"],
}

# Paths
BASE_DIR = "exports"
INPUT_DIR = "data"
PATH_RAW_SNAPSHOTS = os.path.join(INPUT_DIR, "fact_snapshots.csv")

# Output Paths
OUTPUT_DRIVER_WIN = os.path.join(BASE_DIR, "driver_win_rates.csv")
OUTPUT_DRIVER_VEL = os.path.join(BASE_DIR, "driver_velocity.csv")
OUTPUT_DRIVER_DSO = os.path.join(BASE_DIR, "driver_dso_cycle_times.csv")
OUTPUT_MONTHLY_FORECAST = os.path.join(BASE_DIR, "forecast_monthly_2026.csv")
OUTPUT_CONFIDENCE = os.path.join(BASE_DIR, "forecast_confidence_intervals.csv")
OUTPUT_EXECUTIVE = os.path.join(BASE_DIR, "executive_summary.txt")
OUTPUT_ASSUMPTIONS = os.path.join(BASE_DIR, "forecast_assumptions_log.csv")
OUTPUT_DATA_PROFILE = os.path.join(BASE_DIR, "data_profile.csv")


# ==========================================
# HELPER: STAGE CLASSIFICATION
# ==========================================

def is_won(stage):
    if pd.isna(stage):
        return False
    return stage in ASSUMPTIONS['won_stages']

def is_lost(stage):
    if pd.isna(stage):
        return False
    for pattern in ASSUMPTIONS['lost_stage_patterns']:
        if pattern.lower() in stage.lower():
            return True
    return False

def is_closed(stage):
    return is_won(stage) or is_lost(stage)


# ==========================================
# DATA PROFILER - Analyze historical patterns
# ==========================================

def profile_historical_data(df):
    """
    Analyze historical data to understand actual patterns.
    This informs what ranges are realistic for this dataset.
    """
    print("  > Profiling Historical Data...")
    
    df_final = df.sort_values('date_snapshot').groupby('deal_id').last().reset_index()
    df_final['is_won_flag'] = df_final['stage'].apply(is_won)
    df_final['is_lost_flag'] = df_final['stage'].apply(is_lost)
    df_final['is_closed_flag'] = df_final['is_won_flag'] | df_final['is_lost_flag']
    
    profile = {}
    profile_rows = []
    
    total_deals = df['deal_id'].nunique()
    closed_deals = df_final['is_closed_flag'].sum()
    won_deals = df_final['is_won_flag'].sum()
    
    overall_win_rate = won_deals / closed_deals if closed_deals > 0 else 0
    
    profile['overall'] = {
        'total_deals': total_deals,
        'closed_deals': closed_deals,
        'won_deals': won_deals,
        'overall_win_rate': overall_win_rate
    }
    
    print(f"    Overall: {total_deals} deals, {won_deals}/{closed_deals} won ({overall_win_rate:.1%})")
    
    for seg in df['market_segment'].unique():
        seg_final = df_final[df_final['market_segment'] == seg]
        seg_closed = seg_final[seg_final['is_closed_flag']]
        
        won_count = seg_final['is_won_flag'].sum()
        lost_count = seg_final['is_lost_flag'].sum()
        total_closed = won_count + lost_count
        
        seg_wr = won_count / total_closed if total_closed > 0 else 0
        
        won_rev = seg_closed[seg_closed['is_won_flag']]['net_revenue'].sum()
        avg_deal = seg_final['net_revenue'].mean()
        min_deal = seg_final['net_revenue'].min()
        max_deal = seg_final['net_revenue'].max()
        
        profile[seg] = {
            'won_count': won_count,
            'lost_count': lost_count,
            'win_rate': seg_wr,
            'avg_deal_size': avg_deal,
            'min_deal_size': min_deal,
            'max_deal_size': max_deal,
            'total_won_revenue': won_rev
        }
        
        profile_rows.append({
            'market_segment': seg,
            'won_count': won_count,
            'lost_count': lost_count,
            'total_closed': total_closed,
            'win_rate': round(seg_wr, 4),
            'avg_deal_size': round(avg_deal, 0),
            'min_deal_size': round(min_deal, 0),
            'max_deal_size': round(max_deal, 0),
            'total_won_revenue': round(won_rev, 0)
        })
        
        print(f"    {seg}: WR={seg_wr:.1%}, Avg=${avg_deal:,.0f}, Range=${min_deal:,.0f}-${max_deal:,.0f}")
    
    pd.DataFrame(profile_rows).to_csv(OUTPUT_DATA_PROFILE, index=False)
    
    return profile


# ==========================================
# DSO CALCULATION (date_created -> date_closed)
# ==========================================

def calculate_dso_distribution(df):
    print("  > Calculating DSO Distribution (from data)...")
    
    df_final = df.sort_values('date_snapshot').groupby('deal_id').last().reset_index()
    
    df_final['is_won_flag'] = df_final['stage'].apply(is_won)
    df_final['is_lost_flag'] = df_final['stage'].apply(is_lost)
    df_closed = df_final[df_final['is_won_flag'] | df_final['is_lost_flag']].copy()
    
    df_closed['date_created'] = pd.to_datetime(df_closed['date_created'])
    df_closed['date_closed'] = pd.to_datetime(df_closed['date_closed'])
    df_closed['cycle_days'] = (df_closed['date_closed'] - df_closed['date_created']).dt.days
    
    df_valid = df_closed[df_closed['cycle_days'] > 0].copy()
    
    print(f"    Valid closed deals for DSO: {len(df_valid):,}")
    
    dso_stats = []
    dso_dict = {}
    
    if len(df_valid) > 0:
        overall_mean = df_valid['cycle_days'].mean()
        overall_std = df_valid['cycle_days'].std()
        overall_median = df_valid['cycle_days'].median()
    else:
        overall_mean, overall_std, overall_median = 60, 20, 60
    
    for seg in df['market_segment'].unique():
        seg_data = df_valid[df_valid['market_segment'] == seg]
        
        if len(seg_data) >= 3:
            mean_dso = seg_data['cycle_days'].mean()
            std_dso = seg_data['cycle_days'].std()
            median_dso = seg_data['cycle_days'].median()
            min_dso = seg_data['cycle_days'].min()
            max_dso = seg_data['cycle_days'].max()
            p25 = seg_data['cycle_days'].quantile(0.25)
            p75 = seg_data['cycle_days'].quantile(0.75)
        else:
            mean_dso = overall_mean
            std_dso = overall_std
            median_dso = overall_median
            min_dso = df_valid['cycle_days'].min() if len(df_valid) > 0 else 30
            max_dso = df_valid['cycle_days'].max() if len(df_valid) > 0 else 90
            p25 = df_valid['cycle_days'].quantile(0.25) if len(df_valid) > 0 else 45
            p75 = df_valid['cycle_days'].quantile(0.75) if len(df_valid) > 0 else 75
        
        std_dso = max(std_dso, 1) if not pd.isna(std_dso) else overall_std
        
        dso_dict[seg] = {
            'mean': mean_dso,
            'std': std_dso,
            'median': median_dso,
            'min': min_dso,
            'max': max_dso,
            'p25': p25,
            'p75': p75
        }
        
        dso_stats.append({
            'market_segment': seg,
            'num_deals': len(seg_data),
            'mean_days': round(mean_dso, 1),
            'std_days': round(std_dso, 1),
            'median_days': round(median_dso, 1),
            'min_days': round(min_dso, 1),
            'max_days': round(max_dso, 1),
            'p25_days': round(p25, 1),
            'p75_days': round(p75, 1)
        })
        
        print(f"    {seg}: mean={mean_dso:.0f}, median={median_dso:.0f}, std={std_dso:.0f}, range=[{min_dso:.0f}-{max_dso:.0f}]")
    
    pd.DataFrame(dso_stats).to_csv(OUTPUT_DRIVER_DSO, index=False)
    
    return dso_dict


# ==========================================
# WIN RATE CALCULATION (from data)
# ==========================================

def calculate_win_rates(df):
    print("  > Calculating Win Rates (from data)...")
    
    df_final = df.sort_values('date_snapshot').groupby('deal_id').last().reset_index()
    
    df_final['is_won_flag'] = df_final['stage'].apply(is_won)
    df_final['is_lost_flag'] = df_final['stage'].apply(is_lost)
    df_final['is_closed_flag'] = df_final['is_won_flag'] | df_final['is_lost_flag']
    
    df_closed = df_final[df_final['is_closed_flag']].copy()
    
    print(f"    Total closed deals: {len(df_closed):,}")
    print(f"    Won: {df_closed['is_won_flag'].sum():,}, Lost: {df_closed['is_lost_flag'].sum():,}")
    
    overall_won = df_closed['is_won_flag'].sum()
    overall_total = len(df_closed)
    overall_wr = overall_won / overall_total if overall_total > 0 else 0.20
    
    win_rates = {}
    stats_rows = []
    
    for seg in df['market_segment'].unique():
        seg_data = df_closed[df_closed['market_segment'] == seg]
        
        won_rev = seg_data[seg_data['is_won_flag']]['net_revenue'].sum()
        lost_rev = seg_data[seg_data['is_lost_flag']]['net_revenue'].sum()
        total_rev = won_rev + lost_rev
        
        won_count = seg_data['is_won_flag'].sum()
        lost_count = seg_data['is_lost_flag'].sum()
        total_count = won_count + lost_count
        
        vol_wr = won_count / total_count if total_count > 0 else overall_wr
        rev_wr = won_rev / total_rev if total_rev > 0 else vol_wr
        
        final_wr = vol_wr
        
        if total_count == 0:
            final_wr = overall_wr
        
        win_rates[seg] = final_wr
        
        stats_rows.append({
            'market_segment': seg,
            'won_revenue': won_rev,
            'lost_revenue': lost_rev,
            'total_closed_revenue': total_rev,
            'won_count': won_count,
            'lost_count': lost_count,
            'total_closed_count': total_count,
            'win_rate_volume': round(vol_wr, 4),
            'win_rate_revenue': round(rev_wr, 4),
            'win_rate_used': round(final_wr, 4)
        })
        
        print(f"    {seg}: {final_wr:.1%} ({won_count}/{total_count} deals)")
    
    pd.DataFrame(stats_rows).to_csv(OUTPUT_DRIVER_WIN, index=False)
    
    return win_rates


# ==========================================
# VELOCITY CALCULATION (from data)
# ==========================================

def calculate_velocity(df):
    print("  > Calculating Velocity (from data)...")
    
    df['date_snapshot'] = pd.to_datetime(df['date_snapshot'])
    df_first_appearance = df.sort_values('date_snapshot').groupby('deal_id').first().reset_index()
    
    df_first_appearance['appear_year'] = df_first_appearance['date_snapshot'].dt.year
    df_first_appearance['appear_month'] = df_first_appearance['date_snapshot'].dt.month
    
    entry_stage_dist = df_first_appearance['stage'].value_counts(normalize=True)
    print(f"    Deal entry stage distribution:")
    for stage, pct in entry_stage_dist.head(6).items():
        print(f"      {stage}: {pct:.1%}")
    
    latest_year = df_first_appearance['appear_year'].max()
    latest_month = df_first_appearance[df_first_appearance['appear_year'] == latest_year]['appear_month'].max()
    
    if latest_month < 12:
        baseline_year = latest_year - 1 if (latest_year - 1) in df_first_appearance['appear_year'].values else latest_year
    else:
        baseline_year = latest_year
    
    df_baseline = df_first_appearance[df_first_appearance['appear_year'] == baseline_year].copy()
    
    print(f"    Velocity baseline year: {baseline_year} ({len(df_baseline):,} total deals)")
    
    velocity_dict = {}
    export_rows = []
    
    for seg in df['market_segment'].unique():
        velocity_dict[seg] = {}
        seg_data = df_baseline[df_baseline['market_segment'] == seg]
        
        annual_count = len(seg_data)
        annual_avg_size = seg_data['net_revenue'].mean() if len(seg_data) > 0 else df_baseline['net_revenue'].mean()
        
        if pd.isna(annual_avg_size):
            annual_avg_size = df['net_revenue'].mean()
        
        monthly_vols = []
        monthly_sizes = []
        
        for m in range(1, 13):
            m_data = seg_data[seg_data['appear_month'] == m]
            
            monthly_vols.append(len(m_data))
            
            if len(m_data) > 0:
                monthly_sizes.append(m_data['net_revenue'].mean())
            else:
                monthly_sizes.append(annual_avg_size)
        
        if ASSUMPTIONS['velocity_smoothing_window'] > 1:
            monthly_vols_smooth = pd.Series(monthly_vols).rolling(
                window=ASSUMPTIONS['velocity_smoothing_window'],
                center=True,
                min_periods=1
            ).mean().tolist()
        else:
            monthly_vols_smooth = monthly_vols
        
        for m in range(1, 13):
            velocity_dict[seg][m] = {
                'vol': monthly_vols_smooth[m - 1],
                'vol_raw': monthly_vols[m - 1],
                'size': monthly_sizes[m - 1]
            }
            
            export_rows.append({
                'market_segment': seg,
                'month': m,
                'raw_volume': monthly_vols[m - 1],
                'smoothed_volume': round(monthly_vols_smooth[m - 1], 2),
                'avg_deal_size': round(monthly_sizes[m - 1], 0)
            })
        
        avg_monthly = annual_count / 12
        print(f"    {seg}: {annual_count} deals/year ({avg_monthly:.1f}/month avg)")
    
    pd.DataFrame(export_rows).to_csv(OUTPUT_DRIVER_VEL, index=False)
    
    velocity_dict['_entry_stage_distribution'] = entry_stage_dist.to_dict()
    velocity_dict['_baseline_year'] = baseline_year
    
    return velocity_dict


# ==========================================
# EXISTING PIPELINE INITIALIZATION
# ==========================================

def initialize_existing_pipeline(df, win_rates, dso_dict):
    print("  > Initializing Existing Pipeline...")
    
    latest_date = df['date_snapshot'].max()
    print(f"    Latest snapshot: {latest_date.date()}")
    
    df_latest = df[df['date_snapshot'] == latest_date].copy()
    
    df_latest['is_closed_flag'] = df_latest['stage'].apply(is_closed)
    df_open = df_latest[~df_latest['is_closed_flag']].copy()
    
    print(f"    Open deals in pipeline: {len(df_open):,}")
    
    if len(df_open) > 0:
        df_open['date_created'] = pd.to_datetime(df_open['date_created'])
        df_open['create_month'] = df_open['date_created'].dt.to_period('M')
        create_dist = df_open['create_month'].value_counts().sort_index()
        
        print(f"    Creation date distribution:")
        for period, count in create_dist.tail(6).items():
            print(f"      {period}: {count} deals")
    
    existing_deals = []
    
    for _, row in df_open.iterrows():
        seg = row['market_segment']
        create_date = pd.to_datetime(row['date_created'])
        
        base_wr = win_rates.get(seg, 0.20)
        adj_wr = base_wr * ASSUMPTIONS['win_rate_uplift_multiplier']
        is_won = random.random() < adj_wr
        
        seg_dso = dso_dict.get(seg, {'mean': 60, 'std': 20})
        
        days_open = (latest_date - create_date).days
        
        total_cycle = int(np.random.normal(seg_dso['mean'], seg_dso['std']))
        total_cycle = max(days_open + random.randint(7, 30), total_cycle)
        
        close_date = create_date + timedelta(days=total_cycle)
        
        existing_deals.append({
            'deal_id': row['deal_id'],
            'segment': seg,
            'revenue': row['net_revenue'],
            'create_date': create_date,
            'close_date': close_date,
            'is_won': is_won,
            'source': 'Existing Pipeline'
        })
    
    if existing_deals:
        close_periods = pd.Series([d['close_date'] for d in existing_deals]).dt.to_period('M').value_counts().sort_index()
        print(f"    Projected close distribution:")
        for period, count in close_periods.head(8).items():
            print(f"      {period}: {count} deals")
    
    return existing_deals


# ==========================================
# FORECAST ENGINE
# ==========================================

class ForecastEngine:
    def __init__(self, win_rates, velocity, dso_dict, data_profile):
        self.win_rates = win_rates
        self.velocity = {k: v for k, v in velocity.items() if not k.startswith('_')}
        self.dso_dict = dso_dict
        self.data_profile = data_profile
        self.forecast_months = pd.date_range('2026-01-01', '2026-12-31', freq='MS')
    
    def generate_monthly_deals(self, month, segment):
        month_num = month.month
        
        if segment not in self.velocity or month_num not in self.velocity[segment]:
            return []
        
        stats = self.velocity[segment][month_num]
        
        base_vol = stats['vol'] * ASSUMPTIONS['volume_growth_multiplier']
        
        if base_vol <= 0:
            num_deals = 0
        else:
            num_deals = np.random.poisson(base_vol)
        
        avg_size = stats['size'] * ASSUMPTIONS['deal_size_inflation']
        
        base_wr = self.win_rates.get(segment, 0.20)
        adj_wr = base_wr * ASSUMPTIONS['win_rate_uplift_multiplier']
        
        seg_dso = self.dso_dict.get(segment, {'mean': 60, 'std': 20})
        
        seg_profile = self.data_profile.get(segment, {})
        min_size = seg_profile.get('min_deal_size', avg_size * 0.5)
        max_size = seg_profile.get('max_deal_size', avg_size * 1.5)
        
        deals = []
        
        for _ in range(num_deals):
            is_won = random.random() < adj_wr
            
            actual_rev = int(random.uniform(min_size, max_size) * ASSUMPTIONS['deal_size_inflation'])
            
            days_in_month = (month + pd.DateOffset(months=1) - timedelta(days=1)).day
            create_day = random.randint(1, days_in_month)
            create_date = month + timedelta(days=create_day - 1)
            
            cycle_days = max(1, int(np.random.normal(seg_dso['mean'], seg_dso['std'])))
            close_date = create_date + timedelta(days=cycle_days)
            
            deals.append({
                'segment': segment,
                'revenue': actual_rev,
                'create_date': create_date,
                'close_date': close_date,
                'is_won': is_won,
                'create_month': month,
                'source': 'Generated 2026'
            })
        
        return deals
    
    def run_simulation(self, existing_deals):
        monthly_results = {month: defaultdict(lambda: {
            'created': 0, 'won_vol': 0, 'won_rev': 0
        }) for month in self.forecast_months}
        
        all_deals = list(existing_deals)
        
        for month in self.forecast_months:
            for segment in self.velocity.keys():
                new_deals = self.generate_monthly_deals(month, segment)
                all_deals.extend(new_deals)
        
        for deal in all_deals:
            close_period = pd.Period(deal['close_date'], freq='M').to_timestamp()
            
            if close_period not in monthly_results:
                continue
            
            seg = deal['segment']
            
            if deal.get('create_month') in self.forecast_months:
                monthly_results[deal['create_month']][seg]['created'] += 1
            
            if deal['is_won']:
                monthly_results[close_period][seg]['won_vol'] += 1
                monthly_results[close_period][seg]['won_rev'] += deal['revenue']
        
        return monthly_results


# ==========================================
# MAIN EXECUTION
# ==========================================

def run_forecast():
    print("=" * 70)
    print("FORECAST GENERATOR V6 - DATA-DRIVEN MODEL")
    print("=" * 70)
    
    os.makedirs(BASE_DIR, exist_ok=True)
    
    print("\n--- 1. Loading Data ---")
    
    if not os.path.exists(PATH_RAW_SNAPSHOTS):
        print(f"ERROR: File not found: {PATH_RAW_SNAPSHOTS}")
        return
    
    df = pd.read_csv(PATH_RAW_SNAPSHOTS)
    df['date_snapshot'] = pd.to_datetime(df['date_snapshot'])
    df['date_created'] = pd.to_datetime(df['date_created'])
    df['date_closed'] = pd.to_datetime(df['date_closed'])
    
    print(f"    Loaded {len(df):,} snapshot records")
    print(f"    Unique deals: {df['deal_id'].nunique():,}")
    print(f"    Date range: {df['date_snapshot'].min().date()} to {df['date_snapshot'].max().date()}")
    print(f"    Segments: {df['market_segment'].unique().tolist()}")
    
    assumptions_log = pd.DataFrame([{
        'assumption': k,
        'value': str(v),
        'run_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    } for k, v in ASSUMPTIONS.items()])
    assumptions_log.to_csv(OUTPUT_ASSUMPTIONS, index=False)
    
    print("\n--- 2. Profiling Historical Data ---")
    data_profile = profile_historical_data(df)
    
    print("\n--- 3. Calculating Drivers (from data) ---")
    dso_dict = calculate_dso_distribution(df)
    win_rates = calculate_win_rates(df)
    velocity = calculate_velocity(df)
    
    print("\n--- 4. Initializing Pipeline ---")
    existing_deals = initialize_existing_pipeline(df, win_rates, dso_dict)
    
    print(f"\n--- 5. Running {ASSUMPTIONS['num_simulations']} Simulations ---")
    
    engine = ForecastEngine(win_rates, velocity, dso_dict, data_profile)
    all_results = []
    
    for sim in range(ASSUMPTIONS['num_simulations']):
        if (sim + 1) % 100 == 0:
            print(f"    Completed {sim + 1}/{ASSUMPTIONS['num_simulations']}...")
        
        result = engine.run_simulation(existing_deals)
        all_results.append(result)
    
    print("\n--- 6. Aggregating Results ---")
    
    segments = [k for k in velocity.keys() if not k.startswith('_')]
    forecast_rows = []
    confidence_rows = []
    
    for month in engine.forecast_months:
        for seg in segments:
            won_vol_vals = [sim[month][seg]['won_vol'] for sim in all_results]
            won_rev_vals = [sim[month][seg]['won_rev'] for sim in all_results]
            
            clip = ASSUMPTIONS['confidence_interval_clip']
            if len(won_rev_vals) > 0 and max(won_rev_vals) > 0:
                lower = np.percentile(won_rev_vals, (1 - clip) * 50)
                upper = np.percentile(won_rev_vals, 50 + clip * 50)
                won_rev_clipped = [v for v in won_rev_vals if lower <= v <= upper]
            else:
                won_rev_clipped = won_rev_vals
            
            forecast_rows.append({
                'forecast_month': month,
                'market_segment': seg,
                'forecasted_won_volume_median': int(np.median(won_vol_vals)) if won_vol_vals else 0,
                'forecasted_won_volume_mean': round(np.mean(won_vol_vals), 1) if won_vol_vals else 0,
                'forecasted_won_revenue_median': int(np.median(won_rev_vals)) if won_rev_vals else 0,
                'forecasted_won_revenue_mean': int(np.mean(won_rev_clipped)) if won_rev_clipped else 0
            })
            
            if won_rev_clipped and max(won_rev_clipped) > 0:
                confidence_rows.append({
                    'forecast_month': month,
                    'market_segment': seg,
                    'metric': 'won_revenue',
                    'p10': int(np.percentile(won_rev_clipped, 10)),
                    'p50': int(np.percentile(won_rev_clipped, 50)),
                    'p90': int(np.percentile(won_rev_clipped, 90))
                })
    
    df_forecast = pd.DataFrame(forecast_rows)
    df_confidence = pd.DataFrame(confidence_rows)
    
    if ASSUMPTIONS['target_annual_revenue'] > 0:
        print(f"\n--- 7. Applying Management Target Override ---")
        
        raw_annual = df_forecast['forecasted_won_revenue_median'].sum()
        target = ASSUMPTIONS['target_annual_revenue']
        scale_factor = target / raw_annual if raw_annual > 0 else 1.0
        
        print(f"    Raw forecast total: ${raw_annual:,.0f}")
        print(f"    Target override:    ${target:,.0f}")
        print(f"    Scale factor:       {scale_factor:.3f}")
        
        df_forecast['forecasted_won_revenue_median'] = (df_forecast['forecasted_won_revenue_median'] * scale_factor).astype(int)
        df_forecast['forecasted_won_revenue_mean'] = (df_forecast['forecasted_won_revenue_mean'] * scale_factor).astype(int)
        
        for i, row in df_confidence.iterrows():
            df_confidence.at[i, 'p10'] = int(row['p10'] * scale_factor)
            df_confidence.at[i, 'p50'] = int(row['p50'] * scale_factor)
            df_confidence.at[i, 'p90'] = int(row['p90'] * scale_factor)
    
    df_forecast.to_csv(OUTPUT_MONTHLY_FORECAST, index=False)
    df_confidence.to_csv(OUTPUT_CONFIDENCE, index=False)
    
    print("\n--- 8. Sanity Check ---")
    
    monthly_totals = df_forecast.groupby('forecast_month').agg({
        'forecasted_won_volume_median': 'sum',
        'forecasted_won_revenue_median': 'sum'
    })
    
    print("    Monthly forecast totals:")
    for month, row in monthly_totals.iterrows():
        print(f"      {month.strftime('%Y-%m')}: {row['forecasted_won_volume_median']:>4.0f} deals, ${row['forecasted_won_revenue_median']:>12,.0f}")
    
    annual_vol = monthly_totals['forecasted_won_volume_median'].sum()
    annual_rev = monthly_totals['forecasted_won_revenue_median'].sum()
    
    print(f"\n    ANNUAL TOTAL: {annual_vol:,.0f} deals, ${annual_rev:,.0f}")
    
    print("\n--- 9. Generating Executive Summary ---")
    generate_executive_summary(df, df_forecast, df_confidence, velocity, win_rates, dso_dict, data_profile)
    
    print(f"\n    Outputs saved:")
    print(f"      {OUTPUT_MONTHLY_FORECAST}")
    print(f"      {OUTPUT_CONFIDENCE}")
    print(f"      {OUTPUT_EXECUTIVE}")
    
    print("\n" + "=" * 70)
    print("FORECAST COMPLETE")
    print("=" * 70)


def generate_executive_summary(df_raw, df_forecast, df_confidence, velocity, win_rates, dso_dict, data_profile):
    annual_rev = df_forecast['forecasted_won_revenue_median'].sum()
    annual_vol = df_forecast['forecasted_won_volume_median'].sum()
    
    seg_totals = df_forecast.groupby('market_segment')['forecasted_won_revenue_median'].sum().sort_values(ascending=False)
    
    total_conf = df_confidence.groupby('forecast_month')[['p10', 'p50', 'p90']].sum()
    annual_p10 = total_conf['p10'].sum() if len(total_conf) > 0 else 0
    annual_p50 = total_conf['p50'].sum() if len(total_conf) > 0 else 0
    annual_p90 = total_conf['p90'].sum() if len(total_conf) > 0 else 0
    
    entry_dist = velocity.get('_entry_stage_distribution', {})
    baseline_year = velocity.get('_baseline_year', 'N/A')
    
    summary = f"""
{'=' * 70}
EXECUTIVE SUMMARY: 2026 REVENUE FORECAST
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'=' * 70}

FORECAST RESULTS
----------------
Expected Annual Revenue (P50):     ${annual_p50:>15,.0f}
Conservative Case (P10):           ${annual_p10:>15,.0f}
Optimistic Case (P90):             ${annual_p90:>15,.0f}
Expected Deal Volume:              {annual_vol:>15,.0f} deals

{'MANAGEMENT TARGET APPLIED' if ASSUMPTIONS['target_annual_revenue'] > 0 else 'NO MANAGEMENT TARGET OVERRIDE'}

SEGMENT BREAKDOWN
-----------------
"""
    
    for seg, rev in seg_totals.items():
        pct = (rev / annual_rev * 100) if annual_rev > 0 else 0
        wr = win_rates.get(seg, 0)
        dso = dso_dict.get(seg, {}).get('median', 60)
        summary += f"  {seg:20s}: ${rev:>12,.0f}  ({pct:>5.1f}%)  WR: {wr:.1%}  DSO: {dso:.0f} days\n"

    summary += f"""

DATA-DERIVED BASELINES (from {baseline_year})
---------------------------------------------
"""
    for seg in win_rates.keys():
        seg_vel = velocity.get(seg, {})
        avg_monthly = np.mean([seg_vel.get(m, {}).get('vol', 0) for m in range(1, 13)])
        avg_size = np.mean([seg_vel.get(m, {}).get('size', 0) for m in range(1, 13)])
        summary += f"  {seg:20s}: {avg_monthly:.1f} deals/month, ${avg_size:,.0f} avg size\n"

    summary += f"""

GROWTH ASSUMPTIONS APPLIED
--------------------------
Volume Growth:         {((ASSUMPTIONS['volume_growth_multiplier'] - 1) * 100):>+.0f}%
Win Rate Uplift:       {((ASSUMPTIONS['win_rate_uplift_multiplier'] - 1) * 100):>+.0f}%
Deal Size Inflation:   {((ASSUMPTIONS['deal_size_inflation'] - 1) * 100):>+.0f}%
Simulations Run:       {ASSUMPTIONS['num_simulations']:,}

METHODOLOGY
-----------
This forecast uses a DATA-DRIVEN approach:
1. All baselines (velocity, win rates, DSO, deal sizes) are derived from 
   historical data - no arbitrary floors or ceilings.
2. Growth assumptions are applied as MULTIPLIERS on top of data-derived baselines.
3. Monte Carlo simulation captures uncertainty in timing and outcomes.

{'=' * 70}
"""
    
    with open(OUTPUT_EXECUTIVE, 'w', encoding='utf-8') as f:
        f.write(summary)
    
    print(summary)


if __name__ == "__main__":
    run_forecast()