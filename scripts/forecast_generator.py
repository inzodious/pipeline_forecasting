import pandas as pd
import numpy as np
import os
import random
from datetime import datetime, timedelta
from collections import defaultdict

# ==========================================
# CONFIGURATION & ASSUMPTIONS
# ==========================================
ASSUMPTIONS = {
    # Growth Parameters
    "volume_growth_multiplier": 1.10,       # 10% increase in deal creation volume
    "win_rate_uplift_multiplier": 1.05,     # 5% increase in win efficiency
    "deal_size_inflation": 1.03,            # 3% increase in average deal value
    
    # Management Override (set to 0 to disable, or input target annual revenue)
    "target_annual_revenue": 0,             # If > 0, scales forecast to hit this number
    
    # Simulation Parameters
    "num_simulations": 500,
    
    # Statistical Constraints
    "min_monthly_deals_floor": 3,
    "max_monthly_deals_ceiling": 150,       # Raised based on your 60-90/month actuals
    "min_win_rate_floor": 0.05,
    "max_win_rate_ceiling": 0.50,
    "deal_size_variance_cap": 0.25,
    "velocity_smoothing_window": 3,
    "confidence_interval_clip": 0.95,
    
    # Stage Classifications (adjust wildcards as needed)
    "won_stages": ["Closed Won", "Verbal"],
    "lost_stage_patterns": ["Closed Lost", "Declined to Bid"],  # Will use .str.contains()
}

# Paths
BASE_DIR = "data"
PATH_RAW_SNAPSHOTS = os.path.join(BASE_DIR, "fact_snapshots.csv")  # <-- Your actual filename

# Output Paths
OUTPUT_DRIVER_WIN = os.path.join(BASE_DIR, "driver_win_rates.csv")
OUTPUT_DRIVER_VEL = os.path.join(BASE_DIR, "driver_velocity.csv")
OUTPUT_DRIVER_DSO = os.path.join(BASE_DIR, "driver_dso_cycle_times.csv")
OUTPUT_MONTHLY_FORECAST = os.path.join(BASE_DIR, "forecast_monthly_2026.csv")
OUTPUT_CONFIDENCE = os.path.join(BASE_DIR, "forecast_confidence_intervals.csv")
OUTPUT_SENSITIVITY = os.path.join(BASE_DIR, "forecast_sensitivity_matrix.csv")
OUTPUT_EXECUTIVE = os.path.join(BASE_DIR, "executive_summary.txt")
OUTPUT_ASSUMPTIONS = os.path.join(BASE_DIR, "forecast_assumptions_log.csv")


# ==========================================
# HELPER: STAGE CLASSIFICATION
# ==========================================

def is_won(stage):
    """Check if stage indicates a won deal."""
    if pd.isna(stage):
        return False
    return stage in ASSUMPTIONS['won_stages']

def is_lost(stage):
    """Check if stage indicates a lost deal (supports wildcards)."""
    if pd.isna(stage):
        return False
    for pattern in ASSUMPTIONS['lost_stage_patterns']:
        if pattern.lower() in stage.lower():
            return True
    return False

def is_closed(stage):
    """Check if deal is closed (won or lost)."""
    return is_won(stage) or is_lost(stage)


# ==========================================
# DSO CALCULATION (date_created → date_closed)
# ==========================================

def calculate_dso_distribution(df):
    """
    Calculate Days Sales Outstanding (cycle time) distribution by segment.
    Uses ONLY date_created and date_closed - ignores date_implementation entirely.
    """
    print("  > Calculating DSO Distribution (date_created → date_closed)...")
    
    # Get final state of each deal
    df_final = df.sort_values('date_snapshot').groupby('deal_id').last().reset_index()
    
    # Filter to closed deals with valid dates
    df_final['is_won_flag'] = df_final['stage'].apply(is_won)
    df_final['is_lost_flag'] = df_final['stage'].apply(is_lost)
    df_closed = df_final[df_final['is_won_flag'] | df_final['is_lost_flag']].copy()
    
    df_closed['date_created'] = pd.to_datetime(df_closed['date_created'])
    df_closed['date_closed'] = pd.to_datetime(df_closed['date_closed'])
    
    # Calculate cycle days
    df_closed['cycle_days'] = (df_closed['date_closed'] - df_closed['date_created']).dt.days
    
    # Filter reasonable cycles (7 days to 2 years)
    df_valid = df_closed[(df_closed['cycle_days'] >= 7) & (df_closed['cycle_days'] <= 730)].copy()
    
    print(f"    Valid closed deals for DSO: {len(df_valid):,}")
    
    dso_stats = []
    dso_dict = {}
    
    for seg in df['market_segment'].unique():
        seg_data = df_valid[df_valid['market_segment'] == seg]
        
        if len(seg_data) >= 5:
            mean_dso = seg_data['cycle_days'].mean()
            std_dso = seg_data['cycle_days'].std()
            median_dso = seg_data['cycle_days'].median()
            p25 = seg_data['cycle_days'].quantile(0.25)
            p75 = seg_data['cycle_days'].quantile(0.75)
        else:
            # Fallback to overall if segment has few deals
            mean_dso = df_valid['cycle_days'].mean()
            std_dso = df_valid['cycle_days'].std()
            median_dso = df_valid['cycle_days'].median()
            p25 = df_valid['cycle_days'].quantile(0.25)
            p75 = df_valid['cycle_days'].quantile(0.75)
        
        dso_dict[seg] = {
            'mean': mean_dso,
            'std': max(std_dso, 10),  # Floor std at 10 days
            'median': median_dso,
            'p25': p25,
            'p75': p75
        }
        
        dso_stats.append({
            'market_segment': seg,
            'num_deals': len(seg_data),
            'mean_days': round(mean_dso, 1),
            'std_days': round(std_dso, 1),
            'median_days': round(median_dso, 1),
            'p25_days': round(p25, 1),
            'p75_days': round(p75, 1)
        })
        
        print(f"    {seg}: mean={mean_dso:.0f} days, median={median_dso:.0f}, std={std_dso:.0f}")
    
    pd.DataFrame(dso_stats).to_csv(OUTPUT_DRIVER_DSO, index=False)
    
    return dso_dict


# ==========================================
# WIN RATE CALCULATION
# ==========================================

def calculate_win_rates(df):
    """
    Calculate revenue-weighted win rates by segment.
    Win rate = Won Revenue / (Won Revenue + Lost Revenue) for mature deals.
    """
    print("  > Calculating Win Rates...")
    
    # Get final state of each deal
    df_final = df.sort_values('date_snapshot').groupby('deal_id').last().reset_index()
    
    # Classify
    df_final['is_won_flag'] = df_final['stage'].apply(is_won)
    df_final['is_lost_flag'] = df_final['stage'].apply(is_lost)
    df_final['is_closed_flag'] = df_final['is_won_flag'] | df_final['is_lost_flag']
    
    # Only use closed deals for win rate (mature cohort)
    df_closed = df_final[df_final['is_closed_flag']].copy()
    
    print(f"    Total closed deals: {len(df_closed):,}")
    print(f"    Won: {df_closed['is_won_flag'].sum():,}, Lost: {df_closed['is_lost_flag'].sum():,}")
    
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
        
        # Revenue-weighted win rate
        raw_wr = (won_rev / total_rev) if total_rev > 0 else 0.20
        
        # Volume-based win rate (for reference)
        vol_wr = (won_count / total_count) if total_count > 0 else 0.20
        
        # Apply constraints
        constrained_wr = np.clip(raw_wr, ASSUMPTIONS['min_win_rate_floor'], ASSUMPTIONS['max_win_rate_ceiling'])
        
        win_rates[seg] = constrained_wr
        
        stats_rows.append({
            'market_segment': seg,
            'won_revenue': won_rev,
            'lost_revenue': lost_rev,
            'total_closed_revenue': total_rev,
            'won_count': won_count,
            'lost_count': lost_count,
            'raw_win_rate_rev_weighted': round(raw_wr, 4),
            'raw_win_rate_vol_weighted': round(vol_wr, 4),
            'constrained_win_rate': round(constrained_wr, 4)
        })
        
        print(f"    {seg}: {constrained_wr:.1%} (raw: {raw_wr:.1%}, vol-based: {vol_wr:.1%})")
    
    pd.DataFrame(stats_rows).to_csv(OUTPUT_DRIVER_WIN, index=False)
    
    return win_rates


# ==========================================
# VELOCITY CALCULATION (Deal Creation Volume)
# ==========================================

def calculate_velocity(df):
    """
    Calculate monthly deal creation velocity by segment.
    
    IMPORTANT: Counts ALL deals when they first appear in the dataset,
    regardless of what stage they entered at. This captures:
    - 68% entering at Qualified
    - 21% entering directly at Closed Lost
    - 3.5% entering at Alignment
    - Remainder entering at other stages (Solutioning, etc.)
    
    We count the first snapshot appearance of each deal as its "creation" moment.
    """
    print("  > Calculating Velocity (ALL deal first appearances)...")
    
    # Find first appearance of each deal (regardless of stage)
    df['date_snapshot'] = pd.to_datetime(df['date_snapshot'])
    df_first_appearance = df.sort_values('date_snapshot').groupby('deal_id').first().reset_index()
    
    df_first_appearance['appear_year'] = df_first_appearance['date_snapshot'].dt.year
    df_first_appearance['appear_month'] = df_first_appearance['date_snapshot'].dt.month
    
    # Log entry stage distribution for executive summary
    entry_stage_dist = df_first_appearance['stage'].value_counts(normalize=True)
    print(f"    Deal entry stage distribution:")
    for stage, pct in entry_stage_dist.head(6).items():
        print(f"      {stage}: {pct:.1%}")
    
    # Use most recent complete year for baseline
    latest_year = df_first_appearance['appear_year'].max()
    latest_month = df_first_appearance[df_first_appearance['appear_year'] == latest_year]['appear_month'].max()
    
    if latest_month < 12:
        baseline_year = latest_year - 1 if (latest_year - 1) in df_first_appearance['appear_year'].values else latest_year
    else:
        baseline_year = latest_year
    
    df_baseline = df_first_appearance[df_first_appearance['appear_year'] == baseline_year].copy()
    
    print(f"    Velocity baseline year: {baseline_year} ({len(df_baseline):,} deals)")
    
    velocity_dict = {}
    export_rows = []
    
    for seg in df['market_segment'].unique():
        velocity_dict[seg] = {}
        seg_data = df_baseline[df_baseline['market_segment'] == seg]
        
        # Calculate annual averages as fallback
        annual_count = len(seg_data)
        monthly_avg = max(ASSUMPTIONS['min_monthly_deals_floor'], annual_count / 12)
        annual_avg_size = seg_data['net_revenue'].mean() if len(seg_data) > 0 else 50000
        
        monthly_vols = []
        monthly_sizes = []
        
        for m in range(1, 13):
            m_data = seg_data[seg_data['appear_month'] == m]
            
            if len(m_data) >= 3:  # Need at least 3 deals for meaningful month
                monthly_vols.append(len(m_data))
                monthly_sizes.append(m_data['net_revenue'].mean())
            else:
                monthly_vols.append(monthly_avg)
                monthly_sizes.append(annual_avg_size)
        
        # Apply smoothing
        if ASSUMPTIONS['velocity_smoothing_window'] > 1:
            monthly_vols_smooth = pd.Series(monthly_vols).rolling(
                window=ASSUMPTIONS['velocity_smoothing_window'],
                center=True,
                min_periods=1
            ).mean().tolist()
        else:
            monthly_vols_smooth = monthly_vols
        
        for m in range(1, 13):
            vol = np.clip(
                monthly_vols_smooth[m - 1],
                ASSUMPTIONS['min_monthly_deals_floor'],
                ASSUMPTIONS['max_monthly_deals_ceiling']
            )
            
            velocity_dict[seg][m] = {
                'vol': vol,
                'size': monthly_sizes[m - 1]
            }
            
            export_rows.append({
                'market_segment': seg,
                'month': m,
                'raw_volume': monthly_vols[m - 1],
                'smoothed_volume': round(vol, 1),
                'avg_deal_size': round(monthly_sizes[m - 1], 0)
            })
    
    pd.DataFrame(export_rows).to_csv(OUTPUT_DRIVER_VEL, index=False)
    
    # Store entry stage distribution for summary
    velocity_dict['_entry_stage_distribution'] = entry_stage_dist.to_dict()
    
    return velocity_dict


# ==========================================
# EXISTING PIPELINE INITIALIZATION
# ==========================================

def initialize_existing_pipeline(df, win_rates, dso_dict):
    """
    Load the 232 (or however many) open deals from the latest snapshot.
    These will naturally distribute their close dates across 2026 based on DSO.
    
    NO ghost deals are created - this uses only real pipeline data.
    """
    print("  > Initializing Existing Pipeline (real deals only)...")
    
    latest_date = df['date_snapshot'].max()
    print(f"    Latest snapshot: {latest_date.date()}")
    
    df_latest = df[df['date_snapshot'] == latest_date].copy()
    
    # Open deals = not won and not lost
    df_latest['is_closed_flag'] = df_latest['stage'].apply(is_closed)
    df_open = df_latest[~df_latest['is_closed_flag']].copy()
    
    print(f"    Open deals in pipeline: {len(df_open):,}")
    
    # Show creation date distribution
    df_open['date_created'] = pd.to_datetime(df_open['date_created'])
    df_open['create_month'] = df_open['date_created'].dt.to_period('M')
    create_dist = df_open['create_month'].value_counts().sort_index()
    
    print(f"    Creation date distribution (recent months):")
    for period, count in create_dist.tail(6).items():
        print(f"      {period}: {count} deals")
    
    existing_deals = []
    
    for _, row in df_open.iterrows():
        seg = row['market_segment']
        create_date = pd.to_datetime(row['date_created'])
        
        # Determine win/loss based on segment win rate
        base_wr = win_rates.get(seg, 0.20)
        adj_wr = np.clip(
            base_wr * ASSUMPTIONS['win_rate_uplift_multiplier'],
            ASSUMPTIONS['min_win_rate_floor'],
            ASSUMPTIONS['max_win_rate_ceiling']
        )
        is_won = random.random() < adj_wr
        
        # Calculate expected close date using DSO distribution
        seg_dso = dso_dict.get(seg, {'mean': 60, 'std': 30})
        
        # How many days has this deal been open?
        days_open = (latest_date - create_date).days
        
        # Sample total cycle time, ensuring it's at least days_open + some buffer
        total_cycle = int(np.random.normal(seg_dso['mean'], seg_dso['std']))
        total_cycle = max(days_open + random.randint(7, 45), total_cycle)  # At least a week more
        
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
    
    # Show projected close distribution
    close_periods = pd.Series([d['close_date'] for d in existing_deals]).dt.to_period('M').value_counts().sort_index()
    print(f"    Projected close distribution (existing pipeline):")
    for period, count in close_periods.head(8).items():
        print(f"      {period}: {count} deals")
    
    return existing_deals


# ==========================================
# FORECAST ENGINE
# ==========================================

class ForecastEngine:
    """
    Monte Carlo forecast engine.
    - Uses existing pipeline deals (real data)
    - Generates new deals for 2026 based on velocity
    - Applies DSO distribution for close date projection
    """
    
    def __init__(self, win_rates, velocity, dso_dict):
        self.win_rates = win_rates
        self.velocity = {k: v for k, v in velocity.items() if not k.startswith('_')}
        self.dso_dict = dso_dict
        self.forecast_months = pd.date_range('2026-01-01', '2026-12-31', freq='MS')
    
    def generate_monthly_deals(self, month, segment):
        """Generate new deals for a given month/segment."""
        month_num = month.month
        
        if segment not in self.velocity or month_num not in self.velocity[segment]:
            return []
        
        stats = self.velocity[segment][month_num]
        
        # Apply growth
        base_vol = stats['vol'] * ASSUMPTIONS['volume_growth_multiplier']
        base_vol = np.clip(base_vol, ASSUMPTIONS['min_monthly_deals_floor'], ASSUMPTIONS['max_monthly_deals_ceiling'])
        
        num_deals = max(ASSUMPTIONS['min_monthly_deals_floor'], np.random.poisson(base_vol))
        
        avg_size = stats['size'] * ASSUMPTIONS['deal_size_inflation']
        
        base_wr = self.win_rates.get(segment, 0.20)
        adj_wr = np.clip(
            base_wr * ASSUMPTIONS['win_rate_uplift_multiplier'],
            ASSUMPTIONS['min_win_rate_floor'],
            ASSUMPTIONS['max_win_rate_ceiling']
        )
        
        seg_dso = self.dso_dict.get(segment, {'mean': 60, 'std': 30})
        
        deals = []
        
        for _ in range(num_deals):
            is_won = random.random() < adj_wr
            
            # Deal size with variance
            size_mult = 1.0 + random.uniform(
                -ASSUMPTIONS['deal_size_variance_cap'],
                ASSUMPTIONS['deal_size_variance_cap']
            )
            actual_rev = int(avg_size * size_mult)
            
            # Random creation day within month
            days_in_month = (month + pd.DateOffset(months=1) - timedelta(days=1)).day
            create_day = random.randint(1, days_in_month)
            create_date = month + timedelta(days=create_day - 1)
            
            # Sample cycle time from DSO distribution
            cycle_days = max(14, int(np.random.normal(seg_dso['mean'], seg_dso['std'])))
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
        """Run a single simulation."""
        # Track monthly results
        monthly_results = {month: defaultdict(lambda: {
            'created': 0, 'won_vol': 0, 'won_rev': 0
        }) for month in self.forecast_months}
        
        all_deals = list(existing_deals)
        
        # Generate new 2026 deals
        for month in self.forecast_months:
            for segment in self.velocity.keys():
                new_deals = self.generate_monthly_deals(month, segment)
                all_deals.extend(new_deals)
        
        # Aggregate by close month
        for deal in all_deals:
            close_period = pd.Period(deal['close_date'], freq='M').to_timestamp()
            
            # Only count 2026 closings
            if close_period not in monthly_results:
                continue
            
            seg = deal['segment']
            
            # Count creations (only for 2026 created deals)
            if deal.get('create_month') in self.forecast_months:
                if deal.get('create_month') == close_period:
                    pass  # Created this month tracked elsewhere
                monthly_results[deal['create_month']][seg]['created'] += 1
            
            # Count wins
            if deal['is_won']:
                monthly_results[close_period][seg]['won_vol'] += 1
                monthly_results[close_period][seg]['won_rev'] += deal['revenue']
        
        return monthly_results


# ==========================================
# MAIN EXECUTION
# ==========================================

def run_forecast():
    """Main forecast execution."""
    print("=" * 70)
    print("FORECAST GENERATOR V4 - CORRECTED LOGIC")
    print("=" * 70)
    
    # --- 1. Load Data ---
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
    
    # Log assumptions
    assumptions_log = pd.DataFrame([{
        'assumption': k,
        'value': str(v),
        'run_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    } for k, v in ASSUMPTIONS.items()])
    assumptions_log.to_csv(OUTPUT_ASSUMPTIONS, index=False)
    
    # --- 2. Calculate Drivers ---
    print("\n--- 2. Calculating Drivers ---")
    
    dso_dict = calculate_dso_distribution(df)
    win_rates = calculate_win_rates(df)
    velocity = calculate_velocity(df)
    
    # --- 3. Initialize Pipeline ---
    print("\n--- 3. Initializing Pipeline ---")
    
    existing_deals = initialize_existing_pipeline(df, win_rates, dso_dict)
    
    # --- 4. Run Simulations ---
    print(f"\n--- 4. Running {ASSUMPTIONS['num_simulations']} Simulations ---")
    
    engine = ForecastEngine(win_rates, velocity, dso_dict)
    all_results = []
    
    for sim in range(ASSUMPTIONS['num_simulations']):
        if (sim + 1) % 100 == 0:
            print(f"    Completed {sim + 1}/{ASSUMPTIONS['num_simulations']}...")
        
        result = engine.run_simulation(existing_deals)
        all_results.append(result)
    
    # --- 5. Aggregate Results ---
    print("\n--- 5. Aggregating Results ---")
    
    segments = [k for k in velocity.keys() if not k.startswith('_')]
    forecast_rows = []
    confidence_rows = []
    
    for month in engine.forecast_months:
        for seg in segments:
            won_vol_vals = [sim[month][seg]['won_vol'] for sim in all_results]
            won_rev_vals = [sim[month][seg]['won_rev'] for sim in all_results]
            
            # Clip outliers
            clip = ASSUMPTIONS['confidence_interval_clip']
            if len(won_rev_vals) > 0:
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
            
            if won_rev_clipped:
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
    
    # --- 6. Apply Management Target Override (if set) ---
    if ASSUMPTIONS['target_annual_revenue'] > 0:
        print(f"\n--- 6. Applying Management Target Override ---")
        
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
    
    # Save outputs
    df_forecast.to_csv(OUTPUT_MONTHLY_FORECAST, index=False)
    df_confidence.to_csv(OUTPUT_CONFIDENCE, index=False)
    
    # --- 7. Sanity Check ---
    print("\n--- 7. Sanity Check ---")
    
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
    
    # --- 8. Executive Summary ---
    print("\n--- 8. Generating Executive Summary ---")
    generate_executive_summary(df, df_forecast, df_confidence, velocity, win_rates, dso_dict)
    
    print(f"\n    Outputs saved:")
    print(f"      {OUTPUT_MONTHLY_FORECAST}")
    print(f"      {OUTPUT_CONFIDENCE}")
    print(f"      {OUTPUT_EXECUTIVE}")
    
    print("\n" + "=" * 70)
    print("FORECAST COMPLETE")
    print("=" * 70)


def generate_executive_summary(df_raw, df_forecast, df_confidence, velocity, win_rates, dso_dict):
    """Generate executive summary with methodology notes."""
    
    # Key metrics
    annual_rev = df_forecast['forecasted_won_revenue_median'].sum()
    annual_vol = df_forecast['forecasted_won_volume_median'].sum()
    
    # Segment breakdown
    seg_totals = df_forecast.groupby('market_segment')['forecasted_won_revenue_median'].sum().sort_values(ascending=False)
    
    # Confidence intervals
    total_conf = df_confidence.groupby('forecast_month')[['p10', 'p50', 'p90']].sum()
    annual_p10 = total_conf['p10'].sum()
    annual_p50 = total_conf['p50'].sum()
    annual_p90 = total_conf['p90'].sum()
    
    # Entry stage distribution
    entry_dist = velocity.get('_entry_stage_distribution', {})
    
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
{f"Target: ${ASSUMPTIONS['target_annual_revenue']:,.0f}" if ASSUMPTIONS['target_annual_revenue'] > 0 else ""}

SEGMENT BREAKDOWN
-----------------
"""
    
    for seg, rev in seg_totals.items():
        pct = (rev / annual_rev * 100) if annual_rev > 0 else 0
        wr = win_rates.get(seg, 0)
        dso = dso_dict.get(seg, {}).get('median', 60)
        summary += f"  {seg:20s}: ${rev:>12,.0f}  ({pct:>5.1f}%)  WR: {wr:.1%}  DSO: {dso:.0f} days\n"

    summary += f"""

ASSUMPTIONS APPLIED
-------------------
Volume Growth:         {((ASSUMPTIONS['volume_growth_multiplier'] - 1) * 100):>+.0f}%
Win Rate Uplift:       {((ASSUMPTIONS['win_rate_uplift_multiplier'] - 1) * 100):>+.0f}%
Deal Size Inflation:   {((ASSUMPTIONS['deal_size_inflation'] - 1) * 100):>+.0f}%
Simulations Run:       {ASSUMPTIONS['num_simulations']:,}

METHODOLOGY NOTES
-----------------
1. DEAL CREATION VELOCITY
   Velocity is calculated by counting ALL deals when they first appear in 
   the dataset, regardless of entry stage. This captures the full pipeline:
"""
    
    for stage, pct in sorted(entry_dist.items(), key=lambda x: -x[1])[:5]:
        summary += f"     - {stage}: {pct:.1%}\n"
    
    summary += f"""
   This approach ensures no deals are treated as "phantom" - every deal
   that enters the pipeline is counted toward creation volume.

2. DSO (CYCLE TIME) CALCULATION
   Days Sales Outstanding is calculated strictly from date_created to 
   date_closed for all closed deals. date_implementation is NOT used
   for forecasting as it fluctuates and is unreliable.

3. EXISTING PIPELINE
   The forecast begins with the {len(df_raw[df_raw['date_snapshot'] == df_raw['date_snapshot'].max()][~df_raw['stage'].apply(is_closed)]):,} open deals
   in the pipeline as of the latest snapshot. These deals are distributed
   across 2026 close dates using the segment-specific DSO distribution.
   NO synthetic/ghost deals are generated for 2025.

4. WIN RATE
   Revenue-weighted win rates are calculated from closed deals only.
   Win Rate = Won Revenue / (Won + Lost Revenue)

STATISTICAL CONSTRAINTS
-----------------------
Win Rate Bounds:        {ASSUMPTIONS['min_win_rate_floor']:.0%} - {ASSUMPTIONS['max_win_rate_ceiling']:.0%}
Monthly Volume Bounds:  {ASSUMPTIONS['min_monthly_deals_floor']} - {ASSUMPTIONS['max_monthly_deals_ceiling']} deals/segment
Deal Size Variance:     ±{ASSUMPTIONS['deal_size_variance_cap']:.0%}

{'=' * 70}
"""
    
    with open(OUTPUT_EXECUTIVE, 'w') as f:
        f.write(summary)
    
    print(summary)


if __name__ == "__main__":
    run_forecast()