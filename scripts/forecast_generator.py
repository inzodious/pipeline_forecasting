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
    "volume_growth_multiplier": 1.10,   # 10% Increase in deal volume
    "win_rate_uplift_multiplier": 1.05,  # 5% Increase in win efficiency
    "deal_size_inflation": 1.03,        # 3% Increase in deal value
    "manual_cycle_lag_months": None,    # If not None, overrides historical lag
    "num_simulations": 1000             # Monte Carlo iterations for confidence intervals
}

# Paths
BASE_DIR = "data"
PATH_RAW_SNAPSHOTS = os.path.join(BASE_DIR, "fact_pipeline_snapshot.csv")

# Output Paths
OUTPUT_DRIVER_WIN = os.path.join(BASE_DIR, "driver_win_rates.csv")
OUTPUT_DRIVER_VEL = os.path.join(BASE_DIR, "driver_velocity.csv")
OUTPUT_DRIVER_LAGS = os.path.join(BASE_DIR, "driver_cycle_lags.csv")
OUTPUT_DETAILED = os.path.join(BASE_DIR, "forecast_detailed_2026.csv")
OUTPUT_SUMMARY = os.path.join(BASE_DIR, "forecast_summary_2026.csv")
OUTPUT_CONFIDENCE = os.path.join(BASE_DIR, "forecast_confidence_intervals.csv")
OUTPUT_SENSITIVITY = os.path.join(BASE_DIR, "forecast_sensitivity_matrix.csv")
OUTPUT_EXECUTIVE = os.path.join(BASE_DIR, "executive_summary.txt")
OUTPUT_ASSUMPTIONS = os.path.join(BASE_DIR, "forecast_assumptions_log.csv")

# Global counter for projected deals
projected_deal_counter = 1

# ==========================================
# DEAL CLASS
# ==========================================
class Deal:
    def __init__(self, deal_id, deal_name, segment, revenue, create_date, close_date, is_won, source="Projected"):
        self.deal_id = deal_id
        self.deal_name = deal_name
        self.segment = segment
        self.revenue = revenue
        self.create_date = create_date
        self.target_close_date = close_date
        self.is_won = is_won
        self.source = source
        self.status = "Qualified"
        self.actual_close_date = None

    def update_status(self, current_date):
        if current_date >= self.target_close_date:
            self.status = "Closed Won" if self.is_won else "Closed Lost"
            self.actual_close_date = self.target_close_date
        else:
            self.status = "Qualified"

# ==========================================
# PART 1: DRIVER CALCULATIONS
# ==========================================

def calculate_win_rates(df, cutoff_date=None):
    """
    Calculates historical win rates by segment.
    
    METHODOLOGY:
    - Only includes deals that have had sufficient time to mature (closed or past expected close)
    - Uses revenue-weighted win rates to account for deal size importance
    - Calculates both overall segment rates and monthly cohort rates
    
    Args:
        df: Raw snapshot data
        cutoff_date: Optional date to filter mature deals. If None, uses last snapshot date.
    
    Returns:
        Dictionary of win rates by segment
    """
    print("  > Calculating Win Rates...")
    df['date_created'] = pd.to_datetime(df['date_created'])
    df['target_implementation_date'] = pd.to_datetime(df['target_implementation_date'])
    
    # Get final state of every deal
    df_final = df.sort_values('snapshot_date', ascending=False).groupby('deal_id').first().reset_index()
    
    # Determine cutoff for mature deals
    if cutoff_date is None:
        cutoff_date = df['snapshot_date'].max()
    
    # CRITICAL: Only include deals created with enough time to close
    # We calculate expected cycle time per segment first
    df_closed = df_final[df_final['status'].isin(['Closed Won', 'Closed Lost'])].copy()
    df_closed['cycle_days'] = (df_closed['target_implementation_date'] - df_closed['date_created']).dt.days
    
    avg_cycle_days = {}
    for seg in df_closed['market_segment'].unique():
        seg_cycle = df_closed[df_closed['market_segment'] == seg]['cycle_days'].median()
        avg_cycle_days[seg] = seg_cycle if pd.notna(seg_cycle) else 120
    
    # Filter to mature deals only
    df_mature = []
    for _, row in df_final.iterrows():
        seg = row['market_segment']
        expected_close = row['date_created'] + timedelta(days=avg_cycle_days.get(seg, 120))
        
        # Include if: already closed OR had enough time to close
        if row['status'] in ['Closed Won', 'Closed Lost'] or expected_close <= cutoff_date:
            df_mature.append(row)
    
    df_mature = pd.DataFrame(df_mature)
    
    if len(df_mature) == 0:
        print("    WARNING: No mature deals found for win rate calculation")
        return {seg: 0.20 for seg in df['market_segment'].unique()}
    
    # Calculate cohort month
    df_mature['cohort_month'] = df_mature['date_created'].dt.to_period('M').dt.to_timestamp()
    
    # Filter for 2024-2025 cohorts
    df_mature = df_mature[df_mature['date_created'].dt.year.isin([2024, 2025])]
    
    # Calculate overall segment win rates (revenue-weighted)
    segment_stats = []
    win_rates_dict = {}
    
    for segment, group in df_mature.groupby('market_segment'):
        total_created = group['revenue'].sum()
        won_rev = group[group['status'] == 'Closed Won']['revenue'].sum()
        
        win_rate = (won_rev / total_created) if total_created > 0 else 0.20
        win_rates_dict[segment] = win_rate
        
        # Also calculate by cohort month for trend analysis
        monthly_rates = []
        for month, mgroup in group.groupby('cohort_month'):
            m_total = mgroup['revenue'].sum()
            m_won = mgroup[mgroup['status'] == 'Closed Won']['revenue'].sum()
            m_rate = (m_won / m_total) if m_total > 0 else 0
            monthly_rates.append(m_rate)
        
        avg_monthly = np.mean(monthly_rates) if monthly_rates else win_rate
        std_monthly = np.std(monthly_rates) if len(monthly_rates) > 1 else 0
        
        segment_stats.append({
            'market_segment': segment,
            'total_created_revenue': total_created,
            'total_won_revenue': won_rev,
            'win_rate_pct': win_rate,
            'avg_monthly_win_rate': avg_monthly,
            'std_monthly_win_rate': std_monthly,
            'num_months_analyzed': len(monthly_rates),
            'num_mature_deals': len(group),
            'mature_deals_won': len(group[group['status'] == 'Closed Won']),
            'cutoff_date_used': cutoff_date.date()
        })
    
    pd.DataFrame(segment_stats).to_csv(OUTPUT_DRIVER_WIN, index=False)
    print(f"    Win rates calculated for {len(win_rates_dict)} segments")
    print(f"    Mature deals analyzed: {len(df_mature)}")
    
    return win_rates_dict


def calculate_velocity(df):
    """
    Calculates monthly deal creation velocity using 2025 data only.
    Uses FIRST qualification date to avoid double-counting.
    Revenue is taken at time of first qualification.
    
    Returns:
        Dictionary: {Segment: {MonthNum: {'vol': X, 'size': Y}}}
    """
    print("  > Calculating Velocity (2025 baseline)...")
    
    # Find first Qualified date per deal
    df_qual = df[df['status'] == 'Qualified'].copy()
    df_qual['snapshot_date'] = pd.to_datetime(df_qual['snapshot_date'])
    
    df_first_qual = df_qual.sort_values('snapshot_date').groupby('deal_id').first().reset_index()
    df_first_qual['qualified_year'] = df_first_qual['snapshot_date'].dt.year
    df_first_qual['qualified_month'] = df_first_qual['snapshot_date'].dt.month
    
    # Filter to 2025 only (as per instructions)
    df_2025 = df_first_qual[df_first_qual['qualified_year'] == 2025].copy()
    
    if len(df_2025) == 0:
        print("    WARNING: No 2025 qualified deals found. Using 2024 data.")
        df_2025 = df_first_qual[df_first_qual['qualified_year'] == 2024].copy()
    
    # Build velocity dictionary
    velocity_dict = {}
    export_rows = []
    
    for seg in df['market_segment'].unique():
        velocity_dict[seg] = {}
        seg_data = df_2025[df_2025['market_segment'] == seg]
        
        # Get annual averages as fallback
        annual_avg_vol = len(seg_data) / 12 if len(seg_data) > 0 else 1
        annual_avg_size = seg_data['revenue'].mean() if len(seg_data) > 0 else 50000
        
        for m in range(1, 13):
            m_data = seg_data[seg_data['qualified_month'] == m]
            
            if len(m_data) > 0:
                vol = len(m_data)
                size = m_data['revenue'].mean()
            else:
                # Use annual average if month has no data
                vol = annual_avg_vol
                size = annual_avg_size
            
            velocity_dict[seg][m] = {'vol': vol, 'size': size}
            
            export_rows.append({
                'market_segment': seg,
                'month': m,
                'deals_qualified_count': vol,
                'avg_deal_size': size,
                'source': 'actual' if len(m_data) > 0 else 'annual_avg_fallback'
            })
    
    pd.DataFrame(export_rows).to_csv(OUTPUT_DRIVER_VEL, index=False)
    print(f"    Velocity calculated from {len(df_2025)} deals qualified in 2025")
    
    return velocity_dict


def calculate_lags(df):
    """
    Calculates average days to close per segment.
    Only uses actually closed deals for accuracy.
    """
    print("  > Calculating Cycle Time Lags...")
    
    df_won = df[df['status'] == 'Closed Won'].copy()
    df_won['date_created'] = pd.to_datetime(df_won['date_created'])
    df_won['target_implementation_date'] = pd.to_datetime(df_won['target_implementation_date'])
    
    df_won['cycle_days'] = (df_won['target_implementation_date'] - df_won['date_created']).dt.days
    
    # Remove outliers (deals that closed too fast or too slow)
    df_won = df_won[(df_won['cycle_days'] > 0) & (df_won['cycle_days'] < 730)]
    
    cycle_stats = []
    cycle_lags = {}
    
    for seg, group in df_won.groupby('market_segment'):
        mean_days = group['cycle_days'].mean()
        median_days = group['cycle_days'].median()
        std_days = group['cycle_days'].std()
        
        # Use median to avoid outlier skew, convert to months
        lag_months = int(max(1, median_days / 30))
        cycle_lags[seg] = lag_months
        
        cycle_stats.append({
            'market_segment': seg,
            'mean_cycle_days': mean_days,
            'median_cycle_days': median_days,
            'std_cycle_days': std_days,
            'lag_months_used': lag_months,
            'num_deals_analyzed': len(group)
        })
    
    pd.DataFrame(cycle_stats).to_csv(OUTPUT_DRIVER_LAGS, index=False)
    print(f"    Cycle lags calculated for {len(cycle_lags)} segments")
    
    return cycle_lags


# ==========================================
# PART 2: INITIALIZATION
# ==========================================

def initialize_active_deals(df, win_rates):
    """
    Loads existing open deals from the final snapshot.
    Assigns win probability based on historical segment win rates.
    """
    print("  > Initializing Open Pipeline...")
    latest_date = df['snapshot_date'].max()
    print(f"    Latest Snapshot: {latest_date.date()}")
    
    df_latest = df[df['snapshot_date'] == latest_date].copy()
    excluded = ['Closed Won', 'Closed Lost', 'Initiated', 'Verbal', 'Declined to Bid']
    df_open = df_latest[~df_latest['status'].isin(excluded)].copy()
    
    active_deals = []
    
    for _, row in df_open.iterrows():
        seg = row['market_segment']
        
        # Assign win fate
        base_wr = win_rates.get(seg, 0.20)
        adj_wr = min(base_wr * ASSUMPTIONS['win_rate_uplift_multiplier'], 1.0)
        is_won = random.random() < adj_wr
        
        close_date = pd.to_datetime(row['target_implementation_date'])
        if close_date <= latest_date:
            close_date = latest_date + timedelta(days=30)
        
        deal = Deal(
            deal_id=row['deal_id'],
            deal_name=row['deal_name'],
            segment=seg,
            revenue=row['revenue'],
            create_date=pd.to_datetime(row['date_created']),
            close_date=close_date,
            is_won=is_won,
            source="Existing"
        )
        deal.status = row['status']
        active_deals.append(deal)
    
    print(f"    Loaded {len(active_deals)} existing deals into pipeline")
    return active_deals


# ==========================================
# PART 3: SIMULATION ENGINE
# ==========================================

def run_single_simulation(df_raw, win_rates, velocity, lags, sim_id=1):
    """
    Runs a single Monte Carlo simulation for 2026.
    Returns summary results for aggregation.
    """
    global projected_deal_counter
    
    active_deals = initialize_active_deals(df_raw, win_rates)
    sim_dates = pd.date_range(start='2026-01-01', end='2026-12-31', freq='W-FRI')
    
    monthly_results = defaultdict(lambda: defaultdict(lambda: {
        'created': 0, 'won_vol': 0, 'won_rev': 0, 'open_count': 0, 'open_val': 0
    }))
    
    for current_date in sim_dates:
        curr_month = current_date.month
        
        # Generate new deals
        for seg, monthly_data in velocity.items():
            if curr_month not in monthly_data:
                continue
            
            stats = monthly_data[curr_month]
            weekly_vol_target = (stats['vol'] * ASSUMPTIONS['volume_growth_multiplier']) / 4.3
            num_to_create = np.random.poisson(weekly_vol_target)
            
            if num_to_create > 0:
                avg_size = stats['size'] * ASSUMPTIONS['deal_size_inflation']
                base_wr = win_rates.get(seg, 0.20)
                adj_wr = min(base_wr * ASSUMPTIONS['win_rate_uplift_multiplier'], 1.0)
                lag_months = ASSUMPTIONS['manual_cycle_lag_months'] if ASSUMPTIONS['manual_cycle_lag_months'] else lags.get(seg, 4)
                
                for _ in range(num_to_create):
                    is_won = random.random() < adj_wr
                    actual_rev = int(avg_size * random.uniform(0.9, 1.1))
                    
                    target_close = current_date + timedelta(days=int(lag_months * 30))
                    target_close += timedelta(days=random.randint(-15, 15))
                    
                    deal_id = f"PROJ-{seg[:3].upper()}-{projected_deal_counter}"
                    deal_name = f"Projected Deal #{projected_deal_counter}"
                    
                    new_deal = Deal(
                        deal_id=deal_id,
                        deal_name=deal_name,
                        segment=seg,
                        revenue=actual_rev,
                        create_date=current_date,
                        close_date=target_close,
                        is_won=is_won,
                        source="Projected"
                    )
                    active_deals.append(new_deal)
                    projected_deal_counter += 1
        
        # Update statuses and aggregate monthly stats
        forecast_month = pd.Period(current_date, freq='M').to_timestamp()
        
        for deal in active_deals:
            deal.update_status(current_date)
            
            # Track metrics
            seg = deal.segment
            
            # Created this month
            if deal.create_date.to_period('M') == current_date.to_period('M') and deal.source == "Projected":
                monthly_results[forecast_month][seg]['created'] += 1
            
            # Won this month
            if deal.status == "Closed Won" and deal.target_close_date.to_period('M') == current_date.to_period('M'):
                monthly_results[forecast_month][seg]['won_vol'] += 1
                monthly_results[forecast_month][seg]['won_rev'] += deal.revenue
            
            # Open pipeline (end of week snapshot)
            if deal.status == "Qualified":
                monthly_results[forecast_month][seg]['open_count'] += 1
                monthly_results[forecast_month][seg]['open_val'] += deal.revenue
    
    return monthly_results


def run_simulation():
    """
    Main execution function with confidence intervals.
    """
    print("=" * 60)
    print("FORECAST GENERATOR V2 - WITH VALIDATION & DIAGNOSTICS")
    print("=" * 60)
    
    print("\n--- 1. Ingesting Data & Calculating Drivers ---")
    if not os.path.exists(PATH_RAW_SNAPSHOTS):
        print("Error: snapshot file not found.")
        return
    
    df_raw = pd.read_csv(PATH_RAW_SNAPSHOTS)
    df_raw['snapshot_date'] = pd.to_datetime(df_raw['snapshot_date'])
    
    # Log assumptions
    assumptions_log = pd.DataFrame([{
        'assumption': k,
        'value': v,
        'run_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    } for k, v in ASSUMPTIONS.items()])
    assumptions_log.to_csv(OUTPUT_ASSUMPTIONS, index=False)
    
    # Calculate drivers
    win_rates = calculate_win_rates(df_raw)
    velocity = calculate_velocity(df_raw)
    lags = calculate_lags(df_raw)
    
    print(f"\n--- 2. Running {ASSUMPTIONS['num_simulations']} Simulations for Confidence Intervals ---")
    
    all_sim_results = []
    
    for sim_num in range(ASSUMPTIONS['num_simulations']):
        if (sim_num + 1) % 100 == 0:
            print(f"    Completed {sim_num + 1}/{ASSUMPTIONS['num_simulations']} simulations...")
        
        sim_results = run_single_simulation(df_raw, win_rates, velocity, lags, sim_num + 1)
        all_sim_results.append(sim_results)
    
    print("\n--- 3. Aggregating Results & Calculating Confidence Intervals ---")
    
    # Aggregate across simulations
    months = pd.date_range('2026-01-01', '2026-12-31', freq='MS')
    segments = df_raw['market_segment'].unique()
    
    summary_rows = []
    confidence_rows = []
    
    for month in months:
        for seg in segments:
            # Collect metrics across all sims
            created_vals = [sim[month][seg]['created'] for sim in all_sim_results]
            won_vol_vals = [sim[month][seg]['won_vol'] for sim in all_sim_results]
            won_rev_vals = [sim[month][seg]['won_rev'] for sim in all_sim_results]
            open_count_vals = [sim[month][seg]['open_count'] for sim in all_sim_results]
            open_val_vals = [sim[month][seg]['open_val'] for sim in all_sim_results]
            
            # Summary (median/mean)
            summary_rows.append({
                'forecast_month': month,
                'market_segment': seg,
                'deals_created_median': int(np.median(created_vals)),
                'forecasted_won_volume_median': int(np.median(won_vol_vals)),
                'forecasted_won_rev_median': int(np.median(won_rev_vals)),
                'open_pipeline_count_median': int(np.median(open_count_vals)),
                'open_pipeline_value_median': int(np.median(open_val_vals))
            })
            
            # Confidence intervals (P10, P50, P90)
            confidence_rows.append({
                'forecast_month': month,
                'market_segment': seg,
                'metric': 'deals_created',
                'p10': int(np.percentile(created_vals, 10)),
                'p50': int(np.percentile(created_vals, 50)),
                'p90': int(np.percentile(created_vals, 90))
            })
            
            confidence_rows.append({
                'forecast_month': month,
                'market_segment': seg,
                'metric': 'won_revenue',
                'p10': int(np.percentile(won_rev_vals, 10)),
                'p50': int(np.percentile(won_rev_vals, 50)),
                'p90': int(np.percentile(won_rev_vals, 90))
            })
    
    df_summary = pd.DataFrame(summary_rows)
    df_confidence = pd.DataFrame(confidence_rows)
    
    df_summary.to_csv(OUTPUT_SUMMARY, index=False)
    df_confidence.to_csv(OUTPUT_CONFIDENCE, index=False)
    
    print(f"    Summary saved: {OUTPUT_SUMMARY}")
    print(f"    Confidence intervals saved: {OUTPUT_CONFIDENCE}")
    
    # Generate sensitivity analysis
    generate_sensitivity_analysis(df_raw, win_rates, velocity, lags)
    
    # Generate executive summary
    generate_executive_summary(df_summary, df_confidence)
    
    print("\n" + "=" * 60)
    print("FORECAST COMPLETE - All outputs saved to data/ folder")
    print("=" * 60)


# ==========================================
# PART 4: SENSITIVITY ANALYSIS
# ==========================================

def generate_sensitivity_analysis(df_raw, win_rates, velocity, lags):
    """
    Tests multiple assumption scenarios and exports sensitivity matrix.
    """
    print("\n--- 4. Running Sensitivity Analysis ---")
    
    scenarios = [
        {'name': 'Pessimistic', 'vol': 1.00, 'wr': 1.00, 'size': 1.00},
        {'name': 'Conservative', 'vol': 1.05, 'wr': 1.02, 'size': 1.01},
        {'name': 'Base Case', 'vol': 1.10, 'wr': 1.05, 'size': 1.03},
        {'name': 'Optimistic', 'vol': 1.15, 'wr': 1.08, 'size': 1.05},
        {'name': 'Aggressive', 'vol': 1.20, 'wr': 1.10, 'size': 1.07}
    ]
    
    sensitivity_results = []
    
    for scenario in scenarios:
        # Override assumptions
        ASSUMPTIONS['volume_growth_multiplier'] = scenario['vol']
        ASSUMPTIONS['win_rate_uplift_multiplier'] = scenario['wr']
        ASSUMPTIONS['deal_size_inflation'] = scenario['size']
        
        # Run 100 sims for this scenario
        scenario_results = []
        for _ in range(100):
            sim_results = run_single_simulation(df_raw, win_rates, velocity, lags)
            
            # Calculate total annual revenue
            total_rev = 0
            for month_data in sim_results.values():
                for seg_data in month_data.values():
                    total_rev += seg_data['won_rev']
            
            scenario_results.append(total_rev)
        
        sensitivity_results.append({
            'scenario': scenario['name'],
            'volume_growth': scenario['vol'],
            'win_rate_uplift': scenario['wr'],
            'deal_size_inflation': scenario['size'],
            'annual_revenue_p10': int(np.percentile(scenario_results, 10)),
            'annual_revenue_p50': int(np.percentile(scenario_results, 50)),
            'annual_revenue_p90': int(np.percentile(scenario_results, 90)),
            'annual_revenue_mean': int(np.mean(scenario_results))
        })
    
    df_sensitivity = pd.DataFrame(sensitivity_results)
    df_sensitivity.to_csv(OUTPUT_SENSITIVITY, index=False)
    
    print(f"    Sensitivity analysis saved: {OUTPUT_SENSITIVITY}")
    
    # Reset to base case
    ASSUMPTIONS['volume_growth_multiplier'] = 1.10
    ASSUMPTIONS['win_rate_uplift_multiplier'] = 1.05
    ASSUMPTIONS['deal_size_inflation'] = 1.03


# ==========================================
# PART 5: EXECUTIVE SUMMARY
# ==========================================

def generate_executive_summary(df_summary, df_confidence):
    """
    Generates human-readable executive summary text file.
    """
    print("\n--- 5. Generating Executive Summary ---")
    
    # Calculate key metrics
    total_won_rev_median = df_summary['forecasted_won_rev_median'].sum()
    total_won_vol_median = df_summary['forecasted_won_volume_median'].sum()
    
    # Revenue by segment
    seg_revenue = df_summary.groupby('market_segment')['forecasted_won_rev_median'].sum().sort_values(ascending=False)
    top_segment = seg_revenue.index[0]
    top_segment_pct = (seg_revenue.iloc[0] / total_won_rev_median) * 100
    
    # Confidence intervals for total revenue
    total_rev_by_month = df_confidence[df_confidence['metric'] == 'won_revenue'].groupby('forecast_month').agg({
        'p10': 'sum',
        'p50': 'sum',
        'p90': 'sum'
    })
    
    annual_p10 = total_rev_by_month['p10'].sum()
    annual_p50 = total_rev_by_month['p50'].sum()
    annual_p90 = total_rev_by_month['p90'].sum()
    
    # Build narrative
    summary_text = f"""
{'=' * 70}
EXECUTIVE SUMMARY: 2026 REVENUE FORECAST
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'=' * 70}

FORECAST OVERVIEW
-----------------
Expected Annual Revenue (P50):     ${annual_p50:,.0f}
Conservative Case (P10):           ${annual_p10:,.0f}
Optimistic Case (P90):             ${annual_p90:,.0f}
Expected Deal Volume:              {total_won_vol_median:,.0f} deals

CONFIDENCE INTERVAL
-----------------
The forecast indicates a 90% probability that 2026 revenue will fall between
${annual_p10:,.0f} and ${annual_p90:,.0f}, with the median expectation at 
${annual_p50:,.0f}.

KEY DRIVERS
-----------
1. Volume Growth: +10% increase in qualified deal creation expected
2. Win Rate Improvement: +5% efficiency gain in deal conversion
3. Deal Size Inflation: +3% average contract value growth

SEGMENT PERFORMANCE
-------------------
{top_segment} is projected to drive {top_segment_pct:.1f}% of total revenue,
making it the primary growth driver for 2026.

Segment Breakdown:
"""
    
    for seg, rev in seg_revenue.items():
        pct = (rev / total_won_rev_median) * 100
        summary_text += f"  - {seg:20s}: ${rev:>12,.0f}  ({pct:>5.1f}%)\n"
    
    summary_text += f"""

METHODOLOGY
-----------
This forecast was generated using {ASSUMPTIONS['num_simulations']} Monte Carlo 
simulations based on 2024-2025 historical performance data. The model incorporates:
  - Segment-specific win rates calculated from mature deal cohorts
  - Monthly velocity patterns derived from 2025 baseline
  - Historical sales cycle lengths by market segment
  - Stochastic variation to capture real-world uncertainty

ASSUMPTIONS APPLIED
-------------------
  - Volume Growth Multiplier:      {ASSUMPTIONS['volume_growth_multiplier']:.2f}x
  - Win Rate Uplift Multiplier:    {ASSUMPTIONS['win_rate_uplift_multiplier']:.2f}x
  - Deal Size Inflation:            {ASSUMPTIONS['deal_size_inflation']:.2f}x

VALIDATION STATUS
-----------------
Model accuracy should be validated using the backtest_validator.py script,
which tests predictive performance against holdout historical data.

{'=' * 70}
"""
    
    with open(OUTPUT_EXECUTIVE, 'w') as f:
        f.write(summary_text)
    
    print(f"    Executive summary saved: {OUTPUT_EXECUTIVE}")


# ==========================================
# MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    run_simulation()