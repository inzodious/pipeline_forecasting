import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from collections import defaultdict

# ==========================================
# BACKTEST VALIDATION FRAMEWORK V2
# ==========================================
"""
PURPOSE:
Validates the forecast model by training on early data and testing against 
known outcomes in later periods. Operates at MONTHLY AGGREGATION level.

METHODOLOGY:
1. Split data: Train on Jan 2024 - Jun 2025, Test on Jul - Dec 2025
2. Calculate drivers from training period only
3. Generate monthly forecast for test period
4. Compare forecasted monthly totals to actual monthly totals
5. Calculate accuracy metrics (MAPE, bias, hit rate)

This proves the model works before trusting it with future forecasts.

UPDATED V2: 
- Uses correct column names (date_snapshot, date_closed, net_revenue, stage)
- DSO calculated from date_created → date_closed (ignores date_implementation)
- Velocity counts ALL deal first appearances (not just Qualified)
- Stage detection uses pattern matching for lost stages
"""

# Paths
BASE_DIR = "data"
PATH_RAW_SNAPSHOTS = os.path.join(BASE_DIR, "fact_snapshots.csv")  # <-- Your filename
OUTPUT_BACKTEST_RESULTS = os.path.join(BASE_DIR, "backtest_validation_results.csv")
OUTPUT_BACKTEST_SUMMARY = os.path.join(BASE_DIR, "backtest_summary.txt")

# Configuration
TRAIN_END_DATE = '2025-06-30'
TEST_START_DATE = '2025-07-01'
TEST_END_DATE = '2025-12-31'

PATH_ASSUMPTIONS_LOG = os.path.join(BASE_DIR, "forecast_assumptions_log.csv")

# Stage Classifications (must match forecast_generator_v4)
WON_STAGES = ["Closed Won", "Verbal"]
LOST_PATTERNS = ["Closed Lost", "Declined to Bid"]


# ==========================================
# HELPER: STAGE CLASSIFICATION
# ==========================================

def is_won(stage):
    """Check if stage indicates a won deal."""
    if pd.isna(stage):
        return False
    return stage in WON_STAGES

def is_lost(stage):
    """Check if stage indicates a lost deal (supports wildcards)."""
    if pd.isna(stage):
        return False
    for pattern in LOST_PATTERNS:
        if pattern.lower() in stage.lower():
            return True
    return False

def is_closed(stage):
    """Check if deal is closed (won or lost)."""
    return is_won(stage) or is_lost(stage)


# ==========================================
# LOAD ASSUMPTIONS FROM FORECAST RUN
# ==========================================

def load_forecast_assumptions():
    """
    Loads the exact assumptions used in the production forecast.
    This ensures backtest uses identical parameters for fair validation.
    """
    if os.path.exists(PATH_ASSUMPTIONS_LOG):
        print("  > Loading assumptions from production forecast run...")
        df_assumptions = pd.read_csv(PATH_ASSUMPTIONS_LOG)
        
        assumptions = {}
        for _, row in df_assumptions.iterrows():
            key = row['assumption']
            value = row['value']
            
            # Convert string values to appropriate types
            if value == 'None':
                assumptions[key] = None
            elif isinstance(value, str):
                # Handle lists stored as strings
                if value.startswith('['):
                    assumptions[key] = eval(value)
                elif value.replace('.', '').replace('-', '').isdigit():
                    assumptions[key] = float(value) if '.' in value else int(value)
                else:
                    assumptions[key] = value
            else:
                assumptions[key] = value
        
        print(f"    Loaded {len(assumptions)} assumptions from forecast run")
        return assumptions
    else:
        print("  > WARNING: No forecast assumptions log found.")
        print("    Using default backtest assumptions (no growth for pure model test)")
        return {
            'volume_growth_multiplier': 1.00,
            'win_rate_uplift_multiplier': 1.00,
            'deal_size_inflation': 1.00,
            'num_simulations': 200,
            'min_monthly_deals_floor': 3,
            'max_monthly_deals_ceiling': 150,
            'min_win_rate_floor': 0.05,
            'max_win_rate_ceiling': 0.50,
            'deal_size_variance_cap': 0.25,
            'velocity_smoothing_window': 3,
            'confidence_interval_clip': 0.95
        }

# Load assumptions at module level
BACKTEST_ASSUMPTIONS = load_forecast_assumptions()


# ==========================================
# DSO CALCULATION (date_created → date_closed)
# ==========================================

def calculate_dso_backtest(df, cutoff_date):
    """
    Calculate DSO distribution using only training data.
    Uses date_created → date_closed, NOT date_implementation.
    """
    df_final = df.sort_values('date_snapshot').groupby('deal_id').last().reset_index()
    
    # Filter to closed deals
    df_final['is_won_flag'] = df_final['stage'].apply(is_won)
    df_final['is_lost_flag'] = df_final['stage'].apply(is_lost)
    df_closed = df_final[df_final['is_won_flag'] | df_final['is_lost_flag']].copy()
    
    # Only use deals closed before cutoff
    df_closed = df_closed[df_closed['date_closed'] <= cutoff_date].copy()
    
    df_closed['cycle_days'] = (df_closed['date_closed'] - df_closed['date_created']).dt.days
    df_valid = df_closed[(df_closed['cycle_days'] >= 7) & (df_closed['cycle_days'] <= 365)]
    
    dso_dict = {}
    
    for seg in df['market_segment'].unique():
        seg_data = df_valid[df_valid['market_segment'] == seg]
        
        if len(seg_data) >= 5:
            dso_dict[seg] = {
                'mean': seg_data['cycle_days'].mean(),
                'std': max(seg_data['cycle_days'].std(), 10),
                'median': seg_data['cycle_days'].median()
            }
        else:
            # Fallback to overall
            dso_dict[seg] = {
                'mean': df_valid['cycle_days'].mean(),
                'std': max(df_valid['cycle_days'].std(), 10),
                'median': df_valid['cycle_days'].median()
            }
    
    return dso_dict


# ==========================================
# WIN RATE CALCULATION
# ==========================================

def calculate_win_rates_backtest(df, cutoff_date):
    """Calculate revenue-weighted win rates using only training data."""
    
    df_final = df.sort_values('date_snapshot').groupby('deal_id').last().reset_index()
    
    # Filter to deals closed before cutoff
    df_final['is_won_flag'] = df_final['stage'].apply(is_won)
    df_final['is_lost_flag'] = df_final['stage'].apply(is_lost)
    df_final['is_closed_flag'] = df_final['is_won_flag'] | df_final['is_lost_flag']
    
    df_closed = df_final[
        (df_final['is_closed_flag']) & 
        (df_final['date_closed'] <= cutoff_date)
    ].copy()
    
    if len(df_closed) == 0:
        return {seg: 0.20 for seg in df['market_segment'].unique()}
    
    win_rates = {}
    
    for seg in df['market_segment'].unique():
        seg_data = df_closed[df_closed['market_segment'] == seg]
        
        won_rev = seg_data[seg_data['is_won_flag']]['net_revenue'].sum()
        lost_rev = seg_data[seg_data['is_lost_flag']]['net_revenue'].sum()
        total_rev = won_rev + lost_rev
        
        raw_wr = (won_rev / total_rev) if total_rev > 0 else 0.20
        
        win_rates[seg] = np.clip(
            raw_wr,
            BACKTEST_ASSUMPTIONS.get('min_win_rate_floor', 0.05),
            BACKTEST_ASSUMPTIONS.get('max_win_rate_ceiling', 0.50)
        )
    
    return win_rates


# ==========================================
# VELOCITY CALCULATION (ALL first appearances)
# ==========================================

def calculate_velocity_backtest(df, cutoff_date):
    """
    Calculate velocity using only training data.
    Counts ALL deal first appearances, regardless of entry stage.
    """
    
    # Filter to snapshots before cutoff
    df_train = df[df['date_snapshot'] <= cutoff_date].copy()
    
    # Find first appearance of each deal
    df_first = df_train.sort_values('date_snapshot').groupby('deal_id').first().reset_index()
    df_first['appear_year'] = df_first['date_snapshot'].dt.year
    df_first['appear_month'] = df_first['date_snapshot'].dt.month
    
    # Use most recent complete year in training data
    cutoff_year = cutoff_date.year
    cutoff_month = cutoff_date.month
    
    # If cutoff is mid-year, use prior year as baseline
    if cutoff_month < 12:
        baseline_year = cutoff_year - 1
    else:
        baseline_year = cutoff_year
    
    # Make sure we have data for that year
    if baseline_year not in df_first['appear_year'].values:
        baseline_year = df_first['appear_year'].max()
    
    df_baseline = df_first[df_first['appear_year'] == baseline_year].copy()
    
    velocity_dict = {}
    
    for seg in df['market_segment'].unique():
        velocity_dict[seg] = {}
        seg_data = df_baseline[df_baseline['market_segment'] == seg]
        
        annual_count = len(seg_data)
        monthly_avg = max(BACKTEST_ASSUMPTIONS.get('min_monthly_deals_floor', 3), annual_count / 12)
        annual_avg_size = seg_data['net_revenue'].mean() if len(seg_data) > 0 else 50000
        
        monthly_vols = []
        monthly_sizes = []
        
        for m in range(1, 13):
            m_data = seg_data[seg_data['appear_month'] == m]
            
            if len(m_data) >= 3:
                monthly_vols.append(len(m_data))
                monthly_sizes.append(m_data['net_revenue'].mean())
            else:
                monthly_vols.append(monthly_avg)
                monthly_sizes.append(annual_avg_size)
        
        # Apply smoothing
        smoothing = BACKTEST_ASSUMPTIONS.get('velocity_smoothing_window', 3)
        if smoothing > 1:
            monthly_vols_smooth = pd.Series(monthly_vols).rolling(
                window=smoothing, center=True, min_periods=1
            ).mean().tolist()
        else:
            monthly_vols_smooth = monthly_vols
        
        for m in range(1, 13):
            vol = np.clip(
                monthly_vols_smooth[m - 1],
                BACKTEST_ASSUMPTIONS.get('min_monthly_deals_floor', 3),
                BACKTEST_ASSUMPTIONS.get('max_monthly_deals_ceiling', 150)
            )
            velocity_dict[seg][m] = {'vol': vol, 'size': monthly_sizes[m - 1]}
    
    return velocity_dict


# ==========================================
# BACKTEST FORECAST ENGINE
# ==========================================

class BacktestEngine:
    """Forecast engine for backtest - mirrors production logic."""
    
    def __init__(self, win_rates, velocity, dso_dict, test_months):
        self.win_rates = win_rates
        self.velocity = velocity
        self.dso_dict = dso_dict
        self.test_months = test_months
    
    def generate_monthly_deals(self, month, segment):
        """Generate deals for a specific month/segment."""
        month_num = month.month
        
        if segment not in self.velocity or month_num not in self.velocity[segment]:
            return []
        
        stats = self.velocity[segment][month_num]
        
        # Apply growth (for backtest, typically set to 1.0 for fair comparison)
        vol_growth = BACKTEST_ASSUMPTIONS.get('volume_growth_multiplier', 1.0)
        base_vol = stats['vol'] * vol_growth
        base_vol = np.clip(
            base_vol,
            BACKTEST_ASSUMPTIONS.get('min_monthly_deals_floor', 3),
            BACKTEST_ASSUMPTIONS.get('max_monthly_deals_ceiling', 150)
        )
        
        num_deals = max(
            BACKTEST_ASSUMPTIONS.get('min_monthly_deals_floor', 3),
            np.random.poisson(base_vol)
        )
        
        deals = []
        size_inflation = BACKTEST_ASSUMPTIONS.get('deal_size_inflation', 1.0)
        avg_size = stats['size'] * size_inflation
        
        base_wr = self.win_rates.get(segment, 0.20)
        wr_uplift = BACKTEST_ASSUMPTIONS.get('win_rate_uplift_multiplier', 1.0)
        adj_wr = np.clip(
            base_wr * wr_uplift,
            BACKTEST_ASSUMPTIONS.get('min_win_rate_floor', 0.05),
            BACKTEST_ASSUMPTIONS.get('max_win_rate_ceiling', 0.50)
        )
        
        seg_dso = self.dso_dict.get(segment, {'mean': 60, 'std': 30})
        
        for _ in range(num_deals):
            is_won = np.random.random() < adj_wr
            
            variance_cap = BACKTEST_ASSUMPTIONS.get('deal_size_variance_cap', 0.25)
            size_mult = 1.0 + np.random.uniform(-variance_cap, variance_cap)
            actual_rev = int(avg_size * size_mult)
            
            days_in_month = (month + pd.DateOffset(months=1) - timedelta(days=1)).day
            create_day = np.random.randint(1, days_in_month + 1)
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
                'create_month': month
            })
        
        return deals
    
    def run_simulation(self, existing_deals):
        """Run monthly simulation."""
        monthly_results = {month: defaultdict(lambda: {
            'won_vol': 0, 'won_rev': 0
        }) for month in self.test_months}
        
        all_deals = list(existing_deals)
        
        # Generate new deals for each test month
        for month in self.test_months:
            for segment in self.velocity.keys():
                new_deals = self.generate_monthly_deals(month, segment)
                all_deals.extend(new_deals)
        
        # Aggregate by close month
        for deal in all_deals:
            close_month = pd.Period(deal['close_date'], freq='M').to_timestamp()
            
            if close_month in monthly_results and deal['is_won']:
                seg = deal['segment']
                monthly_results[close_month][seg]['won_vol'] += 1
                monthly_results[close_month][seg]['won_rev'] += deal['revenue']
        
        return monthly_results


# ==========================================
# EXTRACT ACTUAL RESULTS
# ==========================================

def extract_actual_monthly_results(df, test_start, test_end):
    """Extract actual monthly results from test period using date_closed."""
    
    test_start = pd.to_datetime(test_start)
    test_end = pd.to_datetime(test_end)
    
    # Get final state of each deal
    df_final = df.sort_values('date_snapshot').groupby('deal_id').last().reset_index()
    
    # Filter to won deals closed in test period
    df_final['is_won_flag'] = df_final['stage'].apply(is_won)
    
    df_won = df_final[
        (df_final['is_won_flag']) &
        (df_final['date_closed'] >= test_start) &
        (df_final['date_closed'] <= test_end)
    ].copy()
    
    df_won['close_month'] = df_won['date_closed'].dt.to_period('M').dt.to_timestamp()
    
    # Aggregate by month and segment
    actual_monthly = defaultdict(lambda: defaultdict(lambda: {
        'actual_won_vol': 0,
        'actual_won_rev': 0
    }))
    
    for _, row in df_won.iterrows():
        month = row['close_month']
        seg = row['market_segment']
        actual_monthly[month][seg]['actual_won_vol'] += 1
        actual_monthly[month][seg]['actual_won_rev'] += row['net_revenue']
    
    return actual_monthly


# ==========================================
# COMPARISON & METRICS
# ==========================================

def compare_forecast_to_actual(forecasted_list, actual):
    """Compare average forecasted results to actual."""
    
    # Average across simulations
    avg_forecasted = defaultdict(lambda: defaultdict(lambda: {
        'forecasted_won_vol': 0,
        'forecasted_won_rev': 0
    }))
    
    for forecasted in forecasted_list:
        for month, seg_data in forecasted.items():
            for seg, metrics in seg_data.items():
                avg_forecasted[month][seg]['forecasted_won_vol'] += metrics['won_vol']
                avg_forecasted[month][seg]['forecasted_won_rev'] += metrics['won_rev']
    
    num_sims = len(forecasted_list)
    for month in avg_forecasted:
        for seg in avg_forecasted[month]:
            avg_forecasted[month][seg]['forecasted_won_vol'] /= num_sims
            avg_forecasted[month][seg]['forecasted_won_rev'] /= num_sims
    
    # Build comparison
    comparison_rows = []
    
    all_months = set(avg_forecasted.keys()) | set(actual.keys())
    all_segments = set()
    
    for month_data in list(avg_forecasted.values()) + list(actual.values()):
        all_segments.update(month_data.keys())
    
    for month in sorted(all_months):
        for seg in sorted(all_segments):
            f_vol = avg_forecasted.get(month, {}).get(seg, {}).get('forecasted_won_vol', 0)
            f_rev = avg_forecasted.get(month, {}).get(seg, {}).get('forecasted_won_rev', 0)
            
            a_vol = actual.get(month, {}).get(seg, {}).get('actual_won_vol', 0)
            a_rev = actual.get(month, {}).get(seg, {}).get('actual_won_rev', 0)
            
            vol_error = f_vol - a_vol
            rev_error = f_rev - a_rev
            
            vol_pct_error = (vol_error / a_vol * 100) if a_vol > 0 else 0
            rev_pct_error = (rev_error / a_rev * 100) if a_rev > 0 else 0
            
            comparison_rows.append({
                'month': month,
                'market_segment': seg,
                'forecasted_won_volume': round(f_vol, 1),
                'actual_won_volume': a_vol,
                'volume_error': round(vol_error, 1),
                'volume_pct_error': round(vol_pct_error, 1),
                'forecasted_won_revenue': int(f_rev),
                'actual_won_revenue': int(a_rev),
                'revenue_error': int(rev_error),
                'revenue_pct_error': round(rev_pct_error, 1)
            })
    
    return pd.DataFrame(comparison_rows)


def calculate_accuracy_metrics(df_comparison):
    """Calculate summary accuracy statistics."""
    
    df_valid = df_comparison[df_comparison['actual_won_revenue'] > 0].copy()
    
    if len(df_valid) == 0:
        return {
            'mape_revenue': 0, 'bias_revenue': 0, 'rmse_revenue': 0,
            'hit_rate_within_20pct': 0, 'num_periods_analyzed': 0
        }
    
    mape_rev = df_valid['revenue_pct_error'].abs().mean()
    bias_rev = df_valid['revenue_pct_error'].mean()
    rmse_rev = np.sqrt((df_valid['revenue_error'] ** 2).mean())
    
    within_20pct = (df_valid['revenue_pct_error'].abs() <= 20).sum()
    hit_rate = (within_20pct / len(df_valid)) * 100
    
    return {
        'mape_revenue': mape_rev,
        'bias_revenue': bias_rev,
        'rmse_revenue': rmse_rev,
        'hit_rate_within_20pct': hit_rate,
        'num_periods_analyzed': len(df_valid)
    }


# ==========================================
# MAIN BACKTEST EXECUTION
# ==========================================

def run_backtest_validation():
    """Main backtest validation workflow."""
    print("=" * 70)
    print("BACKTEST VALIDATION V2 - CORRECTED LOGIC")
    print("=" * 70)
    
    print("\n--- 0. Loading Production Forecast Assumptions ---")
    if os.path.exists(PATH_ASSUMPTIONS_LOG):
        df_show = pd.read_csv(PATH_ASSUMPTIONS_LOG)
        print("\n    Production forecast used these assumptions:")
        for _, row in df_show.iterrows():
            print(f"      {row['assumption']:35s} = {row['value']}")
        print("\n    Backtest will use IDENTICAL assumptions for fair comparison.")
    else:
        print("\n    No forecast log found - using default assumptions.")
        print("    Run forecast_generator_v4.py first to generate assumptions log.")
    
    print("\n--- 1. Loading Historical Data ---")
    if not os.path.exists(PATH_RAW_SNAPSHOTS):
        print(f"Error: Snapshot file not found: {PATH_RAW_SNAPSHOTS}")
        return
    
    df_raw = pd.read_csv(PATH_RAW_SNAPSHOTS)
    df_raw['date_snapshot'] = pd.to_datetime(df_raw['date_snapshot'])
    df_raw['date_created'] = pd.to_datetime(df_raw['date_created'])
    df_raw['date_closed'] = pd.to_datetime(df_raw['date_closed'])
    
    train_end = pd.to_datetime(TRAIN_END_DATE)
    test_start = pd.to_datetime(TEST_START_DATE)
    test_end = pd.to_datetime(TEST_END_DATE)
    
    df_train = df_raw[df_raw['date_snapshot'] <= train_end].copy()
    df_full = df_raw.copy()  # Need full data to extract actuals
    
    print(f"    Training: {df_train['date_snapshot'].min().date()} to {train_end.date()}")
    print(f"    Testing:  {test_start.date()} to {test_end.date()}")
    print(f"    Training records: {len(df_train):,}")
    print(f"    Unique deals in training: {df_train['deal_id'].nunique():,}")
    
    print("\n--- 2. Calculating Drivers (Training Data Only) ---")
    dso_dict = calculate_dso_backtest(df_train, train_end)
    win_rates = calculate_win_rates_backtest(df_train, train_end)
    velocity = calculate_velocity_backtest(df_train, train_end)
    
    print(f"    Win rates calculated for {len(win_rates)} segments")
    for seg, wr in win_rates.items():
        dso = dso_dict.get(seg, {}).get('median', 60)
        print(f"      {seg}: WR={wr:.1%}, DSO={dso:.0f} days")
    
    print("\n--- 3. Initializing Pipeline (as of training cutoff) ---")
    
    # Get open deals as of training end date
    latest_train_snapshot = df_train['date_snapshot'].max()
    df_latest = df_train[df_train['date_snapshot'] == latest_train_snapshot].copy()
    
    df_latest['is_closed_flag'] = df_latest['stage'].apply(is_closed)
    df_open = df_latest[~df_latest['is_closed_flag']].copy()
    
    print(f"    Latest training snapshot: {latest_train_snapshot.date()}")
    print(f"    Open deals at cutoff: {len(df_open):,}")
    
    existing_deals = []
    for _, row in df_open.iterrows():
        seg = row['market_segment']
        create_date = pd.to_datetime(row['date_created'])
        
        base_wr = win_rates.get(seg, 0.20)
        wr_uplift = BACKTEST_ASSUMPTIONS.get('win_rate_uplift_multiplier', 1.0)
        adj_wr = np.clip(
            base_wr * wr_uplift,
            BACKTEST_ASSUMPTIONS.get('min_win_rate_floor', 0.05),
            BACKTEST_ASSUMPTIONS.get('max_win_rate_ceiling', 0.50)
        )
        is_won = np.random.random() < adj_wr
        
        seg_dso = dso_dict.get(seg, {'mean': 60, 'std': 30})
        days_open = (latest_train_snapshot - create_date).days
        
        total_cycle = int(np.random.normal(seg_dso['mean'], seg_dso['std']))
        total_cycle = max(days_open + np.random.randint(7, 45), total_cycle)
        
        close_date = create_date + timedelta(days=total_cycle)
        
        existing_deals.append({
            'segment': seg,
            'revenue': row['net_revenue'],
            'create_date': create_date,
            'close_date': close_date,
            'is_won': is_won
        })
    
    print("\n--- 4. Running Backtest Simulations ---")
    
    test_months = pd.date_range(test_start, test_end, freq='MS')
    num_sims = BACKTEST_ASSUMPTIONS.get('num_simulations', 200)
    
    engine = BacktestEngine(win_rates, velocity, dso_dict, test_months)
    
    all_forecasted = []
    
    for sim in range(num_sims):
        if (sim + 1) % 50 == 0:
            print(f"    Completed {sim + 1}/{num_sims} simulations...")
        
        forecasted = engine.run_simulation(existing_deals)
        all_forecasted.append(forecasted)
    
    print("\n--- 5. Extracting Actual Results (Test Period) ---")
    actual_results = extract_actual_monthly_results(df_full, test_start, test_end)
    
    # Display actual totals
    total_actual_vol = sum(
        seg_data['actual_won_vol'] 
        for month_data in actual_results.values() 
        for seg_data in month_data.values()
    )
    total_actual_rev = sum(
        seg_data['actual_won_rev'] 
        for month_data in actual_results.values() 
        for seg_data in month_data.values()
    )
    print(f"    Actual test period: {total_actual_vol} deals, ${total_actual_rev:,.0f}")
    
    print("\n--- 6. Comparing Forecast to Actual ---")
    df_comparison = compare_forecast_to_actual(all_forecasted, actual_results)
    df_comparison.to_csv(OUTPUT_BACKTEST_RESULTS, index=False)
    
    # Show monthly comparison
    monthly_comp = df_comparison.groupby('month').agg({
        'forecasted_won_volume': 'sum',
        'actual_won_volume': 'sum',
        'forecasted_won_revenue': 'sum',
        'actual_won_revenue': 'sum'
    })
    
    print("\n    Monthly Comparison:")
    print(f"    {'Month':<12} {'Fcst Vol':>10} {'Actual Vol':>12} {'Fcst Rev':>14} {'Actual Rev':>14}")
    print("    " + "-" * 66)
    
    for month, row in monthly_comp.iterrows():
        print(f"    {month.strftime('%Y-%m'):<12} {row['forecasted_won_volume']:>10.0f} {row['actual_won_volume']:>12.0f} ${row['forecasted_won_revenue']:>13,.0f} ${row['actual_won_revenue']:>13,.0f}")
    
    print(f"\n    Results saved: {OUTPUT_BACKTEST_RESULTS}")
    
    print("\n--- 7. Calculating Accuracy Metrics ---")
    metrics = calculate_accuracy_metrics(df_comparison)
    
    # Generate summary
    summary_text = f"""
{'=' * 70}
BACKTEST VALIDATION SUMMARY V2
{'=' * 70}

TEST PERIOD: {test_start.date()} to {test_end.date()} (6 months)
TRAINING PERIOD: {df_train['date_snapshot'].min().date()} to {train_end.date()}
AGGREGATION LEVEL: Monthly totals by segment

ACCURACY METRICS
----------------
Mean Absolute Percentage Error (MAPE):  {metrics['mape_revenue']:.2f}%
Forecast Bias:                          {metrics['bias_revenue']:+.2f}%
Root Mean Squared Error (RMSE):         ${metrics['rmse_revenue']:,.0f}
Hit Rate (within ±20%):                 {metrics['hit_rate_within_20pct']:.1f}%

Month-Segment Periods Analyzed:         {metrics['num_periods_analyzed']}

ACTUAL VS FORECAST TOTALS
-------------------------
Actual Test Period Revenue:             ${total_actual_rev:,.0f}
Forecasted Test Period Revenue:         ${monthly_comp['forecasted_won_revenue'].sum():,.0f}
Difference:                             ${monthly_comp['forecasted_won_revenue'].sum() - total_actual_rev:+,.0f}

INTERPRETATION
--------------
"""
    
    if metrics['mape_revenue'] < 15:
        summary_text += "✓ EXCELLENT: Model demonstrates strong predictive accuracy (MAPE < 15%)\n"
    elif metrics['mape_revenue'] < 25:
        summary_text += "✓ GOOD: Model shows acceptable predictive accuracy (MAPE < 25%)\n"
    elif metrics['mape_revenue'] < 40:
        summary_text += "⚠ MODERATE: Model accuracy is acceptable but could improve (MAPE < 40%)\n"
    else:
        summary_text += "⚠ REVIEW NEEDED: Model accuracy below target (MAPE > 40%)\n"
    
    if abs(metrics['bias_revenue']) < 10:
        summary_text += "✓ UNBIASED: No significant systematic over/under-forecasting\n"
    elif metrics['bias_revenue'] > 10:
        summary_text += "⚠ UPWARD BIAS: Model tends to over-forecast\n"
    else:
        summary_text += "⚠ DOWNWARD BIAS: Model tends to under-forecast\n"
    
    if metrics['hit_rate_within_20pct'] > 60:
        summary_text += "✓ GOOD PRECISION: Majority of forecasts within ±20% of actual\n"
    else:
        summary_text += "⚠ MODERATE PRECISION: Consider tightening variance constraints\n"
    
    summary_text += f"""

METHODOLOGY NOTES (V2 Corrections)
----------------------------------
1. DSO calculated from date_created → date_closed (NOT date_implementation)
2. Velocity counts ALL deal first appearances, not just 'Qualified' stage
3. Win rates are revenue-weighted from closed deals only
4. Stage classification uses pattern matching for lost stages

ASSUMPTIONS USED IN BACKTEST
----------------------------
"""
    
    for key, value in BACKTEST_ASSUMPTIONS.items():
        if not key.startswith('_'):
            summary_text += f"  {key:40s}: {value}\n"
    
    summary_text += f"""

RECOMMENDATION
--------------
"""
    
    if metrics['mape_revenue'] < 25 and abs(metrics['bias_revenue']) < 15:
        summary_text += "✓ Model is suitable for production forecasting.\n"
        summary_text += "  Proceed with 2026 forecast with reasonable confidence.\n"
    else:
        summary_text += "⚠ Model may require calibration.\n"
        summary_text += "  Review driver calculations and consider adjusting assumptions.\n"
    
    summary_text += f"\n{'=' * 70}\n"
    
    print(summary_text)
    
    with open(OUTPUT_BACKTEST_SUMMARY, 'w') as f:
        f.write(summary_text)
    
    print(f"Summary saved: {OUTPUT_BACKTEST_SUMMARY}")
    
    print("\n" + "=" * 70)
    print("BACKTEST VALIDATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    run_backtest_validation()