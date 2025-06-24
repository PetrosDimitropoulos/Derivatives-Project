import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


# Load the cleaned Excel file
file_path = "/Users/petrosdimitropoulos/Downloads/50 stocks_FTSE100.xlsx"
df = pd.read_excel(file_path)

# Convert the 'Date' column to datetime and set it as the index
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Ensure all price columns are numeric
df = df.apply(pd.to_numeric, errors='coerce')

# Calculate monthly percentage returns
stock_returns = df.pct_change().dropna()

# Display first few rows of returns
print(stock_returns.head())

# --- Load risk-free rate (monthly, in percent annualized) ---
rf_df = pd.read_excel("/Users/petrosdimitropoulos/Downloads/Risk free rate.xlsx")
rf_df['Date'] = pd.to_datetime(rf_df['Date'])
rf_df.set_index('Date', inplace=True)

# Convert annual percentage rates to equivalent monthly compounded rates
rf_series = (1 + rf_df.iloc[:, 0] / 100) ** (1/12) - 1
rf_series = rf_series.reindex(stock_returns.index).dropna()


# --- Efficient Portfolio Frontier (EPF) calculation and plotting ---

# Compute expected returns and covariance matrix
mu = stock_returns.mean()
cov_matrix = stock_returns.cov()
num_assets = len(mu)

# Range of target returns for the EPF
target_returns = np.linspace(mu.min(), mu.max(), 100)
frontier_risks = []

# Constraints template
def get_constraints(target_return):
    return [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # sum of weights = 1
        {'type': 'eq', 'fun': lambda w: w @ mu - target_return}  # expected return = target
    ]

# Bounds for weights: no short selling
bounds = tuple((0, 1) for _ in range(num_assets))

# Initial guess: equally weighted portfolio
w0 = np.ones(num_assets) / num_assets

# Solve for each target return
for R_target in target_returns:
    constraints = get_constraints(R_target)
    result = minimize(lambda w: w.T @ cov_matrix @ w, w0, method='SLSQP', bounds=bounds, constraints=constraints)
    if result.success:
        variance = result.fun
        frontier_risks.append(np.sqrt(variance))
    else:
        frontier_risks.append(np.nan)


# --- Tangency Portfolio and Capital Market Line (CML) ---

# Use average risk-free rate over the period
risk_free_rate = rf_series.mean()

# Negative Sharpe ratio (since we minimize)
def neg_sharpe_ratio(w, mu, cov, rf):
    port_return = w @ mu
    port_vol = np.sqrt(w.T @ cov @ w)
    return -(port_return - rf) / port_vol

# Constraints: fully invested, no short-selling
constraints_tangent = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
bounds_tangent = tuple((0, 1) for _ in range(num_assets))

# Solve for Tangency Portfolio
result_tangent = minimize(neg_sharpe_ratio, w0, args=(mu, cov_matrix, risk_free_rate),
                          method='SLSQP', bounds=bounds_tangent, constraints=constraints_tangent)

w_tangent = result_tangent.x
ret_tangent = w_tangent @ mu
std_tangent = np.sqrt(w_tangent.T @ cov_matrix @ w_tangent)

# Capital Market Line (CML)
cml_x = np.linspace(0, std_tangent + 0.02, 100)
cml_y = risk_free_rate + (ret_tangent - risk_free_rate) / std_tangent * cml_x

# --- Global Minimum Variance Portfolio (GMVP) ---

# Constraints: fully invested, no short-selling
gmvp_constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
gmvp_bounds = tuple((0, 1) for _ in range(num_assets))

# Objective: minimize portfolio variance
def portfolio_variance(w, cov):
    return w.T @ cov @ w

gmvp_result = minimize(portfolio_variance, w0, args=(cov_matrix,), method='SLSQP',
                       bounds=gmvp_bounds, constraints=gmvp_constraints)

w_gmvp = gmvp_result.x
gmvp_return = w_gmvp @ mu
gmvp_std = np.sqrt(w_gmvp.T @ cov_matrix @ w_gmvp)


# Plot the EPF with the CML, GMVP and Tangent
plt.figure(figsize=(10, 6))
plt.plot(frontier_risks, target_returns, label='Efficient Frontier', color='b')
plt.plot(cml_x, cml_y, label='Capital Market Line', color='g', linestyle='--')
plt.scatter(std_tangent, ret_tangent, color='r', label='Tangency Portfolio', zorder=5)
plt.scatter(gmvp_std, gmvp_return, color='m', label='GMVP', zorder=5)
plt.xlabel('Portfolio Standard Deviation (Risk)')
plt.ylabel('Portfolio Expected Return')
plt.title('Efficient Frontier and Capital Market Line')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# --- Rolling Window Backtest (Unhedged Tangency Portfolio) ---
rolling_window = 180
portfolio_returns = []
portfolio_weights = []
test_dates = stock_returns.index[rolling_window:]

for i in range(rolling_window, len(stock_returns)):
    window_returns = stock_returns.iloc[i - rolling_window:i]
    mu_t = window_returns.mean()
    cov_t = window_returns.cov()
    rf_t = rf_series.iloc[i - 1]  # align with month t-1

    # Max Sharpe (Tangency) Portfolio at time t
    def neg_sharpe(w):
        port_ret = w @ mu_t
        port_vol = np.sqrt(w.T @ cov_t @ w)
        return -(port_ret - rf_t) / port_vol

    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bounds = tuple((0, 1) for _ in range(len(mu_t)))
    result = minimize(neg_sharpe, w0, method='SLSQP', bounds=bounds, constraints=constraints)

    if result.success:
        w_opt = result.x
        ret = stock_returns.iloc[i] @ w_opt
        portfolio_returns.append(ret)
        portfolio_weights.append(w_opt)
    else:
        portfolio_returns.append(np.nan)
        portfolio_weights.append([np.nan] * len(mu_t))

# Store backtested results
backtest_df = pd.DataFrame({
    'Date': test_dates,
    'Return': portfolio_returns
}).set_index('Date')


# Compute cumulative portfolio value, starting from £1,000,000
initial_value = 1_000_000
backtest_df['Value'] = initial_value * (1 + backtest_df['Return']).cumprod()


# Now that backtest_df exists, compute GMVP returns and DataFrame properly
gmvp_returns_series = stock_returns.loc[backtest_df.index] @ w_gmvp
gmvp_df = pd.DataFrame({'Return': gmvp_returns_series}, index=backtest_df.index)
gmvp_df['Value'] = 1_000_000 * (1 + gmvp_df['Return']).cumprod()

# --- Load FTSE 100 Total Return Index ---
ftse_df = pd.read_excel("/Users/petrosdimitropoulos/Downloads/FTSE100 Index.xlsx")
ftse_df['Date'] = pd.to_datetime(ftse_df['Date'])
ftse_df.set_index('Date', inplace=True)

# Calculate FTSE 100 monthly returns
ftse_returns = ftse_df['FTSE100(RI)'].pct_change().dropna()
ftse_returns = ftse_returns.reindex(backtest_df.index).dropna()

# Compute cumulative value of FTSE 100 starting from £1,000,000
ftse_value = 1_000_000 * (1 + ftse_returns).cumprod()


# --- Hedging the Portfolio with FTSE 100 Futures ---
hedged_returns = []
hedge_flags = []
ftse_prices = ftse_df['FTSE100(RI)'].reindex(stock_returns.index)

for i in range(rolling_window, len(stock_returns) - 1):
    r_index_t = ftse_prices.pct_change().iloc[i]

    if r_index_t < 0:
        # Estimate beta
        w_opt = portfolio_weights[i - rolling_window]
        port_hist_returns = stock_returns.iloc[i - rolling_window:i] @ w_opt
        index_hist_returns = ftse_prices.pct_change().iloc[i - rolling_window:i]
        beta = np.cov(port_hist_returns, index_hist_returns)[0, 1] / np.var(index_hist_returns)

        # Portfolio value at time t
        V_t = backtest_df['Value'].iloc[i - rolling_window]

        # Index level and futures contract value
        index_price_t = ftse_prices.iloc[i]
        contract_value = 10 * index_price_t
        N_t = beta * V_t / contract_value

        # Hedge return at t+1
        r_index_next = ftse_prices.pct_change().iloc[i + 1]
        hedge_return = -beta * r_index_next

        # Adjusted portfolio return
        r_portfolio = portfolio_returns[i - rolling_window + 1] + hedge_return
        hedged_returns.append(r_portfolio)
        hedge_flags.append(True)
    else:
        hedged_returns.append(portfolio_returns[i - rolling_window + 1])
        hedge_flags.append(False)

# Create hedged DataFrame
hedged_dates = stock_returns.index[rolling_window + 1:]
hedged_df = pd.DataFrame({
    'Date': hedged_dates,
    'Return': hedged_returns,
    'Hedged': hedge_flags
}).set_index('Date')

# Compute hedged portfolio value
hedged_df['Value'] = 1_000_000 * (1 + hedged_df['Return']).cumprod()
# Ensure hedged portfolio covers all dates
hedged_df = hedged_df.reindex(backtest_df.index)
hedged_df['Return'] = hedged_df['Return'].fillna(backtest_df['Return'])
hedged_df['Hedged'] = hedged_df['Hedged'].fillna(False)
hedged_df['Value'] = 1_000_000 * (1 + hedged_df['Return']).cumprod()

# --- Hedging the GMVP Portfolio ---
gmvp_hedged_returns = []
gmvp_hedge_flags = []

for i in range(rolling_window, len(stock_returns) - 1):
    r_index_t = ftse_prices.pct_change().iloc[i]

    if r_index_t < 0:
        # Estimate beta of GMVP to index using historical returns
        gmvp_hist_returns = stock_returns.iloc[i - rolling_window:i] @ w_gmvp
        index_hist_returns = ftse_prices.pct_change().iloc[i - rolling_window:i]
        beta = np.cov(gmvp_hist_returns, index_hist_returns)[0, 1] / np.var(index_hist_returns)

        # GMVP value at time t
        V_t = gmvp_df['Value'].iloc[i - rolling_window]

        # Futures contract value
        index_price_t = ftse_prices.iloc[i]
        contract_value = 10 * index_price_t
        N_t = beta * V_t / contract_value

        # Hedge return at t+1
        r_index_next = ftse_prices.pct_change().iloc[i + 1]
        hedge_return = -beta * r_index_next

        # Adjusted GMVP return at t+1
        r_portfolio = gmvp_returns_series.iloc[i - rolling_window + 1] + hedge_return
        gmvp_hedged_returns.append(r_portfolio)
        gmvp_hedge_flags.append(True)
    else:
        gmvp_hedged_returns.append(gmvp_returns_series.iloc[i - rolling_window + 1])
        gmvp_hedge_flags.append(False)

# Create hedged GMVP DataFrame
gmvp_hedged_dates = stock_returns.index[rolling_window + 1:]
gmvp_hedged_df = pd.DataFrame({
    'Date': gmvp_hedged_dates,
    'Return': gmvp_hedged_returns,
    'Hedged': gmvp_hedge_flags
}).set_index('Date')

gmvp_hedged_df['Value'] = 1_000_000 * (1 + gmvp_hedged_df['Return']).cumprod()
# Ensure GMVP hedged portfolio covers all dates
gmvp_hedged_df = gmvp_hedged_df.reindex(gmvp_df.index)
gmvp_hedged_df['Return'] = gmvp_hedged_df['Return'].fillna(gmvp_df['Return'])
gmvp_hedged_df['Hedged'] = gmvp_hedged_df['Hedged'].fillna(False)
gmvp_hedged_df['Value'] = 1_000_000 * (1 + gmvp_hedged_df['Return']).cumprod()


# --- Performance Summary ---
def performance_summary(returns, name):
    total_return = (1 + returns).prod() - 1
    avg_return = returns.mean()
    std_dev = returns.std()
    sharpe = (avg_return - risk_free_rate) / std_dev
    print(f"\n{name} Performance:")
    print(f"Total Return: {total_return:.2%}")
    print(f"Average Monthly Return: {avg_return:.4%}")
    print(f"Standard Deviation: {std_dev:.4%}")
    print(f"Sharpe Ratio: {sharpe:.4f}")
    dsharpe = downside_sharpe(returns, risk_free_rate)
    upr = up_ratio(returns)
    loss = return_loss(returns, ftse_returns.reindex(returns.index).dropna())

    print(f"Downside Sharpe Ratio: {dsharpe:.4f}")
    print(f"U-P Ratio: {upr:.4f}")
    print(f"Return Loss vs Index: {loss:.4%}")

def downside_sharpe(returns, rf):
    downside_returns = returns[returns < rf]
    downside_std = downside_returns.std()
    if downside_std == 0:
        return np.nan
    return (returns.mean() - rf) / downside_std

def up_ratio(returns):
    up = (returns > 0).sum()
    down = (returns < 0).sum()
    return up / down if down != 0 else np.nan

def return_loss(actual_returns, benchmark_returns):
    diff = benchmark_returns - actual_returns
    return np.mean(diff[diff > 0])

performance_summary(backtest_df['Return'], "Unhedged Portfolio")
performance_summary(hedged_df['Return'], "Hedged Portfolio")
performance_summary(gmvp_df['Return'], "GMVP Portfolio")
performance_summary(ftse_returns, "FTSE 100 Index")
performance_summary(gmvp_hedged_df['Return'], "GMVP Hedged Portfolio")


# --- Plot Cumulative Return of All Portfolios ---
def cumulative_return(returns):
    return (1 + returns).cumprod() - 1

cumulative_unhedged = cumulative_return(backtest_df['Return'])
cumulative_hedged = cumulative_return(hedged_df['Return'])
cumulative_gmvp = cumulative_return(gmvp_df['Return'])
cumulative_gmvp_hedged = cumulative_return(gmvp_hedged_df['Return'])
cumulative_ftse = cumulative_return(ftse_returns)

plt.figure(figsize=(10, 6))
plt.plot(cumulative_unhedged.index, cumulative_unhedged, label='Unhedged Portfolio')
plt.plot(cumulative_hedged.index, cumulative_hedged, label='Hedged Portfolio')
plt.plot(cumulative_gmvp.index, cumulative_gmvp, label='GMVP Portfolio')
plt.plot(cumulative_gmvp_hedged.index, cumulative_gmvp_hedged, label='GMVP Hedged Portfolio')
plt.plot(cumulative_ftse.index, cumulative_ftse, label='FTSE 100 Index', linestyle='--')

plt.title('Cumulative Return of All Portfolios')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# --- Compile Performance Metrics into a Report Table ---
def collect_metrics(returns, name):
    total_return = (1 + returns).prod() - 1
    avg_return = returns.mean()
    std_dev = returns.std()
    sharpe = (avg_return - risk_free_rate) / std_dev
    dsharpe = downside_sharpe(returns, risk_free_rate)
    upr = up_ratio(returns)
    loss = return_loss(returns, ftse_returns.reindex(returns.index).dropna())
    return {
        'Portfolio': name,
        'Total Return': total_return,
        'Avg Monthly Return': avg_return,
        'Std Dev': std_dev,
        'Sharpe Ratio': sharpe,
        'Downside Sharpe': dsharpe,
        'U-P Ratio': upr,
        'Return Loss': loss
    }

metrics = [
    collect_metrics(backtest_df['Return'], "Unhedged"),
    collect_metrics(hedged_df['Return'], "Hedged"),
    collect_metrics(gmvp_df['Return'], "GMVP"),
    collect_metrics(gmvp_hedged_df['Return'], "GMVP Hedged"),
    collect_metrics(ftse_returns, "FTSE 100 Index")
]

report_df = pd.DataFrame(metrics)

# Export performance report to Excel
report_df.to_excel("performance_report.xlsx", index=False)
print("Performance report saved to 'performance_report.xlsx'")
