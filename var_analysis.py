import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style='whitegrid')

def fetch_data(tickers, start_date, end_date):
    df = yf.download(tickers, start=start_date, end=end_date)['Close']
    log_returns = np.log(df / df.shift(1)).dropna()
    return log_returns

def cal_portfolio_metrics(log_returns, weights):
    weights = np.array(weights)
    if not np.isclose(np.sum(weights), 1.0):
        weights /= np.sum(weights)

    weights_series = pd.Series(weights, index=log_returns.columns)
    portfolio_returns = log_returns.dot(weights_series)

    portfolio_mean = portfolio_returns.mean()
    portfolio_std = portfolio_returns.std()

    return {
        'mean': portfolio_mean,
        'std': portfolio_std,
        'returns': portfolio_returns
    }

def calc_historical_VaR(portfolio_returns, confidence_level=0.05):
    return -np.percentile(portfolio_returns, (1 - confidence_level) * 100)

def calc_parametric_VaR(portfolio_mean, portfolio_std, confidence_level=0.05):
    ZScore = -norm.ppf(1 - confidence_level)
    return portfolio_mean + ZScore * portfolio_std

def monte_carlo_var(portfolio_returns, confidence_level, num_simulations=10000, forecast_horizon=1):
    mu = portfolio_returns.mean()
    sigma = portfolio_returns.std()
    initial_portfolio_value = 1.0

    random_returns = np.random.normal(mu, sigma, (forecast_horizon, num_simulations))
    simulated_paths = np.zeros((forecast_horizon + 1, num_simulations))
    simulated_paths[0] = initial_portfolio_value

    for t in range(1, forecast_horizon + 1):
        simulated_paths[t] = simulated_paths[t - 1] * np.exp(random_returns[t - 1])

    final_values = simulated_paths[-1, :]
    var_percentile = (1 - confidence_level) * 100
    var_value = np.percentile(final_values, var_percentile)
    monte_carlo_var = initial_portfolio_value - var_value

    return {
        'monte_carlo_var': monte_carlo_var,
        'final_values': final_values
    }

def visualize_var(metrics, historical_var, parametric_var, monte_carlo_var, final_simulated_values, confidence_level):
    returns = metrics['returns']
    mean = metrics['mean']
    std = metrics['std']

    fig, axes = plt.subplots(3, 2, figsize=(15, 18))
    
    returns.cumsum().plot(ax=axes[0, 0])
    axes[0, 0].set_title("Portfolio Cumulative Log Returns")
    axes[0, 0].set_xlabel("Date")
    axes[0, 0].set_ylabel("Cumulative Return")

    sns.histplot(returns, bins=50, kde=True, color='skyblue', ax=axes[0, 1])
    axes[0, 1].axvline(-historical_var, color='red', linestyle='--',
                       label=f"Historical VaR ({confidence_level*100:.0f}%)")
    axes[0, 1].set_title("Distribution of Daily Returns")
    axes[0, 1].set_xlabel("Daily Return")
    axes[0, 1].legend()

    x = np.linspace(mean - 4 * std, mean + 4 * std, 1000)
    y = norm.pdf(x, mean, std)
    axes[1, 0].plot(x, y, label='Normal Distribution')
    axes[1, 0].axvline(parametric_var, color='orange', linestyle='--',
                       label=f"Parametric VaR ({confidence_level*100:.0f}%)")
    axes[1, 0].set_title("Normal Distribution with Parametric VaR")
    axes[1, 0].set_xlabel("Daily Return")
    axes[1, 0].legend()

    sns.histplot(final_simulated_values, bins=50, color='lightgreen', ax=axes[1, 1])
    axes[1, 1].axvline(1 - monte_carlo_var, color='purple', linestyle='--', label='Monte Carlo VaR Cutoff')
    axes[1, 1].set_title("Monte Carlo Simulation of Final Portfolio Value")
    axes[1, 1].set_xlabel("Final Value")
    axes[1, 1].legend()

    returns.rolling(window=30).std().plot(ax=axes[2, 0])
    axes[2, 0].set_title("30-Day Rolling Volatility")
    axes[2, 0].set_xlabel("Date")
    axes[2, 0].set_ylabel("Volatility")

    fig.tight_layout()
    return fig

def run_var_analysis(tickers, weights, start_date, end_date, confidence_level=0.95, num_simulations=10000, forecast_horizon=1):
    log_returns = fetch_data(tickers, start_date, end_date)
    metrics = cal_portfolio_metrics(log_returns, weights)

    historical_VaR = calc_historical_VaR(metrics['returns'], confidence_level)
    parametric_VaR = calc_parametric_VaR(metrics['mean'], metrics['std'], confidence_level)
    monte_carlo = monte_carlo_var(metrics['returns'], confidence_level, num_simulations, forecast_horizon)

    fig = visualize_var(
        metrics,
        historical_VaR,
        parametric_VaR,
        monte_carlo['monte_carlo_var'],
        monte_carlo['final_values'],
        confidence_level
    )

    return {
        'log_returns': log_returns,
        'metrics': metrics,
        'VaR': {
            'historical': historical_VaR,
            'parametric': parametric_VaR,
            'monte_carlo': monte_carlo['monte_carlo_var']
        },
        'fig': fig
    }
