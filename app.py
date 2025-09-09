# app.py

import streamlit as st
from datetime import datetime
from var_analysis import run_var_analysis
import ast

st.set_page_config(page_title="VaR Analyzer", layout="wide")

st.title("Portfolio Value at Risk (VaR) Analyzer")

st.sidebar.header("Portfolio Configuration")

# 1. Tickers
tickers_input = st.sidebar.text_input("Enter Tickers (comma-separated)", value="AAPL, MSFT, GOOGL, AMZN")
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

# 2. Weights
weights_input = st.sidebar.text_input("Enter Weights (comma-separated)", value="0.25, 0.25, 0.25, 0.25")

try:
    weights = [float(w.strip()) for w in weights_input.split(",") if w.strip()]
except ValueError:
    weights = []

# Validation Message
if len(weights) != len(tickers):
    st.sidebar.error("⚠️ Number of weights must match number of tickers.")
elif not abs(sum(weights) - 1.0) < 0.01:
    st.sidebar.warning("⚠️ Weights should sum to 1. They will be normalized automatically.")

# 3. Date Range
start_date = st.sidebar.date_input("Start Date", value=datetime(2020, 1, 1))
end_date = st.sidebar.date_input("End Date", value=datetime(2023, 1, 1))

if start_date >= end_date:
    st.sidebar.error("Start date must be before end date.")

# 4. Confidence Level
confidence_level = st.sidebar.slider("Confidence Level", 0.90, 0.999, 0.95, step=0.01)

# 5. Monte Carlo Simulations
num_simulations = st.sidebar.number_input("Number of Monte Carlo Simulations", min_value=1000, max_value=50000, value=10000, step=1000)

# Run Button
run_analysis = st.sidebar.button("🚀 Run Analysis")

# Footer Info
st.sidebar.markdown("---")
st.sidebar.info("ℹ️ Make sure the number of weights matches the number of tickers and they sum up to 1.")

if run_analysis:
    if len(tickers) == len(weights) and start_date < end_date:
        try:
            results = run_var_analysis(
                tickers=tickers,
                weights=weights,
                start_date=start_date,
                end_date=end_date,
                confidence_level=confidence_level,
                num_simulations=num_simulations
            )

            st.subheader("Portfolio Metrics")
            col1, col2, col3 = st.columns(3)

            col1.metric("Mean Return (Daily)", f"{results['metrics']['mean']*100:.4f}%")
            col2.metric("Volatility (Std Dev)", f"{results['metrics']['std']*100:.4f}%")
            col3.metric(f"{int(confidence_level*100)}% Historical VaR", f"{results['VaR']['historical']*100:.4f}%", delta="-", delta_color="inverse")

            col4, col5 = st.columns(2)
            col4.metric("Parametric VaR", f"{results['VaR']['parametric']*100:.4f}%")
            col5.metric("Monte Carlo VaR", f"{results['VaR']['monte_carlo']*100:.4f}%")

            st.markdown("---")
            st.subheader("Visualizations")
            st.pyplot(results['fig'])

            st.markdown("---")
            st.subheader("Download Portfolio Returns")

            csv_data = results['metrics']['returns'].to_frame(name='Daily Return')
            csv = csv_data.to_csv(index=True).encode("utf-8")
            st.download_button("Download Returns CSV", data=csv, file_name="portfolio_returns.csv", mime="text/csv")

        except Exception as e:
            st.error(f"Error running analysis: {str(e)}")

    else:
        st.warning("Please fix the issues in the sidebar and try again.")
