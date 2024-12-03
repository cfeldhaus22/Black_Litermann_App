# Financial Portfolio Analysis and Optimization with Python

This project is a Python application designed for downloading, analyzing, and optimizing financial asset data. It enables users to convert assets to a specific currency, calculate risk and return metrics, optimize portfolios, and perform comparative backtesting against the S&P 500 index. Additionally, it includes the implementation of the Black-Litterman model to customize portfolio optimization based on financial "views."

This repository was created to publish this project on Streamlit, you can use the app yourself here: https://blacklitermannapp.streamlit.app/

## Project Features

### 1. Financial Data Download
- **Data Source**: Downloads historical financial asset data using the Yahoo Finance API.
- **Currency Conversion**: Converts assets to a specific currency (e.g., USD to MXN) for consistent comparisons across assets.

### 2. Price Visualization
- **Closing Price Charts**: Visualizes historical prices for each asset.
- **Daily Returns**: Calculates and displays daily returns to analyze volatility.

### 3. Risk-Free Rates
- **Interest Rate Sources**: Integrates Banxico and FRED APIs to obtain risk-free rates.

### 4. Risk and Return Metrics Calculation
- **Value at Risk (VaR)**: Estimates the maximum risk over a given period under normal conditions.
- **Excess Kurtosis**: Analyzes the risk of extreme events in the assets.
- **Sortino Ratio**: A risk-adjusted performance metric focusing on negative returns.
- **Sharpe Ratio**: Risk-return ratio adjusted for the risk-free rate.

### 5. Portfolio Optimization
Optimization strategies include:
- **Minimum Volatility**: A portfolio with the lowest volatility.
- **Maximum Sharpe Ratio**: A portfolio with the best risk-return ratio.
- **Minimum Volatility with Target Return**: A portfolio with minimum volatility for a specific return target.

### 6. Portfolio Backtesting
- **Comparison with S&P 500**: Backtesting evaluates the effectiveness of generated strategies by comparing them against the S&P 500 index performance.

### 7. Black-Litterman Model
- **Custom Financial Views**: Adjusts portfolio optimization based on expected returns for each asset using the Black-Litterman model.

## Technologies and Libraries

- **Python**: Primary programming language.
- **Yahoo Finance API**: Market data downloader.
- **Banxico and FRED APIs**: Risk-free interest rate data.
- **Pandas**: Data manipulation.
- **NumPy**: Numerical operations.
- **Matplotlib and Seaborn**: Financial data visualization.
- **SciPy**: Optimization tools.
- **Streamlit**: Interactive app development framework.

## Contribution

Contributions are welcome! If you have ideas or improvements for this project, feel free to fork this repository and open a Pull Request with your changes.

## License

This project is for public and educational use. The data used is publicly accessible.

---

For questions or comments about the analysis or code, feel free to reach out!
