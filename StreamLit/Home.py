
#########   ###########  ####    ###  |  MODELO BLACK-Litterman PARA LA OPTIMIZACION DE PORTAFOLIOS.
########    ##########   ####    ###  |     
###         ###          ####    ###  |  EN ESTE CODIGO USAREMOS IMPLEMENTAREMOS EL MODELO BLACK-
###         #######      ###########  |  Litterman PARA LA OPTIMIZACION DE PORTAFOLIOS.
###         ######       ###########  |  
###         ###          ####    ###  |  ADICIONALMENTE, REALIZAREMOS OTRAS OPTIMIZACIONES,
########    ###          ####    ###  |  CALCULAREMOS METRICAS DE RIESGO DE ACTIVOS FINANCIEROS Y
#########   ###          ####    ###  |  REALIZAREMOS BACKTESTING SOBRE LOS PORTAFOLIOS OPTIMIZADOS.   

#---------------------------------------------------------------------------------------------------#

# Esta pagina corresponde a la pagina de inicio del proyecto en StreamLit. Contiene
# la descripcion de cada página y provee informacion adicional.

# configuracion de la pagina
import streamlit as st
st.set_page_config(
    page_title="Black-Litterman Project",
    page_icon = "mag"
)

texto = '''
# :moneybag: :bar_chart: Analysis and Optimization of Financial Portfolios with Python

This project is a Python application for downloading, analyzing, and optimizing financial asset data. 
It allows transforming assets into a specific currency, calculating risk and return metrics, optimizing 
portfolios, and performing comparative backtesting against the S&P 500 index. Additionally, it includes
the implementation of the Black-Litterman model to customize optimizations based on financial "views".

## :bulb: Project Features

### 1.1. Financial Data Download :open_file_folder:
- **Data Source**: Downloads historical financial asset data using the Yahoo Finance API.
- **Currency Conversion**: Converts assets to a specific currency (e.g., USD to MXN) for 
consistent comparisons across assets.

### 1.2. Price Visualization :chart_with_upwards_trend:
- **Closing Price Charts**: Visualization of historical prices for each asset.
- **Daily Returns**: Calculation of daily returns to analyze volatility.

### 1.3. Risk-Free Rates :heavy_dollar_sign:
- **Interest Rate Sources**: Integrates Banxico and FRED APIs to obtain risk-free rates.

### 1.4. Risk and Return Metrics Calculation :straight_ruler:
- **Value at Risk (VaR)**: Estimates the maximum risk over a given period under normal conditions.
- **Excess Kurtosis**: Analyzes the risk of extreme events in the assets.
- **Sortino Ratio**: Risk-adjusted performance metric focusing on negative returns.
- **Sharpe Ratio**: Risk-return ratio adjusted for the risk-free rate.

### 2.1. Portfolio Optimization : :muscle: 
Optimization based on:
- **Minimum Volatility**: Portfolio with the lowest volatility.
- **Maximum Sharpe Ratio**: Portfolio with the best risk-return ratio.
- **Minimum Volatility with Target Return**: Portfolio with minimum volatility for a specific 
return level.

### 2.2. Portfolio Backtesting  :thinking_face:
- **Comparison with S&P 500**: Backtesting evaluates the effectiveness of generated strategies
by comparing them against the S&P 500 index performance.

### 3.1 Black-Litterman Model :exploding_head:
- **Custom Financial Views**: Adjusts portfolio optimization based on expected returns for each
asset using the Black-Litterman model. For more details, see this 
[Investopedia article](https://www.investopedia.com/terms/b/black-litterman_model.asp).




## Technologies and Libraries :books:

- **Python**: Main programming language.
- **Yahoo Finance API**: Market data download.
- **Banxico and FRED APIs**: Risk-free interest rates.
- **Pandas**: Data manipulation.
- **NumPy**: Numerical operations.
- **Matplotlib and Seaborn**: Financial data visualization.
- **Scipy**: Optimization tools.
- **Streamlit**: App development tool. :streamlit:

## Contribution :raised_hand:

If you have ideas or improvements for this project, they are welcome! Feel free to look at the raw
code in my [GitHub repository](https://www.github.com/cfeldhaus22/Black_Litermann_App) and open a Pull
Request with your changes. 

You can also contact me via E-mail in: c.feldha@gmail.com.

## License :heavy_check_mark:

This project is for public and educational use. The data used is publicly accessible.

---

For any questions or comments about the analysis or code, don’t hesitate to contact me!
'''

st.markdown(texto)

col1, col2 = st.columns([1,0.2])

# podria agregar links a cada seccion de la pagina
with col2:
    if st.button("Start!"):
        st.switch_page("pages/1_Financial Data Download.py")

