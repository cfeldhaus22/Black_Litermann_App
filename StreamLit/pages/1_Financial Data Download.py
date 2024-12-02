
#########   ###########  ####    ###  |  MODELO BLACK-Litterman PARA LA OPTIMIZACION DE PORTAFOLIOS.
########    ##########   ####    ###  |     
###         ###          ####    ###  |  EN ESTE CODIGO USAREMOS IMPLEMENTAREMOS EL MODELO BLACK-
###         #######      ###########  |  Litterman PARA LA OPTIMIZACION DE PORTAFOLIOS.
###         ######       ###########  |  
###         ###          ####    ###  |  ADICIONALMENTE, REALIZAREMOS OTRAS OPTIMIZACIONES,
########    ###          ####    ###  |  CALCULAREMOS METRICAS DE RIESGO DE ACTIVOS FINANCIEROS Y
#########   ###          ####    ###  |  REALIZAREMOS BACKTESTING SOBRE LOS PORTAFOLIOS OPTIMIZADOS.   

#---------------------------------------------------------------------------------------------------#
#                                      CARGA DE LIBRERIAS

import pandas as pd
import numpy as np
#from numpy import *
from numpy.linalg import multi_dot
import yfinance as yf
import scipy as stats
from scipy.stats import kurtosis, skew, norm
import matplotlib.pyplot as plt
import datetime as dt
import seaborn as sns
import streamlit as st
from fredapi import Fred
import requests
import plotly.graph_objects as go

# Configuración de estilo para gráficos
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = [12, 8]  
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

#---------------------------------------------------------------------------------------------------#
#                                             PAGE INFO

# esta pagina corresponde a la primera pagina de la aplicacion
# aqui se descargan los datos usando yahoo finance y las tasas libres de riesgo usando
# la api de la fred y de banxico, tambien se calculan las metricas de riesgo de cada activo

st.set_page_config(
    page_title="Financial Data Download",
    page_icon = "mag"
)
st.title("Financial Data Download")
st.markdown("## Data Download & Currency Conversion :currency_exchange:")

#---------------------------------------------------------------------------------------------------#
#                                      DESCARGA DE DATOS

st.markdown("The first step in this app is downloading your data. Enter the tickers to analyze, \
            ensuring they are spelled correctly and separated by commas.")

# Función para obtener datos de activos
def get_asset_data(tickers, start_date, end_date):
    temp_data = yf.download(tickers, start=start_date, end=end_date)["Close"].dropna()
    temp_data = temp_data.reset_index()
    
    # Convertir datetime a date
    temp_data['Date'] = pd.to_datetime(temp_data['Date']).dt.date
    temp_data = temp_data.set_index('Date')
    
    return temp_data

# Función para convertir a la moneda deseada
def convert_to_currency(data, start_date, end_date, target_currency="MXN"):
    # Almacenamos las tasas de cambio
    conversion_rates = {}
    
    for ticker in data.columns:
        symbol = yf.Ticker(ticker)
        currency = symbol.info.get("currency", "USD")  # Por defecto USD si no hay información
        # Verificamos si la cotización no es en la moneda objetivo
        if currency != target_currency:
            fx_pair = f"{currency}{target_currency}=X"
    
            # Descargar la tasa de cambio si no ha sido descargada ya
            if fx_pair not in conversion_rates:
                fx_data = pd.DataFrame(get_asset_data(fx_pair, start_date, end_date)).rename(
                    columns={"Close": fx_pair})
                conversion_rates[fx_pair] = fx_data  # Guardar para usos posteriores
    
            # Multiplicamos cada precio del activo por la tasa de cambio de cada día
            data[ticker] = data[ticker] * conversion_rates[fx_pair][fx_pair]
            # usamos ffill ya que en algunos dias no pudimos obtener informacion del tipo de cambio
            # de esta forma evitamos perder mas informacion
            data = data.ffill()

    return data

# guardamos las funciones en la sesion para poder usarlas mas adelante
st.session_state.get_asset_data = get_asset_data
st.session_state.convert_to_currency = convert_to_currency

# Entrada de tickers en formato libre
symbols_input = st.text_input(
    "Enter the tickers separated by commas:",
    placeholder="Example: AAPL, MSFT, GOOGL, TSLA"
    ).upper()

# Entrada para las fechas de inicio y fin, guardamos las fechas en la sesion
st.session_state.start_date = st.date_input("Enter the start date:", value = dt.date(2010, 1, 1))
st.session_state.end_date = st.date_input("Enter the end date:")

# para evitar errores, guardamos las variables en la sesion como None
if "data" not in st.session_state:
    st.session_state.data = None
if "target_currency" not in st.session_state:
    st.session_state.target_currency = "MXN"

# Selección de moneda objetivo
st.markdown("If any tickers are quoted in a different currency, the app will automatically convert them to the selected target currency.")
currency_options = ["USD", "MXN", "EUR", "JPY", "GBP", "CAD", "AUD"]
st.session_state.target_currency = st.selectbox("Select the target currency:", currency_options)

# Botón para descargar los datos y convertirlos
if st.button("Get data!"):
    if not symbols_input.strip():
        st.warning("No tickers detected. Please provide tickers separated by commas.")
    else:
        # Procesar la entrada de texto para obtener una lista de tickers
        symbols = [ticker.strip() for ticker in symbols_input.split(",") if ticker.strip()]
        
        # Validar fechas
        if st.session_state.start_date >= st.session_state.end_date:
            st.error("Invalid date range. Please ensure the start date is earlier than the end date.")
        else:
            try:
                # Descargar datos
                st.session_state.data = get_asset_data(
                    symbols,
                    start_date=st.session_state.start_date,
                    end_date=st.session_state.end_date
                )
                    
                if len(st.session_state.data) == 0:
                    st.warning("No data found for the provided tickers.")
                else:
                    # Convertir a la moneda deseada
                    st.session_state.data = convert_to_currency(
                        st.session_state.data,
                        start_date=st.session_state.start_date,
                        end_date=st.session_state.end_date,
                        target_currency=st.session_state.target_currency
                    )
                    st.success(f"Data downloaded and converted to {st.session_state.target_currency}.")
            except Exception as e:
                st.error(f"An error occurred during processing: {e}")


# Mostrar datos descargados si existen
if st.session_state.data is not None and len(st.session_state.data) > 0:
    st.dataframe(st.session_state.data.tail())

st.markdown("*Note: Yahoo Finance data downloads may occasionally be unavailable. If some data is missing, please try again later.*")

#---------------------------------------------------------------------------------------------------#
#                                GRAFICAS PRECIOS DE CIERRE

# Verifica si hay datos cargados en el estado de la aplicación
if st.session_state.data is not None and len(st.session_state.data) > 0:
    st.markdown("## Closing Price Display :chart_with_upwards_trend:")

    # Genera una paleta de colores personalizada con Seaborn
    colors = sns.color_palette("mako", n_colors=len(st.session_state.data.columns)).as_hex()
    st.session_state.colors = colors

    # Crea la figura del gráfico interactivo
    fig = go.Figure()

    # Agrega cada activo como una línea en el gráfico, asignándole un color único
    for idx, column in enumerate(st.session_state.data.columns):
        fig.add_trace(
            go.Scatter(
                x=st.session_state.data.index,
                y=st.session_state.data[column],
                mode='lines',
                name=column,
                line=dict(width=2, color=colors[idx]),  # Asigna el color correspondiente
            )
        )

    # Configura el diseño del gráfico
    fig.update_layout(
        title="",
        xaxis_title="Date",
        yaxis_title=f"Closing Price ({st.session_state.target_currency})",
        template="plotly_white",  # Tema limpio y profesional
        hovermode="x unified",  # Tooltips unificados para todas las líneas
        width=800,
        height=600,
        legend=dict(
            title="",  # Título de la leyenda
            orientation="h",  # Posición horizontal de la leyenda
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
    )

    # Renderiza el gráfico en Streamlit
    st.plotly_chart(fig, use_container_width=True)

#---------------------------------------------------------------------------------------------------#
#                                     RETORNOS DIARIOS

if "returns" not in st.session_state:
    st.session_state.returns = None

# calcular rendimientos
if st.session_state.data is not None:
    st.markdown("## Daily Returns :heavy_dollar_sign:")
    returns = st.session_state.data.copy()
    returns = returns.sort_index()
    for columna in returns.columns:
        returns[columna] = (returns[columna] - returns[columna].shift(1)) / returns[columna].shift(1)
    returns = returns.dropna()
    st.session_state.returns = returns
    st.success("Successfully calculated returns.")

# Mostrar retornos calulados si existen
if st.session_state.returns is not None:
    st.dataframe(st.session_state.returns.tail())


#---------------------------------------------------------------------------------------------------#
#                                   TASAS LIBRE DE RIESGO
    
# la siguiente funcion nos ayudara a obtener la tasa libre de riesgo del bono del tesoro de EUA
# usaremos la API de la FRED para obtener la serie de datos.
# https://fred.stlouisfed.org/docs/api/fred/
# Para obtener una key para la API se debe crear una cuenta de forma gratuita

# funcion para obtener la tasa libre de riesgo US
def get_rf_rate_us(plazo, key, start_date = dt.date(2010, 1, 1), end_date = dt.date.today(), today = False):
    fred = Fred(api_key=key)
    rf_rate_us = fred.get_series(plazo, start_date, end_date)
    rf_rate_us = rf_rate_us / 100
    if today == False:
        rf_rate_us = pd.DataFrame(rf_rate_us).rename(columns ={0: "Rate"})
        rf_rate_us.index.name = "Date"
        return rf_rate_us
    else:
        return rf_rate_us[-1]
    
# Para obtener la tasa libre de riesgo en Mexico usaremos la API de consultas de BANXICO
# esta API es gratuita y tiene un numero de consultas maximo, pero no sera un problema para 
# este estudio
# para mas informacion: https://www.banxico.org.mx/SieAPIRest/service/v1/

# funcion para obtener la tasa libre de riesgo MX
def get_rf_rate_mx(plazo, key, start_date = dt.date(2010, 1, 1), end_date = dt.date.today(), today = False):
    if today:
        # url para obtener el ultimo dato disponible
        url = f"https://www.banxico.org.mx/SieAPIRest/service/v1/series/{plazo}/datos/oportuno?token={key}"
        response = requests.get(url)
        data = response.json()
        return float(data["bmx"]["series"][0]["datos"][0]["dato"]) / 100  # Convertir a decimal
    else:
        # url para obtener la serie de datos
        url = f"https://www.banxico.org.mx/SieAPIRest/service/v1/series/{plazo}/datos/{start_date}/{end_date}?token={key}"
        response = requests.get(url)
        data = response.json()
        
        # Procesa los datos y los convierte en un DataFrame
        dates = []
        values = []
        for entry in data["bmx"]["series"][0]["datos"]:
            dates.append(entry["fecha"])
            values.append(float(entry["dato"]) / 100)  # Convertir a decimal
        
        # Crear DataFrame con las fechas y los valores
        df = pd.DataFrame({"Date": dates, "Rate": values})
        df["Date"] = pd.to_datetime(df["Date"])  # Convierte las fechas al formato datetime
        df.set_index("Date", inplace=True)
        
        return df

# guardamos las funciones para usarlas mas adelante
st.session_state.get_rf_rate_us = get_rf_rate_us
st.session_state.get_rf_rate_mx = get_rf_rate_mx

st.markdown("## Risk Free Rates :heavy_dollar_sign:")
st.text("To retrieve risk-free rates, this app uses the FRED and BANXICO APIs. These rates are essential \
        for calculating key risk metrics. You can choose between Mexican and US rates and select different \
        terms. We recommend using the 1-year risk-free rate.")

if "rf_rate_us" not in st.session_state:
    st.session_state.rf_rate_us = None
if "rf_rate_mx" not in st.session_state:
    st.session_state.rf_rate_mx = None


plazos = ["3m", "1y", "5y", "10y"]
# seleccion del plazo
plazo = st.selectbox("Select the period of the risk free rate:", plazos, index = 1)

if st.button("Get Rates!"):
    # Codigos para bonos del Tesoro en FRED
    us_treasury = {"3m":"GS3M", "1y":"GS1", "5y":"GS5", "10y":"GS10"}
    # key API FRED
    key_fred = '3f2f344c22249ae2ed4577695e869bcd'
    # descarga de datos US
    rf_rate_us = get_rf_rate_us(plazo = us_treasury[plazo], key = key_fred)
        
    us_rf_rate_today = rf_rate_us.iloc[-1].iloc[0]
    #st.success("US information downloaded!")
    #st.success(f"La tasa libre de riesgo actual en EU es: {us_rf_rate_today:.4f}")

    # datos mexico
    # key API Banxico
    key_banxico = "9c64dffdc448adeccfc4ad92a075f06524df61deeae8f2f46206e579f3b2f418"
    # codigos para bonos mexicanos
    mx_treasury = {"3m": "SF3338", "1y": "SF3367", "5y": "SF18608", "10y": "SF30057"}
    # descarga de datos
    rf_rate_mx = get_rf_rate_mx(plazo = mx_treasury[plazo], key = key_banxico)
    mx_rf_rate_today = rf_rate_mx.iloc[-1].iloc[0]
    #st.success("MX information downloaded!")
    #st.success(f"La tasa libre de riesgo actual en Mexico es: {mx_rf_rate_today:.4f}")

    # guardamos la informacion en la sesion
    st.session_state.rf_rate_us = rf_rate_us
    st.session_state.rf_rate_mx = rf_rate_mx
    # guardamos las variables para usarlas mas adelante
    st.session_state.plazo = plazo
    st.session_state.mx_treasury = mx_treasury
    st.session_state.us_treasury = us_treasury
    st.session_state.key_banxico = key_banxico
    st.session_state.key_fred = key_fred


if st.session_state.rf_rate_us is not None and st.session_state.rf_rate_mx is not None:
    st.success(f"- The current US {plazo} Treasury Rate is: {st.session_state.rf_rate_us.iloc[-1].iloc[0] * 100}%")
    st.success(f"- The current MX {plazo} Treasury Rate is: {st.session_state.rf_rate_mx.iloc[-1].iloc[0] * 100}%")

    # Colores personalizados
    color_us = st.session_state.colors[0]
    color_mx = st.session_state.colors[1] if len(st.session_state.colors) > 1 else st.session_state.colors[0]

    # Gráfico interactivo para Treasury Rate US
    fig_us = go.Figure()
    fig_us.add_trace(
        go.Scatter(
            x=st.session_state.rf_rate_us.index,
            y=st.session_state.rf_rate_us["Rate"],
            mode='lines',
            name=f'{plazo} Treasury Rate US',
            line=dict(width=2, color=color_us),
        )
    )
    fig_us.update_layout(
        title=f"{plazo} Treasury Rate US",
        xaxis_title="Date",
        yaxis_title="Rate",
        template="plotly_white",
        hovermode="x unified",
        width=800,
        height=600,
    )
    # Renderiza el gráfico
    st.plotly_chart(fig_us, use_container_width=True)

    # Gráfico interactivo para Treasury Rate MX
    fig_mx = go.Figure()
    fig_mx.add_trace(
        go.Scatter(
            x=st.session_state.rf_rate_mx.index,
            y=st.session_state.rf_rate_mx["Rate"],
            mode='lines',
            name=f'{plazo} Treasury Rate MX',
            line=dict(width=2, color=color_mx),
        )
    )
    fig_mx.update_layout(
        title=f"{plazo} Treasury Rate MX",
        xaxis_title="Date",
        yaxis_title="Rate",
        template="plotly_white",
        hovermode="x unified",
        width=800,
        height=600,
    )
    # Renderiza el gráfico
    st.plotly_chart(fig_mx, use_container_width=True)

#---------------------------------------------------------------------------------------------------#
#                            ESTADISTICAS Y METRICAS DE RIESGO

# la siguiente funcion nos ayudara a calcular metricas de riesgo relevantes para el estudio
# de los activos
def metricas(returns, rf_rate):
    """
    Esta funcion calcula estadisticas financieras para una serie de retornos
    Parametros:
        returns (pd.DataFrame): Data Frame con los retornos de los activos a considerar.
                                Cada columna representa un activo
        rf_rate (float): Tasa libre de riesgo
    Returns:
        pd.DataFrame con las estadisticas calculadas de cada activo
    """
    # definimos un diccionario con los resultados
    resultados = {
        'Mean': [],
        'Skewness': [],
        'Excess Kurtosis': [],
        'PVaR 95%': [],
        'HVaR 95%': [],
        'MCVaR 95%': [],
        'CVaR 95%': [],
        'Sharpe R': [],
        'Sortino R': [],
        'Max Drawdown': []
    }
    # consideramos la tasa libre de riesgo diaria
    rf_d = rf_rate / 252
    
    for i in returns.columns:
        mean = np.mean(returns[i])
        stdev = np.std(returns[i])

        # VaR Paramétrico
        pVaR_95 = norm.ppf(1 - 0.95, mean, stdev)

        # VaR Histórico
        hVaR_95 = returns[i].quantile(0.05)

        # VaR Monte Carlo
        n_sims = 1000  # Incrementar el número de simulaciones para mayor precisión
        sim_returns = np.random.normal(mean, stdev, (n_sims, len(returns[i])))
        MCVaR_95 = np.percentile(sim_returns, 5)

        # CVaR
        CVaR_95 = returns[i][returns[i] <= hVaR_95].mean()

        # Sharpe Ratio
        sharpe_ratio = (mean - rf_d) / stdev

        # Sortino Ratio
        neg_returns = returns[i][returns[i] < rf_d]
        sigma_dp = neg_returns.std()
        sortino_ratio = (mean - rf_d) / sigma_dp

        # Max Drawdown
        cumulative_returns = (1 + returns[i]).cumprod()
        running_max = cumulative_returns.cummax()
        drawdowns = (cumulative_returns - running_max) / running_max
        MDD = drawdowns.min()

        # Añadir los resultados al diccionario
        resultados['Mean'].append(np.round(np.mean(returns[i]), 6))
        resultados['Skewness'].append(np.round(skew(returns[i]), 6))
        resultados['Excess Kurtosis'].append(np.round(kurtosis(returns[i], fisher=True), 6))
        resultados['PVaR 95%'].append(np.round(pVaR_95 , 6))
        resultados['HVaR 95%'].append(np.round(hVaR_95 , 6))
        resultados['MCVaR 95%'].append(np.round(MCVaR_95 , 6))
        resultados['CVaR 95%'].append(np.round(CVaR_95 , 6))
        resultados['Sharpe R'].append(np.round(sharpe_ratio, 6))
        resultados['Sortino R'].append(np.round(sortino_ratio, 6))
        resultados['Max Drawdown'].append(np.round(MDD, 6))
    
    # Crear un DataFrame con los resultados
    estadisticas_df = pd.DataFrame(resultados, index=returns.columns)

    return estadisticas_df

# guardamos la funcion en la sesion
st.session_state.metricas = metricas

st.markdown("## Stats & Risk Metrics :straight_ruler:")

# Seleccionar tasa libre de riesgo
country = ["United States", "Mexico"]
# seleccion del plazo
rf_use = st.selectbox("Select the country which Treasury Rate you’d like to use:", country)

# guardamos la variable
st.session_state.rf_use = rf_use

if "metricas_returns" not in st.session_state:
    st.session_state.metricas_returns = None

if st.button("Get Measures!"):
    if st.session_state.returns is not None and (st.session_state.rf_rate_us is not None or st.session_state.rf_rate_mx is not None):
        if rf_use == "United States":
            metricas_returns = metricas(st.session_state.returns, rf_rate = st.session_state.rf_rate_us.iloc[-1].iloc[0])
            st.session_state.metricas_returns = metricas_returns
            st.success("Successfully calculated stats and risk metrics!")
        elif rf_use == "Mexico":
            metricas_returns = metricas(st.session_state.returns, rf_rate = st.session_state.rf_rate_mx.iloc[-1].iloc[0])
            st.session_state.metricas_returns = metricas_returns
            st.success("Successfully calculated stats and risk metrics!")
    else:
        st.warning("You need to download your data and the Treasury Rates first...")
        

if st.session_state.metricas_returns is not None:
    st.dataframe(st.session_state.metricas_returns)
    st.text("All necessary information has been gathered to optimize the portfolio. Proceed to the next page to continue.")

    # Botón para ir a la siguiente página
    col1, col2 = st.columns([1,0.2])
    with col2:
        if st.button("Optimize!"):
                st.switch_page("pages/2_Portfolio Optimization.py")

