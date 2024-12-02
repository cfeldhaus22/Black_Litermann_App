
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
import scipy as stats
from scipy.stats import kurtosis, skew, norm
import matplotlib.pyplot as plt
import datetime as dt
import seaborn as sns
import streamlit as st

#---------------------------------------------------------------------------------------------------#
#                                             PAGE INFO

# este archivo corresponde a la tercera pagina de la aplicacion de streamlit
# aqui se desarrolla el modelo black literman

st.set_page_config(
    page_title="Black-Litterman Model",
    page_icon = "mag"
)
st.title("Black-Litterman Model")

#---------------------------------------------------------------------------------------------------#

# verificamos que tengamos disponible la informacion necesaria para optimizar los portafolios y 
# descripcion breve del modelo

st.markdown("### About the Black-Litterman Model")
st.markdown("""
The **Black-Litterman model** is a mathematical framework used for portfolio optimization that incorporates subjective views on expected returns. 
It combines prior market equilibrium returns (based on a benchmark) with investor-specific views to produce a posterior distribution of returns.

The model's key components are:
- **Equilibrium Excess Returns (π):** Calculated using the covariance matrix of excess returns and market weights.
- **Investor Views (P):** Matrix representing the views (absolute or relative) on selected assets.
- **Expected Returns on Views (Q):** Vector containing expected returns for the views.
- **Confidence in Views (Ω):** Diagonal matrix representing the investor's confidence in each view.
""")

st.markdown("""
**Key Benefits:**
- Integrates subjective views into a market equilibrium framework.
- Adjusts for the confidence level of each view.
- Outputs optimized portfolio weights with varying risk aversion levels (λ).

---

""")

required = ["data", "returns", "plazo", "rf_use", "mx_treasury", "us_treasury", "key_banxico", "key_fred"]
for i in required:
    if i in st.session_state:
        continue
    else:
        st.warning("No data available. Please return to the main page to load the required data.")
        st.stop()

st.success("Session data loaded successfully.")

#---------------------------------------------------------------------------------------------------#
#                                      MODELO BLACK-Litterman

# Para el siguiente punto debemos obtener las tasas libre de riesgo vigentes al inicio
# de cada anio sobre el que tenemos informacion
# Usaremos las tasas anuales vigentes al inicio de cada periodo

rates = []
if st.session_state.plazo != "1y":
    if st.session_state.rf_use == "Mexico":
        temp_rate = st.session_state.get_rf_rate_mx(st.session_state.mx_treasury["1y"], st.session_state.key_banxico,
                                                     st.session_state.start_date, st.session_state.end_date)
        # nos quedamos con las tasas vigentes el 01 de enero
        for y in range(st.session_state.start_date.year, st.session_state.end_date.year + 1):
            rates.append(temp_rate.loc[f"{y}-01-01"])
    elif st.session_state.rf_use == "United States":
        temp_rate = st.session_state.get_rf_rate_us(st.session_state.us_treasury["1y"], st.session_state.key_fred, 
                                                    st.session_state.start_date, st.session_state.end_date)
        # nos quedamos con las tasas vigentes el 01 de enero
        for y in range(st.session_state.start_date.year, st.session_state.end_date.year + 1):
            rates.append(temp_rate.loc[f"{y}-01-01"])
else:
    if st.session_state.rf_use == "Mexico":
        for y in range(st.session_state.start_date.year, st.session_state.end_date.year + 1):
            rates.append(st.session_state.rf_rate_mx.loc[f"{y}-01-01"])
    elif st.session_state.rf_use == "United States":
        for y in range(st.session_state.start_date.year, st.session_state.end_date.year + 1):
            rates.append(st.session_state.rf_rate_us.loc[f"{y}-01-01"])

# transformamos la lista a un DataFrame
rf_rates = pd.DataFrame(rates)
rf_rates = rf_rates.set_index(rf_rates.index.year)
rf_rates.index.name = "Year"

st.markdown("""
The **Treasury Rate** used in this analysis corresponds to the **1-year yield** from your selected region. 
These rates are utilized to calculate the excess returns of assets and inform the optimization process.
""")


fig, ax = plt.subplots(figsize=(12, 8))
# Crear el gráfico de barras
bars = ax.bar(
    rf_rates.index, 
    rf_rates["Rate"]*100, 
    linewidth=2, 
    color=st.session_state.colors[0]
)
# Agregar valores encima de las barras
for bar in bars:
    height = bar.get_height()  # Altura de cada barra
    if height > 0:  # Solo mostrar valores si son mayores a cero
        ax.text(
            bar.get_x() + bar.get_width() / 2,  # Coordenada x en el centro de la barra
            height + 0.001,  # Coordenada y un poco encima de la barra
            f"{height:.2f}%",  # Formatear el valor
            ha='center',  # Centrar el texto horizontalmente
            va='bottom',  # Alinear el texto desde la parte inferior
            fontsize=11  # Tamaño de la fuente del texto
        )

# Títulos y etiquetas
ax.set_title(
    f"1 Year Treasury Rate {st.session_state.rf_use}", 
    fontsize=20, 
    pad=15
)
ax.set_ylabel("Rate (%)", fontsize=14)

# Etiquetas del eje x
ax.set_xticks(rf_rates.index)  # Asegurarse de que todas las etiquetas estén presentes
ax.set_xticklabels(rf_rates.index, rotation=45, ha='right', fontsize=12)
ax.set_yticklabels([])

# Estilización de los bordes
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Estilización de la cuadrícula
ax.grid(axis = "y", linestyle='--', alpha=0.6)
ax.grid(axis = "x", linestyle='--', alpha=0)

# Mostrar el gráfico en Streamlit
st.pyplot(fig)  
plt.close(fig)  


# ahora vamos a obtener los retornos acumulados anuales de cada activo
annual_returns = st.session_state.returns.copy()
# modificamos el indice a datetime
annual_returns.index = pd.to_datetime(annual_returns.index)
# obtenemos los retornos anuales
annual_returns = annual_returns.resample('YE').apply(lambda x: (1 + x).prod() - 1)
# modificamos el indice para conservar solo los anios
annual_returns = annual_returns.set_index(annual_returns.index.year)

# conservamos las tasas libres de riesgo correspondientes
rf_rates = rf_rates.loc[annual_returns.index]

# ahora obtenemos el exceso de retorno
excess_returns = annual_returns.subtract(rf_rates.iloc[:, 0], axis=0)

st.text("With the data we have, we'll calculate the following:")
st.subheader("Covariance Matrix of the excess annual returns:")
# obtenemos la matriz de varianzas y covarianzas de los excesos de retorno
cov_matrix = excess_returns.cov()

col1, col2 = st.columns([1, 6])
with col1:
    st.latex(r'''\Sigma = ''')
with col2:
    st.dataframe(cov_matrix)

# para calcular la distribucion a priori del modelo, asumimos un benchmark constituido por
# los activos seleccionados con un peso equitativo
ew_pesos =  len(cov_matrix) * [1./len(cov_matrix)]
ew_pesos = np.array(ew_pesos)[:,np.newaxis]

st.subheader("Initial Benchmark:")
st.text("To calculate the priori distribution of the model, we assume a benchmark consisting of \
        the selected assets with equal weights.")
col1, col2 = st.columns([1, 6])
with col1:
    st.latex(r'''w_M = ''')
with col2:
    st.dataframe(pd.DataFrame(ew_pesos, index = cov_matrix.index, columns=["w"]))

st.text('We can compute the equilibruim vector of excess return (the prior). \
        First, we need to estimate the risk aversion.')

st.subheader("Standard Deviation of the market:")

# calculamos la desviacion estandar
st.text("The standard deviation of the market is:")
desv_est_bl = np.sqrt(ew_pesos.T @ cov_matrix @ ew_pesos)
st.latex(rf'''
\sigma_M = \sqrt{{w_M^{{'}} \Sigma w_M}} = {np.round(desv_est_bl.iloc[0,0], 6)}
''')


st.subheader("Risk Aversion:")
# definimos Lambda
Lambda = (1/desv_est_bl)*0.5
#print(Lambda)
st.text("Taking a Sharpe Ratio of 0.5 (same as Black and Litterman), we obtain:")
st.latex(
    rf'''
    \lambda = \frac{{1}}{{\sigma_M}} S_M = \frac{{1}}{{{np.round(desv_est_bl.iloc[0,0], 6)}}} \times 0.5 = {{{np.round(Lambda.iloc[0,0], 6)}}}
    '''
)

# distribucion a priori
vec_ec_bl = (cov_matrix @ ew_pesos) @ Lambda
#print(vec_ec_bl)
st.subheader("Equilibrium Vector:")
col1, col2 = st.columns([1, 3])
with col1:
    st.latex(r'''\Pi = \lambda \Sigma w_M = ''')
with col2:
    st.dataframe(vec_ec_bl)

st.subheader("Pior Variance:")
st.text("To compute the variance of the prior, we compute 'tau' based on the standard error \
        of estimate method.")
# definimos Tau: 1/(numero de periodos (anios))
Tau = 1/annual_returns.shape[0]
#print(Tau)
st.latex(rf'''
    \tau = \frac{{1}}{{T}} = \frac{{1}}{{{annual_returns.shape[0]}}} = {{{np.round(Tau, 6)}}}
    ''')

st.text("The prior distribution is:")
st.latex(r'''P(E) \sim N(\Pi, \tau \Sigma) ''')
st.text("with:")
# varianza a priori
var_priori = Tau * cov_matrix
#print(var_priori)
col1, col2 = st.columns([1, 2])
with col1:
    st.latex(r'''\tau \Sigma = ''')
with col2:
    st.dataframe(var_priori)

#---------------------------------------------------------------------------------------------------#
#                                           VIEWS

# Ahora introduciremos nuestras views abosolutas y relativas sobre cada activo:
# Ej. view absoluta: El activo 3 tendra un rendimiento de 15%
# Ej. view relativa: El activo 2 tendra un rendimiento 10% al rendimiento del activo 1

# debemos definir 3 matrices:
#   1. Views
#   2. Retorno esperado sobre las views
#   3. Confianza sobre la View

st.subheader("Financial Views:")
st.markdown("Now we can introduce our financial views in the model. First, we need our **view matrix**.")
st.markdown("**E.g:** Suposse you have **4 assets** and **3 views**. Your views can be absolute \
            or relative, your views matrix should look like this:")

col1, col2 = st.columns([1, 2])
with col1:
    st.latex(r"P_{{4 \times 3}} =")
with col2:
    st.dataframe(pd.DataFrame(np.array([[0,0,0,1],[1,-1,0,0],[0,0,1,0]]),
                              index = ["View 1", "View 2", "View 3"],
                              columns = ["Asset 1", "Asset 2", "Asset 3", "Asset 4"]))

st.markdown('''**Interpretation:**''')  
st.markdown(''' 
            *1.* The fisrt view is **absolute** on the 4th asset.

            *2.* The second view is **relative** on the 1st asset regarding the 2nd asset.

            *3.* The third view is **absolute** on the 3rd aseet. 
           ''')

st.markdown("""           
            ---
            Then, we need the vector of **expected returns** on each view (Q).
            """)

col1, col2 = st.columns([1, 2])
with col1:
    st.latex(r"Q_{{1 \times 3}} =")
with col2:
    st.dataframe(pd.DataFrame(np.array([[20],[10],[16]]),
                              index = ["View 1", "View 2", "View 3"],
                              columns = ["Expected Return"]))

st.markdown('''**Interpretation:**''')  
st.markdown(''' 
            *1.* We expect the 4th asset to have a 20% return.

            *2.* We expect the 1st asset to have returns **10% greater than** the 2nd asset.

            *3.* We expect the 3rd asset to have a 16% return. 
           ''')

st.markdown("**Note:** The app will automatically turn the percentages to decimal numbers.")

st.markdown("""
            ---
            The last matrix we need is the **confidence matrix** on each view.
            """)

col1, col2 = st.columns([1, 2])
with col1:
    st.latex(r"\Omega_{{1 \times 3}} =")
with col2:
    st.dataframe(pd.DataFrame(np.array([50, 60, 70]),
                              index = ["View 1", "View 2", "View 3"],
                              columns = ["Confidence"]))

st.markdown('''**Interpretation:**''')  
st.markdown(''' 
            *1.* We have 50% confidence on the first view.

            *2.* We have 60% confidence on the second view.

            *3.* We have 70% confidence on the thrid view. 
           ''')

st.markdown("**Note:** The app will automatically turn the percentages to decimal numbers. \
            This vector will be a diagonaliced matrix to perform calculations.")

st.markdown("---")
st.subheader("Now, introduce your information!")

#--------------------- USER DATA

views_col = cov_matrix.columns
num_views = st.number_input("Select the number of views:", min_value=1, step=1)

st.write("Input your views (allowed values: -1, 0, 1):")
# creamos el data frame inicial con ceros
index_views = []
for i in range(num_views):
    name = f"View {i+1}"
    index_views.append(name)

views_editable = pd.DataFrame(0, index = index_views, columns = views_col)
edited_views = st.data_editor(views_editable)

st.write("Input your expected returns (0-100% format):")
returns_editable = pd.DataFrame(0, index = index_views, columns=["Expected Return"])
edited_returns = st.data_editor(returns_editable)

st.write("Input your confidence on each view (0-100% format):")
conf_editable = pd.DataFrame(0, index = index_views, columns=["Confidence"])
edited_conf = st.data_editor(conf_editable)

if "portafolios_bl" not in st.session_state:
    st.session_state.portafolios_bl = None

if st.button("Continue"):
    # Validar que solo se ingresen valores válidos
    invalid_values = ((edited_views != 0) & (edited_views != 1) & (edited_views != -1)).any().any()
    invalid_exp_r = ((edited_returns <= 0) | (edited_returns > 100)).any().any()
    invalid_confidence = ((edited_conf <= 0) | (edited_conf > 100)).any().any()

    if invalid_values:
        st.error("Invalid values detected in the Views dataframe! Only -1, 0, and 1 are allowed.")
        st.stop()
    if invalid_exp_r:
        st.error("Invalid values detected in the Expected Returns dataframe! Values must be between 0 and 100 (exclusive).")
        st.stop()
    if invalid_confidence:
        st.error("Invalid values detected in the Confidence dataframe! Values must be between 0 and 100 (exclusive).")
        st.stop()
    else:
        try:
            P = np.array(edited_views)
            Q = np.array(edited_returns/100)
            O = np.diag(edited_conf/100)
            st.success("Information loaded successfully!")
        except Exception as e:
            st.error(f"An error occurred while processing: {e}")
            st.stop()

    # definimos una matriz auxiliar definida como views @ var_priori @ views.T 
    aux1 = np.array(P @ var_priori @ P.T)

    # obtenemos la matriz diagonalizada
    O2 = np.diag(np.diag(aux1))

    # obtenemos la esperanza del exceso de retorno
    E = np.linalg.inv(np.linalg.inv(Tau*cov_matrix) + P.T@(np.linalg.inv(O2)@P)) \
        @ (np.linalg.inv(Tau*cov_matrix)@vec_ec_bl + P.T@np.linalg.inv(O2)@Q)

    # varianza de los estimados
    varianza_E = np.linalg.inv(np.linalg.inv(Tau*cov_matrix)+ P.T@(np.linalg.inv(O2)@P))
    

    ## Portafolio Optimizado Black-Litterman

    # Lambda representa el caso base de adversion al riesgo al obtener la distribucion
    # a priori. Considerando otros valores de Lambda, podemos obtener los pesos del 
    # portafolio optimizado condistintos niveles de riesgo.
    Lambda1 = Lambda.iloc[0,0]
    # Entre menor sea el valor de Lambda, se toma mas riesgo.
    # Un valor mayor representa mas ADVERSION al riesgo.
    list_lambda = np.arange(1, 7.5, 0.5).tolist()
    list_lambda.insert(0, Lambda1.round(4))

    portafolios_bl = pd.DataFrame()
    # observemos los pesos del portafolio optimizado con distintos valores de riesgo
    for i in list_lambda:
        weights_bl = (np.linalg.inv(cov_matrix * i)) @ E
        portafolios_bl = pd.concat([portafolios_bl, pd.DataFrame(weights_bl).T.rename(index={0:i})])

    portafolios_bl.index.name = "Lambda"
    portafolios_bl.columns = cov_matrix.columns
    portafolios_bl['Total'] = portafolios_bl.sum(axis=1)
    portafolios_bl = portafolios_bl.sort_index()
    st.session_state.portafolios_bl = portafolios_bl

if st.session_state.portafolios_bl is not None:
    st.markdown("""
    ### Optimized Portfolio with Varying Risk Aversion Levels (λ)
    Below are the portfolio weights for different risk aversion levels. 
    You can observe how the allocation changes based on the selected λ value.
    """)

    st.dataframe(st.session_state.portafolios_bl)

    # Resultados de Lambda y total del portafolio
    st.markdown("## Understanding the Results")
    st.markdown("""
    The **Lambda (λ)** parameter represents the investor's level of **risk aversion**:
    - A **low λ value** indicates lower risk aversion and a higher willingness to take risks.
    - A **high λ value** indicates higher risk aversion and a preference for safer investments.

    The **Total Portfolio** value provides insights into leverage:
    - If **Total > 1**, it implies borrowing funds at the risk-free rate to invest more in the portfolio.
    - If **Total < 1**, it indicates a portion of the investment is allocated to the risk-free rate.
    """)

    # Descripción visual adicional del portafolio optimizado
    st.markdown("""
    ### Visualizing Portfolio Allocation
    The following chart shows the portfolio allocation across assets for each λ value. This helps understand how risk preferences influence investment decisions.
    """)

    fig, ax = plt.subplots(figsize=(12, 8))

    # Preparar los datos para Seaborn
    portfolio_data = st.session_state.portafolios_bl.drop(columns="Total").T
    portfolio_data = portfolio_data.reset_index().melt(id_vars='Ticker', var_name='Lambda', value_name='Allocation')
    portfolio_data.rename(columns={'Ticker': 'Assets'}, inplace=True)

    # Gráfico de barras apiladas con Seaborn
    sns.barplot(
        data=portfolio_data,
        x='Assets',
        y='Allocation',
        hue='Lambda',
        palette="mako",
        errorbar=None,  # Deshabilitar barras de error
        ax=ax
    )

    # Personalizar el gráfico
    ax.set_title("Asset Allocation by Risk Aversion Level (λ)", fontsize=17, pad=15)
    ax.set_xlabel("", fontsize=14, labelpad=10)
    ax.set_ylabel("Allocation", fontsize=14, labelpad=10)
    plt.legend(
        title="",
        fontsize=10,
        title_fontsize=13
    )

    sns.move_legend(ax, "lower center", bbox_to_anchor = (0.5, -0.15), ncol = len(portfolio_data)/2, frameon = False)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Mostrar el gráfico en Streamlit
    st.pyplot(fig)

    plt.close(fig)

