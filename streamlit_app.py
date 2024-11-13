import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import plotly.graph_objects as go

#LIBRERIAS
####################################################################################################################################################################################################################################################################################################################################################################################################################################


def black_scholes(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

#BLACKSCHOLES
####################################################################################################################################################################################################################################################################################################################################################################################################################################


def generate_call_heatmap(strike_price, min_spot, max_spot, min_vol, max_vol, time_to_maturity, risk_free_rate):
    spot_prices = np.linspace(min_spot, max_spot, 20)
    volatilities = np.linspace(min_vol, max_vol, 20)

    call_prices = np.zeros((len(volatilities), len(spot_prices)))

    for i, vol in enumerate(volatilities):
        for j, spot in enumerate(spot_prices):
            call_prices[i, j] = black_scholes(spot, strike_price, time_to_maturity, risk_free_rate, vol, option_type='call')

    fig = go.Figure(data=go.Heatmap(
        z=call_prices,
        x=spot_prices,
        y=volatilities,
        colorscale='Viridis',
        colorbar=dict(title="Call Price"),
        hoverongaps=False,
        text=np.round(call_prices, 2),  
        texttemplate="%{text}", 
        textfont={"color": "white"}  
    ))
    fig.update_layout(
        title="Call Price Map",
        xaxis_title="Spot Price",
        yaxis_title="Volatility",
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=True),
        width=1000, 
        height=800  
    )
    st.plotly_chart(fig)


def generate_put_heatmap(strike_price, min_spot, max_spot, min_vol, max_vol, time_to_maturity, risk_free_rate):
    spot_prices = np.linspace(min_spot, max_spot, 20)
    volatilities = np.linspace(min_vol, max_vol, 20)

    put_prices = np.zeros((len(volatilities), len(spot_prices)))

    for i, vol in enumerate(volatilities):
        for j, spot in enumerate(spot_prices):
            put_prices[i, j] = black_scholes(spot, strike_price, time_to_maturity, risk_free_rate, vol, option_type='put')

    fig = go.Figure(data=go.Heatmap(
        z=put_prices,
        x=spot_prices,
        y=volatilities,
        colorscale='Viridis',
        colorbar=dict(title="Put Price"),
        hoverongaps=False,
        text=np.round(put_prices, 2),  # Agregar texto con valores redondeados
        texttemplate="%{text}",  # Mostrar texto como etiquetas
        textfont={"color": "white"}  # Color del texto blanco
    ))
    fig.update_layout(
        title="Put Price Heatmap",
        xaxis_title="Spot Price",
        yaxis_title="Volatility",
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=True),
        width=1000,  # Aumentar el ancho
        height=800  # Aumentar la altura
    )
    st.plotly_chart(fig)
    
#HEATMAPS
####################################################################################################################################################################################################################################################################################################################################################################################################################################


def main():
    st.set_page_config(
        page_title="Black-Scholes Option Pricing",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title('📊 Black-Scholes Option Pricing')
    st.caption('By Fernando Guzman')

    # Elementos de la barra lateral
    st.sidebar.title('📊 Black-Scholes Option Model')
    st.sidebar.write('By Fernando Guzman')

    # Inputs del usuario
    current_asset_price = st.sidebar.number_input('Current Asset Price', min_value=0.0, value=100.0, step=0.01, format="%.2f")
    strike_price = st.sidebar.number_input('Strike Price', min_value=0.0, value=100.0, step=0.01, format="%.2f")
    time_to_maturity = st.sidebar.number_input('Time to Maturity (Years)', min_value=0.0, value=1.0, step=0.01, format="%.2f")
    volatility = st.sidebar.number_input('Volatility (σ)', min_value=0.0, value=0.2, step=0.01, format="%.2f")
    risk_free_rate = st.sidebar.number_input('Risk-Free Interest Rate', min_value=0.0, value=0.05, step=0.01, format="%.2f")

    st.sidebar.markdown("---")

    activate_preferences = st.sidebar.button('Activate Heatmaps')
    min_spot_price = st.sidebar.number_input('Min Spot Price', min_value=0.0, value=current_asset_price * 0.9, step=0.01, format="%.2f")
    max_spot_price = st.sidebar.number_input('Max Spot Price', min_value=0.0, value=current_asset_price * 1.1, step=0.01, format="%.2f")
    min_volatility = st.sidebar.slider('Min Volatility for Heatmap', min_value=0.01, max_value=1.0, value=0.1, step=0.01)
    max_volatility = st.sidebar.slider('Max Volatility for Heatmap', min_value=0.01, max_value=1.0, value=0.5, step=0.01)

    # Mostrar tabla con parámetros seleccionados
    data = {
        'Current Asset Price': [current_asset_price],
        'Strike Price': [strike_price],
        'Time to Maturity (Years)': [time_to_maturity],
        'Volatility (σ)': [volatility],
        'Risk-Free Interest Rate': [risk_free_rate],
        'Min Spot Price': [min_spot_price],
        'Max Spot Price': [max_spot_price],
        'Min Volatility for Heatmap': [min_volatility],
        'Max Volatility for Heatmap': [max_volatility]
    }
    df = pd.DataFrame(data)
    st.table(df)

    call_price = black_scholes(current_asset_price, strike_price, time_to_maturity, risk_free_rate, volatility, option_type='call')
    put_price = black_scholes(current_asset_price, strike_price, time_to_maturity, risk_free_rate, volatility, option_type='put')

    col1, col2 = st.columns(2)
    with col1:
        st.write("### CALL Value")
        st.success(f"${call_price:.2f}")
    with col2:
        st.write("### PUT Value")
        st.error(f"${put_price:.2f}")

    st.write("## Options Price - Interactive Heatmap")
    if activate_preferences:
        st.write("### Call Price Heatmap")
        generate_call_heatmap(strike_price, min_spot_price, max_spot_price, min_volatility, max_volatility, time_to_maturity, risk_free_rate)
        st.write("### Put Price Heatmap")
        generate_put_heatmap(strike_price, min_spot_price, max_spot_price, min_volatility, max_volatility, time_to_maturity, risk_free_rate)

if __name__ == "__main__":
    main()
    
#MAINPAGE Y SIDEBAR
####################################################################################################################################################################################################################################################################################################################################################################################################################################
