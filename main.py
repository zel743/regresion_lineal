import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

st.title("Predicion con regresion lineal simple")
st.subheader("By Oziel Velazquez ITC")
st.subheader("cargar datos")
uploaded_file = st.file_uploader("Sube un archivo CSV con tus datos", type=["csv"])
if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)
    st.write("Vista previa de los datos:")
    st.dataframe(data.head())
    columnas = data.columns.tolist()
    x = st.selectbox("Selecciona la variable independiente (X)", columnas)
    y = st.selectbox("Selecciona la variable dependiente (Y)", columnas)


    x_data = data[x].values
    y_data = data[y].values
    x_ones = np.c_[np.ones(len(x_data)), x_data]

    theta = np.linalg.inv(x_ones.T.dot(x_ones)).dot(x_ones.T.dot(y_data))
    st.write("valor de theta")
    st.write(theta)
    # agregar la r
    y_pred = theta[0] + theta[1] * x_data
    r2 = r2_score(y_data, y_pred)
    st.write(f"coeficiente de determinacion (R²): {r2:.4f}")
    # Interpretación
    if r2 > 0.7:
        st.success("el modelo tiene buen ajuste")
    elif r2 > 0.5:
        st.warning("el modelo tiene ajuste moderado")
    else:
        st.error("el modelo tiene bajo ajuste - reconsidera las variables")
    # escribir ecuacion del modelo
    st.write("### Ecuación del modelo:")
    st.write(f"**Y = {theta[1]:.2f}X + {theta[0]:.2f}**")
    # empezar a graficar
    fig, ax = plt.subplots()
    ax.scatter(data[x], data[y], s=40, c="red")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(f'Scatter Plot: {x} vs {y}')
    ax.grid(True)
    x_line = np.linspace(x_data.min(), x_data.max(), 100)
    y_line = theta[0] + theta[1] * x_line
    ax.plot(x_line, y_line, 'b-', linewidth=2, label='linea de regresion')
    ax.legend()
    st.pyplot(fig)

    n = st.number_input(f'ingresa el valor de la varibale dependiente {x}: ', value=0)

    peso_d = theta[0] + (theta[1] * n)
    st.write(f"prediccion aproximado de {y}:  {peso_d:.2f}")
else:
    st.info("Sube un archivo CSV para continuar.")