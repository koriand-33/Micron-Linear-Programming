import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing,SimpleExpSmoothing


#Read our file
df = pd.read_csv('Ventas_trimestrales_general.csv')
df.head()
df.info()

df['Trimestre']= pd.to_datetime(df['Trimestre'],format='%d/%m/%Y')


print(df['Trimestre'].head())
print(df['Trimestre'].dtype)
ventas_trimestrales = df.groupby(pd.Grouper(key='Trimestre'))['Ventas'].sum()
ventas_trimestrales = df.groupby(df['Trimestre'])['Ventas'].sum()

print(ventas_trimestrales.head())
ventas_trimestrales.head()


#Graficar las ventas trimestrales
plt.figure(figsize=(12, 6))
plt.plot(ventas_trimestrales, marker='o',label='Ventas trimestrales')
plt.title('Ventas trimestrales de Micron')
plt.xlabel('Fecha')
plt.ylabel('Ventas en millones de dólares')
plt.grid(True)
plt.show()



decomposition = seasonal_decompose(ventas_trimestrales,model='additive',period=4)
fig= decomposition.plot()
fig.set_size_inches(14,10)
plt.show()




 
