import numpy as np #Librería numérica
import matplotlib.pyplot as plt # Para crear gráficos con matplotlib

from sklearn.linear_model import LinearRegression #Regresión Lineal con scikit-learn

from sklearn.metrics import mean_squared_error

regresion_lineal = LinearRegression() # creamos una instancia de LinearRegression

def f(x):  # función para crear el ruido gaussiano
    np.random.seed(42) # para poder producir aleatoriamente el ruido
    y = 0.1*x + 1.25 + 0.2*np.random.randn(x.shape[0])
    return y

x = np.arange(0, 20, 0.5) # generamos valores x de 0 a 20 en intervalos de 0.5
y = f(x) # calculamos y a partir de la función que hemos generado

# Instruimos a la regresión lineal con los datos de x & y generados previamente
regresion_lineal.fit(x.reshape(-1,1), y);

# Imprimimos los datos de la regresión lineal
print('w = ' + str(regresion_lineal.coef_) + ', b = ' + str(regresion_lineal.intercept_))

# hacemos un gráfico de los datos que hemos generado
plt.scatter(x,y,label='data', color='blue')
plt.title('Datos');

# Prediciendo el valor de x=5
# prediccion_x = np.array([5])
# predict =regresion_lineal.predict(prediccion_x.reshape(-1, 1))

prediccion_entrenamiento = regresion_lineal.predict(x.reshape(-1,1));

#Calculamos MSE
mse = mean_squared_error(y_true = y, y_pred = prediccion_entrenamiento)

#Calculamos la raíz cuadrada del MSE (Error cuadratico Medio)
rmse = np.sqrt(mse);