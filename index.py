import numpy as np #Librería numérica
import matplotlib.pyplot as plt # Para crear gráficos con matplotlib
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split

boston = datasets.load_boston() #Cargar el dataset
print(boston.keys()) # Mostrar las claves del dataset
print(boston.data.shape) # Mostrar la cantidad de datos del dataset
print(boston.feature_names) # Mostrar los nombres de las columnas

# Seleccionamos la columna 5 del dataset que es la que corresponde al numero de cuartos o "RM" como se encuentra señalada
X = boston.data[: , np.newaxis, 5]

# Definimos los datos correspondientes a las etiquetas 'target' dentro de las keys del dataset
Y = boston.target

# Graficamos los datos a traves de una grafica de dispersión con la librería plt
# plt.scatter(X,Y)
# plt.xlabel('Número de habitaciones')
# plt.ylabel('Valor medio')

# Preparamos los datos de entrenamiento y prueba
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)

# Definimos el algoritmo de regresion lineal
lr = linear_model.LinearRegression()

# Entrenamiento del modelo
lr.fit(X_train, Y_train)

# Realizamos una prediccion
Y_pred = lr.predict(X_test)

# Valor de la pendiente
print(lr.coef_)

# Valor de la interseccion
print(lr.intercept_)

# Precisión del modelo
print('Precisión del modelo')
print(lr.score(X_train, Y_train))

# Graficamos los datos junto con el modelo usando los test
# plt.scatter(X_test, Y_test)
# plt.plot(X_test, Y_pred, color='red', linewidth=3)
# plt.xlabel('Número de habitaciones')
# plt.ylabel('Valor medio')
# plt.show()