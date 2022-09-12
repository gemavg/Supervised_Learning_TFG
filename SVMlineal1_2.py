# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 19:25:38 2022

@author: usuario
"""

# Tratamiento de datos
# ==============================================================================
import pandas as pd
import numpy as np

# Gráficos
# ==============================================================================
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
# from mlxtend.plotting import plot_decision_regions

# Preprocesado y modelado
# ==============================================================================
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

# Configuración matplotlib
# ==============================================================================
plt.rcParams['image.cmap'] = "bwr"
# plt.rcParams['figure.dpi'] = "100"
plt.rcParams['savefig.bbox'] = "tight"
# style.use('ggplot') or plt.style.use('ggplot') #ASUNCIÓN

# Configuración warnings
# ==============================================================================
import warnings

warnings.filterwarnings('ignore')

# lectura de datos
data = pd.read_csv('data.csv')

# dimesiones de los datos
# print("Dimension")
# print(data.shape)

# para ver una parte de los datos
# print(data.head())

# print(data.columns)

color = {"B": "blue", "M": "red"}
diagnosis_color = data.diagnosis.map(color)
marker = {"B": "o", "M": "x"}

fig, ax = plt.subplots()
for diagnosis in set(data.diagnosis):
    ax.scatter(
        data.radius_worst[data.diagnosis == diagnosis],
        data.texture_worst[data.diagnosis == diagnosis],
        s=30,
        c=color[diagnosis],
        marker=marker[diagnosis],
        label=diagnosis)
# plt.legend()
plt.show()
# ax.scatter(data.radius_worst, data.texture_worst, c=diagnosis_color);
# ax.set_title("Datos cancer de mama");

ax.set_xlim(5, 40)
ax.set_ylim(5, 60)

ax.set_xlabel('Radius_worst')  # Nombre del eje x
ax.set_ylabel('Texture_worst')  # Nombre del eje y

fig.set_facecolor('none')
ax.set_facecolor('none')

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# procesamiento
# dividimos los datos en atributos y etiquetas
# X = data.drop(['id','diagnosis'], axis=1)

X = pd.DataFrame(data, columns=['radius_worst', 'texture_worst'])
y = data['diagnosis']

# print(X.head(5))
# print(X)
# print(y)


# dividimos los datos en conjuntos de entrenamiento y prueba
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

##########################################################entrenamiento del algoritmo lineal
from sklearn.svm import SVC

svclassifier = SVC(kernel="linear", C=1000)
#svclassifier = SVC(kernel="poly", C=1000)
#svclassifier = SVC(kernel="rbf", C=1000, gamma=0.3)
svclassifier.fit(X_train, y_train)

print(svclassifier)

##########################################################Representación gráfica de los límites de clasificación
# ==============================================================================
# Grid de valores
x = np.linspace(np.min(X_train.radius_worst), np.max(X_train.radius_worst), 500)  # ASUNCIÓN
y = np.linspace(np.min(X_train.texture_worst), np.max(X_train.texture_worst), 500)  # ASUNCIÓN
Y, X = np.meshgrid(y, x)
grid = np.vstack([X.ravel(), Y.ravel()]).T

# Predicción valores grid
pred_grid = svclassifier.predict(grid)
colors = pred_grid.copy()  # ASUNCIÓN
colors[colors == 'M'] = 'red'  # ASUNCIÓN
colors[colors == 'B'] = 'blue'  # ASUNCIÓN

fig, ax = plt.subplots()
plt.scatter(grid[:, 0], grid[:, 1], c=colors, alpha=0.05, s=1)  # ASUNCIÓN
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

colors_train = y_train.copy()  # ASUNCIÓN
colors_train[colors_train == 'M'] = 'red'  # ASUNCIÓN
colors_train[colors_train == 'B'] = 'blue'  # ASUNCIÓN

ax.scatter(X_train.radius_worst[colors_train == 'blue'], X_train.texture_worst[colors_train == 'blue'], marker='o',
           c='blue', alpha=1)  # ASUNCIÓN
ax.scatter(X_train.radius_worst[colors_train == 'red'], X_train.texture_worst[colors_train == 'red'], marker='x',
           c='red', alpha=1)  # ASUNCIÓN

ax.set_xlabel('Radius_worst')  # Nombre del eje x #ASUNCIÓN
ax.set_ylabel('Texture_worst')  # Nombre del eje y #ASUNCIÓN

# Vectores soporte
ax.scatter(
    svclassifier.support_vectors_[:, 0],
    svclassifier.support_vectors_[:, 1],
    s=200, linewidth=1,
    facecolors='none', edgecolors='black'
)

# Hiperplano de separación
ax.contour(
    X,
    Y,
    svclassifier.decision_function(grid).reshape(X.shape),
    colors='k',
    levels=[-1, 0, 1],
    alpha=0.5,
    linestyles=['--', '-', '--']
)
plt.show()

# hacemos prediciones
y_pred = svclassifier.predict(X_test)

print(y_pred)

# Accuracy de test del modelo
# ==============================================================================
accuracy = accuracy_score(
    y_true=y_test,
    y_pred=y_pred,
    normalize=True
)
print("")
print(f"El accuracy de test es: {100 * accuracy}%")
print(f"{100 * accuracy}%")

# evaluamos el algortimo
from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_test, y_pred))

# Matriz de confusión de las predicciones de test
# ==============================================================================
confusion_matrix = pd.crosstab(
    y_test.ravel(),
    y_pred,
    rownames=['Real'],
    colnames=['Predicción']
)
print(confusion_matrix)
