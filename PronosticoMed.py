#import de librerias:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tc
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import PolynomialFeatures

#funciones

def encoder(df: pd.DataFrame, indx: int):
  ct = ColumnTransformer(
    transformers = [
        ('encoder', OneHotEncoder(), [indx])
    ],
    remainder = 'passthrough')
  df = np.array(ct.fit_transform(df))
  df = df[: ,1:]
  return df


def label_encoder(data: np.array, indx: int):
  le = LabelEncoder()
  data[: , indx] = le.fit_transform(data[: , indx])
  return data




def build_linear_model(data: pd.DataFrame):
  x = data[:, 0:-1]
  y = data[: , -1]
  x_train, x_test, y_train, y_test = tts(x, y, test_size = 0.2, random_state = 42)
  linmod = LinearRegression()
  linmod.fit(x_train, y_train)
  y_pred = linmod.predict(x_test)


  #prueba rápida
  #test_pred = linmod.predict(test)


  mse = mean_squared_error(y_test, y_pred)
  r2 = r2_score(y_test, y_pred)
  output = pd.DataFrame({'Precio real USD': y_test, 'Precio pronosticado USD': y_pred})
  output['Precio pronosticado USD'] = output['Precio pronosticado USD'].apply(lambda x: round(x,2))
  output.to_csv('predicciones_lineal.csv')
  print(output.head(10))
  print(f'mse: {mse}\nr2: {r2}')
  #print(test_pred)


def build_polinomic_model(df: np.array, n: int):
  #import
  x = df[ : , 0:-1]
  y = df[ : , -1]
  #transform to polinomic form
  polimerizer = PolynomialFeatures(degree = n)
  x_poly = polimerizer.fit_transform(x)
  #segmentation
  x_train, x_test, y_train, y_test = tts(x_poly, y, test_size = 0.2, random_state = 42)
  #model
  linmod = LinearRegression()
  linmod.fit(x_train, y_train)
  y_pred = linmod.predict(x_test)
  #metrics
  mse = mean_squared_error(y_test, y_pred)
  r2 = r2_score(y_test, y_pred)
  #output
  output = pd.DataFrame({'Precio real USD': y_test, 'Precio pronosticado USD': y_pred})
  output['Precio pronosticado USD'] = output['Precio pronosticado USD'].apply(lambda x: round(x,2))
  #output.to_csv('predicciones_polinomial.csv')
  print(output.head(10))
  print(f'mse: {mse}\nr2: {r2}')


def visualize(df):
  cols = []
  fig, axs = plt.subplots(1,3, figsize=(12,3))
  for i, var in enumerate(df.columns[:-1]):
    if df[var].dtype != 'object':
      cols.append(var)
  for i, var in enumerate(cols):
      axs[i].scatter(df[var], df[df.columns[-1]])
      axs[i].set_xlabel(var)
      axs[i].set_ylabel(df.columns[-1])
      plt.tight_layout()
  return plt.show()
#carga

df = pd.read_csv('insurance.csv')

#visualización 

#visualize(df)

#conversión

df = encoder(df, 1)
df = encoder(df,5)
df = label_encoder(df, 7)

build_linear_model(df)
build_polinomic_model(df, 2)
print(df[0:5, 0:])