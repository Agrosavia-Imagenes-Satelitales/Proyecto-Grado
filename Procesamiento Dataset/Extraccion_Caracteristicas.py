# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 23:08:09 2021

@author: edwin

script para aumentar el vector de caracteristicas a partir de la base de datos de firmas espectrales de sentinel 2
y muestras carbosol, se realizan los siguientes procedimientos.
    1. Obtención de la primera derivada de la firma espectral
    2. Obtencion de la segunda derivada 
    3. Eliminación del continuo.
"""

import pandas as pd
import copy

def derivate(y1, y2, l1, l2):
    """
    
    Parameters
    ----------
    y1 : TYPE
        Valor de radiancia de la banda actual.
    y2 : TYPE
        Valor de radiancia de la muestra anterior.
    l1 : TYPE
        Longitud de onda de la muestra actual.
    l2 : TYPE
        Longitud de onda de la muestra anterior.

    Returns
    -------
    float.
        Difference formula: se utiliza la formula de diferencia o deridava hacia adelante con paso h
        donde h es la distancia entre muestras que para este caso seria la distancia entre longitudes de onda
        f'(a) = (f(a+h) - f(a))/h
    """
    slope = (y2 - y1)/(l2 - l1)
    return slope

dataset_images_original = pd.read_csv('E:/User/Escritorio/SEMESTRE 9/PROY GRADO 1/Imagenes Satelitales/Bases Datos Imagenes/LUCAS/France_Img_Dataset.csv',index_col=0)
dataset_images = copy.deepcopy(dataset_images_original)  

# Llongitudes de onda de la banda central de sentinel 2A
wavelengths = [4.427E-07,4.924E-07,5.598E-07,6.646E-07,7.041E-07,7.405E-07,7.828E-07,8.328E-07,8.647E-07,9.451E-07,1.3735E-06,1.6137E-06,2.2024E-06]

# Vector donde se almacenaran las derivadas
first_derivative = pd.DataFrame(columns=['B1dx','B2dx','B3dx','B4dx','B5dx','B6dx','B7dx','B8dx','B9dx','B10dx','B11dx','B12dx'])

#-------- CALCULO DE LA PRIMERA DERIVADA
for index, row in dataset_images.iterrows():
     temporal = []         # Lista temporal para almacenar las derivadas de la fila
     for j in range(12):
         l1 = wavelengths[j]
         l2 = wavelengths[j+1]
         y1 = dataset_images.iloc[index, j+17] # Primer valor de radiancia
         y2 = dataset_images.iloc[index, j+18] # Segundo valor de radiancia
         dx = derivate(y1, y2, l1, l2)
         temporal.append(dx) # Agrega a la fila de derivadas un valor columna a columna en cada pasada
     first_derivative.loc[index] = temporal # Agrega al dataframe de derivadas la fila completa 
# inserta en el dataset la primera derivada
for n in range(1,13,1):
    dataset_images.insert(29+n,'B'+str(n)+'dx',first_derivative['B'+str(n)+'dx'])

# Vector donde se almacenaran las segundas derivadas
second_derivative = pd.DataFrame(columns=['B1dx2','B2dx2','B3dx2','B4dx2','B5dx2','B6dx2','B7dx2','B8dx2','B9dx2','B10dx2','B11dx2'])

#-------- CALCULO DE LA SEGUNDA DERIVADA
for index, row in dataset_images.iterrows():
     temporal = []         # Lista temporal para almacenar las derivadas de la fila
     for j in range(11):
         l1 = wavelengths[j]
         l2 = wavelengths[j+1]
         y1 = dataset_images.iloc[index, j+30] # Primer valor de la primera derivada
         y2 = dataset_images.iloc[index, j+31] # Segundo valor de la primera derivada
         dx2 = derivate(y1, y2, l1, l2)
         temporal.append(dx2) # Agrega a la fila de derivadas un valor columna a columna en cada pasada
     second_derivative.loc[index] = temporal # Agrega al dataframe de derivadas la fila completa 

# Inserta en el dataset la segunda derivada
for n in range(1,12,1):
    dataset_images.insert(41+n,'B'+str(n)+'dx2', second_derivative['B'+str(n)+'dx2'])

#--------- ELIMINACIÓN DEL CONTINUO

import numpy as np
import pysptools.spectro as spectro
# https://pysptools.sourceforge.io/spectro.html 
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt

# Vector donde se almacenaran las derivadas B1cr -> B1 continuous removed
continuous_removal = pd.DataFrame(columns=['B1cr','B2cr','B3cr','B4cr','B5cr','B6cr','B7cr','B8cr','B9cr','B10cr','B11cr','B12cr', 'B13cr'])

# remoción del continuo con ConvexHull para todas las firmas espectrales del dataset
for index, row in dataset_images.iterrows():
     bands = dataset_images.iloc[index, 17:30]  # Localiza los 12 valores de la firma espectral en la fila index
     if index == 0:
             print(bands)
     # selecciona de la eliminacion del continuo en el arreglo de valores normalizados ????? 
     continuous = np.array(spectro.convex_hull_removal(bands, wavelengths) [0]).reshape(1, -1) 
     continuous = continuous[0].tolist()
     continuous_removal.loc[index] = continuous # Agrega al dataframe de remocion del continuo la los valores de banda sin continuo y normalizadas???

# inserta en el dataset la firma la remocion del continuo de cada firma espetral
for n in range(1,13,1):
    dataset_images.insert(52+n,'B'+str(n)+'cr', continuous_removal['B'+str(n)+'cr'])


# -- INDICES DE VEGETACION

indices = pd.DataFrame(columns=['GNDVI','EVI','SAVI','BSI','ARVI'])

for index, row in dataset_images.iterrows():
    bands = dataset_images.iloc[index, 17:30]
    band1 = bands[0]
    band2 = bands[1]
    band3 = bands[2]
    band4 = bands[3]
    band5 = bands[4]
    band6 = bands[5]
    band7 = bands[6]
    band8 = bands[7]
    band8a = bands[8]
    band9 = bands[9]
    band10 = bands[10]
    band11 = bands[11]
    band12 = bands[12]
    
    
    GNDVI = (band8-band3)/(band8+band3)
    EVI = 2.5*((band8-band4)/((band8+(6*band4)-(7.5*band2)+1)))
    SAVI = 1.428*((band8-band4)/(band8+band4+0.428))
    BSI = ((band11+band4)-(band8+band2))/((band11+band4)+(band8+band2))
    ARVI = (band8-(2*band4)+band2)/(band8+(2*band4)+band2)
    indices_list = [GNDVI, EVI, SAVI, BSI, ARVI]

    indices.loc[index] = indices_list
    
frames = [dataset_images, indices]
# concatena las tablas dataset_images e indices
dataset_imagesFE = pd.concat(frames, axis=1)         # extracted features dataset

# HABILITAR Linea para guardado de los datos
dataset_images.to_csv('E:/User/Escritorio/SEMESTRE 9/PROY GRADO 1/Imagenes Satelitales/Bases Datos Imagenes/LUCAS/France_Extracted_Features.csv')


