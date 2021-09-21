# -*- coding: utf-8 -*-
"""
Created on Mon May 24 18:50:56 2021

@author: edwin
"""
import os
import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns
import copy
import geopy                 # librerias para el calculo de distancias geodesicas y transformaciones
import geopy.distance        # funcionalidad para calcular distancias en el globo
from sentinelhub import WebFeatureService, BBox, CRS, DataCollection, SHConfig   # libreria para el acceso a SH

ruta = "E:/User/Escritorio/SEMESTRE 9/PROY GRADO 1/Python Imagenes/Procesamiento Dataset/CARBOSOL_horizons.tab"

spain_soil_original= pd.read_csv(ruta , sep = "\t")     #cargo los datos
spain_soil = copy.deepcopy(spain_soil_original)         #genero una copia real de los datos

# reemplazar valores faltantes 
spain_soil.replace("?", np.nan, inplace = True)
spain_soil.head(5)
 
description_soil = spain_soil.describe()  # descripción estadistica del dataset
#print(spain_soil.dtypes)                  # datatype of columns
#print(spain_soil.isnull().sum())          # Cantidad de valores faltantes 

'''
Dado que se desea analizar las muestras superficiales del suelo, solo se conservarán las muestras superficiales
del dataset y las demás seran eliminadas a continuación.
'''
headers = spain_soil.columns.values              # headers columna
#selecciono solo las muestras de suelo superficial
spain_soil = spain_soil[spain_soil['Position (Horizon position in the soil ...)'] <= 1]
# verifico que solo hallan quedado las muestras superficiales aquellas que tienen identificador = 1
print('Los valores unicos de Profundidad son',spain_soil['Position (Horizon position in the soil ...)'].unique())
print('valores unicos relative position:',spain_soil['Position (Horizon relative position in ...)'].unique())

#------------Eliminación de columnas que no son necesarias-----------
spain_soil.drop(['Location', 'Sample ID (Unique identification number ...)', 'Province',\
  'Sample ID (Unique identification number ...).1', 'Horizon (Horizon name in the original ...)', \
   'Position (Horizon position in the soil ...)', 'Position (Horizon relative position in ...)', \
    'Depth top [m]',   'Depth bot [m]', 'Depth [m]', 'Thn [m]', 'Color HLS', 'Density [g/cm**3]', \
      ], axis=1, inplace=True)
    
#---- Eliminación de propiedades quimicas que por su gran ausencia de datos no son reelevantes----

#--------------BORRAR ESTO!
print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
description_soil = spain_soil.describe()  # descripción estadistica del dataset
print(spain_soil.dtypes)                  # datatype of columns
spain_soil.replace(0, np.nan, inplace = True)
print('DATOS FALTANTES POR PROPIEDAD')
print(spain_soil.isnull().sum())          # Cantidad de valores faltantes en cada columna

datosFaltantes = spain_soil.isnull().sum()
unos = np.ones([22,1])                # creo un vector de unos
muestrasTotales = np.multiply(unos, 6599)     # multiplico por la cantidad de muestras
datosFaltantes= datosFaltantes.to_frame() # convierto el datos faltantes de series a dataframe
datosFaltantes.columns = ['Valores Faltantes']
datosFaltantes['Muestras Totales'] = muestrasTotales.tolist() # le agrego a datosFaltantes la columna de muestrasTotales
print(datosFaltantes.dtypes)
datosFaltantes.to_csv(index=True, path_or_buf = 'E:/User/Escritorio/SEMESTRE 9/PROY GRADO 1/Python Imagenes/faltantes.csv')


#spain_soil.drop(['Gp [%]'], axis=1, inplace=True)  # Gp % de yeso tiene 99.2% de datos faltantes
#spain_soil.drop(['Na [mg/kg]'], axis=1, inplace=True)  # Sodium [mg/kg] Sodio 81.6 % de datos faltantes

#-------- Seleccion de las etiquetas con la ubicacion geografica---------------

#----- % Litogenic Labesl (Coarse material) material grueso
lithogenic_labels = spain_soil[['Longitude', 'Latitude', 'Litho [%] (Coarse material)']]
lithogenic_labels["Litho [%] (Coarse material)"].replace(0, np.nan, inplace = True) # conversion de ceros a NaN o valores faltantes
lithogenic_labels = lithogenic_labels.dropna(subset=["Litho [%] (Coarse material)"], axis = 0 )
#reseteo de los indices, pandas conserva el indexado original y luego de la limpieza de datos los indices ya no siguen una secuencia
lithogenic_labels = lithogenic_labels.reset_index(drop = True)
lithogenic_labels.to_csv(index=True, path_or_buf = 'E:/User/Escritorio/SEMESTRE 9/PROY GRADO 1/Python Imagenes/TablasLabels/lithogenic_labels.csv')

#----- % Sand   porcentaje de arena
sand_labels = spain_soil[['Longitude', 'Latitude', 'Sand [%]']]
sand_labels["Sand [%]"].replace(0, np.nan, inplace = True) # conversion de ceros a NaN o valores faltantes
sand_labels = sand_labels.dropna(subset=["Sand [%]"], axis = 0 )
#reseteo de los indices, pandas conserva el indexado original y luego de la limpieza de datos los indices ya no siguen una secuencia
sand_labels = sand_labels.reset_index(drop = True)
sand_labels.to_csv(index=True, path_or_buf = 'E:/User/Escritorio/SEMESTRE 9/PROY GRADO 1/Python Imagenes/TablasLabels/sand_labels.csv')

#----- % Silt   porcentaje de sedimento o cieno
silt_labels = spain_soil[['Longitude', 'Latitude', 'Silt [%]']]
silt_labels["Silt [%]"].replace(0, np.nan, inplace = True) # conversion de ceros a NaN o valores faltantes
silt_labels = silt_labels.dropna(subset=["Silt [%]"], axis = 0 )
#reseteo de los indices, pandas conserva el indexado original y luego de la limpieza de datos los indices ya no siguen una secuencia
silt_labels = silt_labels.reset_index(drop = True)
silt_labels.to_csv(index=True, path_or_buf = 'E:/User/Escritorio/SEMESTRE 9/PROY GRADO 1/Python Imagenes/TablasLabels/silt_labels.csv')

#----- % Clay   porcentaje arcilla
clay_labels = spain_soil[['Longitude', 'Latitude', 'Clay min [%]']]
clay_labels["Clay min [%]"].replace(0, np.nan, inplace = True) # conversion de ceros a NaN o valores faltantes
clay_labels = clay_labels.dropna(subset=["Clay min [%]"], axis = 0 )
#reseteo de los indices, pandas conserva el indexado original y luego de la limpieza de datos los indices ya no siguen una secuencia
clay_labels = clay_labels.reset_index(drop = True)
clay_labels.to_csv(index=True, path_or_buf = 'E:/User/Escritorio/SEMESTRE 9/PROY GRADO 1/Python Imagenes/TablasLabels/clay_labels.csv')

#----- % Organic matter   porcentaje de materia organica
OM_labels = spain_soil[['Longitude', 'Latitude', 'OM [%]']]
OM_labels["OM [%]"].replace(0, np.nan, inplace = True) # conversion de ceros a NaN o valores faltantes
OM_labels = OM_labels.dropna(subset=["OM [%]"], axis = 0 )
#reseteo de los indices, pandas conserva el indexado original y luego de la limpieza de datos los indices ya no siguen una secuencia
OM_labels = OM_labels.reset_index(drop = True)
OM_labels.to_csv(index=True, path_or_buf = 'E:/User/Escritorio/SEMESTRE 9/PROY GRADO 1/Python Imagenes/TablasLabels/OM_labels.csv')

#----- % TOC Total Organic Carbon Percentage
TOC_labels = spain_soil[['Longitude', 'Latitude', 'TOC [%]']]
TOC_labels["TOC [%]"].replace(0, np.nan, inplace = True) # conversion de ceros a NaN o valores faltantes
TOC_labels = TOC_labels.dropna(subset=["TOC [%]"], axis = 0 )
#reseteo de los indices, pandas conserva el indexado original y luego de la limpieza de datos los indices ya no siguen una secuencia
TOC_labels = TOC_labels.reset_index(drop = True)
TOC_labels.to_csv(index=True, path_or_buf = 'E:/User/Escritorio/SEMESTRE 9/PROY GRADO 1/Python Imagenes/TablasLabels/TOC_labels.csv')

#----- % pH
pH_labels = spain_soil[['Longitude', 'Latitude', 'pH']]
pH_labels["pH"].replace(0, np.nan, inplace = True) # conversion de ceros a NaN o valores faltantes
pH_labels = pH_labels.dropna(subset=["pH"], axis = 0 )
#reseteo de los indices, pandas conserva el indexado original y luego de la limpieza de datos los indices ya no siguen una secuencia
pH_labels = pH_labels.reset_index(drop = True)
pH_labels.to_csv(index=True, path_or_buf = 'E:/User/Escritorio/SEMESTRE 9/PROY GRADO 1/Python Imagenes/TablasLabels/pH_labels.csv')

#----- % Carbonates 
carb_labels = spain_soil[['Longitude', 'Latitude', 'Carb [%]']]
carb_labels["Carb [%]"].replace(0, np.nan, inplace = True) # conversion de ceros a NaN o valores faltantes
carb_labels = carb_labels.dropna(subset=["Carb [%]"], axis = 0 )
#reseteo de los indices, pandas conserva el indexado original y luego de la limpieza de datos los indices ya no siguen una secuencia
carb_labels = carb_labels.reset_index(drop = True)
carb_labels.to_csv(index=True, path_or_buf = 'E:/User/Escritorio/SEMESTRE 9/PROY GRADO 1/Python Imagenes/TablasLabels/carb_labels.csv')

#----- Carbon/Nitrogen ratio (C/N) 
CN_rate_labels = spain_soil[['Longitude', 'Latitude', 'C/N']]
CN_rate_labels["C/N"].replace(0, np.nan, inplace = True) # conversion de ceros a NaN o valores faltantes
CN_rate_labels = CN_rate_labels.dropna(subset=["C/N"], axis = 0 )
#reseteo de los indices, pandas conserva el indexado original y luego de la limpieza de datos los indices ya no siguen una secuencia
CN_rate_labels = CN_rate_labels.reset_index(drop = True)
CN_rate_labels.to_csv(index=True, path_or_buf = 'E:/User/Escritorio/SEMESTRE 9/PROY GRADO 1/Python Imagenes/TablasLabels/CN_rate_labels.csv')

#----- Nitrogen, total [mg/kg] 
nitrogen_labels = spain_soil[['Longitude', 'Latitude', 'TN [mg/kg]']]
nitrogen_labels["TN [mg/kg]"].replace(0, np.nan, inplace = True) # conversion de ceros a NaN o valores faltantes
nitrogen_labels = nitrogen_labels.dropna(subset=["TN [mg/kg]"], axis = 0 )
#reseteo de los indices, pandas conserva el indexado original y luego de la limpieza de datos los indices ya no siguen una secuencia
nitrogen_labels = nitrogen_labels.reset_index(drop = True)
nitrogen_labels.to_csv(index=True, path_or_buf = 'E:/User/Escritorio/SEMESTRE 9/PROY GRADO 1/Python Imagenes/TablasLabels/nitrogen_labels.csv')

#----- Phosphorus [mg/kg] (P) 
phosphorus_labels = spain_soil[['Longitude', 'Latitude', 'P [mg/kg]']]
phosphorus_labels["P [mg/kg]"].replace(0, np.nan, inplace = True) # conversion de ceros a NaN o valores faltantes
phosphorus_labels = phosphorus_labels.dropna(subset=["P [mg/kg]"], axis = 0 )
#reseteo de los indices, pandas conserva el indexado original y luego de la limpieza de datos los indices ya no siguen una secuencia
phosphorus_labels = phosphorus_labels.reset_index(drop = True)
phosphorus_labels.to_csv(index=True, path_or_buf = 'E:/User/Escritorio/SEMESTRE 9/PROY GRADO 1/Python Imagenes/TablasLabels/phosphorus_labels.csv')

#----- Potassium [mg/kg] (K) 
potassium_labels = spain_soil[['Longitude', 'Latitude', 'K [mg/kg]']]
potassium_labels["K [mg/kg]"].replace(0, np.nan, inplace = True) # conversion de ceros a NaN o valores faltantes
potassium_labels = potassium_labels.dropna(subset=["K [mg/kg]"], axis = 0 )
#reseteo de los indices, pandas conserva el indexado original y luego de la limpieza de datos los indices ya no siguen una secuencia
potassium_labels = potassium_labels.reset_index(drop = True)
potassium_labels.to_csv(index=True, path_or_buf = 'E:/User/Escritorio/SEMESTRE 9/PROY GRADO 1/Python Imagenes/TablasLabels/potassium_labels.csv')

#----- Calcium [mg/kg] (Ca) 
calcium_labels = spain_soil[['Longitude', 'Latitude', 'Ca [mg/kg]']]
calcium_labels["Ca [mg/kg]"].replace(0, np.nan, inplace = True) # conversion de ceros a NaN o valores faltantes
calcium_labels = calcium_labels.dropna(subset=["Ca [mg/kg]"], axis = 0 )
#reseteo de los indices, pandas conserva el indexado original y luego de la limpieza de datos los indices ya no siguen una secuencia
calcium_labels = calcium_labels.reset_index(drop = True)
calcium_labels.to_csv(index=True, path_or_buf = 'E:/User/Escritorio/SEMESTRE 9/PROY GRADO 1/Python Imagenes/TablasLabels/calcium_labels.csv')

#----- Magnesium [mg/kg] (Mg) 
magnesium_labels = spain_soil[['Longitude', 'Latitude', 'Mg [mg/kg]']]
magnesium_labels["Mg [mg/kg]"].replace(0, np.nan, inplace = True) # conversion de ceros a NaN o valores faltantes
magnesium_labels = magnesium_labels.dropna(subset=["Mg [mg/kg]"], axis = 0 )
#reseteo de los indices, pandas conserva el indexado original y luego de la limpieza de datos los indices ya no siguen una secuencia
magnesium_labels = magnesium_labels.reset_index(drop = True)
magnesium_labels.to_csv(index=True, path_or_buf = 'E:/User/Escritorio/SEMESTRE 9/PROY GRADO 1/Python Imagenes/TablasLabels/magnesium_labels.csv')

#----- Sodium [mg/kg] (sodium) 
sodium_labels = spain_soil[['Longitude', 'Latitude', 'Na [mg/kg]']]
sodium_labels["Na [mg/kg]"].replace(0, np.nan, inplace = True) # conversion de ceros a NaN o valores faltantes
sodium_labels = sodium_labels.dropna(subset=["Na [mg/kg]"], axis = 0 )
#reseteo de los indices, pandas conserva el indexado original y luego de la limpieza de datos los indices ya no siguen una secuencia
sodium_labels = sodium_labels.reset_index(drop = True)
sodium_labels.to_csv(index=True, path_or_buf = 'E:/User/Escritorio/SEMESTRE 9/PROY GRADO 1/Python Imagenes/TablasLabels/sodium_labels.csv')

#----- Cation exchange capacity [cmol/kg] (CEC) 
CEC_labels = spain_soil[['Longitude', 'Latitude', 'CEC [cmol/kg]']]
CEC_labels["CEC [cmol/kg]"].replace(0, np.nan, inplace = True) # conversion de ceros a NaN o valores faltantes
CEC_labels = CEC_labels.dropna(subset=["CEC [cmol/kg]"], axis = 0 )
#reseteo de los indices, pandas conserva el indexado original y luego de la limpieza de datos los indices ya no siguen una secuencia
CEC_labels = CEC_labels.reset_index(drop = True)
CEC_labels.to_csv(index=True, path_or_buf = 'E:/User/Escritorio/SEMESTRE 9/PROY GRADO 1/Python Imagenes/TablasLabels/CEC_labels.csv')

#----- Conductivity, electrical [mS/m]
conductivity_labels = spain_soil[['Longitude', 'Latitude', 'Cond electr [mS/m]']]
conductivity_labels["Cond electr [mS/m]"].replace(0, np.nan, inplace = True) # conversion de ceros a NaN o valores faltantes
conductivity_labels = conductivity_labels.dropna(subset=["Cond electr [mS/m]"], axis = 0 )
#reseteo de los indices, pandas conserva el indexado original y luego de la limpieza de datos los indices ya no siguen una secuencia
conductivity_labels = conductivity_labels.reset_index(drop = True)
conductivity_labels.to_csv(index=True, path_or_buf = 'E:/User/Escritorio/SEMESTRE 9/PROY GRADO 1/Python Imagenes/TablasLabels/conductivity_labels.csv')


#----- Gypsum [%] (Gp)
gypsum_labels = spain_soil[['Longitude', 'Latitude', 'Gp [%]']]
gypsum_labels["Gp [%]"].replace(0, np.nan, inplace = True) # conversion de ceros a NaN o valores faltantes
gypsum_labels = gypsum_labels.dropna(subset=["Gp [%]"], axis = 0 )
#reseteo de los indices, pandas conserva el indexado original y luego de la limpieza de datos los indices ya no siguen una secuencia
gypsum_labels = gypsum_labels.reset_index(drop = True)
gypsum_labels.to_csv(index=True, path_or_buf = 'E:/User/Escritorio/SEMESTRE 9/PROY GRADO 1/Python Imagenes/TablasLabels/gypsum_labels.csv')


#-------Para probar la descarga utilizare 5 ubicaciones inicialmente
lithogenic_labels5 = lithogenic_labels.loc[0:5, 'Longitude':'Latitude']   #ubicaciones de prueba