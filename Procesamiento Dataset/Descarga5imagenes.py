# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 09:42:46 2021

@author: edwin

Este codigo descarga 5 imagenes multiespectrales de una tabla con latitudes y longitudes
"""
#librerias
import os
import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns
import copy
import geopy                 # librerias para el calculo de distancias geodesicas y transformaciones
import geopy.distance
from sentinelhub import WebFeatureService, BBox, CRS, DataCollection, SHConfig   # libreria para el acceso a SH

ruta = "E:/User/Escritorio/SEMESTRE 9/PROY GRADO 1/Python Imagenes/Procesamiento Dataset/CARBOSOL_horizons.tab"

spain_soil_original= pd.read_csv(ruta , sep = "\t")     #cargo los datos
spain_soil = copy.deepcopy(spain_soil_original)         #genero una copia real de los datos

spain_soil.replace("?", np.nan, inplace = True)  # reemplazar valores faltantes por Nan
spain_soil.head(5)                               # Visualizacion de los datos
 
description_soil = spain_soil.describe()  # descripción estadistica del dataset
print(spain_soil.dtypes)                  # datatype of columns
print(spain_soil.isnull().sum())          # Cantidad de valores faltantes 

'''
Dado que se desea analizar las muestras superficiales del suelo, solo se conservarán las muestras superficiales
del dataset y las demás seran eliminadas.
'''
headers = spain_soil.columns.values              # headers columna
#selecciono solo las muestras de suelo superficial
spain_soil = spain_soil[spain_soil['Position (Horizon position in the soil ...)'] <= 1]
# verifico que solo hallan quedado las muestras superficiales 1
print('Los valores unicos de Profundidad son',spain_soil['Position (Horizon position in the soil ...)'].unique())
print('valores unicos relative position:',spain_soil['Position (Horizon relative position in ...)'].unique())

#------------Eliminación de columnas que no son necesarias-----------
#titanicData.drop(['PassengerId','Name','Ticket','Cabin'],axis=1,inplace=True)
spain_soil.drop(['Location', 'Sample ID (Unique identification number ...)', 'Province',\
  'Sample ID (Unique identification number ...).1', 'Horizon (Horizon name in the original ...)', \
   'Position (Horizon position in the soil ...)', 'Position (Horizon relative position in ...)', \
    'Depth top [m]',   'Depth bot [m]', 'Depth [m]', 'Thn [m]', 'Color HLS', 'Density [g/cm**3]', \
      ], axis=1, inplace=True)
    
#---- Eliminacion de propiedades quimicas que por su gran ausencia de datos no son reelevantes----

spain_soil.drop(['Gp [%]'], axis=1, inplace=True)  # Gp % de yeso tiene 99.2% de datos faltantes
spain_soil.drop(['Na [mg/kg]'], axis=1, inplace=True)  # Sodium [mg/kg] Sodio 81.6 % de datos faltantes

#-------- Seleccion de las etiquetas con la ubicacion geografica---------------
lithogenic_labels = spain_soil[['Longitude', 'Latitude', 'Litho [%] (Coarse material)']]
lithogenic_labels["Litho [%] (Coarse material)"].replace(0, np.nan, inplace = True)
lithogenic_labels = lithogenic_labels.dropna(subset=["Litho [%] (Coarse material)"], axis = 0 )
#reseteo de los indices, pandas conserva el indexado original y luego de la limpieza de datos los indices ya no siguen una secuencia
lithogenic_labels = lithogenic_labels.reset_index()

#-------Para probar la descarga utilizare 5 ubicaciones inicialmente
lithogenic_labels5 = lithogenic_labels.loc[0:0, 'Longitude':'Latitude']   #ubicaciones de prueba

#-----------CONFIGURACION DE MI CUENTA DE SENTINEL HUB-------
config = SHConfig() 
config.sh_client_id = '776adf5c-617a-4f21-bd45-bcd6c18853e0'         # usuario que me asigno sentinel hub
config.sh_client_secret = '8T)RFCc7-b]+qd7CD|ZZQ+&ZRYSWu/hRp4?AhNSy' # clave de sentinel hub

# verificacion de que si se escribieron las credenciales
if config.sh_client_id == '' or config.sh_client_secret == '':
    print("Warning! To use Sentinel Hub Process API, please provide the credentials (client ID and client secret).")
    
#------------ LIBRERIA PARA EL MANEJO DE DATOS CON SH-----------
from sentinelhub import MimeType, CRS, BBox, SentinelHubRequest, SentinelHubDownloadClient, \
    DataCollection, bbox_to_dimensions, DownloadRequest
import python_utils


for index, row in lithogenic_labels5.iterrows():
    #------------ Coordenadas del sitio en wgs84 (4 puntos coordenados)------------
    calculo_Bbox = [row['Latitude'], row['Longitude']] # Genero una caja contenedora de 1 Km^2 para la imagen satelital 
    start = geopy.Point(calculo_Bbox)   # latitud , longitud
    d = geopy.distance.GeodesicDistance(kilometers = 0.5)   
    endPoint1 = d.destination(point=start, bearing = 180)     # distancia que se recorre hacia abajo
    endPoint1 = d.destination(point=endPoint1, bearing = 270)     # distancia que se recorre hacia izda
    latitud1 = endPoint1.latitude        # Lat y Long de esquina inferior Izda
    longitud1 = endPoint1.longitude
    print('Lat:',latitud1,'-Long:', longitud1) 
    print(endPoint1)
    
    start = geopy.Point(calculo_Bbox)   # latitud , longitud
    d = geopy.distance.GeodesicDistance(kilometers = 0.5)   
    endPoint2 = d.destination(point=start, bearing = 0)     # distancia que se recorre hacia arriba
    endPoint2 = d.destination(point=endPoint2, bearing = 90)     # distancia que se recorre hacia Derecha
    latitud2 = endPoint2.latitude        # Lat y Long de esquina superior derecha
    longitud2 = endPoint2.longitude
    print('Lat:',latitud2,'-Long:', longitud2)
    print(endPoint2)
    
    par_coordenadas = [longitud1,latitud1,longitud2,latitud2] # Cuadro delimitador (coordenadas)
    resolution = 15   # minima resolucion espacial es de 15 metros
    BBox_coordenadas = BBox(bbox=par_coordenadas, crs=CRS.WGS84) # caja de coordenadas
    BBox_size = bbox_to_dimensions(BBox_coordenadas, resolution=resolution) # resolucion de la imagen
    
    print(f'Image shape at {resolution} m resolution: {BBox_size} pixels') 
    
    #------------Script personalizado de los datos de descarga--------

    evalscript_all_bands = """
        //VERSION=3
        function setup() {
            return {
                input: [{
                    bands: ["B01","B02","B03","B04","B05","B06","B07","B08","B8A","B09","B10","B11","B12"],
                    units: "DN"
                }],
                output: {
                    bands: 13,
                    sampleType: "INT16"
                }
            };
        }
    
        function evaluatePixel(sample) {
            return [sample.B01,
                    sample.B02,
                    sample.B03,
                    sample.B04,
                    sample.B05,
                    sample.B06,
                    sample.B07,
                    sample.B08,
                    sample.B8A,
                    sample.B09,
                    sample.B10,
                    sample.B11,
                    sample.B12];
        }
    """
    
    
    
    test_dir ='E:/User/Escritorio/SEMESTRE 9/PROY GRADO 1/Python Imagenes'  #direccion de la carpeta de guardado
    #-----------Se realiza la solicitud de datos a sentinel hub solicitando todas las bandas espectrales---------
    request_all_bands = SentinelHubRequest(
        data_folder=test_dir,
        evalscript=evalscript_all_bands,
        input_data=[SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L1C,
                time_interval=('2020-06-01', '2020-06-30'),
                mosaicking_order='leastCC')],
        responses=[SentinelHubRequest.output_response('default', MimeType.TIFF)],
        bbox=BBox_coordenadas,
        size=BBox_size,
        config=config
    )
    
    
    all_bands_img = request_all_bands.get_data(save_data=True)  # Guardado de la imagen Multiespectral
    
    print(f'The output directory has been created and a tiff file with all 13 bands was saved into ' \
          'the following structure:\n')

print('FINALIZO')
    
    
    
    



