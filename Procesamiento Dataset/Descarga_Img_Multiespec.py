# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 17:17:02 2021

@author: edwin

Este script descarga de la tabla de propiedades fisicoquimicas con lat,lon, el  NDVI, Bandas de la imagen multiesprectral solicitando 
los datos a Sentinel Hub y guarda los valores de banda en una tabla
"""
#librerias
import cv2
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
# Librerias para manejo de fechas
from datetime import timedelta, date
from datetime import datetime


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

# direccion de la tabla de etiquetas
path = "E:/User/Escritorio/SEMESTRE 9/PROY GRADO 1/Imagenes Satelitales/TablasLabels/LUCAS/France_Georeferenced_Properties.xlsx"

dataset_original = pd.read_excel(path, sheet_name= 1)   #cargo los datos
dataset = copy.deepcopy(dataset_original)           #genero una copia real de los datos

# genero una ventana de tiempo de 34 dias para tener certeza de que Sentinel 2 tomo al menos una imagen.
dataset['Date Before'] = dataset['Date'] - timedelta(days=17)    # sumando 17 dias a la fecha 
dataset['Date After'] = dataset['Date'] + timedelta(days=17)  # restando 17 dias a la fecha 
# Convierto las fechas de datetime a string
dataset['Date Before'] = dataset['Date Before'].dt.strftime("%Y-%m-%d")  
dataset['Date After'] = dataset['Date After'].dt.strftime("%Y-%m-%d")

def getImage(lat, long, dateA, dateB):
    
    """
    Funcion que solicita la imagen multiespectral de Sentinel Hub del satelite Sentinel-2 solo 
    si se encuentra una imagen que cumpla con los parametros especificados

    Parameters
    ----------
    lat : float64
        Latitud de la muestra de suelo.
    long : float 64
        Longitud de la muestra de suelo.
    dateA : strftime
        Fecha de la toma de la muestra - x dias.
    dateB : strftime
        Fecha de la toma de la muestra + x dias.

    Returns
    -------
    TIFF image
        Imagen multiespectral con todas las 13 bandas del sentinel 2.

    """
    dateBefore = dateA
    dateAfter = dateB
    print('El Rango De Fechas Es :', dateBefore,'|', dateAfter)
    # UN PIXEL
    #------------ Coordenadas del sitio en wgs84 (4 puntos coordenados)------------
    # cero grados es el Norte
    calculate_Bbox = [lat, long] # Genero una caja contenedora de x Km^2 para la imagen satelital 
    start = geopy.Point(calculate_Bbox)   # latitud , longitud del punto de inicio 
    d = geopy.distance.GeodesicDistance(kilometers = 0.010)   # genero la distancia a recorren en kilometros
    endPoint1 = d.destination(point=start, bearing = 180)     # distancia que se recorre hacia abajo
    endPoint1 = d.destination(point=endPoint1, bearing = 270)     # distancia que se recorre hacia izda
    lat1 = endPoint1.latitude        # Lat y Long de esquina inferior Izda
    long1 = endPoint1.longitude
    print('Lat:',lat1,'-Long:', long1) 
    print(endPoint1)
    
    start = geopy.Point(calculate_Bbox)   # latitud , longitud del punto de inicio 
    d = geopy.distance.GeodesicDistance(kilometers = 0.010) # genero la distancia a recorren en kilometros 
    endPoint2 = d.destination(point=start, bearing = 0)     # distancia que se recorre hacia arriba
    endPoint2 = d.destination(point=endPoint2, bearing = 90)     # distancia que se recorre hacia Derecha
    lat2 = endPoint2.latitude        # Lat y Long de esquina superior derecha
    long2 = endPoint2.longitude
    print('Lat:',lat2,'-Long:', long2)
    print(endPoint2)
    
    coordenate_pair = [long1,lat1,long2,lat2] # Cuadro delimitador (coordenadas)
    resolution = 15   # minima resolucion espacial es de 15 metros
    BBox_coordinates = BBox(bbox=coordenate_pair, crs=CRS.WGS84) # caja de coordenadas
    BBox_size = bbox_to_dimensions(BBox_coordinates, resolution=resolution) # resolucion de la imagen
    
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
    
    
    
    test_dir ='E:/User/Escritorio/SEMESTRE 9/PROY GRADO 1/Python Imagenes/ubicaciones'  #direccion de la carpeta de guardado
    #-----------Se realiza la solicitud de datos a sentinel hub en todas las bandas espectrales---------
    request_all_bands = SentinelHubRequest(
        data_folder=test_dir,
        evalscript=evalscript_all_bands,
        input_data=[SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L1C,
                time_interval=( dateBefore, dateAfter),
                mosaicking_order='leastCC')],
        responses=[SentinelHubRequest.output_response('default', MimeType.TIFF)],
        bbox=BBox_coordinates,
        size=BBox_size,
        config=config
    )
    
    
    #all_bands_img = request_all_bands.get_data(save_data=True)  # Guardado de la imagen Multiespectral
    all_bands_img = request_all_bands.get_data()  # NO Guardado de la imagen Multiespectral
    
    #print(f'The output directory has been created and a tiff file with all 13 bands was saved into ' \
    #      'the following structure:\n')
    return (all_bands_img)

def BareSoilNDVI(image):   
    '''
    para que se considere suelo desnudo el NDVI debe estar entre 0 y 0.2

    Parameters
    ----------
    imagen : TYPE = numpy array
        DESCRIPTION. --> recibe una imagen en formato numpy array normalmente de dimensiones 1,1,13

    Returns 
    -------
       Retorna TRUE si el NDVI esta ENTRE 0 Y 0.1, FALSE si es DIFERENTE a 0.1

    '''
    Band4 = image[0,0,3]
    print('El valor de la banda 4 es:', Band4, 'tipo de dato:', type(Band4))
    

    Band8= image[0,0,7]
    print('El valor de la banda 8 es:', Band8)

    ndvi = (Band8.astype(float) - Band4.astype(float)) / (Band8 + Band4) #Formula de índice de vegetación de diferencia normalizada.
    print('El NDVI es:', ndvi) 
    
    if 0 < ndvi < 0.2:
        a = True
    else:
        a = False
    return a, ndvi
    
#creacón de la las columnas de la tabla
dataset_images = pd.DataFrame(columns=['Point_ID', 'Clay', 'Sand', 'Silt', 'pH(CaCl2)', 'pH(H2O)', 'EC', 'OC', 'CaCO3',
                                              'P', 'N', 'K', 'Latitude', 'Longitude', 'Date', 'Date Before', 'Date After', 
                                              'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10', 'B11', 'B12', 'B13', 'NDVI'])

#Linea para descarga interrumpida, Primero leo la tabla y luego sigo agregando valores desde el punto de interrupcion
#dataset_images = pd.read_csv('E:/User/Escritorio/SEMESTRE 9/PROY GRADO 1/Python Imagenes/Bases Datos Imagenes/Features_Labels_sand.csv',index_col=0)

#NOTA: Sila descarga es interumpida continuar la descarga desde una muestra despues de la ultima registrada cambiando iloc

#-------------------------------------V Numero de muestra
for index, row in dataset.iloc[0:].iterrows():
    long = row['Longitude']
    lat = row['Latitude']
    dateA = row['Date Before']
    dateB = row['Date After']
    #print(row['Latitude'], row['Longitude'])   
    #getImage(lat, long)
    image = getImage(lat, long, dateA, dateB )    # getimage genera una lista que dentro tiene un numpy array
    image1 = image[0]              # saco de la lista el numpy y me queda un array of uint16
    print('IMAGEN ES:' ,type(image1))
    print(image1.shape)
    print(image1[0,0,0])
    a, ndvi = BareSoilNDVI(image1) 
    print('El suelo esta desnudo', a)
    
    if a == True:
        #-- almaceno en variables los valores del pixel de suelo desnudo
        B1 = image1[0,0,0]
        B2 = image1[0,0,1]
        B3 = image1[0,0,2]
        B4 = image1[0,0,3]
        B5 = image1[0,0,4]
        B6 = image1[0,0,5]
        B7 = image1[0,0,6]
        B8 = image1[0,0,7]
        B8A = image1[0,0,8]
        B9 = image1[0,0,9]
        B10 = image1[0,0,10]
        B11 = image1[0,0,11]
        B12= image1[0,0,12]
        
        # Guardo los valores de las bandas en 
        dataset_images = dataset_images.append(
            {'Point_ID': row['Point_ID'], 'Clay':row['Clay'], 'Sand':row['Sand'], 'Silt':row['Silt'], 'pH(CaCl2)':row['pH(CaCl2)'], 
             'pH(H2O)':row['pH(H2O)'], 'EC':row['EC'], 'OC':row['OC'], 'CaCO3':row['CaCO3'], 'P':row['P'], 'N':row['N'], 'K':row['K'], 
             'Latitude':row['Latitude'], 'Longitude':row['Longitude'], 'Date':row['Date'], 'Date Before':row['Date Before'],
             'Date After':row['Date After'], 'B1':B1, 'B2':B2, 'B3':B3, 'B4':B4,'B5':B5,'B6':B6,'B7':B7,'B8':B8,'B9':B8A,'B10':B9,'B11':B10,
             'B12':B11,'B13':B12, 'NDVI':ndvi}, ignore_index=True)

    print('el numerdo de muestra es:',index)
    #guardo la tabla actualizada
    dataset_images.to_csv('E:/User/Escritorio/SEMESTRE 9/PROY GRADO 1/Imagenes Satelitales/TablasLabels/LUCAS/FRA_Img_Dataset.csv')
    
   
 






