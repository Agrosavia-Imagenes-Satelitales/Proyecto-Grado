# -*- coding: utf-8 -*-
"""
Created on Sun May 23 15:30:20 2021

@author: edwin
"""

import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns
import copy
import geopy
import geopy.distance
'''
ruta = "E:/User/Escritorio/SEMESTRE 9/PROY GRADO 1/Python Imagenes/Procesamiento Dataset/CARBOSOL_profile.tab"

profile_soil_original = pd.read_csv(ruta , sep = "\t")     #cargo los datos
profile_soil = copy.deepcopy(profile_soil_original)         #genero una copia real de los datos
'''

# Define starting point.
start = geopy.Point(41.559831,-5.850220)   # latitud , longitud

# Define a general distance object, initialized with a distance of 1 km.
d = geopy.distance.GeodesicDistance(kilometers = 0.015)

# Use the `destination` method with a bearing of 0 degrees (which is north)
# in order to go from point `start` 1 km to north.
endPoint = d.destination(point=start, bearing = 90)
latitud = endPoint.latitude
longitud = endPoint.longitude
print(41.559831,-5.850220)
print('Lat:',latitud,'-Long:', longitud)
print(endPoint)

#................. medir distancias de una coordenada a otra
# https://geopy.readthedocs.io/en/stable/#module-geopy.distance
from geopy import distance
newport_ri = (-5.01139043001206, 41.489809826062796)
cleveland_oh = (-5.011180994084809, 41.49001866886358)
print(distance.distance(newport_ri, cleveland_oh).miles)