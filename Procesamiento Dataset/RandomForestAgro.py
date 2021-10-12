# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 18:27:19 2021

@author: edwin
"""

import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import copy

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

selector = 4


if selector == 1:    # Base de datos pH Bandas   (0.27 en R2)
    tablaPH = pd.read_csv('E:/User/Escritorio/SEMESTRE 9/PROY GRADO 1/Python Imagenes/Bases Datos Imagenes/Features_Labels_pH.csv',index_col=0)

    dataset = copy.deepcopy(tablaPH)         #genero una copia real de los datos
    dataset.drop(['lat', 'long'], axis=1, inplace=True)
    
    features_x = dataset.iloc[:,0:14 ]
    labels_y = dataset['pH']
    yt = [i for i in dataset['pH']]
    
elif selector == 2:   #Base de datos pH + derivadas + Cont. Removal
    tablapH_der= pd.read_csv('E:/User/Escritorio/SEMESTRE 10/Proyecto de grado/RepositorioEliana/ProyectoGrado_Agrosavia-main/Bases Datos Imagenes/Features_labels_pH_derivatives.csv',index_col=0)
    
    dataset = copy.deepcopy(tablapH_der)         #genero una copia real de los datos
    dataset.drop(['lat', 'long'], axis=1, inplace=True)   # booro columnas latitud y longitud
    
    features_x = dataset.iloc[:,0:49 ]         # Vector de caracteristicas
    labels_y = dataset['pH']               # Vector de etiquetas
    yt = [i for i in dataset['pH']]        # Vector de etiquetas

elif selector == 3:
    tabla_portassium = pd.read_csv('E:/User/Escritorio/SEMESTRE 10/Proyecto de grado/RepositorioEliana/ProyectoGrado_Agrosavia-main/Bases Datos Imagenes/Features_labels_Potassium.csv',index_col=0)
    
    dataset = copy.deepcopy(tabla_portassium)         #genero una copia real de los datos
    dataset.drop(['lat', 'long'], axis=1, inplace=True)   # booro columnas latitud y longitud
    
    features_x = dataset.iloc[:,0:14 ]         # Vector de caracteristicas
    labels_y = dataset['K [mg/kg]']               # Vector de etiquetas
    yt = [i for i in dataset['K [mg/kg]']]        # Vector de etiquetas
    
elif selector == 4:     #Base de datos  LUCAS 
    tablaLUCAS = pd.read_csv('E:/User/Escritorio/SEMESTRE 9/PROY GRADO 1/Imagenes Satelitales/Bases Datos Imagenes/LUCAS/Spain_Img_Dataset.csv',index_col=0)
    dataset = copy.deepcopy(tablaLUCAS)         #genero una copia real de los datos

    print(tablaLUCAS.columns)
    features_x = dataset.iloc[:,17:30 ]         # Vector de caracteristicas
    labels_y = dataset['pH(CaCl2)']               # Vector de etiquetas
    yt = [i for i in dataset['pH(CaCl2)']]        # Vector de etiquetas


plt.figure(figsize=(16, 6))
heatmap = sns.heatmap(dataset.corr(), vmin=-1, vmax=1, annot=False, cmap='BrBG')



# Guarda la Matriz de correlación como una imagen
#plt.savefig('heatmapCorrelationMatrix.png', dpi=300, bbox_inches='tight')
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestRegressor as rfr
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split


# Division de los datos en entrenamiento y validación

X_train, X_test, y_train, y_test = train_test_split(features_x, yt, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
# note that the test set using the fitted scaler in train dataset to transform in the test set
X_test_scaled = scaler.transform(X_test)


estimators = [ 80, 250, 1200 ]
mean_rfrs = []
std_rfrs_upper = []
std_rfrs_lower = []

np.random.seed(11111)

for i in estimators:
    model = rfr(n_estimators=i, max_depth = None)
    scores_rfr = cross_val_score(model , X_train_scaled, y_train , cv = 5, scoring = 'r2')
    print('estimators:',i)
    #     print('explained variance scores for k=10 fold validation:',scores_rfr)
    print("Est. R2 : %0.2f (+/- %0.2f)" % (scores_rfr.mean(), scores_rfr.std() * 2))
    print("")
    mean_rfrs.append(scores_rfr.mean())
    std_rfrs_upper.append(scores_rfr.mean()+scores_rfr.std()*2) # for error plotting
    std_rfrs_lower.append(scores_rfr.mean()-scores_rfr.std()*2) # for error plotting


'''
# -------------Random Forest Regressor con optimizacion de hiperparametros por Grid Search
# https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74 
'''


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
'''
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)

# Fit the random search model
rf_random.fit(X_train_scaled, y_train)

print(rf_random.best_params_)
'''


def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))    
    return accuracy

base_model = RandomForestRegressor(n_estimators = 10, random_state = 42)
base_model.fit(X_train_scaled, y_train)
base_accuracy = evaluate(base_model, X_test_scaled, y_test)


#best_random = rf_random.best_estimator_
#random_accuracy = evaluate(best_random, X_test_scaled, y_test)

#print('Improvement of {:0.2f}%.'.format( 100 * (random_accuracy - base_accuracy) / base_accuracy))

# ----Se monta Metodo de cross validacion y grid searh con valores cercanos a los mejores obtenidos

'''
# Create the parameter grid based on the results of random search for pH + derivatives CARBOSOL
param_grid = {
    'bootstrap': [True],
    'max_depth': [40, 50, 60, 80],
    'max_features': ['sqrt'],
    'min_samples_leaf': [1, 2, 3 ],
    'min_samples_split': [2, 4, 8, 10, 12],
    'n_estimators': [100, 200, 300, 1000]
}
'''
# Create the parameter grid based on the results of random search for pH LUCAS
param_grid = {
    'bootstrap': [True],
    'max_depth': [40, 80, 90, 100],
    'max_features': ['sqrt'],
    'min_samples_leaf': [1, 2, 3 ],
    'min_samples_split': [2, 4, 5, 7, 9],
    'n_estimators': [800, 1200, 1400, 1500]
}
# Create a based model
rf = RandomForestRegressor()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, cv = 5, n_jobs = -1, verbose = 2)

# Fit the grid search to the data
grid_search.fit(X_train_scaled, y_train)
print('los mejores parametros son: ', grid_search.best_params_)

best_grid = grid_search.best_estimator_
grid_accuracy = evaluate(best_grid, X_test_scaled, y_test)

print('Improvement of {:0.2f}%.'.format( 100 * (grid_accuracy - base_accuracy) / base_accuracy))

from sklearn.metrics import r2_score, mean_squared_error

model = best_grid
scores_rfr = cross_val_score(model , X_train_scaled, y_train , cv = 5, scoring = 'r2')
print("Est. R2 : %0.2f (+/- %0.2f)" % (scores_rfr.mean(), scores_rfr.std() * 2))



#R2 = r2_score(y_test, rf.predict(X_test_scaled))

#print('el R2 es :', R2 )



'''
# Escalizacion de los datos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(features_x)
# note that the test set using the fitted scaler in train dataset to transform in the test set
X_test_scaled = scaler.transform(X_test)
'''

'''
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
# note that the test set using the fitted scaler in train dataset to transform in the test set
X_test_scaled = scaler.transform(X_test)
'''
'''
import pandas pd
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(X_crime, y_crime,
                                                   random_state = 0)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
# note that the test set using the fitted scaler in train dataset to transform in the test set
X_test_scaled = scaler.transform(X_test)
'''




