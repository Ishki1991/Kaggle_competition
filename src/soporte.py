import pandas as pd
pd.options.display.max_columns = None
import numpy as np
import pickle

# libreria normalización y estandarización
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler

# para calcular las métricas
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score 
from sklearn.metrics import precision_score 
from sklearn.metrics import recall_score 
from sklearn.metrics import f1_score 
from sklearn.metrics import cohen_kappa_score
from sklearn import metrics



from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV


def metricas(y_test, y_train, y_test_pred, y_train_pred, tipo_modelo):
    
    
    resultados = {'MAE': [metrics.mean_absolute_error(y_test, y_test_pred), metrics.mean_absolute_error(y_train, y_train_pred)],
                'MSE': [metrics.mean_squared_error(y_test, y_test_pred), metrics.mean_squared_error(y_train, y_train_pred)],
                'RMSE': [np.sqrt(metrics.mean_squared_error(y_test, y_test_pred)), np.sqrt(metrics.mean_squared_error(y_train, y_train_pred))],
                'R2':  [metrics.r2_score(y_test, y_test_pred), metrics.r2_score(y_train, y_train_pred)],
                 "set": ["test", "train"]}
    df = pd.DataFrame(resultados)
    df["modelo"] = tipo_modelo
    return df


def detectar_outliers(lista_columnas, dataframe): 
    
    dicc_indices = {} # creamos un diccionario donde almacenaremos índices de los outliers
    
    # iteramos por la lista de las columnas numéricas de nuestro dataframe
    for col in lista_columnas:
        
        #calculamos los cuartiles Q1 y Q3
        Q1 = np.nanpercentile(dataframe[col], 25)
        Q3 = np.nanpercentile(dataframe[col], 75)
        
        # calculamos el rango intercuartil
        IQR = Q3 - Q1
        
        # calculamos los límites
        outlier_step = 1.5 * IQR
        
        # filtramos nuestro dataframe para indentificar los outliers
        outliers_data = dataframe[(dataframe[col] < Q1 - outlier_step) | (dataframe[col] > Q3 + outlier_step)]
        
        
        if outliers_data.shape[0] > 0: # chequeamos si nuestro dataframe tiene alguna fila. 
        
            dicc_indices[col] = (list(outliers_data.index)) # si tiene fila es que hay outliers y por lo tanto lo añadimos a nuestro diccionario
        

    
    return dicc_indices 


def avg_val(df, columna, list_val):

    ordinal_dict = {}

    for val in list_val:
        precio_m = 0
        count = 0
        for i, f in df.iterrows():
            if val == f[columna]:
                precio_m = precio_m + round(f["price"], 2)
                count = count + 1
        print(f"{val} tiene un valor medio de: {round((precio_m/count), 2)} USD")



def ordinal_map(df, columna, orden_valores):
    ordinal_dict = {}
    
    for i, valor in enumerate(orden_valores):
        ordinal_dict[valor] = i
        
    nuevo_nombre = columna + "_mapeada"
    
    df[nuevo_nombre] = df[columna].map(ordinal_dict)
    
    return df

def ordinal_map_con(df, columna, list_val):

    ordinal_dict = {}

    for val in list_val:
        precio_m = 0
        count = 0
        for i, f in df.iterrows():
            if val == f[columna]:
                precio_m = precio_m + round(f["price"], 6)
                count = count + 1
        
        ordinal_dict[val] = round((precio_m/count), 6)

        nuevo_nombre = columna + "_mapeada"

        df[nuevo_nombre] = df[columna].map(ordinal_dict)
    
    return ordinal_dict

def label_encoder(df, columnas):
    for col in df[columnas].columns:
        nuevo_nombre = col + "_encoded"
        df[nuevo_nombre] = le.fit_transform(df[col])
    return 


def one_hot_encoder(dff, columnas):
    
    '''
    columnas: lista
    '''
    
    oh = OneHotEncoder()
    
    transformados = oh.fit_transform(dff[columnas])
    
    oh_df = pd.DataFrame(transformados.toarray(), columns = oh.get_feature_names_out(), dtype = int)
    
    dff[oh_df.columns] = oh_df
    
    dff.drop(columnas, axis = 1, inplace = True)
    
    return dff


def re_trans(df, columna, dic_val):
    lista = []
    for i, f in df.iterrows():
            lista.append(dic_val[f[columna]])
    
    f[columna] = lista

    return lista

def ordinal_encoder(df, columna, orden, num_modelo):
    ordinal = OrdinalEncoder(categories = [orden], dtype = int)
    transformados_oe = ordinal.fit_transform(df[[columna]])
    nuevo_nombre = columna + "_mapeada"
    
    df[nuevo_nombre] = transformados_oe

    with open(f'../data/modelo_{num_modelo}/encoding_{columna}.pkl', 'wb') as s:
        pickle.dump(ordinal, s)
    
    return df
