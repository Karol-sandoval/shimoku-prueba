import pandas as pd
import numpy as np
import random

from typing import Dict

from pycaret.classification import *

# for splitting the data
from sklearn.model_selection import train_test_split

# for model building
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from feature_engine.imputation import AddMissingIndicator, \
    CategoricalImputer, MeanMedianImputer
from feature_engine.transformation import LogCpTransformer
from feature_engine.outliers import Winsorizer
from feature_engine.encoding import OrdinalEncoder, RareLabelEncoder, OneHotEncoder
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, roc_curve, precision_score, \
    recall_score


def get_data(leads_data_path: str, offers_data_path: str):
    leads = pd.read_csv(leads_data_path)
    offers = pd.read_csv(offers_data_path)

    def get_target_var(row):
        if row['Status'] == 'Closed Won':
            val = 1
        elif row['Status'] == 'Closed Lost':
            val = 0
        else:
            val = -1
        return val

    offers['status_num'] = offers.apply(get_target_var, axis=1)

    df_grouped = offers.groupby('Id')

    # Encuentra el índice del valor máximo en 'columna_interes' para cada grupo
    idx_max = df_grouped['status_num'].idxmax()

    # Selecciona las filas correspondientes en el dataframe
    df_max = offers.loc[idx_max]

    offers_final = df_max[['Id','Status','status_num']]

    dataframe_final = pd.merge(left=leads,right=offers_final, how='left', on='Id')

    dataframe_final = dataframe_final[dataframe_final['status_num'] != -1]
    dataframe_final = dataframe_final[dataframe_final['Id'].isna()==False]
    dataframe_final = dataframe_final[~ ((dataframe_final['status_num'].isna()) 
                                         & (dataframe_final['Converted']==1))]
    
    dataframe_final.loc[(dataframe_final['status_num'].isna()) 
                        & (dataframe_final['Converted']==0),['status_num']] = 0
    
    dataframe_final= dataframe_final.rename(columns={'status_num':'purchased'})
    dataframe_final= dataframe_final.rename(columns={'Status_x':'status_lead'})

    dataframe_final['Created Date']=pd.to_datetime(dataframe_final['Created Date'])

    dataframe_final['day_of_week'] = dataframe_final['Created Date'].dt.dayofweek
    dataframe_final['day_of_month'] = dataframe_final['Created Date'].dt.day
    dataframe_final['month'] = dataframe_final['Created Date'].dt.month

    list_category_columns = ['purchased','Use Case','Source','status_lead','Discarded/Nurturing Reason',
                         'Acquisition Campaign','City','day_of_week','day_of_month','month']

    dataframe_final[list_category_columns] = dataframe_final[list_category_columns].astype('category')

    dataframe_final = dataframe_final.drop(columns=['First Name','Created Date','Converted','Status_y'])

    dataframe_final = dataframe_final.set_index('Id')


    #------------------ FEATURE SELECTION AND FEATURE ENGINEERING ------------------#

    dataframe_final = dataframe_final.drop(columns=['status_lead','Discarded/Nurturing Reason'])

    categorical = dataframe_final.drop(columns='purchased').select_dtypes(include=["category","object"]).columns

    cat_vars_with_na = [
        var for var in categorical
        if dataframe_final.drop(columns='purchased')[var].isnull().sum() > 0
    ]
    with_string_missing = [
        var for var in cat_vars_with_na if dataframe_final.drop(columns='purchased')[var].isnull().mean() > 0.1]
    with_frequent_category = [
        var for var in cat_vars_with_na if dataframe_final.drop(columns='purchased')[var].isnull().mean() < 0.1]

    if len(with_string_missing) > 0:
        cat_imp_miss=CategoricalImputer(
            imputation_method='missing', variables=with_string_missing)

    if len(with_frequent_category) > 0:
        cat_imp_freq=CategoricalImputer(
            imputation_method='frequent', variables=with_frequent_category)
        

    dataframe_final[with_string_missing] = cat_imp_miss.fit_transform(dataframe_final[with_string_missing])
    dataframe_final[with_frequent_category] = cat_imp_freq.fit_transform(dataframe_final[with_frequent_category])

    # Assuming X_train is your input features dataframe and it includes categorical data
    encoder = OneHotEncoder()

    # Fit and transform the categorical columns
    X_encoded = encoder.fit_transform(dataframe_final.drop(columns='purchased')[categorical])

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded,  # predictive variables
        dataframe_final['purchased'],  # target
        test_size=0.2,  # portion of dataset to allocate to test set
        random_state=0,  # we are setting the seed here
    )

    sel_ = SelectFromModel(
        LogisticRegression(C= 0.5,
                        penalty='l1',
                        solver='liblinear',
                        random_state=10))

    # remove features with zero coefficient from dataset
    # and parse again as dataframe
    sel_.fit(X_train, y_train)
    importance = sel_.estimator_.coef_[0]

    X_train_new = pd.DataFrame(sel_.transform(X_train),
                    columns = X_train.columns[(sel_.get_support())],
                    index = X_train.index)
    

    #------------------ TRAINING MODELS ------------------#

    train_new = pd.concat([X_train_new,y_train], axis=1)
    test_new = pd.concat([X_test[X_train_new.columns],y_test], axis=1)     

    # Entrenamiento

    setup_model1 = setup(data = train_new, target = 'purchased',
                session_id = 1,
                preprocess = False,
                test_data = test_new,
                index=True,
                verbose = True
                        )

    best = compare_models(sort = 'AUC', cross_validation=True, fold=5, turbo=True
                          )

    test_new = test_new.rename(columns = {'Acquisition Campaign_Follow-up: digital guide':'Acquisition Campaign_Follow-up digital guide',
                           'Acquisition Campaign_Follow-up: digital guide 2':'Acquisition Campaign_Follow-up digital guide 2'})


    #------------------ PREDICTIONS ------------------#

    y_prob = best.predict_proba(test_new.drop(columns=['purchased']))[:,1 ]


    #------------------ PREDICTION TABLE ------------------#
    binary_prediction_table = pd.DataFrame({
        # 'Lead ID': test_new['Lead Number'].values,
        'Lead ID': test_new.index.values,
        'Probability': [round(100 * p, 2) for p in y_prob],
        'Lead Scoring': ['High' if v > 0.75 else 'Medium' if v > 0.5 else 'Low' for v in y_prob],
        # Get random set of two column names
        'Positive Impact Factors': [test_new.columns[np.random.randint(0, len(test_new.columns))] + ', ' +
                                    test_new.columns[np.random.randint(0, len(test_new.columns))]
                                    for i in range(len(y_prob))],
        'Negative Impact Factors': [test_new.columns[np.random.randint(0, len(test_new.columns))] + ', ' +
                                    test_new.columns[np.random.randint(0, len(test_new.columns))]
                                    for i in range(len(y_prob))],
    })

    leads_table = leads.head(10)
    offers_table = offers.head(10)
    lead_final = pd.DataFrame(leads.groupby(['Use Case'])['Id'].count()).reset_index()


    
    #--------------------- RETURN THE DATA ---------------------#
    return {
        'binary_prediction_table': binary_prediction_table,
        'leads_table': leads_table,
        'offers_table': offers_table,
        'lead_final': lead_final
    }