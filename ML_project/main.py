import pandas as pd
from flask import Flask,request,url_for,redirect,render_template,jsonify
from pycaret.regression import *
from logzero import logger
import pickle
import numpy as np
from pycaret.datasets import get_data

data = get_data('insurance')

data.shape

data.info()

# numeric = ['age', 'bmi', 'charges', 'smoker']

numeric = ['age', 'bmi', 'children', 'charges']
categorical = ['smoker', 'sex', 'region']
reg = setup(
    data = data,
    target= 'charges',
    train_size = 0.8,
    session_id = 123,
    normalize = True
)
# dataset=pd.read_excel('insurance_charges_dataset.xlsx')
# r2 = setup(dataset, target = 'charges', session_id = 123,
#            normalize = True,
#            polynomial_features = True, 
#            bin_numeric_features= ['Age', 'BMI'])
# logger.info(r2)

gbr = create_model('gbr',fold =10)
logger.info(gbr)
# logger.info(lr) 
params = {
    'learning_rate': [0.01, 0.02, 0.05],
    'max_depth': [1,2,3,4,5,6,7,8],
    'subsample': [0.4, 0.5, 0.6, 0.7, 0.8],
    'n_estimators': [100,200.300,400,500,600]
}

tuned_model = tune_model(
    gbr,
    optimize='RMSE',
    fold=10,
    custom_grid=params,
    n_iter= 30
)
logger.info(predict_model(gbr))
evaluate_model(gbr)
# # # logger.info(predict_model(lr))
final_lr = finalize_model(gbr) # Final model 
# # # # logger.info(final_lr)

# # # # logger.info(predict_model(final_lr))

save_model(final_lr, model_name = r'C:\Users\aboli\OneDrive\Desktop\ML_project\insurance_prediction')
# # logger.info(save_model)
