import numpy as np

from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor

from sklearn.preprocessing import PowerTransformer
from sklearn.compose import TransformedTargetRegressor
from sklearn.multioutput import MultiOutputRegressor, RegressorChain

# Set transformer for outcomes (y)
y_transformer = PowerTransformer(standardize=True) # transform to unit gaussian

def get_transformed_regressor(regressor, multi=False, chain=False):
    """ Returns transformed target regressor.
        Helps make sure that we predict targets that have been transformed to unit Gaussian,
        but evaluate models in the original space. 
    """
    if chain: regressor = RegressorChain(regressor)
    elif multi: regressor = MultiOutputRegressor(regressor)
    return TransformedTargetRegressor(regressor=regressor, transformer=y_transformer)

# Hyperparameter dictionary for tuning 
models_hyperparameters_dict = {
    'LinearRegression': (
        get_transformed_regressor(LinearRegression()), {}
    ),
    'Ridge': (
        get_transformed_regressor(Ridge()), 
        {'regressor__alpha': (0.1, 1.0, 10.0)}
    ), # L2
    'Lasso': (
        get_transformed_regressor(Lasso()), 
        {'regressor__alpha': (0.1, 1.0, 10.0)}
    ), # L1
    'ElasticNet': (
        get_transformed_regressor(ElasticNet()), 
        {'regressor__alpha': (0.1, 1.0, 10.0), 'regressor__l1_ratio': np.arange(0.1,1.0,0.1)}
    ),
    'SVR_poly_multi': (
        get_transformed_regressor(SVR(), multi=True), {
        'regressor__estimator__kernel': ['poly'],
        'regressor__estimator__degree': [1], # just do 1st order on polynomial features
        'regressor__estimator__C': [1e-1, 1, 1e1],
        'regressor__estimator__epsilon': [1e-3, 1e-1, 1]
    }),
    'SVR_poly_chain': (
        get_transformed_regressor(SVR(), chain=True), {
        'regressor__base_estimator__kernel': ['poly'],
        'regressor__base_estimator__degree': [1], # just do 1st order on polynomial features
        'regressor__base_estimator__C': [1e-1, 1, 1e1],
        'regressor__base_estimator__epsilon': [1e-3, 1e-1, 1]
    }),
    'SVR_rbf_multi': (
        get_transformed_regressor(SVR(), multi=True), {
        'regressor__estimator__kernel': ['rbf'],
        'regressor__estimator__C': [1e-1, 1, 1e1],
        'regressor__estimator__epsilon': [1e-3, 1e-1, 1]
    }),
    'SVR_rbf_chain': (
        get_transformed_regressor(SVR(), chain=True), {
        'regressor__base_estimator__kernel': ['rbf'],
        'regressor__base_estimator__C': [1e-1, 1, 1e1],
        'regressor__base_estimator__epsilon': [1e-3, 1e-1, 1]
    }),
    'KNeighborsRegressor': (
        get_transformed_regressor(KNeighborsRegressor()), {
        'regressor__n_neighbors': [5, 10, 15, 25],
        'regressor__weights': ['uniform', 'distance'],
        # 'regressor__p': [1, 2],
        'regressor__metric': ['l1', 'l2', 'correlation', 'cosine'],
    }),
    'DecisionTreeRegressor': (get_transformed_regressor(DecisionTreeRegressor()), {
        'regressor__min_samples_split': [2, 5, 10],
        'regressor__max_features': ['sqrt', 'log2', None],
    }),
    'RandomForestRegressor': (get_transformed_regressor(RandomForestRegressor()), {
        'regressor__n_estimators': [50, 100, 250],
        'regressor__min_samples_split': [2, 5, 10],
        'regressor__max_features': ['sqrt', 'log2', None]
    }),
    'GradientBoostingRegressor_multi': (get_transformed_regressor(GradientBoostingRegressor(), multi=True), {
        'regressor__estimator__n_estimators': [250, 500],
        # 'estimator__loss': ['squared_error', 'absolute_error', 'huber', 'quantile'],
        'regressor__estimator__learning_rate': [1e-3, 1e-2, 1e-1],
        'regressor__estimator__min_samples_split': [2, 5, 10],
        # 'estimator__max_features': ['sqrt', 'log2', None]
    }),
    'GradientBoostingRegressor_chain': (get_transformed_regressor(GradientBoostingRegressor(), chain=True), {
        'regressor__base_estimator__n_estimators': [250, 500],
        # 'estimator__loss': ['squared_error', 'absolute_error', 'huber', 'quantile'],
        'regressor__base_estimator__learning_rate': [1e-3, 1e-2, 1e-1],
        'regressor__base_estimator__min_samples_split': [2, 5, 10],
        # 'estimator__max_features': ['sqrt', 'log2', None]
    }),
    # 'XGBRegressor': (get_transformed_regressor(XGBRegressor(multi_strategy='multi_output_tree')), { # added due to multi output support
    #     'regressor__n_estimators': [100, 250, 500],
    #     'regressor__learning_rate': [1e-3, 1e-2, 1e-1],
    #     'regressor__max_depth': [3, 5, 7, 10],
    #     # 'alpha': [10e-1, 10e-3, 10e-5],
    # }), # Note: removed because multi output support is a work in progress
    'MLPRegressor': (get_transformed_regressor(MLPRegressor()), {
        'regressor__hidden_layer_sizes': [(16, 16), (32, 32), (64, 64), (128, 128)],
        'regressor__learning_rate_init': [1e-3, 1e-2, 1e-1],
        'regressor__max_iter': [600, 1000]
    }),
}
