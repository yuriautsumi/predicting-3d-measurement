import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm
from scipy import stats
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def get_bootstrap_samples(X, y, num_bootstrap_samples=100):
    # Get bootstrap samples
    bootstrap_size = len(y) # number of data points in each set of sample
    
    bootstrap_data_samples = []
    for _ in tqdm(range(num_bootstrap_samples)):
        indices = np.random.choice(np.arange(bootstrap_size), size=(bootstrap_size, ), replace=True)
        bootstrap_data_samples.append((
            X.iloc[indices, :],
            y.iloc[indices]
        ))
        
    return bootstrap_data_samples

def evaluate_model(clf, X, y, bootstrap_data_samples, q1=0.05, q2=0.5, q3=0.95, num_bootstrap_samples=100):
    # Unbiased mean
    y_hat = clf.predict(X)
    r2_, mse_, mae_ = r2_score(y, y_hat), mean_squared_error(y, y_hat), mean_absolute_error(y, y_hat)
    
    # Recommended: Bootstrap (sample w/ replacement ~1k times) to get quantiles (or st dev for standard errors). Use regular acc/auc for unbiased mean 
    bootstrap_metrics = []
    for (X_test_i, y_test_i) in tqdm(bootstrap_data_samples):
        y_test_i_hat = clf.predict(X_test_i)
        r2, mse, mae = r2_score(y_test_i, y_test_i_hat), mean_squared_error(y_test_i, y_test_i_hat), mean_absolute_error(y_test_i, y_test_i_hat)
        bootstrap_metrics.append((r2, mse, mae))

    r2s  = list(map(lambda x: x[0], bootstrap_metrics))
    mses = list(map(lambda x: x[1], bootstrap_metrics))
    maes = list(map(lambda x: x[2], bootstrap_metrics))

    return pd.DataFrame(
        np.array([
            ['r2', 'r2', 'r2', 'mse', 'mse', 'mse', 'mae', 'mae', 'mae'],
            ['5th', 'mean', '95th', '5th', 'mean', '95th', '5th', 'mean', '95th'],
            [np.quantile(r2s, q1), r2_, np.quantile(r2s, q3), \
             np.quantile(mses, q1), mse_, np.quantile(mses, q3), \
             np.quantile(maes, q1), mae_, np.quantile(maes, q3)]
        ])
    )
    
def compute_residuals(X, y, clf):
    # Compute residuals
    residual_columns = [f'{x} residuals' for x in y.columns]
    X_w_residuals = X.copy()
    residuals = clf.predict(X) - y
    X_w_residuals[residual_columns] = residuals.values
    return X_w_residuals

def plot_residuals(X, y, categorical_columns, continuous_columns,
                   X_w_residuals, model_name, transformation_id, analysis_dir_dict=None, gender_id=None):
    # Plot residuals for each outcome x feature (original)
    nrows, ncols = X.shape[1], len(y.columns)
    fig, ax = plt.subplots(nrows, ncols, figsize=(3*nrows, 4*ncols))
    ax_ = ax.ravel()
    
    for i, xcol in enumerate(categorical_columns):
        for j, ycol in enumerate(y.columns):
            ix = i*len(categorical_columns)+j
            ycol = f'{ycol} residuals'
    
            if len(X[xcol].unique())>2: raise Exception('Only supports categorical variables with 2 categories.')
            c1 = X[xcol].unique()[0]
            c2 = list({'M', 'F'} - {c1})[0]
            # c1, c2 = X[xcol].unique()
            category_mask = X[xcol] == c1
            
            g1 = sns.histplot(data=X_w_residuals.loc[category_mask], x=ycol, color='blue', kde=True, ax=ax_[ix])
            g2 = sns.histplot(data=X_w_residuals.loc[~category_mask], x=ycol, color='orange', kde=True, ax=ax_[ix])
            g1.set_xlabel(ycol, fontsize=7); g1.set_ylabel('Count', fontsize=7)#; g2.set_xlabel(ycol, fontsize=7)
            
            # compute T test 
            rvs = [X_w_residuals.loc[category_mask, ycol], X_w_residuals.loc[~category_mask, ycol]]
            ttest = stats.ttest_ind(*rvs, equal_var=False)
            
            ax_[ix].set_title(f'Residual distributions for {xcol} \n{c1} (blue) v. {c2} (orange), p={ttest.pvalue:.2g}', fontsize=9)
            plt.tight_layout()
    
    offset = ix+1
    for i, xcol in enumerate(continuous_columns):
        for j, ycol in enumerate(y.columns):
            ix = offset+i*len(continuous_columns)+j
            ycol = f'{ycol} residuals'
            g1 = sns.regplot(data=X_w_residuals, x=xcol, y=ycol, ax=ax_[ix])
            g1.set_xlabel(xcol, fontsize=7), g1.set_ylabel(ycol, fontsize=7)
            # compute corr coeff. and p-value
            r, p = stats.pearsonr(X_w_residuals.loc[~X_w_residuals[xcol].isna(), xcol], 
                                  X_w_residuals.loc[~X_w_residuals[xcol].isna(), ycol])
            ax_[ix].set_title(f'{ycol} ~ {xcol} \n(Pearson corr (r)={r:.4f}, p={p:.4f})', fontsize=9)
            plt.tight_layout()
    
    plt.suptitle(f'{model_name} ({transformation_id})',y=0.99,fontsize=14)
    plt.tight_layout()
    if analysis_dir_dict is not None: 
        if gender_id is not None: figure_path = os.path.join(analysis_dir_dict[transformation_id], f'residualPlot_{model_name}_{gender_id}.png')
        else: figure_path = os.path.join(analysis_dir_dict[transformation_id], f'residualPlot_{model_name}.png')
        plt.savefig(figure_path, dpi=300)

def plot_categorical_residuals_by_gender(X, y, categorical_columns, X_w_residuals_dict, 
                                         model_name, transformation_id, analysis_dir_dict=None):
    # Plot residuals for each outcome x feature (original)
    nrows, ncols = len(categorical_columns), len(y.columns)
    fig, ax = plt.subplots(nrows, ncols, figsize=(3*ncols, 2*nrows))
    ax_ = ax.ravel()
    
    for i, xcol in enumerate(categorical_columns):
        for j, ycol in enumerate(y.columns):
            ix = i*len(categorical_columns)+j
            ycol = f'{ycol} residuals'
    
            if len(X[xcol].unique())>2: raise Exception('Only supports categorical variables with 2 categories.')
            c1 = X[xcol].unique()[0]
            c2 = list({'M', 'F'} - {c1})[0]
            # c1, c2 = X[xcol].unique()
            category_mask = X[xcol] == c1
            
            g1 = sns.histplot(data=X_w_residuals_dict['male'], x=ycol, color='blue', kde=True, ax=ax_[ix])
            g2 = sns.histplot(data=X_w_residuals_dict['female'], x=ycol, color='orange', kde=True, ax=ax_[ix])
            g1.set_xlabel(ycol, fontsize=7); g1.set_ylabel('Count', fontsize=7)#; g2.set_xlabel(ycol, fontsize=7)
            
            # compute T test 
            rvs = [X_w_residuals_dict['male'][ycol], X_w_residuals_dict['female'][ycol]]
            ttest = stats.ttest_ind(*rvs, equal_var=False)
            
            ax_[ix].set_title(f'Residual distributions for {xcol} \n{c1} (blue) v. {c2} (orange), p={ttest.pvalue:.2g}', fontsize=9)
            plt.tight_layout()

    plt.suptitle(f'{model_name} by gender ({transformation_id})',y=0.99,fontsize=9)
    plt.tight_layout()
    if analysis_dir_dict is not None: 
        figure_path = os.path.join(analysis_dir_dict[transformation_id], f'residualPlot_{model_name}.png')
        plt.savefig(figure_path, dpi=300)

