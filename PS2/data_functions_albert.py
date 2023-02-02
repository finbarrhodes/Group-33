# -*- coding: utf-8 -*-
"""
Created on Wed May 30 11:34:27 2018

@author: Albert
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def remove_outliers(df,lq=0,hq=1):
    #df: Dataframe with only the variables to trim
    # lq: lowest quantile. hq:Highest quantile
    columns = pd.Series(df.columns.values).tolist()
    for serie in columns:
        df["houtliers_"+serie] = df[serie].quantile(hq)
        df[df[serie]>df["houtliers_"+serie]] = np.nan
        df["loutliers_"+serie] = df[serie].quantile(lq)
        df[df[serie]<df["loutliers_"+serie]]= np.nan
        del df["houtliers_"+serie], df["loutliers_"+serie]
    return df


def gini(array):
    # from: https://github.com/oliviaguest/gini
    #http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm 
    array = np.array(array)
    array = array.flatten() #all values are treated equally, arrays must be 1d
    if np.amin(array) < 0:
        array += np.amin(array) #non-negative
    array += 0.0000001 #non-0
    array = np.sort(array) #values must be sorted
    index = np.arange(1,array.shape[0]+1) 
    n = array.shape[0]
    return ((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array))) 



def data_stats(data, percentiles=[0.01, 0.25, 0.5, 0.75, 0.99]):
    ### Compute mean, sd, min, max, quintiles and Gini of the states in the state list.
    # Inputs -----------------
    # Dataframe
    
    #Obtain the describtive statistics for the state variables.    
        
    summary = data.describe(percentiles)
    summary.reset_index(inplace=True)
        
    #Generate Gini statistic:
    gini_stat = np.empty(len(data.columns)).reshape(1,len(data.columns))
    
    for i, var in enumerate(data.columns):
        gini_stat[:,i] = gini(data[var].dropna())
                
    data_gini = pd.DataFrame(gini_stat, columns=data.columns)    
    data_gini.reset_index(inplace=True)
    data_gini['index'] = 'Gini'
    summary = summary.append(data_gini, ignore_index=True)
    summary.set_index('index', inplace=True)
    
    return summary


def plot_cond_log_distr(data, variable1, variable2, save=False):
        fig, ax = plt.subplots()
        a = data[variable2].unique()
        for value in a:           
            sns.distplot((np.log(data.loc[data[variable2] == value][variable1]).replace([-np.inf, np.inf], np.nan)).dropna()-np.mean((np.log(data[variable1]).replace([-np.inf, np.inf], np.nan)).dropna()), label=variable2+str(value))
           
        plt.title('Distribution of '+variable1+' in Uganda')
        plt.xlabel(variable1)
        ax.legend()
        
        return plt.show()
        
def plot_cum_cond_log_distr(data, variable1, variable2, save=False):
        fig, ax = plt.subplots()
        a = data[variable2].unique()
        for value in a:           
            sns.distplot((np.log(data.loc[data[variable2] == value][variable1]).replace([-np.inf, np.inf], np.nan)).dropna()-np.mean((np.log(data[variable1]).replace([-np.inf, np.inf], np.nan)).dropna()), label=variable2+str(value), hist_kws=dict(cumulative=True),
             kde_kws=dict(cumulative=True))
        plt.title('Cumulative Distribution of '+variable1+' in Uganda')
        plt.xlabel(variable1)
        ax.legend()
        return plt.show()       





