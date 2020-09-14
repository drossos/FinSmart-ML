import matplotlib.pyplot as plt
import pandas as pd
import datetime
# import tensorflow as tf
# from tensorflow import keras
import pickle 
import mpld3
from mpld3 import plugins, utils
import numpy as np

from preprocess import FinDataPreprocessor
from base_obj import pp_base, loaded_model, model

#from preprocess import FinDataPreprocessor

# temp = pd.read_csv('app/data_backend/american_holdings.csv').columns
# tickers = [str(i) for i in temp]
# pp_base = FinDataPreprocessor(tickers)
# pp_base.pre_process()
# #learning network
# loaded_model = 'app/data_backend/saved_models/CNN5_E10000-american_holdings_reduced.h5'

def gen_projected_data(pp,ticker_ind=0,days=10):
    pred_values = list()
    proj_dates = list()

    given_features,given_labels,dates,titles = pp.all_features,pp.all_labels,pp.all_dates,pp.tickers
    model_features, model_labels = pp_base.all_features, pp_base.all_labels
    
    #curr_day_features,next_day_labels = integrate_into_baseframe(model_features,model_labels,given_features,given_labels,ticker_ind)
    # not needed function rn but just using for ease at the moment
    curr_day_features,next_day_labels = integrate_into_baseframe(model_features,model_labels,model_features,model_labels,ticker_ind)
    curr_date =dates[-1]
    for i in range(days):
        curr_day_features = scale_next_day(curr_day_features,next_day_labels)
        #Adding projected dates to the list
        if (curr_date + datetime.timedelta(days=1)).weekday() > 4:
            incr = 8-(curr_date + datetime.timedelta(days=1)).weekday()
        else:
            incr = 1
        proj_dates.append(curr_date + datetime.timedelta(days=incr))
        curr_date = curr_date + datetime.timedelta(days=incr)

        next_day_labels = model.predict(np.array([curr_day_features]))[0]
        pred_values.append(next_day_labels[ticker_ind])
    
    # turns the projected changes to percentage changes of last known security price
    percent_pred = np.array(pred_values) / next_day_labels[ticker_ind]
    scaled_proj = percent_pred * given_labels[-1][ticker_ind]

    return scaled_proj, proj_dates    

def gen_projected_plot(pp,proj_dates,scaled_proj):
    iven_features,given_labels,dates,titles = pp.all_features,pp.all_labels,pp.all_dates,pp.tickers
    model_features, model_labels = pp_base.all_features, pp_base.all_labels

    fig, ax = plt.subplots()
    ax.plot_date(x=proj_dates,y=scaled_proj,ls='-',marker='',lw=1)
    titles = [i+"_Actual" for i in titles]
    titles.append("Projected Stock Price")
    ax.legend(titles, loc='upper right')
    ax.set_ylabel("Price per Share")
    ax.set_title(loaded_model.split("/")[-1])
    
    html_fig = mpld3.fig_to_html(fig)
    # mpld3.show(fig)
    
    return scaled_proj

#returns a scalling factor array to project the given stock
def scale_projection(base_stock, next_day_labels):
    proj_scaling = list()
    for i in range(len(next_day_labels)):
        proj_scaling.append(base_stock/next_day_labels[i])
    return proj_scaling

def integrate_into_baseframe(model_features,model_labels,given_features,given_labels,ticker_ind):
    #putting desired stock into model frame to pass into dataframe
    for i in range(model_features.shape[1]):
        model_features[-1][i][ticker_ind] = given_features[-1][i][ticker_ind]
    curr_day_features = model_features[-1]
    model_labels[-1][ticker_ind] = given_labels[-1][ticker_ind]
    next_day_labels = model_labels[-1]

    return curr_day_features,next_day_labels
    
    
def scale_next_day(curr_feature, next_day_labels):
    for i in range(curr_feature.shape[0]-1):
        curr_feature[i] = curr_feature[i+1]
    curr_feature[-1] = next_day_labels
    return curr_feature

def gen_display_actual(pp):
    
    features,labels,dates,titles = pp.all_features,pp.all_labels,pp.all_dates,pp.tickers

    fig, ax = plt.subplots()
    ax.plot_date(x=dates,y=labels,ls='-',marker='',lw=1)
    titles = [i+"_Actual" for i in titles]
    titles.append(titles[0] + "_Predicted")
    ax.legend(titles, loc='upper right')
    ax.set_ylabel("Price per Share")
    
   
    html_fig = mpld3.fig_to_html(fig)
    return html_fig

def def_model_display():
    model_features, model_labels, dates,titles= pp_base.all_features, pp_base.all_labels, pp_base.all_dates, pp_base.tickers

    pred_values = model.predict(model_features)

    fig, ax = plt.subplots()
    #ax.plot_date(x=dates,y=model_labels[:,0:5],ls='-',marker='',lw=1)
    ax.plot_date(x=dates,y=pred_values[:,0:5],ls='-',marker='',lw=1)
    titles = [i+"_Actual" for i in titles[0:5]]
    titles.append(titles[0] + "_Predicted")
    ax.legend(titles, loc='upper right')
    ax.set_ylabel("Price per Share")
    ax.set_title(loaded_model.split("/")[-1])
    
   
    mpld3.show(fig)

def gen_plot_proj_actual(pp, proj_labels, proj_dates):
    labels,dates,titles = pp.all_labels,pp.all_dates,pp.tickers

    fig, ax = plt.subplots()
    ax.plot_date(x=dates,y=labels,ls='-',marker='',lw=1)
    ax.plot_date(x=proj_dates,y=proj_labels,ls='-',marker='',lw=1)
    titles = [i+"_Actual" for i in titles]
    titles_proj = [i+"_Projected" for i in titles]
    titles = titles + titles_proj
    ax.legend(titles, loc='upper right')
    ax.set_ylabel("Price per Share")
    ax.set_title(loaded_model.split("/")[-1])
    
    html_fig = mpld3.fig_to_html(fig)

    return html_fig

if __name__ == "__main__":
    #temp = FinDataPreprocessor(tickers=['BNS'])
    #temp.pre_process()
    #x, y = gen_projected_data(temp)
    #gen_plot_proj_actual(temp,x,y)
    # def_model_display()
    print("in dv")