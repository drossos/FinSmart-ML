import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import yfinance as yf
import dill
#https://towardsdatascience.com/time-series-forecasting-with-lstms-using-tensorflow-2-and-keras-in-python-6ceee9c6c651

train_val_ratio = (.7,.3)

class FinDataPreprocessor:
    #Used for cerating object from database binary dictionary
    def __dic_construct(self,dic):
        self.tickers = dic['tickers']
        self.stocks = list()
        self.num_outputs = dic['num_outputs']
        #Total labels for plotting purposes
        self.all_labels = dic['all_labels']
        self.all_features = dic['all_features']
        self.all_dates = dic['all_dates']
        #Index 0 is features and index 1 is the labels
        self.train = dic['train']
        self.val = dic['val']
        self.train_dates = dic['train_dates']
        self.val_dates = dic['val_dates']

    def __init__(self,tickers=None, dic=None):
        if tickers is None:
            self.__dic_construct(dic)
        else:
            self.tickers = tickers
            self.stocks = list()
            self.num_outputs = len(tickers)
            #Total labels for plotting purposes
            self.all_labels = np.zeros((1,len(self.tickers)))
            self.all_features = np.zeros((1,1,1))
            self.all_dates = list()
            #Index 0 is features and index 1 is the labels
            self.train = []
            self.val = []
            self.train_dates = []
            self.val_dates = []
        #create the stock objects for each security
        for i in self.tickers:
            self.stocks.append(yf.Ticker(i))
      
    def pre_process(self,train_val_ratio=(.7,.3),time_steps=10):
        #make sure data is present to process
        if len(self.tickers) == 0:
            print('no data to pre_proc')
            return 0;
        self.all_labels = np.zeros((1,len(self.tickers)))
        self.all_features = np.zeros((1,time_steps,len(self.tickers)))

        #Load in each data set
        data_frames = []
        for i in range(len(self.tickers)):
            data_frames.append([(self.stocks[i].history(period="max")["Close"]).to_frame()])
            data_frames[i][0].columns = [self.tickers[i]]
            data_frames[i][0]["Date"] = pd.to_datetime(data_frames[i][0].index)

        #Take intersection of dates
        date_intersect = data_frames[0][0]["Date"]
        for i in range(1,len(data_frames)):
            date_intersect = pd.Series(np.intersect1d(date_intersect,data_frames[i][0]["Date"]))
        
        for i in range(len(data_frames)):
            data_frames[i].append(self.extract_collision(date_intersect,data_frames[i][0]["Date"],data_frames[i][0][self.tickers[i]]))
        #Create DataFrame
        cols = [el[1] for el in data_frames]
        cols = list(zip(*cols))     #Transpose and then zip matrix
        multivar = pd.DataFrame(cols,columns = self.tickers, index = date_intersect)
        train,val = multivar.iloc[:int(len(multivar)*train_val_ratio[0])] , multivar.iloc[int(len(multivar)*train_val_ratio[0]):]

        time_steps = time_steps
        features_train, labels_train = self.create_dataset(train,train.values,time_steps)
        features_val, labels_val = self.create_dataset(val,val.values,time_steps)

        self.train= (features_train, labels_train)
        self.val = (features_val, labels_val)
        self.train_dates=train.index
        self.val_dates = val.index
        self.all_labels = np.delete(self.all_labels,0,0)
        self.all_features = np.delete(self.all_features,0,0)
        self.all_dates = self.train_dates[10:].append(self.val_dates[10:])
        
    def extract_collision(self,dates,unfilt_dates,values):
        inds = np.in1d(unfilt_dates,dates)
        mask = np.zeros(len(unfilt_dates))
        mask[inds] = 1
        mask = np.nonzero(mask)

        return [values[i] for i in mask[0]]

    def create_dataset(self,features,labels,time_steps=1):
        features_dat ,labels_dat = [],[]
        for i in range(len(features) - time_steps):
            v = features.iloc[i:(i + time_steps)].values
            features_dat.append(v)
            labels_dat.append(labels[i + time_steps][:self.num_outputs])
            #total labels for plotting
            temp_lab = labels[i + time_steps]
            temp_features = v
            self.all_labels= np.vstack((self.all_labels,temp_lab.reshape(1,temp_lab.shape[0])))
            self.all_features = np.vstack((self.all_features,v.reshape(1,v.shape[0],v.shape[1])))
        return np.array(features_dat), np.array(labels_dat)

    def to_dict(self):
        dic  = {
            'tickers':self.tickers,
            'num_outputs':self.num_outputs,
            #Total labels for plotting purposes
            'all_labels':self.all_labels,
            'all_features':self.all_features,
            'all_dates':self.all_dates,
            #Index 0 is features and index 1 is the labels
            'train':self.train,
            'val':self.val,
            'train_dates':self.train_dates,
            'val_dates':self.val_dates
        }
        return dic
           
if __name__ == "__main__":
    # preprpoc = Univar_Preprocess()
    # train,val,(train_dates,val_dates) = preprpoc.pre_process(ticker, isETF, train_val_ratio)
    # data_visulization.univar_display_actual_vs_predicted(train,train_dates[10:])
    #prefix = ""
    #tickers = ["BNS","BRK-B","F","GLD"]
    #temp = pd.read_csv('app/data_backend/american_holdings.csv').columns
    #tickers = [str(i) for i in temp]
    #pp = FinDataPreprocessor(tickers)

    #pp.pre_process(train_val_ratio) 
    print('in pp')
