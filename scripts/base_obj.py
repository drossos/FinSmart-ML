import pandas as pd

from preprocess import FinDataPreprocessor
from tensorflow import keras

temp = pd.read_csv('app/data_backend/american_holdings.csv').columns
tickers = [str(i) for i in temp]
pp_base = FinDataPreprocessor(tickers)
pp_base.pre_process()
#learning network
loaded_model = 'app/data_backend/saved_models/CNN5_E10000-american_holdings_reduced.h5'
model = keras.models.load_model(loaded_model)