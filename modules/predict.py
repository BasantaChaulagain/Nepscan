from tensorflow.contrib import predictor
from preprocessing import load_data
from config import PREDICT_DATA_DIRECTORY

predict_fn = predictor.from_saved_model('./tmp/model_saved/1533889934')

def predictions():
    predict_data, predict_label = load_data(PREDICT_DATA_DIRECTORY)
    return predict_fn({'x': predict_data})

