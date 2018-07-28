import os

from tensorflow.contrib import predictor
from preprocessing import load_data
from config import PREDICT_DATA_DIRECTORY

predict_fn = predictor.from_saved_model('./model_saved/1532714754')

predict_data, predict_label = load_data(PREDICT_DATA_DIRECTORY)
print(predict_label)
predictions = predict_fn(
    {'x': predict_data}
)
print(predictions['classses'])
