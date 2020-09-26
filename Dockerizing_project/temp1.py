import numpy as np
import pandas as pd
from flask import Flask, request
import pickle

######################################
# Using postman
######################################


app  = Flask(__name__)
pickle_in = open('classifier.pkl', 'rb')
classifier = pickle.load(pickle_in)

@app.route('/')
def welcome():
    return "Welcome All"


@app.route('/predict')
def predict_note_authentication():
    variance = request.args.get("variance")
    skewness = request.args.get("skewness")
    curtosis = request.args.get("curtosis")
    entropy = request.args.get("entropy")
    
    prediction = classifier.predict([[variance,skewness,curtosis,entropy]])
    return "The predicted value is :" + str(prediction)


@app.route('/predict_file', methods=['POST'])
def predict_note_file():
    df_test = pd.read_csv(request.files.get("test_file"))
    prediction = classifier.predict(df_test)
    return "The predicted value for the csv file are :"+str(list(prediction))



if __name__ == '__main__':
    app.run()