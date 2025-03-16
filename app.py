import pickle
from flask import Flask, request, app, jsonify, url_for,render_template
import numpy as np
import pandas as pd


app = Flask(__name__)  # initializing a Flask app from here
model  = pickle.load(open('regmodel.pkl','rb'))  # loading the ml model through pickle
scaler = pickle.load(open('scaler.pkl','rb'))


@app.route('/')  # this route the web app to home page
def home():
    return render_template('home.html')

'''
 this route to predict function to POST the data from html form to ML model to predict the housing price
 then the ml model use this data to make a prediction and send back the prediction through GET method
'''
@app.route('/predict_api',methods=['POST'])  ## this is a sample api created to call Ml model and get the output
def predict_api():
    data = request.json['data'] # raw data comming from html form
    print(f'data=',data)
    new_data = scaler.transform(np.array(list(data.values())).reshape(1,-1))
    print(f'new data=',new_data)
    output = model.predict(new_data)
    print(f'output=', output[0])
    return jsonify(output[0])

@app.route('/predict',methods=['POST'])  # this is predict method to send html form data to the model prediction
def predict():
    data = [float(x) for x in request.form.values()]
    new_data = scaler.transform(np.array(data).reshape(1,-1))
    output = model.predict(new_data)[0]
    return render_template('home.html',prediction_text ='The Predicted value of house is {}'.format(output))

if __name__=='__main__':
    app.run(debug=True)
