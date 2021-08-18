from flask import Flask, request, jsonify
import joblib
import traceback
import pandas as pd
import numpy as np
import sys

# Your API definition
app = Flask(__name__)


@app.route('/', methods=['GET'])
def home():
    return "Alive!!"


@app.route('/predict', methods=['POST'])
def predict():
    if clf:
        try:
            json_ = request.json
            print("Json data request",json_)
            #Not a good way bc every column here will be dummies.. but my data as already been Dummies
            #Need to found a way to delete those 2 lines here
            query = pd.DataFrame(json_)
            #query = pd.get_dummies(query)
            query = query.reindex(columns=model_columns, fill_value=0)

            prediction = list(clf.predict(query))

            return jsonify({'prediction': str(prediction)})

        except Exception as ex:
            print(ex)
            return jsonify({'trace': traceback.format_exc()})

    else:
        print ('Train the model first')
        return ('No model here to use')

if __name__ == '__main__':
    try:
        port = int(sys.argv[1]) # This is for a command-line input
    except:
        port = 12345 # If you don't provide any port the port will be set to 12345

    clf = joblib.load("model.pkl") # Load "model.pkl"
    print ('Model loaded')
    model_columns = joblib.load("model_columns.pkl") # Load "model_columns.pkl"
    print ('Model columns loaded')

    app.run(port=port, debug=True)