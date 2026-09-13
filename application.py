from flask import Flask,request,jsonify,render_template
import pickle
import numpy as np
import pandas as pd
from  sklearn.preprocessing import StandardScaler

application=Flask(__name__)
app=application

## import  ridge  regressor and Standard scaler pickle 
ridge_model=pickle.load(open('models/ridge.pkl','rb'))
standard_scaler=pickle.load(open('models/scaler.pkl','rb'))




@app.route("/")
def index():
    return render_template('index.html')


@app.route("/predictdata",methods=['GET','POST'])
def predict_datapoint():
    if request.method =="POST":
        data = request.get_json()
            # Numeric conversions
        temperature = float(data['temperature'])
        rh = float(data['RH'])
        ws = float(data['WS'])
        rain = float(data['RAIN'])
        ffmc = float(data['FFMC'])
        dmc = float(data['DMC'])
        isi = float(data['ISI'])

    # Manual mapping for Classes
        class_map = {
            "Low": 0,
            "Moderate": 1,
            "High": 2,
            "Extreme": 3
        }
        classes = class_map.get(data['Classes'], -1)  # default -1 if not found

        # Manual mapping for Region (example)
        region_map = {
            "North": 0,
            "South": 1,
            "East": 2,
            "West": 3
        }
        region = region_map.get(data['Region'], -1)

    # Example: pass to ML model
        new_data_sclaed = standard_scaler.transform([[temperature, rh, ws, rain, ffmc, dmc, isi, classes, region]])
        results=ridge_model.predict(new_data_sclaed)
        return jsonify({"prediction": results})
    else:
        return render_template('home.html')
if __name__ =="__main__":
    app.run(host="0.0.0.0")


