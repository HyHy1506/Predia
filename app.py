from flask import Flask,redirect,render_template,request,url_for,jsonify
import pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler

app=Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
sc = pickle.load(open('sc.pkl', 'rb'))

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict/",methods=['GET','POST'])
def predict():
    if(request.method=='POST'):
       
        # scaled_features = sc.transform(final_features)
        # prediction = model.predict(scaled_features)
        # probability = model.predict_proba(scaled_features)
        # probability_of_diabetes = probability[0][1] * 100
        #-------create string
        age = float(request.form['age'])
        hypertension = float(request.form['hypertension'])
        bmi = float(request.form['bmi'])
        HbA1c_level = float(request.form['HbA1c_level'])
        blood_glucose_level = float(request.form['blood_glucose_level'])
        print(hypertension)
        features = [age, hypertension, bmi, HbA1c_level, blood_glucose_level]
        # chuyen list sang numpy.array
        features_array = np.array(features).reshape(1, -1)
        # chuan hoa
        features_array_scaled=sc.transform(features_array)
        # tinh phan tram 
        probability = model.predict_proba(features_array_scaled)
        probability_of_diabetes = probability[0][1] * 100
        output = f"Bạn có khả năng: {probability_of_diabetes:.2f}% mắc bệnh tiểu đường"
       
        return render_template("predict.html",isPredicted=True, resultFinal=output,blood_glucose_level=blood_glucose_level,HbA1c_level=HbA1c_level,hypertension=hypertension,bmi=bmi,age=age)
    else:
        return render_template("predict.html",isPredicted=False)
if __name__=="__main__":
    app.run(debug=True)