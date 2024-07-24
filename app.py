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
        float_features = [float(x) for x in request.form.values()]
        final_features = [np.array(float_features)]
        scaled_features = sc.transform(final_features)
        prediction = model.predict(scaled_features)
        probability = model.predict_proba(scaled_features)
        probability_of_diabetes = probability[0][1] * 100
        #-------create string
        glucose_level = float(request.form['Glucose Level'])
        insulin = float(request.form['Insulin'])
        bmi = float(request.form['BMI'])
        age = float(request.form['Age'])

        output = f"Bạn có khả năng: {probability_of_diabetes:.2f}% mắc bệnh tiểu đường"
        return render_template("predict.html",isPredicted=True, resultFinal=output,glucose_level=glucose_level,insulin=insulin,bmi=bmi,age=age)
    else:
        return render_template("predict.html",isPredicted=False)
if __name__=="__main__":
    app.run(debug=True)