from flask import Flask,redirect,render_template,request,url_for,jsonify
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

app=Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
sc = pickle.load(open('sc.pkl', 'rb'))
pca=pickle.load(open('pca.pkl','rb'))
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
        useFile= 'useFile'in request.form
        if useFile:
            #xu ly dung file
            file = request.files['file']
            if file:

                custom_data=pd.read_csv(file,delimiter=';')
                data_array = custom_data.values#chuyen thanh numpy array
                features = ['age', 'hypertension', 'bmi', 'HbA1c_level', 'blood_glucose_level']

                # Strip whitespace from column names
                custom_data.columns = custom_data.columns.str.strip()
                print(custom_data)

                # Ensure the custom data contains the required features
                missing_features = [feature for feature in features if feature not in custom_data.columns]
                if missing_features:
                    return (f"Missing features in custom data: {missing_features}")
                else:
                    custom_X = sc.transform(custom_data[features])
                    custom_X_pca = pca.transform(custom_X)

                    # Make predictions on custom data
                    probability = model.predict_proba(custom_X_pca)
                    data_array_new=np.column_stack((data_array,probability[:,1]))
                    return render_template("predict.html",isPredicted=True,isListPatient=True,data_array=data_array_new)


                # # tinh phan tram 
                
                # #chuyen du lieu vao html
              
                #return render_template("predict.html",isPredicted=True,isListPatient=True,ageList=ageList,hypertensionList=hypertensionList,bmiList=bmiList,HbA1c_levelList=HbA1c_levelList,blood_glucose_levelList=blood_glucose_levelList)

        else:
             #xu ly khong dung file       
            age = float(request.form['age'])
            hypertension = float(request.form['hypertension'])
            bmi = float(request.form['bmi'])
            HbA1c_level = float(request.form['HbA1c_level'])
            blood_glucose_level = float(request.form['blood_glucose_level'])
            features = ['age', 'hypertension', 'bmi', 'HbA1c_level', 'blood_glucose_level']

            features_np = [[age, hypertension, bmi, HbA1c_level, blood_glucose_level]]
            custom_df = pd.DataFrame(features_np, columns=features)
            custom_X = sc.transform(custom_df[features])

            custom_X_pca = pca.transform(custom_X)
            # tinh phan tram 
            probability = model.predict_proba(custom_X_pca)
            probability_of_diabetes = probability[0][1] * 100
            output = f"Bạn có khả năng: {probability_of_diabetes:.2f}% mắc bệnh tiểu đường"
        
            return render_template("predict.html",isPredicted=True,isListPatient=False, resultFinal=output,blood_glucose_level=blood_glucose_level,HbA1c_level=HbA1c_level,hypertension=hypertension,bmi=bmi,age=age)
    else:
        return render_template("predict.html",isPredicted=False)
if __name__=="__main__":
    app.run(debug=True)