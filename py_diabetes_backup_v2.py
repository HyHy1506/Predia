import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from imblearn.under_sampling import NearMiss
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from matplotlib import rcParams
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import itertools
from sklearn import tree
import warnings
warnings.filterwarnings('ignore')



datas_ = pd.read_csv('Dataset_model_new.csv', delimiter=';')
df = datas_




features = ['gender', 'age', 'hypertension','heart_disease', 'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level']
# df = pd.get_dummies(df , drop_first=True)

df = df.drop_duplicates()

le = LabelEncoder()
df['gender'] = le.fit_transform(df['gender'])
df['smoking_history'] = le.fit_transform(df['smoking_history'])


df.loc[df['bmi'] > 35, 'bmi'] = 35
df.loc[df['bmi'] < 15, 'bmi'] = 15
df.loc[df['HbA1c_level'] >= 8, 'HbA1c_level'] = 8
df.loc[df['blood_glucose_level'] >= 250, 'blood_glucose_level'] = 250
df = df[df['age'] >= 10]
df = df[df['HbA1c_level'] > 3.5]





features = ['age', 'hypertension', 'bmi', 'HbA1c_level', 'blood_glucose_level']
X = df[features]
y = df['diabetes']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA to reduce data dimensionality
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)


model_RFC = RandomForestClassifier(random_state= 42, n_estimators= 200, max_depth= 10, min_samples_split= 10 )



nm = SMOTE()
X , y = nm.fit_resample(X_pca, y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model_RFC.fit(X_train, y_train)


y_predict = model_RFC.predict(X_test)


custom_data = [
    [ 40 , 0, 20.6, 4.5, 90],
    [ 35, 1, 28.2, 7.2, 130],
    [ 55, 1, 31.4, 8.0, 150],
    [ 42, 0, 26.9, 7.0, 120],
    [ 50, 1, 29.7, 7.8, 140],
]

custom_df = pd.DataFrame(custom_data, columns=features)
custom_X = scaler.transform(custom_df[features])

custom_X_pca = pca.transform(custom_X)

custom_predictions = model_RFC.predict(custom_X_pca)
print(custom_predictions)

for i, pred in enumerate(custom_predictions):
    if pred == 0:
        print(f"Person {i+1} is not predicted to have diabetes.")
    else:
        print(f"Person {i+1} is predicted to have diabetes.")

