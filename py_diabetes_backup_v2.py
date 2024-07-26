import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import itertools
import warnings
import pickle

warnings.filterwarnings('ignore')

dataset = pd.read_csv('BTD_dataset.csv', delimiter=';')
# gender	age	  hypertension	heart_disease	smoking_history  	bmi	  HbA1c_level	blood_glucose_level	  diabetes
df = dataset
df = df.drop_duplicates()
df.isnull().sum()
#
 # trực quan hóa dữ liệu đếm các chỉ số
def add_counts(ax):
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                    textcoords='offset points')
fig, axes = plt.subplots(3, 2, figsize=(10, 10))

ax = sns.countplot(ax=axes[0, 0], x='gender', hue='diabetes', data=df)
axes[0, 0].set_title('Giới tính nhóm theo bệnh tiểu đường ')
add_counts(ax)

ax = sns.countplot(ax=axes[0, 1], x='hypertension', hue='diabetes', data=df)
axes[0, 1].set_title('Huyết áp cao nhóm theo bệnh tiểu đường')
add_counts(ax)

ax = sns.countplot(ax=axes[1, 0], x='heart_disease', hue='diabetes', data=df)
axes[1, 0].set_title('bệnh tim nhóm theo bệnh tiểu đường')
add_counts(ax)

ax = sns.countplot(ax=axes[1, 1], x='smoking_history', hue='diabetes', data=df)
axes[1, 1].set_title('Lich sử hút thuôc nhóm theo bệnh tiểu đường')
add_counts(ax)

ax = sns.countplot(ax=axes[2, 0], x='diabetes', data=df)
axes[2, 0].set_title("Số lượng bệnh tiểu đường")
add_counts(ax)

diabetes_counts = df['diabetes'].value_counts()
# startangle=90 vẽ từ góc 90 độ
axes[2, 1].pie(diabetes_counts, labels=diabetes_counts.index, autopct='%1.1f%%', startangle=90)
axes[2, 1].set_title('Diabetes Distribution')
# đặt tỷ lệ khung hình trục bằng nhau
axes[2, 1].axis('equal')
# hiển thị trus thích góc trên bên phải
axes[2, 1].legend(title='Diabetes:', loc='upper right')
# tự động điều chỉnh khoảng cách
plt.tight_layout()
plt.show()


# Chuyển đổi các cột categorical thành số
le = LabelEncoder()
df['gender'] = le.fit_transform(df['gender'])
df['smoking_history'] = le.fit_transform(df['smoking_history'])

# vẽ biểu đồ đánh giá dữ liệu
features = ['gender', 'age', 'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level']
fig, ax = plt.subplots(3, 2, figsize=(15, 15))
for i, feature in enumerate(features):
    row = i // 2
    col = i % 2
    sns.boxplot(y=df[feature], ax=ax[row, col])
    ax[row, col].set_title(f' Biểu đồ về {feature}')
    ax[row, col].set_xlabel('')
plt.tight_layout()
plt.show()


# biểu đồ sau khi loại bỏ bớt outline
df.loc[df['bmi'] > 45, 'bmi'] = 45
df.loc[df['bmi'] < 15, 'bmi'] = 15
df.loc[df['HbA1c_level'] >= 8, 'HbA1c_level'] = 8
df.loc[df['blood_glucose_level'] >= 250, 'blood_glucose_level'] = 250


fig, ax = plt.subplots(3, 2, figsize=(15, 15))
for i, feature in enumerate(features):
    row = i // 2
    col = i % 2
    sns.boxplot(y=df[feature], ax=ax[row, col])
    ax[row, col].set_title(f' Biểu đồ về {feature}')
    ax[row, col].set_xlabel('')
plt.tight_layout()
plt.show()

# đánh giá sự phân tán của HbA1c và blood_glucose
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='HbA1c_level', y='blood_glucose_level', hue='diabetes')
plt.title('Phân tán của HbA1c_level và blood_glucose_level')
plt.xlabel('HbA1c level')
plt.ylabel('glucose level')
plt.show()



plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Heatmap cho mối quan hệ giữa các biến số')
plt.show()


# Giả sử df là DataFrame ban đầu
# dataset_new cần dùng để dự đoán
# Thay thế các giá trị 0 bằng NaN trong các cột liên quan
features = ['bmi', 'HbA1c_level', 'blood_glucose_level']
df[features] = df[features].replace(0, np.nan)

# Điền các giá trị NaN bằng giá trị trung bình của cột
for col in features:
    df[col].fillna(df[col].mean(), inplace=True)

# Chuẩn hóa các thuộc tính bằng MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0, 1))
df_scaled = sc.fit_transform(df)

# Chuyển đổi dataset_scaled thành DataFrame để dễ dàng thao tác (tùy chọn)
df_scaled = pd.DataFrame(df_scaled, columns=df.columns)

# Tách dữ liệu
X = df_scaled.iloc[:, [1, 2, 5, 6, 7]].values
y = df_scaled.iloc[:, [8]].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Random forest Algorithm
from sklearn.ensemble import RandomForestClassifier
ranfor = RandomForestClassifier(n_estimators=41, criterion='entropy', random_state=42)
ranfor.fit(X_train, y_train)

# Support vector Algorithm
from sklearn.svm import SVC
svc = SVC(kernel='linear', random_state=42)
svc.fit(X_train, y_train)

# Logistic Regression Algorithm
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(random_state=42)
logreg.fit(X_train, y_train)

# KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Comparing models
print('Random Forest: ' + str(ranfor.score(X_test, y_test)))
print('SVM: ' + str(svc.score(X_test, y_test)))
print('Logreg: ' + str(logreg.score(X_test, y_test)))
print('KNN: ' + str(knn.score(X_test, y_test)))

# Choose model has the highest accuracy
y_predict = ranfor.predict(X_test)

# tạo biểu đồ heatmap dự đoán
confusion = confusion_matrix(y_test, y_predict)
plt.figure(figsize=(6, 4))
sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Dự đoán')
plt.ylabel('tập thử nghiệm')
plt.title('Tần xuất dự đoán')
plt.show()

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Confusion Matrix
print(confusion_matrix(y_test, y_predict))

# Classification Report
print(classification_report(y_test, y_predict))

# ROC AUC Score
roc_auc = roc_auc_score(y_test, ranfor.predict_proba(X_test)[:, 1])
print('ROC AUC Score: ', roc_auc)

cm = confusion_matrix(y_test, y_predict)
accuracy = accuracy_score(y_test, y_predict)
print("Confusion matrix: ")
print(cm)
print("accuracy score: ")
print(accuracy)

pickle.dump(ranfor, open('model.pkl', 'wb'))
model = pickle.load(open('model.pkl', 'rb'))
