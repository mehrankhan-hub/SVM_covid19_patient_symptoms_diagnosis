import pandas as pd
import pickle as pkl
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
df = pd.read_csv('covid19_patient_symptoms_diagnosis.csv')
#print(df)
# print(df.isnull().sum())
# print(df.shape)
X = df[['age','gender','fever','dry_cough','sore_throat','fatigue','headache',
        'shortness_of_breath','loss_of_smell','loss_of_taste','oxygen_level',
        'body_temperature','comorbidity','travel_history','contact_with_patient',
        'chest_pain']]

y = df['covid_result']
for col in X.select_dtypes(include=['object', 'string']).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str)).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = SVC(kernel = 'rbf')

model.fit(X_train,y_train)
print('Prediction :',model.predict(X_test))
print('Train data Score :',model.score(X_train,y_train))
print('Test data Score :',model.score(X_test,y_test))


with open('Covid19 Prediction.pkl','wb') as f:
    pkl.dump(model, f)

