# Load the necessary packAs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(
    'diabetes.csv',
    names = ['P','G','BP','ST','I','BMI','DPF','A','O'],
    header = 0
)
# print(df.head(5))

# print(df.describe())
# print(df.info())
# print(df.isnull().sum())

# dfP = df['P'][df['P'] == 0].count()
# dfG = df['G'][df['G'] == 0].count()
# dfBP = df['BP'][df['BP'] == 0].count()
# dfST = df['ST'][df['ST'] == 0].count()
# dfI = df['I'][df['I'] == 0].count()
# dfBMI = df['BMI'][df['BMI'] == 0].count()
# dfA = df['A'][df['A'] == 0].count()
# dfPF = df['DPF'][df['DPF'] == 0].count()

# print(dfP)
# print(dfG)
# print(dfBP)
# print(dfST)
# print(dfI)
# print(dfBMI)
# print(dfA)
# print(dfDPF)


# Split: feature X & target Y
x = df.drop(['O'], axis=1)
# print(x.iloc[0])
y = df['O']
# print(y)

x = np.array(x)
# print(x[0])
from sklearn.model_selection import train_test_split
xtr, xts, ytr, yts = train_test_split(
    x,
    y,
    test_size = .1
)

# Logistic Regression
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(solver = 'liblinear')
model.fit(xtr,ytr)

print(xtr)
#predict all data
df['Predict']=model.predict(x)      #variabel prediksi
print(df.head())
print('Score model = {}%'.format(round(model.score(xts,yts)*100,2)))
print(model.predict([[6,148,72,35,0,33.6,0.627,52]]))

import joblib
joblib.dump(model,'modelDiabetes')