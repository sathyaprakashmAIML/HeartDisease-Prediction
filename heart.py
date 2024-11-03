import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

data='hearts.csv'
df=pd.read_csv(data)

le=LabelEncoder()
print(df.columns)

print(df.info())

df['Sex']=le.fit_transform(df['Sex'])
df['ChestPainType']=le.fit_transform(df['ChestPainType'])
df['RestingECG']=le.fit_transform(df['RestingECG'])
df['ExerciseAngina']=le.fit_transform(df['ExerciseAngina'])
df['ST_Slope']=le.fit_transform(df['ST_Slope'])

x=df.drop(columns=['HeartDisease'])
y=df['HeartDisease']
print('x',x)
print('y',y)
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.25,random_state=12)
NB=GaussianNB()
NB.fit(xtrain,ytrain)
ypred=NB.predict(xtest)
print(accuracy_score(ypred,ytest))


prediction=NB.predict([[29,0,2,100,106,1,2,80,1,1,1]])

if prediction==1:
    print('you have')
else:
    print('dont have')
