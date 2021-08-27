import pandas as pd
import numpy  as np
import matplotlib.pyplot as mpl
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn import preprocessing
import math


#LOAD DATA
data = pd.read_csv('covid_vaccine_statewise.csv',sep=',')
data['id']=data.index
data=data[['Date','Vaccinated','id']]
print(data)
#PREPARATION
x=np.array(data['id']).reshape(-1,1)
y=np.array(data['Vaccinated']).reshape(-1,1)

mpl.plot(y,'-m')
#mpl.show()

pf= PolynomialFeatures(degree=10)
x=pf.fit_transform(x)

#TRAIN
mod=linear_model.LinearRegression()
mod.fit(x,y)
acc=mod.score(x,y)
print(f'ACCuracy:',{round(acc*100,3)})

y0=mod.predict(x)
mpl.plot(y0,'--b')
mpl.show()

#PREDICT
days=2
pred=(mod.predict(pf.fit_transform([[122+days]]))/10000000)
diff=(mod.predict(pf.fit_transform([[122+days]]))-185191602)
print(pred,'crores')
print('Total Persons vaccinated on given predicted day =',round(int(diff)),'(a negative sign(-)indicates that much amount of people were not vaccinated in comparison to previous day)')
