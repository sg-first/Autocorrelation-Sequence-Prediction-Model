from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras import Input,layers
from keras.models import Model
ensemble_model=load_model("A_0_v3.h5")
x=np.arange(1,46,1)
x_std=(x-x.mean())/x.std()
y=ensemble_model.predict(x)
"""
layer_input=Input(shape=(1,))
x=layer_input
for layer in ensemble_model.layers[2:]:
    x=layer(x)
new_ensemble_model=Model(layer_input,x)
y=new_ensemble_model.predict(x_std)
"""
df=pd.read_csv("infection_A.csv",names=["城市","区域","日期","增加人数"])
df=df.drop(columns=["城市","日期"])
area=df[df["区域"]==0]
area=area.reset_index()
area["index"]=(area["index"])%45+1
area.columns=["天数","区域","增加人数"]
print(area)
train_data=area["天数"]
train_data=np.array(train_data)
train_data_std=(train_data-train_data.mean())/train_data.std()
target=area["增加人数"]
target=np.array(target)
y=y.reshape((45,))
print(y)
print(target)
print(y-target)
print(((y-target)**2).mean())
plt.figure()
plt.plot(x,y,label="train_model")
plt.plot(train_data,target,label="data")
plt.legend(loc="best")
plt.show()
