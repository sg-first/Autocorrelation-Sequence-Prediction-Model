import pandas as pd
import numpy as np
from keras import layers
from keras.layers.normalization import BatchNormalization
from keras import Input
from keras.models import Model
from keras import callbacks
import keras.optimizers as opt
from keras import regularizers
import matplotlib.pyplot as plt
callback_list=[
    callbacks.EarlyStopping(monitor="loss",patience=200),
    callbacks.ReduceLROnPlateau(monitor="loss",factor=0.85,verbose=1,patience=12)
]

infetion_A=pd.read_csv("infection_A.csv",names=["城市","区域","日期","新增感染人数"])
infetion_A=infetion_A[["日期","区域","新增感染人数"]]


worst_area=infetion_A.groupby("区域")["新增感染人数"].max()
worst_area=pd.DataFrame(worst_area[worst_area<400])
worst_area=worst_area.reset_index()
infetion_A=infetion_A[infetion_A["区域"].isin(worst_area["区域"])]
print(infetion_A)
lower=infetion_A.groupby("日期")["新增感染人数"].sum()
label=np.array(lower)
print(label)

train=np.arange(1,46)
train=(train-np.mean(train))/np.std(train)


data_input=Input(shape=(1,))
x=layers.Dense(64,activation="relu")(data_input)
x=layers.Dense(128,activation="relu")(x)
x=layers.Dense(128,activation="relu")(x)
predict_inf=layers.Dense(1)(x)

predict_inf_model=Model(data_input,predict_inf)
predict_inf_model.compile(optimizer=opt.adam(),loss="mse")
predict_inf_model.fit(train,label,epochs=40000,batch_size=45,callbacks=callback_list)
predict_inf_model.save("toy_v5_zt.h5")
predict_data=np.arange(1,76)
predict_data=(predict_data-np.mean(predict_data))/np.std(predict_data)
pre=predict_inf_model.predict(predict_data)
plt.figure()
plt.plot(train, label, label="data")
plt.plot(predict_data,pre,label="model")
plt.legend(loc="best")
plt.show()
