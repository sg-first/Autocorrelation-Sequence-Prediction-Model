import pandas as pd
import numpy as np
from keras import layers
from keras.layers.normalization import BatchNormalization
from keras import Input
from keras.models import Model
from keras import callbacks
from keras.optimizers import adam
from keras import regularizers
callback_list=[
    callbacks.EarlyStopping(monitor="loss",patience=200),
    callbacks.ModelCheckpoint(filepath="toy_v5.h5",monitor="loss",save_best_only=True),
    callbacks.ReduceLROnPlateau(monitor="loss",factor=0.8,verbose=1,patience=12)
]
infetion_A=pd.read_csv("infection_A.csv",names=["城市","区域","日期","新增感染人数"])
infetion_A=infetion_A[["日期","区域","新增感染人数"]]
print(infetion_A)
worst_area=infetion_A[infetion_A["区域"]==39].iloc[:,2:]
print(worst_area)
label=np.array(worst_area)
print(label)
train=np.arange(1,46)
train=(train-np.mean(train))/np.std(train)
print(train)
data_input=Input(shape=(1,))
x=layers.Dense(128,activation="relu")(data_input)
x=layers.Dense(256,activation="relu")(x)
x=layers.Dense(512,activation="relu")(x)

x=layers.Dense(2048,activation="relu")(x)
x=layers.Dense(4096,activation="relu")(x)
predict_inf=layers.Dense(1)(x)
predict_inf_model=Model(data_input,predict_inf)
predict_inf_model.compile(optimizer=adam(),loss="mae")
predict_inf_model.fit(train,label,epochs=10000,batch_size=9,callbacks=callback_list)
