import pandas as pd
import numpy as np
from keras import layers
from keras.layers.normalization import BatchNormalization
from keras import Input
from keras.models import Model
from keras import callbacks
import keras.optimizers as opt
from keras import regularizers

callback_list=[
    callbacks.EarlyStopping(monitor="loss",patience=200),
    callbacks.ModelCheckpoint(filepath="toy_v5_s.h5",monitor="loss",save_best_only=True),
    callbacks.ReduceLROnPlateau(monitor="loss",factor=0.85,verbose=1,patience=12)
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
x=layers.Dense(32,activation="relu")(data_input)
x=layers.Dense(64,activation="relu")(x)
x=layers.Dense(64,activation="relu")(x)
predict_inf=layers.Dense(1)(x)

predict_inf_model=Model(data_input,predict_inf)
predict_inf_model.compile(optimizer=opt.adam(),loss="mse")
predict_inf_model.fit(train,label,epochs=40000,batch_size=45,callbacks=callback_list)
