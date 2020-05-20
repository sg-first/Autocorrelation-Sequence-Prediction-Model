from keras import callbacks
import keras.models
import pandas as pd
import numpy as np

predict_inf_model=keras.models.load_model('toy_v5_s.h5')
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

predict_inf_model.fit(train,label,epochs=40000,batch_size=45,callbacks=callback_list)