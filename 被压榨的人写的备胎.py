import numpy as np
import pandas as pd
from keras import callbacks
from keras.models import load_model
import keras
import keras.optimizers as opt
from keras import Input,layers
from keras.models import Model
import matplotlib.pyplot as plt

df=pd.read_csv("infection_A.csv",names=["城市","区域","日期","增加人数"])
df=df.drop(columns=["城市","日期"])


def generate_arrays_from_file(x,y):
    while True:
        yield (x,y)

for i in range(0,1):
    callback_list = [
        callbacks.EarlyStopping(monitor="loss", patience=500),
        #callbacks.ModelCheckpoint(filepath="A_{}_v4.h5".format(i), monitor="loss", save_best_only=True),
        callbacks.ReduceLROnPlateau(monitor="loss", factor=0.7, verbose=1, patience=12)
    ]
    area=df[df["区域"]==i]
    area=area.reset_index()
    area["index"]=(area["index"])%45+1
    area.columns=["天数","区域","增加人数"]
    print(area)
    train_data=area["天数"]
    train_data=np.array(train_data)
    target=area["增加人数"]
    target=np.array(target)

    model__1=load_model("toy_v5_s.h5")
    model__1.summary()
    model__1.trainable=False

    data_input = Input(shape=(1,))
    y = layers.Dense(2, activation="relu")(data_input)
    y = layers.Dense(4, activation="relu")(y)
    y = layers.Dense(8, activation="relu")(y)
    y = layers.Dense(16, activation="relu")(y)
    y = layers.Dense(32, activation="relu")(y)
    y = layers.Dense(16, activation="relu")(y)
    y = layers.Dense(8, activation="relu")(y)
    y = layers.Dense(4, activation="relu")(y)
    y = layers.Dense(2, activation="relu")(y)
    y = layers.Dense(1)(y)
    predict_2 = layers.normalization.BatchNormalization()(y)

    middle = model__1(predict_2)

    z = layers.Dense(128,activation="relu")(middle)
    predict_3 = layers.Dense(1)(z)

    ensemble_model=Model(inputs=data_input,outputs=predict_3)
    ensemble_model.summary()
    ensemble_model.compile(optimizer=opt.adam(), loss="mse")
    ensemble_model.fit_generator(generate_arrays_from_file(train_data,target), steps_per_epoch=1, epochs=40000, callbacks=callback_list)
    # ensemble_model.fit(train_data,target,batch_size=45,epochs=40000,callbacks=callback_list)

    print(ensemble_model.evaluate(train_data, target, 45))
    y = ensemble_model.predict(train_data)
    y = y.reshape((45,))
    plt.figure()
    plt.plot(train_data, y, label="train_model")
    plt.plot(train_data, target, label="data")
    plt.legend(loc="best")
    plt.show()

