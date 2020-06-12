import numpy as np
import pandas as pd
from keras import callbacks
from keras.models import load_model
import keras
import keras.optimizers as opt
from keras import Input, layers
from keras.models import Model
import matplotlib.pyplot as plt
import keras.backend as K
city = "A"


def sort(df):
    df = df.sort_values(by=["区域", "日期"])
    print(df)
    print(list(set(df["区域"])) == list(range(0, max(df["区域"]) + 1)))
    df.to_csv("infection_{}.csv".format(city), header=False, index=False)
def RMSLE(y_true,y_pred):
    first_log=K.log(K.clip(y_pred,K.epsilon(),None)+1.)
    second_log=K.log(K.clip(y_true,K.epsilon(),None)+1.)
    return K.sqrt(K.mean(K.square(first_log-second_log)))

df = pd.read_csv("infection_{}.csv".format(city), names=["城市", "区域", "日期", "增加人数"])
df = df.drop(columns=["城市", "日期"])
for i in range(0,118):
    callback_list = [
        callbacks.EarlyStopping(monitor="loss", patience=60),
        # callbacks.ModelCheckpoint(filepath="A_{}_v4.h5".format(i), monitor="loss", save_best_only=True),
        callbacks.ReduceLROnPlateau(monitor="loss", factor=0.8, verbose=1, patience=12)
    ]
    area = df[df["区域"] == i]
    area = area.reset_index()
    area["index"] = (area["index"]) % 45 + 1
    area.columns = ["天数", "区域", "增加人数"]
    train_data = area["天数"]
    train_data = np.array(train_data)
    target = area["增加人数"]
    target = np.array(target)
    xr = dict()

    for j in range(8):
        model__1 = load_model("toy_v5_A.h5")
        model__1.trainable = False
        model__1.name = "model_1"

        data_input = Input(shape=(1,))
        x = layers.BatchNormalization()(data_input)
        x = layers.Dense(128, activation="relu")(x)

        y = layers.Dense(128, activation="relu")(x)
        # y = layers.Dense(64, activation="relu")(y)
        y = layers.Dense(128, activation="relu")(y)
        x = layers.add([x, y])

        y = layers.Dense(128, activation="relu")(x)
        # y = layers.Dense(64, activation="relu")(y)
        y = layers.Dense(128, activation="relu")(y)
        x = layers.add([x, y])

        y = layers.Dense(128, activation="relu")(x)
        # y = layers.Dense(64, activation="relu")(y)
        y = layers.Dense(128, activation="relu")(y)
        x = layers.add([x, y])

        y = layers.Dense(128, activation="relu")(x)
        # y = layers.Dense(64, activation="relu")(y)
        y = layers.Dense(128, activation="relu")(y)
        x = layers.add([x, y])

        y = layers.Dense(32, activation="relu")(x)

        y = layers.Dense(1)(y)
        predict_3 = layers.normalization.BatchNormalization()(y)
        model__2 = Model(inputs=data_input, outputs=predict_3)
        model__2.name = "model_2"

        data_input = Input(shape=(1,))
        z = layers.Dense(64)(data_input)
        predict_3 = layers.Dense(1)(z)
        model__3 = Model(data_input, predict_3)
        model__3.name = "model_3"

        ensemble_input = keras.Input(shape=(1,))
        ensemble_output = model__3(model__1(model__2(ensemble_input)))
        ensemble_model = Model(ensemble_input, ensemble_output)

        ensemble_model.compile(optimizer=opt.adam(), loss="mse")
        ensemble_model.fit(train_data, target, epochs=7000, batch_size=45, callbacks=callback_list)
        y = ensemble_model.predict(train_data)
        y = y.reshape((45,))
        print(((y - target) ** 2).mean())
        print([i] * 50)
        xr[((y - target) ** 2).mean()] = ensemble_model

    ensemble_model = xr[min(xr, key=lambda x: x)]
    x_test = np.arange(46, 76, 1)
    y_test = ensemble_model.predict(x_test)
    y_test = np.where(y_test >= 0, y_test, 0)
    result = np.concatenate((x_test.reshape(30, 1), y_test.reshape(30, 1)), axis=1)
    predict_em = ensemble_model.predict(np.arange(1, 76)).reshape(75, )
    predict_em = np.where(predict_em >= 0, predict_em, 0)
    plt.figure()
    plt.plot(np.arange(1, 76), predict_em, label="train_model")
    plt.plot(train_data, target, label="data")
    plt.legend(loc="best")
    plt.savefig("F:\emsenble_slight_{}\{}城{}区拟合.png".format(city,city,str(i)))
    if (i == 0):
        result_final = pd.DataFrame(result, index=[i] * result.shape[0], columns=["天数", "感染人数"])
        result_final.to_csv("predict_bei_v2_{}.csv".format(city), columns=["天数", "感染人数"])
    else:
        result_final = pd.read_csv("predict_bei_v2_{}.csv".format(city), names=["天数", "感染人数"])
        result = pd.DataFrame(result, index=[i] * result.shape[0], columns=["天数", "感染人数"])
        result_final = pd.concat([result_final, result])
        result_final.to_csv("predict_bei_v2_{}.csv".format(city), columns=["天数", "感染人数"])


