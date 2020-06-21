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
city = "E"#改
slight_list_A=[28,59,69,73,85,91,95,100,115,116,6,8,45,49,48]
slight_list_B=[2,8,10,11,12,18,19,22,25,27]+[6,13,14,15,21,23,29]
slight_list_C=[2,3,4,6,7,8,9,14,15,16,20,21,24,25,35,39,44,47,50,56,64,68,75,78,79,84,86,92,95,97,116,121,124,126,128,132,134]
slight_list_D=[4,5,6,7,10,11,15,18,21,22,24,26,30,32,34,36,41,42,45,46,47,51,53,54,56,58,59,62,63,64,66,67,68,69,72,73,74]+[1,2,3,9,16,17,19,25,33,35,38,43,44,49,52,60,61,65,70,71]
slight_list_E=[0,3,5,9,16,18,19,20,22,23,27,29,30,32,33]+[1,2,4,6,10,12,13,14,21,24,25,26,28,31]
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
cant_list=list()
for i in slight_list_E[slight_list_E.index(25):]:#改
    callback_list = [
        callbacks.EarlyStopping(monitor="loss", patience=60),
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
    em=True

    def reduce(x):
        return x/(11000/(1*max(target)))


    count = 0
    while(em):
        for j in range(1,6):
            if(max(target)>200):
                model__1 = load_model("toy_v5_xr.h5")
            else:
                model__1 = load_model("toy_v5_zt.h5")

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
            y= layers.BatchNormalization()(y)
            model__2 = Model(inputs=data_input, outputs=y)
            model__2.name = "model_2"

            data_input = Input(shape=(1,))
            z = layers.Lambda(reduce)(data_input)
            z = layers.Dense(64)(z)
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
            print([j] * 50)
            xr[((y - target) ** 2).mean()] = ensemble_model


        ensemble_model = xr[min(xr, key=lambda x: x)]
        x_test = np.arange(46, 76, 1)
        y_test = ensemble_model.predict(x_test)
        y_test = np.where(y_test >= 0, y_test, 0)
        def judge(y_test):
            for i in range(0,15,3):
                if(y_test[i][0]<y_test[i+15][0]):
                    return True
            return False
        if(not judge(y_test)):#max(y_test)[0]< 1.3*max(target)
            em=False
        elif(count>=5):
            cant_list.append(i)
            break
        else:
            xr.clear()
            count+=1
        print(cant_list * 10)









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
    plt.savefig("F:\emsenble_slight_{}_v2\{}城{}区{}拟合.png".format(city, city, str(i), str(j)))

    if ( slight_list_E.index(i)== 0):#改
        result_final = pd.DataFrame(result, index=[i] * result.shape[0], columns=["天数", "感染人数"])
        result_final.to_csv("predict_bei_slight_{}_v2.csv".format(city), columns=["天数", "感染人数"])
    else:
        result_final = pd.read_csv("predict_bei_slight_{}_v2.csv".format(city), names=["天数", "感染人数"])
        result = pd.DataFrame(result, index=[i] * result.shape[0], columns=["天数", "感染人数"])
        result_final = pd.concat([result_final, result])
        result_final.to_csv("predict_bei_slight_{}_v2.csv".format(city), columns=["天数", "感染人数"])
print(cant_list*10)