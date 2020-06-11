from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras import Input,layers
from keras.models import Model
target="E"
x=np.arange(1,46,1)
infection_B=pd.read_csv("infection_{}.csv".format(target),names=["城市","区域","日期","新增感染人数"])
infection_B=infection_B[["日期","区域","新增感染人数"]]
print(infection_B)
zt=np.argmax(infection_B.groupby(infection_B["区域"])["新增感染人数"].sum())
for i in [0,35]:
    worst_area=infection_B[infection_B["区域"]==i].iloc[:,2:]
    print(worst_area)
    label=np.array(worst_area)
    print(label)
    plt.figure()
    plt.plot(x, label, label="data")
    plt.legend(loc="best")
    plt.savefig("F:\emsenble_{}\{}城{}区图像.png".format(target,target,str(i)))
