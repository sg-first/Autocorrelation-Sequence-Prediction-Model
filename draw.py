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
input=np.arange(1,45)
input=(input-input.mean())/input.std()
print(input)
model__1 = load_model("toy_v5_zt.h5")
output=model__1.predict(input)
print(output)
plt.figure()
plt.plot(input,output)
plt.show()
