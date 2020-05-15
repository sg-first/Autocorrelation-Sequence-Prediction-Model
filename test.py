import keras
import numpy as np
import matplotlib.pyplot as plt
toy_v4=keras.models.load_model("toy_v4.h5")
x_label=np.arange(1,46,0.0001)
x=(x_label-x_label.mean())/x_label.std()
y=toy_v4.predict(x)
plt.figure()
plt.plot(x,y)
plt.show()
