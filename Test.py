import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt

xs = np.array([-1,0,1,2,3,4])
ys = np.array([-3,-1,1,3,5,7])

plt.plot(xs,ys)
plt.show()

# 3 = INFO, WARNING, and ERROR messages are not printed
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = 3

model = tf.keras.Sequential()


model.add(tf.keras.layers.Dense(1,input_dim=1))

model.compile(loss='mean_squared_error',optimizer='sgd')



model.fit(xs,ys,epochs=500)

