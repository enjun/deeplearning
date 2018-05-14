# %% 读图片代码
import matplotlib.pyplot as plt  # plt 用于显示图片
import matplotlib.image as mpimg  # mpimg 用于读取图片
import numpy as np

t = list(range(5))
x = np.sin(t)
y= np.cos(t)
plt.plot(t, x)
plt.plot(t, y)
plt.show()