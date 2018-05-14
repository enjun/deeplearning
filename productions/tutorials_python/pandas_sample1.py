import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dates = pd.date_range('20180101', '20180105')
d = pd.DataFrame(data=np.random.randn(5, 4), index=dates, columns=list('ABCD'))
print(d)
print(d.head(2))  # it's fit for acquire some part of big data
print(d.tail(2))

print(d.index)  # panda's standard data include these three main parts
print(d.columns)
print(d.values)

print(d.loc['20180101':'20180103'])  #
print(d.loc[:, ['A', 'B']])
print(d.at[dates[0], 'A'])  # fast access data compare with prior method

print(d.iloc[3:5, 0:1])  # it is identical to slice of python

print('\n')
nd = np.array(d)
print(nd)

days = 12
x = pd.Series(data=np.random.randn(days), index=pd.date_range('20180501', periods=days))
x = x.cumsum()
plt.figure()
x.plot()
# plt.show()

d.to_csv('pandas_temp_data.csv')
d2 = pd.read_csv('pandas_temp_data.csv')
print(d2)