import pandas as pd

winedatac=pd.read_csv('D:/MCA3/wine.csv')

print(winedatac)
print(winedatac.head())
print("shape\n",winedatac.shape)
print("columns\n",winedatac.columns)
print("dtypes\n",winedatac.dtypes)
print("ndim\n",winedatac.ndim)
print("size\n",winedatac.size)


#winedatae=pd.read_excel('D:/MCA3/wine.xls')


print("\nExcel Data:")
print(winedatac)
print(winedatac.head())
