import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

df = pd.read_csv("tempbook.csv")

print(df)

x = df[['color', 'size', 'price']].values

color_le = LabelEncoder()
x[:,0] = color_le.fit_transform(x[:,0])

print(x)

ct = ColumnTransformer(
    [("color", OneHotEncoder(), [0])],
    remainder='passthrough'
)

x = ct.fit_transform(x)

print(x)

print(pd.get_dummies(df[['color','price']]))
 
