import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import OneHotEncoder


data = pd.read_csv('tg_3_.csv', delimiter=';', encoding='cp1251')
x = data[['возраст', 'мощность', 'кузов']]
encoder = OneHotEncoder(sparse_output=False, drop='first')

x_encoded = pd.DataFrame(encoder.fit_transform(x[['кузов']]), columns=encoder.get_feature_names_out(['кузов']))
x = pd.concat([x, x_encoded], axis=1)
x = x.drop(['кузов'], axis=1)
y = data['цена']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=42)
model = LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(round(r2, 2), round(mae, 2))
