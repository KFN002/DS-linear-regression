import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np

data = pd.read_csv('tg_2_.csv', delimiter=';', encoding='cp1251')
x = data[['температура', 'дождь']]
y = data['продажи']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=42)
preprocessor = ColumnTransformer(
    transformers=[
        ('rain_category', OneHotEncoder(), ['дождь'])
    ],
    remainder='passthrough'
)
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
rounded_predictions = np.round(y_pred).astype(float)
flat_predictions = rounded_predictions.flatten()
print(flat_predictions)
