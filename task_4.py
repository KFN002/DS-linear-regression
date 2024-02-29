import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import OneHotEncoder


data = pd.read_csv('tg_4_.csv', sep=';', encoding='cp1251')
data = data.dropna()


def replace_outliers(column):
    outliers = column > (column.quantile(0.75) + (column.quantile(0.75) - column.quantile(0.25)) * 3)
    column[outliers] = column.median()
    return column


data[["эке", "спо"]] = data[["эке", "спо"]].apply(lambda x: x.str.replace(',', '.').astype(float))
data["удой"] = data.groupby(["порода"], group_keys=False)['удой'].transform(replace_outliers)
data['порода'] = data['порода'].replace({'РефлешнСоверинггггг': 'РефлешнСоверинг'})
data['спо_кат'] = (data['спо'] > 0.9).astype(int)
x = data[['эке', 'протеин', 'порода', 'спо_кат']]
y = data['удой']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=42)
ohe = OneHotEncoder(sparse_output=False, drop='first')
ohe.fit(x_train[['порода', 'спо_кат']])
x_train_cat = pd.DataFrame(ohe.transform(x_train[['порода', 'спо_кат']]), columns=ohe.get_feature_names_out())
x_test_cat = pd.DataFrame(ohe.transform(x_test[['порода', 'спо_кат']]), columns=ohe.get_feature_names_out())
x_train = pd.concat([x_train.reset_index(drop=True), x_train_cat], axis=1).drop(columns=['порода', 'спо_кат'])
x_test = pd.concat([x_test.reset_index(drop=True), x_test_cat], axis=1).drop(columns=['порода', 'спо_кат'])
model = LinearRegression()
model.fit(x_train, y_train)
prediction = model.predict(x_test)
print(round(r2_score(y_test, prediction), 2),
      round(mean_absolute_percentage_error(y_test, prediction) * 100, 2))
