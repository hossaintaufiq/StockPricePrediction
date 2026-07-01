import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor

# Load the prepared dataset
try:
    df = pd.read_csv('stock_news.csv', parse_dates=['Date'], index_col=['Date'])
except FileNotFoundError:
    raise SystemExit('stock_news.csv not found. Run the notebook preprocessing cells first.')

# Match the notebook preprocessing style
sc = MinMaxScaler()
new_df = pd.DataFrame(sc.fit_transform(df))
new_df.columns = df.columns
new_df.index = df.index

X = new_df.drop('Close_stationary', axis=1)
X.drop('Close', axis=1, inplace=True)
y = new_df['Close']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

models = [
    ('RandomForest', RandomForestRegressor(random_state=42)),
    ('AdaBoost', AdaBoostRegressor(random_state=42)),
    ('DecisionTree', DecisionTreeRegressor(random_state=42)),
]

for name, model in models:
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    mae = mean_absolute_error(pred, y_test)
    print(f'{name}: MAE={mae:.6f}')
