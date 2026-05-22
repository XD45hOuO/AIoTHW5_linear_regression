import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# 1. 讀取資料
df = pd.read_csv('laptop_price.csv', encoding='latin-1')

# 2. 資料清理
df['Ram'] = df['Ram'].str.replace('GB', '').astype(int)
df['Weight'] = df['Weight'].str.replace('kg', '').astype(float)

# 3. 特徵選擇
features = ['Company', 'TypeName', 'Ram', 'Weight', 'OpSys']
target = 'Price_euros'

df_model = df[features + [target]].dropna()

# 4. 類別編碼
X = pd.get_dummies(df_model[features], drop_first=True)
y = df_model[target]

# 5. 切分訓練集 / 測試集並訓練模型
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# 6. 模型評估
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f'R-squared: {r2:.4f}')
print(f'MAE: {mae:.2f}')

# 6b. 儲存測試集預測結果為 CSV
results = df_model.iloc[X_test.index][features].copy()
results['Actual_Price'] = y_test.values
results['Predicted_Price'] = y_pred.round(2)
results['Error'] = (results['Predicted_Price'] - results['Actual_Price']).round(2)
results = results.reset_index(drop=True)
results.to_csv('test_predictions.csv', index=False, encoding='utf-8-sig')
print(f'test_predictions.csv saved ({len(results)} rows).')

# 7. 實際價格 vs 預測價格散佈圖
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5, edgecolors='k', linewidths=0.3)
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1.5, label='Perfect Prediction')
plt.xlabel('Actual Price (euros)')
plt.ylabel('Predicted Price (euros)')
plt.title('Actual vs Predicted Laptop Price')
plt.legend()
plt.tight_layout()
plt.savefig('evaluation_plot.png', dpi=150)
plt.close()
print('evaluation_plot.png saved.')

# 8. Price vs 各特徵圖表
cat_features = ['Company', 'TypeName', 'OpSys']
num_features = ['Ram', 'Weight']

# 數值特徵：散佈圖 + 回歸曲線
for feat in num_features:
    fig, ax = plt.subplots(figsize=(7, 5))
    x_vals = df_model[feat]
    y_vals = df_model['Price_euros']
    ax.scatter(x_vals, y_vals, alpha=0.4, edgecolors='k', linewidths=0.2, label='Data')
    # 回歸線
    m, b = np.polyfit(x_vals, y_vals, 1)
    x_line = np.linspace(x_vals.min(), x_vals.max(), 200)
    ax.plot(x_line, m * x_line + b, 'r-', linewidth=2, label=f'Regression (y={m:.1f}x+{b:.0f})')
    ax.set_xlabel(feat)
    ax.set_ylabel('Price (euros)')
    ax.set_title(f'Price vs {feat}')
    ax.legend()
    plt.tight_layout()
    fname = f'price_vs_{feat.lower()}.png'
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f'{fname} saved.')

# 類別特徵：箱型圖
for feat in cat_features:
    order = df_model.groupby(feat)['Price_euros'].median().sort_values(ascending=False).index
    fig, ax = plt.subplots(figsize=(max(8, len(order) * 0.8), 5))
    data_groups = [df_model.loc[df_model[feat] == cat, 'Price_euros'].values for cat in order]
    ax.boxplot(data_groups, tick_labels=order, vert=True)
    ax.set_xlabel(feat)
    ax.set_ylabel('Price (euros)')
    ax.set_title(f'Price vs {feat}')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    fname = f'price_vs_{feat.lower()}.png'
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f'{fname} saved.')
