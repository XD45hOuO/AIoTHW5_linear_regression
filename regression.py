# ==========================================
# Phase 1: 環境準備與資料載入
# ==========================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# 讀取資料 (此資料集常帶有特殊字元，使用 latin-1 編碼讀取)
df = pd.read_csv('laptop_price.csv', encoding='latin-1')

# 移除不必要的 ID 欄位
if 'laptop_ID' in df.columns:
    df = df.drop(columns=['laptop_ID'])

# ==========================================
# Phase 2: 資料準備 (Data Preparation)
# ==========================================
# 1. 數值特徵清理：將 Ram 的 'GB' 與 Weight 的 'kg' 移除，並轉換為數值型態
df['Ram'] = df['Ram'].str.replace('GB', '').astype(int)
df['Weight'] = df['Weight'].str.replace('kg', '').astype(float)

# 2. 選擇特徵：為了示範，我們挑選最具代表性的 5 個特徵 (2個數值, 3個類別)
# 在實際 CRISP-DM 中，你可以進一步萃取 CPU 時脈與螢幕解析度 (PPI)
features = ['Company', 'TypeName', 'Ram', 'Weight', 'OpSys']
X = df[features]
y = df['Price_euros'] # 目標變數：價格 (歐元)

# 3. 類別特徵處理：使用 One-Hot Encoding 轉換文字標籤
# drop_first=True 可以避免虛擬變數陷阱 (Dummy Variable Trap)，解決完全共線性問題
X = pd.get_dummies(X, drop_first=True)

# ==========================================
# Phase 3: 模型建立 (Modeling)
# ==========================================
# 1. 切分訓練集與測試集 (80% 訓練, 20% 測試)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. 初始化並訓練多元線性回歸模型
model = LinearRegression()
model.fit(X_train, y_train)

# ==========================================
# Phase 4: 模型評估 (Evaluation)
# ==========================================
# 1. 預測測試集
y_pred = model.predict(X_test)

# 2. 計算評估指標
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"📊 模型評估結果:")
print(f"R-squared (決定係數): {r2:.4f}")
print(f"MAE (平均絕對誤差): {mae:.2f} 歐元")

# 3. 視覺化：實際價格 vs 預測價格
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2) # 完美的 45 度線
plt.xlabel("Actual Price (Euros)")
plt.ylabel("Predicted Price (Euros)")
plt.title("Actual vs Predicted Laptop Prices")
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()