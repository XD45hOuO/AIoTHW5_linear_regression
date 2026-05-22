# Laptop Price Prediction — Multiple Linear Regression

使用多元線性回歸模型，依據筆記型電腦的品牌、類型、記憶體、重量與作業系統預測售價（歐元）。

---

## 專案結構

```
AIoTHW3/
├── venv/                    # Python 虛擬環境
├── laptop_price.csv         # 原始資料集
├── train_model.py           # 主程式（資料清理、訓練、評估、視覺化）
├── test_predictions.csv     # 測試集預測結果
├── evaluation_plot.png      # 實際 vs 預測價格散佈圖
├── price_vs_ram.png         # Price vs RAM（含回歸線）
├── price_vs_weight.png      # Price vs Weight（含回歸線）
├── price_vs_company.png     # Price vs 品牌（箱型圖）
├── price_vs_typename.png    # Price vs 產品類型（箱型圖）
├── price_vs_opsys.png       # Price vs 作業系統（箱型圖）
├── requirements.txt         # 套件版本清單
└── README.md
```

---

## 資料集

- **來源**：[Kaggle — Laptop Price](https://www.kaggle.com/datasets/muhammetvarl/laptop-price)
- **檔案**：`laptop_price.csv`（1,303 筆，編碼 latin-1）
- **目標欄位**：`Price_euros`

---

## 環境建置

**建立虛擬環境**

```bash
python -m venv venv
```

**啟動虛擬環境**

```bash
# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

**安裝套件**

```bash
pip install -r requirements.txt
```

---

## 執行方式

```bash
python train_model.py
```

執行後會在當前目錄產生 `test_predictions.csv` 與所有圖表檔案。

---

## CRISP-DM 流程

| 階段 | 內容 |
|------|------|
| 資料理解 | 讀取 CSV，檢視欄位結構 |
| 資料準備 | 移除 `Ram` 的 `GB`、`Weight` 的 `kg` 並轉型；使用 `pd.get_dummies(drop_first=True)` 進行類別編碼 |
| 建模 | scikit-learn `LinearRegression`，80/20 切分（`random_state=42`） |
| 評估 | R-squared、MAE |
| 部署 | 輸出預測結果 CSV 與視覺化圖表 |

**使用特徵**：`Company`、`TypeName`、`Ram`、`Weight`、`OpSys`

---

## 模型結果

| 指標 | 數值 |
|------|------|
| R-squared | 0.6884 |
| MAE | 282.55 歐元 |

模型可解釋約 **68.8%** 的價格變異，平均預測誤差約 **282 歐元**。

---

## 輸出檔案說明

### `test_predictions.csv`

測試集（261 筆）的逐筆預測結果，欄位如下：

| 欄位 | 說明 |
|------|------|
| Company | 品牌 |
| TypeName | 產品類型 |
| Ram | 記憶體（GB） |
| Weight | 重量（kg） |
| OpSys | 作業系統 |
| Actual_Price | 實際價格（歐元） |
| Predicted_Price | 預測價格（歐元） |
| Error | 預測誤差（預測 − 實際） |

### 圖表

| 檔案 | 說明 |
|------|------|
| `evaluation_plot.png` | 實際 vs 預測價格散佈圖，紅色虛線為完美預測參考線 |
| `price_vs_ram.png` | RAM 與價格的散佈圖，含線性回歸線 |
| `price_vs_weight.png` | 重量與價格的散佈圖，含線性回歸線 |
| `price_vs_company.png` | 各品牌價格分布箱型圖（依中位數排序） |
| `price_vs_typename.png` | 各產品類型價格分布箱型圖 |
| `price_vs_opsys.png` | 各作業系統價格分布箱型圖 |
