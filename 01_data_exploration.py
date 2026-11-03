import pandas as pd

# Đọc dataset
df = pd.read_csv("data/final_dataset.csv")

# 1. Hiển thị dữ liệu
print("===== 5 DÒNG ĐẦU TIÊN =====")
print(df.head())

# 2. Kích thước
print("\n===== KÍCH THƯỚC DATASET =====")
print("Số dòng:", df.shape[0])
print("Số cột:", df.shape[1])

# 3. Danh sách cột
print("\n===== DANH SÁCH CỘT =====")
print(df.columns.tolist())

# 4. Thông tin dataset
print("\n===== THÔNG TIN DATASET =====")
df.info()

# 5. Thống kê cơ bản
print("\n===== THỐNG KÊ CƠ BẢN =====")
print(df.describe(include="all"))

# 6. Missing values
print("\n===== DỮ LIỆU THIẾU =====")
print(df.isnull().sum())

# 7. Duplicate
print("\n===== DỮ LIỆU TRÙNG =====")
print("Số dòng trùng:", df.duplicated().sum())