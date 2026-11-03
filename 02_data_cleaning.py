import pandas as pd
import os

# ============================================================
# 1. ĐỌC DỮ LIỆU GỐC
# ============================================================

df = pd.read_csv("data/final_dataset.csv")

print("===== THÔNG TIN TRƯỚC KHI LÀM SẠCH =====")
print("Số dòng:", len(df))
print("Số cột:", len(df.columns))


# ============================================================
# 2. CHUẨN HÓA TÊN CỘT
# ============================================================

# Sửa lỗi chính tả: incriment -> increment
df = df.rename(columns={
    "incriment": "increment"
})

print("\n===== TÊN CỘT SAU KHI CHUẨN HÓA =====")
print(df.columns.tolist())


# ============================================================
# 3. XỬ LÝ DÒNG TRÙNG LẶP
# ============================================================

duplicate_count = df.duplicated().sum()

print("\n===== DỮ LIỆU TRÙNG LẶP =====")
print("Số dòng trùng trước khi xử lý:", duplicate_count)

# Xóa các dòng trùng hoàn toàn
df = df.drop_duplicates()

print("Số dòng trùng sau khi xử lý:",
      df.duplicated().sum())


# ============================================================
# 4. KIỂM TRA GIÁ TRỊ RATING BẤT THƯỜNG
# ============================================================

print("\n===== KIỂM TRA RATING =====")

white_invalid = (
    (df["white_rating"] <= 0) |
    (df["white_rating"] > 4000)
).sum()

black_invalid = (
    (df["black_rating"] <= 0) |
    (df["black_rating"] > 4000)
).sum()

print("White rating bất thường:", white_invalid)
print("Black rating bất thường:", black_invalid)

# Không xóa các giá trị rating bị thiếu hoặc bất thường ở đây
# vì cần giữ dữ liệu cho các phân tích khác.


# ============================================================
# 5. KIỂM TRA TIME CONTROL
# ============================================================

print("\n===== TIME CONTROL =====")

print("Số giá trị thiếu:",
      df["time_control"].isna().sum())

print("Số time_control = 0:",
      (df["time_control"] == 0).sum())

print("Giá trị nhỏ nhất:",
      df["time_control"].min())

print("Giá trị lớn nhất:",
      df["time_control"].max())


# ============================================================
# 6. KIỂM TRA INCREMENT
# ============================================================

print("\n===== INCREMENT =====")

print("Số giá trị thiếu:",
      df["increment"].isna().sum())

print("Increment < 0:",
      (df["increment"] < 0).sum())

print("Giá trị nhỏ nhất:",
      df["increment"].min())

print("Giá trị lớn nhất:",
      df["increment"].max())


# ============================================================
# 7. KIỂM TRA PGN
# ============================================================

print("\n===== PGN =====")

print("Số PGN bị thiếu:",
      df["pgn"].isna().sum())


# ============================================================
# 8. KIỂM TRA THÔNG TIN NGƯỜI CHƠI VÀ RATING
# ============================================================

print("\n===== THÔNG TIN NGƯỜI CHƠI =====")

missing_player = df[
    df["white"].isna() |
    df["black"].isna() |
    df["white_rating"].isna() |
    df["black_rating"].isna()
]

print(
    "Số dòng thiếu thông tin người chơi/rating:",
    len(missing_player)
)

print("\nSố lượng thiếu theo từng cột:")

print(
    df[
        ["white", "white_rating",
         "black", "black_rating"]
    ].isna().sum()
)


# ============================================================
# 9. XỬ LÝ GIÁ TRỊ THIẾU
# ============================================================

print("\n===== XỬ LÝ GIÁ TRỊ THIẾU =====")

# Không xóa các dòng có winner bị thiếu.
# Các trường hợp winner thiếu có thể tương ứng với
# các ván hòa hoặc các trạng thái không xác định bên thắng.

# Không xóa các dòng thiếu PGN.
# PGN chỉ cần thiết cho một số phân tích như số nước đi.

# Không điền giá trị giả cho thông tin người chơi/rating.
# Các giá trị thiếu được giữ nguyên dưới dạng NaN.

print("Các giá trị thiếu không cần thiết cho toàn bộ phân tích")
print("được giữ nguyên để tránh làm mất dữ liệu.")


# ============================================================
# 10. KIỂM TRA DATASET SAU KHI LÀM SẠCH
# ============================================================

print("\n===== SAU KHI LÀM SẠCH =====")

print("Số dòng:", len(df))
print("Số cột:", len(df.columns))

print("\nDanh sách cột:")
print(df.columns.tolist())

print("\nSố dòng trùng:",
      df.duplicated().sum())

print("\nDữ liệu thiếu:")
print(df.isnull().sum())


# ============================================================
# 11. TẠO THƯ MỤC OUTPUT
# ============================================================

os.makedirs("output", exist_ok=True)


# ============================================================
# 12. LƯU DATASET SAU KHI LÀM SẠCH
# ============================================================

output_path = "output/cleaned_dataset.csv"

df.to_csv(
    output_path,
    index=False
)

print("\n===== HOÀN TẤT =====")
print("Dataset sau khi làm sạch đã được lưu tại:")
print(output_path)