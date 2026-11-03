
import os

import pandas as pd


# ============================================================
# 1. DOC DATASET VA KIEM TRA DU LIEU
# ============================================================

DATA_PATH = "output/processed_dataset.csv"
OUTPUT_DIR = "output/analysis"

REQUIRED_COLUMNS = [
    "result",
    "rating_diff",
    "avg_rating",
    "total_moves",
]

VALID_RESULTS = ["White", "Black", "Draw", "Unknown"]

if not os.path.isfile(DATA_PATH):
    raise FileNotFoundError(
        f"Không tìm thấy dataset: {DATA_PATH}. "
        "Hãy chạy các bước xử lý dữ liệu trước."
    )

df = pd.read_csv(DATA_PATH)

missing_columns = [
    column
    for column in REQUIRED_COLUMNS
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Dataset thiếu các cột bắt buộc: {missing_columns}"
    )

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("THÔNG TIN DATASET")
print("=" * 60)
print(f"Số dòng: {len(df):,}")
print(f"Số cột: {len(df.columns)}")


# Chuẩn hóa nhãn kết quả để tránh khác biệt do chữ hoa,
# chữ thường hoặc khoảng trắng.

df["result"] = (
    df["result"]
    .astype("string")
    .str.strip()
    .str.lower()
    .map({
        "white": "White",
        "black": "Black",
        "draw": "Draw",
        "unknown": "Unknown",
    })
    .fillna("Unknown")
)

# Chuyển các cột phân tích sang kiểu số.
# Giá trị không hợp lệ sẽ thành NaN.

NUMERIC_COLUMNS = [
    "rating_diff",
    "avg_rating",
    "total_moves",
]

for column in NUMERIC_COLUMNS:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce",
    )

print("\n===== KIỂM TRA SỐ LƯỢNG MẪU =====")

print("\nSố dòng theo result:")
print(
    df["result"]
    .value_counts()
    .reindex(VALID_RESULTS, fill_value=0)
)

print(
    "\nSố rating_diff hợp lệ:",
    int(df["rating_diff"].notna().sum()),
)

print("\nSố dòng theo result có rating_diff hợp lệ:")
print(
    df.groupby("result")["rating_diff"]
    .count()
    .reindex(VALID_RESULTS, fill_value=0)
)

print("\nSố dòng thiếu rating_diff theo result:")
print(
    df[df["rating_diff"].isna()]
    .groupby("result")
    .size()
    .reindex(VALID_RESULTS, fill_value=0)
)


# ============================================================
# 2. PHAN TICH KET QUA VAN DAU
# ============================================================

print("\n===== 1. PHÂN TÍCH KẾT QUẢ =====")

result_counts = (
    df["result"]
    .value_counts()
    .reindex(VALID_RESULTS, fill_value=0)
)

result_percent = (
    result_counts.div(len(df)).mul(100).round(2)
    if len(df) > 0
    else result_counts.astype(float)
)

result_analysis = pd.DataFrame({
    "count": result_counts,
    "percentage": result_percent,
})

print(result_analysis)


# ============================================================
# 3. PHAN TICH RATING VA KET QUA
# ============================================================

print("\n===== 2. PHÂN TÍCH RATING VÀ KẾT QUẢ =====")

rating_data = df[
    df["result"].isin(["White", "Black", "Draw"])
].dropna(
    subset=["rating_diff"]
).copy()

rating_analysis = (
    rating_data
    .groupby("result")["rating_diff"]
    .agg(["count", "mean", "median", "min", "max"])
    .reindex(["White", "Black", "Draw"])
    .round(2)
)

print(rating_analysis)


# ============================================================
# 4. RATING CAO HON VA KET QUA VAN DAU
# ============================================================

print("\n===== 3. RATING CAO HƠN VÀ KẾT QUẢ =====")

# Giả định rating_diff = rating Trắng - rating Đen.
# Kiểm tra lại định nghĩa này trong File 03 trước khi diễn giải.

rating_data["higher_rated_player"] = "Equal"

rating_data.loc[
    rating_data["rating_diff"] > 0,
    "higher_rated_player",
] = "White"

rating_data.loc[
    rating_data["rating_diff"] < 0,
    "higher_rated_player",
] = "Black"


def determine_match_result(row):
    """Phân loại kết quả so với người có rating cao hơn."""

    higher = row["higher_rated_player"]
    result = row["result"]

    if higher == "Equal":
        return "Equal rating"

    if result == "Draw":
        return "Draw"

    if result == higher:
        return "Higher-rated player wins"

    return "Lower-rated player wins"


rating_data["rating_result"] = rating_data.apply(
    determine_match_result,
    axis=1,
)

rating_result_order = [
    "Higher-rated player wins",
    "Lower-rated player wins",
    "Draw",
    "Equal rating",
]

rating_result_counts = (
    rating_data["rating_result"]
    .value_counts()
    .reindex(rating_result_order, fill_value=0)
)

# Tỷ lệ ở đây được tính trên toàn bộ rating_data,
# bao gồm cả nhóm Equal rating.

if len(rating_data) > 0:
    rating_result_percent = (
        rating_result_counts
        .div(len(rating_data))
        .mul(100)
        .round(2)
    )
else:
    rating_result_percent = rating_result_counts.astype(float)

rating_result_analysis = pd.DataFrame({
    "count": rating_result_counts,
    "percentage": rating_result_percent,
})

print(rating_result_analysis)


# ============================================================
# 5. PHAN TICH DO DAI VAN CO
# ============================================================

print("\n===== 4. PHÂN TÍCH ĐỘ DÀI VÁN CỜ =====")

moves_data = df["total_moves"].dropna()

if not moves_data.empty:
    print("\nThống kê tổng quan:")
    print(moves_data.describe().round(2))
else:
    print("Không có dữ liệu total_moves hợp lệ.")


# ============================================================
# 6. DO DAI VAN THEO KET QUA
# ============================================================

print("\n===== 5. ĐỘ DÀI VÁN THEO KẾT QUẢ =====")

moves_by_result_data = df[
    df["result"].isin(["White", "Black", "Draw"])
].dropna(
    subset=["total_moves"]
)

moves_by_result = (
    moves_by_result_data
    .groupby("result")["total_moves"]
    .agg(["count", "mean", "median", "min", "max"])
    .reindex(["White", "Black", "Draw"])
    .round(2)
)

print(moves_by_result)


# ============================================================
# 7. TUONG QUAN RATING VA DO DAI VAN
# ============================================================

print("\n===== 6. TƯƠNG QUAN RATING VÀ ĐỘ DÀI VÁN =====")

correlation_data = df.dropna(
    subset=["avg_rating", "total_moves"]
)

if (
    len(correlation_data) >= 2
    and correlation_data["avg_rating"].nunique() > 1
    and correlation_data["total_moves"].nunique() > 1
):
    correlation = correlation_data["avg_rating"].corr(
        correlation_data["total_moves"],
        method="pearson",
    )
    print("Hệ số tương quan Pearson:", round(correlation, 4))
else:
    correlation = float("nan")
    print(
        "Không đủ dữ liệu hoặc không có biến thiên "
        "để tính tương quan Pearson."
    )


# ============================================================
# 8. TAO BANG TONG HOP
# ============================================================

print("\n===== 7. BẢNG TỔNG HỢP =====")


def safe_mean(series):
    return round(series.mean(), 2) if series.notna().any() else float("nan")


def safe_median(series):
    return round(series.median(), 2) if series.notna().any() else float("nan")


summary = pd.DataFrame({
    "Chỉ số": [
        "Tổng số ván đấu",
        "Số ván White thắng",
        "Số ván Black thắng",
        "Số ván hòa",
        "Số ván không xác định",
        "Rating trung bình",
        "Số nước đi trung bình",
        "Số nước đi trung vị",
        "Tương quan rating - số nước đi",
    ],
    "Giá trị": [
        len(df),
        int((df["result"] == "White").sum()),
        int((df["result"] == "Black").sum()),
        int((df["result"] == "Draw").sum()),
        int((df["result"] == "Unknown").sum()),
        safe_mean(df["avg_rating"]),
        safe_mean(df["total_moves"]),
        safe_median(df["total_moves"]),
        round(correlation, 4) if pd.notna(correlation) else float("nan"),
    ],
})

print(summary.to_string(index=False))


# ============================================================
# 9. LUU CAC KET QUA PHAN TICH
# ============================================================

summary.to_csv(
    os.path.join(OUTPUT_DIR, "summary.csv"),
    index=False,
    encoding="utf-8-sig",
)

result_analysis.to_csv(
    os.path.join(OUTPUT_DIR, "result_analysis.csv"),
    encoding="utf-8-sig",
)

rating_analysis.to_csv(
    os.path.join(OUTPUT_DIR, "rating_analysis.csv"),
    encoding="utf-8-sig",
)

rating_result_analysis.to_csv(
    os.path.join(OUTPUT_DIR, "rating_result_analysis.csv"),
    encoding="utf-8-sig",
)

moves_by_result.to_csv(
    os.path.join(OUTPUT_DIR, "moves_by_result.csv"),
    encoding="utf-8-sig",
)

print("\n===== HOÀN TẤT =====")
print("Các kết quả phân tích đã được lưu tại:")
print(f"{OUTPUT_DIR}/")
print("\nCác tệp đầu ra:")
print("- summary.csv")
print("- result_analysis.csv")
print("- rating_analysis.csv")
print("- rating_result_analysis.csv")
print("- moves_by_result.csv")
