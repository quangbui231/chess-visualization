
import os
import re

import pandas as pd


# ============================================================
# 1. CẤU HÌNH VÀ ĐỌC DATASET
# ============================================================

input_path = "output/cleaned_dataset.csv"
output_path = "output/processed_dataset.csv"

df = pd.read_csv(input_path)

print("===== DATASET BAN ĐẦU =====")
print("Số dòng:", len(df))
print("Số cột:", len(df.columns))


# ============================================================
# 2. KIỂM TRA CÁC CỘT CẦN THIẾT
# ============================================================

required_columns = [
    "white_rating",
    "black_rating",
    "winner",
    "status",
    "pgn",
    "game_id",
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Dataset thiếu các cột bắt buộc: {missing_columns}"
    )


# ============================================================
# 3. CHUẨN HÓA RATING
# ============================================================

for col in ["white_rating", "black_rating"]:
    original = df[col]

    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

    invalid_count = (
        original.notna() & df[col].isna()
    ).sum()

    print(
        f"Số giá trị không hợp lệ ở {col}: "
        f"{invalid_count}"
    )


# ============================================================
# 4. FEATURE ENGINEERING: RATING
# ============================================================

# Rating của Trắng trừ rating của Đen.
df["rating_diff"] = (
    df["white_rating"] - df["black_rating"]
)

# Rating trung bình của hai người chơi.
df["avg_rating"] = (
    df["white_rating"] + df["black_rating"]
) / 2


# ============================================================
# 5. FEATURE ENGINEERING: RESULT
# ============================================================

winner = (
    df["winner"]
    .astype("string")
    .str.strip()
    .str.lower()
)

status = (
    df["status"]
    .astype("string")
    .str.strip()
    .str.lower()
)

print("\n===== PHÂN BỐ WINNER =====")
print(winner.value_counts(dropna=False))

print("\n===== PHÂN BỐ STATUS =====")
print(status.value_counts(dropna=False))

df["result"] = "Unknown"

df.loc[winner.eq("white"), "result"] = "White"
df.loc[winner.eq("black"), "result"] = "Black"

# Chỉ xác định hòa khi winner bị thiếu và status
# thuộc các trạng thái hòa đã xác định.
draw_statuses = {"draw", "stalemate"}

df.loc[
    winner.isna() & status.isin(draw_statuses),
    "result"
] = "Draw"


# ============================================================
# 6. ĐẾM NƯỚC ĐI TỪ PGN - PHIÊN BẢN NHANH
# ============================================================
# Không dựng bàn cờ, không gọi read_game cho từng ván.
#
# Các bước:
# - Bỏ header PGN.
# - Bỏ comment trong {...} và biến thể trong (...).
# - Bỏ số thứ tự nước đi.
# - Bỏ annotation và ký hiệu kết thúc ván.
# - Đếm các token nước đi còn lại.
#
# Lưu ý:
# Đây là cách đếm nhanh cho PGN tiêu chuẩn.
# Các ván có ký hiệu thả quân như N@e6 được để trống
# vì dataset có thể chứa biến thể Crazyhouse.
# Các PGN rỗng hoặc không thể phân tích sơ bộ cũng để trống.


DROP_MOVE_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])[KQRBNP]@[a-h][1-8]"
)

HEADER_PATTERN = re.compile(
    r'(?m)^\s*\[[A-Za-z0-9_]+\s+"(?:[^"\\]|\\.)*"\]\s*$'
)

COMMENT_PATTERN = re.compile(
    r"\{[^}]*\}|;[^\r\n]*"
)

MOVE_NUMBER_PATTERN = re.compile(
    r"(?<!\S)\d+\.(?:\.\.)?"
)

RESULT_PATTERN = re.compile(
    r"^(1-0|0-1|1/2-1/2|\*)$"
)

# SAN thông dụng của cờ vua tiêu chuẩn:
# e4, Nf3, Bxc6, Qh5+, O-O, O-O-O, e8=Q...
SAN_PATTERN = re.compile(
    r"^(?:"
    r"O-O-O[+#]?|O-O[+#]?|"
    r"[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8]"
    r"(?:=[QRBN])?[+#]?"
    r")$"
)


def count_moves_fast(pgn_text):

    if pd.isna(pgn_text):
        return pd.NA

    text = str(pgn_text).strip()

    if not text:
        return pd.NA

    # Không dùng parser cờ tiêu chuẩn để đếm biến thể
    # có nước thả quân.
    if DROP_MOVE_PATTERN.search(text):
        return pd.NA

    # Loại header PGN.
    text = HEADER_PATTERN.sub("", text)

    # Loại comment và comment cuối dòng.
    text = COMMENT_PATTERN.sub(" ", text)

    # Loại các biến thể đặt trong ngoặc.
    # Lặp nhiều lần để xử lý ngoặc lồng nhau ở mức cơ bản.
    previous_text = None

    while previous_text != text:
        previous_text = text
        text = re.sub(r"\([^()]*\)", " ", text)

    # Tách token.
    tokens = text.split()

    move_tokens = []

    for token in tokens:

        # Bỏ số thứ tự nước đi, kể cả dạng 1.e4.
        token = MOVE_NUMBER_PATTERN.sub("", token)

        token = token.strip()

        if not token:
            continue

        # Bỏ kết quả ván đấu.
        if RESULT_PATTERN.fullmatch(token):
            continue

        # Bỏ annotation độc lập.
        if token in {"!", "?", "!!", "??", "!?", "?!"}:
            continue

        # Bỏ annotation gắn sau nước đi, ví dụ e4! hoặc Nf3?!
        token = re.sub(r"[!?]+$", "", token)

        # Bỏ ký hiệu check/mate để kiểm tra dạng SAN.
        san_for_check = token.rstrip("+#")

        if SAN_PATTERN.fullmatch(token) or SAN_PATTERN.fullmatch(
            san_for_check
        ):
            move_tokens.append(token)

    if not move_tokens:
        return pd.NA

    # Đếm số ply, sau đó chuyển thành số nước đi hoàn chỉnh.
    ply_count = len(move_tokens)

    return (ply_count + 1) // 2


print("\nĐang đếm nước đi từ PGN bằng phương pháp nhanh...")

df["total_moves"] = pd.array(
    [
        count_moves_fast(value)
        for value in df["pgn"]
    ],
    dtype="Int64"
)


# ============================================================
# 7. KIỂM TRA CÁC BIẾN MỚI
# ============================================================

print("\n===== 10 DÒNG ĐẦU SAU FEATURE ENGINEERING =====")

columns_to_show = [
    "game_id",
    "white_rating",
    "black_rating",
    "rating_diff",
    "avg_rating",
    "winner",
    "status",
    "result",
    "total_moves",
]

print(
    df[columns_to_show]
    .head(10)
    .to_string(index=False)
)


# ============================================================
# 8. THỐNG KÊ CÁC BIẾN
# ============================================================

print("\n===== THỐNG KÊ RATING_DIFF =====")
print(df["rating_diff"].describe())

print("\n===== THỐNG KÊ AVG_RATING =====")
print(df["avg_rating"].describe())

print("\n===== THỐNG KÊ TOTAL_MOVES =====")
print(df["total_moves"].describe())

print("\n===== PHÂN BỐ RESULT =====")
print(df["result"].value_counts(dropna=False))


# ============================================================
# 9. KIỂM TRA GIÁ TRỊ THIẾU
# ============================================================

print("\n===== GIÁ TRỊ THIẾU =====")

new_columns = [
    "rating_diff",
    "avg_rating",
    "result",
    "total_moves",
]

print(df[new_columns].isna().sum())


# ============================================================
# 10. KIỂM TRA VÁN CỜ BẤT THƯỜNG
# ============================================================

print("\n===== VÁN CÓ TOTAL_MOVES <= 2 =====")

short_games = df.loc[
    df["total_moves"].le(2).fillna(False),
    ["game_id", "status", "total_moves"],
]

print(short_games.head(20).to_string(index=False))


print("\n===== VÁN CÓ TOTAL_MOVES >= 150 =====")

long_games = df.loc[
    df["total_moves"].ge(150).fillna(False),
    ["game_id", "status", "total_moves"],
]

print(long_games.head(10).to_string(index=False))


# ============================================================
# 11. LƯU DATASET
# ============================================================

os.makedirs(
    os.path.dirname(output_path),
    exist_ok=True
)

df.to_csv(
    output_path,
    index=False
)

print("\n===== HOÀN TẤT =====")
print("Số dòng:", len(df))
print("Số cột:", len(df.columns))
print(
    "Số total_moves bị thiếu:",
    int(df["total_moves"].isna().sum())
)
print("Dataset đã lưu tại:", output_path)
