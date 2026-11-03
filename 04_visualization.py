
import os

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. CAU HINH GIAO DIEN CHUNG
# ============================================================

COLORS = {
    "White": "#3B82F6",
    "Black": "#F97316",
    "Draw": "#10B981",
    "Unknown": "#94A3B8",
}

RESULT_LABELS = {
    "White": "Trắng thắng",
    "Black": "Đen thắng",
    "Draw": "Hòa",
    "Unknown": "Chưa xác định",
}

RESULT_ORDER = ["White", "Black", "Draw", "Unknown"]
RESULT_ORDER_BOX = ["White", "Black", "Draw"]

plt.rcParams.update({
    "figure.figsize": (10, 6),
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#CBD5E1",
    "axes.labelcolor": "#334155",
    "text.color": "#172033",
    "xtick.color": "#475569",
    "ytick.color": "#475569",
    "axes.titlesize": 15,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "font.size": 10,
    "grid.color": "#E2E8F0",
    "grid.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "DejaVu Sans",
    "axes.unicode_minus": False,
    "savefig.dpi": 300,
})


# ============================================================
# 2. DOC DU LIEU VA KIEM TRA COT
# ============================================================

DATA_PATH = "output/processed_dataset.csv"
OUTPUT_DIR = "output/figures"

if not os.path.isfile(DATA_PATH):
    raise FileNotFoundError(
        f"Không tìm thấy tệp dữ liệu: {DATA_PATH}. "
        "Hãy kiểm tra đường dẫn và chạy các bước xử lý dữ liệu trước."
    )

df = pd.read_csv(DATA_PATH)

required_columns = [
    "result",
    "avg_rating",
    "rating_diff",
    "total_moves",
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Dataset thiếu các cột bắt buộc: {missing_columns}"
    )

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 55)
print("THÔNG TIN DATASET")
print("=" * 55)
print(f"Số dòng: {len(df):,}")
print(f"Số cột: {len(df.columns)}")
print(f"Thư mục lưu biểu đồ: {OUTPUT_DIR}")


# Chuẩn hóa nhãn kết quả:
# - Loại bỏ khoảng trắng.
# - Không phân biệt chữ hoa, chữ thường.
# - Giá trị trống hoặc không hợp lệ được gán Unknown.

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

# Chuyển các cột cần phân tích sang dạng số.
# Giá trị không chuyển đổi được sẽ trở thành NaN.

NUMERIC_COLUMNS = [
    "avg_rating",
    "rating_diff",
    "total_moves",
]

for column in NUMERIC_COLUMNS:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce",
    )

print("\nPhân bố kết quả:")
print(
    df["result"]
    .value_counts()
    .reindex(RESULT_ORDER, fill_value=0)
)


# ============================================================
# 3. HAM LUU BIEU DO
# ============================================================

def save_chart(fig, filename):
    """Căn chỉnh, lưu biểu đồ và giải phóng bộ nhớ."""

    for ax in fig.axes:
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=0.8)
        ax.grid(axis="x", visible=False)

    fig.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, filename)

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)

    print(f"Đã lưu: {output_path}")


def plot_boxplot_by_result(
    data,
    value_column,
    title,
    ylabel,
    filename,
    show_zero_line=False,
):
    """
    Vẽ boxplot theo kết quả ván cờ.
    Chỉ hiển thị nhóm có dữ liệu hợp lệ.
    """

    box_data = []
    box_labels = []
    box_colors = []

    for result in RESULT_ORDER_BOX:
        values = data.loc[
            data["result"] == result,
            value_column,
        ].dropna()

        if values.empty:
            continue

        box_data.append(values.to_numpy())
        box_labels.append(RESULT_LABELS[result])
        box_colors.append(COLORS[result])

    if not box_data:
        print(
            f"Cảnh báo: Không có dữ liệu hợp lệ cho biểu đồ "
            f"'{title}'."
        )
        return

    fig, ax = plt.subplots(figsize=(9, 5.8))

    box = ax.boxplot(
        box_data,
        labels=box_labels,
        patch_artist=True,
        showfliers=False,
        widths=0.55,
        medianprops={
            "color": "#172033",
            "linewidth": 2,
        },
        whiskerprops={
            "color": "#64748B",
            "linewidth": 1.3,
        },
        capprops={
            "color": "#64748B",
            "linewidth": 1.3,
        },
        boxprops={
            "linewidth": 1.2,
        },
    )

    for patch, color in zip(box["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)
        patch.set_edgecolor("white")

    if show_zero_line:
        ax.axhline(
            0,
            color="#64748B",
            linestyle="--",
            linewidth=1.3,
            label="Chênh lệch bằng 0",
        )
        ax.legend(frameon=False)

    ax.set_title(title, loc="left", pad=16)
    ax.set_xlabel("Kết quả ván cờ")
    ax.set_ylabel(ylabel)

    save_chart(fig, filename)


# ============================================================
# 4. BIEU DO 1 - PHAN BO KET QUA VAN CO
# ============================================================

result_counts = (
    df["result"]
    .value_counts()
    .reindex(RESULT_ORDER, fill_value=0)
)

fig, ax = plt.subplots(figsize=(9, 5.5))

bars = ax.bar(
    [RESULT_LABELS[result] for result in RESULT_ORDER],
    result_counts.values,
    color=[COLORS[result] for result in RESULT_ORDER],
    width=0.65,
    edgecolor="white",
    linewidth=1.5,
)

ax.bar_label(
    bars,
    labels=[f"{value:,}" for value in result_counts.values],
    padding=5,
    fontsize=9,
)

ax.set_title(
    "Phân bố kết quả các ván cờ trực tuyến",
    loc="left",
    pad=16,
)
ax.set_xlabel("Kết quả ván cờ")
ax.set_ylabel("Số lượng ván cờ")

# Tránh giới hạn trục y bằng 0 khi dataset rỗng.
max_count = int(result_counts.max()) if not result_counts.empty else 0
ax.set_ylim(0, max(1, max_count * 1.14))

save_chart(fig, "01_result_distribution.png")


# ============================================================
# 5. BIEU DO 2 - PHAN BO RATING TRUNG BINH
# ============================================================

rating_data = df["avg_rating"].dropna()

if not rating_data.empty:
    fig, ax = plt.subplots(figsize=(10, 5.8))

    ax.hist(
        rating_data,
        bins=35,
        color="#3B82F6",
        edgecolor="white",
        linewidth=1.0,
        alpha=0.95,
    )

    mean_rating = rating_data.mean()
    median_rating = rating_data.median()

    ax.axvline(
        mean_rating,
        color="#EA580C",
        linestyle="--",
        linewidth=2,
        label=f"Trung bình: {mean_rating:,.0f}",
    )

    ax.axvline(
        median_rating,
        color="#0F766E",
        linestyle=":",
        linewidth=2,
        label=f"Trung vị: {median_rating:,.0f}",
    )

    ax.set_title(
        "Phân bố rating trung bình của các ván cờ",
        loc="left",
        pad=16,
    )
    ax.set_xlabel("Rating trung bình của hai người chơi")
    ax.set_ylabel("Số lượng ván cờ")
    ax.legend(frameon=False)

    save_chart(fig, "02_avg_rating_distribution.png")
else:
    print(
        "Cảnh báo: Không có dữ liệu avg_rating hợp lệ "
        "để vẽ biểu đồ 2."
    )


# ============================================================
# 6. BIEU DO 3 - CHENH LECH RATING THEO KET QUA
# ============================================================

rating_diff_data = df[
    df["result"].isin(RESULT_ORDER_BOX)
].copy()

plot_boxplot_by_result(
    data=rating_diff_data,
    value_column="rating_diff",
    title="Chênh lệch rating theo kết quả ván cờ",
    ylabel="Chênh lệch rating (Trắng − Đen)",
    filename="03_rating_difference_by_result.png",
    show_zero_line=True,
)


# ============================================================
# 7. BIEU DO 4 - PHAN BO SO NUOC DI
# ============================================================

moves_data = df["total_moves"].dropna()

if not moves_data.empty:
    fig, ax = plt.subplots(figsize=(10, 5.8))

    ax.hist(
        moves_data,
        bins=40,
        color="#14B8A6",
        edgecolor="white",
        linewidth=1.0,
        alpha=0.96,
    )

    median_moves = moves_data.median()
    mean_moves = moves_data.mean()

    ax.axvline(
        median_moves,
        color="#0F766E",
        linestyle="--",
        linewidth=2,
        label=f"Trung vị: {median_moves:.0f}",
    )

    ax.axvline(
        mean_moves,
        color="#EA580C",
        linestyle=":",
        linewidth=2,
        label=f"Trung bình: {mean_moves:.1f}",
    )

    ax.set_title(
        "Phân bố số nước đi trong các ván cờ",
        loc="left",
        pad=16,
    )
    ax.set_xlabel("Số nước đi")
    ax.set_ylabel("Số lượng ván cờ")
    ax.legend(frameon=False)

    save_chart(fig, "04_move_distribution.png")
else:
    print(
        "Cảnh báo: Không có dữ liệu total_moves hợp lệ "
        "để vẽ biểu đồ 4."
    )


# ============================================================
# 8. BIEU DO 5 - SO NUOC DI THEO KET QUA
# ============================================================

moves_by_result = df[
    df["result"].isin(RESULT_ORDER_BOX)
].copy()

plot_boxplot_by_result(
    data=moves_by_result,
    value_column="total_moves",
    title="Số nước đi theo kết quả ván cờ",
    ylabel="Số nước đi",
    filename="05_moves_by_result.png",
)


# ============================================================
# 9. BIEU DO 6 - RATING TRUNG BINH VA SO NUOC DI
# ============================================================

# Các cột đã được chuyển sang numeric ở phần đầu.
# Loại bỏ giá trị thiếu sau khi chuyển đổi.

scatter_data = df.dropna(
    subset=["avg_rating", "total_moves"]
).copy()

# Lấy mẫu tối đa 25.000 điểm để giảm tải khi vẽ.
if len(scatter_data) > 25000:
    scatter_data = scatter_data.sample(
        n=25000,
        random_state=42,
    )

if not scatter_data.empty:
    fig, ax = plt.subplots(figsize=(10, 6))

    for result in RESULT_ORDER:
        subset = scatter_data[
            scatter_data["result"] == result
        ]

        if subset.empty:
            continue

        ax.scatter(
            subset["avg_rating"],
            subset["total_moves"],
            color=COLORS[result],
            label=RESULT_LABELS[result],
            alpha=0.4,
            s=17,
            edgecolors="none",
            rasterized=True,
        )

    ax.set_title(
        "Mối liên hệ giữa rating trung bình và số nước đi",
        loc="left",
        pad=16,
    )
    ax.set_xlabel("Rating trung bình của hai người chơi")
    ax.set_ylabel("Số nước đi")

    # Chỉ hiện chú giải khi có nhóm dữ liệu được vẽ.
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(
            title="Kết quả",
            frameon=False,
            ncol=2,
        )

    save_chart(fig, "06_rating_vs_moves.png")
else:
    print(
        "Cảnh báo: Không có dữ liệu hợp lệ cho biểu đồ "
        "rating trung bình và số nước đi."
    )


# ============================================================
# 10. TONG KET CAC BIEU DO
# ============================================================

expected_files = [
    "01_result_distribution.png",
    "02_avg_rating_distribution.png",
    "03_rating_difference_by_result.png",
    "04_move_distribution.png",
    "05_moves_by_result.png",
    "06_rating_vs_moves.png",
]

print("\n" + "=" * 55)
print("KẾT QUẢ TẠO BIỂU ĐỒ")
print("=" * 55)

created_count = 0

for filename in expected_files:
    output_path = os.path.join(OUTPUT_DIR, filename)

    if os.path.isfile(output_path):
        print(f"[OK] {filename}")
        created_count += 1
    else:
        print(f"[CHƯA TẠO] {filename}")

print(f"\nĐã tạo {created_count}/{len(expected_files)} biểu đồ.")
print(f"Thư mục kết quả: {OUTPUT_DIR}")
