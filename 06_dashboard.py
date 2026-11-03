
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# 1. CAU HINH
# ============================================================

st.set_page_config(
    page_title="Chess Games Analytics",
    page_icon="♟️",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_PATH = Path("output/processed_dataset.csv")

RESULT_ORDER = ["White", "Black", "Draw", "Unknown"]

RESULT_LABELS = {
    "White": "White thắng",
    "Black": "Black thắng",
    "Draw": "Hòa",
    "Unknown": "Chưa xác định",
}

COLORS = {
    "White": "#3B82F6",
    "Black": "#F97316",
    "Draw": "#10B981",
    "Unknown": "#94A3B8",
}

st.markdown(
    """
    <style>
    .stApp {
        background: #F5F7FB;
    }

    .block-container {
        max-width: 1450px;
        padding-top: 1.4rem;
        padding-bottom: 2rem;
    }

    section[data-testid="stSidebar"] {
        background: #FFFFFF;
        border-right: 1px solid #E2E8F0;
    }

    .hero {
        padding: 25px 28px;
        margin-bottom: 22px;
        border-radius: 16px;
        background: linear-gradient(115deg, #102A43, #1D4E89, #3B82F6);
    }

    .hero h1 {
        color: #FFFFFF !important;
        margin: 0 0 8px 0;
    }

    .hero p {
        color: #EAF2FF !important;
        margin: 0;
        font-size: 15px;
    }

    [data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 15px;
    }

    [data-testid="stPlotlyChart"] {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 4px;
    }

    .section-label {
        color: #64748B;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: -10px;
    }

    div[data-testid="stDataFrame"] {
        background: white;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>♟ Chess Games Analytics</h1>
        <p>
            Khám phá kết quả ván đấu, trình độ người chơi,
            độ dài ván cờ và mối quan hệ giữa các biến.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 2. DOC DU LIEU
# ============================================================

if not DATA_PATH.exists():
    st.error(
        f"Không tìm thấy {DATA_PATH}. "
        "Hãy chạy file 03_feature_engineering.py trước."
    )
    st.stop()

try:
    df = pd.read_csv(DATA_PATH)
except Exception as error:
    st.error(f"Không đọc được dataset: {error}")
    st.stop()

required_columns = [
    "result",
    "avg_rating",
    "rating_diff",
    "total_moves",
]

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    st.error(f"Dataset thiếu các cột: {missing_columns}")
    st.stop()

for col in ["avg_rating", "rating_diff", "total_moves"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["result"] = df["result"].fillna("Unknown").astype(str)

df.loc[
    ~df["result"].isin(RESULT_ORDER),
    "result"
] = "Unknown"

df["result_label"] = df["result"].map(RESULT_LABELS)

if df.empty:
    st.error("Dataset không có dữ liệu.")
    st.stop()


# ============================================================
# 3. BO LOC
# ============================================================

st.sidebar.title("Bộ lọc dữ liệu")
st.sidebar.caption("Các biểu đồ sẽ cập nhật theo bộ lọc.")

selected_results = st.sidebar.multiselect(
    "Kết quả ván đấu",
    options=RESULT_ORDER,
    default=RESULT_ORDER,
    format_func=lambda x: RESULT_LABELS[x],
)

rating_values = df["avg_rating"].dropna()
move_values = df["total_moves"].dropna()

if rating_values.empty or move_values.empty:
    st.error("Không đủ dữ liệu rating hoặc số lượt để tạo dashboard.")
    st.stop()

rating_min = float(rating_values.min())
rating_max = float(rating_values.max())
move_min = int(move_values.min())
move_max = int(move_values.max())

if rating_min < rating_max:
    rating_range = st.sidebar.slider(
        "Rating trung bình",
        min_value=rating_min,
        max_value=rating_max,
        value=(rating_min, rating_max),
        step=10.0,
    )
else:
    rating_range = (rating_min, rating_max)

if move_min < move_max:
    move_range = st.sidebar.slider(
        "Số lượt trong ván",
        min_value=move_min,
        max_value=move_max,
        value=(move_min, move_max),
        step=1,
    )
else:
    move_range = (move_min, move_max)

if st.sidebar.button("Đặt lại bộ lọc", width="stretch"):
    st.rerun()

# Loc ket qua chung.
result_df = df[df["result"].isin(selected_results)].copy()

# Loc rating rieng cho cac bieu do su dung rating.
rating_df = result_df[
    result_df["avg_rating"].between(
        rating_range[0], rating_range[1]
    )
].copy()

# Loc so luot rieng cho cac bieu do su dung total_moves.
moves_df = result_df[
    result_df["total_moves"].between(
        move_range[0], move_range[1]
    )
].copy()

# Scatter plot can ca hai bien hop le.
scatter_df = result_df[
    result_df["avg_rating"].between(
        rating_range[0], rating_range[1]
    )
    & result_df["total_moves"].between(
        move_range[0], move_range[1]
    )
].dropna(subset=["avg_rating", "total_moves"]).copy()

if result_df.empty:
    st.warning("Không có dữ liệu phù hợp với bộ lọc.")
    st.stop()


# ============================================================
# 4. HAM DINH DANG BIEU DO
# ============================================================

def style_chart(fig, height=390):
    fig.update_layout(
        template="plotly_white",
        height=height,
        autosize=True,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(
            family="Arial, sans-serif",
            size=12,
            color="#334155",
        ),
        margin=dict(l=55, r=25, t=65, b=60),
        title=dict(
            font=dict(size=16, color="#172033"),
            x=0.02,
            xanchor="left",
        ),
        legend=dict(
            title_text="",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
        ),
    )

    fig.update_xaxes(
        showgrid=False,
        linecolor="#CBD5E1",
        automargin=True,
    )

    fig.update_yaxes(
        gridcolor="#E2E8F0",
        zerolinecolor="#CBD5E1",
        automargin=True,
    )

    return fig


def show_chart(fig, height=390):
    style_chart(fig, height)
    st.plotly_chart(
        fig,
        width="stretch",
        config={
            "displaylogo": False,
            "scrollZoom": False,
            "responsive": True,
        },
    )


def add_vertical_stat(fig, value, color, dash, label):
    fig.add_vline(
        x=value,
        line_color=color,
        line_dash=dash,
        line_width=2,
    )

    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            name=label,
            line=dict(
                color=color,
                dash=dash,
                width=2,
            ),
            hoverinfo="skip",
        )
    )


# ============================================================
# 5. KPI
# ============================================================

st.markdown(
    '<p class="section-label">Tổng quan</p>',
    unsafe_allow_html=True,
)
st.subheader("Các chỉ số chính")

total_games = len(result_df)

white_wins = int((result_df["result"] == "White").sum())
black_wins = int((result_df["result"] == "Black").sum())
draws = int((result_df["result"] == "Draw").sum())
unknown = int((result_df["result"] == "Unknown").sum())

avg_rating = result_df["avg_rating"].mean()
avg_moves = result_df["total_moves"].mean()

kpi = st.columns(4, gap="medium")

kpi[0].metric("Tổng số ván", f"{total_games:,}")
kpi[1].metric(
    "White thắng",
    f"{white_wins / total_games:.1%}" if total_games else "0%",
)
kpi[2].metric(
    "Black thắng",
    f"{black_wins / total_games:.1%}" if total_games else "0%",
)
kpi[3].metric(
    "Hòa",
    f"{draws / total_games:.1%}" if total_games else "0%",
)

kpi = st.columns(4, gap="medium")

kpi[0].metric(
    "Rating trung bình",
    f"{avg_rating:,.1f}" if pd.notna(avg_rating) else "N/A",
)
kpi[1].metric(
    "Số lượt trung bình",
    f"{avg_moves:.1f}" if pd.notna(avg_moves) else "N/A",
)
kpi[2].metric("Số ván hòa", f"{draws:,}")
kpi[3].metric("Chưa xác định", f"{unknown:,}")

st.caption(
    f"Dữ liệu sau lọc: {total_games:,} ván. "
    f"Dữ liệu gốc: {len(df):,} ván."
)

st.divider()


# ============================================================
# 6. BIEU DO 1 - PHAN BO KET QUA
# ============================================================

st.markdown(
    '<p class="section-label">01 · Kết quả</p>',
    unsafe_allow_html=True,
)
st.subheader("Phân bố kết quả ván đấu")

counts = (
    result_df["result"]
    .value_counts()
    .reindex(RESULT_ORDER, fill_value=0)
)

fig1 = go.Figure()

fig1.add_trace(
    go.Bar(
        x=[RESULT_LABELS[r] for r in RESULT_ORDER],
        y=[int(counts[r]) for r in RESULT_ORDER],
        marker=dict(
            color=[COLORS[r] for r in RESULT_ORDER],
            line=dict(color="white", width=1.2),
        ),
        text=[f"{int(counts[r]):,}" for r in RESULT_ORDER],
        textposition="outside",
        cliponaxis=False,
        hovertemplate="Kết quả: %{x}<br>Số ván: %{y:,}<extra></extra>",
    )
)

fig1.update_layout(
    title="Số lượng ván theo kết quả",
    xaxis_title="Kết quả ván đấu",
    yaxis_title="Số lượng ván",
    showlegend=False,
    bargap=0.25,
)

fig1.update_yaxes(rangemode="tozero")

show_chart(fig1, 350)


# ============================================================
# 7. BIEU DO 2 VA 3
# ============================================================

left, right = st.columns(2, gap="large")

with left:
    st.markdown(
        '<p class="section-label">02 · Trình độ người chơi</p>',
        unsafe_allow_html=True,
    )
    st.subheader("Phân bố rating trung bình")

    values = rating_df["avg_rating"].dropna()
    fig2 = go.Figure()

    fig2.add_trace(
        go.Histogram(
            x=values,
            nbinsx=35,
            name="Số ván",
            marker=dict(
                color=COLORS["White"],
                line=dict(color="white", width=1),
            ),
            hovertemplate="Rating: %{x:.0f}<br>Số ván: %{y:,}<extra></extra>",
        )
    )

    if not values.empty:
        mean_value = values.mean()
        median_value = values.median()

        add_vertical_stat(
            fig2,
            mean_value,
            "#EA580C",
            "dash",
            f"Trung bình: {mean_value:.0f}",
        )
        add_vertical_stat(
            fig2,
            median_value,
            "#0F766E",
            "dot",
            f"Trung vị: {median_value:.0f}",
        )

    fig2.update_layout(
        title="Phân bố rating trung bình của các ván đấu",
        xaxis_title="Rating trung bình",
        yaxis_title="Số lượng ván đấu",
        bargap=0.04,
    )

    show_chart(fig2, 410)


with right:
    st.markdown(
        '<p class="section-label">03 · Chênh lệch rating</p>',
        unsafe_allow_html=True,
    )
    st.subheader("Rating theo kết quả ván đấu")

    box_df = rating_df[
        rating_df["result"].isin(["White", "Black", "Draw"])
    ].dropna(subset=["rating_diff"])

    fig3 = go.Figure()

    for result in ["White", "Black", "Draw"]:
        values = box_df.loc[
            box_df["result"] == result,
            "rating_diff",
        ]

        if values.empty:
            continue

        fig3.add_trace(
            go.Box(
                y=values,
                name=RESULT_LABELS[result],
                marker_color=COLORS[result],
                line=dict(color=COLORS[result]),
                fillcolor=COLORS[result],
                opacity=0.8,
                boxpoints=False,
                hovertemplate="Chênh lệch rating: %{y:.0f}<extra></extra>",
            )
        )

    fig3.add_hline(
        y=0,
        line_dash="dash",
        line_color="#64748B",
        line_width=1.5,
        annotation_text="Chênh lệch bằng 0",
        annotation_position="top left",
    )

    fig3.update_layout(
        title="Chênh lệch rating theo kết quả",
        xaxis_title="Kết quả",
        yaxis_title="Rating (White - Black)",
        showlegend=False,
    )

    show_chart(fig3, 410)


st.divider()


# ============================================================
# 8. BIEU DO 4 VA 5
# ============================================================

left, right = st.columns(2, gap="large")

with left:
    st.markdown(
        '<p class="section-label">04 · Độ dài ván</p>',
        unsafe_allow_html=True,
    )
    st.subheader("Phân bố số lượt trong các ván cờ")

    values = moves_df["total_moves"].dropna()
    fig4 = go.Figure()

    fig4.add_trace(
        go.Histogram(
            x=values,
            nbinsx=40,
            name="Số ván",
            marker=dict(
                color=COLORS["Draw"],
                line=dict(color="white", width=1),
            ),
            hovertemplate="Số lượt: %{x:.0f}<br>Số ván: %{y:,}<extra></extra>",
        )
    )

    if not values.empty:
        median_value = values.median()
        mean_value = values.mean()

        add_vertical_stat(
            fig4,
            median_value,
            "#0F766E",
            "dash",
            f"Trung vị: {median_value:.0f}",
        )
        add_vertical_stat(
            fig4,
            mean_value,
            "#EA580C",
            "dot",
            f"Trung bình: {mean_value:.1f}",
        )

    fig4.update_layout(
        title="Phân bố số lượt trong các ván cờ",
        xaxis_title="Số lượt",
        yaxis_title="Số lượng ván",
        bargap=0.04,
    )

    show_chart(fig4, 410)


with right:
    st.markdown(
        '<p class="section-label">05 · So sánh kết quả</p>',
        unsafe_allow_html=True,
    )
    st.subheader("Số lượt theo kết quả")

    box_df = moves_df[
        moves_df["result"].isin(["White", "Black", "Draw"])
    ].dropna(subset=["total_moves"])

    fig5 = go.Figure()

    for result in ["White", "Black", "Draw"]:
        values = box_df.loc[
            box_df["result"] == result,
            "total_moves",
        ]

        if values.empty:
            continue

        fig5.add_trace(
            go.Box(
                y=values,
                name=RESULT_LABELS[result],
                marker_color=COLORS[result],
                line=dict(color=COLORS[result]),
                fillcolor=COLORS[result],
                opacity=0.8,
                boxpoints=False,
                hovertemplate="Số lượt: %{y:.0f}<extra></extra>",
            )
        )

    fig5.update_layout(
        title="Độ dài ván cờ theo kết quả",
        xaxis_title="Kết quả ván đấu",
        yaxis_title="Số lượt",
        showlegend=False,
    )

    show_chart(fig5, 410)


st.divider()


# ============================================================
# 9. BIEU DO 6 - RATING VA SO LUOT
# ============================================================

st.markdown(
    '<p class="section-label">06 · Mối quan hệ</p>',
    unsafe_allow_html=True,
)
st.subheader("Rating và độ dài ván")

# Giong file 04: toi da 25.000 diem, random_state co dinh.
sampled = len(scatter_df) > 25000

if sampled:
    scatter_plot_df = scatter_df.sample(
        n=25000,
        random_state=42,
    )
else:
    scatter_plot_df = scatter_df

fig6 = go.Figure()

for result in RESULT_ORDER:
    subset = scatter_plot_df[
        scatter_plot_df["result"] == result
    ]

    if subset.empty:
        continue

    fig6.add_trace(
        go.Scattergl(
            x=subset["avg_rating"],
            y=subset["total_moves"],
            mode="markers",
            name=RESULT_LABELS[result],
            marker=dict(
                color=COLORS[result],
                size=5,
                opacity=0.42,
            ),
            customdata=subset[["result_label"]].to_numpy(),
            hovertemplate=(
                "Kết quả: %{customdata[0]}<br>"
                "Rating trung bình: %{x:.0f}<br>"
                "Số lượt: %{y:.0f}<extra></extra>"
            ),
        )
    )

fig6.update_layout(
    title="Mối quan hệ giữa rating trung bình và độ dài ván",
    xaxis_title="Rating trung bình",
    yaxis_title="Số lượt",
)

show_chart(fig6, 440)

if sampled:
    st.caption(
        "Biểu đồ phân tán hiển thị mẫu ngẫu nhiên 25.000 ván "
        "để cải thiện tốc độ. Các thống kê khác sử dụng dữ liệu "
        "được lọc đầy đủ."
    )
else:
    st.caption(
        f"Biểu đồ hiển thị {len(scatter_plot_df):,} ván có đủ "
        "rating và số lượt trong phạm vi lọc."
    )


# ============================================================
# 10. BANG THONG KE
# ============================================================

st.divider()

st.markdown(
    '<p class="section-label">Dữ liệu chi tiết</p>',
    unsafe_allow_html=True,
)
st.subheader("Bảng thống kê theo kết quả")

summary_rows = []

for result in RESULT_ORDER:
    subset = result_df[result_df["result"] == result]

    if subset.empty:
        continue

    summary_rows.append({
        "Kết quả": RESULT_LABELS[result],
        "Số ván": len(subset),
        "Rating trung bình": subset["avg_rating"].mean(),
        "Số lượt trung bình": subset["total_moves"].mean(),
        "Số lượt trung vị": subset["total_moves"].median(),
    })

summary_df = pd.DataFrame(summary_rows)

for col in [
    "Rating trung bình",
    "Số lượt trung bình",
    "Số lượt trung vị",
]:
    summary_df[col] = summary_df[col].round(2)

st.dataframe(
    summary_df,
    width="stretch",
    hide_index=True,
)

st.download_button(
    label="Tải dữ liệu đang lọc (CSV)",
    data=result_df.to_csv(index=False).encode("utf-8-sig"),
    file_name="filtered_chess_games.csv",
    mime="text/csv",
)

st.caption(
    "Nguồn: output/processed_dataset.csv. "
    "Các biểu đồ sử dụng cùng quy ước màu cho từng kết quả. "
    "Histogram Plotly có thể chia khoảng khác Matplotlib trong file 04."
)
