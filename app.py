import hashlib
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, StandardScaler


APP_TITLE = "SPKC Menu Makanan Sehat dan Ekonomis"
KAGGLE_URL = "https://www.kaggle.com/datasets/ahsanneural/global-food-and-nutrition-database-2026"
SAMPLE_DATA_PATH = Path("sample_data.csv")

BENEFIT_CRITERIA = ["protein", "fiber", "health_score"]
COST_CRITERIA = ["price", "calories", "fat", "sugar", "sodium"]
TOPSIS_CRITERIA = COST_CRITERIA + BENEFIT_CRITERIA

CATEGORY_PRICE_RANGES = {
    "fruit": (4000, 18000),
    "vegetable": (3000, 15000),
    "grain": (2500, 14000),
    "cereal": (2500, 14000),
    "dairy": (6000, 22000),
    "meat": (12000, 45000),
    "poultry": (10000, 35000),
    "seafood": (14000, 55000),
    "fish": (14000, 55000),
    "legume": (3500, 16000),
    "nut": (8000, 35000),
    "snack": (5000, 25000),
    "beverage": (3000, 20000),
    "fast": (12000, 45000),
}
DEFAULT_PRICE_RANGE = (5000, 30000)

COLUMN_CANDIDATES = {
    "food_name": ["food_name", "food", "name", "product_name", "description", "food_description"],
    "food_category": ["food_category", "category", "categories", "food_type", "group", "main_category"],
    "calories": ["calories", "calorie", "energy_kcal", "energy_kcal_100g", "energy-kcal_100g"],
    "protein": ["protein", "protein_g", "proteins", "proteins_100g"],
    "fat": ["fat", "fat_g", "total_fat", "fat_100g"],
    "sugar": ["sugar", "sugar_g", "sugars", "sugars_100g"],
    "fiber": ["fiber", "fiber_g", "dietary_fiber", "fiber_100g"],
    "sodium": ["sodium", "sodium_mg", "sodium_100g"],
    "health_score": ["health_score", "nutrition_score", "nutritional_score", "score"],
    "price": ["price", "price_idr", "harga", "harga_rp"],
    "carbohydrate": ["carbohydrate", "carbohydrates", "carbs", "carbs_g", "carbohydrates_100g"],
    "saturated_fat": ["saturated_fat", "saturated_fat_g", "saturated-fat_100g"],
    "cholesterol": ["cholesterol", "cholesterol_mg"],
    "nova_group": ["nova_group", "nova"],
    "nutri_score": ["nutri_score", "nutriscore_score", "nutriscore_grade", "nutrition_grade"],
}


st.set_page_config(page_title=APP_TITLE, layout="wide", initial_sidebar_state="expanded")


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names into lowercase snake_case."""
    cleaned = df.copy()
    cleaned.columns = [
        re.sub(r"_+", "_", re.sub(r"[^a-zA-Z0-9]+", "_", str(col).strip().lower())).strip("_")
        for col in cleaned.columns
    ]
    return cleaned


def detect_column(df: pd.DataFrame, target: str) -> str | None:
    candidates = COLUMN_CANDIDATES.get(target, [])
    columns = list(df.columns)
    exact_lookup = {col.lower(): col for col in columns}
    for candidate in candidates:
        if candidate in exact_lookup:
            return exact_lookup[candidate]

    for candidate in candidates:
        for column in columns:
            if candidate in column:
                return column
    return None


@st.cache_data(show_spinner=False)
def load_data(uploaded_file) -> tuple[pd.DataFrame | None, str]:
    """Load uploaded CSV or bundled sample data."""
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file), "uploaded"
    if SAMPLE_DATA_PATH.exists():
        return pd.read_csv(SAMPLE_DATA_PATH), "sample"
    return None, "empty"


def stable_random_value(seed_text: str, low: float, high: float) -> float:
    digest = hashlib.md5(seed_text.encode("utf-8")).hexdigest()
    ratio = int(digest[:8], 16) / 0xFFFFFFFF
    return low + ratio * (high - low)


def generate_price_column(df: pd.DataFrame) -> pd.Series:
    """Generate deterministic simulated food prices in IDR."""
    prices = []
    for _, row in df.iterrows():
        category = str(row.get("food_category", "other")).lower()
        food_name = str(row.get("food_name", "food"))
        low, high = DEFAULT_PRICE_RANGE
        for keyword, price_range in CATEGORY_PRICE_RANGES.items():
            if keyword in category:
                low, high = price_range
                break

        base_price = stable_random_value(f"{category}-{food_name}", low, high)
        protein = pd.to_numeric(row.get("protein", 0), errors="coerce")
        calories = pd.to_numeric(row.get("calories", 0), errors="coerce")
        adjustment = 1.0
        if pd.notna(protein) and protein >= 15:
            adjustment += 0.08
        if pd.notna(calories) and calories >= 350:
            adjustment += 0.04
        prices.append(int(round((base_price * adjustment) / 500) * 500))
    return pd.Series(prices, index=df.index, name="price")


def estimate_health_score(df: pd.DataFrame) -> pd.Series:
    features = df[["protein", "fiber", "calories", "fat", "sugar", "sodium"]].fillna(0)
    scaled = pd.DataFrame(MinMaxScaler().fit_transform(features), columns=features.columns)
    score = (
        50
        + 18 * scaled["protein"]
        + 18 * scaled["fiber"]
        - 9 * scaled["calories"]
        - 8 * scaled["fat"]
        - 13 * scaled["sugar"]
        - 10 * scaled["sodium"]
    )
    return score.clip(0, 100)


def preprocess_data(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Detect, clean, impute, deduplicate, enrich, and normalize dataset."""
    df = clean_column_names(raw_df)
    detected_columns = {target: detect_column(df, target) for target in COLUMN_CANDIDATES}

    processed = pd.DataFrame(index=df.index)
    processed["food_name"] = df[detected_columns["food_name"]] if detected_columns["food_name"] else "Unknown Food"
    processed["food_category"] = (
        df[detected_columns["food_category"]] if detected_columns["food_category"] else "Uncategorized"
    )

    numeric_targets = [
        "calories",
        "protein",
        "fat",
        "sugar",
        "fiber",
        "sodium",
        "health_score",
        "price",
        "carbohydrate",
        "saturated_fat",
        "cholesterol",
        "nova_group",
    ]
    for target in numeric_targets:
        source = detected_columns.get(target)
        processed[target] = pd.to_numeric(df[source], errors="coerce") if source else np.nan

    source = detected_columns.get("nutri_score")
    processed["nutri_score"] = df[source].astype(str) if source else "-"

    processed["food_name"] = processed["food_name"].fillna("Unknown Food").astype(str).str.strip()
    processed["food_category"] = processed["food_category"].fillna("Uncategorized").astype(str).str.strip()

    # Convert sodium from gram-like values to milligrams when the scale appears small.
    if processed["sodium"].dropna().between(0, 10).mean() > 0.8:
        processed["sodium"] = processed["sodium"] * 1000

    for column in ["calories", "protein", "fat", "sugar", "fiber", "sodium"]:
        fallback = processed[column].median()
        processed[column] = processed[column].fillna(0 if pd.isna(fallback) else fallback).clip(lower=0)

    if processed["health_score"].isna().all():
        processed["health_score"] = estimate_health_score(processed)
    else:
        processed["health_score"] = processed["health_score"].fillna(estimate_health_score(processed)).clip(0, 100)

    if processed["price"].isna().all():
        processed["price"] = generate_price_column(processed)
        price_note = "Kolom price dibuat otomatis karena dataset tidak menyediakan harga."
    else:
        processed["price"] = processed["price"].fillna(generate_price_column(processed)).clip(lower=0)
        price_note = "Kolom price tersedia sebagian/utuh; nilai kosong diisi dengan simulasi."

    for column in ["carbohydrate", "saturated_fat", "cholesterol", "nova_group"]:
        processed[column] = processed[column].fillna(0).clip(lower=0)

    before_dedup = len(processed)
    processed = processed.drop_duplicates(subset=["food_name", "food_category"]).reset_index(drop=True)
    duplicate_count = before_dedup - len(processed)

    scaler = MinMaxScaler()
    processed[[f"{col}_norm" for col in TOPSIS_CRITERIA]] = scaler.fit_transform(processed[TOPSIS_CRITERIA])

    metadata = {
        "detected_columns": detected_columns,
        "price_note": price_note,
        "duplicate_count": duplicate_count,
        "cleaned_columns": list(df.columns),
    }
    return processed, metadata


def calculate_ahp_weights(priority_scores: dict[str, int]) -> pd.Series:
    """Simple AHP: normalize user priority scores into criterion weights."""
    scores = pd.Series(priority_scores, dtype=float)
    total = scores.sum()
    if total == 0:
        return pd.Series(1 / len(scores), index=scores.index)
    return scores / total


def calculate_topsis(df: pd.DataFrame, weights: pd.Series) -> pd.DataFrame:
    matrix = df[TOPSIS_CRITERIA].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0)
    denominator = np.sqrt((matrix**2).sum(axis=0)).replace(0, 1)
    normalized = matrix / denominator
    weighted = normalized * weights[TOPSIS_CRITERIA]

    ideal_best = {}
    ideal_worst = {}
    for criterion in TOPSIS_CRITERIA:
        if criterion in BENEFIT_CRITERIA:
            ideal_best[criterion] = weighted[criterion].max()
            ideal_worst[criterion] = weighted[criterion].min()
        else:
            ideal_best[criterion] = weighted[criterion].min()
            ideal_worst[criterion] = weighted[criterion].max()

    best = pd.Series(ideal_best)
    worst = pd.Series(ideal_worst)
    distance_best = np.sqrt(((weighted - best) ** 2).sum(axis=1))
    distance_worst = np.sqrt(((weighted - worst) ** 2).sum(axis=1))

    result = df.copy()
    result["topsis_score"] = distance_worst / (distance_best + distance_worst + 1e-12)
    result["rank"] = result["topsis_score"].rank(ascending=False, method="dense").astype(int)
    return result.sort_values(["rank", "topsis_score"], ascending=[True, False])


def label_clusters(df: pd.DataFrame) -> dict[int, str]:
    summary = df.groupby("cluster").agg(
        health_score=("health_score", "mean"),
        price=("price", "mean"),
        topsis_score=("topsis_score", "mean"),
    )
    health_median = summary["health_score"].median()
    price_median = summary["price"].median()

    labels = {}
    used_labels = set()
    for cluster_id, row in summary.iterrows():
        if row["health_score"] >= health_median and row["price"] <= price_median:
            label = "Sehat dan Ekonomis"
        elif row["health_score"] >= health_median and row["price"] > price_median:
            label = "Sehat tapi Mahal"
        elif row["health_score"] < health_median and row["price"] <= price_median:
            label = "Murah tapi Kurang Sehat"
        else:
            label = "Kurang Direkomendasikan"

        if label in used_labels and len(summary) >= 4:
            remaining = [
                "Sehat dan Ekonomis",
                "Sehat tapi Mahal",
                "Murah tapi Kurang Sehat",
                "Kurang Direkomendasikan",
            ]
            label = next((item for item in remaining if item not in used_labels), label)
        labels[cluster_id] = label
        used_labels.add(label)
    return labels


def run_kmeans(df: pd.DataFrame, n_clusters: int = 4) -> pd.DataFrame:
    result = df.copy()
    if len(result) < 2:
        result["cluster"] = 0
        result["cluster_label"] = "Sehat dan Ekonomis"
        return result

    n_clusters = max(2, min(n_clusters, len(result)))
    features = result[TOPSIS_CRITERIA].astype(float).fillna(0)
    scaled_features = StandardScaler().fit_transform(features)
    result["cluster"] = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(scaled_features)
    result["cluster_label"] = result["cluster"].map(label_clusters(result))
    return result


def create_visualizations(df: pd.DataFrame) -> dict[str, object]:
    figures = {}
    for column in ["calories", "protein", "fat", "sugar", "sodium", "health_score"]:
        figures[column] = px.histogram(df, x=column, nbins=35, title=f"Distribusi {column}")

    corr = df[TOPSIS_CRITERIA].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(corr, annot=True, cmap="RdYlGn", center=0, fmt=".2f", ax=ax)
    ax.set_title("Korelasi Antar Kriteria Numerik")
    figures["correlation"] = fig
    return figures


def render_header() -> None:
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
        .food-card {
            min-height: 185px;
            border: 1px solid #d8e0e8;
            border-radius: 8px;
            padding: 14px;
            background: #ffffff;
        }
        .food-card h3 {font-size: 1rem; line-height: 1.25; margin: .45rem 0;}
        .food-card p, .mini {color: #53616f; font-size: .86rem; margin: .2rem 0;}
        .rank {
            width: 34px; height: 34px; display: grid; place-items: center;
            border-radius: 50%; background: #1f7a63; color: white; font-weight: 700;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title(APP_TITLE)
    st.write(
        "Sistem Pendukung Keputusan Cerdas untuk rekomendasi menu makanan sehat dan ekonomis "
        "menggunakan AHP sederhana, TOPSIS, dan K-Means Clustering."
    )


def render_dataset_instruction() -> None:
    st.info(
        "Belum ada dataset Kaggle yang diupload. Aplikasi menampilkan data contoh agar prototype tetap bisa diuji. "
        "Untuk analisis penuh, unduh dataset dari Kaggle, ekstrak file CSV, lalu upload melalui sidebar."
    )
    st.markdown(f"Dataset Kaggle: {KAGGLE_URL}")


def render_summary(df: pd.DataFrame, metadata: dict) -> None:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Jumlah baris", f"{df.shape[0]:,}")
    col2.metric("Jumlah kolom", f"{df.shape[1]:,}")
    col3.metric("Missing value", f"{int(df.isna().sum().sum()):,}")
    col4.metric("Duplikasi dihapus", f"{metadata['duplicate_count']:,}")

    st.caption(metadata["price_note"])
    with st.expander("Kolom yang terdeteksi otomatis"):
        detected = pd.DataFrame(
            [{"target_column": key, "detected_from_dataset": value or "-"} for key, value in metadata["detected_columns"].items()]
        )
        st.dataframe(detected, use_container_width=True, hide_index=True)

    left, right = st.columns([1.2, 1])
    with left:
        st.subheader("Statistik Deskriptif")
        st.dataframe(df[TOPSIS_CRITERIA].describe().T, use_container_width=True)
    with right:
        st.subheader("Distribusi Kategori")
        category_counts = df["food_category"].value_counts().head(15).reset_index()
        category_counts.columns = ["food_category", "count"]
        st.plotly_chart(
            px.bar(category_counts, x="count", y="food_category", orientation="h"),
            use_container_width=True,
        )


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Filter Interaktif")
    categories = sorted(df["food_category"].dropna().astype(str).unique())
    selected_categories = st.sidebar.multiselect("Kategori makanan", categories, default=categories[: min(8, len(categories))])

    max_price = int(max(1000, df["price"].quantile(0.98)))
    price_max = st.sidebar.slider("Harga maksimum (Rp)", 0, max_price, max_price, step=1000)
    calorie_max = st.sidebar.slider("Kalori maksimum", 0, int(max(1, df["calories"].quantile(0.98))), int(df["calories"].quantile(0.90)))
    protein_min = st.sidebar.slider("Protein minimum", 0.0, float(max(1, df["protein"].quantile(0.98))), 0.0, step=0.5)
    sugar_max = st.sidebar.slider("Gula maksimum", 0.0, float(max(1, df["sugar"].quantile(0.98))), float(df["sugar"].quantile(0.90)), step=0.5)
    sodium_max = st.sidebar.slider("Sodium maksimum (mg)", 0, int(max(1, df["sodium"].quantile(0.98))), int(df["sodium"].quantile(0.90)))
    health_min = st.sidebar.slider("Health score minimum", 0, 100, 40)

    category_mask = df["food_category"].isin(selected_categories) if selected_categories else True
    return df[
        category_mask
        & (df["price"] <= price_max)
        & (df["calories"] <= calorie_max)
        & (df["protein"] >= protein_min)
        & (df["sugar"] <= sugar_max)
        & (df["sodium"] <= sodium_max)
        & (df["health_score"] >= health_min)
    ].copy()


def render_weight_inputs() -> pd.Series:
    st.sidebar.header("Bobot Kriteria AHP")
    defaults = {
        "price": 8,
        "calories": 5,
        "protein": 7,
        "fat": 4,
        "sugar": 6,
        "fiber": 6,
        "sodium": 5,
        "health_score": 9,
    }
    priority_scores = {
        criterion: st.sidebar.slider(criterion, 1, 9, default)
        for criterion, default in defaults.items()
    }
    return calculate_ahp_weights(priority_scores)


def render_top_cards(ranked: pd.DataFrame) -> None:
    st.subheader("10 Rekomendasi Makanan Terbaik")
    top_items = ranked.head(10)
    for row_group in np.array_split(top_items, 2):
        columns = st.columns(len(row_group) if len(row_group) else 1)
        for column, (_, row) in zip(columns, row_group.iterrows()):
            with column:
                st.markdown(
                    f"""
                    <div class="food-card">
                        <div class="rank">#{int(row['rank'])}</div>
                        <h3>{row['food_name'][:58]}</h3>
                        <p>{row['food_category']}</p>
                        <strong>TOPSIS {row['topsis_score']:.3f}</strong>
                        <div class="mini">Health {row['health_score']:.1f} | Rp{row['price']:,.0f}</div>
                        <div class="mini">{row['calories']:.0f} kkal | Protein {row['protein']:.1f}g</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def render_interpretation(ranked: pd.DataFrame) -> None:
    best = ranked.iloc[0]
    cluster_summary = (
        ranked.groupby("cluster_label")
        .agg(jumlah=("food_name", "count"), rata_topsis=("topsis_score", "mean"), rata_harga=("price", "mean"))
        .sort_values("rata_topsis", ascending=False)
        .reset_index()
    )
    st.subheader("Ringkasan Interpretasi")
    st.write(
        f"Makanan dengan prioritas tertinggi adalah **{best['food_name']}** dengan skor TOPSIS "
        f"**{best['topsis_score']:.3f}**. Skor ini menunjukkan kedekatan alternatif terhadap profil ideal: "
        "protein, fiber, dan health_score tinggi, dengan harga, kalori, lemak, gula, dan sodium lebih rendah."
    )
    st.dataframe(cluster_summary, use_container_width=True, hide_index=True)


def main() -> None:
    render_header()

    with st.sidebar:
        st.header("Dataset")
        uploaded_file = st.file_uploader("Upload dataset CSV Kaggle", type=["csv"])

    raw_df, data_source = load_data(uploaded_file)
    if raw_df is None:
        st.error("Dataset tidak tersedia dan sample_data.csv belum ditemukan.")
        return
    if data_source == "sample":
        render_dataset_instruction()

    st.subheader("Preview Dataset Mentah")
    st.dataframe(raw_df.head(20), use_container_width=True)

    try:
        df, metadata = preprocess_data(raw_df)
    except Exception as exc:
        st.error(f"Preprocessing gagal: {exc}")
        st.write("Pastikan file CSV memiliki minimal kolom nama makanan/kategori dan beberapa kolom nutrisi.")
        return

    if df.empty:
        st.warning("Dataset kosong setelah preprocessing.")
        return

    weights = render_weight_inputs()
    filtered = apply_filters(df)
    n_clusters = st.sidebar.slider("Jumlah cluster K-Means", 2, 8, 4)

    tab_summary, tab_eda, tab_result, tab_data = st.tabs(
        ["Ringkasan Dataset", "Eksplorasi Data", "Hasil Rekomendasi", "Data Preprocessed"]
    )

    with tab_summary:
        render_summary(df, metadata)

    with tab_eda:
        figures = create_visualizations(df)
        columns = st.columns(2)
        for index, column in enumerate(["calories", "protein", "fat", "sugar", "sodium", "health_score"]):
            with columns[index % 2]:
                st.plotly_chart(figures[column], use_container_width=True)
        st.pyplot(figures["correlation"], use_container_width=True)

    with tab_result:
        if filtered.empty:
            st.warning("Tidak ada makanan yang memenuhi filter. Longgarkan filter di sidebar.")
            return

        ranked = calculate_topsis(filtered, weights)
        ranked = run_kmeans(ranked, n_clusters)
        render_top_cards(ranked)

        st.subheader("Tabel Ranking TOPSIS")
        result_columns = [
            "rank",
            "food_name",
            "food_category",
            "price",
            "calories",
            "protein",
            "fat",
            "sugar",
            "fiber",
            "sodium",
            "health_score",
            "topsis_score",
            "cluster_label",
        ]
        st.dataframe(ranked[result_columns].head(100), use_container_width=True, hide_index=True)

        left, right = st.columns(2)
        with left:
            top_scores = ranked.head(20).sort_values("topsis_score")
            st.plotly_chart(
                px.bar(top_scores, x="topsis_score", y="food_name", orientation="h", title="Grafik Skor TOPSIS Top 20"),
                use_container_width=True,
            )
        with right:
            st.plotly_chart(
                px.scatter(
                    ranked,
                    x="price",
                    y="health_score",
                    size="protein",
                    color="cluster_label",
                    hover_name="food_name",
                    title="Scatter Plot Cluster: Harga vs Health Score",
                ),
                use_container_width=True,
            )
        render_interpretation(ranked)

        st.subheader("Bobot Kriteria")
        st.dataframe(weights.rename("weight").reset_index().rename(columns={"index": "criterion"}), hide_index=True)

    with tab_data:
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.caption(
            "Kolom *_norm adalah hasil normalisasi numerik. Kolom tambahan seperti carbohydrate, saturated_fat, "
            "cholesterol, nova_group, dan nutri_score dipertahankan sebagai konteks analisis bila tersedia."
        )


if __name__ == "__main__":
    main()
