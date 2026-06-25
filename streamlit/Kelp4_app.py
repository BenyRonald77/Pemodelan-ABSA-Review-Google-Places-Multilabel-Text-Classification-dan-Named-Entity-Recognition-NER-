import re
import html
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences


# ============================================================
# STREAMLIT PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Kelp4 ABSA Multilabel & NER",
    layout="wide"
)


# ============================================================
# PATH CONFIG
# ============================================================
BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "models"

MULTILABEL_MODEL_PATH = MODEL_DIR / "Kelp4_best_multilabel_model.pkl"
MULTILABEL_LABELS_PATH = MODEL_DIR / "Kelp4_multilabel_labels.pkl"
MULTILABEL_THRESHOLDS_PATH = MODEL_DIR / "Kelp4_multilabel_thresholds.pkl"

NER_MODEL_PATH = MODEL_DIR / "Kelp4_best_ner_model.keras"
NER_WORD2IDX_PATH = MODEL_DIR / "Kelp4_ner_word2idx.pkl"
NER_IDX2TAG_PATH = MODEL_DIR / "Kelp4_ner_idx2tag.pkl"
NER_CONFIG_PATH = MODEL_DIR / "Kelp4_ner_config.pkl"


# ============================================================
# LOAD MODELS
# ============================================================
@st.cache_resource
def load_all_models():
    multilabel_model = joblib.load(MULTILABEL_MODEL_PATH)
    multilabel_labels = joblib.load(MULTILABEL_LABELS_PATH)

    if MULTILABEL_THRESHOLDS_PATH.exists():
        multilabel_thresholds = joblib.load(MULTILABEL_THRESHOLDS_PATH)
    else:
        multilabel_thresholds = {
            label: 0.5
            for label in multilabel_labels
        }

    ner_model = tf.keras.models.load_model(
        NER_MODEL_PATH,
        compile=False
    )

    word2idx = joblib.load(NER_WORD2IDX_PATH)
    idx2tag = joblib.load(NER_IDX2TAG_PATH)
    ner_config = joblib.load(NER_CONFIG_PATH)

    idx2tag = {
        int(k): v
        for k, v in idx2tag.items()
    }

    return (
        multilabel_model,
        multilabel_labels,
        multilabel_thresholds,
        ner_model,
        word2idx,
        idx2tag,
        ner_config
    )


(
    multilabel_model,
    multilabel_labels,
    multilabel_thresholds,
    ner_model,
    word2idx,
    idx2tag,
    ner_config
) = load_all_models()

MAX_LEN = ner_config["MAX_LEN"]


# ============================================================
# HELPER FUNCTIONS - TEXT CLEANING
# ============================================================
def normalize_raw_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-ZÀ-ÿ\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_text(text):
    """
    Cleaning teks untuk model multilabel.
    Fungsi ini dibuat sama dengan preprocessing notebook,
    termasuk penanganan negasi sederhana.
    """

    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-zA-ZÀ-ÿ\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = text.split()

    negation_words = {
        "tidak", "tdk", "nggak", "ngga", "gak", "ga",
        "kurang", "tak", "bukan"
    }

    new_tokens = []
    i = 0

    while i < len(tokens):
        token = tokens[i]

        if token in negation_words and i + 1 < len(tokens):
            next_token = tokens[i + 1]

            new_tokens.append(token)
            new_tokens.append(token + "_" + next_token)

            i += 2
        else:
            new_tokens.append(token)
            i += 1

    return " ".join(new_tokens)


# ============================================================
# HELPER FUNCTIONS - CONTEXT DETECTION
# ============================================================
def has_product_context(text):
    text = normalize_raw_text(text)

    product_words = [
        "makanan", "makanannya", "bakso", "baksonya", "menu",
        "rasa", "rasanya", "minuman", "kuah", "daging", "produk",
        "sop", "mie", "ayam", "sapi", "tetelan", "porsi",
        "masakan", "hidangan", "makan", "minum"
    ]

    return any(word in text for word in product_words)


def has_price_context(text):
    text = normalize_raw_text(text)

    price_words = [
        "harga", "harganya", "mahal", "murah", "terjangkau",
        "worth", "worthit", "uang", "bayar", "tarif",
        "kemahalan", "murahan", "biaya"
    ]

    return any(word in text for word in price_words)


def has_place_context(text):
    text = normalize_raw_text(text)

    place_words = [
        "tempat", "tempatnya", "lokasi", "parkir", "suasana",
        "ruangan", "meja", "kursi", "bersih", "nyaman",
        "luas", "sempit", "ac", "toilet", "area", "jalan",
        "cabang", "outlet"
    ]

    return any(word in text for word in place_words)


def has_promotion_context(text):
    text = normalize_raw_text(text)

    promotion_words = [
        "promo", "promosi", "diskon", "voucher", "cashback",
        "paket", "potongan", "gratis", "bonus", "deal"
    ]

    return any(word in text for word in promotion_words)


# ============================================================
# HELPER FUNCTIONS - ERROR ANALYSIS RULES
# ============================================================
def detect_product_negative_negation(text):
    text = normalize_raw_text(text)

    negation_phrases = [
        "tidak enak",
        "tdk enak",
        "nggak enak",
        "ngga enak",
        "gak enak",
        "ga enak",
        "kurang enak",
        "tidak lezat",
        "kurang lezat",
        "tidak mantap",
        "kurang mantap",
        "tidak recommended",
        "tidak rekomendasi",
        "tidak direkomendasikan",
        "tidak cocok",
        "kurang cocok",
        "rasanya aneh",
        "rasa aneh",
        "rasanya hambar",
        "rasa hambar",
        "tidak segar",
        "kurang segar"
    ]

    return has_product_context(text) and any(
        phrase in text
        for phrase in negation_phrases
    )


def detect_product_positive(text):
    text = normalize_raw_text(text)

    positive_product_phrases = [
        "makanan enak",
        "makanannya enak",
        "bakso enak",
        "baksonya enak",
        "rasanya enak",
        "rasa enak",
        "enak banget",
        "sangat enak",
        "lezat",
        "mantap",
        "nikmat",
        "recommended",
        "rekomendasi"
    ]

    return has_product_context(text) and any(
        phrase in text
        for phrase in positive_product_phrases
    )


def detect_price_negative(text):
    text = normalize_raw_text(text)

    negative_price_phrases = [
        "harga mahal",
        "harganya mahal",
        "terlalu mahal",
        "mahal banget",
        "mahal sekali",
        "sangat mahal",
        "agak mahal",
        "cukup mahal",
        "kurang worth",
        "tidak worth",
        "ga worth",
        "gak worth",
        "ngga worth",
        "nggak worth",
        "tidak worthit",
        "ga worthit",
        "gak worthit",
        "kemahalan"
    ]

    return any(
        phrase in text
        for phrase in negative_price_phrases
    )


def detect_price_positive(text):
    text = normalize_raw_text(text)

    positive_price_phrases = [
        "harga murah",
        "harganya murah",
        "murah banget",
        "murah sekali",
        "murah meriah",
        "harga terjangkau",
        "terjangkau",
        "worth it",
        "worthit",
        "sesuai harga",
        "harga sesuai",
        "harga oke",
        "harga ok"
    ]

    return any(
        phrase in text
        for phrase in positive_price_phrases
    )


def detect_place_positive(text):
    text = normalize_raw_text(text)

    positive_place_phrases = [
        "tempat nyaman",
        "tempatnya nyaman",
        "tempat luas",
        "tempatnya luas",
        "lokasi strategis",
        "suasana nyaman",
        "tempat bersih",
        "tempatnya bersih",
        "parkir luas",
        "ruangan nyaman"
    ]

    return any(
        phrase in text
        for phrase in positive_place_phrases
    )


def detect_place_negative(text):
    text = normalize_raw_text(text)

    negative_place_phrases = [
        "tempat sempit",
        "tempatnya sempit",
        "tempat kotor",
        "tempatnya kotor",
        "parkir susah",
        "parkir sempit",
        "lokasi susah",
        "ruangan panas",
        "toilet kotor",
        "tidak nyaman",
        "kurang nyaman"
    ]

    return any(
        phrase in text
        for phrase in negative_place_phrases
    )


def detect_promotion_positive(text):
    text = normalize_raw_text(text)

    positive_promotion_phrases = [
        "promo menarik",
        "promonya menarik",
        "diskon menarik",
        "ada diskon",
        "ada promo",
        "banyak promo",
        "voucher menarik",
        "cashback",
        "bonus",
        "gratis"
    ]

    return any(
        phrase in text
        for phrase in positive_promotion_phrases
    )


def detect_promotion_negative(text):
    text = normalize_raw_text(text)

    negative_promotion_phrases = [
        "promo tidak jelas",
        "promonya tidak jelas",
        "diskon tidak berlaku",
        "voucher tidak bisa",
        "voucher gagal",
        "promo mengecewakan"
    ]

    return any(
        phrase in text
        for phrase in negative_promotion_phrases
    )


# ============================================================
# HELPER FUNCTIONS - POST-PROCESSING MULTILABEL
# ============================================================
def apply_postprocessing_multilabel(text, result_df):
    """
    Post-processing berdasarkan error analysis.
    Model utama tetap digunakan, lalu hasilnya dirapikan agar tidak kontradiktif.
    """

    postprocess_notes = []

    # PRODUCT NEGATIVE
    if detect_product_negative_negation(text):
        result_df.loc[
            result_df["label"].isin([
                "PRODUCT_POSITIVE",
                "PRODUCT_NEUTRAL"
            ]),
            "prediksi"
        ] = 0

        result_df.loc[
            result_df["label"] == "PRODUCT_NEGATIVE",
            "prediksi"
        ] = 1

        result_df.loc[
            result_df["label"] == "OUT_OF_TOPIC",
            "prediksi"
        ] = 0

        postprocess_notes.append(
            "Terdeteksi pola negasi pada aspek product, sehingga prediksi disesuaikan menjadi PRODUCT_NEGATIVE."
        )

    # PRODUCT POSITIVE
    if detect_product_positive(text) and not detect_product_negative_negation(text):
        result_df.loc[
            result_df["label"].isin([
                "PRODUCT_NEGATIVE",
                "PRODUCT_NEUTRAL"
            ]),
            "prediksi"
        ] = 0

        result_df.loc[
            result_df["label"] == "PRODUCT_POSITIVE",
            "prediksi"
        ] = 1

        result_df.loc[
            result_df["label"] == "OUT_OF_TOPIC",
            "prediksi"
        ] = 0

        postprocess_notes.append(
            "Terdeteksi pola product positif, sehingga prediksi diperkuat menjadi PRODUCT_POSITIVE."
        )

    # PRICE NEGATIVE
    if detect_price_negative(text):
        result_df.loc[
            result_df["label"].isin([
                "PRICE_POSITIVE",
                "PRICE_NEUTRAL"
            ]),
            "prediksi"
        ] = 0

        result_df.loc[
            result_df["label"] == "PRICE_NEGATIVE",
            "prediksi"
        ] = 1

        result_df.loc[
            result_df["label"] == "OUT_OF_TOPIC",
            "prediksi"
        ] = 0

        postprocess_notes.append(
            "Terdeteksi pola harga negatif seperti mahal/kemahalan, sehingga prediksi disesuaikan menjadi PRICE_NEGATIVE."
        )

    # PRICE POSITIVE
    if detect_price_positive(text):
        result_df.loc[
            result_df["label"].isin([
                "PRICE_NEGATIVE",
                "PRICE_NEUTRAL"
            ]),
            "prediksi"
        ] = 0

        result_df.loc[
            result_df["label"] == "PRICE_POSITIVE",
            "prediksi"
        ] = 1

        result_df.loc[
            result_df["label"] == "OUT_OF_TOPIC",
            "prediksi"
        ] = 0

        postprocess_notes.append(
            "Terdeteksi pola harga positif seperti murah/terjangkau/worthit, sehingga prediksi disesuaikan menjadi PRICE_POSITIVE."
        )

    # PLACE POSITIVE
    if detect_place_positive(text):
        result_df.loc[
            result_df["label"].isin([
                "PLACE_NEGATIVE",
                "PLACE_NEUTRAL"
            ]),
            "prediksi"
        ] = 0

        result_df.loc[
            result_df["label"] == "PLACE_POSITIVE",
            "prediksi"
        ] = 1

        result_df.loc[
            result_df["label"] == "OUT_OF_TOPIC",
            "prediksi"
        ] = 0

        postprocess_notes.append(
            "Terdeteksi pola tempat positif, sehingga prediksi disesuaikan menjadi PLACE_POSITIVE."
        )

    # PLACE NEGATIVE
    if detect_place_negative(text):
        result_df.loc[
            result_df["label"].isin([
                "PLACE_POSITIVE",
                "PLACE_NEUTRAL"
            ]),
            "prediksi"
        ] = 0

        result_df.loc[
            result_df["label"] == "PLACE_NEGATIVE",
            "prediksi"
        ] = 1

        result_df.loc[
            result_df["label"] == "OUT_OF_TOPIC",
            "prediksi"
        ] = 0

        postprocess_notes.append(
            "Terdeteksi pola tempat negatif, sehingga prediksi disesuaikan menjadi PLACE_NEGATIVE."
        )

    # PROMOTION POSITIVE
    if detect_promotion_positive(text):
        result_df.loc[
            result_df["label"].isin([
                "PROMOTION_NEGATIVE",
                "PROMOTION_NEUTRAL"
            ]),
            "prediksi"
        ] = 0

        result_df.loc[
            result_df["label"] == "PROMOTION_POSITIVE",
            "prediksi"
        ] = 1

        result_df.loc[
            result_df["label"] == "OUT_OF_TOPIC",
            "prediksi"
        ] = 0

        postprocess_notes.append(
            "Terdeteksi pola promosi positif, sehingga prediksi disesuaikan menjadi PROMOTION_POSITIVE."
        )

    # PROMOTION NEGATIVE
    if detect_promotion_negative(text):
        result_df.loc[
            result_df["label"].isin([
                "PROMOTION_POSITIVE",
                "PROMOTION_NEUTRAL"
            ]),
            "prediksi"
        ] = 0

        result_df.loc[
            result_df["label"] == "PROMOTION_NEGATIVE",
            "prediksi"
        ] = 1

        result_df.loc[
            result_df["label"] == "OUT_OF_TOPIC",
            "prediksi"
        ] = 0

        postprocess_notes.append(
            "Terdeteksi pola promosi negatif, sehingga prediksi disesuaikan menjadi PROMOTION_NEGATIVE."
        )

    # Hapus prediksi aspek yang tidak punya konteks di teks
    if not has_promotion_context(text):
        result_df.loc[
            result_df["label"].isin([
                "PROMOTION_POSITIVE",
                "PROMOTION_NEGATIVE",
                "PROMOTION_NEUTRAL"
            ]),
            "prediksi"
        ] = 0

    if not has_price_context(text):
        result_df.loc[
            result_df["label"].isin([
                "PRICE_POSITIVE",
                "PRICE_NEGATIVE",
                "PRICE_NEUTRAL"
            ]),
            "prediksi"
        ] = 0

    if not has_place_context(text):
        result_df.loc[
            result_df["label"].isin([
                "PLACE_POSITIVE",
                "PLACE_NEGATIVE",
                "PLACE_NEUTRAL"
            ]),
            "prediksi"
        ] = 0

    if not has_product_context(text):
        result_df.loc[
            result_df["label"].isin([
                "PRODUCT_POSITIVE",
                "PRODUCT_NEGATIVE",
                "PRODUCT_NEUTRAL"
            ]),
            "prediksi"
        ] = 0

    # Jika ada aspek valid, OUT_OF_TOPIC dimatikan
    aspect_labels = [
        label for label in result_df["label"].tolist()
        if label != "OUT_OF_TOPIC"
    ]

    has_any_aspect_prediction = result_df[
        (result_df["label"].isin(aspect_labels)) &
        (result_df["prediksi"] == 1)
    ].shape[0] > 0

    if has_any_aspect_prediction:
        result_df.loc[
            result_df["label"] == "OUT_OF_TOPIC",
            "prediksi"
        ] = 0

    # Jika setelah filtering tidak ada label, ambil label confidence tertinggi
    # sesuai konteks agar output tidak kosong
    if result_df["prediksi"].sum() == 0:
        allowed_labels = []

        if has_product_context(text):
            allowed_labels.extend([
                "PRODUCT_POSITIVE",
                "PRODUCT_NEGATIVE",
                "PRODUCT_NEUTRAL"
            ])

        if has_price_context(text):
            allowed_labels.extend([
                "PRICE_POSITIVE",
                "PRICE_NEGATIVE",
                "PRICE_NEUTRAL"
            ])

        if has_place_context(text):
            allowed_labels.extend([
                "PLACE_POSITIVE",
                "PLACE_NEGATIVE",
                "PLACE_NEUTRAL"
            ])

        if has_promotion_context(text):
            allowed_labels.extend([
                "PROMOTION_POSITIVE",
                "PROMOTION_NEGATIVE",
                "PROMOTION_NEUTRAL"
            ])

        if len(allowed_labels) > 0:
            candidate_df = result_df[
                result_df["label"].isin(allowed_labels)
            ]

            if len(candidate_df) > 0:
                best_idx = candidate_df["confidence"].idxmax()
                result_df.loc[best_idx, "prediksi"] = 1
        else:
            result_df.loc[
                result_df["label"] == "OUT_OF_TOPIC",
                "prediksi"
            ] = 1

    # Tambahkan catatan post-processing
    result_df["postprocess_note"] = ""

    if postprocess_notes:
        note_text = " ".join(postprocess_notes)

        active_labels = result_df[
            result_df["prediksi"] == 1
        ]["label"].tolist()

        result_df.loc[
            result_df["label"].isin(active_labels),
            "postprocess_note"
        ] = note_text

    return result_df


# ============================================================
# HELPER FUNCTIONS - MULTILABEL
# ============================================================
def predict_multilabel(text):
    clean = clean_text(text)

    try:
        proba = multilabel_model.predict_proba([clean])[0]
    except Exception:
        pred_default = multilabel_model.predict([clean])[0]
        proba = pred_default.astype(float)

    pred = []

    for label, score in zip(multilabel_labels, proba):
        threshold = multilabel_thresholds.get(label, 0.5)
        pred.append(1 if score >= threshold else 0)

    result_df = pd.DataFrame({
        "label": multilabel_labels,
        "confidence": proba,
        "threshold": [
            multilabel_thresholds.get(label, 0.5)
            for label in multilabel_labels
        ],
        "prediksi": pred
    })

    result_df = apply_postprocessing_multilabel(
        text,
        result_df
    )

    selected_labels = result_df[
        result_df["prediksi"] == 1
    ]["label"].tolist()

    result_df = result_df.sort_values(
        by="confidence",
        ascending=False
    ).reset_index(drop=True)

    return selected_labels, result_df


def interpret_sentiment(label):
    if label.endswith("_POSITIVE"):
        return "Positive"
    elif label.endswith("_NEGATIVE"):
        return "Negative"
    elif label.endswith("_NEUTRAL"):
        return "Neutral"
    elif label == "OUT_OF_TOPIC":
        return "Out of Topic"
    else:
        return "-"


def make_multilabel_summary_df(selected_labels):
    rows = []

    for label in selected_labels:
        if label == "OUT_OF_TOPIC":
            aspect = "OUT_OF_TOPIC"
        else:
            aspect = label.split("_")[0]

        rows.append({
            "label": label,
            "aspect": aspect,
            "sentiment": interpret_sentiment(label)
        })

    return pd.DataFrame(rows)


def extract_aspects_from_multilabel(labels):
    aspects = []

    for label in labels:
        if label.startswith("PRODUCT"):
            aspects.append("PRODUCT")
        elif label.startswith("PRICE"):
            aspects.append("PRICE")
        elif label.startswith("PLACE"):
            aspects.append("PLACE")
        elif label.startswith("PROMOTION"):
            aspects.append("PROMOTION")
        elif label == "OUT_OF_TOPIC":
            aspects.append("OUT_OF_TOPIC")

    return sorted(list(set(aspects)))


# ============================================================
# HELPER FUNCTIONS - NER
# ============================================================
def simple_tokenize(text):
    tokens = re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)
    return tokens


def encode_tokens_for_ner(tokens):
    token_ids = [
        word2idx.get(token, word2idx.get("UNK", 1))
        for token in tokens
    ]

    padded = pad_sequences(
        [token_ids],
        maxlen=MAX_LEN,
        padding="post",
        truncating="post",
        value=word2idx.get("PAD", 0)
    )

    return padded


def predict_ner(text):
    tokens = simple_tokenize(text)

    if len(tokens) == 0:
        return [], [], pd.DataFrame()

    encoded = encode_tokens_for_ner(tokens)

    pred_prob = ner_model.predict(encoded, verbose=0)
    pred_ids = np.argmax(pred_prob, axis=-1)[0]

    used_len = min(len(tokens), MAX_LEN)

    used_tokens = tokens[:used_len]
    pred_tags = [
        idx2tag[int(tag_id)]
        for tag_id in pred_ids[:used_len]
    ]

    token_df = pd.DataFrame({
        "token": used_tokens,
        "predicted_tag": pred_tags
    })

    return used_tokens, pred_tags, token_df


def bio_to_entities(tokens, tags):
    entities = []
    current_tokens = []
    current_label = None

    for token, tag in zip(tokens, tags):
        if tag == "O":
            if current_tokens:
                entities.append({
                    "entity_text": " ".join(current_tokens),
                    "aspect": current_label
                })

                current_tokens = []
                current_label = None

            continue

        if "-" not in tag:
            continue

        prefix, label = tag.split("-", 1)

        if prefix == "B":
            if current_tokens:
                entities.append({
                    "entity_text": " ".join(current_tokens),
                    "aspect": current_label
                })

            current_tokens = [token]
            current_label = label

        elif prefix == "I":
            if current_tokens and current_label == label:
                current_tokens.append(token)
            else:
                current_tokens = [token]
                current_label = label

    if current_tokens:
        entities.append({
            "entity_text": " ".join(current_tokens),
            "aspect": current_label
        })

    return pd.DataFrame(entities)


def render_highlight(tokens, tags):
    html_tokens = []

    for token, tag in zip(tokens, tags):
        safe_token = html.escape(token)

        if tag == "O":
            html_tokens.append(
                f"<span style='padding:3px 5px; margin:2px; "
                f"display:inline-block;'>{safe_token}</span>"
            )
        else:
            safe_tag = html.escape(tag)

            html_tokens.append(
                f"<span style='background-color:#fff3b0; color:#000000; "
                f"padding:3px 6px; border-radius:6px; margin:2px; "
                f"display:inline-block;'>"
                f"{safe_token} <small>({safe_tag})</small></span>"
            )

    return " ".join(html_tokens)


# ============================================================
# STREAMLIT UI
# ============================================================
st.title("ABSA Review Google Places")
st.subheader("Multilabel Text Classification dan Named Entity Recognition")

st.markdown(
    """
    Aplikasi ini digunakan untuk demo prediksi ABSA dari review baru.
    Model pertama memprediksi label multilabel pada level review,
    sedangkan model kedua mengekstraksi span/aspek menggunakan NER.
    """
)

review_text = st.text_area(
    "Masukkan review baru:",
    value="Baksonya enak, tempatnya luas, tapi harganya mahal.",
    height=120
)

tab1, tab2, tab3 = st.tabs([
    "Prediksi Multilabel ABSA",
    "Prediksi NER ABSA",
    "Hubungan Multilabel dan NER"
])


# ============================================================
# TAB 1 - MULTILABEL
# ============================================================
with tab1:
    st.header("Prediksi Multilabel ABSA")

    if review_text.strip():
        selected_labels, multilabel_result_df = predict_multilabel(
            review_text
        )

        st.write("**Review input:**")
        st.info(review_text)

        st.write("**Label yang diprediksi:**")

        if selected_labels:
            st.success(", ".join(selected_labels))

            summary_df = make_multilabel_summary_df(
                selected_labels
            )

            st.write("**Ringkasan aspek dan sentimen:**")
            st.dataframe(
                summary_df,
                use_container_width=True
            )
        else:
            st.warning("Tidak ada label yang melewati threshold prediksi.")

        st.write("**Detail confidence, threshold, prediksi, dan post-processing:**")
        st.dataframe(
            multilabel_result_df,
            use_container_width=True
        )

        st.markdown(
            """
            **Interpretasi:**  
            Kolom `confidence` menunjukkan skor probabilitas model.  
            Kolom `threshold` adalah batas minimum hasil tuning untuk setiap label.  
            Kolom `prediksi = 1` berarti label tersebut diprediksi muncul pada review.  
            Kolom `postprocess_note` menunjukkan catatan perbaikan berbasis error analysis jika ada.  

            Karena ini multilabel classification, satu review dapat memiliki lebih dari satu label.
            """
        )
    else:
        st.warning("Masukkan teks review terlebih dahulu.")


# ============================================================
# TAB 2 - NER
# ============================================================
with tab2:
    st.header("Prediksi NER ABSA")

    if review_text.strip():
        tokens, pred_tags, token_df = predict_ner(review_text)
        entities_df = bio_to_entities(tokens, pred_tags)

        st.write("**Token dan tag prediksi:**")
        st.dataframe(
            token_df,
            use_container_width=True
        )

        st.write("**Entitas/aspek yang diekstraksi:**")

        if len(entities_df) > 0:
            st.dataframe(
                entities_df,
                use_container_width=True
            )
        else:
            st.warning("Tidak ada entitas/aspek yang terdeteksi.")

        st.write("**Highlight hasil NER:**")
        highlighted_html = render_highlight(tokens, pred_tags)

        st.markdown(
            highlighted_html,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            **Interpretasi:**  
            Tag `B-` menunjukkan awal entitas, `I-` menunjukkan lanjutan entitas,
            dan `O` menunjukkan token yang bukan bagian dari entitas/aspek.
            """
        )
    else:
        st.warning("Masukkan teks review terlebih dahulu.")


# ============================================================
# TAB 3 - HUBUNGAN MULTILABEL DAN NER
# ============================================================
with tab3:
    st.header("Hubungan Output Multilabel dan NER")

    if review_text.strip():
        selected_labels, _ = predict_multilabel(review_text)
        tokens, pred_tags, _ = predict_ner(review_text)

        entities_df = bio_to_entities(tokens, pred_tags)
        multilabel_aspects = extract_aspects_from_multilabel(
            selected_labels
        )

        if len(entities_df) > 0:
            ner_aspects = sorted(
                entities_df["aspect"].unique().tolist()
            )
        else:
            ner_aspects = []

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Aspek dari Multilabel Classification:**")

            if multilabel_aspects:
                st.success(", ".join(multilabel_aspects))
            else:
                st.warning("Tidak ada aspek terdeteksi dari multilabel.")

            if selected_labels:
                st.write("**Label multilabel:**")
                st.dataframe(
                    make_multilabel_summary_df(selected_labels),
                    use_container_width=True
                )

        with col2:
            st.write("**Aspek dari NER:**")

            if ner_aspects:
                st.success(", ".join(ner_aspects))
            else:
                st.warning("Tidak ada aspek terdeteksi dari NER.")

            if len(entities_df) > 0:
                st.write("**Span/aspek hasil NER:**")
                st.dataframe(
                    entities_df,
                    use_container_width=True
                )

        st.markdown(
            """
            **Penjelasan hubungan:**  
            Multilabel classification memprediksi aspek dan sentimen pada level review,
            misalnya `PRODUCT_POSITIVE`, `PRODUCT_NEGATIVE`, atau `PRICE_NEGATIVE`.
            NER memprediksi bagian teks yang menjadi span/aspek,
            misalnya token yang termasuk `PRODUCT`, `PRICE`, `PLACE`, atau `PROMOTION`.

            Jadi, multilabel menjawab **review ini membahas aspek apa dan sentimennya apa**,
            sedangkan NER menjawab **bagian teks mana yang menunjukkan aspek tersebut**.
            """
        )
    else:
        st.warning("Masukkan teks review terlebih dahulu.")