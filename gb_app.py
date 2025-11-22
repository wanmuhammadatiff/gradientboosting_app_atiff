import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

st.set_page_config(page_title="Gradient Boosting Classification App", layout="wide")

st.title("Gradient Boosting Classification Web App")
st.write(
    "Upload a dataset, select features & target, train a Gradient Boosting classifier, "
    "inspect performance and try single-sample predictions."
)

# Sidebar - logo / authors
st.sidebar.image(
    "https://brand.umpsa.edu.my/images/2024/02/29/umpsa-bangunan__1764x719.png",
    use_container_width=True,
)
st.sidebar.header("Developers:")
st.sidebar.write("- Ku Muhammad Naim Ku Khalif")

# -----------------------------
# Dataset upload
# -----------------------------
st.sidebar.header("Dataset Upload")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
if uploaded_file is None:
    st.info("Please upload a CSV file to continue.")
    st.stop()

df = pd.read_csv(uploaded_file)
st.success("Dataset loaded.")
st.subheader("Dataset Preview")
st.dataframe(df.head())

# -----------------------------
# Features & target selection
# -----------------------------
st.sidebar.header("Features & Target")
all_columns = df.columns.tolist()
if len(all_columns) == 0:
    st.error("Uploaded file appears empty.")
    st.stop()

target_col = st.sidebar.selectbox("Select target column (y)", all_columns, index=len(all_columns) - 1)
feature_cols = st.sidebar.multiselect(
    "Select feature columns (X)",
    [c for c in all_columns if c != target_col],
    default=[c for c in all_columns if c != target_col],
)
if len(feature_cols) == 0:
    st.error("Please select at least one feature column.")
    st.stop()

# -----------------------------
# Encode categorical features & target
# -----------------------------
df_processed = df.copy()
label_encoders = {}

for col in feature_cols:
    if not np.issubdtype(df_processed[col].dtype, np.number):
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col].astype(str))
        label_encoders[col] = le
        st.sidebar.info(f"Encoded feature: `{col}`")

if not np.issubdtype(df_processed[target_col].dtype, np.number):
    le = LabelEncoder()
    df_processed[target_col] = le.fit_transform(df_processed[target_col].astype(str))
    label_encoders[target_col] = le
    st.sidebar.info(f"Encoded target: `{target_col}`")

X = df_processed[feature_cols].values
y = df_processed[target_col].values

# -----------------------------
# Train/test split settings
# -----------------------------
st.sidebar.header("Train/Test Split")
test_size = st.sidebar.slider("Test size (proportion)", 0.1, 0.5, 0.25, 0.05)
random_state = int(st.sidebar.number_input("Random state", value=42, step=1))

# Optional scaling (not required for trees but offered)
scale_features = st.sidebar.checkbox("Apply StandardScaler", value=False)
scaler = None
if scale_features:
    scaler = StandardScaler()

# -----------------------------
# Gradient Boosting hyperparameters
# -----------------------------
st.sidebar.header("Gradient Boosting Hyperparameters")
n_estimators = st.sidebar.slider("n_estimators", 50, 1000, 100, 50)
learning_rate = st.sidebar.slider("learning_rate", 0.01, 1.0, 0.1, 0.01)
max_depth = st.sidebar.slider("max_depth (tree depth)", 1, 10, 3, 1)
subsample = st.sidebar.slider("subsample (stochastic boosting)", 0.1, 1.0, 1.0, 0.1)
min_samples_split = st.sidebar.number_input("min_samples_split", min_value=2, value=2, step=1)
min_samples_leaf = st.sidebar.number_input("min_samples_leaf", min_value=1, value=1, step=1)
max_features = st.sidebar.selectbox("max_features", ["auto", "sqrt", "log2", None], index=1)

# -----------------------------
# Train/test split (stratify if possible)
# -----------------------------
try:
    strat = y if len(np.unique(y)) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=strat
    )
except Exception:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

# apply scaler if requested
if scale_features:
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

# -----------------------------
# Train model
# -----------------------------
gb_model = GradientBoostingClassifier(
    n_estimators=int(n_estimators),
    learning_rate=float(learning_rate),
    max_depth=int(max_depth),
    subsample=float(subsample),
    min_samples_split=int(min_samples_split),
    min_samples_leaf=int(min_samples_leaf),
    max_features=None if max_features == "None" else (None if max_features is None else max_features),
    random_state=random_state,
)
gb_model.fit(X_train, y_train)

# -----------------------------
# Evaluation
# -----------------------------
st.subheader("Model Evaluation")
y_pred = gb_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# Map class names back if target encoded
if target_col in label_encoders:
    classes_numeric = gb_model.classes_
    display_class_names = label_encoders[target_col].inverse_transform(classes_numeric)
else:
    display_class_names = [str(c) for c in gb_model.classes_]

cm_df = pd.DataFrame(
    cm,
    index=[f"True {c}" for c in display_class_names],
    columns=[f"Pred {c}" for c in display_class_names],
)

col1, col2 = st.columns(2)
with col1:
    st.markdown("**Confusion Matrix (interactive)**")
    st.dataframe(cm_df)
with col2:
    st.markdown("**Accuracy**")
    st.metric("Accuracy", f"{acc:.3f}")

report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
report_df = pd.DataFrame(report_dict).T
st.markdown("**Classification Report**")
st.dataframe(report_df.style.format("{:.3f}", na_rep="-"))

# -----------------------------
# Feature importances
# -----------------------------
st.subheader("Feature Importances")
try:
    importances = gb_model.feature_importances_
    fi_df = pd.DataFrame({"feature": feature_cols, "importance": importances}).sort_values("importance", ascending=False)
    st.dataframe(fi_df)
    st.bar_chart(fi_df.set_index("feature"))
except Exception:
    st.info("Feature importances not available.")

# -----------------------------
# Single-sample prediction UI
# -----------------------------
st.subheader("Try a New Prediction")
input_values = []
use_columns = len(feature_cols) <= 4
cols = st.columns(len(feature_cols)) if use_columns else None

for i, col_name in enumerate(feature_cols):
    if not np.issubdtype(df[col_name].dtype, np.number):
        unique_vals = df[col_name].unique().tolist()
        if use_columns:
            with cols[i]:
                selected = st.selectbox(col_name, unique_vals, key=f"pred_{i}")
        else:
            selected = st.selectbox(col_name, unique_vals, key=f"pred_{i}")
        encoded = label_encoders[col_name].transform([selected])[0]
        input_values.append(float(encoded))
    else:
        default_val = float(df[col_name].median()) if df[col_name].count() > 0 else 0.0
        if use_columns:
            with cols[i]:
                val = st.number_input(col_name, value=default_val, key=f"pred_{i}")
        else:
            val = st.number_input(col_name, value=default_val, key=f"pred_{i}")
        input_values.append(float(val))

if st.button("Predict"):
    sample = np.array(input_values).reshape(1, -1)
    if scale_features and scaler is not None:
        sample = scaler.transform(sample)
    pred_numeric = gb_model.predict(sample)[0]
    if target_col in label_encoders:
        pred_label = label_encoders[target_col].inverse_transform([pred_numeric])[0]
    else:
        pred_label = pred_numeric
    st.write(f"**Predicted class:** `{pred_label}`")
    if hasattr(gb_model, "predict_proba"):
        proba = gb_model.predict_proba(sample)[0]
        proba_df = pd.DataFrame([proba], columns=[str(c) for c in display_class_names])
        st.markdown("**Class probabilities:**")
        st.dataframe(proba_df)

st.caption("Model trained in-session. For production save the model and encoders.")
