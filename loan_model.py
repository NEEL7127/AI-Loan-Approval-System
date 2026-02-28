import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

DATA_PATH = "loan_data.csv"
MODEL_PATH = "loan_model.pkl"
PREDICTIONS_PATH = "improved_predictions.csv"
CLASS_PLOT_PATH = "loan_status_distribution.png"
CORR_PLOT_PATH = "feature_correlation_heatmap.png"
TARGET_COL = "Loan_Status"

CATEGORICAL_COLS = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Property_Area",
    TARGET_COL,
]


def run_eda(data: pd.DataFrame) -> None:
    print("\nDataset Info:")
    data.info()

    print("\nDataset Summary:")
    print(data.describe(include="all"))

    print("\nMissing Values:")
    print(data.isnull().sum())

    print("\nClass Distribution (Loan_Status):")
    print(data[TARGET_COL].value_counts())
    print("\nClass Distribution (%):")
    print((data[TARGET_COL].value_counts(normalize=True) * 100).round(2))


def encode_categorical(data: pd.DataFrame):
    label_encoders = {}
    encoded = data.copy()

    for col in CATEGORICAL_COLS:
        encoder = LabelEncoder()
        encoded[col] = encoder.fit_transform(encoded[col])
        label_encoders[col] = encoder

    return encoded, label_encoders


def save_eda_plots(encoded_data: pd.DataFrame) -> None:
    plt.figure(figsize=(6, 4))
    sns.countplot(x=TARGET_COL, data=encoded_data)
    plt.title("Loan Status Distribution (Encoded)")
    plt.tight_layout()
    plt.savefig(CLASS_PLOT_PATH)
    plt.close()

    corr = encoded_data.corr(numeric_only=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, cmap="Blues", annot=False)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(CORR_PLOT_PATH)
    plt.close()

    print(f"Saved class distribution plot to {CLASS_PLOT_PATH}")
    print(f"Saved correlation heatmap to {CORR_PLOT_PATH}")


def main() -> None:
    data = pd.read_csv(DATA_PATH)
    run_eda(data)

    # Modern missing-value handling
    data = data.ffill().bfill()

    encoded_data, label_encoders = encode_categorical(data)

    X = encoded_data.drop(TARGET_COL, axis=1)
    y = encoded_data[TARGET_COL]
    feature_columns = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    log_model = LogisticRegression(random_state=42, max_iter=1000)
    log_model.fit(X_train_scaled, y_train)
    log_pred = log_model.predict(X_test_scaled)

    dt_model = DecisionTreeClassifier(random_state=42)
    dt_model.fit(X_train_scaled, y_train)
    dt_pred = dt_model.predict(X_test_scaled)

    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 5, 10],
    }

    grid = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid,
        cv=3,
        scoring="roc_auc",
        n_jobs=-1,
    )
    grid.fit(X_train_scaled, y_train)

    print("\nBest Parameters:", grid.best_params_)

    rf_model = grid.best_estimator_
    rf_pred = rf_model.predict(X_test_scaled)

    probs = rf_model.predict_proba(X_test_scaled)[:, 1]
    print("AUC Score:", roc_auc_score(y_test, probs))

    print("\nLogistic Regression Accuracy:", accuracy_score(y_test, log_pred))
    print("Decision Tree Accuracy:", accuracy_score(y_test, dt_pred))
    print("Random Forest Accuracy:", accuracy_score(y_test, rf_pred))

    cm = confusion_matrix(y_test, rf_pred)
    print("\nConfusion Matrix (Random Forest):\n", cm)

    print("\nFeature Importances (Random Forest):")
    for feature, importance in sorted(
        zip(feature_columns, rf_model.feature_importances_),
        key=lambda item: item[1],
        reverse=True,
    ):
        print(f"{feature}: {importance:.4f}")

    output = pd.DataFrame(
        {
            "Actual": label_encoders[TARGET_COL].inverse_transform(y_test),
            "Predicted_RF": label_encoders[TARGET_COL].inverse_transform(rf_pred),
        }
    )
    output.to_csv(PREDICTIONS_PATH, index=False)
    print(f"\nPredictions saved to {PREDICTIONS_PATH}")

    artifact = {
        "model": rf_model,
        "scaler": scaler,
        "encoders": label_encoders,
        "feature_columns": feature_columns,
        "target_column": TARGET_COL,
    }
    joblib.dump(artifact, MODEL_PATH)
    print(f"Saved trained artifact to {MODEL_PATH}")

    save_eda_plots(encoded_data)


if __name__ == "__main__":
    main()
