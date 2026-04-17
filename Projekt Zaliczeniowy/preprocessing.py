from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE


def create_preprocessor(num_cols, cat_cols):
    """Create a column transformer with median imputation + scaling for numeric features and constant imputation + one-hot encoding for categorical features."""
    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), num_cols),

        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy='constant', fill_value='MISSING')),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]), cat_cols)
    ])
    return preprocessor


def apply_preprocessing(preprocessor, X_train, X_test, y_train, cat_cols, num_cols):
    """Fit preprocessor on training data, transform train and test sets, and return extracted feature names."""
    X_train_pre = preprocessor.fit_transform(X_train, y_train)
    X_test_pre = preprocessor.transform(X_test)

    ohe = preprocessor.named_transformers_["cat"].named_steps["ohe"]
    feature_names = num_cols + list(ohe.get_feature_names_out(cat_cols))

    print(f"Po preprocessingu: {X_train_pre.shape[1]} cech")
    print(f"  - {len(num_cols)} cech numerycznych (zstandaryzowanych)")
    print(f"  - {X_train_pre.shape[1] - len(num_cols)} cech binarnych (po one-hot encoding)")

    return X_train_pre, X_test_pre, feature_names


def apply_smote(X_train, y_train, random_state=42):
    """Apply SMOTE oversampling to balance the class distribution in the training set."""
    print(f"Przed SMOTE - klasa 0: {(y_train == 0).sum()}, klasa 1: {(y_train == 1).sum()}")

    smote = SMOTE(random_state=random_state)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    print(f"Po SMOTE - klasa 0: {(y_train_res == 0).sum()}, klasa 1: {(y_train_res == 1).sum()}")
    print("Zbalansowano dane (metoda: oversampling)")

    return X_train_res, y_train_res
