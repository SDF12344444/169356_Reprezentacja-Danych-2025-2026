import os
import warnings
import joblib

warnings.filterwarnings('ignore')


def winsorize_outliers(df, cols, limits=(0.01, 0.99)):
    """Cap extreme values in specified columns using quantile-based winsorization."""
    df_winsorized = df.copy()
    for col in cols:
        lower = df_winsorized[col].quantile(limits[0])
        upper = df_winsorized[col].quantile(limits[1])
        df_winsorized[col] = df_winsorized[col].clip(lower, upper)
    return df_winsorized


def save_model(model, filepath):
    """Save a trained model to disk using joblib."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    print(f"Zapisano model: {filepath}")


def load_model(filepath):
    """Load a saved model from disk using joblib."""
    return joblib.load(filepath)


def create_directories(dirs=["data", "models"]):
    """Create necessary directories if they do not already exist."""
    for dir_name in dirs:
        os.makedirs(dir_name, exist_ok=True)
