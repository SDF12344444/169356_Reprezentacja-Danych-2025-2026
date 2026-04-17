from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier


def train_dummy_model(X_train, y_train, strategy='most_frequent', random_state=42):
    """Train a baseline dummy classifier for comparison with real models."""
    dummy = DummyClassifier(strategy=strategy, random_state=random_state)
    dummy.fit(X_train, y_train)
    return dummy


def train_random_forest(X_train, y_train, n_iter=15, cv=3, random_state=42):
    """Train a Random Forest classifier with randomized hyperparameter search."""
    print("\nTrenowanie Random Forest z optymalizacja...")

    rf_params = {
        'n_estimators': [100, 200],
        'max_depth': [10, 15],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }

    rf_base = RandomForestClassifier(random_state=random_state, n_jobs=-1)
    rf_random = RandomizedSearchCV(
        rf_base, rf_params, n_iter=n_iter, cv=cv,
        scoring='roc_auc', n_jobs=-1, random_state=random_state
    )
    rf_random.fit(X_train, y_train)

    rf = rf_random.best_estimator_
    print(f"Najlepsze parametry RF: {rf_random.best_params_}")
    print(f"Walidacja krzyzowa (CV={cv}) - sredni ROC AUC: {rf_random.best_score_:.4f}")

    return rf


def train_xgboost(X_train, y_train, n_iter=10, cv=3, random_state=42):
    """Train an XGBoost classifier with randomized hyperparameter search."""
    print("\nTrenowanie XGBoost z optymalizacja...")

    xgb_params = {
        'max_depth': [3, 7],
        'learning_rate': [0.05, 0.1],
        'n_estimators': [100, 200],
        'subsample': [0.8]
    }

    xgb_base = XGBClassifier(eval_metric="logloss", random_state=random_state, n_jobs=-1)
    xgb_random = RandomizedSearchCV(
        xgb_base, xgb_params, n_iter=n_iter, cv=cv,
        scoring='roc_auc', n_jobs=-1, random_state=random_state
    )
    xgb_random.fit(X_train, y_train)

    xgb = xgb_random.best_estimator_
    print(f"Najlepsze parametry XGB: {xgb_random.best_params_}")
    print(f"Walidacja krzyzowa (CV={cv}) - sredni ROC AUC: {xgb_random.best_score_:.4f}")

    return xgb


def train_ensemble(models, X_train, y_train, voting='soft'):
    """Train a soft/hard voting ensemble classifier from a list of trained models."""
    print("\nTrenowanie Ensemble Voting...")
    ensemble = VotingClassifier(estimators=models, voting=voting)
    ensemble.fit(X_train, y_train)
    return ensemble
