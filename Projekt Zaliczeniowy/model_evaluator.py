import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, accuracy_score


def evaluate_models(models, X_test, y_test):
    """Evaluate multiple models on test set and return comparison DataFrame with accuracy and ROC AUC."""
    print("\nEWALUACJA MODELI NA ZBIORZE TESTOWYM")

    results = []

    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

        roc_auc = roc_auc_score(y_test, y_proba) if y_proba is not None else 0
        acc = accuracy_score(y_test, y_pred)

        results.append({
            'Model': name,
            'Accuracy': acc,
            'ROC AUC': roc_auc
        })

        print(f"\n{name}")
        print(classification_report(y_test, y_pred))
        print(f"ROC AUC: {roc_auc:.4f}")
        print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

    results_df = pd.DataFrame(results).sort_values('ROC AUC', ascending=False)
    print("\nPODSUMOWANIE WYNIKOW")
    print(results_df.to_string(index=False))

    return results_df


def plot_feature_importance(model, feature_names, title, model_name):
    """Plot and save feature importance bar chart for tree-based models."""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        return

    feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)

    plt.figure(figsize=(10, 8))
    top_n = min(15, len(feat_imp))
    sns.barplot(x=feat_imp.values[:top_n], y=feat_imp.index[:top_n], palette='viridis')
    plt.title(f'{title} - Top {top_n} cech', fontsize=14)
    plt.xlabel('Istotność', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{model_name}_feature_importance.png', dpi=150)
    plt.close()

    print(f"\n{model_name} - Top 10 najważniejszych cech:")
    for i, (feat, imp) in enumerate(feat_imp.head(10).items(), 1):
        print(f"   {i}. {feat}: {imp:.4f}")

    return feat_imp


def plot_roc_curves(models, X_test, y_test):
    """Plot and save ROC curves comparing all models that support predict_proba."""
    plt.figure(figsize=(10, 8))
    colors = ['blue', 'green', 'red', 'orange']

    for (name, model), color in zip(models.items(), colors):
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            auc = roc_auc_score(y_test, y_proba)
            plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})', linewidth=2, color=color)

    plt.plot([0, 1], [0, 1], 'k--', label='Losowy klasyfikator (AUC = 0.5)', linewidth=1)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Krzywe ROC - Porownanie modeli', fontsize=14)
    plt.legend(loc='lower right', fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('roc_curves_comparison.png', dpi=150)
    plt.close()
    print("Zapisano wykres porownania krzywych ROC")


def analyze_misclassified(model, X_test, y_test, X_test_pre, feature_names):
    """Analyze misclassified samples and save detailed results to CSV."""
    y_pred = model.predict(X_test_pre)
    misclassified = X_test[y_pred != y_test].copy()
    misclassified['True'] = y_test[y_pred != y_test]
    misclassified['Predicted'] = y_pred[y_pred != y_test]

    print(f"\nAnaliza błędów:")
    print(f"  - Liczba błędnych klasyfikacji: {len(misclassified)} / {len(y_test)} ({len(misclassified) / len(y_test) * 100:.1f}%)")

    if len(misclassified) > 0:
        print(f"\n  - Rozkład błędów według płci i klasy:")
        pivot_table = pd.crosstab(
            misclassified['Sex'],
            misclassified['Pclass'],
            values=misclassified['True'],
            aggfunc='count',
            margins=True
        )
        print(pivot_table)

        os.makedirs("data", exist_ok=True)
        misclassified.to_csv("data/misclassified_analysis.csv", index=False)
        print(f"\n  - Zapisano szczegółową analizę do 'data/misclassified_analysis.csv'")

    return misclassified


def save_final_data(X_train_pre, y_train, feature_names, target, data_dir="data"):
    """Save the final preprocessed training data as a Parquet file."""
    final_data = pd.DataFrame(X_train_pre, columns=feature_names)
    final_data[target] = y_train.values
    final_data.to_parquet(f"{data_dir}/final_training_data.parquet", index=False)
    print("Zapisano końcowe dane treningowe do formatu Parquet")


def save_models(models, preprocessor, results_df, data_dir="models"):
    """Save trained models and preprocessor as joblib files, including the best model."""
    import joblib
    os.makedirs(data_dir, exist_ok=True)

    best_model_name = results_df.iloc[0]['Model']
    best_model = models[best_model_name]

    joblib.dump(best_model, f"{data_dir}/best_model.joblib")
    joblib.dump(models['Random Forest'], f"{data_dir}/random_forest_best.joblib")
    joblib.dump(models['XGBoost'], f"{data_dir}/xgboost_best.joblib")
    joblib.dump(models['Ensemble Voting'], f"{data_dir}/ensemble_voting.joblib")
    joblib.dump(preprocessor, f"{data_dir}/preprocessor.joblib")

    print("Zapisano modele do katalogu 'models/'")
