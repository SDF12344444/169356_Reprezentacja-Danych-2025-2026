import pandas as pd
from utils import winsorize_outliers
from data_loader import load_data_from_csv, scrape_wikipedia_data, save_raw_data, merge_data
from feature_engineering import create_features
from data_preprocessor import clean_data, split_data
from preprocessing import create_preprocessor, apply_preprocessing, apply_smote
from model_trainer import train_dummy_model, train_random_forest, train_xgboost, train_ensemble
from model_evaluator import (evaluate_models, plot_feature_importance, plot_roc_curves,
                             analyze_misclassified, save_models, save_final_data)



if __name__ == "__main__":
    df_csv = load_data_from_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")
    df_wiki = scrape_wikipedia_data("https://en.wikipedia.org/wiki/List_of_Titanic_passengers")
    save_raw_data(df_csv, df_wiki)

    df = merge_data(df_csv, df_wiki)
    df = create_features(df)
    df.to_parquet("data/titanic_prepared.parquet", index=False)
    print("Zapisano przygotowane dane")

    df = pd.read_parquet("data/titanic_prepared.parquet")
    print("Wczytano dane: titanic_prepared.parquet")

    target = "Survived"
    num_cols = ["Age", "Fare", "Fare_Ratio_To_Class", "SibSp", "Parch", "FamilySize"]
    cat_cols = ["Pclass", "Sex", "Sex_Pclass", "Embarked", "Title", "SocialStatus",
                "IsAlone", "Cabin_Letter", "Age_Group", "Has_Cabin", "FamilySize_Cat",
                "Has_Child", "Has_Spouse"]

    df = clean_data(df, target, num_cols, cat_cols)

    X_train, X_test, y_train, y_test = split_data(df, num_cols, cat_cols, target)

    X_train = winsorize_outliers(X_train, num_cols, limits=(0.01, 0.99))
    print(f"\nZastosowano winsoryzacje (percentyle 1-99) aby zachować wszystkie {len(X_train)} wiersze")


    preprocessor = create_preprocessor(num_cols, cat_cols)
    X_train_pre, X_test_pre, feature_names = apply_preprocessing(preprocessor, X_train, X_test, y_train, cat_cols,
                                                                 num_cols)

    X_train_res, y_train_res = apply_smote(X_train_pre, y_train)


    dummy = train_dummy_model(X_train_res, y_train_res)
    rf = train_random_forest(X_train_res, y_train_res)
    xgb = train_xgboost(X_train_res, y_train_res)

    models_dict = [('Random Forest', rf), ('XGBoost', xgb)]
    ensemble = train_ensemble(models_dict, X_train_res, y_train_res)

    all_models = {
        'Dummy (Baseline)': dummy,
        'Random Forest': rf,
        'XGBoost': xgb,
        'Ensemble Voting': ensemble
    }

    results_df = evaluate_models(all_models, X_test_pre, y_test)

    plot_feature_importance(rf, feature_names, "Random Forest", "rf")
    plot_feature_importance(xgb, feature_names, "XGBoost", "xgb")

    best_model_name = results_df.iloc[0]['Model']
    best_model = all_models[best_model_name]
    print(f"Analiza dla najlepszego modelu: {best_model_name}")
    analyze_misclassified(best_model, X_test, y_test, X_test_pre, feature_names)

    plot_roc_curves(all_models, X_test_pre, y_test)

    save_models(all_models, preprocessor, results_df)
    save_final_data(X_train_pre, y_train, feature_names, target)

    print('\n' + f"Najlepszy model: {results_df.iloc[0]['Model']}")
    print(f"Najwyzszy ROC AUC: {results_df.iloc[0]['ROC AUC']:.4f}")
