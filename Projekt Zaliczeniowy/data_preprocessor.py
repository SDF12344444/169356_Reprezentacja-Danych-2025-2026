from sklearn.model_selection import train_test_split


def clean_data(df, target, num_cols, cat_cols):
    """Remove duplicate rows and drop missing values in the target column."""
    df = df.drop_duplicates().dropna(subset=[target]).reset_index(drop=True)
    print(f"Po usunieciu duplikatow i brakow: {len(df)} wierszy")
    return df


def split_data(df, num_cols, cat_cols, target, test_size=0.2, random_state=42):
    """Split data into train/test sets with stratified sampling on the target variable."""
    X = df[num_cols + cat_cols]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"Zbior treningowy: {len(X_train)} wierszy")
    print(f"Zbior testowy: {len(X_test)} wierszy")
    print(f"Rozklad klas (train): {y_train.value_counts().to_dict()}")
    print(f"Rozklad klas (test): {y_test.value_counts().to_dict()}")

    return X_train, X_test, y_train, y_test
