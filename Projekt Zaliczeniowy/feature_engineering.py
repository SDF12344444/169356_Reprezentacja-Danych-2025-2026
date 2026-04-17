import pandas as pd
import numpy as np


def create_features(df):
    """Create new features including family size, title, cabin indicators, age groups, and fare ratios."""
    # Podstawowe cechy
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    df["Title"] = df["Name"].str.extract(r',\s*(\w+)\.')
    df["SocialStatus"] = df["Pclass"].map({1: "upper", 2: "middle", 3: "lower"})

    # Cechy z kabin
    df["Cabin_Letter"] = df["Cabin"].str.extract(r'([A-Za-z])').fillna('U')
    df['Has_Cabin'] = df['Cabin'].notna().astype(int)

    # Cechy z wieku
    df["Age_Group"] = pd.cut(df["Age"], bins=[0, 12, 18, 35, 60, 100],
                             labels=["Child", "Teen", "Young Adult", "Adult", "Senior"])

    # Cechy interakcyjne
    df['Sex_Pclass'] = df['Sex'] + '_' + df['Pclass'].astype(str)

    # Cena biletu wzgledem klasy
    avg_fare_by_class = df.groupby('Pclass')['Fare'].transform('median')
    df['Fare_Ratio_To_Class'] = df['Fare'] / avg_fare_by_class
    df['Fare_Ratio_To_Class'] = df['Fare_Ratio_To_Class'].fillna(1).replace([np.inf, -np.inf], 1)

    # Kategoryczny rozmiar rodziny
    df['FamilySize_Cat'] = pd.cut(df['FamilySize'], bins=[0, 1, 3, 5, 100],
                                  labels=['Alone', 'Small', 'Medium', 'Large'])

    # Cechy rodzinne
    df['Has_Child'] = (df['Parch'] > 0).astype(int)
    df['Has_Spouse'] = (df['SibSp'] > 0).astype(int)

    print("Dodano nowe cechy:")
    print("   - Podstawowe: FamilySize, IsAlone, Title, SocialStatus, Cabin_Letter, Age_Group")
    print("   - Rozszerzone: Has_Cabin, Sex_Pclass, Fare_Ratio_To_Class, FamilySize_Cat, Has_Child, Has_Spouse")

    return df
