import pandas as pd
import requests
from bs4 import BeautifulSoup


def load_data_from_csv(url):
    """Load tabular data from a CSV file at the given URL."""
    df = pd.read_csv(url)
    print(f"Zrodlo 1 (CSV): {len(df)} wierszy, {len(df.columns)} kolumn")
    return df


def scrape_wikipedia_data(url):
    """Scrape the first table from a Wikipedia page and return it as a DataFrame."""
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        html = response.text.replace("“", '"').replace("”", '"')
        soup = BeautifulSoup(html, "lxml")
        tables = pd.read_html(str(soup))
        df_wiki = tables[0]
        print(f"Zrodlo 2 (Wikipedia): {len(df_wiki)} wierszy, {len(df_wiki.columns)} kolumn")
        return df_wiki
    else:
        raise Exception(f"Blad pobierania strony: {response.status_code}")


def save_raw_data(df_csv, df_wiki, data_dir="data"):
    """Save raw DataFrames as Parquet files in the specified directory."""
    import os
    os.makedirs(data_dir, exist_ok=True)
    df_csv.to_parquet(f"{data_dir}/titanic_raw_csv.parquet", index=False)
    df_wiki.to_parquet(f"{data_dir}/titanic_raw_wiki.parquet", index=False)
    print("Zapisano surowe dane do plikow .parquet")


def merge_data(df_csv, df_wiki):
    """Merge CSV and Wikipedia DataFrames on 'Name', preferring CSV Age and filling missing values from Wikipedia."""
    df_wiki_small = df_wiki[['Name', 'Age']].copy()
    df_wiki_small["Age"] = pd.to_numeric(df_wiki_small["Age"], errors="coerce")
    df = df_csv.merge(df_wiki_small, on="Name", how="left", suffixes=('', '_wiki'))
    df['Age'] = df['Age'].fillna(df['Age_wiki'])
    df.drop(columns=['Age_wiki'], inplace=True)
    print(f"Po merge: {len(df)} wierszy")
    return df
