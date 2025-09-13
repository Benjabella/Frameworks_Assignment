# data_cleaning.py
import pandas as pd
import numpy as np
from datetime import datetime

def clean_and_prepare_data(df):
    """Clean and prepare the CORD-19 dataset for analysis"""
    print("🧹 Cleaning and preparing data...")
    
    # Create a copy to avoid modifying the original
    df_clean = df.copy()
    
    # Handle missing values
    print("Handling missing values...")
    
    # Columns to keep (focus on important ones for analysis)
    important_cols = [
        'title', 'abstract', 'authors', 'journal', 'publish_time',
        'source_x', 'has_full_text', 'cord_uid'
    ]
    
    # Keep only columns that exist in the dataframe
    available_cols = [col for col in important_cols if col in df_clean.columns]
    df_clean = df_clean[available_cols]
    
    # Fill missing titles with 'Unknown Title'
    df_clean['title'] = df_clean['title'].fillna('Unknown Title')
    
    # Fill missing abstracts with empty string
    df_clean['abstract'] = df_clean['abstract'].fillna('')
    
    # Convert publish_time to datetime and extract year
    print("Processing dates...")
    df_clean['publish_time'] = pd.to_datetime(df_clean['publish_time'], errors='coerce')
    df_clean['year'] = df_clean['publish_time'].dt.year
    
    # Create new features
    print("Creating new features...")
    df_clean['abstract_word_count'] = df_clean['abstract'].apply(lambda x: len(str(x).split()))
    df_clean['title_word_count'] = df_clean['title'].apply(lambda x: len(str(x).split()))
    df_clean['has_abstract'] = df_clean['abstract_word_count'] > 0
    
    # Clean journal names
    df_clean['journal'] = df_clean['journal'].fillna('Unknown Journal')
    df_clean['journal'] = df_clean['journal'].str.strip().str.title()
    
    print(f"✅ Data cleaning complete! Final shape: {df_clean.shape}")
    return df_clean

def analyze_missing_data(df):
    """Analyze and report on missing data patterns"""
    print("\n🔍 Missing Data Analysis:")
    
    missing_percent = (df.isnull().sum() / len(df)) * 100
    missing_data = missing_percent[missing_percent > 0].sort_values(ascending=False)
    
    print("Columns with missing values (percentage):")
    for col, percent in missing_data.items():
        print(f"{col}: {percent:.2f}%")
    
    return missing_data