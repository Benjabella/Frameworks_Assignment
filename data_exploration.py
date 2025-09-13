import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import re
from collections import Counter
import streamlit as st

# Set style for visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_and_explore_data():
    """Load the metadata.csv file and perform basic exploration"""
    print("Loading metadata.csv...")
    
    # Load the data
    try:
        df = pd.read_csv('metadata.csv', low_memory=False)
        print("✅ Data loaded successfully!")
    except FileNotFoundError:
        print("❌ metadata.csv not found. Please download it from CORD-19 dataset")
        return None
    
    # Basic exploration
    print(f"\n📊 DataFrame Dimensions: {df.shape[0]} rows, {df.shape[1]} columns")
    
    print("\n📋 First few rows:")
    print(df.head())
    
    print("\n🔍 Data Types:")
    print(df.dtypes)
    
    print("\n❓ Missing Values:")
    missing_values = df.isnull().sum()
    print(missing_values[missing_values > 0])
    
    print("\n📈 Basic Statistics for Numerical Columns:")
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    if len(numerical_cols) > 0:
        print(df[numerical_cols].describe())
    else:
        print("No numerical columns found")
    
    return df

if __name__ == "__main__":
    df = load_and_explore_data()