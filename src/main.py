# main.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import os

def detect_columns():
    """Detect available columns in the metadata file"""
    print("🔍 Detecting available columns...")
    try:
        # Read just the first row to get column names
        sample_df = pd.read_csv('data/metadata.csv', nrows=1)
        available_columns = sample_df.columns.tolist()
        print(f"✅ Found {len(available_columns)} columns")
        return available_columns
    except FileNotFoundError:
        print("❌ metadata.csv not found. Please download it from CORD-19 dataset")
        return None
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return None

def load_and_explore_data():
    """Load the metadata.csv file and perform basic exploration"""
    print("📂 Loading metadata.csv...")
    
    # First detect available columns
    available_columns = detect_columns()
    if available_columns is None:
        return None
    
    # Load the data with available columns
    try:
        df = pd.read_csv('data/metadata.csv', usecols=available_columns, low_memory=False)
        print("✅ Data loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        # Fallback: try without specifying columns
        try:
            df = pd.read_csv('data/metadata.csv', low_memory=False)
            print("✅ Data loaded with fallback method")
        except Exception as e2:
            print(f"❌ Failed to load data: {e2}")
            return None
    
    # Basic exploration
    print(f"\n📊 DataFrame Dimensions: {df.shape[0]:,} rows, {df.shape[1]} columns")
    
    print("\n📋 First few rows:")
    print(df.head(3))
    
    print("\n🔍 Data Types:")
    print(df.dtypes)
    
    print("\n❓ Missing Values (top 10 columns with most missing values):")
    missing_values = df.isnull().sum()
    missing_percentage = (missing_values / len(df)) * 100
    missing_data = pd.DataFrame({
        'Missing_Count': missing_values,
        'Missing_Percentage': missing_percentage
    })
    print(missing_data.sort_values('Missing_Count', ascending=False).head(10))
    
    return df

def clean_and_prepare_data(df):
    """Clean and prepare the CORD-19 dataset for analysis"""
    print("\n🧹 Cleaning and preparing data...")
    
    # Create a copy to avoid modifying the original
    df_clean = df.copy()
    
    # Handle missing values in key columns
    if 'title' in df_clean.columns:
        df_clean['title'] = df_clean['title'].fillna('Unknown Title')
    else:
        df_clean['title'] = 'Unknown Title'
    
    if 'abstract' in df_clean.columns:
        df_clean['abstract'] = df_clean['abstract'].fillna('')
    else:
        df_clean['abstract'] = ''
    
    if 'journal' in df_clean.columns:
        df_clean['journal'] = df_clean['journal'].fillna('Unknown Journal')
    else:
        df_clean['journal'] = 'Unknown Journal'
    
    # Handle publish_time and extract year
    if 'publish_time' in df_clean.columns:
        df_clean['publish_time'] = pd.to_datetime(df_clean['publish_time'], errors='coerce')
        df_clean['year'] = df_clean['publish_time'].dt.year
        # Fill NaN years with 2020 (most common COVID year)
        df_clean['year'] = df_clean['year'].fillna(2020).astype(int)
    else:
        df_clean['year'] = 2020  # Default year
    
    # Create new features
    df_clean['abstract_word_count'] = df_clean['abstract'].apply(lambda x: len(str(x).split()))
    df_clean['title_word_count'] = df_clean['title'].apply(lambda x: len(str(x).split()))
    df_clean['has_abstract'] = df_clean['abstract_word_count'] > 0
    
    print(f"✅ Data cleaning complete! Final shape: {df_clean.shape}")
    return df_clean

def analyze_publication_trends(df):
    """Analyze publication trends over time"""
    print("\n📈 Analyzing publication trends...")
    
    # Count papers by year
    yearly_counts = df['year'].value_counts().sort_index()
    
    # Plot publications over time
    plt.figure(figsize=(12, 6))
    yearly_counts.plot(kind='bar', color='skyblue', alpha=0.8)
    plt.title('Number of COVID-19 Publications by Year', fontsize=16, pad=20)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Number of Publications', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('publications_by_year.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return yearly_counts

def analyze_journals(df):
    """Analyze publications by journal"""
    print("📚 Analyzing journal distribution...")
    
    # Get top journals
    top_journals = df['journal'].value_counts().head(15)
    
    # Plot top journals
    plt.figure(figsize=(12, 8))
    top_journals.plot(kind='barh', color='lightgreen', alpha=0.8)
    plt.title('Top 15 Journals Publishing COVID-19 Research', fontsize=16, pad=20)
    plt.xlabel('Number of Publications', fontsize=12)
    plt.ylabel('Journal', fontsize=12)
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig('top_journals.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return top_journals

def analyze_titles(df):
    """Analyze word frequencies in titles"""
    print("🔤 Analyzing title word frequencies...")
    
    # Combine all titles
    all_titles = ' '.join(df['title'].astype(str))
    
    # Clean text
    words = re.findall(r'\b[a-zA-Z]{3,}\b', all_titles.lower())
    
    # Remove common stop words and COVID-related terms
    stop_words = {
        'the', 'and', 'of', 'in', 'to', 'a', 'for', 'with', 'on', 'by', 
        'as', 'an', 'from', 'that', 'this', 'is', 'are', 'was', 'were',
        'covid', '19', 'sars', 'cov', '2', 'coronavirus', 'study', 'research',
        'based', 'using', 'analysis', 'during', 'among', 'between'
    }
    filtered_words = [word for word in words if word not in stop_words]
    
    # Get word frequencies
    word_freq = Counter(filtered_words)
    top_words = word_freq.most_common(20)
    
    # Plot top words
    plt.figure(figsize=(12, 8))
    words, counts = zip(*top_words)
    plt.barh(words, counts, color='lightcoral', alpha=0.8)
    plt.title('Top 20 Words in Paper Titles (excluding common terms)', fontsize=16, pad=20)
    plt.xlabel('Frequency', fontsize=12)
    plt.ylabel('Words', fontsize=12)
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig('top_title_words.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return word_freq

def analyze_abstracts(df):
    """Analyze abstract statistics"""
    print("📝 Analyzing abstract statistics...")
    
    # Abstract length distribution
    plt.figure(figsize=(10, 6))
    df['abstract_word_count'].hist(bins=50, color='purple', alpha=0.7, edgecolor='black')
    plt.title('Distribution of Abstract Word Count', fontsize=16, pad=20)
    plt.xlabel('Word Count', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('abstract_length_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Papers with/without abstracts
    has_abstract = df['has_abstract'].sum()
    no_abstract = len(df) - has_abstract
    
    plt.figure(figsize=(8, 6))
    plt.pie([has_abstract, no_abstract], 
            labels=['With Abstract', 'Without Abstract'],
            colors=['lightblue', 'lightcoral'],
            autopct='%1.1f%%',
            startangle=90)
    plt.title('Papers with Abstracts', fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig('abstract_presence.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return {
        'with_abstract': has_abstract,
        'without_abstract': no_abstract,
        'avg_word_count': df['abstract_word_count'].mean()
    }

def generate_summary_report(df, results):
    """Generate a comprehensive summary report"""
    print("\n" + "=" * 60)
    print("📊 COMPREHENSIVE ANALYSIS SUMMARY")
    print("=" * 60)
    
    print(f"📈 Total papers analyzed: {len(df):,}")
    print(f"📅 Time span: {int(df['year'].min())} - {int(df['year'].max())}")
    print(f"🏛️  Number of unique journals: {df['journal'].nunique():,}")
    print(f"📝 Papers with abstracts: {results['abstract_stats']['with_abstract']:,} ({results['abstract_stats']['with_abstract']/len(df)*100:.1f}%)")
    print(f"📊 Average abstract length: {results['abstract_stats']['avg_word_count']:.1f} words")
    
    print(f"\n📈 Publication Trends:")
    for year, count in results['yearly_counts'].items():
        print(f"   {year}: {count:,} papers")
    
    print(f"\n🏆 Top 5 Journals:")
    for i, (journal, count) in enumerate(results['top_journals'].head(5).items(), 1):
        print(f"   {i}. {journal}: {count:,} papers")
    
    print(f"\n🔤 Top 5 Title Words:")
    for i, (word, count) in enumerate(results['word_frequencies'].most_common(5), 1):
        print(f"   {i}. '{word}': {count:,} occurrences")
    
    print("\n💾 Visualizations saved:")
    print("   - publications_by_year.png")
    print("   - top_journals.png")
    print("   - top_title_words.png")
    print("   - abstract_length_distribution.png")
    print("   - abstract_presence.png")
    
    print("\n🎉 Analysis complete! Run 'streamlit run app_optimized.py' to launch the web application.")

def main():
    """Main function to run the complete analysis"""
    print("=" * 60)
    print("CORD-19 COVID-19 RESEARCH DATA ANALYSIS")
    print("=" * 60)
    
    # Step 1: Load and explore data
    df = load_and_explore_data()
    if df is None:
        return
    
    # Step 2: Clean and prepare data
    df_clean = clean_and_prepare_data(df)
    
    # Step 3: Run analyses
    print("\n🚀 Starting complete analysis...")
    
    results = {}
    from src.analysis import analysis_visualization
    results['yearly_counts'] = analysis_visualization.analyze_publication_trends(df_clean)
    results['top_journals'] = analysis_visualization.analyze_journals(df_clean)
    results['word_frequencies'] = analysis_visualization.analyze_titles(df_clean)
    results['abstract_stats'] = analysis_visualization.analyze_abstracts(df_clean)
    
    # Step 4: Save cleaned data
    try:
        df_clean.to_parquet('data/cleaned_metadata.parquet', index=False)
        print("💾 Cleaned data saved as 'data/cleaned_metadata.parquet'")
    except Exception as e:
        print(f"⚠️  Could not save as parquet: {e}")
        df_clean.to_csv('data/cleaned_metadata.csv', index=False)
        print("💾 Cleaned data saved as 'data/cleaned_metadata.csv'")
    
    # Step 5: Generate summary report
    generate_summary_report(df_clean, results)
    
    # Step 6: Suggest next steps
    print("\n" + "=" * 60)
    print("🚀 NEXT STEPS")
    print("=" * 60)
    print("1. Run the Streamlit app: streamlit run app_optimized.py")
    print("2. Explore interactive visualizations")
    print("3. Check the generated PNG files for static charts")
    print("4. Use data/cleaned_metadata.parquet for faster future analysis")

if __name__ == "__main__":
    main()