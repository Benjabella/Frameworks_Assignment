# preprocess_data.py
import pandas as pd
import numpy as np

def preprocess_and_save():
    """Preprocess the large CSV and save as optimized format"""
    print("Loading metadata.csv (this may take a while)...")
    
    # First, let's check what columns are actually available
    try:
        # Read just the first row to get column names
        sample_df = pd.read_csv('metadata.csv', nrows=1)
        available_columns = sample_df.columns.tolist()
        print(f"Available columns: {available_columns}")
        
        # Define essential columns that might exist
        possible_columns = [
            'title', 'abstract', 'authors', 'journal', 'publish_time',
            'source_x', 'has_full_text', 'cord_uid', 'source', 'doi',
            'pmcid', 'pubmed_id', 'license', 'who_covidence_id', 'arxiv_id',
            'pdf_json_files', 'pmc_json_files'
        ]
        
        # Only use columns that actually exist
        essential_columns = [col for col in possible_columns if col in available_columns]
        
        # Make sure we have at least the basic columns
        if 'title' not in essential_columns:
            essential_columns = available_columns[:8]  # Use first 8 columns as fallback
        
        print(f"Using columns: {essential_columns}")
        
        # Read the CSV with only existing columns
        df = pd.read_csv('metadata.csv', usecols=essential_columns, low_memory=False)
        
    except Exception as e:
        print(f"Error reading CSV: {e}")
        # Fallback: read without specifying columns
        df = pd.read_csv('metadata.csv', low_memory=False)
        print("Loaded all columns as fallback")
    
    print(f"✅ Data loaded successfully! Shape: {df.shape}")
    
    # Basic cleaning
    df['title'] = df['title'].fillna('Unknown Title')
    df['abstract'] = df['abstract'].fillna('')
    
    # Handle journal column if it exists
    if 'journal' in df.columns:
        df['journal'] = df['journal'].fillna('Unknown Journal')
    else:
        df['journal'] = 'Unknown Journal'
    
    # Handle publish_time and extract year
    if 'publish_time' in df.columns:
        df['publish_time'] = pd.to_datetime(df['publish_time'], errors='coerce')
        df['year'] = df['publish_time'].dt.year
        # Fill NaN years with 2020 (most common COVID year)
        df['year'] = df['year'].fillna(2020).astype(int)
    else:
        df['year'] = 2020  # Default year
    
    # Create abstract word count
    df['abstract_word_count'] = df['abstract'].apply(lambda x: len(str(x).split()))
    df['has_abstract'] = df['abstract_word_count'] > 0
    
    # Save as Parquet (much faster to load)
    df.to_parquet('metadata_optimized.parquet', index=False)
    print("✅ Optimized data saved as 'metadata_optimized.parquet'")
    
    # Also save a sample for quick testing
    df_sample = df.sample(min(10000, len(df)), random_state=42)
    df_sample.to_parquet('metadata_sample.parquet', index=False)
    print(f"✅ Sample data saved as 'metadata_sample.parquet' ({len(df_sample)} records)")
    
    # Print some basic info
    print(f"\n📊 Dataset Info:")
    print(f"Total records: {len(df):,}")
    print(f"Years range: {df['year'].min()} - {df['year'].max()}")
    print(f"Records with abstract: {df['has_abstract'].sum():,}")
    
    return df

if __name__ == "__main__":
    preprocess_and_save()