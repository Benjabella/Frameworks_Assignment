# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from collections import Counter
import numpy as np
import os

# Set page configuration
st.set_page_config(
    page_title="CORD-19 Data Explorer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_optimized_data(use_sample=False):
    """Load optimized data format"""
    try:
        if use_sample:
            df = pd.read_parquet('data/metadata_sample.parquet')
            st.sidebar.info("📊 Using sample data (10k records)")
        else:
            df = pd.read_parquet('data/metadata_optimized.parquet')
            st.sidebar.info("📊 Using full dataset")
        return df
    except FileNotFoundError:
        st.error("Optimized data files not found. Please run preprocess_data.py first.")
        return None

def safe_value_counts(series, default_name="Unknown"):
    """Safe value counts handling NaN values"""
    if series.isna().all():
        return pd.Series([len(series)], index=[default_name])
    return series.fillna(default_name).value_counts()

def main():
    st.title("📊 CORD-19 COVID-19 Research Data Explorer")
    st.write("Exploring patterns in COVID-19 research publications")
    
    # Sidebar options
    st.sidebar.header("⚡ Performance Options")
    use_sample = st.sidebar.checkbox("Use sample data (faster)", value=True)
    
    # Load data with loading indicator
    with st.spinner("Loading data..."):
        df = load_optimized_data(use_sample)
    
    if df is None:
        st.stop()
    
    # Sidebar filters
    st.sidebar.header("🔍 Filters")
    
    # Year range filter
    min_year, max_year = int(df['year'].min()), int(df['year'].max())
    year_range = st.sidebar.slider(
        "Select Year Range", 
        min_year, max_year, 
        (max(2019, min_year), max_year)
    )
    
    # Apply filters
    filtered_df = df[(df['year'] >= year_range[0]) & (df['year'] <= year_range[1])]
    
    # Display statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Papers", f"{len(filtered_df):,}")
    with col2:
        st.metric("Year Range", f"{year_range[0]} - {year_range[1]}")
    with col3:
        unique_journals = filtered_df['journal'].nunique()
        st.metric("Unique Journals", f"{unique_journals:,}")
    with col4:
        has_abstract = filtered_df['has_abstract'].sum()
        st.metric("With Abstracts", f"{has_abstract:,}")
    
    # Tabs for different analyses
    tab1, tab2, tab3, tab4 = st.tabs(["Publication Trends", "Journal Analysis", "Title Analysis", "Data Sample"])
    
    with tab1:
        st.header("📈 Publication Trends Over Time")
        
        yearly_counts = safe_value_counts(filtered_df['year'], "Unknown")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        yearly_counts.sort_index().plot(kind='bar', ax=ax, color='skyblue', alpha=0.8)
        ax.set_title('Number of Publications by Year', fontsize=16, pad=20)
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Number of Publications', fontsize=12)
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        
        # Monthly trends if we have publish_time
        if 'publish_time' in filtered_df.columns and not filtered_df['publish_time'].isna().all():
            monthly_data = filtered_df.dropna(subset=['publish_time']).copy()
            monthly_data['month'] = monthly_data['publish_time'].dt.to_period('M')
            monthly_counts = monthly_data['month'].astype(str).value_counts().sort_index()
            
            fig2, ax2 = plt.subplots(figsize=(14, 6))
            monthly_counts.plot(ax=ax2, color='orange', linewidth=2.5)
            ax2.set_title('Monthly Publication Trends', fontsize=16, pad=20)
            ax2.set_xlabel('Month', fontsize=12)
            ax2.set_ylabel('Number of Publications', fontsize=12)
            ax2.tick_params(axis='x', rotation=45)
            plt.tight_layout()
            st.pyplot(fig2)
    
    with tab2:
        st.header("📚 Journal Analysis")
        
        top_n = st.slider("Number of top journals to show", 5, 30, 15, key="journals_slider")
        journal_counts = safe_value_counts(filtered_df['journal']).head(top_n)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        journal_counts.plot(kind='barh', ax=ax, color='lightgreen', alpha=0.8)
        ax.set_title(f'Top {top_n} Journals by Publication Count', fontsize=16, pad=20)
        ax.set_xlabel('Number of Publications', fontsize=12)
        ax.set_ylabel('Journal', fontsize=12)
        plt.tight_layout()
        st.pyplot(fig)
        
        # Journal statistics
        st.subheader("Journal Statistics")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Total journals:** {unique_journals:,}")
            st.write(f"**Top journal:** {journal_counts.index[0]} ({journal_counts.iloc[0]:,} papers)")
        with col2:
            avg_papers = len(filtered_df) / unique_journals if unique_journals > 0 else 0
            st.write(f"**Average papers per journal:** {avg_papers:.1f}")
    
    with tab3:
        st.header("🔤 Title Word Analysis")
        
        # Combine all titles
        all_titles = ' '.join(filtered_df['title'].astype(str))
        
        # Extract words and clean
        words = re.findall(r'\b[a-zA-Z]{3,}\b', all_titles.lower())
        stop_words = {'the', 'and', 'of', 'in', 'to', 'a', 'for', 'with', 'on', 'by', 
                     'as', 'an', 'from', 'that', 'this', 'is', 'are', 'was', 'were',
                     'covid', '19', 'sars', 'cov', '2', 'coronavirus', 'study', 'research'}
        filtered_words = [word for word in words if word not in stop_words]
        
        # Get top words
        word_freq = Counter(filtered_words)
        top_n_words = st.slider("Number of top words to show", 10, 30, 20, key="words_slider")
        top_words = word_freq.most_common(top_n_words)
        
        # Plot
        if top_words:
            fig, ax = plt.subplots(figsize=(12, 8))
            words, counts = zip(*top_words)
            ax.barh(words, counts, color='lightcoral', alpha=0.8)
            ax.set_title(f'Top {top_n_words} Words in Paper Titles', fontsize=16, pad=20)
            ax.set_xlabel('Frequency', fontsize=12)
            ax.set_ylabel('Words', fontsize=12)
            plt.tight_layout()
            st.pyplot(fig)
        
        # Word statistics
        st.subheader("Word Statistics")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Total unique words:** {len(word_freq):,}")
            if top_words:
                st.write(f"**Most frequent word:** '{top_words[0][0]}' ({top_words[0][1]:,} times)")
        with col2:
            total_words = len(filtered_words)
            st.write(f"**Total words analyzed:** {total_words:,}")
    
    with tab4:
        st.header("📋 Data Sample")
        
        sample_size = st.slider("Number of rows to show", 5, 50, 10, key="sample_slider")
        
        # Select columns to display
        display_columns = ['title', 'journal', 'year']
        if 'authors' in filtered_df.columns:
            display_columns.append('authors')
        if 'publish_time' in filtered_df.columns:
            display_columns.append('publish_time')
        
        # Show sample data
        sample_data = filtered_df[display_columns].head(sample_size)
        st.dataframe(
            sample_data,
            use_container_width=True,
            height=400
        )
        
        # Download option
        csv = sample_data.to_csv(index=False)
        st.download_button(
            label="Download sample as CSV",
            data=csv,
            file_name="cord19_sample.csv",
            mime="text/csv"
        )
    
    # Footer
    st.sidebar.header("ℹ️ About")
    st.sidebar.info("""
    **CORD-19 Data Explorer**  
    Analyzing COVID-19 research trends from the CORD-19 dataset.
    
    **Data Source:** [Allen Institute for AI](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge)
    
    **Features:**  
    • Publication trends analysis  
    • Journal distribution  
    • Title word frequency  
    • Interactive filtering
    """)

if __name__ == "__main__":
    main()