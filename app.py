# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from analysis_visualization import analyze_publication_trends, analyze_journals, analyze_titles

# Set page configuration
st.set_page_config(
    page_title="CORD-19 Data Explorer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    """Main Streamlit application"""
    st.title("📊 CORD-19 COVID-19 Research Data Explorer")
    st.write("""
    Explore patterns and trends in COVID-19 research papers from the CORD-19 dataset.
    This app provides insights into publication trends, journal distribution, and common themes in research titles.
    """)
    
    # Load data
    @st.cache_data
    def load_data():
        try:
            df = pd.read_csv('metadata.csv', low_memory=False)
            # Basic cleaning for the app
            df['publish_time'] = pd.to_datetime(df['publish_time'], errors='coerce')
            df['year'] = df['publish_time'].dt.year
            df['abstract'] = df['abstract'].fillna('')
            df['title'] = df['title'].fillna('Unknown Title')
            df['journal'] = df['journal'].fillna('Unknown Journal')
            return df
        except FileNotFoundError:
            st.error("❌ metadata.csv file not found. Please ensure it's in the same directory.")
            return None
    
    df = load_data()
    
    if df is None:
        return
    
    # Sidebar for filters
    st.sidebar.header("🔍 Filters")
    
    # Year range filter
    min_year = int(df['year'].min())
    max_year = int(df['year'].max())
    year_range = st.sidebar.slider(
        "Select Year Range",
        min_value=min_year,
        max_value=max_year,
        value=(2020, max_year)
    )
    
    # Journal filter
    journals = ['All'] + sorted(df['journal'].unique().tolist())
    selected_journal = st.sidebar.selectbox("Select Journal", journals)
    
    # Apply filters
    filtered_df = df[(df['year'] >= year_range[0]) & (df['year'] <= year_range[1])]
    if selected_journal != 'All':
        filtered_df = filtered_df[filtered_df['journal'] == selected_journal]
    
    # Display basic stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Papers", len(filtered_df))
    with col2:
        st.metric("Years Covered", f"{year_range[0]} - {year_range[1]}")
    with col3:
        st.metric("Unique Journals", filtered_df['journal'].nunique())
    with col4:
        st.metric("Papers with Abstracts", filtered_df['abstract'].str.len().gt(0).sum())
    
    # Tabs for different analyses
    tab1, tab2, tab3, tab4 = st.tabs(["Publication Trends", "Journal Analysis", "Title Analysis", "Sample Data"])
    
    with tab1:
        st.header("📈 Publication Trends")
        fig, ax = plt.subplots(figsize=(10, 6))
        yearly_counts = filtered_df['year'].value_counts().sort_index()
        yearly_counts.plot(kind='bar', ax=ax, color='skyblue')
        ax.set_title('Publications by Year')
        ax.set_xlabel('Year')
        ax.set_ylabel('Number of Publications')
        st.pyplot(fig)
    
    with tab2:
        st.header("📚 Journal Analysis")
        top_n = st.slider("Number of top journals to show", 5, 20, 10)
        top_journals = filtered_df['journal'].value_counts().head(top_n)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        top_journals.plot(kind='barh', ax=ax, color='lightgreen')
        ax.set_title(f'Top {top_n} Journals')
        ax.set_xlabel('Number of Publications')
        st.pyplot(fig)
    
    with tab3:
        st.header("🔤 Title Word Analysis")
        
        # Word frequency analysis
        all_titles = ' '.join(filtered_df['title'].astype(str))
        words = re.findall(r'\b[a-zA-Z]{3,}\b', all_titles.lower())
        stop_words = {'the', 'and', 'of', 'in', 'to', 'a', 'for', 'with', 'on', 'by'}
        filtered_words = [word for word in words if word not in stop_words]
        
        from collections import Counter
        word_freq = Counter(filtered_words)
        top_words = word_freq.most_common(15)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        words, counts = zip(*top_words)
        ax.barh(words, counts, color='lightcoral')
        ax.set_title('Top Words in Titles')
        ax.set_xlabel('Frequency')
        st.pyplot(fig)
    
    with tab4:
        st.header("📋 Sample Data")
        sample_size = st.slider("Number of rows to show", 5, 50, 10)
        st.dataframe(filtered_df[['title', 'journal', 'year', 'publish_time']].head(sample_size))
    
    # Additional information
    st.sidebar.header("ℹ️ About")
    st.sidebar.info("""
    This app analyzes the CORD-19 dataset metadata to provide insights into COVID-19 research trends.
    Data source: [CORD-19 on Kaggle](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge)
    """)

if __name__ == "__main__":
    main()