# analysis_visualization.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
from wordcloud import WordCloud
import numpy as np

def analyze_publication_trends(df):
    """Analyze publication trends over time"""
    print("📈 Analyzing publication trends...")
    
    # Count papers by year
    yearly_counts = df['year'].value_counts().sort_index()
    
    # Plot publications over time
    plt.figure(figsize=(12, 6))
    yearly_counts.plot(kind='bar', color='skyblue')
    plt.title('Number of COVID-19 Publications by Year', fontsize=16)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Number of Publications', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('publications_by_year.png')
    plt.show()
    
    return yearly_counts

def analyze_journals(df):
    """Analyze publications by journal"""
    print("📚 Analyzing journal distribution...")
    
    # Get top journals
    top_journals = df['journal'].value_counts().head(15)
    
    # Plot top journals
    plt.figure(figsize=(12, 8))
    top_journals.plot(kind='barh', color='lightgreen')
    plt.title('Top 15 Journals Publishing COVID-19 Research', fontsize=16)
    plt.xlabel('Number of Publications', fontsize=12)
    plt.ylabel('Journal', fontsize=12)
    plt.tight_layout()
    plt.savefig('top_journals.png')
    plt.show()
    
    return top_journals

def analyze_titles(df):
    """Analyze word frequencies in titles"""
    print("🔤 Analyzing title word frequencies...")
    
    # Combine all titles
    all_titles = ' '.join(df['title'].dropna().astype(str))
    
    # Clean text
    words = re.findall(r'\b[a-zA-Z]{3,}\b', all_titles.lower())
    
    # Remove common stop words
    stop_words = {'the', 'and', 'of', 'in', 'to', 'a', 'for', 'with', 'on', 'by', 
                 'as', 'an', 'from', 'that', 'this', 'is', 'are', 'was', 'were'}
    filtered_words = [word for word in words if word not in stop_words]
    
    # Get word frequencies
    word_freq = Counter(filtered_words)
    top_words = word_freq.most_common(20)
    
    # Plot top words
    plt.figure(figsize=(12, 8))
    words, counts = zip(*top_words)
    plt.barh(words, counts, color='lightcoral')
    plt.title('Top 20 Words in Paper Titles', fontsize=16)
    plt.xlabel('Frequency', fontsize=12)
    plt.ylabel('Words', fontsize=12)
    plt.tight_layout()
    plt.savefig('top_title_words.png')
    plt.show()
    
    # Create word cloud
    plt.figure(figsize=(12, 8))
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(filtered_words))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud of Paper Titles', fontsize=16)
    plt.tight_layout()
    plt.savefig('title_wordcloud.png')
    plt.show()
    
    return word_freq

def analyze_sources(df):
    """Analyze paper distribution by source"""
    print("🌐 Analyzing source distribution...")
    
    if 'source_x' in df.columns:
        source_counts = df['source_x'].value_counts().head(10)
        
        plt.figure(figsize=(12, 6))
        source_counts.plot(kind='bar', color='orange')
        plt.title('Paper Distribution by Source (Top 10)', fontsize=16)
        plt.xlabel('Source', fontsize=12)
        plt.ylabel('Number of Papers', fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('sources_distribution.png')
        plt.show()
        
        return source_counts
    
    return None

def run_complete_analysis(df):
    """Run all analysis functions"""
    print("🚀 Starting complete analysis...")
    
    results = {}
    
    results['yearly_counts'] = analyze_publication_trends(df)
    results['top_journals'] = analyze_journals(df)
    results['word_frequencies'] = analyze_titles(df)
    results['source_distribution'] = analyze_sources(df)
    
    # Additional analysis: Abstract length distribution
    plt.figure(figsize=(10, 6))
    df['abstract_word_count'].hist(bins=50, color='purple', alpha=0.7)
    plt.title('Distribution of Abstract Word Count', fontsize=16)
    plt.xlabel('Word Count', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.tight_layout()
    plt.savefig('abstract_length_distribution.png')
    plt.show()
    
    print("✅ Analysis complete!")
    return results