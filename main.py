# main.py
import pandas as pd
from data_exploration import load_and_explore_data
from data_cleaning import clean_and_prepare_data, analyze_missing_data
from analysis_visualization import run_complete_analysis

def main():
    """Main function to run the complete analysis"""
    print("=" * 60)
    print("CORD-19 COVID-19 RESEARCH DATA ANALYSIS")
    print("=" * 60)
    
    # Step 1: Load and explore data
    df = load_and_explore_data()
    if df is None:
        return
    
    # Step 2: Analyze missing data
    missing_data = analyze_missing_data(df)
    
    # Step 3: Clean and prepare data
    df_clean = clean_and_prepare_data(df)
    
    # Step 4: Run complete analysis
    results = run_complete_analysis(df_clean)
    
    # Step 5: Save cleaned data
    df_clean.to_csv('cleaned_metadata.csv', index=False)
    print("💾 Cleaned data saved as 'cleaned_metadata.csv'")
    
    # Summary report
    print("\n" + "=" * 60)
    print("ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"Total papers analyzed: {len(df_clean):,}")
    print(f"Time span: {int(df_clean['year'].min())} - {int(df_clean['year'].max())}")
    print(f"Number of unique journals: {df_clean['journal'].nunique()}")
    print(f"Papers with abstracts: {df_clean['has_abstract'].sum():,}")
    
    print("\n🎉 Analysis complete! Run 'streamlit run app.py' to launch the web application.")

if __name__ == "__main__":
    main()