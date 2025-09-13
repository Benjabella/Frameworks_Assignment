# CORD-19 COVID-19 Research Data Analysis

## Overview
This project analyzes the CORD-19 dataset metadata to uncover patterns and trends in COVID-19 research publications. The analysis includes publication trends, journal distribution, and title word frequency analysis.

## Features
- Data loading and exploration
- Data cleaning and preparation
- Publication trend analysis
- Journal distribution analysis
- Title word frequency analysis
- Interactive Streamlit web application

## Installation
1. Clone this repository
2. Install required packages:
   ```bash
   pip install -r requirements.txt
3. Download `metadata.csv` from the [CORD-19 dataset](https://www.kaggle.com/datasets/batprem/cord-19-metadata) and place it in the project directory.

## Usage
1. Run the complete analysis:
    ```bash
    python main.py
2. Launch the Streamlit app:
    ```bash
    streamlit run app.py

## File Structure

├── main.py                 # Main execution script
├── data_exploration.py     # Data loading and exploration
├── data_cleaning.py        # Data cleaning functions
├── analysis_visualization.py # Analysis and visualization functions
├── app.py                  # Streamlit web application
├── requirements.txt        # Python dependencies
└── README.md              # Project documentation

