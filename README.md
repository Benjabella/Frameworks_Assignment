# CORD-19 COVID-19 Research Data Explorer

## 📖 Project Description
This project analyzes the COVID-19 Open Research Dataset (CORD-19) to uncover trends in scientific publications during the pandemic. It includes:
- Batch processing for data cleaning and analysis
- Interactive web application for data exploration
- Automated visualization generation

## ✨ Features
- Publication trend analysis by year/month
- Journal distribution statistics
- Title word frequency analysis
- Abstract presence and length metrics
- Interactive filtering and data sampling
- Static visualizations for reports

## ⚙️ Installation
1. Clone this repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🚀 Usage
### Batch Processing (Generate Reports & Visualizations)
```bash
python main.py
```
Generates:
- Publication trend charts
- Journal distribution plots
- Word frequency analysis
- Data quality reports

### Interactive Web Application
```bash
streamlit run app.py
```
Launch a web-based dashboard with:
- Dynamic filters
- Interactive visualizations
- Data sampling
- Export capabilities

## 📂 File Structure
| File | Purpose |
|------|---------|
| `app.py` | Streamlit web application |
| `main.py` | Batch processing and analysis |
| `data_cleaning.py` | Data cleaning utilities |
| `preprocess_data.py` | Data preprocessing pipeline |
| `analysis_visualization.py` | Visualization functions |
| `requirements.txt` | Python dependencies |

## 📊 Generated Visualizations
1. `publications_by_year.png` - Annual publication trends
2. `top_journals.png` - Top publishing journals
3. `top_title_words.png` - Most frequent title words
4. `abstract_length_distribution.png` - Abstract word count distribution
5. `abstract_presence.png` - Papers with/without abstracts
6. `sources_distribution.png` - Data source distribution

## 📦 Dependencies
- pandas
- numpy
- matplotlib
- seaborn
- streamlit
- wordcloud
- altair
- requests
- scikit-learn
- python-dateutil

Full list in `requirements.txt`
