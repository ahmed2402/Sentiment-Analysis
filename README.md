# Music Album Review Sentiment Analysis

A comprehensive sentiment analysis system for music album reviews that predicts ratings and sentiment categories using machine learning. This project includes a web application for real-time sentiment analysis and extensive data processing capabilities.

## 🎵 Features

- **Real-time Sentiment Analysis**: Web interface for analyzing music review sentiment
- **Rating Prediction**: Predicts numerical ratings (0.5-5.0) for music album reviews
- **Sentiment Classification**: Categorizes reviews as Positive, Neutral, or Negative
- **Advanced Text Preprocessing**: Comprehensive NLP pipeline with tokenization, stopword removal, and lemmatization
- **Synthetic Data Generation**: Tools for generating balanced training datasets
- **Interactive Web Interface**: User-friendly Flask web application
- **Comprehensive EDA**: Detailed exploratory data analysis notebooks

## 🚀 Quick Start

### Prerequisites

- Python 3.7+
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Sentiment-Analysis
   ```

2. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download NLTK data** (automatically handled by the application)
   ```bash
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
   ```

### Running the Application

1. **Start the web application**
   ```bash
   python app.py
   ```

2. **Open your browser and navigate to**
   ```
   http://localhost:5000
   ```

3. **Enter a music review and get instant sentiment analysis results**

## 📁 Project Structure

```
Sentiment-Analysis/
├── app.py                          # Main Flask web application
├── generate_synthetic_data.py      # Synthetic data generation script
├── main.ipynb                      # Main data processing notebook
├── eda.ipynb                       # Exploratory data analysis notebook
├── README.md                       # Project documentation
├── dataset/                        # Data files
│   ├── music_album_reviews.csv     # Original dataset
│   ├── cleaned_music_reviews.csv   # Preprocessed dataset
│   ├── cleaned_music_reviews2.csv  # Final processed dataset
│   └── synthetic_low_ratings_4000.csv # Generated synthetic data
├── models/                         # Trained machine learning models
│   ├── model.pkl                   # Main sentiment model
│   ├── model_ridge.pkl            # Ridge regression model
│   └── tfidf.pkl                   # TF-IDF vectorizer
├── static/                         # Static web assets
│   └── style.css                   # CSS styling
└── templates/                      # HTML templates
    ├── index.html                  # Main input page
    └── results.html                # Results display page
```

## 🔧 Technical Details

### Data Processing Pipeline

1. **Text Preprocessing**:
   - Lowercase conversion
   - Special character removal (preserving !? for sentiment)
   - Tokenization using NLTK
   - Stopword removal
   - Lemmatization using WordNet

2. **Feature Engineering**:
   - TF-IDF vectorization
   - Text normalization

3. **Model Architecture**:
   - Ridge Regression for rating prediction
   - Sentiment classification based on predicted ratings:
     - Positive: ≥ 3.5
     - Neutral: 2.0 - 3.4
     - Negative: < 2.0

### Dataset Information

- **Original Dataset**: 80,271 music album reviews
- **Synthetic Data**: 20,000 additional low-rating reviews for balance
- **Final Dataset**: 97,841 total reviews
- **Rating Distribution**: Balanced across 0.5-5.0 scale

### Model Performance

The system uses a Ridge Regression model trained on TF-IDF features with the following characteristics:
- Handles imbalanced rating distributions
- Robust text preprocessing pipeline
- Real-time prediction capabilities

## 📊 Usage Examples

### Web Interface

1. Navigate to the web application
2. Enter a music review text
3. Submit for analysis
4. View predicted rating and sentiment category

### Programmatic Usage

```python
from app import preprocess_text
import joblib

# Load models
model = joblib.load('models/model.pkl')
tfidf = joblib.load('models/tfidf.pkl')

# Analyze text
text = "This album is absolutely amazing! The production quality is outstanding."
clean_text = preprocess_text(text)
X = tfidf.transform([clean_text])
rating = model.predict(X)[0]
sentiment = "Positive" if rating >= 3.5 else "Neutral" if rating >= 2 else "Negative"
```

## 🔬 Data Generation

The project includes a synthetic data generation script (`generate_synthetic_data.py`) that creates balanced training data for low-rating reviews, addressing dataset imbalance issues.

### Running Data Generation

```bash
python generate_synthetic_data.py
```

This generates 20,000 synthetic reviews across different rating levels (0.5-2.5) with realistic music review patterns.

## 📈 Analysis Notebooks

- **`eda.ipynb`**: Comprehensive exploratory data analysis
- **`main.ipynb`**: Data processing and model training pipeline

## 🛠️ Dependencies

- Flask (web framework)
- scikit-learn (machine learning)
- pandas (data manipulation)
- numpy (numerical computing)
- nltk (natural language processing)
- matplotlib & seaborn (visualization)
- joblib (model serialization)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- NLTK library for natural language processing capabilities
- Scikit-learn for machine learning algorithms
- Flask for web framework
- Music review dataset contributors

## 📞 Support

For questions or issues, please open an issue on the project repository or contact the development team.

---

**Note**: This project is designed for educational and research purposes. The sentiment analysis results should be used as guidance rather than definitive assessments.
