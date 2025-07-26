from flask import Flask, render_template, request
import joblib
# text_preprocessor.py
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s!?]', '', text)  # Keep !? for sentiment
    words = word_tokenize(text)
    stop_words = list(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

app = Flask(__name__)

# Load models
model = joblib.load('models/model.pkl')
tfidf = joblib.load('models/tfidf.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    # Get user input
    review = request.form['review']
    
    # Process and predict
    clean_text = preprocess_text(review)
    X = tfidf.transform([clean_text])
    rating = round(model.predict(X)[0], 1)
    if rating > 5.0:
        rating = 5.0
    
    # Determine sentiment
    sentiment = "Positive" if rating >= 3.5 else "Neutral" if rating >= 2 else "Negative"
    
    return render_template('results.html', 
                         review=review,
                         rating=rating,
                         sentiment=sentiment)

if __name__ == '__main__':
    app.run(debug=True)