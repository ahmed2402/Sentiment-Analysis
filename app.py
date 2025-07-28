from flask import Flask, render_template, request, flash, redirect, url_for
import joblib
# text_preprocessor.py
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s!?]', '', text)  # Keep !? for sentiment
    words = word_tokenize(text)
    stop_words = list(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

def validate_input(text):
    """
    Validate user input to check for irrelevant or gibberish text
    Returns: (is_valid, error_message)
    """
    # Check if input is empty or too short
    if not text or len(text.strip()) < 10:
        return False, "Please provide a more detailed review (at least 10 characters)."
    
    # Check if input is too long
    if len(text) > 1000:
        return False, "Review is too long. Please keep it under 1000 characters."
    
    # Check for repetitive characters (like "aaaaaaa" or "!!!!!")
    if re.search(r'(.)\1{4,}', text):
        return False, "Please provide meaningful text instead of repetitive characters."
    
    # Check for random character sequences
    random_patterns = [
        r'[a-z]{15,}',  # Very long sequences of random letters (increased threshold)
        r'[0-9]{8,}',   # Long sequences of numbers
        r'[!@#$%^&*()]{5,}',  # Long sequences of special characters
    ]
    
    for pattern in random_patterns:
        if re.search(pattern, text.lower()):
            return False, "Please provide meaningful text instead of random characters."
    
    # Check if text contains meaningful words (not just stopwords)
    words = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    meaningful_words = [word for word in words if word not in stop_words and len(word) > 2]
    
    if len(meaningful_words) < 2:
        return False, "Please provide more meaningful content in your review."
    
    # Check for music/album related keywords (optional but helpful)
    music_keywords = [
        'album', 'song', 'music', 'track', 'melody', 'rhythm', 'beat', 'sound',
        'voice', 'singing', 'instrument', 'guitar', 'piano', 'drums', 'bass',
        'lyrics', 'chorus', 'verse', 'bridge', 'hook', 'vibe', 'mood', 'feeling',
        'emotion', 'atmosphere', 'energy', 'tone', 'style', 'genre', 'artist',
        'band', 'singer', 'producer', 'recording', 'production', 'mix', 'master',
        'good', 'bad', 'great', 'amazing', 'terrible', 'love', 'hate', 'like',
        'dislike', 'enjoy', 'boring', 'exciting', 'beautiful', 'ugly', 'powerful',
        'weak', 'strong', 'soft', 'loud', 'quiet', 'fast', 'slow', 'upbeat',
        'melancholic', 'happy', 'sad', 'angry', 'peaceful', 'energetic', 'relaxing'
    ]
    
    text_lower = text.lower()
    music_word_count = sum(1 for keyword in music_keywords if keyword in text_lower)
    
    if music_word_count == 0:
        return False, "Please provide a review about music or an album. Your text doesn't seem to be related to music."
    
    return True, ""

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Required for flash messages

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
    
    # Validate input
    is_valid, error_message = validate_input(review)
    
    if not is_valid:
        # Store error message and redirect back to home
        flash(error_message, 'error')
        return redirect(url_for('home'))
    
    try:
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
    
    except Exception as e:
        # Handle any errors during processing
        flash("An error occurred while analyzing your review. Please try again with different text.", 'error')
        return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)