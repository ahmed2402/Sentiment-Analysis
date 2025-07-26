import pandas as pd
import numpy as np
import random
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
except:
    pass

def preprocess_text(text):
    """Preprocess text similar to the original dataset"""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s!?]', '', text)  # Keep !? for sentiment
    words = word_tokenize(text)
    stop_words = list(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

def generate_synthetic_reviews():
    """Generate synthetic music album reviews for low ratings"""
    
    # Genre-specific terminology and phrases
    genres = ['rock', 'pop', 'hip-hop', 'electronic', 'jazz', 'country', 'folk', 'metal', 'punk', 'indie', 'blues', 'reggae', 'classical', 'r&b', 'soul']
    instruments = ['guitar', 'drums', 'bass', 'synth', 'piano', 'violin', 'saxophone', 'trumpet', 'harmonica', 'keyboard', 'organ', 'flute', 'clarinet', 'trombone', 'cello']
    production_terms = ['mixing', 'mastering', 'arrangement', 'composition', 'melody', 'rhythm', 'harmony', 'dynamics', 'production', 'engineering', 'sound quality', 'audio', 'recording']
    
    # Common negative phrases for different rating levels
    rating_phrases = {
        0.5: [
            "absolute garbage", "complete waste of time", "unlistenable", "painful to hear", "worst album ever",
            "terrible production", "amateurish", "embarrassing", "cringe-worthy", "avoid at all costs",
            "trainwreck", "disaster", "horrible", "awful", "dreadful", "atrocious", "abysmal", "excruciating",
            "torture to listen to", "ear-splitting", "mind-numbing", "soul-crushing", "depressingly bad",
            "complete failure", "utter rubbish", "pure trash", "garbage fire", "dumpster fire", "hot mess",
            "absolute disaster", "complete joke", "laughably bad", "pathetically awful", "shockingly terrible",
            "unbearable", "intolerable", "insufferable", "repulsive", "revolting", "nauseating", "vomit-inducing"
        ],
        1.0: [
            "really bad", "poor quality", "disappointing", "mediocre at best", "not worth it",
            "weak", "boring", "forgettable", "lackluster", "subpar", "below average",
            "uninspired", "generic", "derivative", "soulless", "flat", "dull", "tedious",
            "monotonous", "repetitive", "predictable", "unoriginal", "stale", "dated", "outdated",
            "irrelevant", "pointless", "meaningless", "empty", "hollow", "shallow", "superficial",
            "pretentious", "overrated", "overhyped", "overproduced", "underwhelming", "underdeveloped",
            "half-baked", "unfinished", "incomplete", "rushed", "sloppy", "careless", "thoughtless"
        ],
        1.5: [
            "not great", "below average", "somewhat disappointing", "has some issues", "could be better",
            "lacks direction", "inconsistent", "rough around the edges", "needs work", "underwhelming",
            "misses the mark", "doesn't quite work", "has potential but", "struggles with", "fails to",
            "flawed", "problematic", "troubled", "confused", "muddled", "unclear", "unfocused",
            "scattered", "disjointed", "fragmented", "incoherent", "messy", "chaotic", "disorganized",
            "unstructured", "formless", "shapeless", "directionless", "aimless", "purposeless",
            "unconvincing", "unpersuasive", "uncompelling", "unengaging", "uninteresting", "unexciting",
            "uninspiring", "unmotivating", "unmemorable", "unremarkable", "unexceptional", "ordinary"
        ],
        2.0: [
            "okay", "average", "mediocre", "nothing special", "run of the mill", "standard fare",
            "decent but", "acceptable", "passable", "competent", "adequate", "serviceable",
            "middle of the road", "neither good nor bad", "forgettable", "unremarkable", "standard",
            "conventional", "traditional", "typical", "usual", "normal", "regular", "routine",
            "everyday", "common", "ordinary", "basic", "simple", "plain", "straightforward",
            "straight-ahead", "by-the-book", "by-the-numbers", "formulaic", "cookie-cutter",
            "paint-by-numbers", "assembly-line", "mass-produced", "commercial", "mainstream",
            "popular", "accessible", "approachable", "easy-listening", "background music"
        ],
        2.5: [
            "slightly above average", "has some good moments", "shows promise", "decent effort",
            "not bad", "reasonably good", "has potential", "some redeeming qualities", "worth a listen",
            "better than expected", "has its moments", "shows improvement", "getting there",
            "promising", "encouraging", "hopeful", "optimistic", "positive", "constructive",
            "developmental", "progressive", "advancing", "evolving", "growing", "maturing",
            "developing", "emerging", "rising", "ascending", "climbing", "improving", "enhancing",
            "refining", "polishing", "perfecting", "mastering", "excelling", "thriving", "flourishing",
            "blossoming", "blooming", "sprouting", "budding", "germinating", "taking root", "establishing"
        ]
    }
    
    # Album names and artists for variety
    artists = [
        "The Generic Band", "Average Collective", "Mediocre Masters", "Subpar Sounds", "Forgettable Five",
        "The Underwhelming", "Bland Brothers", "The Uninspired", "Lackluster Legends", "The Disappointing",
        "Generic Group", "The Forgettable", "Boring Band", "The Unremarkable", "Standard Sounds",
        "The Ordinary", "Basic Beats", "Common Collective", "Regular Rhythm", "Typical Tunes",
        "The Average", "Mediocre Music", "Subpar Sessions", "Forgettable Favorites", "Bland Beats",
        "The Uninspired", "Lackluster Legends", "The Disappointing", "Generic Grooves", "The Forgettable",
        "Boring Ballads", "The Unremarkable", "Standard Sessions", "Common Collective", "Regular Records",
        "The Ordinary", "Basic Band", "Typical Tunes", "Average Artists", "Mediocre Musicians"
    ]
    
    albums = [
        "Generic Album", "Average Record", "Mediocre Music", "Subpar Songs", "Forgettable Tracks",
        "The Underwhelming Collection", "Bland Beats", "Uninspired Melodies", "Lackluster Lyrics", "Disappointing Sounds",
        "Generic Grooves", "Forgettable Favorites", "Boring Ballads", "Unremarkable Rhythms", "Standard Songs",
        "Basic Beats", "Common Collection", "Regular Recordings", "Typical Tunes", "Ordinary Output",
        "Average Album", "Mediocre Mix", "Subpar Sessions", "Forgettable Favorites", "Bland Ballads",
        "Uninspired Undertones", "Lackluster Lullabies", "Disappointing Disc", "Generic Grooves", "Forgettable Favorites",
        "Boring Beats", "Unremarkable Undertones", "Standard Sessions", "Common Collection", "Regular Recordings",
        "The Ordinary", "Basic Beats", "Typical Tunes", "Average Album", "Mediocre Music", "Subpar Songs"
    ]
    
    synthetic_reviews = []
    
    for rating in [0.5, 1.0, 1.5, 2.0, 2.5]:
        for i in range(4000):
            # Select random elements
            artist = random.choice(artists)
            album = random.choice(albums)
            genre = random.choice(genres)
            instrument = random.choice(instruments)
            production_term = random.choice(production_terms)
            negative_phrase = random.choice(rating_phrases[rating])
            
            # Generate review based on rating
            if rating == 0.5:
                templates = [
                    f"this {genre} album by {artist} is {negative_phrase}. the {instrument} work is {negative_phrase} and the {production_term} is {negative_phrase}.",
                    f"avoid {album} at all costs. {artist} created {negative_phrase} with terrible {instrument} and awful {production_term}.",
                    f"this is {negative_phrase}. {artist}'s {album} has {negative_phrase} {instrument} and {negative_phrase} {production_term}.",
                    f"{album} by {artist} is {negative_phrase}. the {genre} elements are {negative_phrase} and the {production_term} is {negative_phrase}.",
                    f"i regret listening to this {genre} album. {artist} delivered {negative_phrase} with {negative_phrase} {instrument} work.",
                    f"what a {negative_phrase} {genre} album. {artist} completely failed with {negative_phrase} {instrument} and {negative_phrase} {production_term}.",
                    f"this {genre} effort from {artist} is {negative_phrase}. the {instrument} sounds {negative_phrase} and the {production_term} is {negative_phrase}.",
                    f"{album} represents everything wrong with {genre}. {artist} produced {negative_phrase} with {negative_phrase} {instrument} work.",
                    f"i can't believe {artist} released this {negative_phrase} {genre} album. the {instrument} is {negative_phrase} and {production_term} is {negative_phrase}.",
                    f"this {genre} album is {negative_phrase}. {artist} should be ashamed of this {negative_phrase} {instrument} and {negative_phrase} {production_term}."
                ]
            elif rating == 1.0:
                templates = [
                    f"this {genre} album is {negative_phrase}. {artist} struggles with {instrument} and the {production_term} is {negative_phrase}.",
                    f"{album} by {artist} is {negative_phrase}. the {genre} style feels {negative_phrase} and the {production_term} lacks quality.",
                    f"not impressed with this {genre} effort. {artist} shows {negative_phrase} {instrument} skills and {negative_phrase} {production_term}.",
                    f"this {genre} album from {artist} is {negative_phrase}. the {instrument} work is {negative_phrase} and {production_term} needs improvement.",
                    f"{album} is {negative_phrase}. {artist}'s {genre} approach feels {negative_phrase} with {negative_phrase} {instrument}.",
                    f"disappointed with this {genre} release. {artist} delivered {negative_phrase} {instrument} work and {negative_phrase} {production_term}.",
                    f"this {genre} album from {artist} is {negative_phrase}. the {instrument} performance is {negative_phrase} and {production_term} is {negative_phrase}.",
                    f"{album} shows {artist}'s {negative_phrase} side. the {genre} elements are {negative_phrase} and {instrument} work is {negative_phrase}.",
                    f"not what i expected from {artist}. this {genre} album is {negative_phrase} with {negative_phrase} {instrument} and {negative_phrase} {production_term}.",
                    f"this {genre} effort is {negative_phrase}. {artist} failed to deliver with {negative_phrase} {instrument} and {negative_phrase} {production_term}."
                ]
            elif rating == 1.5:
                templates = [
                    f"this {genre} album has some issues. {artist} shows {negative_phrase} {instrument} work but the {production_term} is {negative_phrase}.",
                    f"{album} by {artist} is {negative_phrase}. the {genre} elements are {negative_phrase} though the {instrument} has moments.",
                    f"not great but not terrible. {artist}'s {genre} album has {negative_phrase} {production_term} but decent {instrument}.",
                    f"this {genre} effort from {artist} is {negative_phrase}. the {instrument} work is {negative_phrase} and {production_term} needs work.",
                    f"{album} shows potential but {negative_phrase}. {artist}'s {genre} style is {negative_phrase} with {negative_phrase} {instrument}.",
                    f"mixed feelings about this {genre} album. {artist} has {negative_phrase} {instrument} but the {production_term} is {negative_phrase}.",
                    f"this {genre} release from {artist} is {negative_phrase}. the {instrument} work shows promise but {production_term} is {negative_phrase}.",
                    f"{album} is {negative_phrase}. {artist}'s {genre} approach has {negative_phrase} moments but {instrument} work is {negative_phrase}.",
                    f"somewhat disappointed with this {genre} album. {artist} delivers {negative_phrase} {instrument} though {production_term} is {negative_phrase}.",
                    f"this {genre} effort is {negative_phrase}. {artist} shows {negative_phrase} {instrument} skills but {production_term} needs improvement."
                ]
            elif rating == 2.0:
                templates = [
                    f"this {genre} album is {negative_phrase}. {artist} delivers {negative_phrase} {instrument} work and {negative_phrase} {production_term}.",
                    f"{album} by {artist} is {negative_phrase}. the {genre} elements are {negative_phrase} and the {instrument} is {negative_phrase}.",
                    f"average {genre} music from {artist}. {negative_phrase} {instrument} and {negative_phrase} {production_term} make this {negative_phrase}.",
                    f"this {genre} album is {negative_phrase}. {artist} shows {negative_phrase} {instrument} skills and {negative_phrase} {production_term}.",
                    f"{album} is {negative_phrase}. {artist}'s {genre} approach results in {negative_phrase} {instrument} and {negative_phrase} {production_term}.",
                    f"standard {genre} fare from {artist}. {negative_phrase} {instrument} work and {negative_phrase} {production_term} make this {negative_phrase}.",
                    f"this {genre} album is {negative_phrase}. {artist} provides {negative_phrase} {instrument} and {negative_phrase} {production_term}.",
                    f"{album} represents {negative_phrase} {genre} music. {artist} offers {negative_phrase} {instrument} and {negative_phrase} {production_term}.",
                    f"typical {genre} album from {artist}. {negative_phrase} {instrument} work and {negative_phrase} {production_term} make this {negative_phrase}.",
                    f"this {genre} effort is {negative_phrase}. {artist} delivers {negative_phrase} {instrument} and {negative_phrase} {production_term}."
                ]
            else:  # 2.5
                templates = [
                    f"this {genre} album is {negative_phrase}. {artist} shows some promise with {instrument} though the {production_term} is {negative_phrase}.",
                    f"{album} by {artist} has {negative_phrase} moments. the {genre} elements are {negative_phrase} and {instrument} work is decent.",
                    f"not bad for a {genre} album. {artist} delivers {negative_phrase} {instrument} and {negative_phrase} {production_term}.",
                    f"this {genre} effort from {artist} is {negative_phrase}. the {instrument} has {negative_phrase} moments but {production_term} is {negative_phrase}.",
                    f"{album} shows {negative_phrase} potential. {artist}'s {genre} style works with {negative_phrase} {instrument} though {production_term} is {negative_phrase}.",
                    f"decent {genre} album from {artist}. {negative_phrase} {instrument} work and {negative_phrase} {production_term} show improvement.",
                    f"this {genre} release is {negative_phrase}. {artist} demonstrates {negative_phrase} {instrument} skills though {production_term} is {negative_phrase}.",
                    f"{album} has {negative_phrase} qualities. {artist}'s {genre} approach shows {negative_phrase} {instrument} work and {negative_phrase} {production_term}.",
                    f"reasonable {genre} effort from {artist}. {negative_phrase} {instrument} and {negative_phrase} {production_term} make this {negative_phrase}.",
                    f"this {genre} album is {negative_phrase}. {artist} provides {negative_phrase} {instrument} work and {negative_phrase} {production_term}."
                ]
            
            # Add natural language imperfections
            review = random.choice(templates)
            
            # Add some natural variations and imperfections
            if random.random() < 0.3:
                review = review.replace("the", "da")
            if random.random() < 0.2:
                review = review.replace("and", "&")
            if random.random() < 0.15:
                review = review.replace("you", "u")
            if random.random() < 0.1:
                review = review.replace("are", "r")
            
            # Add some typos occasionally
            if random.random() < 0.05:
                review = review.replace("album", "albun")
            if random.random() < 0.05:
                review = review.replace("music", "musik")
            if random.random() < 0.05:
                review = review.replace("guitar", "guitar")
            
            # Add emotional expressions based on rating
            if rating <= 1.0:
                if random.random() < 0.3:
                    review += " waste of money."
                if random.random() < 0.2:
                    review += " never listening again."
            elif rating <= 1.5:
                if random.random() < 0.2:
                    review += " disappointing."
                if random.random() < 0.15:
                    review += " expected better."
            elif rating <= 2.0:
                if random.random() < 0.15:
                    review += " nothing special."
                if random.random() < 0.1:
                    review += " forgettable."
            else:  # 2.5
                if random.random() < 0.1:
                    review += " has potential."
                if random.random() < 0.05:
                    review += " worth a listen."
            
            # Clean up any double spaces
            review = re.sub(r'\s+', ' ', review).strip()
            
            # Generate cleaned version
            cleaned_review = preprocess_text(review)
            
            synthetic_reviews.append({
                'Review': review,
                'Rating': rating,
                'Cleaned_Review': cleaned_review,
                'is_synthetic': True
            })
    
    return pd.DataFrame(synthetic_reviews)

if __name__ == "__main__":
    print("Generating synthetic music album reviews...")
    synthetic_df = generate_synthetic_reviews()
    
    print(f"Generated {len(synthetic_df)} synthetic reviews")
    print(f"Rating distribution:\n{synthetic_df['Rating'].value_counts().sort_index()}")
    
    # Save to CSV
    output_file = "dataset/synthetic_low_ratings_4000.csv"
    synthetic_df.to_csv(output_file, index=False)
    print(f"\nSynthetic data saved to: {output_file}")
    
    # Display sample reviews
    print("\nSample reviews by rating:")
    for rating in [0.5, 1.0, 1.5, 2.0, 2.5]:
        sample = synthetic_df[synthetic_df['Rating'] == rating].iloc[0]
        print(f"\nRating {rating}: {sample['Review']}") 