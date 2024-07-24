import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from textblob import TextBlob

# Load the dataset from a CSV file
file_path = r"E:\TYBSC CS\TCS INTERSHIP\reviews.csv"  # Replace with your actual CSV file path
try:
    dataset = pd.read_csv(file_path)
    print("Dataset loaded successfully.")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

# Strip leading and trailing whitespace from column names
dataset.columns = dataset.columns.str.strip()

# Ensure that the 'Review' and 'Rating' columns exist in the dataset
if 'Review' not in dataset.columns or 'Rating' not in dataset.columns:
    raise ValueError("Dataset must contain 'Review' and 'Rating' columns.")

# Sentiment analysis implementation
def perform_sentiment_analysis(text):
    analysis = TextBlob(str(text))
    polarity = analysis.sentiment.polarity
    if polarity > 0:
        sentiment_label = "Positive"
    elif polarity == 0:
        sentiment_label = "Neutral"
    else:
        sentiment_label = "Negative"
    return sentiment_label

# Apply sentiment analysis
dataset['Sentiment_Label'] = dataset['Review'].apply(perform_sentiment_analysis)

# Filter dataset for positive sentiment reviews
positive_reviews = dataset[dataset['Sentiment_Label'] == 'Positive']

# Concatenate all positive sentiment reviews into a single string
positive_text = ' '.join(positive_reviews['Review'])

# Generate word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=None).generate(positive_text)

# Display the word cloud
plt.figure(figsize=(10, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.title('Word Cloud of Positive Sentiments')
plt.axis('off')
plt.show()
