import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression

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
    return polarity

# Apply sentiment analysis
dataset['Sentiment_Polarity'] = dataset['Review'].apply(perform_sentiment_analysis)

# Ensure the 'Rating' column is numeric
dataset['Rating'] = pd.to_numeric(dataset['Rating'], errors='coerce')

# Drop rows with missing values in 'Sentiment_Polarity' or 'Rating'
dataset.dropna(subset=['Sentiment_Polarity', 'Rating'], inplace=True)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(dataset['Review'], dataset['Rating'], test_size=0.2, random_state=42)

# Define a pipeline for text feature extraction and modeling
text_pipeline = Pipeline([
    ('vectorizer', CountVectorizer(ngram_range=(1, 2))),  # Include both unigrams and bigrams
    ('regressor', LinearRegression())
])

# Train the model
text_pipeline.fit(X_train, y_train)

# Predict ratings
y_pred = text_pipeline.predict(X_test)

# Visualizing Regression Analysis Results
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel('Actual Ratings')
plt.ylabel('Predicted Ratings')
plt.title('Actual vs. Predicted Ratings')
plt.show()
