import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
    return polarity

# Apply sentiment analysis
dataset['Sentiment_Polarity'] = dataset['Review'].apply(perform_sentiment_analysis)

# Ensure the 'Rating' column is numeric
dataset['Rating'] = pd.to_numeric(dataset['Rating'], errors='coerce')

# Drop rows with missing values in 'Sentiment_Polarity' or 'Rating'
dataset.dropna(subset=['Sentiment_Polarity', 'Rating'], inplace=True)

# Scatter Plot of Sentiment Polarity vs. Rating
sns.set_style("whitegrid")
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Sentiment_Polarity', y='Rating', data=dataset, color='green')
plt.title('Sentiment Polarity vs. Rating', fontsize=16)
plt.xlabel('Sentiment Polarity', fontsize=14)
plt.ylabel('Rating', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()
