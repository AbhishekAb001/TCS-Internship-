import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

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

# Preprocessing and cleaning data (if needed)
# You can add your text preprocessing here if necessary

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(dataset['Review'], dataset['Rating'], test_size=0.2, random_state=42)

# Define a pipeline for text feature extraction and modeling
text_pipeline = Pipeline([
    ('vectorizer', CountVectorizer(ngram_range=(1, 2))),  # Include both unigrams and bigrams
    ('regressor', LinearRegression())
])

# Train the model
text_pipeline.fit(X_train, y_train)

# Evaluate the model
y_pred = text_pipeline.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Plot histogram of Ratings
sns.set_style("whitegrid")
plt.figure(figsize=(10, 6))
sns.histplot(dataset['Rating'], bins=10, kde=True, color='skyblue')
plt.title('Distribution of Ratings', fontsize=16)
plt.xlabel('Rating', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()
