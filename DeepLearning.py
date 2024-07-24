import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
import string
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D
from tensorflow.keras.callbacks import EarlyStopping

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# Dataset loading
file_path = "e:/TYBSC CS/TCS INTERSHIP/reviews.csv"
dataset = pd.read_csv(file_path)

# Text preprocessing
def preprocess_text(text):
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = []
    for word, tag in pos_tags:
        if tag.startswith('NN'):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'
        lemmatized_tokens.append(lemmatizer.lemmatize(word, pos))
    table = str.maketrans('', '', string.punctuation)
    no_punctuation_tokens = [token.translate(table) for token in lemmatized_tokens]
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in no_punctuation_tokens if word.lower() not in stop_words]
    return ' '.join(filtered_tokens)

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
    return sentiment_label, polarity

# Integrate text preprocessing and sentiment analysis results into the dataset
dataset['Preprocessed_Review'] = dataset['Review'].apply(preprocess_text)
dataset['Sentiment_Label'], dataset['Sentiment_Polarity'] = zip(*dataset['Preprocessed_Review'].apply(perform_sentiment_analysis))

# Convert sentiment labels to numerical values
label_encoder = LabelEncoder()
dataset['Sentiment_Label'] = label_encoder.fit_transform(dataset['Sentiment_Label'])

# Prepare data for deep learning model
tokenizer = Tokenizer()
tokenizer.fit_on_texts(dataset['Preprocessed_Review'])
X = tokenizer.texts_to_sequences(dataset['Preprocessed_Review'])
X = pad_sequences(X)
y = dataset['Sentiment_Label']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the deep learning model architecture
embedding_dim = 100
max_sequence_length = X.shape[1]

model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=embedding_dim, input_length=max_sequence_length))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(units=128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(units=3, activation='softmax'))  # 3 classes for positive, neutral, negative

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Define early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.1, callbacks=[early_stopping])

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')
