from flask import Flask, render_template, request
import fasttext.util
import numpy as np
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import re
from nltk.corpus import stopwords
import pickle
from sklearn.model_selection import train_test_split

app = Flask(__name__)
loaded_model = load_model('model_LSTM.h5')
df = pd.read_csv('dataset_headline_news.csv')
X = df['Title']
y = df['label']
messages = df.copy()
X = messages['Title']
y = messages['label']
stopwords_indonesian = stopwords.words('indonesian')

def create_corpus(message):
    # Assuming 'message' is a string
    review = re.sub('[^a-zA-Z]', ' ', message)
    review = review.lower()
    review = ' '.join([word for word in review.split() if word not in stopwords_indonesian])
    return review

def load_tokenizer_from_file(tokenizer_path):
    with open(tokenizer_path, 'rb') as handle:
        tokenizer = pickle.load(handle)
    return tokenizer

def fake_news_det(news, tokenizer):
    input_data = [news]
    corpus_text = create_corpus(input_data[0])  # Process a single message
    sequences = tokenizer.texts_to_sequences([corpus_text])
    padded_sequences = pad_sequences(sequences, padding='pre', maxlen=229)
    X_trial = np.array(padded_sequences)
    y_probs = loaded_model.predict(X_trial)
    y_pred = (y_probs > 0.5).astype(int)
    return y_pred

# Load FastText model
fasttext.util.download_model('id', if_exists='ignore')
ft_model = fasttext.load_model('cc.id.300.bin')
tokenizer = load_tokenizer_from_file('tokenizer.pickle')
sequences = tokenizer.texts_to_sequences(X)
padded_sequences = pad_sequences(sequences, padding='pre', maxlen=229)
X_final = np.array(padded_sequences)
y_final = np.array(y)
X_temp, X_test, y_temp, y_test = train_test_split(X_final, y_final, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        pred = fake_news_det(message, tokenizer)
        print(pred)
        return render_template('index.html', y_pred=pred)
    else:
        return render_template('index.html', y_pred="Something went wrong")

if __name__ == '__main__':
    app.run(debug=True, port=1143)
