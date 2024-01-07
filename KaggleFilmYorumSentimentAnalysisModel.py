import re
import string
import numpy as np
import pandas as pd
import html
import os
import openpyxl

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Embedding, GRU, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences

class SentimentAnalysisModel:
    def __init__(self, data_path, num_words=10000, max_tokens=None, embedding_size=50):
        self.data_path = data_path
        self.num_words = num_words
        self.max_tokens = max_tokens
        self.embedding_size = embedding_size
        self.model = None
        self.tokenizer = Tokenizer(num_words=num_words)

    def load_data(self):
        df = pd.read_csv(self.data_path)
        # Data preprocessing steps
        df["comment"] = df["comment"].apply(lambda x: x[23:-24])
        df["point"] = df["point"].apply(lambda x: float(x[0:-2]))
        df.drop(df[df["point"] == 3].index, inplace=True)
        df["point"] = df["point"].replace(1, 0)
        df["point"] = df["point"].replace(2, 0)
        df["point"] = df["point"].replace(4, 1)
        df["point"] = df["point"].replace(5, 1)
        df.reset_index(inplace=True)
        df.drop("index", axis=1, inplace=True)
        df["comment"] = df["comment"].apply(lambda x: x.lower())
        df["comment"] = df["comment"].apply(lambda x: x.replace("\r", " "))
        df["comment"] = df["comment"].apply(lambda x: x.replace("\n", " "))
        return df

    def preprocess_data(self, df):
        df["comment"] = df["comment"].apply(self.preprocess_text)
        df.drop_duplicates(subset=["comment"], inplace=True)
        df.dropna(subset=["comment"], inplace=True)
        df["comment"] = df["comment"].apply(lambda x: x.lower())
        df["comment"] = df["comment"].apply(self.remove_punctuation)
        df["comment"] = df["comment"].apply(self.remove_numeric)
        X_train, X_test, y_train, y_test = self.split_data(df)
        return X_train, X_test, y_train, y_test

    def preprocess_text(self, text):
        text = html.unescape(text)
        special_chars = {'&': 'and', '@': 'at', '#': 'hash', '$': 'dollar', '%': 'percent', '*': 'star'}
        for char, replacement in special_chars.items():
            text = text.replace(char, replacement)
        return text

    def remove_punctuation(self, text):
        no_punc = [words for words in text if words not in string.punctuation]
        word_wo_punc = "".join(no_punc)
        return word_wo_punc

    def remove_numeric(self, corpus):
        return "".join(words for words in corpus if not words.isdigit())

    def split_data(self, df):
        target = df["point"].values.tolist()
        data = df["comment"].values.tolist()
        cutoff = int(len(data) * 0.80)
        X_train, X_test = data[:cutoff], data[cutoff:]
        y_train, y_test = target[:cutoff], target[cutoff:]
        return X_train, X_test, y_train, y_test

    def build_model(self):
        model = Sequential()
        model.add(Embedding(input_dim=self.num_words, output_dim=self.embedding_size, input_length=self.max_tokens, name="embedding_layer"))
        model.add(GRU(units=16, return_sequences=True))
        model.add(GRU(units=8, return_sequences=True))
        model.add(GRU(units=4))
        model.add(Dense(1, activation="sigmoid"))
        optimizer = Adam(learning_rate=1e-3)  # 'lr' yerine 'learning_rate' kullanımı
        model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
        self.model = model
        return model

    def train_model(self, X_train, y_train, epochs=5, batch_size=256):
        X_train_tokens = self.tokenizer.texts_to_sequences(X_train)
        X_train_pad = pad_sequences(X_train_tokens, maxlen=self.max_tokens)
        y_train = np.array(y_train)
        self.model.fit(X_train_pad, y_train, epochs=epochs, batch_size=batch_size) #verbose=1 veya verbose=2 daha fazla bilgi

    def predict(self, X_test):
        X_test_tokens = self.tokenizer.texts_to_sequences(X_test)
        X_test_pad = pad_sequences(X_test_tokens, maxlen=self.max_tokens)
        y_pred = self.model.predict(X_test_pad)
        y_pred = y_pred.T[0]
        cls_pred = np.array([1.0 if p > 0.5 else 0.0 for p in y_pred])
        return cls_pred

    def evaluate(self, cls_pred, cls_true):
        incorrect = np.where(cls_pred != cls_true)[0]
        return incorrect


    def save_predictions_to_excel(self, incorrect_indices):
        # Hatalı tahminlerin indekslerini içeren DataFrame oluştur
        incorrect_df = pd.DataFrame({"Incorrect_Index": incorrect_indices})

        # Hatalı tahminlerin indekslerini içeren Excel dosyasını kaydet
        output_excel_path = os.path.join(current_directory, "kaggle/export/incorrect_predictions.xlsx")
        incorrect_df.to_excel(output_excel_path, index=False, engine="openpyxl")
        print(f"Incorrect predictions saved to: {output_excel_path}")


# Mevcut çalışma dizinini al
current_directory = os.getcwd()
data_path = os.path.join(current_directory, "kaggle/input/turkish_movie_sentiment_dataset.csv")
#data_path = "kaggle/input/turkish_movie_sentiment_dataset.csv"
sentiment_model = SentimentAnalysisModel(data_path, max_tokens=100)
df = sentiment_model.load_data()
X_train, X_test, y_train, y_test = sentiment_model.preprocess_data(df)

sentiment_model.build_model()
#-----------------------------------------------------------------
#verideki point değeri 4.0 olmayan tüm yorumları içeren dataframe oluşturur konsol çıktısı verir.
df_filtered = df[df["point"] != 4.0]
print(df_filtered)
#----------------------------------------------------------------------------

print(sentiment_model.model.summary())  # Model: "sequential" Model özetini yazdırın
sentiment_model.train_model(X_train, y_train, epochs=5, batch_size=256)
#Model: "sequential" özet çıktısını alıyoruz.
output_path = os.path.join(current_directory, 'kaggle/export/model_summary.txt')
with open(output_path, 'w') as f:
    sentiment_model.model.summary(print_fn=lambda x: f.write(x + '\n'))
print(f"Model summary saved to: {output_path}")
#-----------------------------------------------------------------

cls_pred = sentiment_model.predict(X_test)

incorrect_indices = sentiment_model.evaluate(cls_pred, np.array(y_test))
sentiment_model.save_predictions_to_excel(incorrect_indices)
