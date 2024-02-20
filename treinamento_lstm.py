import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from joblib import dump, load
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.models import Sequential

def contaNaNTexto():
    print('Contando valores NaN na coluna TEXTO')
    num_nan = df['TEXTO'].isna().sum()
    print("Número de registros com valor NaN na coluna TEXTO:", num_nan)

print('Carregando os dados do CSV')
df = pd.read_csv('dados.csv', sep='\t')

contaNaNTexto()
df = df.dropna(subset=['TEXTO'])
contaNaNTexto()

print('Inicializando o LabelEncoder')
label_encoder = LabelEncoder()
print('Criando a coluna LABEL com o LabelEncoder')
df['LABEL'] = label_encoder.fit_transform(df['COD_ASSUNTO_CLASSIFICACAO'])

print('Salvando o LabelEncoder em um arquivo')
dump(label_encoder, 'label_encoder.joblib')

print('\nVisualizar as primeiras linhas do dataframe')
print(df.head())

print('\nVerificando informações sobre o dataframe')
print(df.info())

print('\nExplorando os rótulos de classificação')
print(df['COD_ASSUNTO_CLASSIFICACAO'].value_counts())
print(df['LABEL'].value_counts())

# sys.exit()

print('Separando os dados em features (X) e labels (y)')
X = df['TEXTO']
y = df['LABEL']

print('Dividindo os dados em conjunto de treinamento e conjunto de teste')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# print('Definindo o vocab_size')
# print('Extraindo os textos da coluna TEXTO')
# textos = df['TEXTO'].tolist()
# print('Criando uma instância do CountVectorizer')
# vectorizer = CountVectorizer()
# print('Realizar o fit_transform nos textos')
# X_textos = vectorizer.fit_transform(textos)
# print('Obtendo o vocabulário do CountVectorizer')
# vocabulario = vectorizer.vocabulary_
# print('Obtendo as palavras únicas do vocabulário')
# palavras_unicas = list(vocabulario.keys())
# print('Calculando o vocab_size')
# vocab_size = len(palavras_unicas)

# codigo acima executado uma vez, colocando agora o valor fixo resultante:
# vocab_size = 2056447
# print('=> vocab_size:', vocab_size)

# print('Calculando o comprimento de cada sequência de texto')
# comprimentos = [len(texto.split()) for texto in textos]
# print('Calculando o comprimento máximo das sequências')
# max_length = max(comprimentos)

# codigo acima executado uma vez, colocando agora o valor fixo resultante:
# max_length = 1575215
# print("=> max_length:", max_length)

# print('Contando número de classes')
# num_classes = df['LABEL'].nunique()
# print("=> num_classes:", num_classes)

# print('Definindo a arquitetura da rede neural')
# embedding_dim = 128
# model = Sequential([
#     Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),
#     LSTM(units=100),
#     Dense(units=num_classes, activation='softmax')
# ])

print('Tokenização e padding dos textos')
max_words = 10000
max_length = 1000
tokenizer = Tokenizer(num_words=max_words)
print('Aplicando fit_on_texts')
tokenizer.fit_on_texts(X_train)
print('Aplicando texts_to_sequences X_train')
X_train_sequences = tokenizer.texts_to_sequences(X_train)
print('Aplicando texts_to_sequences X_test')
X_test_sequences = tokenizer.texts_to_sequences(X_test)
print('Aplicando pad_sequences X_train_sequences')
X_train_padded = pad_sequences(X_train_sequences, maxlen=max_length, padding='post')
print('Aplicando pad_sequences X_test_sequences')
X_test_padded = pad_sequences(X_test_sequences, maxlen=max_length, padding='post')

print('Definindo modelo LSTM')
embedding_dim = 128
model = Sequential([
    Embedding(input_dim=max_words, output_dim=embedding_dim, input_length=max_length),
    LSTM(units=100),
    Dense(units=len(label_encoder.classes_), activation='softmax')
])

print('Compilando o modelo')
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# print('Criando um pipeline que transforma os textos em vetores e treina o modelo Naive Bayes')
# model = make_pipeline(CountVectorizer(), model)

print('Treinando o modelo')
# model.fit(X_train, y_train)
# Treinar modelo
model.fit(X_train_padded, y_train, epochs=10, batch_size=64, validation_data=(X_test_padded, y_test))

# print('Fazendo previsões no conjunto de teste')
# y_pred = model.predict(X_test)

print('Calculando a precisão do modelo')
# accuracy = accuracy_score(y_test, y_pred)
loss, accuracy = model.evaluate(X_test_padded, y_test)
print(f'Acurácia do modelo: {accuracy}')

print('Salvando o modelo treinado')
# dump(model, 'modelo_classificacao.pkl')
model.save('modelo_lstm.keras')

# resultado 0.6