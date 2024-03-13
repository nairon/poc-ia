import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense

print('Carregando os dados do CSV')
df = pd.read_csv('dados.csv', sep='\t')

print('Removendo linhas com valores NaN na coluna TEXTO')
df = df.dropna(subset=['TEXTO'])

print('Dividindo os dados em features (X) e labels (y)')
X = df['TEXTO']
y = df['COD_ASSUNTO_CLASSIFICACAO']

print('Dividindo os dados em conjunto de treinamento e conjunto de teste')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print('Tokenizando os textos')
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)
X_train_tokenized = tokenizer.texts_to_sequences(X_train)
X_test_tokenized = tokenizer.texts_to_sequences(X_test)

print('Obtendo o tamanho máximo da sequência')
max_length = max([len(seq) for seq in X_train_tokenized])

print('Realizando padding das sequências')
X_train_padded = pad_sequences(X_train_tokenized, maxlen=max_length, padding='post')
X_test_padded = pad_sequences(X_test_tokenized, maxlen=max_length, padding='post')

print('Codificando os rótulos de classe')
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

print('Construindo o modelo BiLSTM')
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=128, input_length=max_length))
model.add(Bidirectional(LSTM(64)))
model.add(Dense(units=len(label_encoder.classes_), activation='softmax'))

print('Compilando o modelo')
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print('Treinando o modelo')
model.fit(X_train_padded, y_train_encoded, epochs=5, batch_size=32, validation_split=0.1)

print('Avaliando o modelo')
accuracy = model.evaluate(X_test_padded, y_test_encoded)[1]
print(f'Acurácia do modelo BiLSTM: {accuracy}')

# nao foi possivel executar, faltou memoria na maquina
# numpy.core._exceptions._ArrayMemoryError: Unable to allocate 9.40 GiB for an array with shape (1143, 2208227) and data type int32