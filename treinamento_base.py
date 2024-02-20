import sys
import csv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from joblib import dump, load

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

print('Criando um pipeline que transforma os textos em vetores e treina o modelo Naive Bayes')
model = make_pipeline(CountVectorizer(), MultinomialNB())

print('Treinando o modelo')
model.fit(X_train, y_train)

print('Fazendo previsões no conjunto de teste')
y_pred = model.predict(X_test)

print('Calculando a precisão do modelo')
accuracy = accuracy_score(y_test, y_pred)
print(f'Acurácia do modelo: {accuracy}')

# Salvar o modelo treinado
# joblib.dump(model, 'modelo_classificacao.pkl')
