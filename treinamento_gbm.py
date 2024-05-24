import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from joblib import dump

print('Carregando os dados do CSV')
df = pd.read_csv('dados.csv', sep='\t')

print('Removendo linhas com valores NaN na coluna TEXTO')
df = df.dropna(subset=['TEXTO'])

print('Inicializando o LabelEncoder')
label_encoder = LabelEncoder()
print('Criando a coluna LABEL com o LabelEncoder')
df['LABEL'] = label_encoder.fit_transform(df['COD_ASSUNTO_CLASSIFICACAO'])

print('Salvando o LabelEncoder em um arquivo')
dump(label_encoder, 'label_encoder_gbm.joblib')

print('Separando os dados em features (X) e labels (y)')
X = df['TEXTO']
y = df['LABEL']

print('Dividindo os dados em conjunto de treinamento e conjunto de teste')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print('Realizando vetorização das palavras')
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

print('Salvando CountVectorizer em um arquivo')
dump(vectorizer, 'count_vectorizer_gbm.joblib')

print("Criando e treinando o classificador Gradient Boosting Machines (GBM)")
gbm_classifier = GradientBoostingClassifier(n_estimators=25, learning_rate=0.1, random_state=42)
gbm_classifier.fit(X_train_vectorized, y_train)

print("Fazendo previsões no conjunto de teste")
y_pred = gbm_classifier.predict(X_test_vectorized)

print('Calculando a acurácia')
accuracy = accuracy_score(y_test, y_pred)
print(f'Acurácia do modelo Gradient Boosting Machines (GBM): {accuracy}')

print('Calculando o MAE')
mae = mean_absolute_error(y_test, y_pred)
print(f'O MAE do modelo Gradient Boosting Machines (GBM): {mae}')

print('Salvando modelo em um arquivo')
dump(gbm_classifier, 'modelo_gbm.joblib')

# resultado 0.839622641509434 (n_estimators = 10)
# O MAE do modelo Gradient Boosting Machines (GBM): 0.5220125786163522