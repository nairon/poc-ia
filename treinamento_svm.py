import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

print('Carregando os dados do CSV')
df = pd.read_csv('dados.csv', sep='\t')

print('Removendo linhas com valores NaN na coluna TEXTO')
df = df.dropna(subset=['TEXTO'])

print('Dividindo os dados em features (X) e labels (y)')
X = df['TEXTO']
y = df['COD_ASSUNTO_CLASSIFICACAO']

print('Dividindo os dados em conjunto de treinamento e conjunto de teste')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print('Realizando vetorização das palavras')
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

print('Criando e treinando o classificador SVM')
svm_classifier = SVC(kernel='linear')  # SVM com kernel linear
svm_classifier.fit(X_train_vectorized, y_train)

print('Fazendo previsões')
y_pred = svm_classifier.predict(X_test_vectorized)

print('Calculando a acurácia')
accuracy = accuracy_score(y_test, y_pred)
print(f'Acurácia do modelo SVM: {accuracy}')

# resultado 0.789308176100629