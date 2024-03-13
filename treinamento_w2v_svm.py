import pandas as pd
from gensim.models import Word2Vec
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

print('Carregando os dados do CSV')
df = pd.read_csv('dados.csv', sep='\t')

print('Removendo linhas com valores NaN na coluna TEXTO')
df = df.dropna(subset=['TEXTO'])

print('Preparando os dados para treinamento do Word2Vec')
sentences = [text.split() for text in df['TEXTO']]

print('Treinando o modelo Word2Vec')
model_w2v = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

print('Construindo vetores de documentos usando Word2Vec')
X = []
for sentence in sentences:
    vector = sum(model_w2v.wv[word] for word in sentence if word in model_w2v.wv.key_to_index) / len(sentence)
    X.append(vector.tolist())

print('Separando os dados em conjuntos de treinamento e teste')
X_train, X_test, y_train, y_test = train_test_split(X, df['COD_ASSUNTO_CLASSIFICACAO'], test_size=0.2, random_state=42)

print('Criando e treinando o modelo SVM')
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

print('Fazendo previsões')
y_pred = svm_model.predict(X_test)

print('Avaliando o modelo')
accuracy = accuracy_score(y_test, y_pred)
print(f'Acurácia do modelo SVM com Word2Vec: {accuracy}')

# resultado 0.8144654088050315