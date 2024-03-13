import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
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
dump(label_encoder, 'label_encoder_gbm2.joblib')

print('\nVisualizando as primeiras linhas do dataframe')
print(df.head())

print('\nVerificando informações sobre o dataframe')
print(df.info())

print('\nExplorando os rótulos de classificação')
print(df['COD_ASSUNTO_CLASSIFICACAO'].value_counts())
print(df['LABEL'].value_counts())

print('Separando os dados em features (X) e labels (y)')
X = df['TEXTO']
y = df['LABEL']

print('Dividindo os dados em conjunto de treinamento e conjunto de teste')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print('Realizando vetorização das palavras')
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

print("Configurando o GridSearchCV para GBM")
param_grid = {
    'n_estimators': [10, 25],
    'learning_rate': [0.01, 0.05, 0.1]
}
gbm_classifier = GradientBoostingClassifier(random_state=42)
grid_search = GridSearchCV(gbm_classifier, param_grid, cv=3, n_jobs=-1)

print("Treinando o modelo com GridSearchCV")
grid_search.fit(X_train_vectorized, y_train)

print("Melhores parâmetros encontrados:")
print(grid_search.best_params_)

print("Salvando o melhor modelo encontrado")
best_model = grid_search.best_estimator_
dump(best_model, 'modelo_gbm2.joblib')

print("Fazendo previsões no conjunto de teste")
y_pred = best_model.predict(X_test_vectorized)

print('Calculando a acurácia')
accuracy = accuracy_score(y_test, y_pred)
print(f'Acurácia do modelo Gradient Boosting Machines (GBM): {accuracy}')

# Melhores parâmetros encontrados:
# {'learning_rate': 0.1, 'n_estimators': 25}
# Acurácia do modelo Gradient Boosting Machines (GBM): 0.8427672955974843