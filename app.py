from flask import Flask, request, render_template, redirect, url_for, session, jsonify
from PyPDF2 import PdfReader
from io import BytesIO
from joblib import load

app = Flask(__name__)

app.secret_key = 'secret_key'

# Carregando o modelo treinado
model = load('modelo_naive_bayes.pkl')
# Carregando o LabelEncoder
label_encoder = load('label_encoder.joblib')

@app.route('/formulario')
def formulario():
    success_message = session.pop('success_message', None)
    return render_template('formulario.html', success_message=success_message)

def extrair_texto(file):
    print('Extraindo texto do PDF')
    pdf_bytes = file.read()
    pdf_file = BytesIO(pdf_bytes)
    pdf_reader = PdfReader(pdf_file)
    full_text = ""
    for page in pdf_reader.pages:
        full_text += page.extract_text()
    # print("Texto do PDF\n", full_text)
    return full_text

@app.route('/upload', methods=['POST'])
def upload_file():
    # Verifica se a solicitação contém um arquivo
    if 'file' not in request.files:
        return 'Nenhum arquivo enviado', 400

    file = request.files['file']

    # Verifica se o nome do arquivo está vazio
    if file.filename == '':
        return 'Nome do arquivo vazio', 400

    try:
        full_text = extrair_texto(file)

        # Realiza a previsão com o modelo
        print('Realizando o predict do texto')
        prediction = model.predict([full_text])
        prediction_list = prediction.tolist()

        # Traduzir o label predito de volta para o código original
        original_label = label_encoder.inverse_transform(prediction_list)

        print('Resultado')
        print(prediction_list)
        print(original_label)

        print('Realizando a previsão das probabilidades para cada classe')
        probabilities = model.predict_proba([full_text])

        # Prepara a lista de resultados com probabilidades
        results = []
        for class_index, prob in enumerate(probabilities[0]):
            results.append({'class': class_index, 'probability': prob})

        # Ordena os resultados por probabilidade (maior para menor)
        results = sorted(results, key=lambda x: x['probability'], reverse=True)
        print(results)

        # jsonify({'prediction': prediction.tolist()})

        session['success_message'] = f'{file.filename} => {original_label}'
        return redirect(url_for('formulario'))

    except Exception as e:
        return f'Ocorreu um erro ao ler o arquivo PDF: {str(e)}', 500

if __name__ == '__main__':
    app.run(debug=True)