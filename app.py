from flask import Flask, request, render_template, redirect, url_for, session
from PyPDF2 import PdfReader
from io import BytesIO

app = Flask(__name__)

app.secret_key = 'secret_key'

@app.route('/formulario')
def formulario():
    success_message = session.pop('success_message', None)
    return render_template('formulario.html', success_message=success_message)

@app.route('/upload', methods=['POST'])
def upload_file():
    # Verifica se a solicitação contém um arquivo
    if 'file' not in request.files:
        return 'Nenhum arquivo enviado', 400

    file = request.files['file']

    # Verifica se o nome do arquivo está vazio
    if file.filename == '':
        return 'Nome do arquivo vazio', 400

    # Salva o arquivo enviado para um diretório local
    # file.save(file.filename)

    # Lê o conteúdo do arquivo PDF
    try:
        # with open(file.filename, "rb") as input_pdf:

        pdf_bytes = file.read()
        pdf_file = BytesIO(pdf_bytes)

        # Criando um objeto PdfFileReader
        # pdf_reader = PdfReader(input_pdf)
        pdf_reader = PdfReader(pdf_file)

        # Obtendo o número de páginas do arquivo PDF
        # num_pages = len(pdf_reader.pages)

        # Inicializa uma string para armazenar o texto completo do PDF
        full_text = ""

        # Lendo o texto de cada página e concatenando na string full_text
        for page in pdf_reader.pages:
            full_text += page.extract_text()

        print("Texto do PDF\n", full_text)

        session['success_message'] = f'Arquivo {file.filename} enviado com sucesso'
        return redirect(url_for('formulario'))

    except Exception as e:
        return f'Ocorreu um erro ao ler o arquivo PDF: {str(e)}', 500

if __name__ == '__main__':
    app.run(debug=True)