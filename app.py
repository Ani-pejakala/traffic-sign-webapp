
from flask import Flask, render_template, request, send_file
import os
from run_model import predict

app = Flask(__name__)
UPLOAD_FOLDER = "/static/uploads/"
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
@app.route('/', methods=['GET', 'POST'])
def home_page():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', msg='No file selected')
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', msg='No file selected')

        if file and allowed_file(file.filename):
            file.save(os.path.join(os.getcwd() + UPLOAD_FOLDER, file.filename))
            out = predict("custom.h5", file)
            print(out)
            filename, file_extension = os.path.splitext(file.filename)
            return render_template('result.html', image=UPLOAD_FOLDER+file.filename,sign=out)
        else:
            return render_template('index.html', msg='No style selected')

    elif request.method == 'GET':
        return render_template('index.html')
