from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import json
import similarity_search

app = Flask(__name__)

with open('paths.json', 'r')as f:
    paths = json.load(f)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == "POST":
        image = request.files['file']  # file is name of input tag

        filename = secure_filename(image.filename)

        target_path = f'target_image_path/{filename}'
        results = similarity_search.index_search(target_path, 4)
        result_paths = []
        print(paths)

        for i in results[0]:
            result_paths.append(paths[f'{i}'])

        return str(result_paths)

    return '''
        <!doctype html>
        <title>Upload new File</title>
        <h1>Upload new File</h1>
        <form method=post enctype=multipart/form-data>
          <p><input type=file name=file>
             <input type=submit value=Upload>
             
        </form>
        '''


if __name__ == '__main__':
    app.run(debug=True)
