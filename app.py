from flask import Flask, render_template, request, send_file
import pandas as pd

from clustering import *

app = Flask(__name__)

def get_data(file):
  df = pd.read_csv(file)
  current_data = df.iloc[:]

  return current_data

@app.route("/")
def index():
	return render_template("index.html")

@app.route("/ta")
def ta_view():
  current_data = get_data('files/data_clustering.csv')
  return render_template("scopeTA.html", data=current_data)

@app.route("/newcluster", methods=['GET', 'POST'])
def newcluster_view():
  if request.method == 'POST':
    file1 = request.files['file1']
    file2 = request.files['file2']
    file3 = request.files['file3']
    file1.save('files/' + file1.filename)
    file2.save('files/' + file2.filename)
    file3.save('files/' + file3.filename)

    df1 = preprocessing('files/' + file1.filename)
    df2 = preprocessing('files/' + file2.filename)
    df3 = preprocessing('files/' + file3.filename)
    df = concatenate(df1, df2, df3)
    data = fuzzy_clustering(df)

    current_data = get_data('files/new_clustering.csv')
    data['tabel'] = current_data
    return render_template("newcluster.html", data=data)

  return render_template("newcluster.html")

@app.route('/download/<filename>')
def download_file(filename):
    file_path = 'files/' + filename
    return send_file(file_path, as_attachment=True)

if __name__ == '__main__':
	app.run(debug=True)