from flask import Flask, redirect, url_for,request
from num2words import num2words

app = Flask(__name__)
@app.route("/")
def home():
    return "<html><form Action='http://127.0.0.1:5000/numtext' Metod=get><input type=text size=20 name=name><input type=submit value='Перевести'></form></html>"

@app.route("/<name>")
def user(name):
    return f"Hello, {name}!"

@app.route("/admin")
def admin():
    return redirect(url_for("num_text",name=12))

@app.route("/numtext/<name>")
def num_text(name):
    return num2words(name,lang='eng')

@app.route("/numtext",methods=['GET'])
def numtext():
    data = request.args
    return num2words(data['name'],lang='ru')

if __name__ == "__main__":
    app.run(debug=True)