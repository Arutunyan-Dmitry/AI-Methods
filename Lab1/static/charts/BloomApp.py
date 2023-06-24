from flask import Flask, request
from BloomFilter import BloomFilter
import pandas as pd

app = Flask(__name__)
bloomfilter = BloomFilter(10, 30)


@app.route("/")
def home():
    global our_dataset
    our_dataset = pd.read_csv("D:\\Ulstu\\МИИ\\Lab1\\static\\datasets\\our-dataset.csv", encoding='windows-1251')
    return "<html>" \
           "<form Action='/search' Method='get' style='margin: 0 40%'>" \
           "<h5>Введите ключевые слова для поиска</h5>" \
           "<input type=text name='search_value'/>" \
           "<input type=submit value='Найти'/></form></html>"


@app.route("/search", methods=['GET'])
def search():
    data = request.args
    for item in our_dataset['Ключевые слова']:
        for word in item.split(" "):
            bloomfilter.add_to_filter(word)

    if bloomfilter.check_is_not_in_filter(data['search_value'].lower()):
        return '<h2 align="center"> Ничего не найдено :( </h2>'
    else:
        result = our_dataset[our_dataset['Ключевые слова'].str.contains(data['search_value'].lower())]
        html_out = ""
        for i in result.index.tolist():
            html_out += '<div style="margin: 0 10%" align"center"><p>' + str(result['Ключевые слова'][i]) + '<a ' \
            + 'href="' + str(result['Наборы данных'][i]) + '"> Ссылка </a>' + '</div>'
        return '<h2 align="center"> Вот что удалось найти </h2>' + html_out



if __name__ == "__main__":
    app.run(debug=True)
