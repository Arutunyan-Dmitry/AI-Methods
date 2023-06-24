import math

from flask import Flask, request
import matplotlib.pyplot as plt
import os.path
import random
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
import pymorphy2
import pandas as pd


app = Flask(__name__)
picfld = os.path.join('static', 'images')
txtfld = os.path.join('static', 'text')
app.config['UPLOAD_FOLDER'] = picfld
app.config['FILE_FOLDER'] = txtfld


@app.route("/")
def home():
    return '<html><body><h1 align="center">Методы искусственного интеллекта</h1>' \
           '<div style="margin: 3% 30% 5% 30%;">' \
           '<p><a href="/GeneticAlg">Генетический алгоритм</a></p>' \
           '<p><a href="/FuzzySetsInput">Нечёткие множества</a></p>' \
           '<p><a href="/LinquaParamInput">Лингвистические переменные и шкалы</a></p>' \
           '<p><a href="/ClasterParamInput">Нечёткая кластеризация объектов</a></p>' \
           '<p><a href="/LogicOutInput">Нечёткий логический вывод</a></p>' \
           '<p><a href="/TextAnalysis">Анализ текста</a></p>' \
           '<p><a href="/NeuralNet">Нейронная сеть</a></p>' \
           '</div></body></html>'


@app.route("/GeneticAlg")
def genetic():
    return "<form Action='/GeneticAlgExc' Method='get' style='margin: 0 30%'>" \
           "<h2>Ввод данных городов и поставщиков</h2><h3>Поставщик 1 </h3>" \
           "<p>Цена за км <input type=text name=price1 value=100 /></p>" \
           "<span>Расстояние до города 1 </span><input type=text name=from1to1 value=1 />" \
           "<span> Расстояние до города 2 </span><input type=text name=from1to2 value=2 />" \
           "<span> Расстояние до города 3 </span><input type=text name=from1to3 value=3 />" \
           "<h3>Поставщик 2 </h3><p>Цена за км <input type=text name=price2 value=150 /></p>" \
           "<span>Расстояние до города 1 </span><input type=text name=from2to1 value=3 />" \
           "<span> Расстояние до города 2 </span><input type=text name=from2to2 value=1 />" \
           "<span> Расстояние до города 3 </span><input type=text name=from2to3 value=2 />" \
           "<h3>Поставщик 3 </h3><p>Цена за км <input type=text name=price3 value=130 /></p>" \
           "<span>Расстояние до города 1 </span><input type=text name=from3to1 value=2 />" \
           "<span>Расстояние до города 2 </span><input type=text name=from3to2 value=3 />" \
           "<span>Расстояние до города 3 </span><input type=text name=from3to3 value=1 />" \
           "<h3> Потребности городов </h3>" \
           "<span>Потребность города 1 (не более 15) </span><input type=text name=req1 value=10 />" \
           "<span> Потребность города 2 (не более 15) </span><input type=text name=req2 value=5 />" \
           "<span> Потребность города 3 (не более 15) </span><input type=text name=req3 value=7 />" \
           "<p><input type=submit value='Выполнить' /></p></form></html>"


@app.route("/GeneticAlgExc", methods=['GET'])
def geneticex():
    data = request.args
    requirements = [int(data['req1']), int(data['req2']), int(data['req3'])]
    durations = [[int(data['from1to1']) * int(data['price1']), int(data['from1to2']) * int(data['price1']),
                  int(data['from1to3']) * int(data['price1'])],
                 [int(data['from2to1']) * int(data['price2']), int(data['from2to2']) * int(data['price2']),
                  int(data['from2to3']) * int(data['price2'])],
                 [int(data['from3to1']) * int(data['price3']), int(data['from3to2']) * int(data['price3']),
                  int(data['from3to3']) * int(data['price3'])]]
    coordinates = [[random.randint(0, 1) for i in range(3)] for j in range(3)]

    # requirements output
    reqout = '<p>Необходимое кол-во продуктов для городов</p><table><tr><td style="border: 1px solid #333;">' \
             'Город 1 </td><td style="border: 1px solid #333;"> Город 2 </td><td style="border: 1px solid #333;">' \
             'Город 3 </td></tr><tr><td style="border: 1px solid #333;">' + str(requirements[0]) + '</td><td ' \
                                                                                                   'style="border: 1px solid #333;">' + str(
        requirements[1]) + '</td><td  style="border: 1px solid #333;"' \
                           '>' + str(requirements[2]) + '</td></tr></table>'

    # durations output
    durout = '<p>Цены перевозки от одного поставщика до другого</p>' \
             '<table><tr><th style="border: 1px solid #333;">Поставщики</th><th style="border: 1px solid #333;">Г1</th>' \
             '<th style="border: 1px solid #333;">Г2</th><th style="border: 1px solid #333;">Г3</th></tr>'
    for i in range(3):
        durout += '<tr><td style="border: 1px solid #333;">Поставщик ' + str(i + 1) + '</td>'
        for j in range(3):
            durout += '<td style="border: 1px solid #333;">' + str(durations[j][i]) + '</td>'
        durout += '</tr>'
    durout += '</table>'

    # array output
    out = '<p>Случайно расставленные маршруты</p>' \
          '<table><tr><th style="border: 1px solid #333;">Поставщики</th><th style="border: 1px solid #333;">Г1</th>' \
          '<th style="border: 1px solid #333;">Г2</th><th style="border: 1px solid #333;">Г3</th></tr>'
    for i in range(3):
        out += '<tr><td style="border: 1px solid #333;">Поставщик ' + str(i + 1) + '</td>'
        for j in range(3):
            out += '<td style="border: 1px solid #333;">' + str(coordinates[i][j]) + '</td>'
        out += "</tr>"
    out += "</table>"

    # Mutation
    for i in range(3):
        for j in range(3):
            sum = (coordinates[0][i] + coordinates[1][i] + coordinates[2][i]) * 5
            if (requirements[i] + 5) <= sum:
                if coordinates[j][i] == 1:
                    coordinates[j][i] = 0
    for i in range(3):
        for j in range(3):
            sum = (coordinates[0][i] + coordinates[1][i] + coordinates[2][i]) * 5
            if sum < requirements[i]:
                if coordinates[j][i] == 0:
                    coordinates[j][i] = 1

    # affordable price
    medprice = [0, 0, 0]
    for i in range(3):
        min = 999
        max = 0
        for j in range(3):
            if durations[j][i] > max:
                max = durations[j][i]
            if durations[j][i] < min:
                min = durations[j][i]
        medprice[i] = int((min + max) / 2 + 1)

    # medprice output
    medout = '<p>Возможные затраты на перевозку каждого города</p><table><tr><td style="border: 1px solid #333;">' \
             'Город 1 </td><td style="border: 1px solid #333;"> Город 2 </td><td style="border: 1px solid #333;">' \
             'Город 3 </td></tr><tr><td style="border: 1px solid #333;">' + str(medprice[0]) + '</td><td ' \
                                                                                               'style="border: 1px solid #333;">' + str(
        medprice[1]) + '</td><td  style="border: 1px solid #333;"' \
                       '>' + str(medprice[2]) + '</td></tr></table>'

    # pricing after mutation
    pricing = [0, 0, 0]
    for i in range(3):
        sum = 0
        for j in range(3):
            if coordinates[j][i] != 0:
                sum += durations[j][i]
        pricing[i] = sum

    # array output
    out1 = '<p>Гарантированно подходящие по кол-ву продуктов маршруты (мутации)</p>' \
           '<table><tr><th style="border: 1px solid #333;">Поставщики</th><th style="border: 1px solid #333;">Г1</th>' \
           '<th style="border: 1px solid #333;">Г2</th><th style="border: 1px solid #333;">Г3</th></tr>'
    for i in range(3):
        out1 += '<tr><td style="border: 1px solid #333;">Поставщик ' + str(i + 1) + '</td>'
        for j in range(3):
            out1 += '<td style="border: 1px solid #333;">' + str(coordinates[i][j]) + '</td>'
        out1 += "</tr>"
    out1 += '<tr><td style="border: 1px solid #333;"> Цена: </td>'
    for i in range(3):
        out1 += '<td style="border: 1px solid #333;">' + str(pricing[i]) + '</td>'
    out1 += "</table>"

    # crossover
    for i in range(3):
        count = 0
        for j in range(3):
            if coordinates[j][i] == 1:
                count += 1
        while pricing[i] > (medprice[i] * count):
            sum = 0
            a = random.randint(0, 2)
            b = random.randint(0, 2)
            tmp = coordinates[a][i]
            coordinates[a][i] = coordinates[b][i]
            coordinates[b][i] = tmp
            for j in range(3):
                if coordinates[j][i] != 0:
                    sum += durations[j][i]
            pricing[i] = sum

    pricing1 = [0, 0, 0]
    for i in range(3):
        sum = 0
        for j in range(3):
            if coordinates[j][i] != 0:
                sum += durations[j][i]
        pricing1[i] = sum

    # array output
    out2 = '<p>Гарантированно подходящие по кол-ву продуктов и цене маршруты (скрещивание), но не самые оптимальные</p>' \
           '<table><tr><th style="border: 1px solid #333;">Поставщики</th><th style="border: 1px solid #333;">Г1</th>' \
           '<th style="border: 1px solid #333;">Г2</th><th style="border: 1px solid #333;">Г3</th></tr>'
    for i in range(3):
        out2 += '<tr><td style="border: 1px solid #333;">Поставщик ' + str(i + 1) + '</td>'
        for j in range(3):
            out2 += '<td style="border: 1px solid #333;">' + str(coordinates[i][j]) + '</td>'
        out2 += "</tr>"
    out2 += '<tr><td style="border: 1px solid #333;"> Цена: </td>'
    for i in range(3):
        out2 += '<td style="border: 1px solid #333;">' + str(pricing[i]) + '</td>'
    out2 += "</table>"

    return "<h2>Генетический алгоритм</h2>" \
           "<p>" + reqout + "</p>" \
                            "<p>" + durout + "</p>" \
                                             "<p>" + medout + "</p>" \
                                                              "<p>" + out + "</p>" \
                                                                            "<p>" + out1 + "</p>" \
                                                                                           "<p>" + out2 + "</p>"


@app.route("/FuzzySetsInput")
def fuzzysetsinput():
    return '<body style="background: #B0C4DE;"><form Action="/FuzzySets" Method="get">' \
           '<div style="margin: 2% 20% 5% 20%; background: #FFFFFF; padding: 1% 5%">' \
           '<h2 align="center">Множество 1</h2>' \
           '<p>Название множества <input style="min-width: 35%;" type=text name=s1name value="Натуральные числа, близкие к (4, 6)" /></p>' \
           '<p>Ввод параметров множества </p>' \
           '<p style="margin: 1%;">a: <input style="max-width: 5%;" type=number name=s1a value=2 /></p>' \
           '<p style="margin: 1%;">b: <input style="max-width: 5%;" type=number name=s1b value=4 /></p>' \
           '<p style="margin: 1%;">c: <input style="max-width: 5%;" type=number name=s1c value=6 /></p>' \
           '<p style="margin: 1%;">d: <input style="max-width: 5%;" type=number name=s1d value=8 /></p>' \
           '<p>Ввод объектов множества (числа через пробел) <input style="max-width: 13%;" type=text name=s1obj value="1 3 5 6 7" /></div>' \
           '<div style="margin: 2% 20% 1% 20%; background: #FFFFFF; padding: 1% 5%">' \
           '<h2 align="center">Множество 2</h2>' \
           '<p>Название множества <input style="min-width: 35%;" type=text name=s2name value="Небольшие натуральные числа" /></p>' \
           '<p>Ввод параметров множества </p>' \
           '<p style="margin: 1%;">a: <input style="max-width: 5%;" type=number name=s2a value=1 /></p>' \
           '<p style="margin: 1%;">b: <input style="max-width: 5%;" type=number name=s2b value=1 /></p>' \
           '<p style="margin: 1%;">c: <input style="max-width: 5%;" type=number name=s2c value=3 /></p>' \
           '<p style="margin: 1%;">d: <input style="max-width: 5%;" type=number name=s2d value=10 /></p>' \
           '<p>Ввод объектов множества (числа через пробел) <input style="max-width: 13%;" type=text name=s2obj value="1 2 3 5 6 7 10" /></div>' \
           '<h1 align="center" style="margin: 3% 45% 10% 45%"><input type=submit value="Далее" /></h1></form></html>'


@app.route("/FuzzySets", methods=['GET'])
def fuzzysets():
    data = request.args
    name1 = str(data['s1name'])
    name2 = str(data['s2name'])
    a1 = int(data['s1a'])
    a2 = int(data['s2a'])
    b1 = int(data['s1b'])
    b2 = int(data['s2b'])
    c1 = int(data['s1c'])
    c2 = int(data['s2c'])
    d1 = int(data['s1d'])
    d2 = int(data['s2d'])
    obj1 = str(data['s1obj']).split()
    for i in range(len(obj1)):
        obj1[i] = int(obj1[i])
    obj2 = str(data['s2obj']).split()
    for i in range(len(obj2)):
        obj2[i] = int(obj2[i])

    # plot
    plt.plot([0, a1, b1, c1, d1, 11], [0, 0, 1, 1, 0, 0], color='#8B008B')
    plt.plot([0, a2, b2, c2, d2, 11], [0, 0, 1, 1, 0, 0], color='#228B22')
    plt.savefig('static/images/fuzzyfunc.png')

    # applience
    appl1 = [[0.00] * len(obj1) for i in range(2)]
    for i in range(len(appl1[0])):
        appl1[0][i] = obj1[i]
    for i in range(len(appl1[0])):
        if b1 <= appl1[0][i] <= c1:
            appl1[1][i] = 1
        elif a1 < appl1[0][i] < b1:
            appl1[1][i] = round((appl1[0][i] - a1) / (b1 - a1), 2)
        elif c1 < appl1[0][i] < d1:
            appl1[1][i] = round((appl1[0][i] - d1) / (c1 - d1), 2)
        elif appl1[0][i] <= a1:
            appl1[1][i] = 0
        elif appl1[0][i] >= d1:
            appl1[1][i] = 0

    appl2 = [[0.00] * len(obj2) for i in range(2)]
    for i in range(len(appl2[0])):
        appl2[0][i] = obj2[i]
    for i in range(len(appl2[0])):
        if b2 <= appl2[0][i] <= c2:
            appl2[1][i] = 1
        elif a2 < appl2[0][i] < b2:
            appl2[1][i] = round((appl2[0][i] - a2) / (b2 - a2), 2)
        elif c2 < appl2[0][i] < d2:
            appl2[1][i] = round((appl2[0][i] - d2) / (c2 - d2), 2)
        elif appl2[0][i] <= a2:
            appl2[1][i] = 0
        elif appl2[0][i] >= d2:
            appl2[1][i] = 0

    # output
    appl1out = '<table><tr><th style="border: 1px solid #333;"></th><th style="border: 1px solid #333;">Значение</th>' \
               '<th style="border: 1px solid #333;">Принадлежность</th></tr>'
    for i in range(len(appl1[0])):
        appl1out += '<tr><td style="border: 1px solid #333;">Объект ' + str(i + 1) + '</td>'
        for j in range(len(appl1)):
            appl1out += '<td style="border: 1px solid #333; text-align: center;">' + str(appl1[j][i]) + '</td>'
        appl1out += '</tr>'
    appl1out += '</table>'
    appl2out = '<table><tr><th style="border: 1px solid #333;"></th><th style="border: 1px solid #333;">Значение</th>' \
               '<th style="border: 1px solid #333;">Принадлежность</th></tr>'
    for i in range(len(appl2[0])):
        appl2out += '<tr><td style="border: 1px solid #333;">Объект ' + str(i + 1) + '</td>'
        for j in range(len(appl2)):
            appl2out += '<td style="border: 1px solid #333; text-align: center;">' + str(appl2[j][i]) + '</td>'
        appl2out += '</tr>'
    appl2out += '</table>'

    # crossing
    crossing = '<p style="margin-left: 15%; margin-bottom: 12%;">M1 ⋂ M2 = {'
    for i in range(len(appl1[0])):
        for j in range(len(appl2[0])):
            if appl1[0][i] == appl2[0][j]:
                crossing += ' <' + str(appl1[0][i]) + '/'
                if appl1[1][i] <= appl2[1][j]:
                    crossing += str(appl1[1][i]) + '>,'
                else:
                    crossing += str(appl2[1][j]) + '>,'
    crossing = crossing[:-1]
    crossing += ' } </p>'

    return '<body style="background: #B0C4DE;"><div style="margin: 2% 10% 3% 10%; background: #FFFFFF; padding: 1% 5%">' \
           '<h1 align="center">Нечёткие множства</h1><div style="display: flex; justify-content: space-between">' \
           '<div><h3>Множество 1</h3><h4>' + name1 + '</h4><p>Задано функцией принадлежности трапециидального вида<p>' \
                                                     '<p>Параметры функции: a = ' + str(a1) + ', b = ' + str(
        b1) + ', c = ' + str(c1) + ', d = ' + str(d1) + '</p>' \
                                                        '<p>Обьекты функции: ' + str(obj1) + '</p></div>' \
                                                                                             '<div><h3>Множество 2</h3><h4>' + name2 + '</h4><p>Задано функцией принадлежности трапециидального вида<p>' \
                                                                                                                                       '<p>Параметры функции: a = ' + str(
        a2) + ', b = ' + str(b2) + ', c = ' + str(c2) + ', d = ' + str(d2) + '</p>' \
                                                                             '<p>Обьекты функции: ' + str(
        obj2) + '</p></div></div>' \
                '<h2 align="center" style="margin-bottom: 0;">Графики функций принадлежности обеих множеств</h2>' \
                '<h2 align="center" style="margin: 0;"><img src="' + os.path.join(app.config['UPLOAD_FOLDER'],
                                                                                  'fuzzyfunc.png') + '"/></h2>' \
                                                                                                     '<p style="margin: 0% 35%;">Фиолетовый - 1е множество, зелёный - 2е множество</p>' \
                                                                                                     '<h2 align="center">Рассчёт принадлежности объектов</h2><div style="display: flex; justify-content: space-around">' \
                                                                                                     '<div><h3>Множество 1</h3>' + appl1out + '</div><div><h3>Множество 2</h3>' + appl2out + '</div></div>' \
                                                                                                                                                                                             '<h2 align="center" style="margin-bottom: 4%">Пересечение объектов двух множеств</h2>' + crossing


@app.route("/LinquaParamInput")
def linquaparaminput():
    return '<body style="background: #B0C4DE;"><form Action="/LinquaParam" Method="get">' \
           '<div style="margin: 2% 20% 5% 20%; background: #FFFFFF; padding: 1% 5%">' \
           '<h2 align="center">Лингвистические переменные и шкалы (трапецевидная функция)</h2>' \
           '<h3>Оценка затрат на оплату труда</h3>' \
           '<p>Ввод параметров множества </p>' \
           '<p style="margin: 1%;">a: <input style="max-width: 15%;" type=number name=a value=5000 /></p>' \
           '<p style="margin: 1%;">b: <input style="max-width: 15%;" type=number name=b value=20000 /></p>' \
           '<p style="margin: 1%;">c: <input style="max-width: 15%;" type=number name=c value=35000 /></p>' \
           '<p style="margin: 1%;">d: <input style="max-width: 15%;" type=number name=d value=50000 /></p>' \
           '<p>Ввод кол-ва лингвистических параметров (от 3 до 7): <input style="max-width: 5%;" type=number name=div value="7"/></div>' \
           '<h1 align="center" style="margin: 0% 45% 10% 45%"><input type=submit value="Далее" /></h1></form></html>'


@app.route("/LinquaParam", methods=['GET'])
def linquaparam():
    linqArray = ['Очень низкие', 'Низкие', 'Более менее средние', 'Средние', 'Более менее средние', 'Высокие',
                 'Очень высокие']
    colors = ['#800080', '#8B0000', '#B22222', '#DC143C', '#CD5C5C', '#F08080', '#FFB6C1']

    data = request.args
    a = int(data['a'])
    b = int(data['b'])
    c = int(data['c'])
    d = int(data['d'])
    div = int(data['div'])

    # plot
    plt.plot([0, a, b, c, d, (d + 5000)], [0, 0, 1, 1, 0, 0], color='#8B008B')
    plt.savefig('static/images/linqfunc.png')
    plt.close()

    # linqdivision
    result = [[0, 0, '', '']] * div
    if div == 3:
        result[0] = [0, a, linqArray[0], colors[0]]
        result[1] = [a, d, linqArray[3], colors[3]]
        result[2] = [d, d + 5000, linqArray[6], colors[6]]
    if div == 4:
        result[0] = [0, a, linqArray[0], colors[0]]
        result[1] = [a, ((b + c) / 2), linqArray[2], colors[2]]
        result[2] = [((b + c) / 2), d, linqArray[4], colors[4]]
        result[3] = [d, d + 5000, linqArray[6], colors[6]]
    if div == 5:
        result[0] = [0, a, linqArray[0], colors[0]]
        result[1] = [a, b, linqArray[1], colors[1]]
        result[2] = [b, c, linqArray[3], colors[3]]
        result[3] = [c, d, linqArray[5], colors[5]]
        result[4] = [d, d + 5000, linqArray[6], colors[6]]
    if div == 6:
        result[0] = [0, a, linqArray[0], colors[0]]
        result[1] = [a, b, linqArray[1], colors[1]]
        result[2] = [b, ((b + c) / 2), linqArray[2], colors[2]]
        result[3] = [((b + c) / 2), c, linqArray[4], colors[4]]
        result[4] = [c, d, linqArray[5], colors[5]]
        result[5] = [d, d + 5000, linqArray[6], colors[6]]
    if div == 7:
        result[0] = [0, a, linqArray[0], colors[0]]
        result[1] = [a, ((a + b) / 2), linqArray[1], colors[1]]
        result[2] = [((a + b) / 2), b, linqArray[2], colors[2]]
        result[3] = [b, c, linqArray[3], colors[3]]
        result[4] = [c, ((c + d) / 2), linqArray[4], colors[4]]
        result[5] = [((c + d) / 2), d, linqArray[5], colors[5]]
        result[6] = [d, d + 5000, linqArray[6], colors[6]]

    # scale
    k = 155
    plt.title("Затраты")
    for i in range(div):
        for j in range(40):
            plt.plot([result[i][0], result[i][1]], [j, j], color=result[i][3])
        plt.text(result[i][0], k, result[i][2])
        plt.plot([result[i][0], result[i][0]], [0, k - 5], color=result[i][3])
        k = k - 15
    y_ticks = [0, 40, 100, 160, 220, 280]
    y_labels = [' ', ' ', ' ', ' ', ' ', ' ']
    plt.yticks(ticks=y_ticks, labels=y_labels)
    plt.savefig('static/images/linqparam.png')

    # output
    linqout = ''
    for i in range(div):
        linqout += '<p><b>' + result[i][2] + ' затраты:</b> от: ' + str(result[i][0]) + ' до: ' + str(
            result[i][1]) + '</p>'

    return '<body style="background: #B0C4DE;"><div style="margin: 2% 10% 3% 10%; background: #FFFFFF; padding: 1% 5%">' \
           '<h1 align="center">Лингвистические переменные и шкалы</h1><h2>Оценка стоимости оплаты труда</h2>' \
           '<p>Параметры функции: a = ' + str(a) + ', b = ' + str(b) + ', c = ' + str(c) + ', d = ' + str(d) + '</p>' \
           '<p>Количество лингвистических параметров: ' + str(div) + '</p>' \
           '<h2 align="center" style="margin-bottom: 0;">График функции принадлежности</h2>' \
           '<h2 align="center" style="margin: 0;"><img src="' + os.path.join(app.config['UPLOAD_FOLDER'], 'linqfunc.png') + '"/></h2>' \
           '<h2 align="center" style="margin-bottom: 0;">Шкала лингвистических параметров</h2>' \
           '<h2 align="center" style="margin: 0;"><img src="' + os.path.join(app.config['UPLOAD_FOLDER'], 'linqparam.png') + '"/></h2>' \
           '<h2 align="center" style="margin-bottom: 0;">Значения шкалы лингвистических параметров</h2>' + linqout + '</body>'


@app.route("/ClasterParamInput")
def clasterparaminput():
    return '<body style="background: #B0C4DE;"><form Action="/ClasterisationAlg" Method="get">' \
           '<div style="margin: 2% 20% 5% 20%; background: #FFFFFF; padding: 1% 5%">' \
           '<h2 align="center">Нечёткая кластеризация объектов</h2><h3>Кластеризация по возрасту людей</h3>' \
           '<p>Введите кол-во кластеров (максимум 6)<input style="max-width: 5%;" type=number name=numb value=4 /></p>' \
           '<p>Введите возраста людей для кластеризации (через запяую, значения от 0 до 100) </p>' \
           '<input style="min-width: 73%; margin-bottom: 5%" type=text name=ages value="10, 12, 23, 35, 60, 71" /></div>' \
           '<h1 align="center" style="margin: 3% 45% 10% 45%"><input type=submit value="Далее" /></h1></form></body>'


@app.route("/ClasterisationAlg", methods=['GET'])
def clasterisationalg():
    # initialization
    linqAxe = [[0, 20, 'очень молодой'], [20, 40, 'молодой'], [40, 60, 'взрослый'], [60, 70, 'старый'], [70, 100, 'очень старый']]
    plotColors = ['#800080', '#8B0000', '#B22222', '#DC143C', '#CD5C5C']
    highlighters = ["#87CEFA", "#F08080", "#90EE90", "#EE82EE", "#FFDEAD", "#7FFFD4"]
    m = 1.6

        # getting params
    data = request.args
    nb = int(data['numb'])
    ages = str(data['ages']).split(',')

    objects = [float(i) for i in ages]
    clasterCentres = [random.randint(0, 100) for i in range(nb)]
    affilationDegrees = [[0 for i in range(nb)] for j in range(len(ages))]
    lambdaCenters = [[0 for i in range(nb)] for j in range(len(ages))]
    lambdaDegrees = [[[0 for k in range(nb)] for j in range(len(ages))] for i in range(nb)]
    out6 = ''

        # part-time
    result = [[0 for i in range(nb)] for j in range(len(ages))]

    # formalisation
    for i in range(len(affilationDegrees)):
        summ = 0
        j: int
        for j in range(len(affilationDegrees[0]) - 1):
            affilationDegrees[i][j] = round(random.uniform(0.01, float(1 / nb)), 2)
            summ += affilationDegrees[i][j]
        affilationDegrees[i][len(affilationDegrees[0]) - 1] = round(1 - summ, 2)
    # output
    out1 = '<table style="min-width: 30%;"><tr><th style="border: 1px solid #333;" rowspan="2" align="center"> Объекты' \
           '</th><th style="border: 1px solid #333;" align="center" colspan=' + str(nb) + '> Центры кластеров </th></tr><tr>'
    for i in range(len(clasterCentres)):
        out1 += '<td align="center" style="border: 1px solid #333;">' + str(clasterCentres[i]) + '</td>'
    for i in range(len(affilationDegrees)):
        out1 += '<tr><td align="center" style="border: 1px solid #333;">' + str(int(objects[i])) + '</td>'
        for j in range(len(affilationDegrees[0])):
            out1 += '<td align="right" style="border: 1px solid #333;">' + str(affilationDegrees[i][j]) + '</td>'
        out1 += '</tr>'
    out1 += '</table>'

    # iterations
    iterations = round((len(objects) + nb + 1) / 3)
    for iteration in range(iterations):

        # claster centers
        for i in range(len(affilationDegrees[0])):
            summU = 0
            summUX = 0
            for j in range(len(affilationDegrees)):
                summU += affilationDegrees[j][i] ** m
            for j in range(len(affilationDegrees)):
                summUX += (affilationDegrees[j][i] ** m) * objects[j]
            clasterCentres[i] = summUX / summU
        # output
        if iteration == 0:
            out2 = '<table style="min-width: 20%;"><tr><th style="border: 1px solid #333;" align="center"' \
                    'colspan=' + str(nb) + '> Центры кластеров </th></tr><tr>'
            for i in range(len(clasterCentres)):
                out2 += '<td align="center" style="border: 1px solid #333; background: ' + str(highlighters[i]) + '">' + str(round(clasterCentres[i], 2)) + '</td>'
            out2 += '</table>'

        # degrees of affilation
        for i in range(len(lambdaCenters)):
            for j in range(len(lambdaCenters[0])):
                lambdaCenters[i][j] = abs(objects[i] - clasterCentres[j])
        # output
        if iteration == 0:
            out3 = '<table style="min-width: 20%;"><tr><th style="border: 1px solid #333;" align="center"' \
                   'colspan=' + str(nb) + '> Лямбда центры (Cl) </th></tr>'
            for i in range(len(lambdaCenters)):
                out3 += '<tr>'
                for j in range(len(lambdaCenters[0])):
                    out3 += '<td align="center" style="border: 1px solid #333; background: ' + str(highlighters[j]) + '"' \
                            '>' + str(round(lambdaCenters[i][j], 2)) + '</td>'
                out3 += '</tr>'
            out3 += '</table>'

        for k in range(len(lambdaDegrees)):
            for i in range(len(lambdaDegrees[0])):
                for j in range(len(lambdaDegrees[0][0])):
                    lambdaDegrees[k][i][j] = (abs(objects[i] - clasterCentres[j]) / lambdaCenters[i][k]) ** 3.33
        # output
        if iteration == 0:
            out4 = ''
            for k in range(len(lambdaDegrees)):
                out4 += '<div><table style="min-width: 20%;"><tr><th style="border: 1px solid #333;" align="center"' \
                       'colspan=' + str(nb) + '> Лямбда принадлежности (C' + str(k) + ') </th></tr>'
                for i in range(len(lambdaDegrees[0])):
                    out4 += '<tr>'
                    for j in range(len(lambdaDegrees[0][0])):
                        out4 += '<td align="center" style="border: 1px solid #333; background: ' + str(highlighters[k]) + '"' \
                                '>' + str(round(lambdaDegrees[k][i][j], 2)) + '</td>'
                    out4 += '</tr>'
                out4 += '</table></div>'

            # result
        for j in range(len(lambdaDegrees[0][0])):
            for i in range(len(lambdaDegrees[0])):
                summR = 0
                for k in range(len(lambdaDegrees)):
                    summR += lambdaDegrees[k][i][j]
                result[i][j] = 1 / summR
        # output
        if iteration == 0:
            out5 = '<table style="min-width: 30%;"><tr><th style="border: 1px solid #333;" rowspan="2" align="center"> Объекты' \
                   '</th><th style="border: 1px solid #333;" align="center" colspan=' + str(nb) + '> Центры кластеров </th></tr><tr>'
            for i in range(len(clasterCentres)):
                out5 += '<td align="center" style="border: 1px solid #333; background: ' + str(highlighters[i]) + '">' + str(round(clasterCentres[i], 2)) + '</td>'
            for i in range(len(result)):
                out5 += '<tr><td align="center" style="border: 1px solid #333;">' + str(int(objects[i])) + '</td>'
                for j in range(len(result[0])):
                    out5 += '<td align="right" style="border: 1px solid #333;background: ' + str(highlighters[j]) + '">' + str(round(result[i][j], 2)) + '</td>'
                out5 += '</tr>'
            out5 += '</table>'

        affilationDegrees = result
        #output
        if iteration != 0:
            out6 += '<div style="min-width: 30%;"><h4>Итерация ' + str(iteration + 1) + '</h4><table style="min-width: 100%;"><tr><th style="border: 1px solid #333;" rowspan="2" align="center"> Объекты' \
                   '</th><th style="border: 1px solid #333;" align="center" colspan=' + str(nb) + '> Центры кластеров </th></tr><tr>'
            for i in range(len(clasterCentres)):
                out6 += '<td align="center" style="border: 1px solid #333;">' + str(round(clasterCentres[i], 2)) + '</td>'
            for i in range(len(affilationDegrees)):
                out6 += '<tr><td align="center" style="border: 1px solid #333;">' + str(int(objects[i])) + '</td>'
                for j in range(len(affilationDegrees[0])):
                    out6 += '<td align="right" style="border: 1px solid #333;">' + str(round(affilationDegrees[i][j], 2)) + '</td>'
                out6 += '</tr>'
            out6 += '</table></div>'

    # tests
    # '<p>' + str(objects) + '</p>' \
    # '<p>' + str(clasterCentres) + '</p>' \
    # '<p>' + str(affilationDegrees) + '</p>' \
    # '<p>' + str(lambdaCenters) + '</p>' \
    # '<p>' + str(lambdaDegrees) + '</p>' \
    # '<p>' + str(result) + '</p>' \
    # '<p>' + str(iterations) + '</p>'

    # linguistic axe
    step = 155
    plt.title("Возраст")
    for i in range(len(linqAxe)):
        for j in range(40):
            plt.plot([linqAxe[i][0], linqAxe[i][1]], [j, j], color=plotColors[i])
        plt.text(linqAxe[i][0], step, linqAxe[i][2])
        plt.plot([linqAxe[i][0], linqAxe[i][0]], [0, step - 5], color=plotColors[i])
        step = step - 25
    y_ticks = [0, 40, 100, 160, 220, 280]
    y_labels = [' ', ' ', ' ', ' ', ' ', ' ']
    plt.yticks(ticks=y_ticks, labels=y_labels)
    plt.savefig('static/images/linqaxe.png')

    # determinating clasters to linq params
    out7 = '<div>'
    for i in range(len(clasterCentres)):
        if clasterCentres[i] < 20:
            out7 += '<p>Кластер ' + str(i) + ' со значением ' + str(round(clasterCentres[i], 2)) + ' определяется ' \
                    'лингвистической переменной <b>' + str(linqAxe[0][2]) + '</b></p>'
        if 20 < clasterCentres[i] < 40:
            out7 += '<p>Кластер ' + str(i) + ' со значением ' + str(round(clasterCentres[i], 2)) + ' определяется ' \
            'лингвистической переменной <b>' + str(linqAxe[1][2]) + '</b></p>'
        if 40 < clasterCentres[i] < 60:
            out7 += '<p>Кластер ' + str(i) + ' со значением ' + str(round(clasterCentres[i], 2)) + ' определяется ' \
                    'лингвистической переменной <b>' + str(linqAxe[2][2]) + '</b></p>'
        if 60 < clasterCentres[i] < 70:
            out7 += '<p>Кластер ' + str(i) + ' со значением ' + str(round(clasterCentres[i], 2)) + ' определяется ' \
                    'лингвистической переменной <b>' + str(linqAxe[3][2]) + '</b></p>'
        if 70 < clasterCentres[i] < 100:
            out7 += '<p>Кластер ' + str(i) + ' со значением ' + str(round(clasterCentres[i], 2)) + ' определяется ' \
                    'лингвистической переменной <b>' + str(linqAxe[4][2]) + '</b></p>'
    out7 += '</div>'

    # linguistic axe with args
    step1 = 220
    for i in range(len(objects)):
        plt.text(objects[i], step1, str(objects[i]))
        plt.plot([objects[i], objects[i]], [0, step1 - 5], color='#32CD32')
        step1 = step1 - 15
    y_ticks = [0, 40, 100, 160, 220, 280]
    y_labels = [' ', ' ', ' ', ' ', ' ', ' ']
    plt.yticks(ticks=y_ticks, labels=y_labels)
    plt.savefig('static/images/linqaxewithargs.png')

    # output
    return '<body style="background: #B0C4DE;"><div style="margin: 2% 10% 3% 10%; background: #FFFFFF; padding: 1% 5%">' \
           '<h1 align="center">Нечёткая кластеризация объектов</h1><h2>Кластеризация людей по возрасту</h2>' \
           '<p>Перед началом работы определимся с поверочной шкалой лингвистических переменных возраста.' \
           'Представим её в виде столбчатой диаграммы.</p>' \
           '<h2 align="center" style="margin: 0;"><img src="' + os.path.join(app.config['UPLOAD_FOLDER'],'linqaxe.png') + '"/></h2>' \
           '<p>Теперь перейдём непосредственно к процессу кластеризации</p><h2>Процесс кластеризации</h2><h4>Данные от пользователя:</h4>' \
           '<p>Количество кластеров: ' + str(nb) + '</p><p>Список возростов (объектов) для кластеризации: ' + str(data["ages"]) + '</p>' \
           '<h3>Инициализация</h3><p>Построим таблицу, где в левой шапке у нас будут значения объектов, а в верхней значения ' \
           'центров кластеров. Тогда на пересечении у нас будут находиться степени принадлежностей объектов к кластерам.' \
           'На этапе инициализации степени принадлежности и центры кластеров заполним случайно. </p>' + out1 + '<p>Переходим к выполнению алгоритма</p>' \
           '<h3 align="center">Итерация 1</h3><p>Сначала найдём центры кластеров по имеющимся степеням принадлежности:</p>' + out2 + '' \
           '<p>Теперь через значения объектов и центров кластеров построим матрицу лямбда центров: </p>' + out3 + '<p>Теперь ' \
           'для каждого столбца данной матрицы построим матрицы лямбда принадлежностей</p><div style="display: flex; ' \
           'justify-content: space-around;">' + out4 + '</div><p>По обратной сумме строк данных матриц найдём новые ' \
           'степени принадлежности объектов к кластерам</p><h6 align="center">' + out5 + '</h6><p>Следующие итерации ' \
           'выполняются анологичным образом:</p><div style="display: flex; justify-content: space-around;">' + out6 + '</div>' \
           '<p>По последней таблице степеней принадлежности мы можем однозначно определить каждый объект к одному из кластеров, ' \
           'а значит процесс кластеризации закончен. Переходим к определению кластеров на лингвистической шкале</p>' \
           '<h2>Назначение кластерам лингвистических переменных</h2>' + out7 + '<p>Теперь остаётся только нанести ' \
           'объекты на поверочную шкалу и проверить результат выполнения алгоритма</p>' \
           '<h2>Нанесение объектов на лингвистическую шкалу</h2>' \
           '<h6 align="center" style="margin: 0;"><img src="' + os.path.join(app.config['UPLOAD_FOLDER'], 'linqaxewithargs.png') + '"/></h6>' \
           '<h3>Вывод</h3><p style="margin-bottom: 10%;">Таким образом, путём нечёткой кластеризации объектов мы получили достаточную точность их распределения.</p></body>'


@app.route("/LogicOutInput")
def logicoutinput():
    return '<body style="background: #B0C4DE;"><form Action="/LogicOut" Method="get">' \
           '<div style="margin: 2% 20% 5% 20%; background: #FFFFFF; padding: 1% 5%">' \
           '<h2 align="center">Нечёткий логический вывод</h2><h3>Объём топлива от скорости и расстояния</h3>' \
           '<p>Введите скорость передвижения (от 20 до 190 км/ч) ' \
           '<input style="max-width: 5%;" type=number name=speed value=64 /> км/ч</p>' \
           '<p>Введите расстояние (от 100 до 800 км) ' \
           '<input style="max-width: 8%; margin-bottom: 5%" type=number name=distance value="237" /> км</p></div>' \
           '<h1 align="center" style="margin: 3% 45% 10% 45%"><input type=submit value="Далее" /></h1></form></body>'


@app.route("/LogicOut")
def logicout():
    #initialization
    plot_colors = ['#DC143C', '#4682B4', '#008000']

    speed_func = [[20, 20, 40, 60], [50, 80, 80, 110], [100, 120, 190, 190]]
    dist_func = [[100, 100, 200, 300], [300, 450, 450, 600], [600, 700, 800, 800]]
    volume_func = [[10, 10, 10, 20], [15, 20, 20, 25], [25, 30, 30, 45], [45, 50, 60, 60]]

    speed_affiliation = [0, 0, 0]
    dist_affiliation = [0, 0, 0]

    #rules
    #speed 0 - низкая, 1 - средняя, 2 - высокая
    #dist 0 - маленькое, 1 - среднее, 2 - большое
    #volume 0 - очень маленький, 1 - маленький, 2 - средний, 3 - большой
    rules = [[2, 0, 0],
             [1, 0, 0],
             [0, 0, 1],
             [2, 1, 1],
             [1, 1, 2],
             [0, 1, 2],
             [2, 2, 2],
             [1, 2, 3],
             [0, 2, 3]]

    result = [[0 for i in range(5)] for i in range(9)]
    agr_result = [0, 0]
    linq_result = ''
    volume_result = 0

    # getting params
    data = request.args
    speed = int(data['speed'])
    dist = int(data['distance'])

    #drawing
    drawing_arr = [speed_func, dist_func, volume_func]
    for i in range(len(drawing_arr)):
        for j in range(len(drawing_arr[i])):
            plt.plot([drawing_arr[i][j][0], drawing_arr[i][j][1]], [0, 1], color=plot_colors[i])
            plt.plot([drawing_arr[i][j][1], drawing_arr[i][j][2]], [1, 1], color=plot_colors[i])
            plt.plot([drawing_arr[i][j][2], drawing_arr[i][j][3]], [1, 0], color=plot_colors[i])
        plt.savefig('static/images/logicout' + str(i) + '.png')
        plt.close()

    #output
    out0 = '<table align="center" style="min-width: 40%;"><tr><th  style="border: 1px solid #333;">Скорость</th>' \
           '<th  style="border: 1px solid #333;">Расстояние</th><th style="border: 1px solid #333;">Объём</th></tr>'
    for i in range(len(rules)):
        out0 += '<tr>'
        for j in range(len(rules[0])):
            if j == 0:
                if rules[i][j] == 0:
                    out0 += '<td  style="border: 1px solid #333;">Низкая</td>'
                if rules[i][j] == 1:
                    out0 += '<td  style="border: 1px solid #333;">Средняя</td>'
                if rules[i][j] == 2:
                    out0 += '<td  style="border: 1px solid #333;">Высокая</td>'
            if j == 1:
                if rules[i][j] == 0:
                    out0 += '<td  style="border: 1px solid #333;">Маленькое</td>'
                if rules[i][j] == 1:
                    out0 += '<td  style="border: 1px solid #333;">Среднее</td>'
                if rules[i][j] == 2:
                    out0 += '<td  style="border: 1px solid #333;">Большое</td>'
            if j == 2:
                if rules[i][j] == 0:
                    out0 += '<td  style="border: 1px solid #333;">Очень маленький</td>'
                if rules[i][j] == 1:
                    out0 += '<td  style="border: 1px solid #333;">Маленький</td>'
                if rules[i][j] == 2:
                    out0 += '<td  style="border: 1px solid #333;">Средний</td>'
                if rules[i][j] == 3:
                    out0 += '<td  style="border: 1px solid #333;">Большой</td>'
        out0 += '</tr>'
    out0 += '</table>'

    #execution
    for i in range(len(speed_func)):
        if speed < speed_func[i][0]:
            speed_affiliation[i] = 0
        if speed > speed_func[i][3]:
            speed_affiliation[i] = 0
        if speed_func[i][1] <= speed <= speed_func[i][2]:
            speed_affiliation[i] = 1
        if speed_func[i][0] <= speed < speed_func[i][1]:
            speed_affiliation[i] = (speed - speed_func[i][0]) / (speed_func[i][1] - speed_func[i][0])
        if speed_func[i][2] < speed <= speed_func[i][3]:
            speed_affiliation[i] = (speed - speed_func[i][3]) / (speed_func[i][2] - speed_func[i][3])

    for i in range(len(dist_func)):
        if dist < dist_func[i][0]:
            dist_affiliation[i] = 0
        if dist > dist_func[i][3]:
            dist_affiliation[i] = 0
        if dist_func[i][1] <= dist <= dist_func[i][2]:
            dist_affiliation[i] = 1
        if dist_func[i][0] <= dist < dist_func[i][1]:
            dist_affiliation[i] = (dist - dist_func[i][0]) / (dist_func[i][1] - dist_func[i][0])
        if dist_func[i][2] < dist <= dist_func[i][3]:
            dist_affiliation[i] = (dist - dist_func[i][3]) / (dist_func[i][2] - dist_func[i][3])

    for i in range(len(result)):
        result[i][0] = i + 1
        for j in range(len(result[0])):
            if j == 1:
                result[i][j] = speed_affiliation[rules[i][0]]
            if j == 2:
                result[i][j] = dist_affiliation[rules[i][1]]
            if j == 3:
                result[i][j] = min(result[i][1], result[i][2])
            if j == 4:
                result[i][j] = result[i][j-1]
    maximum = 0
    for i in range(len(result)):
        if result[i][4] > maximum:
            maximum = result[i][4]
            agr_result[0] = i
            agr_result[1] = maximum

    if rules[agr_result[0]][2] == 0:
        linq_result = 'очень маленький'
    if rules[agr_result[0]][2] == 1:
        linq_result = 'маленький'
    if rules[agr_result[0]][2] == 2:
        linq_result = 'средний'
    if rules[agr_result[0]][2] == 3:
        linq_result = 'большой'

    volume_result = round((volume_func[rules[agr_result[0]][2]][1] + ((volume_func[rules[agr_result[0]][2]][1] + volume_func[rules[agr_result[0]][2]][2]) / 2) * agr_result[1]), 2)

    #output
    out1 = '<table align="center" style="max-width: 40%;"><tr><th  style="border: 1px solid #333;">№</th>' \
           '<th  style="border: 1px solid #333;">Степень принадлежности для значения "Скорости"</th>' \
           '<th style="border: 1px solid #333;">Степень принадлежности для значения "Расстояния"</th>' \
           '<th style="border: 1px solid #333;">Нечёткое И</th>' \
           '<th style="border: 1px solid #333;">Нечёткая импликация</th></tr>'
    for i in range(len(result)):
        out1 += '<tr>'
        for j in range(len(result[0])):
            out1 += '<td  style="border: 1px solid #333;">' + str(round(result[i][j], 3)) + '</td>'
        out1 += '</tr>'
    out1 += '<tr><td style="border: 1px solid #333;" colspan="4">Результат агрегации</td>' \
            '<td style="border: 1px solid #333;">' + str(round(agr_result[1], 3)) + '</td></tr>' \
            '<tr><td style="border: 1px solid #333;" colspan="4">Полученное нечёткое значение объёма</td>' \
            '<td style="border: 1px solid #333;">' + linq_result + '</td></tr>' \
            '<tr><td style="border: 1px solid #333;" colspan="4">Полученное чёткое значение объёма</td>' \
            '<td style="border: 1px solid #333;">' + str(volume_result) + ' л</td></tr></table>'

    return '<body style="background: #B0C4DE;"><div style="margin: 2% 10% 3% 10%; background: #FFFFFF; padding: 1% 5%">' \
           '<h1 align="center">Нечёткий логический вывод</h1><h2>Зависимость объёма топлива от скорости и расстояния</h2>' \
           '<p>Заданная скорость: ' + str(speed) + ' км/ч </p><p>Заданное расстояние: ' + str(dist) + ' км </p>' \
           '<h3>Графики функций принадлежности</h3><div style="display: flex; justify-content: space-around;">' \
           '<div><p style="margin-bottom: 0;">Скорости, км/ч</p><img style="max-width: 100%;" src="' + os.path.join(app.config['UPLOAD_FOLDER'], 'logicout0.png') + '"/></div>' \
           '<div><p style="margin-bottom: 0;">Расстояния, км</p><img style="max-width: 100%;" src="' + os.path.join(app.config['UPLOAD_FOLDER'], 'logicout1.png') + '"/></div>' \
           '<div><p style="margin-bottom: 0;">Объёма, л</p><img style="max-width: 100%;" src="' + os.path.join(app.config['UPLOAD_FOLDER'], 'logicout2.png') + '"/></div></div>' \
           '<h3>Таблица правил</h3>' + out0 + '<h3>Таблица результатов</h3>' + out1 + '<h3>Вывод</h3><p style="margin-bottom: 10%;">' \
           'Чтобы преодолеть заданное расстояние с заданной скоростью необходимо заправить Mitsubishi lancer X на' + str(volume_result) + ' л бензина АИ-95</p></body>'


@app.route("/TextAnalysis")
def textAnalysis():
    f = open(os.path.join(app.config['FILE_FOLDER'], 'ForAnalys.txt'), mode='r', encoding='UTF-8')
    text = f.read()

    # cleaning
    text = re.sub(r'[^\w\s]+|\d+|([a-zA-Z])\w+|([A-Z])+', r'', text).strip()
    text = text.lower()

    # splitting
    word_array = text.split()
    before_normalisation_words = text.split()

    # removing stop words
    for f in range(5):
        for word in word_array:
            if word in stopwords.words('russian'):
                word_array.remove(word)
                before_normalisation_words.remove(word)

    # normalisaton
    morph = pymorphy2.MorphAnalyzer()
    for i in range(len(word_array)):
        p = morph.parse(word_array[i])[0]
        word_array[i] = p.normal_form

    # counting frequency
    frequency = {}
    for word in word_array:
        count = frequency.get(word, 0)
        frequency[word] = count + 1

    #counting TF*IDF
    ru_corpora = pd.read_csv("D:\\ULSTU\\AIM\\Project\\static\\sets\\ru_corpora.csv", sep=';', encoding='utf-8')
    TF_IDF = {}
    for word in word_array:
        # TF
        tf = frequency[word] / len(word_array)

        #IDF
        if ru_corpora[ru_corpora['lex'] == word].size != 0:
            idf = math.log2((2000000000 - ru_corpora[ru_corpora['lex'] == word]['ipm']) / ru_corpora[ru_corpora['lex'] == word]['ipm'])
        else:
            idf = math.log2((2000000000 - 200) / 200)
        TF_IDF[word] = tf*idf

    #Max
    maximum = 0
    word_key = ""
    for word in TF_IDF:
        if TF_IDF[word] > maximum:
            maximum = TF_IDF[word]
            word_key = word

    #Max doubleword
    filter_arr = ['структуры', 'конечное', 'вкладке', 'поиск']
    doub_words = [['', '']]
    doub_words.remove(doub_words[0])
    for i in range(len(before_normalisation_words)):
        if before_normalisation_words[i] == word_key:
            if before_normalisation_words[i - 1] not in filter_arr:
                doub_words.append([before_normalisation_words[i - 1], before_normalisation_words[i]])
            if before_normalisation_words[i + 1] not in filter_arr:
                doub_words.append([before_normalisation_words[i], before_normalisation_words[i + 1]])
    doub_words.remove(doub_words[3])

    # past verbs
    verbs = []
    for word in before_normalisation_words:
        word_t = morph.parse(word)[0].tag.POS
        if word_t in {'VERB'}:
            verbs.append(word)
    past_verbs = []
    for word in verbs:
        sub_word = word[-2::]
        if sub_word in ['ла', 'ли', 'ло']:
            past_verbs.append(word)
        else:
            sub_word = word[-1::]
            if sub_word == 'л':
                past_verbs.append(word)

    # output
    out1 = ""
    for i in range(len(doub_words)):
        out1 += doub_words[i][0] + ' ' + doub_words[i][1] + ', '

    out2 = ""
    for word in past_verbs:
        out2 += word + ', '

    return '<body style="background: #B0C4DE;"><div style="margin: 2% 10% 3% 10%; background: #FFFFFF; padding: 1% 5%">' \
           '<h1 align="center">Анализ текста</h1><h3>Задание 1:</h3><p>Выявите наиболее статистически значимые ' \
           'двусловия по методу TF×IDF</p><h3>Решение:</h3><p>' + out1 + '</p><h3>Задание 2:</h3><p>Программно ' \
           'определите количество глаголов, стоящих в прошедшем времени</p><h3>Решение:</h3><p>' + out2 + '</p>' \
           '<p>Количество глаголов прошедшего времени в тексте: ' + str(len(past_verbs)) + '</p></body>'


def activation(x):
    return 1 / (1 + np.exp(-x))


def sigma_derivative(x):
    return x * (1 - x)


@app.route("/NeuralNet")
def neuralnet():
    np.random.seed(1)
    # params -------------------------------------------------------
    # input
    obj_dim = 3
    #body
    h1_dim = 4
    h2_dim = 5
    #output
    out_dim = 1
    # x = [count_object, count_signs]
    x = np.array([[0, 1, 0],
                  [1, 0, 0],
                  [1, 1, 1]])
    # y = [count_objects, 1]
    y = np.array([[1],
                  [1],
                  [0]])
    # result
    global z

    # weight matrices [left_neuron, right_neuron]
    w1 = 2 * np.random.random((obj_dim, h1_dim)) - 1
    w2 = 2 * np.random.random((h1_dim, h2_dim)) - 1
    w3 = 2 * np.random.random((h2_dim, out_dim)) - 1
    # processing
    iterations_num = 100000
    antgr_speed = 1.5

    # iterations -------------------------------------------------------
    for i in range(iterations_num):
        # forward propagation
            # First level - X
            # Second level
        t1 = x @ w1
        h1 = activation(t1)
            # Third level
        t2 = h1 @ w2
        h2 = activation(t2)
            # Forth (last) level (without activation)
        z = h2 @ w3

        # backward propagation
            # Clear z error
        e_full = y - z
            # Forth level local error gradient
        sigma_z = e_full * sigma_derivative(z)
            # Clear h2 error
        e_h2 = sigma_z @ w3.T
            # Third level local error gradient
        sigma_h2 = e_h2 * sigma_derivative(h2)
            # Clear h1 error
        e_h1 = sigma_h2 @ w2.T
            # Second level local error gradient
        sigma_h1 = e_h1 * sigma_derivative(h1)
            # Update weights
        w3 += antgr_speed * h2.T @ sigma_z
        w2 += antgr_speed * h1.T @ sigma_h2
        w1 += antgr_speed * x.T @ sigma_h1

    print(z)

    # normalisation --------------------------------------------
    norm_z = z * iterations_num / 10
    for i in range(len(norm_z)):
        if norm_z[i] > 1:
            norm_z[i] = 1
        else:
            norm_z[i] = 0
    print(norm_z)

    return ""


if __name__ == "__main__":
    app.run(port=5001, debug=True)
