import os.path

from flask import Flask, request
import pandas as pd
import matplotlib.pyplot as plt
import numpy
import math

app = Flask(__name__)
picfld = os.path.join('static', 'charts')
app.config['UPLOAD_FOLDER'] = picfld


@app.route("/")
def home():
    return "<html>" \
            "<form Action='/table' Method='get' style='margin: 0 30%'>" \
            "<h5>Укажите имя файла на диске D</h5>" \
            "<input type=text name=file value='healthcare-dataset-stroke-data.csv'/>" \
            "<h5>Укажите диапазон строк</h5>" \
            "<input type=number name=from min=0 value=4 />" \
            "<input type=number name=to min=0 value=18 />" \
            "<h5>Укажите диапазон столбцов</h5>" \
            "<input type=number name=from_st min=0 value=1 />" \
            "<input type=number name=to_st min=0 value=3 />" \
            "<input type=submit value='Открыть' /></form></html>"


@app.route("/table",methods=['GET'])
def table():
        data = request.args
        global initial_df
        global added_df
        global filename
        global sized_df
        filename = "D:\\Ulstu\\МИИ\\" + data['file']
        initial_df = pd.read_csv(filename, sep=',')
        added_df = pd.read_csv(filename, sep=',')
        initial_df = initial_df.rename(columns={'age': 'возраст', 'gender': 'пол', "hypertension": 'гипертензия',
                                                "heart_disease": 'сердечные заболевания', "ever_married": 'замужество',
                                                "work_type": 'тип работы',
                                                "Residence_type": 'место жительства',
                                                "avg_glucose_level": 'средний уровень глюкозы',
                                                "bmi": 'ИМТ',
                                                "smoking_status": 'статус курильщика', "stroke": 'инсульт'})
        added_df = added_df.rename(columns={'age': 'возраст', 'gender': 'пол', "hypertension": 'гипертензия',
                                               "heart_disease": 'сердечные заболевания', "ever_married": 'замужество',
                                               "work_type": 'тип работы',
                                               "Residence_type": 'место жительства',
                                               "avg_glucose_level": 'средний уровень глюкозы',
                                               "bmi": 'ИМТ',
                                               "smoking_status": 'статус курильщика', "stroke": 'инсульт'})

        if int(data['from']) > int(data['to']):
                from_str = int(data['to'])
                to_str = int(data['from'])
        else:
                to_str = int(data['to'])
                from_str = int(data['from'])

        if int(data['from_st']) > int(data['to_st']):
                from_clm = int(data['to_st'])
                to_clm = int(data['from_st'])
        else:
                to_clm = int(data['to_st'])
                from_clm = int(data['from_st'])

        sized_df = initial_df.iloc[from_str:to_str]
        prim = initial_df.iloc[from_str:to_str, from_clm:to_clm]

        count = initial_df.isna().sum()
        countAll = initial_df.count()

        description_data = 'Описание: По данным Всемирной организации здравоохранения (ВОЗ), инсульт является второй по значимости причиной смерти в мире, на его долю приходится примерно 11% от общего числа смертей.' \
                           'Этот набор данных используется для прогнозирования вероятности инсульта у пациента на основе входных параметров, таких как пол, возраст, различные заболевания и статус курения. <br><br>' \
                           'Каждая строка в данных предоставляет актуальную информацию о пациенте. <br>' \
                           'Описание столбцов таблицы: <br>' \
                           '1) id: уникальный идентификатор <br>' \
                           '2) пол: "Мужской", "Женский" или "Другой" <br>' \
                           '3) возраст: возраст пациента <br>' \
                           '4) гипертония: 0, если у пациента нет гипертонии, 1, если у пациента гипертония <br>' \
                           '5) heart_disease: 0, если у пациента нетболезни сердца, 1, если у пациента есть заболевание сердца <br>' \
                           '6) ever_married: "Нет" или "Да" <br>'\
                           '7) work_type: "дети", "Govt_jov", "Never_worked", "Частный" или "Самозанятый" <br>' \
                           '8) Residence_type: "Сельский" или "Городской" <br>' \
                           '9) avg_glucose_level: средний уровень глюкозы в крови <br>' \
                           '10) ИМТ: индекс массы тела <br>' \
                           '11) smoking_status: "ранее курил", "никогда не курил", "курит" или "Неизвестно" * <br>' \
                           '12) инсульт: 1, если у пациента был инсульт, или 0, если нет <br>' \
                           '* Примечание: "Неизвестно" в smoking_status означает, что информация для этого пациента недоступна <br><br>' \
                           'Описание типов данных столбцов таблицы: <br>' \
                           + str(initial_df.dtypes) + "<br><br>"

        return '<h3 align="center">Набор данных для прогнозирования инсульта</h3>' \
                + '<div style="margin: 0 10%"><div align"center">' + prim.to_html() + '</div>' \
                + '<h5>Количество пустых ячеек = %s , ' % count  \
                + '<br>Количество заполненных ячеек = %s , ' % countAll \
                + '<br>Количество строк = %s , ' % len(initial_df.axes[0]) \
                + '<br>Количество столбцов = %s</h5>' % len(initial_df.axes[1]) + '<h5>%s</h5>' % description_data \
                + '<h3 align="center">Только выбранные данные</h3>' \
                + '<a style="margin: 10% 3%" href="/married/0">Married</a>' \
                + '<a style="margin: 10% 3%" href="/singe/0">Single</a>' \
                + '<a style="margin: 10% 3%" href="/man_bmi/0">Man BMI</a>' \
                + '<a style="margin: 10% 3%" href="/woman_bmi/0">Woman BMI</a>' \
                + '<h3 align="center">Глобальные данные</h3>' \
                + '<a style="margin: 10% 3%" href="/married/1">Married</a>' \
                + '<a style="margin: 10% 3%" href="/singe/1">Single</a>' \
                + '<a style="margin: 10% 3%" href="/man_bmi/1">Man BMI</a>' \
                + '<a style="margin: 10% 3%" href="/woman_bmi/1">Woman BMI</a>' \
                + '<h3 align="center">Динамика изменений при добавлении данных </h3>' \
                + '<a style="margin: 10% 3%" href="/added">Added</a>' \
                + '<h3 align="center">Парная линейная регрессия (ИМТ от возраста) </h3>' \
                + '<a style="margin: 10% 3%" href="/regression">Regression</a>' \
                + '<h3 align="center">Дерево решений (ИМТ от пола и возраста) </h3>' \
                + '<a style="margin: 10% 3%" href="/solutionTree">Solution Tree</a>' \
               + '<h3 align="center">Кластеризация </h3>' \
               + '<a style="margin: 10% 3%" href="/classterisation">Classterisation</a></div>' \
               + '<div style="min-height: 10%"></div>'


@app.route("/married/<flag>", methods=['GET'])
def married(flag):
    if flag == "1":
        df_married = pd.read_csv(filename, sep=',')
    else:
        df_married = sized_df
    return '<h2 align="center">Возраст пациентов с инсультом состоящих в браке</h2>' \
            '<div style="margin: 0 10%" align"center"><h3> Минимальный возраст пациента - ' + age_construct(df_married[(df_married['замужество'] == "Yes")]["возраст"].min()) + ', таким(и) является(ются):</h3></div>' \
            '<div style="margin: 0 10%" align"center">' + df_married[(df_married['замужество'] == "Yes") & (df_married["возраст"] == df_married[(df_married['замужество'] == "Yes")]["возраст"].min())].to_html() + '</div>' \
            '<div style="margin: 0 10%" align"center"><h3> Максимальный возраст пациента - ' + age_construct(df_married[(df_married['замужество'] == "Yes")]["возраст"].max()) + ', таким(и) является(ются):</h3></div>' \
            '<div style="margin: 0 10%" align"center">' + df_married[(df_married['замужество'] == "Yes") & (df_married["возраст"] == df_married[(df_married['замужество'] == "Yes")]["возраст"].max())].to_html() + '</div>' \
            '<div style="margin: 0 10%" align"center"><h3> Средний возраст пациентов - ' + age_construct(df_married[(df_married['замужество'] == "Yes")]["возраст"].median()) + ', количество таковых: ' \
            + str(len(df_married[(df_married['замужество'] == "Yes") & (df_married["возраст"] == df_married[(df_married['замужество'] == "Yes")]["возраст"].median())])) + '</h3></div>' \
            '<div style="margin: 0 10%" align"center"><h3> По проанализированным данным можно сделать вывод, что минимальнй возраст у пациентов в браке страдающих инсультом - ' \
            + age_construct(df_married[(df_married['замужество'] == "Yes")]["возраст"].min()) + ' максимальный - ' + age_construct(df_married[(df_married['замужество'] == "Yes")]["возраст"].max()) + ' наиболее ' \
            'подверженны инсульту люди в воздасте ' + age_construct(df_married[(df_married['замужество'] == "Yes")]["возраст"].median()) + '</h3></div>'


@app.route("/singe/<flag>", methods=['GET'])
def singe(flag):
    if flag == "1":
        df_singe = pd.read_csv(filename, sep=',')
    else:
        df_singe = sized_df
    return '<h2 align="center">Возраст пациентов с инсультом не состоящих в браке</h2>' \
           '<div style="margin: 0 10%" align"center"><h3> Минимальный возраст пациента - ' + age_construct(df_singe[(df_singe['замужество'] == "No")]["возраст"].min()) + ', таким(и) является(ются):</h3></div>' \
           '<div style="margin: 0 10%" align"center">' + df_singe[(df_singe['замужество'] == "No") & (df_singe["возраст"] == df_singe[(df_singe['замужество'] == "No")]["возраст"].min())].to_html() + '</div>' \
           '<div style="margin: 0 10%" align"center"><h3> Максимальный возраст пациента - ' + age_construct(df_singe[(df_singe['замужество'] == "No")]["возраст"].max()) + ', таким(и) является(ются):</h3></div>' \
           '<div style="margin: 0 10%" align"center">' + df_singe[(df_singe['замужество'] == "No") & (df_singe["возраст"] == df_singe[(df_singe['замужество'] == "No")]["возраст"].max())].to_html() + '</div>' \
           '<div style="margin: 0 10%" align"center"><h3> Средний возраст пациентов - ' + age_construct(df_singe[(df_singe['замужество'] == "No")]["возраст"].median()) + ', количество таковых: ' \
           + str(len(df_singe[(df_singe['замужество'] == "No") & (df_singe["возраст"] == df_singe[(df_singe['замужество'] == "No")]["возраст"].median())])) + '</h3></div>' \
           '<div style="margin: 0 10%" align"center"><h3> По проанализированным данным можно сделать вывод, что минимальнй возраст у пациентов не состоящих в браке страдающих инсультом - ' \
           + age_construct(df_singe[(df_singe['замужество'] == "No")]["возраст"].min()) + ' максимальный - ' + age_construct(df_singe[(df_singe['замужество'] == "No")]["возраст"].max()) + ' наиболее ' \
           'подверженны инсульту люди в воздасте ' + age_construct(df_singe[(df_singe['замужество'] == "No")]["возраст"].median()) + '</h3></div>'


@app.route("/man_bmi/<flag>", methods=['GET'])
def man_bmi(flag):
    if flag == "1":
        df_man_bmi = pd.read_csv(filename, sep=',')
    else:
        df_man_bmi = sized_df
    return '<h2 align="center">ИМТ пациентов мужчин с инсультом</h2>'\
           '<div style="margin: 0 10%" align"center"><h3> Минимальный ИМТ пациента - ' + str(df_man_bmi[(df_man_bmi['пол'] == "Male") & (df_man_bmi['ИМТ'] != "N/A")]['ИМТ'].min()) + 'кг/м^2, таким(и) является(ются):</h3></div>' \
           '<div style="margin: 0 10%" align"center">' + df_man_bmi[(df_man_bmi['пол'] == "Male") & (df_man_bmi['ИМТ'] == df_man_bmi[(df_man_bmi['пол'] == "Male") & (df_man_bmi['ИМТ'] != "N/A")]['ИМТ'].min())].to_html() + '</div>' \
           '<div style="margin: 0 10%" align"center"><h3> Максимальный ИМТ пациента - ' + str(df_man_bmi[(df_man_bmi['пол'] == "Male") & (df_man_bmi['ИМТ'] != "N/A")]['ИМТ'].max()) + 'кг/м^2, таким(и) является(ются):</h3></div>' \
           '<div style="margin: 0 10%" align"center">' + df_man_bmi[(df_man_bmi['пол'] == "Male") & (df_man_bmi['ИМТ'] == df_man_bmi[(df_man_bmi['пол'] == "Male") & (df_man_bmi['ИМТ'] != "N/A")]['ИМТ'].max())].to_html() + '</div>' \
           '<div style="margin: 0 10%" align"center"><h3> Средний ИМТ пациента - ' + str(df_man_bmi[(df_man_bmi['пол'] == "Male") & (df_man_bmi['ИМТ'] != "N/A")]['ИМТ'].median()) + 'кг/м^2, количество таковых: ' \
           + str(len(df_man_bmi[(df_man_bmi['пол'] == "Male") & (df_man_bmi['ИМТ'] == df_man_bmi[(df_man_bmi['пол'] == "Male") & (df_man_bmi['ИМТ'] != "N/A")]['ИМТ'].median())])) + '</h3></div>' \
           '<div style="margin: 0 10%" align"center"><h3> По проанализированным данным можно сделать вывод, что минимальнй ИМТ у мужчин страдающих инсультом - ' \
           + str(df_man_bmi[(df_man_bmi['пол'] == "Male") & (df_man_bmi['ИМТ'] != "N/A")]['ИМТ'].min()) + 'кг/м^2 максимальный - ' + str(df_man_bmi[(df_man_bmi['пол'] == "Male") & (df_man_bmi['ИМТ'] != "N/A")]['ИМТ'].max()) + 'кг/м^2 наиболее ' \
           'подверженны инсульту мужчины с ИМТ ' + str(df_man_bmi[(df_man_bmi['пол'] == "Male") & (df_man_bmi['ИМТ'] != "N/A")]['ИМТ'].median()) + 'кг/м^2 </h3></div>'


@app.route("/woman_bmi/<flag>", methods=['GET'])
def woman_bmi(flag):
    if flag == "1":
        df_woman_bmi = pd.read_csv(filename, sep=',')
    else:
        df_woman_bmi = sized_df
    return '<h2 align="center">ИМТ пациентов женщин с инсультом</h2>'\
           '<div style="margin: 0 10%" align"center"><h3> Минимальный ИМТ пациента - ' + str(df_woman_bmi[(df_woman_bmi['пол'] == "Female") & (df_woman_bmi['ИМТ'] != "N/A")]['ИМТ'].min()) + 'кг/м^2, таким(и) является(ются):</h3></div>' \
           '<div style="margin: 0 10%" align"center">' + df_woman_bmi[(df_woman_bmi['пол'] == "Female") & (df_woman_bmi['ИМТ'] == df_woman_bmi[(df_woman_bmi['пол'] == "Female") & (df_woman_bmi['ИМТ'] != "N/A")]['ИМТ'].min())].to_html() + '</div>' \
           '<div style="margin: 0 10%" align"center"><h3> Максимальный ИМТ пациента - ' + str(df_woman_bmi[(df_woman_bmi['пол'] == "Female") & (df_woman_bmi['ИМТ'] != "N/A")]['ИМТ'].max()) + 'кг/м^2, таким(и) является(ются):</h3></div>' \
           '<div style="margin: 0 10%" align"center">' + df_woman_bmi[(df_woman_bmi['пол'] == "Female") & (df_woman_bmi['ИМТ'] == df_woman_bmi[(df_woman_bmi['пол'] == "Female") & (df_woman_bmi['ИМТ'] != "N/A")]['ИМТ'].max())].to_html() + '</div>' \
           '<div style="margin: 0 10%" align"center"><h3> Средний ИМТ пациента - ' + str(df_woman_bmi[(df_woman_bmi['пол'] == "Female") & (df_woman_bmi['ИМТ'] != "N/A")]['ИМТ'].median()) + 'кг/м^2, количество таковых: ' \
           + str(len(df_woman_bmi[(df_woman_bmi['пол'] == "Female") & (df_woman_bmi['ИМТ'] == df_woman_bmi[(df_woman_bmi['пол'] == "Female") & (df_woman_bmi['ИМТ'] != "N/A")]['ИМТ'].max())])) + '</h3></div>' \
           '<div style="margin: 0 10%" align"center"><h3> По проанализированным данным можно сделать вывод, что минимальнй ИМТ у женщин страдающих инсультом - ' \
           + str(df_woman_bmi[(df_woman_bmi['пол'] == "Female") & (df_woman_bmi['ИМТ'] != "N/A")]['ИМТ'].min()) + 'кг/м^2 максимальный - ' + str(df_woman_bmi[(df_woman_bmi['пол'] == "Female") & (df_woman_bmi['ИМТ'] != "N/A")]['ИМТ'].max()) + 'кг/м^2 наиболее ' \
           'подверженны инсульту женщины с ИМТ ' + str(df_woman_bmi[(df_woman_bmi['пол'] == "Female") & (df_woman_bmi['ИМТ'] != "N/A")]['ИМТ'].median()) + 'кг/м^2 </h3></div>'


@app.route("/added", methods=['GET'])
def added():
    for i in range(int(len(initial_df) / 10) - 1):
        added_df.loc[len(initial_df) + 1 + i] = [90000 + i, list(initial_df['пол'].mode())[0], initial_df['возраст'].median(),
                                                 int(initial_df['гипертензия'].median()), int(initial_df['сердечные заболевания'].median()),
                                                 list(initial_df['замужество'].mode())[0], list(initial_df['тип работы'].mode())[0],
                                                 list(initial_df['место жительства'].mode())[0], initial_df['средний уровень глюкозы'].median(),
                                                 initial_df['ИМТ'].median(), list(initial_df['статус курильщика'].mode())[0],
                                                 int(initial_df['инсульт'].median())]

    married_s_prev = initial_df[initial_df['замужество'] == "Yes"]['возраст']
    married_s_new = added_df[(added_df['замужество'] == "Yes")]['возраст']
    married = pd.DataFrame(dict(s1=married_s_prev, s2=married_s_new))
    married = married.rename(columns={'s1': 'Возраст без доп данных', 's2': 'Возраст с доп данными'})
    married.plot.kde()
    plt.savefig('static/charts/married.png')

    single_s_prev = initial_df[(initial_df['замужество'] == "No")]['возраст']
    single_s_new = added_df[(added_df['замужество'] == "No")]['возраст']
    single = pd.DataFrame(dict(s1=single_s_prev, s2=single_s_new))
    single = single.rename(columns={'s1': 'Возраст без доп данных', 's2': 'Возраст с доп данными'})
    single.plot.hist(stacked=True, bins=20)
    plt.savefig('static/charts/single.png')

    men_s_prew = initial_df[(initial_df['пол'] == "Male")]["ИМТ"]
    men_s_new = added_df[(added_df['пол'] == "Male")]["ИМТ"]
    men = pd.DataFrame(dict(s1=men_s_prew, s2=men_s_new))
    men = men.rename(columns={'s1': 'ИМТ без доп данных', 's2': 'ИМТ с доп данными'})
    men.plot.scatter(x="ИМТ без доп данных", y="ИМТ с доп данными")
    plt.savefig('static/charts/men.png')

    women_s_prew = initial_df[(initial_df['пол'] == "Female")]["ИМТ"]
    women_s_new = added_df[(added_df['пол'] == "Female")]["ИМТ"]
    women = pd.DataFrame(dict(s1=women_s_prew, s2=women_s_new))
    women = women.rename(columns={'s1': 'ИМТ без доп данных', 's2': 'ИМТ с доп данными'})
    women.plot.kde()
    plt.savefig('static/charts/women.png')

    return '<h2 align="center"> Графики до добавления усреднённых данных и после</h2>' \
           + '<div style="margin: 0 10%"><a href="/seeAll">To see all data set with added values click here</a>' \
           + '<div style="display: flex; margin: 7% 0 0 0;"> <div> <h3 style="margin: 0">Возраст пациентов с инсультом состоящих в браке</h3>' \
           + '<img src="' + os.path.join(app.config['UPLOAD_FOLDER'], 'married.png') + '"/> </div>' \
           + '<div> <h3 style="margin: 0">Возраст пациентов с инсультом не состоящих в браке</h3>' \
           + '<img src="' + os.path.join(app.config['UPLOAD_FOLDER'], 'single.png') + '"/> </div></div>' \
           + '<div style="display: flex; margin: 5% 0 0 0;"> <div> <h3 style="margin: 0">ИМТ пациентов мужчин с инсультом</h3>' \
           + '<img src="' + os.path.join(app.config['UPLOAD_FOLDER'], 'men.png') + '"/> </div>' \
           + '<div> <h3 style="margin: 0">ИМТ пациентов женщин с инсультом</h3>' \
           + '<img src="' + os.path.join(app.config['UPLOAD_FOLDER'], 'women.png') + '"/> </div></div></div>'


@app.route("/seeAll", methods=['GET'])
def seeall():
    return '<div style="margin: 0 10%"> <h2>Изначальное кол-во элементов: ' + str(len(initial_df)) +' Текущее кол-во элементов: ' + str(len(added_df)) + '</h2>' \
            + added_df.to_html() + '</div>'


@app.route("/regression", methods=["GET"])
def regression():
    reg_s = initial_df.iloc[0:int(len(initial_df) * 0.99)][["возраст", "ИМТ"]]
    reg_s = reg_s[reg_s["ИМТ"].notna()]

    x = numpy.array(reg_s["возраст"].tolist())
    y = numpy.array(reg_s["ИМТ"].tolist())
    sumY = sum(y)
    sumX = sum(x)
    sumXY = sum(x*y)
    sumXX = sum(x*x)
    n = int(len(initial_df) * 0.99)
    b1 = (sumXY - (sumY*sumX)/n)/(sumXX-sumX*sumX/n)
    b0 = (sumY-b1*sumX)/n

    reg_s.plot.scatter(x="возраст", y="ИМТ", s=3, color='#00b003')
    plt.plot([0, 82], [(b1 * 0 + b0), (b1 * 82 + b0)], color='#ff0000')
    plt.savefig('static/charts/reg.png')

    check_s = initial_df.iloc[int(len(initial_df) * 0.99):int(len(initial_df))][["возраст", "ИМТ"]]
    check_s = check_s[check_s["ИМТ"].notna()]

    check_s.plot.scatter(x="возраст", y="ИМТ", s=50, color="#ff6208")
    plt.plot([0, 82], [(b1 * 0 + b0), (b1 * 82 + b0)], color='#000000')
    plt.savefig('static/charts/check.png')

    midRealY = sumY / n
    midRealY = midRealY*midRealY
    predY = x
    for i in predY:
        i = b1 * i + b0
    sumYY = sum(y*y)
    sumPYPY = sum(predY*predY)

    kphDtm = ((sumYY - n*midRealY)/(sumPYPY - n*midRealY)) ** 0.5
    return '<h2 align="center">Парная линейная регрессия (ИМТ от возраста)</h2>' \
           '<div style="margin: 0 10%"><h4>Рассчёт парной линейной регрессии на 90% данных:</h4>' \
           '<h4>Значение b0: ' + str(b0) + '</h4>' \
           '<h4>Значение b1: ' + str(b1) + '</h4>' \
           '<h4>Тогда, формула парной линейной регрессии y=b1x + b0 принимает вид: y=' + str(b1) + 'x + ' + str(b0) + '</h4>' \
           '<h3 align="center">График линейной функцие на диаграмме распределения для 99% записей</h3>' \
           '<img src="' + os.path.join(app.config['UPLOAD_FOLDER'], 'reg.png') + '"/>' \
           '<h3 align="center">График линейной функцие на диаграмме распределения для 1% записей</h3>' \
           '<img src="' + os.path.join(app.config['UPLOAD_FOLDER'], 'check.png') + '"/>' \
           '<h4>Значение коэфициента детерминации: ' + str(kphDtm) + '</h4>' \
           '<h4 style="margin: 0 0 10% 0">Исходя из этого, точность полученной модели парной линейной регрессии  ' + str(round(kphDtm*100)) + \
           '%, что соответствует точности ниже среднего. Однако, по графику распределения, ИМТ теснее связан с возрастом только' +\
           ' в начале жизни человека, а дальше становится очень хаотичен, точность построенной модели достаточная.</h4></div>'


@app.route("/solutionTree", methods=["GET"])
def solutionTree():
    tree_df = initial_df[initial_df["ИМТ"].notna()]
    tree_df = tree_df.reset_index()
    tree_rows = tree_df.iloc[0:25][["ИМТ", "возраст", "пол"]]
    checking_rows = tree_df.iloc[30:35][["ИМТ", "возраст", "пол"]]
    min_age = int(tree_rows["возраст"].min() / 10) * 10
    max_age = int(tree_rows["возраст"].max() / 10) * 10 + 10
    tmp_age = min_age

    age_to_show = ""
    for i in range(int((max_age - min_age) / 10)):
        age_to_show += str(tmp_age) + "-" + str(tmp_age + 10) + ", "
        tmp_age += 10

    tmp_age = min_age
    age_div_mass = [[0] * 2 for i in range(int((max_age - min_age) / 10))]
    for i in range(int((max_age - min_age) / 10)):
        for i2 in range(2):
            age_div_mass[i][i2] = tmp_age
            tmp_age += 10
        tmp_age -= 10

    general_entrophy = abs((float(1/25) * math.log10(float(1/25))) * 25)
    amount_of_men = len(tree_rows[tree_rows["пол"] == "Male"])
    amount_of_women = len(tree_rows[tree_rows["пол"] == "Female"])

    gender_entrophy = abs((float(amount_of_men/25) * math.log10(float(amount_of_men/25))) + (float(amount_of_women/25) * math.log10(float(amount_of_women/25))))

    age_entr_table = ""
    for i in range(int((max_age - min_age) / 10)):
        tmp_tree_rows = tree_rows[tree_rows["возраст"] >= age_div_mass[i][0]]
        age_entr_table += '<h4 style="margin: 0">Кол-во ' + str(age_div_mass[i][0]) + '-' + str(age_div_mass[i][1]) + ': ' + str(len(tmp_tree_rows[tmp_tree_rows["возраст"] < age_div_mass[i][1]])) + '</h4>'

    age_entrophy = 0
    for i in range(int((max_age - min_age) / 10)):
        tmp_tree_rows = tree_rows[tree_rows["возраст"] >= age_div_mass[i][0]]
        age_entrophy += abs((float(len(tmp_tree_rows[tmp_tree_rows["возраст"] < age_div_mass[i][1]])/25) * math.log10(float(len(tmp_tree_rows[tmp_tree_rows["возраст"] < age_div_mass[i][1]])/25))))

    solution_tree = '<h3>Мужчины:</h3><div style="display: flex; justify-content: space-between;">'
    tree_mass = [[0] * 5 for i in range(int((max_age - min_age) / 10) * 2)]
    age_div_mass_i = 0
    for i in range(len(age_div_mass) * 2):
        gender_str = 'Male'
        if i >= 5:
            gender_str = 'Female'
        if i == 5:
            age_div_mass_i = 0
            solution_tree += '</div><h3>Женщины:</h3><div style="display: flex; justify-content: space-between;">'
        div_flag = 0
        tree_mass_i = 0
        for index, tmp_tree_chain in tree_rows.iterrows():
            if str(tmp_tree_chain["пол"]) == gender_str:
                if(tmp_tree_chain["возраст"] >= age_div_mass[age_div_mass_i][0]) & (tmp_tree_chain["возраст"] < age_div_mass[age_div_mass_i][1]):
                    if div_flag == 0:
                        solution_tree += '<div><h4 style="margin: 2% 0 0 0">Возраст: ' + str(age_div_mass[age_div_mass_i][0]) + "-" + str(age_div_mass[age_div_mass_i][1]) + "</h4>"
                    solution_tree += '<h4 style="margin: 0">Пол: ' + str(tmp_tree_chain["пол"]) + " Возраст: " + str(tmp_tree_chain["возраст"]) + " ИМТ: " + str(tmp_tree_chain["ИМТ"]) + "</h4>"
                    tree_mass[i][tree_mass_i] = tmp_tree_chain["ИМТ"]
                    tree_mass_i += 1
                    div_flag = 1
        if div_flag == 1:
            solution_tree += "</div>"
        age_div_mass_i += 1
    solution_tree += "</div>"

    age_div_mass_i = 0
    checking_charts_show = ""
    for i in range(len(tree_mass)):
        gender_str = 'Male'
        if i >= 5:
            gender_str = 'Female'
        if i == 5:
            age_div_mass_i = 0
        plot_flag = 0
        for index, tmp_checking_chain in checking_rows.iterrows():
            if str(tmp_checking_chain["пол"]) == gender_str:
                if(tmp_checking_chain["возраст"] >= age_div_mass[age_div_mass_i][0]) & (tmp_checking_chain["возраст"] < age_div_mass[age_div_mass_i][1]):
                    if plot_flag == 0:
                        checking_charts_show += '<div><h4 style="margin: 2% 0 0 0">Пол: ' + gender_str + ' Возраст: ' + str(age_div_mass[age_div_mass_i][0]) + '-' + str(age_div_mass[age_div_mass_i][1]) + '</h4>'
                    plt.plot([0, 50], [float(tmp_checking_chain["ИМТ"]), float(tmp_checking_chain["ИМТ"])], color="#bd0000")
                    checking_charts_show += '<h4 style="margin: 0">Пол: ' + str(tmp_checking_chain["пол"]) + " Возраст: " + str(tmp_checking_chain["возраст"]) + " ИМТ: " + str(tmp_checking_chain["ИМТ"]) + "</h4>"
                    plot_flag = 1
        if plot_flag == 1:
            for j in range(5):
                if float(tree_mass[i][j]) != 0:
                    plt.plot([0, 50], [float(tree_mass[i][j]), float(tree_mass[i][j])], color="#1300bd")
            checking_charts_show += '<img src="' + os.path.join(app.config['UPLOAD_FOLDER'], 'st' + str(i) + '.png') + '"/></div>'
            plt.savefig('static/charts/st' + str(i) + '.png')
        plt.close()
        age_div_mass_i += 1

    return '<h1 align="center">Дерево решений (ИМТ от пола и возраста)</h1>' \
           '<div style="margin: 0 10%"><div style="display: flex; justify-content: space-start;"><div><h4>Данные для построения дерева решений:' \
           '</h4>' + tree_rows.to_html() + '</div><div style="margin: 0 0 0 2%"><h4>Данные для проверки дерева решений:</h4>' + checking_rows.to_html() + '</div></div>' \
           '<h3 style="margin: 2% 0 0 0">Вариативность данных (имеем 2 критерия)</h3><h4 style="margin: 0">Пол: Male, Female</h4><h4 style="margin: 0 0 2% 0">Возраст: ' + age_to_show + "</h4>" \
           '<h2 align="center">Подсчёт энтропии</h2> <div style="display: flex; justify-content: space-between;"> <div><h3>Полная энтропия: ' + str(general_entrophy) + '</h3></div>' \
           '<div><h3 style="margin: 2% 0 0 0">Энтропия, критерий пол: </h3><h4 style="margin: 0">Кол-во мужчин: ' + str(amount_of_men) + '</h4><h4 style="margin: 0">Кол-во женщин: ' + str(amount_of_women) + '</h4><h4 style="margin: 0 0 1% 0">Энтропия: ' + str(gender_entrophy) + '</h4></div>' \
           '<div><h3 style="margin: 2% 0 0 0">Энтропия, критерий возраст: </h3>' + age_entr_table + '<h4 style="margin: 0 0 1% 0">Энтропия: ' + str(age_entrophy) + '</h4></div></div>' \
           '</div><div style="margin: 0 3%"><h2 align="center">Дерево решений</h2>' + str(solution_tree) + '</div><div style="margin: 0 6%; max-width: 100%;">' \
           '<h2 align="center">Поверка дерева решений на 5ти данных из dataset</h2><div style="display: flex; flex-wrap: wrap;"> ' + checking_charts_show + '</div></div>'


@app.route("/classterisation", methods=['GET'])
def classterisation():
    notna_df = initial_df[initial_df["ИМТ"].notna()]
    notna_df = notna_df.reset_index()
    divider = int(len(notna_df) / 4)

    class_df = notna_df.iloc[0:5][["ИМТ", "возраст"]]
    for i in range(15):
        if i < 5:
            class_df.loc[5 + i] = notna_df.loc[divider + i][["ИМТ", "возраст"]]
        if i >= 5 & i < 10:
            class_df.loc[5 + i] = notna_df.loc[divider * 2 + (i-5)][["ИМТ", "возраст"]]
        if i >= 10:
            class_df.loc[5 + i] = notna_df.loc[divider * 3 + (i - 10)][["ИМТ", "возраст"]]
    class_df = class_df.reset_index()

    class_df.plot.scatter(x="возраст", y="ИМТ", s=50, color="#4682B4")
    plt.savefig('static/charts/checkclass.png')

    centroid_table_1 = [[22, 2], [33, 26], [30, 50], [41, 69], [28, 77]]
    centroid_table_2 = [[22.05, 1.16], [32.2, 27.8], [29.75, 47], [37.575, 66.25], [26.76, 78.6]]

    return '<body style="padding: 7% 0 13% 0; background-color: #C0C0C0;"> <h1 style="color: #FFF5EE;" align="center">Кластеризация (критерии: ИМТ и возраст)</h1>' \
        + '' \
        + '<div style="margin: 3% 10% 0 10%; background-color: #FFFFFF; padding: 3%;"><div style="display: flex; justify-content: space-start;"> ' \
        + '<div><h3>Этап 1. Выборка данных</h3><p>Проводить кластеризацию сразу на всех данных будет ' \
        + 'малоэффективно, поскольку либо кластеров будет слишком много, либо данные в кластерах будут ' \
        + 'чересчур отличаться друг от друга. Начнём кластеризацию с 20 значений, взятых случайно ' \
        + 'из разных частей начального датафрейма. Построим точечную диаграмму</p></div>' \
        + '<div><img style="width: 640px; height: 480px;" src="' + os.path.join(app.config['UPLOAD_FOLDER'], 'checkclass.png') + '"/></div></div>' \
        + '' \
        + '<div style="display: flex; justify-content: space-start;"> <div>' \
        + '<img style="width: 640px; height: 480px;" src="' + os.path.join(app.config['UPLOAD_FOLDER'], 'class1.png') + '"/></div>' \
        + '<div><h3>Этап 2. Предполагаемое разбиение на кластеры</h3><p>Разбить данный график ' \
        + 'на кластеры сразу достаточно сложно. Наиболее выделяющимися выглядят два элемента внизу '\
        + 'и три элемента наверху. Центральную же часть попробуем разделить на три кластера: '\
        + 'слева, по центру и справа. Таким образом, мы не получим видимых пересечений кластеров. '\
        + 'Построим график и обозначим элементы 5ти кластеров разными цветами. '\
        + 'Выберем центры масс (возьмём пересечания диагоналей предполагамеых кластеров)</p>' + centroid_table_show(centroid_table_1) + '</div></div>' \
        + '' \
        + '<h3>Этап 3: Кластеризация</h3><h3>Итерация 1</h3><div style="display: flex; justify-content: space-start;">' \
        + '<div>' + claster_shower(claster_maker(centroid_table_1, class_df)) + '</div><div style="margin-left: 3%; max-width: 35%;">' \
        + '<p>Теперь, по получившимся данным, найдём координаты новых центройдов кластеров ' \
        + 'как среднее между всеми ИМТ и Возрастами соответствующих эл-тов кластера: </p>' + centroid_table_show(centroid_table_2) + '<p>' \
        + 'Заканчиваем нашу итерацию и начинаем новую также с рассчёта расстояний. </p></div></div>' \
        + '' \
        + '<h3>Итерация 2</h3><div style="display: flex; justify-content: space-start;">' \
        + '<div>' + claster_shower(claster_maker(centroid_table_2, class_df)) + '</div><div style="margin-left: 3%; max-width: 35%;">' \
        + '<p>Как мы можем заметить, на второй итерации элементы не сменили свои кластеры, а значение  ' \
        + 'среднеквадратической ошибки уменьшилось. Следовательно, мы можем закончить кластеризацию и вывести результат.</p></div></div>' \
        + '<h3> Результат кластеризации </h3><img style="display: block;  margin-left: auto; margin-right: auto; width: 70%;" src="' + os.path.join(app.config['UPLOAD_FOLDER'], 'class2.png') + '"/>' \
        + '' \
        + '<h3>Вывод</h3><p>Подводя итоги, кластеризация хорошо показала себя на небольшом количестве данных ' \
        + 'и даже выдала достаточно логичное разделение на группы, но при увеличении кол-ва данных логика ' \
        + 'терялась. Данный алгоритм применим на данных параметрах только к небольшому кол-ву элементов и ' \
        + 'неприменим к большому.</p></div></body>'


def claster_maker(centers, elements):
    result = [[0] * (len(centers) + 3) for i in range(len(elements))]
    for i in range(int(len(elements))):
        result[i][0] = i
        tmp_IMT = float(elements.loc[i]["ИМТ"])
        tmp_AGE = float(elements.loc[i]["возраст"])
        for j in range(int(len(centers))):
            result[i][j+1] = round(((centers[j][0] - tmp_IMT)**2 + (centers[j][1] - tmp_AGE)**2)**0.5, 4)
        tmp_MIN = result[i][1]
        for j in range(int(len(centers) - 1)):
            if tmp_MIN > result[i][j + 2]:
                tmp_MIN = result[i][j + 2]
        result[i][len(centers) + 1] = tmp_MIN
        result[i][len(centers) + 2] = round(tmp_MIN**2, 4)
    return result


def centroid_table_show(table):
    result = '<table style="border: 1px solid #333;"><tr><th style="border: 1px solid #333;">ЦМ</th><th style="border: 1px solid #333;">ИМТ</th><th style="border: 1px solid #333;">Возраст</th></tr>'
    for i in range(5):
        if i == 0:
            result += '<tr><td style="background-color: #FF0000; border: 1px solid #333;">Red </td>'
        if i == 1:
            result += '<tr><td style="background-color: #FFFF00; border: 1px solid #333;">Yellow</td>'
        if i == 2:
            result += '<tr><td style="background-color: #4682B4; border: 1px solid #333;">Blue</td>'
        if i == 3:
            result += '<tr><td style="background-color: #00FF00; border: 1px solid #333;">Green</td>'
        if i == 4:
            result += '<tr><td style="background-color: #800080; border: 1px solid #333;">Purple</td>'
        result += '<td style="border: 1px solid #333;">' + str(table[i][0]) + '</td><td style="border: 1px solid #333;">' + str(table[i][1]) + '</td></tr>'
    result += '</table>'
    return result


def claster_shower(elements):
    result = '<table style="border: 1px solid #333;"><tr><th style="border: 1px solid #333;">Эл-ты</th>' \
             + '<th style="border: 1px solid #333;">До Red</th><th style="border: 1px solid #333;">До Yellow</th>' \
             + '<th style="border: 1px solid #333;">До Blue</th><th style="border: 1px solid #333;">До Green</th>' \
             + '<th style="border: 1px solid #333;">До Purple</th><th style="border: 1px solid #333;">Минимум</th>' \
             + '<th style="border: 1px solid #333;">Квадраты расстояний</th></tr>'

    for i in range(int(len(elements))):
        result += '<tr><td style="border: 1px solid #333;">' + str(elements[i][0]) + '</td>' \
                + '<td style="border: 1px solid #333;">' + str(elements[i][1]) + '</td>' \
                + '<td style="border: 1px solid #333;">' + str(elements[i][2]) + '</td>' \
                + '<td style="border: 1px solid #333;">' + str(elements[i][3]) + '</td>' \
                + '<td style="border: 1px solid #333;">' + str(elements[i][4]) + '</td>' \
                + '<td style="border: 1px solid #333;">' + str(elements[i][5]) + '</td>'

        for j in range(5):
            if elements[i][j+1] == elements[i][6]:
                if j == 0:
                    result += '<td style="background-color: #FF0000; border: 1px solid #333;">' + str(elements[i][6]) + '</td>'
                if j == 1:
                    result += '<td style="background-color: #FFFF00; border: 1px solid #333;">' + str(elements[i][6]) + '</td>'
                if j == 2:
                    result += '<td style="background-color: #4682B4; border: 1px solid #333;">' + str(elements[i][6]) + '</td>'
                if j == 3:
                    result += '<td style="background-color: #00FF00; border: 1px solid #333;">' + str(elements[i][6]) + '</td>'
                if j == 4:
                    result += '<td style="background-color: #800080; border: 1px solid #333;">' + str(elements[i][6]) + '</td>'

        result += '<td style="border: 1px solid #333;">' + str(elements[i][7]) + '</td></tr>'
    result += '</table><p>Среднеквадратическая ошибка: '
    summ = 0
    for i in range(int(len(elements))):
        summ += elements[i][7]
    result += str(summ) + '</p>'

    return result


def age_construct(age):
    if age % 1 == 0:
        if age % 10 == 2 or age % 10 == 3 or age % 10 == 4:
            value = str(int(age)) + " года"
        if age % 10 == 1:
            value = str(int(age)) + " год"
        if age % 10 == 0 or age % 10 == 5 or age % 10 == 6 or age % 10 == 7 or age % 10 == 8 or age % 10 == 9:
            value = str(int(age)) + " лет"
    else:
        if int(age) == 0:
            value = str(int((age % 1) * 100)) + " месяцев "
        else:
            value = str(int(age)) + " лет " \
            + str(age % 1) + " месяцев "
    return value


if __name__ == "__main__":
    app.run(port=5001, debug=True)