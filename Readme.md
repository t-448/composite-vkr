# Прогнозирование конечных свойств новых материалов. #

*Выпускная квалификационная работа по курсу «Data Science» в Образовательном Центре МГТУ им. Н.Э. Баумана по теме: "Прогнозирование конечных свойств новых материалов (композиционных материалов)".*

## Поставленная задача данной работы: ## 

Имеется ряд входных параметров, по которым необходимо спрогнозировать конечные свойства композиционных материалов. 

## Входные данные: ##
Изначально было предотавлено 2 датасета. Путем объединения файлов методом inner был сформирован общий датасет, содержащий следующие параметры:

- Соотношение матрица-наполнитель

- Плотность, кг/м3

- Модуль упругости, ГПа

- Количество отвердителя, м.%

- Содержание эпоксидных групп,%_2

- Температура вспышки, С_2

- Поверхностная плотность, г/м2

- Потребление смолы, г/м2

- Прочность при растяжении, МПа

- Потребление смолы, г/м2

- Угол нашивки, град

- Шаг нашивки

- Плотность нашивки

## Шаги выполнения ВКР: ##
### Этапы создания программы ###
1. Изучение теоретической информации, предоставленной во время обучения. А также, различных методов прогнозирования.
2. Проведение разведочного анализа и представление визуализации предложенных данных. В таблице представлены для каждой колонки среднее, медианное значение, проведен анализ и исключены выбросы, проверка на наличие пропусков. Представлены гистограммы распределения переменных, диаграммы boxplot, попарные графики рассеяния точек.
3. Проведение предобработки данных (удалены выбросы, нормализация).
4. Обучение нескольких моделей для формирования прогноза *модуля упругости при растяжении* и *прочности при растяжении*.
5. Формирование общей таблицы с полученными значениями прогнозирования по каждой модели/методу прогнозирования, используемого при написании ВКР. В каждом столбце показателей выделено наилучшее значение.
6. Написание нейронных сетей для последующего расчета прогноза *Соотношение "матрица-наполнитель"*
7. Разработка пользовательского приложения на Flask для расчета прогноза *Соотношение "матрица - наполнитель"*.
8. Оценка точности модели на тренировочном и тестовом датасете.
### Этапы загрузки проекта ###
1. Создание аккаунта и репозитория на Github
2. Загрузка кода и материалов выполненного исследования
3. Состаление и размещение в корне репозитория файла Readme.md

## Структура репозитория:##

- Datasets - папка с файлами: X_bp.xlsx - Первый датасет, X_nup.xlsx - Второй датасет; df_clean.csv - данные для обучения моделей (simple and best) df_norm.csv - данные для обучения моделей (norm)
- App/app.py - папка с файлом приложение Flask; 
- App/app.ipynb - ноутбук с файлом приложение Flask;
- App/templates - папка с html документом фронтэнда для flask приложения;
- vkr_Предобработка данных.ipynb - ноутбук с кодом проекта и всеми полученными данными;
- Построение_и_обучение_моделей.ipynb - файл с кодом и результатми построения и прогнозирования моделей *модуля упругости при растяжении* и *прочности при растяжении*;
- Построение_нейронно_сети.ipynb - файл с кодом построения нейронных сетей;
- models/model1 - результат (модель) построения первой нейронной сети;
- models/model2 - результат (модель) построения второй нейронной сети;
- models/models_result.xlsx - таблица с полученными значениями во время построения и обучения данных;
- Пояснительная записка;
- Презентация.


## Инструкция использования пользовательского приложения: ##

Приложение позволяет решать задачу прогнозирования "Соотношение матрица наполнитель".

Для получения прогноза необходимо:

Инструкция использования приложения:
1. Запустить из папки App приложение app.py;
2. В появившейся строке  ```(* Running on http://127.0.0.1:5000/ )``` - нажать на ссылку: http://127.0.0.1:5000/
3. В открывшемся окне браузера указать значения параметров для рассчета и нажать "Рассчитать";
4. Ознакомиться с появившемся результатом. Рузультатом программы явялется число с плавающей точкой.

*Автор: Бойко Татьяна Сергеевна*