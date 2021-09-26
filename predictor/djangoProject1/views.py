from django.shortcuts import render
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def home(request):
    return render(request, 'home.html')


def predict(request):
    return render(request, 'predict.html')


def result(request):
    dataframe = pd.read_csv(r'C:\Users\91936\heart.csv')
    df = dataframe.dropna()
    df = df.drop(columns=['slope', 'thal', 'fbs', 'restecg', 'exang', 'sex'])
    X = df.drop(['target'], axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=40)

    model = DecisionTreeClassifier(max_depth=5, criterion='entropy')
    model.fit(X, y)

    val1 = float(request.GET['n1'])
    val2 = float(request.GET['n2'])
    val3 = float(request.GET['n3'])
    val4 = float(request.GET['n4'])
    val5 = float(request.GET['n5'])
    val6 = float(request.GET['n6'])
    val7 = float(request.GET['n7'])

    pred1 = (val1, val2, val3, val4, val5, val6, val7)
    input_as_numpy = np.asarray(pred1)
    input_reshaped = input_as_numpy.reshape(1, -1)
    pred = model.predict(input_reshaped)

    if pred == [1]:
        result1 = "You might have Heart Disease"
    else:
        result1 = "You might not have Heart Risk"

    return render(request, "predict.html", {"result2": result1})
