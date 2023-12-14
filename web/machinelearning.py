import streamlit as st
import numpy as np
from io import BytesIO

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pymysql as MySQLdb
import numpy as np
import matplotlib.pyplot as plt
import xlrd
import pandas as pd
import xlwt
import math
from pylab import *
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import pymysql
import base64
from tensorflow.keras import datasets, layers, models
from keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Flatten,Conv1D,MaxPooling1D
from keras.utils import np_utils
import tensorflow as tf
from tensorflow import keras
from livelossplot import PlotLossesKeras
from time import sleep
from PIL import Image

def get_database(hostName):  # hostName:'localhost'
    db = pymysql.connect(host=hostName, port=3306, user="root", passwd="fu123456", db='imu', charset='utf8')
    cursor = db.cursor()
    cursor.execute("select value,type from imu_dl")
    result = cursor.fetchall()
    df = pd.DataFrame(result, columns=["value", "type"])
    dbX_train = np.array(df["value"])
    dbX_train = dbX_train.tolist()
    X_train = []
    for i in range(0, len(dbX_train)):
        B = dbX_train[i].replace('][', ']\n[')
        C = B.split('\n')
        D = [eval(a) for a in C]
        #     E = (D-np.min(D))/(np.max(D)-np.min(D))
        #     E=E.tolist()
        X_train.append(D)
    dbX_label = np.array(df["type"])
    dbX_label = dbX_label.tolist()
    X_label = []
    for i in range(0, len(dbX_label)):
        X_label.append(int(dbX_label[i]))
    class_names = ['凹陷', '弯头', '弯曲应变', '环焊缝异常']

    wo_train = []
    train_12 = []
    train_14 = []
    train_16 = []
    train_18 = []
    for i in range(0, len(X_train)):
        #     wo_train.append(X_train[i][0:2])
        wo_l20 = [X_train[i][0][1:19], X_train[i][1][1:19]]
        l18 = [X_train[i][0][1:19], X_train[i][1][1:19], X_train[i][2][1:19]]
        l16 = [X_train[i][0][1:19], X_train[i][1][1:19], X_train[i][2][1:19]]
        l14 = [X_train[i][0][1:19], X_train[i][1][1:19], X_train[i][2][1:19]]
        l12 = [X_train[i][0][1:19], X_train[i][1][1:19], X_train[i][2][1:19]]
        train_18.append(l18)
        train_16.append(l16)
        train_14.append(l14)
        train_12.append(l12)
        wo_train.append(wo_l20)

    the_lable = []
    for i in range(len(X_label)):
        if X_label[i] == 0:
            x = "凹陷"
        elif X_label[i] == 1:
            x = "弯头"
        elif X_label[i] == 2:
            x = "弯曲变形段"
        elif X_label[i] == 3:
            x = "环焊缝干扰段"
        the_lable.append(x)

    import random
    index = [i for i in range(len(X_train))]
    random.shuffle(index)

    X_train_20 = np.array(X_train)
    X_train_20 = X_train_20[index]
    wo_train = np.array(wo_train)[index]
    train_12 = np.array(train_12)[index]
    train_14 = np.array(train_14)[index]
    train_16 = np.array(train_16)[index]
    train_18 = np.array(train_18)[index]
    X_label = np.array(X_label)
    X_label = X_label[index]

    plt.figure(figsize=(30, 5))
    for i in range(0, 1):
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(wo_train[i], cmap=plt.cm.hot)
        #     cb=plt.colorbar()
        #     cb.ax.tick_params(labelsize=30)
        plt.xlabel(the_lable[i], fontsize=30)
    #     plt.savefig("E:/imu/biye/环焊缝异常1423-1424",dpi=300)
    #     plt.clf()

    d = int(len(train_12) * 0.8)
    train_data = train_12[0:d]
    train_label = X_label[0:d]

    test_data = train_12[d:len(train_12)]
    test_label = X_label[d:len(train_12)]

    # from sklearn.model_selection import KFold
    # train_data,test_data,train_label,test_label = train_test_split(X_train_20, X_label, test_size=0.2)
    X = train_data.reshape(train_data.shape[0], 18 * 3, 1).astype('float32')
    X1 = test_data.reshape(test_data.shape[0], 18 * 3, 1).astype('float32')
    y = np_utils.to_categorical(train_label)
    y1 = np_utils.to_categorical(test_label)
    return X, X1, y, y1

def get_dataset(name):
    data = None
    if name == 'Iris':
        data = datasets.load_iris()
    elif name == 'Wine':
        data = datasets.load_wine()
    else:
        data = datasets.load_breast_cancer()
    X = data.data
    y = data.target
    return X, y


def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == 'SVM':
        C = st.sidebar.slider('C', 0.01, 10.0)
        params['C'] = C
    elif clf_name == 'KNN':
        K = st.sidebar.slider('K', 1, 15)
        params['K'] = K
    else:
        max_depth = st.sidebar.slider('max_depth', 2, 15)
        params['max_depth'] = max_depth
        n_estimators = st.sidebar.slider('n_estimators', 1, 100)
        params['n_estimators'] = n_estimators
    return params


def get_classifier(clf_name, params):
    clf = None
    if clf_name == 'SVM':
        clf = SVC(C=params['C'])
    elif clf_name == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=params['K'])
    else:
        clf = clf = RandomForestClassifier(n_estimators=params['n_estimators'],
                                           max_depth=params['max_depth'], random_state=1234)
    return clf


def app():
    st.write("""
            # 机器学习
            """)
    data_type = st.sidebar.selectbox(
        "选择数据类型",
        ("MYSQL", "Excel", "CSV"))
    hostName = st.text_input('数据位置', 'localhost')
    inputsize = st.text_input("数据尺寸", 32)
    inputsize = int(inputsize)
    classifier_name = st.sidebar.selectbox(
        'Select classifier',
        ('KNN', 'SVM', 'Random Forest')
    )
    if data_type == "MYSQL":
        X, X1, y, y1 = get_database(hostName)
    elif data_type == "Excel":
        import pd
        df = pd.read_excel(hostName, header=None)
        dataInput = df.iloc[0:len(df) - 1, 0:inputsize]
        dataOutput = df.iloc[0:len(df) - 1, -1]
        X, X1, y, y1 = train_test_split(dataInput, dataOutput, test_size=0.2)
    elif data_type == "CSV":
        import pd
        df = pd.read_CSV(hostName, header=None)
        dataInput = df.iloc[0:len(df) - 1, 0:inputsize]
        dataOutput = df.iloc[0:len(df) - 1, -1]
        X, X1, y, y1 = train_test_split(dataInput, dataOutput, test_size=0.2)

    # Get dataset

    st.write('Shape of dataset:', X.shape)
    st.write('number of classes:', len(np.unique(y)))

    # Hyper parameters
    params = add_parameter_ui(classifier_name)

    # Model
    clf = get_classifier(classifier_name, params)

    # Train model and plot results

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    st.write(f'Classifier = {classifier_name}')
    st.write(f'Accuracy =', acc)

    #### PLOT DATASET ####
    # Project the data onto the 2 primary principal components
    pca = PCA(2)
    X_projected = pca.fit_transform(X)

    x1 = X_projected[:, 0]
    x2 = X_projected[:, 1]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x1, x2,
               c=y, alpha=0.8,
               cmap='viridis')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')

    # Resize image
    buf = BytesIO()
    fig.savefig(buf, format="png")
    st.image(buf)