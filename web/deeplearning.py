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

##
def get_database(hostName):#hostName:'localhost'
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
    return X,X1,y,y1
def DeepLearningModel(inputsize):
    model = Sequential()
    model.add(Conv1D(filters=inputsize,
                     kernel_size=7,
                     padding='same',
                     input_shape=(18 * 3, 1),
                     activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=32,
                     kernel_size=7,
                     padding='same',
                     activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=64,
                     kernel_size=5,
                     padding='same',
                     activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(128, kernel_regularizer=keras.regularizers.l2(0.01), activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(64, kernel_regularizer=keras.regularizers.l2(0.01), activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(32, kernel_regularizer=keras.regularizers.l2(0.01), activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(4, activation='softmax'))
    return model
def modelStructure(model):
    from tensorflow.keras import utils
    utils.plot_model(model, 'C:\\Users\\Lannister\\OneDrive\\桌面\\WebFig\\model1.png', show_shapes=True, show_dtype=True, show_layer_names=True)






def app():
    with st.container():
        st.write('__________')
        st.title("卷积神经网络训练")
        l_col, r_col = st.columns((1, 3.5))
        with l_col:
            data_type = st.sidebar.selectbox(
                "选择数据类型",
                ("MYSQL", "Excel", "CSV"))
            hostName = st.text_input('数据位置', 'localhost')
            inputsize = st.text_input("数据尺寸", 32)
            epochNum = st.text_input("最大迭代数", 5)
            inputsize = int(inputsize)
            epochNum = int(epochNum)

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

            model = DeepLearningModel(inputsize)
            st.button("Reset", type="primary")
            if st.button('Train'):
                from fastprogress import master_bar, progress_bar
                model.compile(loss='categorical_crossentropy',
                              optimizer=tf.keras.optimizers.Adam(learning_rate=10e-5), metrics=['accuracy'])
                train_history = model.fit(x=X,
                                    y=y, batch_size=128,
                                    epochs=epochNum, validation_data=(X1, y1),
                                    callbacks=[PlotLossesKeras()])
                import matplotlib.pyplot as plt
                # 绘制训练 & 验证的损失值
                import matplotlib as mpl
                epoch = range(1,epochNum+1)
                clf()  # 清图。
                cla()  # 清坐标轴。
                import matplotlib.pyplot as plt
                # 绘制训练 & 验证的损失值
                plt.rcParams['font.sans-serif'] = ['SimHei']
                plt.rcParams['axes.unicode_minus'] = False
                plt.style.use('seaborn')

                plt.plot(train_history.history['loss'])
                plt.plot(train_history.history['val_loss'])
                plt.title('损失', fontsize=20)
                plt.ylabel('损失', fontsize=20)
                plt.xlabel('迭代次数', fontsize=20)
                plt.yticks(fontsize=20)
                plt.xticks(fontsize=20)
                plt.legend(['Train', 'Test'], loc='upper right', fontsize=15)
                plt.savefig("C:\\Users\\Lannister\\OneDrive\\桌面\\卷积神经网络loss.jpg",dpi=400,bbox_inches = 'tight')
            if st.button('StoreData'):
                loss = pd.DataFrame(train_history.history['loss'], columns=["loss"])
                loss["val_loss"] = train_history.history['val_loss']
                loss.to_excel("C:\\Users\\Lannister\\OneDrive\\桌面\\ABC.xlsx")
        with r_col:
            if st.button('ShowFigure📈'):
                image_Loss = Image.open("C:\\Users\\Lannister\\OneDrive\\桌面\\卷积神经网络loss.jpg")
                st.image(image_Loss)
    with st.container():
        st.title("深度学习模型")
        Fig20 = Image.open('C:\\Users\\Lannister\\OneDrive\\桌面\\WebFig\\20.png')
        st.image(Fig20)
        modelStructure(model)
        l_col, r_col = st.columns((1.5, 3.5))
        with l_col:
            on = st.toggle('生成模型示意图')
        with r_col:
            if on:
                st.write("     本模型示意图")
                Fig21 = Image.open('C:\\Users\\Lannister\\OneDrive\\桌面\\WebFig\\model1.png')
                st.image(Fig21)
            else:
                st.write("等待本模型图生成")











