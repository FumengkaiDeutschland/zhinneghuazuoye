import streamlit as st
import requests
from streamlit_lottie import st_lottie
from PIL import Image
import pandas as pd
import numpy as np


def load_lottiie(url):
    r = requests.get(url)
    if r.status_code !=200:
        return None
    return r.json()


def app():
    with st.container():
        st.write('____________________')
        l_col, r_col = st.columns((2, 2))
        with l_col:
            Fig10 = Image.open('C:\\Users\\Lannister\\OneDrive\\桌面\\WebFig\\11.png')
            st.image(Fig10)
        with r_col:
            st.write(" ")
            st.write(" ")
            st.write(" ")
            st.write(" ")
            st.title("PID控制参数选择")
            st.sidebar.slider('比例部分参数📈', 0, 100)
            st.sidebar.slider('积分部分参数📈', 0, 100)
            st.sidebar.slider('微分部分参数📈', 0, 100)

    with st.container():
        st.write('---------------------------')
        st.subheader("三项分离器液位控制系统简介")
        st.write("  在三相分离器中，对于井口采出液的流量有着较高的要求，当流量过高时,设备无法容纳来液，当流量过低时，其会导致分离不充分，因此需要流量机来对井口采出来液进行测量。")
        #load_css('C:\\Users\\Lannister\\OneDrive\\桌面\\style.css')
        video_file = open('C:\\Users\\Lannister\\OneDrive\\桌面\\WebFig\\1.mp4', 'rb')
        video_bytes = video_file.read()
        st.video(video_bytes)
    with st.container():
        st.write('---------------------------')
        st.subheader("三项分离器液位控制系统")
        l_col, r_col = st.columns((1, 4))
        with l_col:
            st.write(" ")
            st.write(" ")
            st.write(" ")
            st.write(" ")
            st.write(" ")
            st.write(" ")
            inputsize = st.text_input("输入控制液位", 32)
            on = st.toggle('开始')
            if on:
                st.write('开始调节！！')
            else:
                st.write('准备中')
        with r_col:
            if on:
                video_file = open('C:\\Users\\Lannister\\OneDrive\\桌面\\WebFig\\22.mp4', 'rb')
                video_bytes = video_file.read()
                st.video(video_bytes)
            else:
                Fig10 = Image.open('C:\\Users\\Lannister\\OneDrive\\桌面\\WebFig\\10.png')
                st.image(Fig10)
    with st.container():
        st.write('------------------')
        st.title("三相分离器液位")
        url = 'C:\\Users\\Lannister\\OneDrive\\桌面\\WebFig\\液位.xlsx'
        chart_data = pd.read_excel(url)
        oil = chart_data[["时间","油位"]]
        water = chart_data[["时间", "水位"]]
        st.bar_chart(oil, x="时间", y="油位")
        st.write('------------------')
        st.bar_chart(water, x="时间", y="水位",color=["#0000FF"])
        on = st.toggle('检测是否存在液位超标')
        if on:
            st.write("检测开始")
            for i in range(len(oil)):
                if oil.iloc[i,1]>0.9:
                    st.write("第"+str(i+1)+"时"+"油位超标！！！")
                    st.write("请及时调节！！！！")
                    exit()
                if water.iloc[i,1]>0.9:
                    st.write("第"+str(i+1)+"时""水超标！！！")
                    st.write("请及时调节！！！！")
                    exit()



        else:
            st.write("未检测")



