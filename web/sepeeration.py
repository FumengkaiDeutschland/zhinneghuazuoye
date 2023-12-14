import streamlit as st
import requests
from streamlit_lottie import st_lottie
from PIL import Image

def load_lottiie(url):
    r = requests.get(url)
    if r.status_code !=200:
        return None
    return r.json()


def app():
    #load_css('C:\\Users\\Lannister\\OneDrive\\桌面\\style.css')

    image_sphere =Image.open('C:\\Users\\Lannister\\OneDrive\\桌面\\1.jpg')
    image_phase_separation = Image.open('C:\\Users\\Lannister\\OneDrive\\桌面\\2.jpg')
    image_nano = Image.open('C:\\Users\\Lannister\\OneDrive\\桌面\\2.jpg')

    with st.container():
        st.write('__________')
        st.subheader("集输站场")

        l_col, r_col = st.columns((3, 2))
        with l_col:
            st.title("简介")
            st.write(
            "把分散的油井所生产的石油、伴生天然气和其他产品集中起来，经过必要的处理、初加工，合格的油和天然气分别外输到炼油厂和天然气用户的工艺全过程称为油气集输。主要包括油气分离、油气计量、原油脱水、天然气净化、原油稳定、轻烃回收等工艺。其中对于油品的流量监测至关重要。"
            )
        with r_col:
            Fig1 = Image.open('C:\\Users\\Lannister\\OneDrive\\桌面\\WebFig\\1.png')
            st.image(Fig1)

    with st.container():
        st.write('__________')
        st.title("主要设备")
        l_col, r_col = st.columns((2, 3))
        with l_col:
            Fig2 = Image.open('C:\\Users\\Lannister\\OneDrive\\桌面\\WebFig\\2.jpg')
            st.image(Fig2)
        with r_col:
            st.write(" ")
            st.write(" ")
            st.write(" ")
            st.write(" ")
            st.write(" ")
            st.write(" ")

            st.write('油气集输主要设备有油气分离器、加热炉、原油电脱 水器、塔器及泵等设备。')
    with st.container():
        st.write('__________')
        st.title("系统功能")
        Fig3 = Image.open('C:\\Users\\Lannister\\OneDrive\\桌面\\WebFig\\3.png')
        st.image(Fig3)
        st.write('油气集输 SCADA 系统是指用于监控和控制油气集输过程中液态、气态等各种介质流动的自动化监控系统。它通过获取油气集输管道的各种参数、状态等信息，实现对油气集输过程的全面了解和实时监管。油气集输 SCADA 系统可以显示多种信息，对运行情况进行实时监控，实现远程自动控制。')
        st.write('主要功能有：')
        st.write(
            '1. 数据采集和传输：系统通过现场数据采集器采集现场传感器或仪表测量的各种数据（如流量、压力、温度、液位等），并通过网络传输至控制中心。')
        st.write(
            '2. 数据显示和报警：系统将采集到的数据显示在控制中心的监控界面上，以图表或曲线的形式展现实时状态。同时，系统会对数据进行监控，如果数据异常，则会发出报警提示。')
        st.write(
            '3. 实时监测和控制： SCADA 系统能够实现对油气集输过程中的各个环节进行实时监测，当发现异常情况时，可以主动发出控制信号进行处理，以保证设备或管道的安全运行。')
        st.write(
            '4. 历史数据存储和分析： SCADA 系统对采集到的数据进行存储，可以长期保存历史数据，用于分析设备或管道的运行情况，预测故障风险，提高设备的利用率和可靠性。')
