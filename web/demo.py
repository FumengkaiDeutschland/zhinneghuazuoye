
import streamlit as st
from multipage import MultiPage
import home, deeplearning,machinelearning,separation
st.set_page_config(
        page_title="集输站场信息📈",
)
st.title("集输站场信息事实数据采集系统")



app = MultiPage()



app.add_page('首页',home.app)
app.add_page('三相分离器液位控制',separation.app)
app.add_page('机器学习模型构建',machinelearning.app)
app.add_page('深度学习模型构建',deeplearning.app)


if __name__=='__main__':
    app.run()