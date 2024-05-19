import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
from requests import get, post
import json


st.title('Дашборд для номенклатуры')


def check_password():
    if st.session_state.get("password_correct", False):
        return True
    placeholder = st.empty()
    with placeholder.container():
        username = st.text_input("Username", "")
        password = st.text_input("Password", "")
        col1, col2, *_ = st.columns(7)
        with col1:
            reg = st.button("Register")
        with col2:
            logi = st.button("Login")
        if reg:

            r = post("http://localhost:8000/register", json={"username": username, "password": password})
            if r.status_code == 200 and len(username) >= 3 and len(password) >= 3:
                st.session_state["password_correct"] = True
                placeholder.empty()
                return True
            else:
                st.write("Этот username уже занят")
        if logi:
            r = post("http://localhost:8000/login", json={"username": username, "password": password})
            if r.status_code == 200 and len(username) >= 3 and len(password) >= 3:
                st.session_state["password_correct"] = True
                placeholder.empty()
                return True
            else:
                st.write("Неверное имя пользователя или пароль")
        return False


if not check_password():
    st.stop()





uploaded_file = st.file_uploader("Выберите данные")
if uploaded_file is not None:
     stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
     string_data = stringio.read()
     ans = post("http://localhost:8000/get_nearest_documents", json={"data": string_data})

     st.download_button('Скачать ответ', bytes(ans.text, encoding='utf-8'), 'answer.csv')
st.write(' ')
title = st.text_input("Запрос", "")
if title:
    ans = post("http://localhost:8000/get_nearest_documents", json={"data": title})
    st.write(ans.text)