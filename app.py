import streamlit as st
import pandas as pd

st.write("Hello World")

number = st.number_input('Insert a number')
st.write('The current number is ', number)