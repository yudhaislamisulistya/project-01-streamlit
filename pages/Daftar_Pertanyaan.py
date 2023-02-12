import random
import sqlite3

import numpy as np
import pandas as pd
import streamlit as st
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import \
    StopWordRemoverFactory
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from streamlit_chat import message
from streamlit_option_menu import option_menu
from symspellpy import SymSpell, Verbosity

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 


# Connect to the database (or create it if it doesn't exist)
conn = sqlite3.connect('faq.db')

# Create the table to store the FAQ data (if it doesn't already exist)
conn.execute('''CREATE TABLE IF NOT EXISTS faq (id integer primary key,question text, answer text)''')

def delete_faq():
    st.markdown('## Hapus Pertanyaan')
    faq_id = st.text_input('Masukkan ID Pertanyaan Untuk Menghapus:', key='deleteFaq')
    if st.button('Delete'):
        # Delete the FAQ from the database
        conn.execute("DELETE FROM faq WHERE id=?", (faq_id,))
        conn.commit()
        st.success('FAQ deleted successfully')

def update_faq():
    st.markdown('## Update Pertanyaan')
    faq_id = st.text_input('Masukkan ID Pertanyaan Untuk Memperbaharui:', key='updateFaqWithId')
    updated_answer = st.text_input('Masukkan Jawaban Baru:', key='updateFaqAnswer')
    if st.button('Update'):
        # Update the answer in the database
        conn.execute("UPDATE faq SET answer=? WHERE id=?", (updated_answer, faq_id))
        conn.commit()
        st.success('FAQ updated successfully')
        
def add_faq():
    st.markdown('## Tambah Pertanyaan')
    question = st.text_input('Masukkan Pertanyaan:', key='addFaqQuestion')
    answer = st.text_input('Masukkan Jawaban:', key='addFaqAnswer')
    if st.button('Save'):
        conn.execute("INSERT INTO faq (question, answer) VALUES (?, ?)", (question, answer))
        conn.commit()
        st.success('Daftar Berhasil Ditambahkan')

def view_faq():
    st.markdown('## Daftar Pertanyaan')
    df = pd.read_sql_query("SELECT * FROM faq", conn)
    st.table(df)

def list_questions():
    st.title('Daftar Pertanyaan')
    add_faq()
    update_faq()
    delete_faq()
    view_faq()
    
if __name__ == '__main__':
    list_questions()

