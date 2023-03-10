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

# Load data
st.write('# Dataset')
df = pd.read_sql_query("SELECT * FROM faq", conn)
df = df.drop(['id'], axis=1)
df.rename(columns={'question': 'Pertanyaan', 'answer': 'Jawaban'}, inplace=True)
st.table(df)


# Text Prerocessing

## Casefolding
st.write('# Text Preprocessing')
st.write('## Casefolding')
df = df.apply(lambda x: x.astype(str).str.lower())
st.table(df['Pertanyaan'])

## Stopword Removal
st.write('## Stopword Removal')
factory = StopWordRemoverFactory()
stopwords = factory.get_stop_words()
df['Pertanyaan'] = df['Pertanyaan'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords)]))
st.table(df['Pertanyaan'])

## Stemming
st.write('## Stemming')
factory = StemmerFactory()
stemmer = factory.create_stemmer()
df['Pertanyaan'] = df['Pertanyaan'].apply(lambda x: stemmer.stem(x))
st.table(df['Pertanyaan'])


# TF IDF FOR Pertanyaan
st.write('# TF IDF')
df_pertanyaan = df['Pertanyaan'].values.tolist()
count_vectorizer = CountVectorizer()
count_vectorizer.fit_transform(df_pertanyaan)

tfidf_matrix = TfidfVectorizer().fit_transform(df_pertanyaan)

st.write('## TABEL TF')
# Menghitung TF From Dataset Table
df_tf = pd.DataFrame(np.zeros((len(df_pertanyaan), len(count_vectorizer.get_feature_names_out()))), columns=count_vectorizer.get_feature_names_out())
for i in range(len(df_pertanyaan)):
    words = df_pertanyaan[i].split(' ')
    for w in words:
        df_tf[w][i] = df_tf[w][i] + (1 / len(words))
st.table(df_tf)

st.write('## TABEL IDF')
# Menghitung IDF From Dataset Table
idf = {}

for w in count_vectorizer.get_feature_names_out():
    k = 0
    
    for i in range(len(df_pertanyaan)):
        if w in df_pertanyaan[i].split():
            k += 1
            
    idf[w] =  np.log10(len(df_pertanyaan) / k)

df_idf = pd.DataFrame.from_dict(idf, orient='index', columns=['IDF'])
st.table(df_idf)

st.write('## TABEL TF-IDF')
# Menghitung TF-IDF From Dataset Table
df_tf_idf = df_tf.copy()
for w in count_vectorizer.get_feature_names_out():
    for i in range(len(df_pertanyaan)):
        df_tf_idf[w][i] = df_tf[w][i] * idf[w]
        
df_tf_idf

df_tf_idf_final = pd.DataFrame(index=count_vectorizer.get_feature_names_out(), columns=["KATA", 'TF', 'IDF', 'TF-IDF'])
df_tf_idf_final['KATA'] = df_tf_idf_final.index
df_tf_idf_final['TF'] = df_tf.sum(axis=0)
df_tf_idf_final['IDF'] = df_idf['IDF']
df_tf_idf_final['TF-IDF'] = df_tf_idf_final['TF'] * df_tf_idf_final['IDF']
st.table(df_tf_idf_final)



st.write('# LSI (Latent Semantic Indexing)')
num_components=10
lsa = TruncatedSVD(n_components=num_components, n_iter=100, random_state=42)

st.write('## Matrix M')
matrix_m = df_tf_idf.to_numpy().T
index_name = count_vectorizer.get_feature_names_out()
columns_name = ['DOKUMEN' + str(i) for i in range(1, len(df_pertanyaan)+1)]
df = pd.DataFrame(matrix_m, index=index_name, columns=columns_name)
st.table(df)

st.write("## Matrix UK/U (Singular Vector)")
matrix_uk = lsa.fit_transform(df_tf_idf.to_numpy())
df = pd.DataFrame(matrix_uk, columns=['UK' + str(i) for i in range(1, num_components+1)])
st.table(df)

st.write("## Matrix SK/D (Singular Value)")
matrix_sk = np.diag(lsa.singular_values_)
df = pd.DataFrame(matrix_sk, columns=['SK' + str(i) for i in range(1, num_components+1)])
st.table(df)

st.write("## Matrix VK Transpose")
matrix_vk_transpose = lsa.components_.T
df = pd.DataFrame(matrix_vk_transpose, columns=['VK' + str(i) for i in range(1, num_components+1)])
st.table(df)

st.write("## Matrix P")
matrix_p = np.dot(matrix_uk, matrix_sk)
df = pd.DataFrame(matrix_p, columns=['P' + str(i) for i in range(1, num_components+1)])
st.table(df)

st.write("# Pengujian Question Answering System Helpdesk")
text = st.text_input('Pertanyaan', '', key='pertanyaanProses')
if text != '':
    sym_spell = SymSpell()

    path_corpus = "tempo.txt"
    sym_spell.create_dictionary(path_corpus)
    text_to_list = text.split(' ')
    update_text = []

    for w in text_to_list:
        suggestions = sym_spell.lookup(w, Verbosity.CLOSEST, max_edit_distance=2, include_unknown=True)
        update_text.append([s.term for s in suggestions][0])

    text = ' '.join(update_text)

    if text != '':
        st.write('Teks : ', text)
        
        text = text.lower()
        text = ' '.join([word for word in text.split() if word not in (stopwords)])
        text = stemmer.stem(text)
        text_to_column = [w for w in text.split(' ') if w in count_vectorizer.get_feature_names_out()]

        st.write('## TABEL TF PENGUJIAN')
        df_tf = pd.DataFrame(np.zeros((1, len(count_vectorizer.get_feature_names_out()))), columns=count_vectorizer.get_feature_names_out())
        for w in text_to_column:
            df_tf[w][0] = df_tf[w][0] + (1 / len(text_to_column))
        df_tf

        st.write('## TABEL TF-IDF PENGUJIAN')
        df_idf = df_tf.T.copy()
        df_idf.columns = ['TF']
        df_idf['IDF'] = np.log10(len(df_pertanyaan) / df_idf['TF'])
        df_idf['IDF'] = df_idf['IDF'].apply(lambda x: 0 if x == np.inf else x)
        df_idf['TF-IDF'] = df_idf['TF'] * df_idf['IDF']
        st.table(df_idf)
        final_tf_idf = df_idf['TF-IDF'].to_numpy()
        final_tf_idf = final_tf_idf.reshape(1, -1)

        q = np.dot(np.dot(final_tf_idf, matrix_vk_transpose), matrix_sk)

        df_q = pd.DataFrame(q, columns=['Q' + str(i) for i in range(1, num_components+1)])
        st.table(df_q)

        st.write('# Cosine Similarity')


        q
        matrix_p

        result = []


        for i in range(len(matrix_p)):
            result.append(np.dot(matrix_p[i], q.T) / (np.linalg.norm(matrix_p[i]) * np.linalg.norm(q)))
            
        df_result = pd.DataFrame(result, columns=['Cosine Similarity'])
        df_result.sort_values(by='Cosine Similarity', ascending=False, inplace=True)
        df_result

        top_1_index = df_result.index[0]

        nan_value = df_result.loc[top_1_index, 'Cosine Similarity']

        df = pd.read_sql_query("SELECT * FROM faq", conn)
        df = df.drop(['id'], axis=1)
        df.rename(columns={'question': 'Pertanyaan', 'answer': 'Jawaban'}, inplace=True)

        get_pertanyaan = df['Jawaban'][top_1_index]
        if text != '':
            if str(nan_value) != 'nan':
                st.write('Jawaban : ', get_pertanyaan)
            elif str(nan_value) == 'nan':
                st.write('Jawaban : ', 'Maaf, saya tidak mengerti pertanyaan anda')