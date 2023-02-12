import streamlit as st

from streamlit_option_menu import option_menu


    
def admin():
    hide_streamlit_style = """
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                .css-j7qwjs {content-visibility: visible !important;}
                </style>
                """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
    st.success("Anda Login Sebagai Admin")
    st.title('QAS (Question Answer System) Chatbot')
    st.write('Selamat datang di halaman tentang aplikasi QAS (Question Answer System) Chatbot yang memadukan metode Cosine dan LSI. Aplikasi ini dirancang untuk membantu pengguna menemukan jawaban atas pertanyaan mereka dengan lebih cepat dan akurat.')
    st.write('Metode Cosine digunakan untuk menentukan kemiripan antara pertanyaan pengguna dan dokumen sumber yang tersedia. Sedangkan metode LSI (Latent Semantic Indexing) digunakan untuk menganalisis konteks dari pertanyaan dan dokumen sumber untuk menemukan jawaban yang relevan.')
    st.write('Berkat kombinasi kedua metode ini, aplikasi QAS Chatbot mampu memberikan jawaban yang tepat dan akurat pada pertanyaan pengguna. Dengan ini, pengguna dapat menemukan informasi yang mereka butuhkan dengan lebih cepat dan efisien.')

            
    
    