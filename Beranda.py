import streamlit as st
import streamlit_authenticator as stauth
from admin import admin
from user import user

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            .css-j7qwjs {content-visibility: hidden !important;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

def login():
    names = ['Admin','User']
    usernames = ['admin','user']
    passwords = ['admin123','user123']
    hashed_passwords = stauth.Hasher(passwords).generate()
    authenticator = stauth.Authenticate(names,usernames,hashed_passwords,'some_cookie_name','some_signature_key',cookie_expiry_days=30)
    name, authentication_status, username = authenticator.login('Login', 'main')
        
    if st.session_state["authentication_status"]:
        if st.session_state["name"] == 'Admin':
            test=authenticator.logout('Logout', 'sidebar')
            admin()
        elif st.session_state["name"] == 'User':
            st.sidebar.success('Anda Login Sebagai User')
            test=authenticator.logout('Logout', 'sidebar')
            user()
    elif st.session_state["authentication_status"] == False:
        st.sidebar.error('Username/password is incorrect')
    elif st.session_state["authentication_status"] == None:
        st.sidebar.warning('Username/password is incorrect')
        
def main():
    login()

if __name__ == "__main__":
    main()