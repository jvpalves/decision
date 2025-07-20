import streamlit as st
import pandas as pd
from utils import VAGAS_DB, recommend_cvs_for_job

st.set_page_config(page_title="Cvs - Decision", layout="wide")

if not st.session_state.get("logged_in", False) or st.session_state.get("user_type") != "admin":
    st.error("Acesso negado. Você precisa estar logado como administrador para acessar esta página.")
    if st.button("Ir para Home"):
        st.switch_page("Home.py")
    st.stop()


col_nav_home, col_nav_sobre, col_nav_spacer_cvs = st.columns([1, 1, 8])
with col_nav_home:
    if st.button("Home", key="home_cvs"):
        st.switch_page('Home.py')

st.divider()

st.title("Recomendação de Cvs")

if "nova_vaga_id" in st.session_state:
    recommend_cvs_for_job(VAGAS_DB[st.session_state.nova_vaga_id])

else:
    st.warning("Nada a mostrar")


