# utils.py
# Importing libs/ importando libs
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import warnings
import pickle
import streamlit as st
import hashlib
import fitz
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import os
from datetime import datetime
from google.oauth2 import service_account
from google.cloud import bigquery
import pandas as pd
from sentence_transformers import SentenceTransformer
from sentence_transformers import util as util_trans
import json
import numpy as np
import joblib
import torch
import requests
import zipfile
import io
warnings.simplefilter("ignore")

# Load applicants database once
with open("applicants.json", "r", encoding="utf-8") as f:
    applicants_dict = json.load(f)

from pathlib import Path

model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

df_data = pd.read_pickle('app_embs')
df = torch.from_numpy(df_data.values)
df = df.float()

def extract_job_text(job_json: dict) -> str:
    job_text = (
        job_json.get("perfil_vaga", {}).get("principais_atividades", "") + " " +
        " ".join(job_json.get("perfil_vaga", {}).get("competencia_tecnicas_e_comportamentais", []))
    )

    job_text = job_text.strip()

    job_text = preprocess(job_text)

    return job_text

def extract_applicant_skills(applicant):

    texts = []
    ids = []
    for app_id, data in applicants.items():
        cv = data.get("cv_pt", "").strip()
        skills = data["informacoes_profissionais"].get("conhecimentos_tecnicos", "")
        info_app = skills + " " + cv.lower()
        info_app = preprocess(info_app)
        if cv:
            ids.append(app_id)
            texts.append(info_app)

    return ids, texts


def recommend_cvs_for_job(job_json: str, top_k: int = 5):
   
    job_text = extract_job_text(job_json)
    job_embedding = model.encode(job_text)
    applicant_ids: list = joblib.load("applicant_ids.pkl")
    applicant_embeddings = df
    
    print(applicant_embeddings.dtype, job_embedding.dtype)
    similarities = util_trans.cos_sim(job_embedding, applicant_embeddings)[0]
    top_results = np.argsort(-similarities)[:top_k]
    print(f"topresults {top_results}")
    app_ids = applicant_ids[0].values.tolist()
    print(f"\nTop {top_k} candidatos mais recomendados para esta vaga:\n")
    for idx in top_results:
        app_id = app_ids[idx]
        applicant = applicants_dict.get(app_id)

        if not applicant:
            print(f"⚠️ Applicant ID {app_id} not found.")
            return

        basic = applicant.get("infos_basicas", {})
        personal = applicant.get("informacoes_pessoais", {})
        prof = applicant.get("informacoes_profissionais", {})
        education = applicant.get("formacao_e_idiomas", {})

        name = basic.get("nome") or personal.get("nome", "Nome não disponível")
        location = basic.get("local", "Local não informado")
        title = prof.get("titulo_profissional", "Título profissional não informado")
        academic = education.get("nivel_academico", "Nível acadêmico não informado")
        english = education.get("nivel_ingles", "Inglês não informado")
        cv = applicant.get("cv_pt", "")

        st.subheader(f"Nome: {name} (ID: {app_id})", )
        st.write(f"Score de similaridade: {similarities[idx]:.4f}")
        st.write(f"Localização: {location}")
        st.write(f"Título profissional: {title}")
        st.write(f"Formação: {academic}")
        st.write(f"Inglês: {english}")
        st.write("Currículo (trecho):")
        st.write(cv[:500] + "..." if len(cv) > 500 else cv)

        if st.button("Ver mais informações", key=f"ver_{app_id}"):
            st.session_state.candidato_selecionado = app_id
            st.switch_page("pages/Detalhes_Candidato.py")
        
        st.write("--------------------------------------------------\n")


with open("vagas.pkl", "rb") as f:
    jobs = pickle.load(f)

logreg = joblib.load("logistic_model.pkl")
xgb = joblib.load("xgboost_model.pkl")
embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

job_data = joblib.load("job_data.pkl")
job_ids = job_data["job_ids"]
job_titles = job_data["job_titles"]
job_embeddings = job_data["job_embeddings"]

def preprocess(text):
    import nltk
    nltk.download('stopwords')
    nltk.download('punkt_tab')
    import re, string
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    
    stop_words = set(stopwords.words('portuguese'))
    
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text, language="portuguese")
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    return ' '.join(tokens)

def predict_jobs_for_cv(cv_text, top_n=5):

    cleaned_cv = preprocess(cv_text)

    cv_vec = embedding_model.encode([cleaned_cv])

    sims = cosine_similarity(cv_vec, job_embeddings).flatten()

    results = []
    for i, sim in enumerate(sims):
        logreg_prob = logreg.predict_proba([[sim]])[0][1]
        xgb_prob = xgb.predict_proba([[sim]])[0][1]
        ensemble_prob = (logreg_prob + xgb_prob) / 2

        job = jobs.get(job_ids[i], {})
        title = job.get("informacoes_basicas", {}).get("titulo_vaga", "N/A")
        area = job.get("perfil_vaga", {}).get("areas_atuacao", "N/A")
        skills = job.get("perfil_vaga", {}).get("competencia_tecnicas_e_comportamentais", "")
        activities = job.get("perfil_vaga", {}).get("principais_atividades", "")

        results.append({
            "job_id": job_ids[i],
            "title": title,
            "area": area,
            "skills": skills,
            "activities": activities,
            "similarity": sim,
            "hire_prob": ensemble_prob
        })

    with st.container(border=True):

        top_jobs = sorted(results, key=lambda x: x["hire_prob"], reverse=True)[:top_n]

        for idx, job in enumerate(top_jobs, 1):

            st.write(f"\n🔹 Recomendação #{idx}")
            if st.session_state.user_type == "candidato":
                from utils import CANDIDATURAS_DB

                ja_candidatou_real = any(
                    c["id_vaga"] == job['job_id'] and c["username_candidato"] == st.session_state.username
                    for c in CANDIDATURAS_DB
                )

                if ja_candidatou_real:
                    st.success("Você já se candidatou!")
                elif st.button("Candidatar-se", key=f"apply_{job['job_id']}", use_container_width=True):
                    success, message = register_prospect(job['job_id'], st.session_state.username)
                    if success:
                        st.switch_page("pages/Minhas_Candidaturas.py")
                        st.success(message)
                        st.rerun()
                        
                    else:
                        st.error(message)
            st.write(f"Título da vaga: {job['title']}")
            st.write(f"Área: {preprocess(job['area'])}")
            st.write(f"Score de similaridade: {job['similarity']:.2f}")
            st.write(f"Probabilidade de contratação: {job['hire_prob']:.2%}")
            st.write(f"Competências: {preprocess(job['skills'][:200])}...")
            st.write(f"Atividades: {preprocess(job['activities'][:200])}...")
            st.write("-" * 80)

USUARIOS_DB = {
    "admin": {
        "senha_hash": hashlib.sha256("admin".encode()).hexdigest(),
        "tipo": "admin",
        "nome": "Administrador",
        "curriculo": None
    },
    "candidato1": {
        "senha_hash": hashlib.sha256("candidato123".encode()).hexdigest(),
        "tipo": "candidato",
        "nome": "José da Silva",
        "curriculo": "assistente administrativo\n\n\nsantosbatista\nitapecerica da serra/sp\n29 anos ▪ brasileiro ▪ casado\nformação acadêmica\n bacharel - ciências contábeis\ncentro universitário ítalo brasileiro\njul/2015 - dez/2018\n graduação - gestão financeira\ncentro universitário anhanguera\njan/2013 - dez/2014\nhabilidades\n contas a pagar e receber\n excel avançado\n indicadores kpi’s\n notas fiscais, cfop’s\n fechamento contábil\n emissão de boletos\n guias\n impostos\n budget\n controladoria\n sistemas integrados:\ntotvs;\nfolha matic;\nnavision\nresumo profissional\nprofissional com experiência nos departamentos financeiro,\ncontábil, fiscal e controladoria jurídica. elaboração e análise de\nindicadores kpi’s de resultado, relatórios, guias, gestão de\npagamentos, notas fiscais, boletos, fechamento financeiro e\ncontábil fiscal.\nsoftwares erp protheus, folha matic, navision, elaw e sapiens,\nexcel avançado, (kpi's, painéis de dashboard e automatização).\nhistórico profissional\n 01/2021 – 07/2021 fcn contabilidade freight forwarder\n\nassistente contábil\nconciliações contábeis, financeira, folha de pagamento,\nfiscal, lançamentos contábeis, exportações txt, análise e\nelaboração de relatórios, fechamento contábil, análise\nfiscal e contabilização de folha de pagamento, sistema\nfolha matic.\n 10/2020 – 01/2021 almeida advogados\nassistente financeiro\ngestão de pagamentos, baixa de boletos, relatórios gerenciais.\n 04/2019 – 06/2019 fedex brasil logistica e transporte ltda\nassistente juridico\nresponsável pelo fechamento mensal através das\napurações de provisões e reclassificações contábeis,\nelaboração de indicadores financeiros e desempenho,\nautomatização de planilhas, análise de budget e real vs\norçado.\n 07/2017 – 11/2018 atonanni construções e serviços ltda\nassistente contábil / fiscal\nlançamento de notas fiscais, apurações dos impostos (iss,\npis, cofins, cprb, ir, csll).\nguias de pagamentos, sped fiscais, relatórios, xml, cfop,\nncm.\n 06/2014 – 07/2017 iss servisytem do brasil ltda\nassistente de controladoria\ncontas a pagar e a receber, análises contábeis e\nfinanceiras, reembolsos, p.o’s.\ngestão de custos, budget, real vs orçado, indicadores, kpi’s\ne mapeamento de melhorias.\n 04/2013 – 06/2014 n & n comércio de alimentos ltda\nassistente financeiro\ncontas a pagar e a receber, boletos, relatórios gerenciais.\nbaixa de notas fiscais, concilação financeira, negociações\nde pagamentos\n"
    },
    "candidato2": {
        "senha_hash": hashlib.sha256("candidato234".encode()).hexdigest(),
        "tipo": "candidato",
        "nome": "Maria dos Santos",
        "curriculo": "formação acadêmica\nensino médio (2º grau) em ensino médio (2º grau), beatriz lopes em sp\njan. 2010 até dez. 2012\nensino superior em administração de empresas, unip em sp\njun. 2016 - trancado\nexperiência profissional\nanalista administrativo de operações, liq em são paulo - sp\nmai. 2018 até o momento\n\nadministração - administração geral (analista)\n\nauxiliar na área de bi (business intelligence). extração de informação e análise de relatórios gerenciais, acompanhamento dos processos para as áreas financeiras,rh,operacional. suporte a toda área de backoffice, suporte a todos os supervisores. criação de indicadores (dashboard) pelo excel. analista de operações em trade com todo suporte a equipe de supervisores e gerentes.\n\nestagiaria, ballmash modas e confecções ltda eireli epp em sp\njan. 2017 até nov. 2017\n\nadministração - administração geral (estagiário)\n\natividades: administração geral. auxilio na conferencia do caixa e controle de vendas no cartão, lançamentos de dados em planilha, contas a pagar, emissão e lançamento de notas fiscal. auxiliar no arquivo de documentos, atendimento telefônico e anotações, atendimento ao público.\n\noperador de caixa, parque da mônica em sp\nout. 2015 até jul. 2016\n\ncomercial, vendas - atendimento (operacional)\n\noperação de pdv atendimento ao publico .\n\noperadora de teleatendimento, rede bem estar em sp\nfev. 2015 até jul. 2015\n\ntelemarketing - telemarketing / call center ativo (operacional)\n\nagendamentos e vendas .\n\nvendas atendente, cinemark brasil em sp\njan. 2012 até fev. 2014\n\ncultura, lazer, entretenimento - entretenimento (operacional)\n\natendimento e recepção ao cliente em todos os setores .matemática comercial transição de cartão seja ele de débito ou crédito até mesmo título de crédito dentre outras atribuições. promover vendas de produtos e serviços em comunicação direta ao cliente.\n\ninformática:\nbanco de dados: caché\nprogramação: html\naplicações de escritório: microsoft access, microsoft excel, microsoft outlook, microsoft powerpoint, microsoft word, open office\nsistemas operacionais: windows, linux\noutros programas: edição de som, edição de video\n"
    }
}

CURRICULOS_DIR = "curriculos_uploaded"
if not os.path.exists(CURRICULOS_DIR):
    os.makedirs(CURRICULOS_DIR)

VAGAS_DB = jobs

CANDIDATURAS_DB = []

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(plain_password, hashed_password):
    return hash_password(plain_password) == hashed_password

def check_login(username, password):
    if username in USUARIOS_DB and verify_password(password, USUARIOS_DB[username]["senha_hash"]):
        return True, USUARIOS_DB[username]["tipo"], USUARIOS_DB[username]["nome"]
    return False, None, None

def extract_text_from_pdf(file_path):
    text = ""
    with fitz.open(file_path) as doc:
        for page in doc:
            text += page.get_text()
    return text


def register_candidate(nome_completo, username, password, uploaded_file):
    if username in USUARIOS_DB:
        return False, "Nome de usuário já existe."
    
    curriculo_filename = None
    if uploaded_file is not None:
        curriculo_filename = os.path.join(CURRICULOS_DIR, f"{username}_{uploaded_file.name}")
        try:
            with open(curriculo_filename, "wb") as f:
                f.write(uploaded_file.getbuffer())
        except Exception as e:
            return False, f"Erro ao salvar currículo: {e}"

        cv_text = extract_text_from_pdf(curriculo_filename)

    USUARIOS_DB[username] = {
        "senha_hash": hash_password(password),
        "tipo": "candidato",
        "nome": nome_completo,
        "curriculo": cv_text
    }
    return True, "Cadastro realizado com sucesso! Faça login para continuar."


def hide_streamlit_sidebar_css():
    ...

def load_css(file_name="style.css"):
    try:
        with open(file_name, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"Arquivo CSS \'{file_name}\' não encontrado.")

def create_job(titulo, empresa, local, descricao, requisitos, tipo_contrato, salario):
    id_vaga =  len(VAGAS_DB) + 1
    nova_vaga = {
        "informacoes_basicas": {
            "data_requicisao": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "limite_esperado_para_contratacao": None,
            "titulo_vaga": titulo,
            "vaga_sap": None,
            "cliente": empresa,
            "solicitante_cliente": None,
            "empresa_divisao": "Decision São Paulo",
            "requisitante": None,
            "analista_responsavel": None,
            "tipo_contratacao": tipo_contrato,
            "prazo_contratacao": None,
            "data_inicial": None,
            "data_final": None,
            "objetivo_vaga": "Contratação",
            "prioridade_vaga": None,
            "origem_vaga": None,
            "superior_imediato": None
        },
        "perfil_vaga": {
            "pais": None,
            "estado": None,
            "cidade": None,
            "bairro": "",
            "regiao": "",
            "local_trabalho": local,
            "vaga_especifica_para_pcd": None,
            "faixa_etaria": salario,
            "horario_trabalho": None,
            "nivel profissional": None,
            "nivel_academico": None,
            "nivel_ingles": None,
            "nivel_espanhol": None,
            "outro_idioma": None,
            "areas_atuacao": None,
            "principais_atividades": descricao,
            "competencia_tecnicas_e_comportamentais": requisitos,
            "demais_observacoes": None,
            "viagens_requeridas": None,
            "equipamentos_necessarios": None
        },
        "beneficios": {
            "valor_venda": None,
            "valor_compra_1": None,
            "valor_compra_2": None
        }
    }
    VAGAS_DB[id_vaga] = nova_vaga
    return id_vaga

def list_jobs():
    return VAGAS_DB

def list_job_by_id(job_id):
    for vaga_id, vaga in VAGAS_DB.items():
        
        if vaga_id == job_id:
            return vaga
    return None

def register_prospect(id_vaga, username_candidato):
    
    if any(c["id_vaga"] == id_vaga and c["username_candidato"] == username_candidato for c in CANDIDATURAS_DB):
        return False, "Você já se candidatou para esta vaga."
    
    nova_candidatura = {
        "id_candidatura": f"cand{len(CANDIDATURAS_DB) + 1}",
        "id_vaga": id_vaga,
        "username_candidato": username_candidato,
        "status": "Em Análise",
        "data_candidatura": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    CANDIDATURAS_DB.append(nova_candidatura)
    return True, "Candidatura registrada com sucesso!"

def listar_candidaturas_por_usuario(username_candidato):
    return [c for c in CANDIDATURAS_DB if c["username_candidato"] == username_candidato]

def get_vaga_titulo(id_vaga):
    vaga = list_job_by_id(id_vaga)
    info = vaga["informacoes_basicas"]

    return info.get("titulo_vaga", "Título não informado")


