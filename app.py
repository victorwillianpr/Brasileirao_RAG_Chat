import streamlit as st
import pandas as pd
import requests
import json
import re
from io import StringIO
import sys
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# --- CONFIGURAÇÃO DA API GEMINI ---
API_KEY = "AIzaSyCG28UmTonPzKXHxIn_SEWgAzUcS2F0Gjw"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={API_KEY}"

# --- CARREGAMENTO DOS DADOS ---
@st.cache_data
def carregar_dados():
    """Carrega todos os CSVs em um dicionário de DataFrames."""
    try:
        dfs = {
            "full_df": pd.read_csv('campeonato_brasileiro\campeonato-brasileiro-full.csv'),
            "gols_df": pd.read_csv('campeonato_brasileiro\campeonato-brasileiro-gols.csv'),
            "cartoes_df": pd.read_csv('campeonato_brasileiro\campeonato-brasileiro-cartoes.csv'),
            "stats_df": pd.read_csv('campeonato_brasileiro\campeonato-brasileiro-estatisticas-full.csv')
        }
        dfs['full_df']['data'] = pd.to_datetime(dfs['full_df']['data'], format='%d/%m/%Y')
        dfs['full_df']['mandante_Placar'] = pd.to_numeric(dfs['full_df']['mandante_Placar'], errors='coerce')
        dfs['full_df']['visitante_Placar'] = pd.to_numeric(dfs['full_df']['visitante_Placar'], errors='coerce')
        return dfs
    except FileNotFoundError as e:
        st.error(f"Erro: Arquivo não encontrado - {e.filename}. Certifique-se de que os arquivos .csv estão na pasta correta.")
        return None

# --- ARQUITETURA HÍBRIDA: RAG + TEXT-TO-CODE ---

# 1. CLASSIFICADOR DE INTENÇÃO
def classificar_intencao(pergunta):
    """Classifica a pergunta como 'analitica' ou 'busca_simples'."""
    prompt = f"""
    Analise a seguinte pergunta do usuário e classifique sua intenção.
    Responda apenas com 'analitica' ou 'busca_simples'.

    - 'analitica': Perguntas que requerem cálculos, agregações, comparações, ou a busca pelo "maior", "menor", "mais", "menos", etc. Exemplos: "qual time fez mais gols em 2019?", "quem levou mais cartões vermelhos?".
    - 'busca_simples': Perguntas que buscam um fato específico sobre um ou poucos jogos. Exemplos: "qual foi o placar do jogo entre Palmeiras e Corinthians em 2022?".

    Pergunta do usuário: "{pergunta}"
    Intenção:
    """
    try:
        response_text = fazer_chamada_llm(prompt)
        if 'analitica' in response_text.lower():
            return 'analitica'
        return 'busca_simples'
    except Exception:
        return 'busca_simples'

# 2. GERAÇÃO DE CÓDIGO PANDAS (CAMINHO ANALÍTICO)
def gerar_codigo_pandas(pergunta, dfs):
    """Gera código Python/Pandas para responder a uma pergunta analítica."""
    schema_info = ""
    for name, df in dfs.items():
        schema_info += f"DataFrame '{name}':\nColunas: {list(df.columns)}\n\n"

    prompt = f"""
    Você é um assistente de análise de dados especialista em Python e Pandas. Sua tarefa é gerar um código Python para responder à pergunta do usuário usando os DataFrames disponíveis.

    DataFrames disponíveis e seus esquemas:
    {schema_info}

    Instruções CRÍTICAS:
    1. O código deve ser completo, autônomo e NÃO PODE conter dados de exemplo ou placeholders.
    2. Use os nomes dos DataFrames exatamente como fornecidos (full_df, gols_df, etc.).
    3. O resultado final da sua análise DEVE ser impresso com a função `print()`. A impressão deve ser clara e conter os dados reais encontrados.
    4. NÃO invente dados. A sua resposta DEVE ser baseada nos DataFrames fornecidos.
    5. NÃO gere nenhum texto explicativo, apenas o bloco de código Python.
    6. A coluna 'data' no 'full_df' é do tipo datetime. Você pode usar `df['data'].dt.year`.

    Agora, gere o código para a seguinte pergunta.

    Pergunta do usuário: "{pergunta}"

    Código Python:
    ```python
    # Seu código aqui
    ```
    """
    response_text = fazer_chamada_llm(prompt)
    match = re.search(r'```python\n(.*?)\n```', response_text, re.DOTALL)
    if match:
        return match.group(1)
    return "print('Não foi possível gerar um código para esta pergunta. Tente reformular a questão.')"

def executar_codigo(codigo, dfs):
    """Executa o código gerado de forma segura e captura o resultado."""
    buffer = StringIO()
    original_stdout = sys.stdout
    sys.stdout = buffer

    local_scope = {**dfs, 'pd': pd}
    
    try:
        exec(codigo, local_scope)
        sys.stdout = original_stdout
        resultado = buffer.getvalue()
        if not resultado:
            return "O código foi executado com sucesso, mas não produziu nenhuma saída para ser exibida."
        return resultado
    except Exception as e:
        sys.stdout = original_stdout
        return f"Erro ao executar o código: {e}\n\nCódigo com problema:\n{codigo}"


# 3. RAG AVANÇADO COM BUSCA SEMÂNTICA (CAMINHO DE BUSCA SIMPLES)
@st.cache_resource
def setup_rag_pipeline(_dfs):
    """
    Cria a base de conhecimento, gera embeddings e constrói um índice FAISS.
    Esta função é decorada com @st.cache_resource para que seja executada apenas uma vez.
    """
    corpus = []
    # Usando um subconjunto maior para uma base de conhecimento mais rica
    full_df = _dfs['full_df'].head(4000).dropna(subset=['mandante_Placar', 'visitante_Placar'])
    
    for _, partida in full_df.iterrows():
        # Cria um "documento" mais rico para cada partida
        info = (f"Partida em {partida['data'].strftime('%d de %B de %Y')}: "
                f"{partida['mandante']} enfrentou {partida['visitante']} no estádio {partida['arena']}. "
                f"O resultado foi {int(partida['mandante_Placar'])} a {int(partida['visitante_Placar'])}. "
                f"O vencedor foi {partida['vencedor'] if partida['vencedor'] != '-' else 'empate'}.")
        corpus.append(info)

    # Carrega um modelo de embedding pré-treinado. 
    # 'paraphrase-multilingual-MiniLM-L12-v2' é bom para português.
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    
    # Converte o corpus em embeddings
    embeddings = model.encode(corpus, show_progress_bar=True)
    
    # Cria um índice FAISS para busca rápida
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))

    return model, index, corpus

def recuperar_contexto_semantico(pergunta, model, index, corpus, max_resultados=3):
    """Realiza uma busca semântica para encontrar os documentos mais relevantes."""
    query_embedding = model.encode([pergunta])
    _, indices = index.search(np.array(query_embedding), max_resultados)
    
    # Recupera os documentos do corpus com base nos índices encontrados
    contexto = [corpus[i] for i in indices[0]]
    return "\n---\n".join(contexto)

# 4. FUNÇÃO DE GERAÇÃO FINAL
def gerar_resposta_final(pergunta, contexto):
    prompt = f"""
    Você é um assistente de futebol. Com base no CONTEXTO abaixo, forneça uma resposta clara e amigável em português para a PERGUNTA do usuário.
    Se o contexto for o resultado de uma análise de dados, formate-o de maneira agradável.
    Se o contexto não tiver a informação, admita que não encontrou.

    CONTEXTO:
    {contexto}

    PERGUNTA:
    {pergunta}
    """
    return fazer_chamada_llm(prompt)

# FUNÇÃO AUXILIAR PARA CHAMADA DE API
def fazer_chamada_llm(prompt):
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    headers = {'Content-Type': 'application/json'}
    try:
        response = requests.post(API_URL, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        result = response.json()
        return result['candidates'][0]['content']['parts'][0]['text']
    except Exception as e:
        return f"Erro na comunicação com a API: {e}"

# --- INTERFACE COM STREAMLIT ---
st.set_page_config(page_title="RAG Brasileirão ", page_icon="⚽") 
st.title("⚽ Chatbot RAG Brasileirão")
st.caption("Dados sobre o Campeonato Brasileiro de 2003 até 2024")

dataframes = carregar_dados()

if dataframes:
    # Prepara o pipeline de RAG avançado (será executado apenas uma vez)
    rag_model, rag_index, rag_corpus = setup_rag_pipeline(dataframes)

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Olá! Sou um especialista nos dados do Brasileirão. O que você gostaria de saber?"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Escreva sua pergunta"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analisando..."):
                intencao = classificar_intencao(prompt)
            
            if intencao == 'analitica':
                with st.spinner("Analisando..."):
                    codigo_gerado = gerar_codigo_pandas(prompt, dataframes)
                
                with st.spinner("Pensando..."):
                    resultado_codigo = executar_codigo(codigo_gerado, dataframes)
                
                with st.spinner("Resposta final..."):
                    contexto_final = f"O resultado da análise de dados para a pergunta foi: {resultado_codigo}"
                    resposta_final = gerar_resposta_final(prompt, contexto_final)

            else: # busca_simples
                st.info(f"Buscando melhor resposta...")
                with st.spinner("Buscando informações..."):
                    contexto_rag = recuperar_contexto_semantico(prompt, rag_model, rag_index, rag_corpus)
                    if not contexto_rag:
                        resposta_final = "Não encontrei informações sobre isso nos meus registos. Tente ser mais específico."
                    else:
                        resposta_final = gerar_resposta_final(prompt, contexto_rag)

            st.markdown(resposta_final)
            st.session_state.messages.append({"role": "assistant", "content": resposta_final})

