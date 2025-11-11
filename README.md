Chatbot RAG do Campeonato Brasileiro
Este é um projeto simples em Python que utiliza a arquitetura RAG (Retrieval-Augmented Generation) para criar um chatbot. A interface é construída com Streamlit e o chatbot responde a perguntas sobre dados do Campeonato Brasileiro de futebol.

Funcionalidades
Interface de Chat: Uma interface web simples e interativa para conversar com o assistente de IA.

Carregamento de Dados: O sistema carrega e processa múltiplos arquivos CSV contendo estatísticas, jogos, gols e cartões do campeonato.

Arquitetura RAG:

Recuperação (Retrieval): Quando o usuário faz uma pergunta, o sistema busca na base de dados processada os trechos mais relevantes.

Aumento (Augmentation): Os trechos encontrados são combinados com a pergunta original.

Geração (Generation): O novo prompt (pergunta + contexto) é enviado a um modelo de linguagem (LLM) para gerar uma resposta fundamentada nos dados.

Como Executar o Projeto
Pré-requisitos
Python 3.8 ou superior instalado.

Os seguintes arquivos CSV devem estar na mesma pasta que o app.py:

campeonato-brasileiro-full.csv

campeonato-brasileiro-gols.csv

campeonato-brasileiro-cartoes.csv

campeonato-brasileiro-estatisticas-full.csv

Passos para Instalação
Crie um Ambiente Virtual (Recomendado):

python -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate

Instale as Dependências:
Com seu ambiente virtual ativado, instale as bibliotecas necessárias a partir do arquivo requirements.txt.

pip install -r requirements.txt

Execute o Aplicativo Streamlit:
No terminal, execute o seguinte comando:

streamlit run app.py

Acesse o Chatbot:
Abra seu navegador e acesse o endereço IP local fornecido pelo Streamlit (geralmente http://localhost:8501).

Agora você pode interagir com o chatbot e fazer perguntas sobre o Brasileirão!
