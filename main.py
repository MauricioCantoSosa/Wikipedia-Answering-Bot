import streamlit as st
from streamlit_chat import message
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.docstore.document import Document
import requests
import pinecone
import os

os.environ['OPENAI_API_KEY'] = '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'
os.environ['PINECONE_API_KEY'] = 'cdc32a3a-e280-4633-80e8-6556bdcab127'
os.environ['PINECONE_API_ENV'] = 'us-central1-gcp'

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_API_ENV = os.environ.get("PINECONE_API_ENV")

texts= ''

# Downloads the text data from wikipedia articles
def get_wiki_data(title, first_paragraph_only):
    title = title.rstrip()
    title = title.replace('\r', '').replace('\n', '')
    url = f"https://en.wikipedia.org/w/api.php?format=json&action=query&prop=extracts&explaintext=1&titles={title}"
    if first_paragraph_only:
        url += "&exintro=1"
    data = requests.get(url).json()
    return Document(
        page_content=list(data["query"]["pages"].values())[0]["extract"],
        metadata={"source": f"https://en.wikipedia.org/wiki/{title}"},
    )

# Document list of wikipedia articles to feed the model
def wikipedia_topinecone(wiki_article):
    sources = [
        get_wiki_data(wiki_article, False),
    ]

    # Chunking data into smaller documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(sources)
    return texts

# Gets answer from LLM
def get_answer(query, OPENAI_API_KEY, texts=''):
    docsearch = Pinecone.from_texts([t.page_content for t in texts], embeddings, index_name=index_name)
    llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
    chain = load_qa_chain(llm, chain_type="stuff")

    docs = docsearch.similarity_search(query, include_metadata=True)
    return(
        chain.run(
                input_documents = docs,
                question = query)
    )

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# initialize pinecone
pinecone.init(
    api_key=PINECONE_API_KEY,  
    environment=PINECONE_API_ENV
)
index_name = "langchain-customknowledge"









st.set_page_config(page_title='Custom Knowledge Chat-GPT', page_icon=':robot:')
st.header('Ask Wikipedia')

col1, col2 = st.columns(2)

with col1:
    st.write('''Reading a whole Wikipedia article looking for a simple answer can be a daunting task. Which is the reason this bot was created. Copy-Paste the title from any Wikipedia
                article of your choice in the textbox to the right and press CTRL+Enter. Afterwards, ask the bot below any question or query from the wikipedia article you chose.''')

with col2:
    wiki_article = st.text_input(label='Wikipedia Article Title', placeholder='Write the Wikipedia Article Title Here', key='wiki_article')
    if wiki_article:
        texts = wikipedia_topinecone(wiki_article)
        st.write('Information Retrieved')

st.markdown('## Chat with it')

def get_query():
    question = st.text_input(label='', placeholder='Write Here', key='question')
    return question

question = get_query()

if "history" not in st.session_state:
    st.session_state.history = []

if question:
    #message(question, is_user=True)
    st.session_state.history.append({"message": question, "is_user": True})

    answer = get_answer(question, OPENAI_API_KEY, texts)
    st.session_state.history.append({"message": answer, "is_user": False})

for i, chat in enumerate(st.session_state.history):
    message(**chat, key=str(i)) #unpacking