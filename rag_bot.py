
!huggingface-cli login #login to hugginface

!cat /root/.cache/huggingface/token  #check if your token is stored

from huggingface_hub import whoami
print(whoami())

import os
import time
from IPython.display import display, HTML
def tunnel_prep():
    for f in ('cloudflared-linux-amd64', 'logs.txt', 'nohup.out'):
        try:
            os.remove(f'/content/{f}')
            print(f"Deleted {f}")
        except FileNotFoundError:
            continue

    !wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -q
    !chmod +x cloudflared-linux-amd64
    !nohup /content/cloudflared-linux-amd64 tunnel --url http://localhost:8501 &
    url = ""
    while not url:
        time.sleep(1)
        result = !grep -o 'https://.*\.trycloudflare.com' nohup.out | head -n 1
        if result:
            url = result[0]
    return display(HTML(f'Your tunnel URL <a href="{url}" target="_blank">{url}</a>'))

!pip install -qqq -U langchain-huggingface
!pip install -qqq -U langchain
!pip install -qqq -U langchain-community
!pip install -qqq -U faiss-cpu
!pip install -qqq -U streamlit


%%writefile rag_app.py
import os
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import streamlit as st
import base64
from PIL import Image

#HuggingFace LLM-Endpoint
hf_token = os.getenv("HF_TOKEN")  #load Toekn from your secrets
hf_model = "mistralai/Mistral-7B-Instruct-v0.3"
llm = HuggingFaceEndpoint(
    repo_id=hf_model,
    temperature=0.5,
    top_p=0.9,
    huggingface_api_key=hf_token,
)

# Your Embeddings-Modell
embedding_model = "sentence-transformers/msmarco-MiniLM-L-6-v3"
embeddings_folder = "/content/"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model, cache_folder=embeddings_folder)

#Load FAISS-Index
vector_db = FAISS.load_local("/content/faiss_index", embeddings, allow_dangerous_deserialization=True)

#Prompt Template
input_template = """Answer the question based only on the following context. Keep your answers short and succinct.

Context to answer question:
{context}

Question to be answered: {question}
Response:"""
prompt = PromptTemplate(template=input_template, input_variables=["context", "question"])

#Build RAG-Chain with  RetrievalQA
@st.cache_resource
def init_rag_bot():
    retriever = vector_db.as_retriever(search_kwargs={"k": 5})
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )
    return qa_chain

rag_bot = init_rag_bot()

##### Streamlit App #####

st.title("THE AI ASSISTANT")

#Initialise Chat-Historie
if "messages" not in st.session_state:
    st.session_state.messages = []
#sho chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=message["avatar"]):
        st.markdown(message["content"])
# Define avatars for human and assistant
user_avatar = "🔎"  #Human avatar (can be an emoji, image URL, etc.)
assistant_avatar = "🤖"  #Assistant avatar (can be an emoji, image URL, etc.)

#add the human message with avatar only once (when the user submits)
if user_input := st.chat_input("Ask me anything!"):
    #display human message and append to the history
    st.chat_message("human", avatar=user_avatar).markdown(user_input)
    st.session_state.messages.append({"role": "human", "content": user_input, "avatar": user_avatar})

    with st.spinner("Please wait while I think..."):
        answer = rag_bot({"query": user_input})
        response = answer.get("result", "I couldn't find an answer.")
        sources = []
        if "source_documents" in answer:
            for doc in answer["source_documents"]:
                source_name = doc.metadata.get("source", "Unknown Source")
                sources.append(source_name)
        if sources:
            source_text = "\n".join(f"- {source}" for source in sources)
        else:
            source_text = "No sources found."

        response_with_sources = f"{response}\n\n**Sources:**\n{source_text}"

        st.chat_message("assistant", avatar=assistant_avatar).markdown(response_with_sources)
        st.session_state.messages.append({"role": "assistant", "content": response_with_sources, "avatar": assistant_avatar})




tunnel_prep()

!streamlit run rag_app.py &>/content/logs.txt &

"""# Debugging"""

#OPTIONAL: Debugging (remove or set DEBUG = False for production)
DEBUG = False
if DEBUG:
    st.write("Debugging Chat History:", st.session_state.messages)


"""# Debuggin - uncomment the code below for purpose of debuggin """

#import requests

#headers = {
#    "Authorization": "Bearer <your token>"
#}

#data = {
#    "inputs": "Can you provide a summary of the key benefits of AI in cybersecurity?"
#}

#model_name = "mistralai/Mistral-7B-Instruct-v0.3"
#response = requests.post(f"https://api-inference.huggingface.co/models/{model_name}", headers=headers, json=data)

#print(response.json())
