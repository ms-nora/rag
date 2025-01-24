# Install libraries
"""

# Commented out IPython magic to ensure Python compatibility.
# %%bash
# pip install -qqq -U langchain-huggingface
# pip install -qqq -U langchain
# pip install -qqq -U langchain-community
# pip install -qqq -U faiss-cpu

import os
from google.colab import userdata #if token is stored in secrets

os.environ["HUGGINGFACEHUB_API_TOKEN"] = userdata.get('HF_TOKEN')

from langchain_huggingface import HuggingFaceEndpoint
hf_model = "mistralai/Mistral-7B-Instruct-v0.3"

llm = HuggingFaceEndpoint(repo_id = hf_model)

from google.colab import drive
drive.mount('/content/drive') #here you might have stored your books
import os
from langchain.document_loaders import TextLoader

folder_path = '/content/drive/MyDrive/books/' #get your books from here
documents = []
for filename in os.listdir(folder_path):
    if filename.endswith('.txt'):
        file_path = os.path.join(folder_path, filename)
        loader = TextLoader(file_path)
        docs = loader.load()
        documents.extend(docs)

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

#Initialize the text splitter with your desired chunk size and overlap
text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)

#Split the documents into smaller chunks
docs = text_splitter.split_documents(documents)

#Initialize the embeddings model (for example, sentence-transformers)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/msmarco-MiniLM-L-6-v3")

#Create the FAISS index from the chunked documents
vector_db = FAISS.from_documents(docs, embeddings)

vector_db.save_local("/content/faiss_index")

test_text = "Whhat is a bug"
query_result = embeddings.embed_query(test_text)
query_result




"""# Debuggin - (uncomment the code below for debuggin)"""

"""# check the dimensions"""

#characters = len(test_text)
#dimensions = len(query_result)
#print(f"The {characters} character sentence was transformed into a {dimensions} dimension vector")

#Similarity search with FAISS
#query = "What is ethical hacking?"
#results = vector_db.similarity_search(query, k=2)

#for result in results:
    print(result)

"""# Test the model """

#from langchain.prompts.prompt import PromptTemplate

#input_template = """Answer the question based only on the following context. Keep your answers short and succinct.

#Context to answer question:
#{context}

#Question to be answered: {question}
#Response:"""


#prompt = PromptTemplate(template=input_template,
                        input_variables=["context", "question"])

#from langchain.chains import RetrievalQA

#qa_chain = RetrievalQA.from_chain_type(
#    llm = llm,
#   retriever = vector_db.as_retriever(search_kwargs={"k": 2}), #top 2 results only, speed things up
#   return_source_documents = True,
#   chain_type_kwargs = {"prompt": prompt},
#)

#answer = qa_chain.invoke("What is a bug?")

#answer

"""# Test if the model is using sources"""

#answer.keys()
#answer['source_documents']
#answer['source_documents'][0].page_content
#print(answer['source_documents'][0].page_content)

#answer['source_documents'][0].metadata