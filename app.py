import fitz  # PyMuPDF
import re
import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
load_dotenv()

os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")
os.environ['HF_TOKEN']=os.getenv("HF_TOKEN")
embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
groq_api_key =os.getenv("GROQ_API_KEY")

loader = PyMuPDFLoader("Sample.pdf")
data = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
splits = text_splitter.split_documents(data)
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

retriever = vectorstore.as_retriever()
llm= llm = ChatGroq(groq_api_key=groq_api_key,
    model="Llama-3.1-8b-instant",
    temperature=0,
)


system_prompt = (
    "You are an assistant for checking details in the invoice. "
    "the invoice details have been provided to you as below "
    "Given the invoice data , you have to provide the details as asked by the user"
    "Make sure to provide only the required details , and avoid providing any extra details"
    "Make sure the deatils provided are consistent with the invoice data, and concrete as well"
    
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

customer_details=rag_chain.invoke({"input" : "Provide only the  details of the customer for the invoice"})
product_details=rag_chain.invoke({"input" : "Provide the details of each product purchased by the customer , for the  given invoice and do not provide any other information."})
total_amount=rag_chain.invoke({"input" : "Provide just the total amount for the invoice,in number format.The output should be like /'total amount:x'/"})
l=[customer_details['answer'],product_details['answer'],total_amount['answer']]
for i in l:
    print(i)
    print("\n")

