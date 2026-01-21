import os 
from dotenv import load_dotenv
from langchain_coummunity.document_loaders import TextLoader
from langchain_text_splitter import RecursiveCharacterTextSpliter
from langchain_huggingface import HuggingfaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_gemini import ChatGemini
from langchain_deepseek import ChatDeepSeek

from langchain_openai import ChatOpenAI
from langchain_core.chains import RetrivalQA

markdown_path = "../../data/C1/markdown/easy-rl-chapter1.md"
loader = TextLoader(markdown_path)
docs = loader.load()

path = "../data/chunked_docs"
loader = TextLoader(path)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter()
texts = text_splitter.split_documents(docs)

embeddings = HuggingfaceEmbeddings(
    model_name= "   sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
    encode_kwargs={"normalize_embeddings":True}
)