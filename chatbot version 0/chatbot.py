# Import necessary packages
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM, OllamaEmbeddings
# Document processing and retrieval  
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Splits text into smaller chunks for better embedding and retrieval
from langchain.schema import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import re  # Provides tools for working with regular expressions, useful for text cleaning and pattern matching
import json
import os
from langdetect import detect
def load_data(json_file):
    with open(json_file, 'r', encoding='utf-8') as file:
        json_data = json.load(file)
        # Convert JSON into LangChain Documents
        data = [
            Document(
                page_content=f"Section: {item['section']}\n"
                            f"Question: {item['question']}\n"
                            f"Subtopic: {item['subtopic']}\n"
                            f"Answer: {' '.join(item['answer'])}",  # Keeps all answer elements together
                metadata={"section": item['section'], "subtopic": item['subtopic'], "question": item["question"]}
            )
            for item in json_data
        ]
    return data

# Language Detection Function
def detect_language(text):
    try:
        lang = detect(text)
        print(f"Detected language: {lang}")
        return lang if lang in ["en", "fr", "ar"] else "None"
    except Exception:
        return "None"

#main programme 

# Create embeddings using OllamaEmbeddings 
embeddings = OllamaEmbeddings(model="mistral")
#loading data
data = load_data("data_en.json")
# Define your persist directory
persist_directory = "./chroma_db"
if os.path.exists(persist_directory):
    print("Vectorstore exist , loading ...")
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    print("Vectoresore is loaded")
else:
    print("Creating new vectorstore")
    # Splitting documents while keeping metadata
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(data)  # This keeps metadata
    for i,chunk in enumerate(chunks[:3]):
        print(f"Chunk {i+1}:\n{chunk.page_content}\n")# Define the prompt template
        if i==1:
            break
    # Create and persist vectorstore with metadata
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=persist_directory)

    print("New vectorstore created and persisted with metadata!")

# Enhanced prompt template
prompt = ChatPromptTemplate.from_template("""
You are a legal expert on Tunisian real estate law. Answer the question using ONLY the provided context.
If the context doesn't contain the answer, say "I don't have information about this specific aspect of Tunisian law."

Context: {context}

Question: {question}

Answer in clear, professional language:
""")



llm = OllamaLLM(model="mistral")
#retriever
retriever = vectorstore.as_retriever(search_type="similarity",search_kwargs={"k":8})
chain = (
    {"context":retriever,"question":RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
user_queries=["Co-Ownership In Tunisian Law","Understanding Land Classification In Tunisia","Registered And Unregistered Property In Tunisian Law"]


for query in user_queries:
    print(f"\nQuery: {query}")
    print("____________________________")
    
    try:
        # Get the LLM response
        result = chain.invoke(query)
        
        # Get the retrieved documents separately
        retrieved_docs = vectorstore.similarity_search(query, k=2)
        
        # Print retrieved documents and their metadata
        if retrieved_docs:
            print("Retrieved Documents:")
            for doc in retrieved_docs:
                print(f"Content: {doc.page_content}")
                print(f"Metadata: {doc.metadata}")
                print("=" * 50)
            
            # Print the LLM's response
            clean_text = re.sub(r"<think>.*?</think>", "", result, flags=re.DOTALL)
            print("\nLLM Response:")
            print(clean_text)
        else:
            print("No documents retrieved")
            
    except Exception as e:
        print(f"Error processing query: {str(e)}")
    
    print("____________________________")
    

