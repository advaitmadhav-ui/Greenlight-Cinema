import os
from tqdm import tqdm
from datasets import load_dataset
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

def build_script_vector_store():
    print("1.  Downloading Movie Scripts from alternative dataset (IsmaelMousa/movies)...")
    try:
        dataset = load_dataset("IsmaelMousa/movies", split="train[:100]")
    except Exception as e:
        print(f"Error loading dataset. Please check your internet connection: {e}")
        return
        
    documents = []
    for row in dataset:
        title = row.get('Name', 'Unknown Title')
        script_text = row.get('Script', '')
        
        if script_text:
            documents.append(Document(page_content=script_text, metadata={"title": title}))
            
    print(f" Successfully loaded {len(documents)} full screenplays.")
    
    print("\n2. ️ Chunking scripts by Scene (INT/EXT)...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=150,
        separators=["\n\nINT.", "\n\nEXT.", "\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f" Split scripts into {len(chunks)} overlapping scene chunks.")

    print("\n3.  Initializing Ollama Nomic Embedding Model...")
    # Swapped from HuggingFace to OllamaEmbeddings
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    print("\n4. ️ Building ChromaDB Vector Store in BATCHES...")
    persist_directory = "./chroma_db"
    
    # Initialize an empty ChromaDB instance first
    vectordb = Chroma(embedding_function=embeddings, persist_directory=persist_directory)
    
    # Process the chunks in safe batches of 200 to prevent CPU freezing
    batch_size = 200
    for i in tqdm(range(0, len(chunks), batch_size), desc="Embedding Batches", unit="batch"):
        batch = chunks[i:i + batch_size]
        vectordb.add_documents(batch)
    
    print(f"\n Success! Vector database fully built and saved to the '{persist_directory}' folder.")

if __name__ == "__main__":
    build_script_vector_store()