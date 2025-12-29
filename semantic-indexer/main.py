import os
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict

# LangChain Imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- Configuration et Modèles ---

# Le chemin vers le dossier où FAISS est stocké
VECTOR_FOLDER = os.getenv("VECTOR_FOLDER", "vector_store")
os.makedirs(VECTOR_FOLDER, exist_ok=True)

# Instanciation de l'Embedding Model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Dictionnaire de bases de données vectorielles par conversation_id (chargées en mémoire)
vectorstores: Dict[str, FAISS] = {}

def get_vector_store_path(conversation_id: str) -> str:
    """Retourne le chemin du dossier pour une conversation spécifique."""
    conv_folder = os.path.join(VECTOR_FOLDER, conversation_id)
    os.makedirs(conv_folder, exist_ok=True)
    return conv_folder

def load_vector_store(conversation_id: str) -> Optional[FAISS]:
    """Charge ou initialise l'index FAISS pour une conversation spécifique."""
    if conversation_id in vectorstores:
        return vectorstores[conversation_id]
    
    conv_folder = get_vector_store_path(conversation_id)
    index_path = os.path.join(conv_folder, "faiss.index")
    
    # On vérifie si le fichier .faiss existe réellement
    if os.path.exists(index_path + ".faiss"):
        try:
            vectorstore = FAISS.load_local(
                conv_folder, 
                embeddings, 
                "faiss.index",
                allow_dangerous_deserialization=True
            )
            vectorstores[conversation_id] = vectorstore
            print(f"✅ Index FAISS chargé pour conversation {conversation_id} ! ({vectorstore.index.ntotal} documents)")
            return vectorstore
        except Exception as e:
            print(f"⚠️ Erreur de chargement de l'index pour {conversation_id} : {e}. L'index sera recréé.")
            return None
    else:
        print(f"ℹ️ Aucun index FAISS trouvé pour conversation {conversation_id}. Il sera créé lors de la première ingestion.")
        return None

# --- Schémas de données Pydantic ---

class RetrievalRequest(BaseModel):
    """Schéma pour la requête de recherche de fragments."""
    question: str = Field(..., description="La question de l'utilisateur.")
    conversation_id: str = Field(..., description="L'ID de la conversation.")
    k: int = 8
    score_threshold: float = 0.75

class IngestRequest(BaseModel):
    """Schéma pour l'ingestion de contenu."""
    content: str = Field(..., description="Le texte brut du document à indexer.")
    source: str = Field(..., description="Le nom du fichier source (ex: 'doc.pdf').")
    conversation_id: str = Field(..., description="L'ID de la conversation.")

class Chunk(BaseModel):
    """Représentation d'un fragment de document pour la réponse de recherche."""
    content: str
    source: str
    score: float 

class RetrievalResponse(BaseModel):
    """Schéma de la réponse pour une recherche sémantique."""
    chunks: List[Chunk] = Field(..., description="Liste des fragments de document pertinents.")

# --- FastAPI App ---

app = FastAPI(title="Semantic Indexer Microservice")

@app.on_event("startup")
async def startup_event():
    """Initialisation au démarrage."""
    print("✅ Semantic Indexer démarré avec support multi-conversations.")

# --- Endpoints ---

@app.post("/index-chunks", status_code=200)
def index_document(request: IngestRequest):
    text = request.content
    source = request.source
    conversation_id = request.conversation_id

    if not text.strip():
        raise HTTPException(status_code=400, detail="Contenu du document vide.")
    
    if not conversation_id:
        raise HTTPException(status_code=400, detail="conversation_id est requis.")

    # 1. Découpage (Chunking)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,   
        chunk_overlap=200,  
        separators=["\n\n", "\n", ".", " ", ""] 
    )
    chunks = splitter.split_text(text)
    docs = [Document(page_content=c, metadata={"source": source}) for c in chunks]

    # 2. Charger ou créer l'index pour cette conversation
    vectorstore = load_vector_store(conversation_id)
    
    if vectorstore is None:
        # Créer un nouvel index pour cette conversation
        vectorstore = FAISS.from_documents(docs, embeddings)
        vectorstores[conversation_id] = vectorstore
    else:
        # Ajouter les documents à l'index existant
        vectorstore.add_documents(docs)

    # 3. Sauvegarde sur le disque
    try:
        conv_folder = get_vector_store_path(conversation_id)
        vectorstore.save_local(conv_folder, "faiss.index")
        return {"status": "success", "message": f"Indexé : {source} ({len(docs)} morceaux) pour conversation {conversation_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la sauvegarde FAISS : {e}")


@app.post("/retrieve-chunks", response_model=RetrievalResponse)
def retrieve_chunks(request: RetrievalRequest):
    conversation_id = request.conversation_id
    
    if not conversation_id:
        raise HTTPException(status_code=400, detail="conversation_id est requis.")
    
    # Charger l'index pour cette conversation
    vectorstore = load_vector_store(conversation_id)
    
    if vectorstore is None:
        # Si pas d'index pour cette conversation, renvoie liste vide
        return RetrievalResponse(chunks=[])
    
    docs_scores = vectorstore.similarity_search_with_score(
        request.question,
        k=request.k
    )
    
    # Filtrage
    relevant = [
        Chunk(content=doc.page_content, source=doc.metadata["source"], score=score) 
        for doc, score in docs_scores 
        if score < request.score_threshold
    ]

    # Sécurité : si le seuil est trop strict, on prend quand même les meilleurs
    if not relevant and docs_scores:
        relevant = [
            Chunk(content=doc.page_content, source=doc.metadata["source"], score=score) 
            for doc, score in docs_scores[:3]
        ]
        
    return RetrievalResponse(chunks=relevant)