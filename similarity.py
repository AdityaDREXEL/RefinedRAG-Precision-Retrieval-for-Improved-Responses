import json
import os
import time
import warnings

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import PyPDFium2Loader
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import Annoy, Qdrant
from langchain_core.language_models import BaseLanguageModel
from langchain_core.vectorstores import VectorStore, VectorStoreRetriever
from langchain_groq import ChatGroq

warnings.filterwarnings("ignore")


def load_document(file_path: str) -> str:
    """
    Loads a document from the given file path.
    """
    loaded_documents = PyPDFium2Loader(file_path).load()
    document_text = "\n".join([doc.page_content for doc in loaded_documents])

    return document_text


def create_embeddings() -> FastEmbedEmbeddings:
    """
    Initializes and returns the FastEmbedEmbeddings instance.
    """
    return FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")


def create_vector_db(documents: list, audio_name: str) -> Qdrant:
    """
    Creates a vector store (Qdrant) and returns the vector database.
    """
    embeddings = create_embeddings()
    vector_db = Qdrant.from_documents(
        documents=documents,
        embedding=embeddings,
        path="data/embeddings",
        collection_name=f"document_collection_{audio_name}",
    )
    return vector_db


def retreive_similar_docs_hybrid_Search(prompt: str, vector_db: Qdrant) -> str:
    """
    Retreive vectors similar to the prompt from the vector db
    """
    retriever = vector_db.as_retriever(
        search_type="similarity", search_kwargs={"body_search": "alpha"}
    )

    search_results = retriever.invoke(prompt)
    # Format the retrieved documents
    context = "\n\n\n".join([result.page_content for result in search_results])

    return context


def retreive_similar_docs_hnsw(prompt: str, vector_db: Qdrant) -> str:
    """
    Retreive vectors similar to the prompt from the vector db
    """

    # Perform similarity search on the vector database
    search_results = vector_db.similarity_search([prompt], k=3)

    # Format the retrieved documents
    context = "\n\n\n".join([result.page_content for result in search_results])

    return context


def retreive_similar_docs_mmr(prompt: str, vector_db: Qdrant) -> str:
    """
    Retreive vectors similar to the prompt from the vector db
    """
    retriever = vector_db.as_retriever(search_type="mmr", search_kwargs={"k": 3})

    search_results = retriever.invoke(prompt)
    # print(search_results)

    # Format the retrieved documents
    context = "\n\n\n".join([result.page_content for result in search_results])

    return context


def retreive_similar_docs_similarity_score_threshold(
    prompt: str, vector_db: Qdrant
) -> str:
    """
    Retreive vectors similar to the prompt from the vector db
    """

    retriever = vector_db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": 0.6, "k": 3},
    )

    search_results = retriever.invoke(prompt)
    # print(search_results)

    # Format the retrieved documents
    context = "\n\n\n".join([result.page_content for result in search_results])

    return context


def retreive_similar_docs_top_k(prompt: str, vector_db: Qdrant) -> str:
    """
    Retreive vectors similar to the prompt from the vector db
    """

    retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    search_results = retriever.invoke(prompt)
    # print(search_results)

    # Format the retrieved documents
    context = "\n\n\n".join([result.page_content for result in search_results])

    return context


def main():
    # Record the start time
    start_time = time.time()

    # Load the document from the provided path
    document_path = "data\\transcription\\transcribed_audio_1.pdf"
    document_text = load_document(document_path)

    audio_name = document_path[-11:-4]
    print(audio_name)

    # Embed the document
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2048,
        chunk_overlap=128,
        add_start_index=True,
    )

    recursive_chunks = recursive_splitter.split_text(document_text)
    splits = [Document(page_content=chunk) for chunk in recursive_chunks]

    # Store documents in the vector database (Qdrant)
    vector_db = create_vector_db(splits, audio_name)
    print("Vector db created!\n\n")

    # Get the user prompt and generate a response
    user_prompt = """
        What large language models are discussed in the class? List all the LLMs.
        """

    # top k
    similarity_search_results_top_k = retreive_similar_docs_top_k(
        user_prompt, vector_db
    )
    print(similarity_search_results_top_k)

    # # similarity score threshold
    # similarity_search_results_similarity_Score_threshold = retreive_similar_docs_similarity_score_threshold(user_prompt, vector_db)
    # print(similarity_search_results_similarity_Score_threshold)

    # mmr
    # similarity_search_results_mmr = retreive_similar_docs_mmr(user_prompt, vector_db)
    # print(similarity_search_results_mmr)

    # # Approximate Nearest Neighbour
    # similarity_search_results_hnsw = retreive_similar_docs_hnsw(user_prompt, vector_db)
    # print(similarity_search_results_hnsw)

    # # Hybrid Search
    # similarity_search_results_hybrid_search = retreive_similar_docs_hybrid_Search(user_prompt, vector_db)
    # print(similarity_search_results_hybrid_search)

    # Record the end time
    end_time = time.time()

    # Calculate and print the time taken
    time_taken = end_time - start_time
    print(f"Time taken to generate the results: {time_taken:.2f} seconds")


if __name__ == "__main__":
    main()
