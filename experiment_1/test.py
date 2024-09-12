import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import UnstructuredPowerPointLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_and_split_pptx(filepath):
    """
    Loads a PowerPoint file and splits the content into smaller chunks.

    :param filepath: Path to the PowerPoint file.
    :return: List of document chunks after splitting.
    """
    loader = UnstructuredPowerPointLoader(filepath)
    docs = loader.load()

    # Split the content into manageable chunks for embedding
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(docs)


def create_vector_store(docs):
    """
    Creates a vector store by embedding the document chunks.

    :param docs: List of document chunks.
    :return: Vector store created from document embeddings.
    """
    # Embed the document chunks using Ollama embeddings
    return Chroma.from_documents(
        docs,
        embedding=OllamaEmbeddings(model="nomic-embed-text"),
    )


def setup_rag_chain(retriever, llm_model):
    """
    Sets up the RAG (Retrieval-Augmented Generation) chain for answering queries.

    :param retriever: The document retriever object.
    :param llm_model: The language model for generating responses.
    :return: A RAG chain to process questions and provide answers.
    """

    prompt = ChatPromptTemplate(
        messages=[
            HumanMessagePromptTemplate(
                prompt=PromptTemplate(
                    input_variables=["context", "question"],
                    template="""
                        You are an Virgin Voyages assistant for question-answering tasks.
                        Start each message with 'AHOY!' and speak like a pirate.
                        Use the following pieces of retrieved context to answer the question.
                        If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
                        Question: {question}
                        Context: {context}
                        Answer:
                        """,
                )
            ),
        ]
    )

    # Helper function to format retrieved documents into readable text
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Build the RAG chain LCEL (LangChain Expression Language) <- LangChain Library
    # LlamaIndex
    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm_model
        | StrOutputParser()
    )


def vectorize_pptx_and_answer_questions(filepath):
    """
    End-to-end function that vectorizes a PowerPoint file and answers a question based on its content.

    :param filepath: Path to the PowerPoint file.
    :param question: The question to ask about the PowerPoint content.
    """
    # Initialize the language model
    llm = ChatOllama(model="llama3.1")

    # Load, split, and vectorize the PowerPoint content
    splits = load_and_split_pptx(filepath)
    vectorstore = create_vector_store(splits)
    retriever = vectorstore.as_retriever()

    # Setup RAG chain and invoke it with the question
    rag_chain = setup_rag_chain(retriever, llm)
    print("type 'exit' to leave chat")
    while True:
        user_input = input(
            os.getenv("PPTX_INPUT_PROMPT"),
        )
        if user_input == "exit":
            break
        response = rag_chain.invoke(user_input)
        print(response)


if __name__ == "__main__":
    # Load environment variables (such as API keys) from the .env file
    load_dotenv("../.env")

    # Vectorize PowerPoint and answer a question about its content
    vectorize_pptx_and_answer_questions(filepath=os.getenv("PPTX"))
