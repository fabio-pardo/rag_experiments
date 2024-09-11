# Experiment 1

<!--toc:start-->
- [Experiment 1](#experiment-1)
  - [Setup and Installation](#setup-and-installation)
  - [What's the functionality for this experiment?](#whats-the-functionality-for-this-experiment)
<!--toc:end-->

This is a sheet where I'm going to jot down the work that's been completed and
some other notes like possible experiments and what not.

## Setup and Installation

- Run `mkvirtualenv rag_experiments`
- `pip install -r requirements.txt` in the base directory for this project
- Install `Ollama` locally.
- Using `Ollama` we've pulled models `llama3.1` and `nomic-embed-text` for our
chat and embedding purposes, respectively
- There may be some other downloads that must be done for Unstructured to
properly work. I'm going to leave this link here: <https://python.langchain.com/v0.2/docs/integrations/providers/unstructured/#installation-and-setup>
as the answer for what programs are missing should be here.
- In the `.env.example` file, set `LANGCHAIN_TRACING_V2` to `false` if
you don't want LangChain Smith to be functioning. Otherwise, gather an API Key
set them in the `.env.example` and change `.env.example` file to `.env`.
- Make sure that `.env.example` PPTX vars are filled or else it won't work.

## What's the functionality for this experiment?

1. Load LLM to Chat with. In this instance we're chatting with LLam3.1 using Ollama
2. Load the PPTX that we want to work with.
3. Recursively split the documents into chunks, trying to keep paragraphs (and
then sentences, and then words) together as long as possible. As those are the
strongest semantically related pieces of text.
4. Create a Chroma vectorstore and converts the split documents into numerical
vector representations of the text. The model "nomic-embed-text" is the specific
embedding model used to generate vector embeddings for the text. This model
specializes in text embeddings.
5. A vector store retriever is initialized from our used vector store. In this case
a Chroma Vector Store Retriever.
6. We pull a prompt from the LangChain hub to use. If you look online
this prompt is the following:

```text
You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Use three sentences maximum and keep the answer concise.
Question: {question}
Context: {context}
Answer:

```

7. Define a Function to Format Retrieved Documents. Used to format the retrieved
documents into a readable format by concatenating the content of each doc
with two line breaks.
8. Create a RAG chain by chaining together different components for processing
the query:

- The `retriever` retrieves relevant document chunks, which are formatted using the `format_docs` function
- The question is passed as a passthrough via `RunnablePassthrough()`.
- The retrieved and formatted docs are combined with the question and passed
to the `prompt`.
- The result is passed to the `llm` which generates a response based on the
combined input
- Finally the response is parsed using `StrOutputParser()`, which coverts the model's
output into a string.
