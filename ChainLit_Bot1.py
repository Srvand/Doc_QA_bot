from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
import os
import chainlit as cl
import tempfile
from langchain import PromptTemplate, OpenAI, LLMChain

os.environ["OPENAI_API_KEY"] = "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"

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

system_template = """you are an Sales representative who will have conversation with potential customer. 
Use the following pieces of context to answer the users question.Your role is to convince the customer to enroll for the product.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Begin!
----------------
Question: {question}
Answer: Let's think step by step."""

# system_template="""Question: {question}
# Answer: Let's think step by step."""



prompt = PromptTemplate(template=system_template, input_variables=["question"])

@cl.on_chat_start
async def load():
    files = None

    # Wait for the user to upload a file
    while files == None:
        files = await cl.AskFileMessage(content="Please upload a file to start chatting!", accept=["pdf"]).send()

    file = files[0]

    msg = cl.Message(content=f"Processing `{file.name}`...")
    await msg.send()

    with tempfile.NamedTemporaryFile(delete=False) as temp:
        temp.write(file.content)
        temp_path = temp.name

    # Load the PDF using PyPDFLoader into an array of documents, where each document contains the page content and metadata with page number.
    loader = PyPDFLoader(temp_path)
    pages = loader.load_and_split()
    
    # Combine the page content into a single text variable.
    text = ' '.join([page.page_content for page in pages])

    # Split the text into chunks
    texts = text_splitter.split_text(text)

    # Create a Chroma vector store
    embeddings = OpenAIEmbeddings()
    docsearch = await cl.make_async(Chroma.from_texts)(
        texts, embeddings
    )
    
    # Create a chain that uses the Chroma vector store
    chain = LLMChain(prompt=prompt,
        llm=ChatOpenAI(temperature=0)
    )

    # Save the metadata and texts in the user session
    cl.user_session.set("llm_chain", chain)

    # Let the user know that the system is ready
    await msg.update(content=f"`{file.name}` processed. You can now ask questions!")

    return chain


@cl.on_message
async def main(message: str):
    llm_chain = cl.user_session.get("llm_chain")
    res = await llm_chain.acall(message, callbacks=[cl.AsyncLangchainCallbackHandler()])
    await cl.Message(content=res["text"]).send()

    