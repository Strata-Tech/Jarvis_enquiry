
from langchain.vectorstores import FAISS
from langchain.llms import GooglePalm
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import os

from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env (especially openai api key)

# Create Google Palm LLM model
llm = GooglePalm(google_api_key=os.environ["GOOGLE_API_KEY"], temperature=0.1)
# # Initialize instructor embeddings using the Hugging Face model
instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")
vectordb_file_path = "faiss_index"

def create_vector_db():
    # Load data from FAQ sheet
    loader = CSVLoader(file_path='HDBfaq.csv', source_column="prompt",encoding='ISO-8859-1')
    data = loader.load()

    # Create a FAISS instance for vector database from 'data'
    vectordb = FAISS.from_documents(documents=data,
                                    embedding=instructor_embeddings)

    # Save vector database locally
    vectordb.save_local(vectordb_file_path)


def get_qa_chain():

    # Load the vector database from the local folder
    vectordb = FAISS.load_local(vectordb_file_path, instructor_embeddings)

    # Create a retriever for querying the vector database
    retriever = vectordb.as_retriever(score_threshold=0.7)

    prompt_template = """Given the following context and a question, generate an answer based on this context only.
    In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
    If the answer is not found in the context, kindly state "I'm sorry ,I don't know. This is not within my knowledge base at the moment." Don't try to make up an answer.

    CONTEXT: {context}

    QUESTION: {question}"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    chain = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type="stuff",
                                        retriever=retriever,
                                        input_key="query",
                                        return_source_documents=True,
                                        chain_type_kwargs={"prompt": PROMPT})

    return chain

if __name__ == "__main__":
    create_vector_db()
    chain = get_qa_chain()
    print(chain("Do you have javascript course?"))





import streamlit as st
from langchain_helper import get_qa_chain, create_vector_db

from googletrans import Translator

from gtts import gTTS
from playsound import playsound

st.title("Ask J.A.R.V.I.S. 🙋🏻‍❓🤖")

# Create a list to store the last three bot responses
bot_responses = []

btn = st.button("Create Knowledgebase")
if btn:
    create_vector_db()

# Add a sidebar with a language dropdown
selected_language = st.sidebar.selectbox("Select Language for Answer", ["English", "Chinese", "Malay", "Tamil"])

question = st.text_input("Question:")

if question:
    chain = get_qa_chain()
    response = chain(question)

    # Append bot response to the bot_responses list
    bot_responses.append(response["result"])

    # Keep only the last three bot responses
    #while len(bot_responses) > 3:
        #bot_responses.pop(0)

# Display the last three bot responses in the main interface
st.header("J.A.R.V.I.S. says: ")
for bot_response in bot_responses:
    st.write(bot_response)

st.header("Translation: ")

if question:
    # Translate the response to the selected language if not English
    translated_response = response["result"]
    if selected_language != "English" and selected_language!="Chinese":
        translator = Translator()
        translated_response = translator.translate(response["result"], src="en", dest=selected_language.lower()).text

    elif selected_language == "Chinese":
        selected_language="zh-cn"
        translator = Translator()
        translated_response = translator.translate(response["result"], src="en", dest="zh-cn").text



    # Display the translated response in the selected language
    st.write(translated_response)

    # Convert the translated response to speech in the selected language
    language_code = {
        "English": "en",
        "zh-cn": "zh-cn",
        "Malay": "ms",
        "Tamil": "ta"
    }

    tts = gTTS(translated_response, lang=language_code[selected_language], slow=False,tld='com')
    tts.save("output.mp3")
    playsound("output.mp3")

