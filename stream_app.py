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

