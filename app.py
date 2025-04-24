import asyncio
import os
import random
import tempfile

import speech_recognition as sr
import streamlit as st
from dotenv import load_dotenv

from refinedrag.chain import ask_question, create_chain
from refinedrag.config import Config
from refinedrag.ingestor import Ingestor
from refinedrag.model import create_llm
from refinedrag.retriever import create_retriever
from refinedrag.uploader import upload_files

load_dotenv()

LOADING_MESSAGES = [
    "Calculating your answer through multiverse...",
    "Adjusting quantum entanglement...",
    "Summoning star wisdom... almost there!",
    "Consulting Schrödinger's cat...",
    "Warping spacetime for your response...",
    "Balancing neutron star equations...",
    "Analyzing dark matter... please wait...",
    "Engaging hyperdrive... en route!",
    "Gathering photons from a galaxy...",
    "Beaming data from Andromeda... stand by!",
]


@st.cache_resource(show_spinner=False)
def build_qa_chain(files):
    file_paths = upload_files(files)
    vector_store = Ingestor().ingest(file_paths)
    llm = create_llm()
    retriever = create_retriever(llm, vector_store=vector_store)
    return create_chain(llm, retriever)


async def ask_chain(question: str, chain):
    full_response = ""
    assistant = st.chat_message(
        "assistant", avatar=str(Config.Path.IMAGES_DIR / "assistant-avatar.png")
    )
    with assistant:
        message_placeholder = st.empty()
        message_placeholder.status(random.choice(LOADING_MESSAGES), state="running")
        documents = []
        async for event in ask_question(chain, question, session_id="session-id-42"):
            if type(event) is str:
                full_response += event
                message_placeholder.markdown(full_response)
            if type(event) is list:
                documents.extend(event)
        for i, doc in enumerate(documents):
            with st.expander(f"Source #{i+1}"):
                st.write(doc.page_content)

    st.session_state.messages.append({"role": "assistant", "content": full_response})


def handle_video_and_pdf_upload():
    st.header("Refined RAG")
    st.subheader("Get answers from your documents or videos")

    # PDF Upload Section
    st.write("### Upload PDF Files")
    uploaded_pdfs = st.file_uploader(
        label="Upload PDF files", type=["pdf"], accept_multiple_files=True
    )

    # Video Upload Section
    st.write("### Upload Video File")
    uploaded_video = st.file_uploader("Upload Video File", type=["mp4", "mov", "avi"])

    # Process PDF Documents
    if uploaded_pdfs:
        with st.spinner("Processing your PDF files..."):
            chain = build_qa_chain(uploaded_pdfs)
            st.session_state.chain = chain
            st.success("PDF processing completed!")
            return

    # Process Video File
    if uploaded_video:
        st.info("Video uploaded successfully. Ready to process.")
        if st.button("Process Video"):
            try:
                with st.spinner("Saving video..."):
                    temp_dir = tempfile.TemporaryDirectory()
                    temp_video_path = os.path.join(temp_dir.name, uploaded_video.name)
                    sanitized_path = temp_video_path.replace(" ", "_")
                    with open(sanitized_path, "wb") as f:
                        f.write(uploaded_video.read())

                with st.spinner("Extracting audio and transcribing..."):
                    transcription = transcribe_video_to_text(sanitized_path)

                if transcription:
                    st.success("Transcription completed!")
                    st.text_area("Video Transcription", transcription, height=200)

                    # Process transcription
                    with st.spinner("Processing transcription text..."):
                        ingestor = Ingestor()
                        vector_store = ingestor.ingest_text(
                            transcription, source="transcription"
                        )
                        llm = create_llm()
                        retriever = create_retriever(llm, vector_store=vector_store)
                        st.session_state.chain = create_chain(llm, retriever)
                        return
                else:
                    st.error("Failed to transcribe video.")
            except Exception as e:
                st.error(f"Error during video processing: {e}")


def transcribe_video_to_text(video_path):
    try:
        # Extract audio using ffmpeg
        audio_path = video_path.replace(".mp4", ".wav")
        os.system(f"ffmpeg -i {video_path} -q:a 0 -map a {audio_path} -y")

        # Transcribe audio
        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_path) as source:
            audio = recognizer.record(source)
        transcription = recognizer.recognize_google(audio)

        # Clean up
        os.remove(audio_path)
        return transcription
    except Exception as e:
        st.error(f"Error during transcription: {e}")
        return None


def show_message_history():
    if "messages" in st.session_state:
        for message in st.session_state.messages:
            role = message["role"]
            avatar_path = (
                Config.Path.IMAGES_DIR / "assistant-avatar.png"
                if role == "assistant"
                else Config.Path.IMAGES_DIR / "user-avatar.png"
            )
            with st.chat_message(role, avatar=str(avatar_path)):
                st.markdown(message["content"])


def show_chat_input():
    if "chain" not in st.session_state or st.session_state.chain is None:
        st.warning(
            "Please upload and process a document or video before asking questions."
        )
        return

    if prompt := st.chat_input("Ask your question here"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message(
            "user",
            avatar=str(Config.Path.IMAGES_DIR / "user-avatar.png"),
        ):
            st.markdown(prompt)

        # Process the question using the chain
        asyncio.run(ask_chain(prompt, st.session_state.chain))


st.set_page_config(page_title="Refined RAG", page_icon="🐧")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hi! What do you want to know about your documents or videos?",
        }
    ]

if "chain" not in st.session_state:
    st.session_state.chain = None

handle_video_and_pdf_upload()

show_message_history()
show_chat_input()
