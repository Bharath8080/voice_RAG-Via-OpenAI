import os
import base64
import tempfile
import requests
from io import BytesIO
import streamlit as st
from dotenv import load_dotenv
from audio_recorder_streamlit import audio_recorder
import faiss
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core import VectorStoreIndex, StorageContext, SimpleDirectoryReader
from llama_index.core import Settings
import pygame
import time
import uuid
from openai import OpenAI

# Load API keys
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize OpenAI client
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize pygame mixer for audio playback
pygame.mixer.init()

# Streamlit setup
st.set_page_config(page_title="🎤 Voice RAG Assistant", layout="centered")
st.title("🎤 Voice-Based RAG Chatbot")
st.markdown("---")

# Sidebar: Upload PDF and service selection
with st.sidebar:
    st.header("📄 Upload your PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
    
    st.header("🔧 Service Selection")
    
    # TTS service selection
    tts_service = st.radio(
        "Text-to-Speech Provider",
        ("OpenAI", "Groq"),
        horizontal=True
    )
    
    # STT service selection
    stt_service = st.radio(
        "Speech-to-Text Provider",
        ("OpenAI", "Groq"),
        horizontal=True
    )
    
    # Voice selection based on the selected TTS service
    st.subheader("🎤 Voice Selection")
    
    if tts_service == "OpenAI":
        # OpenAI voice options
        voice_options = [
            "alloy", "echo", "fable", "onyx", "nova", "shimmer"
        ]
        selected_voice = st.selectbox("Select OpenAI Voice", voice_options)
        
        # Add model selection for OpenAI TTS
        tts_models = ["tts-1", "tts-1-hd", "gpt-4o-mini-tts"]
        selected_tts_model = st.selectbox("Select OpenAI TTS Model", tts_models)
    else:
        # Groq voice options
        voice_options = [
            "Fritz-PlayAI", "Arista-PlayAI", "Atlas-PlayAI", "Basil-PlayAI", 
            "Briggs-PlayAI", "Calum-PlayAI", "Celeste-PlayAI", "Cheyenne-PlayAI", 
            "Chip-PlayAI", "Cillian-PlayAI", "Deedee-PlayAI", "Gail-PlayAI", 
            "Indigo-PlayAI", "Mamaw-PlayAI", "Mason-PlayAI", "Mikail-PlayAI", 
            "Mitch-PlayAI", "Quinn-PlayAI", "Thunder-PlayAI"
        ]
        selected_voice = st.selectbox("Select Groq Voice", voice_options)
    
    st.markdown("---")
    st.markdown("🎙️ Use mic to ask questions or type below.")

# Initialize index
def init_index(file_path):
    llm = Gemini(model="models/gemini-2.0-flash-exp")
    embed_model = GeminiEmbedding(model_name="models/text-embedding-004")
    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.chunk_size = 1024
    
    reader = SimpleDirectoryReader(input_files=[file_path])
    documents = reader.load_data()
    faiss_index = faiss.IndexFlatL2(768)
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
    return index

# Stream TTS response using OpenAI
def stream_tts_response_openai(answer_text, voice, model):
    # Display speaker gif if available
    gif_path = "speaker.gif"
    placeholder = None
    if os.path.exists(gif_path):
        with open(gif_path, "rb") as f:
            data_url = base64.b64encode(f.read()).decode("utf-8")
            placeholder = st.empty()
            placeholder.markdown(f'<img src="data:image/gif;base64,{data_url}" style="height:100px;">', unsafe_allow_html=True)
    
    with st.spinner("🔊 Generating audio response with OpenAI..."):
        try:
            # Create a temporary file for the audio output
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
                temp_path = temp_file.name
            
            # Generate speech with OpenAI TTS
            response = openai_client.audio.speech.create(
                model=model,
                voice=voice,
                input=answer_text,
                response_format="mp3"
            )
            
            # Save the audio to the temporary file
            with open(temp_path, "wb") as f:
                f.write(response.content)
            
            # Use pygame to play audio 
            pygame.mixer.music.load(temp_path)
            pygame.mixer.music.play()
            
            # Create a progress bar to show audio is playing
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Get audio duration (approximate - ~5 words per second)
            word_count = len(answer_text.split())
            estimated_duration = word_count / 5  # about 5 words per second
            
            # Update progress bar while audio is playing
            start_time = time.time()
            while pygame.mixer.music.get_busy():
                elapsed = time.time() - start_time
                progress = min(elapsed / estimated_duration, 1.0)
                progress_bar.progress(progress)
                status_text.text(f"Playing audio... ({int(elapsed)}s)")
                time.sleep(0.1)
            
            # Clear progress indicators when done
            progress_bar.empty()
            status_text.empty()
            
        except Exception as e:
            st.error(f"Error with OpenAI TTS: {str(e)}")
        finally:
            # Clean up temp file after we're done with it
            try:
                # Ensure the mixer is done with the file before deleting
                pygame.mixer.music.stop()
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
            except Exception as e:
                pass
    
    if placeholder:
        placeholder.empty()

# Stream TTS response using Groq
def stream_tts_response_groq(answer_text, voice):
    # Display speaker gif if available
    gif_path = "speaker.gif"
    placeholder = None
    if os.path.exists(gif_path):
        with open(gif_path, "rb") as f:
            data_url = base64.b64encode(f.read()).decode("utf-8")
            placeholder = st.empty()
            placeholder.markdown(f'<img src="data:image/gif;base64,{data_url}" style="height:100px;">', unsafe_allow_html=True)
    
    with st.spinner("🔊 Generating audio response with Groq..."):
        try:
            # Create a temporary file for the audio output
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = temp_file.name
            
            # Prepare headers for Groq API
            headers = {
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            }
            
            # Prepare payload for Groq TTS
            payload = {
                "model": "playai-tts",
                "voice": voice,
                "input": answer_text,
                "response_format": "wav"
            }
            
            # Make request to Groq API
            response = requests.post(
                "https://api.groq.com/openai/v1/audio/speech",
                headers=headers,
                json=payload,
                stream=False
            )
            
            # Check if the request was successful
            if response.status_code == 200:
                # Save the audio to the temporary file
                with open(temp_path, "wb") as f:
                    f.write(response.content)
                
                # Use pygame to play audio 
                pygame.mixer.music.load(temp_path)
                pygame.mixer.music.play()
                
                # Create a progress bar to show audio is playing
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Get audio duration (approximate - ~5 words per second)
                word_count = len(answer_text.split())
                estimated_duration = word_count / 5  # about 5 words per second
                
                # Update progress bar while audio is playing
                start_time = time.time()
                while pygame.mixer.music.get_busy():
                    elapsed = time.time() - start_time
                    progress = min(elapsed / estimated_duration, 1.0)
                    progress_bar.progress(progress)
                    status_text.text(f"Playing audio... ({int(elapsed)}s)")
                    time.sleep(0.1)
                
                # Clear progress indicators when done
                progress_bar.empty()
                status_text.empty()
            else:
                st.error(f"Failed to generate speech with Groq: {response.text}")
        except Exception as e:
            st.error(f"Error with Groq TTS: {str(e)}")
        finally:
            # Clean up temp file after we're done with it
            try:
                # Ensure the mixer is done with the file before deleting
                pygame.mixer.music.stop()
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
            except Exception as e:
                pass
    
    if placeholder:
        placeholder.empty()

# Transcribe audio using OpenAI
def transcribe_audio_openai(audio_data):
    # Create a unique temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(audio_data)
        tmp_path = tmp_file.name
    
    try:
        # Transcribe using OpenAI
        with open(tmp_path, "rb") as audio_file:
            transcript = openai_client.audio.transcriptions.create(
                model="gpt-4o-transcribe",
                file=audio_file,
                language="en"
            )
        return transcript.text
    except Exception as e:
        st.error(f"Error with OpenAI transcription: {str(e)}")
        return ""
    finally:
        # Clean up the temporary file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

# Transcribe audio using Groq
def transcribe_audio_groq(audio_data):
    # Create a unique temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(audio_data)
        tmp_path = tmp_file.name
    
    try:
        # Transcribe using Groq direct API call
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}"
        }
        
        with open(tmp_path, "rb") as f:
            files = {
                "file": (os.path.basename(tmp_path), f, "audio/wav")
            }
            
            data = {
                "model": "whisper-large-v3-turbo",
                "language": "en",
                "response_format": "text"
            }
            
            response = requests.post(
                "https://api.groq.com/openai/v1/audio/transcriptions",
                headers=headers,
                data=data,
                files=files
            )
        
        if response.status_code == 200:
            return response.text
        else:
            st.error(f"Failed to transcribe audio with Groq: {response.text}")
            return ""
    except Exception as e:
        st.error(f"Error with Groq transcription: {str(e)}")
        return ""
    finally:
        # Clean up the temporary file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

# Main app logic
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name
    
    index = init_index(tmp_path)
    query_engine = index.as_query_engine()
    
    st.subheader("🎙️ Speak your question or type below")
    audio_bytes = audio_recorder(recording_color="#f10c49", neutral_color="#6aa36f")
    user_query = ""
    
    if audio_bytes:
        st.info("🛠️ Transcribing audio...")
        
        # Use selected STT service
        if stt_service == "OpenAI":
            user_query = transcribe_audio_openai(audio_bytes)
        else:  # Groq
            user_query = transcribe_audio_groq(audio_bytes)
            
        if user_query:
            st.success(f"📝 Transcribed: `{user_query}`")
        else:
            st.error("Failed to transcribe audio. Please try again or type your question.")
    
    user_query = st.text_input("✍️ Or type your question here:", value=user_query)
    
    if user_query:
        with st.spinner("🤖 Gemini is processing..."):
            response = query_engine.query(user_query)
            answer = response.response
            
            st.markdown("### 📢 Gemini's Answer")
            st.write(answer)
            
            # Use selected TTS service
            if tts_service == "OpenAI":
                stream_tts_response_openai(answer, selected_voice, selected_tts_model)
            else:  # Groq
                stream_tts_response_groq(answer, selected_voice)
else:
    st.warning("📄 Please upload a PDF from the sidebar to begin.")
