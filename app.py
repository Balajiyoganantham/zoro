from flask import Flask, render_template, request, Response
import google.generativeai as genai
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from google.api_core import retry
from kokoro import KPipeline
import soundfile as sf
import io
import numpy as np
import os

app = Flask(__name__)
app.static_folder = 'static'

# Load API key
GOOGLE_API_KEY = "AIzaSyA8o58BN_OY0vfv1wIt_TR0cNIZ6W9sfOI"  # Replace with your actual API key
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize Kokoro TTS pipeline
tts_pipeline = KPipeline(lang_code='a')  # 'a' for American English

# Define the embedding function
class PsychologyEmbeddingFunction(EmbeddingFunction):
    document_mode = True

    def __call__(self, input: Documents) -> Embeddings:
        if self.document_mode:
            embedding_task = "retrieval_document"
        else:
            embedding_task = "retrieval_query"
        retry_policy = {"retry": retry.Retry(predicate=retry.if_transient_error)}
        response = genai.embed_content(
            model="models/text-embedding-004",
            content=input,
            task_type=embedding_task,
            request_options=retry_policy,
        )
        return response["embedding"]

# Sample documents (replace with your actual mental health content)
DOCUMENT1 = "Managing emotions involves recognizing triggers and practicing mindfulness techniques..."
DOCUMENT2 = "Stress can be managed through deep breathing exercises, physical activity, and proper sleep..."
DOCUMENT3 = "Building resilience requires fostering positive relationships and maintaining a growth mindset..."
documents = [DOCUMENT1, DOCUMENT2, DOCUMENT3]

# Setup ChromaDB
DB_NAME = "psychology_chatbot_db"
embed_fn = PsychologyEmbeddingFunction()
embed_fn.document_mode = True
chroma_client = chromadb.Client()
db = chroma_client.get_or_create_collection(name=DB_NAME, embedding_function=embed_fn)

# Add documents to the database
db.add(documents=documents, ids=[str(i) for i in range(len(documents))])

# Initialize the generative model
model = genai.GenerativeModel("gemini-1.5-flash-latest")

# Dictionary to store conversation history
conversation_history = {}

# Chatbot logic
def get_bot_response(userText, session_id):
    # Retrieve conversation history for the session
    history = conversation_history.get(session_id, [])

    # Search ChromaDB for relevant content
    result = db.query(query_texts=[userText], n_results=1)

    # Access the passage correctly, assuming single query and single result
    passage = result["documents"][0][0] if result["documents"] else ""

    if not passage:
        response = "I'm sorry, I couldn't find any specific information for your question. However, remember that self-care, mindfulness, and talking to trusted people are great ways to maintain mental health."
    else:
        # Format passage and prompt
        passage_oneline = passage.replace("\n", " ")
        query_oneline = userText.replace("\n", " ")

        # Include conversation history in the prompt
        history_text = "\n".join([f"{msg['role']}: {msg['text']}" for msg in history])
        prompt = f"""Your name is Zoro, created by Balaji. You are a supportive and understanding mental health chatbot that provides thoughtful responses using text from the reference passage included below. Your goal is to offer empathetic, clear, and helpful advice based on the provided information.
        Be sure to:
        - Respond in a warm, reassuring, and conversational tone.
        - Break down complex ideas into simple, easy-to-understand explanations.
        - Encourage positive thinking and actionable steps when appropriate.
        - Only use information from the passage when relevant; otherwise, provide general mental health advice.
        CONVERSATION HISTORY:
        {history_text}
        QUESTION: {query_oneline}
        PASSAGE: {passage_oneline}
        """

        # Generate response
        answer = model.generate_content(prompt)
        response = answer.text

    # Update conversation history
    history.append({"role": "user", "text": userText})
    history.append({"role": "bot", "text": response})
    conversation_history[session_id] = history

    return response

# Function to convert text to speech using Kokoro
def text_to_speech(text):
    # Create a BytesIO buffer to store the audio
    buffer = io.BytesIO()
    
    # Clean and prepare the text
    # Remove excess whitespace and normalize line endings
    text = ' '.join(text.split())
    
    # Generate audio using Kokoro pipeline without splitting
    generator = tts_pipeline(
        text,
        voice='af_heart',
        speed=1,
        split_pattern=None  # Don't split the text
    )
    
    # Collect all audio segments
    all_audio = []
    for _, _, audio in generator:
        all_audio.append(audio)
    
    # Concatenate all audio segments
    if all_audio:
        combined_audio = np.concatenate(all_audio)
        
        # Write the combined audio to the buffer
        sf.write(buffer, combined_audio, 24000, format='wav')
    
    # Reset buffer position
    buffer.seek(0)
    return buffer

# Flask routes
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_response():
    userText = request.args.get('msg')
    session_id = request.args.get('session_id', 'default_session')
    bot_response = get_bot_response(userText, session_id)
    return bot_response

@app.route("/get_audio")
def get_audio():
    userText = request.args.get('msg')
    session_id = request.args.get('session_id', 'default_session')
    bot_response = get_bot_response(userText, session_id)

    # Convert bot response to speech
    audio_buffer = text_to_speech(bot_response)

    return Response(
        audio_buffer.getvalue(),
        mimetype="audio/wav"
    )

if __name__ == "__main__":
    app.run(debug=True)