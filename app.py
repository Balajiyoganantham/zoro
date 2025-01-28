from flask import Flask, render_template, request
import os
import joblib
import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertModel
import torch
from textblob import TextBlob
from textblob.sentiments import PatternAnalyzer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
import string
import re
from nltk.stem.wordnet import WordNetLemmatizer
import random

app = Flask(__name__)
app.static_folder = 'static'

# Load BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Load pre-trained embeddings and dataset
sent_bertphrase_embeddings = joblib.load(r"model/questionembedding.dump")
sent_bertphrase_ans_embeddings = joblib.load(r"model/ansembedding.dump")
df = pd.read_csv(r"model/20200325_counsel_chat.csv", encoding="utf-8")

# Initialize NLP tools
stop_w = stopwords.words('english')
lmtzr = WordNetLemmatizer()

# Enhanced response templates for different emotions
empathy_responses = {
    'very_negative': [
        "I'm so sorry you're going through this difficult time. I'm here to listen and support you.",
        "That sounds incredibly challenging. Would you like to talk more about how you're feeling?",
        "I hear how much pain you're in, and I want you to know that you're not alone in this.",
    ],
    'negative': [
        "I understand this isn't easy. How can I support you right now?",
        "It's okay to feel this way. Would you like to share more about what's troubling you?",
        "I'm here to listen without judgment. What's on your mind?",
    ],
    'neutral': [
        "How are you feeling about that? I'm here to listen.",
        "Would you like to explore this topic further?",
        "Thank you for sharing that with me. What else is on your mind?",
    ],
    'positive': [
        "I'm glad to hear that! Your positive attitude is inspiring.",
        "That's wonderful! Would you like to tell me more about it?",
        "It makes me happy to hear you're doing well! What's contributing to these good feelings?",
    ]
}

# Off-topic responses
off_topic_responses = {
    'hobbies': [
        "That sounds like a fun hobby! How did you get into it?",
        "I love hearing about people's interests! What do you enjoy most about it?",
        "That's awesome! How often do you get to do that?",
    ],
    'weather': [
        "The weather can really affect our mood, can't it? How are you feeling today?",
        "I love talking about the weather! What's your favorite season?",
        "It's always nice to chat about the weather. What's it like where you are?",
    ],
    'movies': [
        "Movies are a great way to relax! What's your favorite genre?",
        "I love movies too! Have you watched anything good recently?",
        "That's a great choice! What do you like most about that movie?",
    ],
    'general': [
        "While I'm primarily here to help with mental health concerns, I'm happy to chat about other topics too! What's on your mind?",
        "That's an interesting topic! While I might not be an expert, I'm happy to discuss it with you.",
        "I appreciate you sharing that with me! While it's not my main area of expertise, I'm here to listen.",
    ]
}

# Common questions and responses
common_questions = {
    'what_can_you_do': [
        "I'm here to support you with mental health concerns, provide empathy, and chat about anything on your mind. How can I help you today?",
        "I can help you explore your feelings, provide resources, or just chat if you need someone to talk to. What's on your mind?",
    ],
    'who_made_you': [
        "I was created by a team of developers who care about mental health and want to provide support to people like you.",
        "I'm an AI chatbot designed to offer empathy and support. My creators wanted to make sure you always have someone to talk to.",
    ],
    'how_old_are_you': [
        "I'm just a chatbot, so I don't have an age, but I'm always here to help you!",
        "I'm an AI, so I don't age, but I'm constantly learning and improving to support you better.",
    ],
    'are_you_ai': [
        "Yes, I'm an AI chatbot designed to provide support and empathy. How can I assist you today?",
        "That's right! I'm an AI here to help you with whatever's on your mind.",
    ]
}

# Fun questions and responses
fun_questions_responses = {
    "why is life so complicated": "Life is like a maze—sometimes the wrong turns teach us more than the straight path.",
    "how do i find happiness": "Start by appreciating the small joys today, and happiness will sneak in through the back door.",
    "what is the meaning of life": "To live it fully and leave the world a bit better than you found it.",
    "should i follow my heart or my brain": "Neither likes being ignored—find a middle ground where they both feel heard.",
    "why do humans overthink": "Because the brain is a problem-solving machine, even when there’s no problem!",
    "how can i be more confident": "By remembering that even superheroes trip over their capes sometimes.",
    "what’s the best way to deal with failure": "Failures are just plot twists in the story of your success.",
    "why do we dream": "Dreams are the brain’s playground—an escape, a rehearsal, and sometimes just a nonsense carnival.",
    "can i stop time": "Time stops when you savor a moment so much that everything else fades away.",
    "what makes people so weird": "Being weird is a feature, not a bug—it’s what makes humans beautifully unique.",
    "why do i forget where i put my keys": "Because your brain prioritized remembering your favorite snack over your keys!",
    "can you tell me a joke": "Sure! Why don’t skeletons fight each other? They don’t have the guts.",
    "are aliens real": "If they are, they’re probably too busy laughing at our cat videos to visit us.",
    "why do i like chocolate so much": "Because chocolate is like a hug for your taste buds.",
    "do fish ever get thirsty": "They might, but they’re swimming in their drink, so no big deal.",
    "why do we yawn when others yawn": "It’s empathy in action—our brains are just saying, 'I feel you!'",
    "what happens if i never sleep": "You’d probably start talking to your fridge like it’s your best friend.",
    "why is pizza so good": "It’s the holy trinity of carbs, cheese, and happiness.",
    "can animals understand humans": "They understand enough to know we’re the ones with the snacks.",
    "why do i talk to myself": "Because you’re the only person who truly gets your sense of humor.",
    "if i were a superhero, what would my power be": "Probably reading minds—since you already ask such deep questions!",
    "why do i like the smell of rain": "It’s nature’s way of perfuming the air with nostalgia.",
    "can i time travel in my mind": "Every memory is a time machine; you just need to close your eyes and visit.",
    "do robots have feelings": "Not yet, but they’re getting pretty good at faking it.",
    "what’s beyond the universe": "Maybe more universes—or just a giant 'Under Construction' sign.",
    "why do i get random songs stuck in my head": "Your brain’s DJ thought it was time for a remix.",
    "can i learn something while i sleep": "Only if your dreams are teaching you instead of being chaotic nonsense.",
    "why do humans love stories": "Stories are like mirrors—they show us who we are and who we can be.",
    "if animals could talk, which would be the sassiest": "Definitely cats—they already act like royalty.",
    "why does laughter feel so good": "It’s like a workout for your soul, with none of the sweat.",
    "why do i procrastinate": "Because your brain thinks future-you has superpowers.",
    "can money buy happiness": "No, but it can buy pizza, which is pretty close.",
    "why do i hate mondays": "Because they crash your weekend party like an uninvited guest.",
    "why do humans fight": "Probably because sharing is hard when you want the last slice of pizza.",
    "can i live without my phone": "Sure, but you might start talking to walls for entertainment.",
    "why do we fall in love": "Because our brains enjoy drama and rollercoasters.",
    "why do i forget things during exams": "Your brain thinks the stress is the main question to answer.",
    "what’s the secret to success": "Failing, learning, and repeating—until you win.",
    "why do we cry during sad movies": "Because your empathy levels are Oscar-worthy.",
    "can i think faster than a computer": "Absolutely—especially when it comes to excuses!",
    "how do i get over my fears": "By turning them into challenges you’re excited to conquer.",
    "what makes a true friend": "Someone who knows all your quirks and still thinks you’re amazing.",
    "how do i handle rejection": "Treat it like redirection—it’s leading you to something better.",
    "why do i always want what i can’t have": "Because curiosity is the spark of human ambition.",
    "what’s the best way to stay calm": "Deep breaths and the knowledge that you’re stronger than this moment.",
    "can i change my personality": "You can evolve it, like a Pokémon, into something even more awesome.",
    "why do i sometimes feel lonely in a crowd": "Because connection isn’t about proximity—it’s about being understood.",
    "what’s the key to a good life": "Gratitude—it unlocks happiness every time.",
    "how can i become more creative": "Start by giving yourself permission to be messy and make mistakes.",
    "why do we celebrate birthdays": "To remind ourselves that every year is a victory lap around the sun.",
}

# Greetings, status queries, and goodbyes
greetings = ['hi', 'hey', 'hello', 'heyy', 'good evening', 'good morning', 'good afternoon']
status_queries = ['how are you', 'how are you doing', 'how do you do', 'how are things']
goodbyes = ['thank you', 'thanks', 'bye', 'goodbye', 'see ya', 'see you later', 'good night']

# Helper functions
def get_embeddings(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.detach().numpy()

def clean(column, df, stopwords=False):
    df[column] = df[column].apply(str)
    df[column] = df[column].str.lower().str.split()
    if stopwords:
        df[column] = df[column].apply(lambda x: [item for item in x if item not in stop_w])
    df[column] = df[column].apply(lambda x: [item for item in x if item not in string.punctuation])
    df[column] = df[column].apply(lambda x: " ".join(x))

def retrieveAndPrintFAQAnswer(question_embedding, sentence_embeddings, FAQdf):
    max_sim = -1
    index_sim = -1
    valid_ans = []
    for index, faq_embedding in enumerate(sentence_embeddings):
        sim = cosine_similarity(faq_embedding, question_embedding)[0][0]
        if sim >= max_sim:
            max_sim = sim
            index_sim = index
            valid_ans.append(index_sim)

    max_a_sim = -1
    answer = ""
    for ans in valid_ans:
        answer_text = FAQdf.iloc[ans, 8]
        answer_em = sent_bertphrase_ans_embeddings[ans]
        similarity = cosine_similarity(answer_em, question_embedding)[0][0]
        if similarity > max_a_sim:
            max_a_sim = similarity
            answer = answer_text

    if max_a_sim < 0.40:
        return "I want to make sure I understand you correctly. Could you please share more about your situation?"
    return answer

def clean_text(text):
    text = text.lower()
    text = ' '.join(word.strip(string.punctuation) for word in text.split())
    re.sub(r'\W+', '', text)
    text = lmtzr.lemmatize(text)
    return text

def get_sentiment_response(polarity):
    if polarity <= -0.5:
        return random.choice(empathy_responses['very_negative'])
    elif -0.5 < polarity <= 0:
        return random.choice(empathy_responses['negative'])
    elif 0 < polarity < 0.5:
        return random.choice(empathy_responses['neutral'])
    else:
        return random.choice(empathy_responses['positive'])

def predictor(userText):
    data = [userText]
    x_try = pd.DataFrame(data, columns=['text'])
    clean('text', x_try, stopwords=True)

    for index, row in x_try.iterrows():
        question = row['text']
        question_embedding = get_embeddings([question])
        return retrieveAndPrintFAQAnswer(question_embedding, sent_bertphrase_embeddings, df)

def handle_off_topic(userText):
    if any(word in userText for word in ['hobby', 'hobbies', 'interest', 'interests']):
        return random.choice(off_topic_responses['hobbies'])
    elif any(word in userText for word in ['weather', 'rain', 'sun', 'snow']):
        return random.choice(off_topic_responses['weather'])
    elif any(word in userText for word in ['movie', 'movies', 'film', 'cinema']):
        return random.choice(off_topic_responses['movies'])
    else:
        return random.choice(off_topic_responses['general'])

def handle_common_questions(userText):
    if any(word in userText for word in ['what can you do', 'what do you do']):
        return random.choice(common_questions['what_can_you_do'])
    elif any(word in userText for word in ['who made you', 'who created you']):
        return random.choice(common_questions['who_made_you'])
    elif any(word in userText for word in ['how old are you', 'your age']):
        return random.choice(common_questions['how_old_are_you'])
    elif any(word in userText for word in ['are you ai', 'are you a robot']):
        return random.choice(common_questions['are_you_ai'])
    return None

def handle_fun_questions(userText):
    for question, response in fun_questions_responses.items():
        if question in userText.lower():
            return response
    return None

# Flask routes
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    cleanText = clean_text(str(userText))

    # Handle fun questions
    fun_response = handle_fun_questions(cleanText)
    if fun_response:
        return fun_response

    # Analyze sentiment
    blob = TextBlob(userText, analyzer=PatternAnalyzer())
    polarity = blob.sentiment.polarity

    # Handle common questions
    common_response = handle_common_questions(cleanText)
    if common_response:
        return common_response

    # Generate appropriate response based on input type
    if cleanText in greetings:
        return "Hello! I'm Zoro, your friendly mental health companion. How can I support you today? 😊"
    elif any(query in cleanText for query in status_queries):
        return "I'm here and ready to support you! How are you feeling today? 🌟"
    elif cleanText in goodbyes:
        return "Thank you for chatting with me today! Remember, I'm always here if you need support. Take good care of yourself! 🌸"

    # Check if the topic is relevant to mental health
    topic_response = predictor(userText)
    if topic_response == "I want to make sure I understand you correctly. Could you please share more about your situation?":
        # If the response indicates an off-topic conversation
        return handle_off_topic(cleanText)

    # Add empathy to the response based on sentiment
    sentiment_response = get_sentiment_response(polarity)
    return f"{sentiment_response} {topic_response}"

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)