import streamlit as st
import re
import pickle
import streamlit.components.v1 as components
from hunspell import Hunspell

# Initialize Hunspell
h = Hunspell('ar', hunspell_data_dir='.')

# Load your model and vectorizer
vect = pickle.load(open("vectorizerBAG.pickle", "rb"))
model = pickle.load(open("bestModelBAG.pickle", "rb"))

# App title and introduction
st.title("Video Title Categorizer")
st.write("Welcome to the Video Title Categorizer! Leave a comment, and our AI model will predict the category of your video title.")

# Initialize session state for corrected text
if 'corrected_text' not in st.session_state:
    st.session_state['corrected_text'] = ""

# User input
text = st.text_input("Leave Your Comment", placeholder="Type your comment here...", value=st.session_state['corrected_text'])

# Function to check spelling and suggest corrections
def spellcheck_and_suggest(text):
    words = text.split()
    corrected_text = text
    suggestions = {}
    for word in words:
        if not h.spell(word):
            # Hunspell can suggest corrections for misspelled words
            suggested_corrections = h.suggest(word)
            if suggested_corrections:
                # Pick the first suggestion
                suggestions[word] = suggested_corrections[0]

    if suggestions:
        corrected_text = " ".join([suggestions.get(word, word) for word in words])
        st.session_state['corrected_text'] = corrected_text
        if st.button("Did You Mean ? : " + corrected_text):
            # If the user clicks on the suggestion, use the corrected text for further processing
            st.text_input("Leave Your Comment", value=corrected_text, key="corrected")
            predict(corrected_text)
    else:
        return text

# Text preprocessing function
def preprocess_text(text):
    tmp = re.sub(r'[^\u0620-\u063F\u0641-\u064A\u0660-\u0669]', ' ', str(text))
    tmp = re.sub(r'\d+', ' ', tmp)  # Remove numbers
    tmp = re.sub(r'\s+', ' ', tmp).strip()  # Remove extra spaces
    tmp = tmp.lower()
    return tmp

def predict(text):
    
    X = vect.transform([text]).toarray()
    probabilities = model.predict_proba(X)[0]
    max_probability = max(probabilities)
    max_index = probabilities.argmax()
    prediction = model.classes_[max_index]
    prediction_percentage = max_probability * 100
    # Display prediction results
    if max_probability < 0.8:
        st.warning(f"Prediction confidence is below 80%: {prediction_percentage:.2f}%. Classified as Other.")
    else:
        components.html(f"""
<div style="background-color: #2c3e50; border-radius: 10px; padding: 20px; text-align: center;margin:-20px">
    <h3 style="color: #18bc9c; margin: 0;">Prediction: <b>{prediction}</b></h3>
    <h4 style="color: #3498db; margin: 0;">Confidence: <b>{prediction_percentage:.2f}%</b></h4>
</div>
""", height=60) 

# Predict button
if st.button("Predict"):
    with st.spinner('Analyzing...'):
        text = preprocess_text(text)
        corrected_text = spellcheck_and_suggest(text)
        # Use the corrected text if suggestions were accepted, otherwise, use the original text
        predict(text)
