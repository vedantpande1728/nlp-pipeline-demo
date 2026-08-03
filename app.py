import streamlit as st
import nltk
import spacy
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# 1. Download NLTK data (cached so it only runs once and doesn't slow down the app)
@st.cache_resource
def setup_nltk():
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('stopwords')
    nltk.download('wordnet')  # Fixed missing resource for Lemmatization
    nltk.download('averaged_perceptron_tagger_eng')

setup_nltk()

# 2. Load spaCy model (cached to avoid reloading on every keystroke)
@st.cache_resource
def load_spacy():
    return spacy.load("en_core_web_sm")

nlp = load_spacy()

st.title("NLP Pipeline Demo")
st.write("Enter text and explore NLP steps")

# Added a default sentence with proper nouns so you can see NER working immediately
default_text = "Sundar Pichai is the CEO of Google, which is headquartered in California."
text = st.text_area("Enter your text here:", value=default_text)

if text:

    st.subheader("1. Sentence Tokenization")
    sentences = sent_tokenize(text)
    st.write(sentences)

    st.subheader("2. Word Tokenization")
    tokens = word_tokenize(text)
    st.write(tokens)

    st.subheader("3. Stemming")
    stemmer = PorterStemmer()
    stemmed = [stemmer.stem(word) for word in tokens]
    st.write(stemmed)

    st.subheader("4. Lemmatization")
    lemmatizer = WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(word) for word in tokens]
    st.write(lemmatized)

    st.subheader("5. Stopword Removal")
    stop_words = set(stopwords.words('english'))
    filtered = [word for word in tokens if word.lower() not in stop_words]
    st.write(filtered)

    st.subheader("6. POS Tagging")
    pos_tags = nltk.pos_tag(tokens)
    st.write(pos_tags)

    st.subheader("7. Named Entity Recognition (NER)")
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    
    # Check if entities exist, otherwise show a helpful warning
    if entities:
        st.write(entities)
    else:
        st.warning("No entities found. Ensure proper nouns are capitalized (e.g., 'Google' instead of 'google').")

    st.subheader("8. Chunking")
    grammar = "NP: {<DT>?<JJ>*<NN>}"
    # Fixed invalid Streamlit attribute assignment
    chunk_parser = nltk.RegexpParser(grammar)
    tree = chunk_parser.parse(pos_tags)
    
    # Display the tree structure properly formatted
    st.text(str(tree))