import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegressionCV
from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
import nltk
import joblib
nltk.download('punkt')
nltk.download('stopwords')


@st.cache_data
def load_data():
    data1 = pd.read_csv("D:/streamlit/sentiment_dataset_10000.csv")
    data2 = pd.read_csv("D:/streamlit/training2_preprocessed_data.csv")
    combined_data = pd.concat([data1, data2], ignore_index=True)
    return combined_data

st.sidebar.title("Sentiment Analysis App")


with st.spinner("Loading data..."):
    data = load_data().dropna(subset=['text'])


tfidf = TfidfVectorizer(min_df=2, max_df=0.9, stop_words='english', ngram_range=(1, 2), max_features=10000)
X = tfidf.fit_transform(data['text'])
y = data['sentiment']

pipeline = make_pipeline(
    SMOTE(random_state=42, n_jobs=-1),
    NearMiss(version=3, n_jobs=-1),
    LogisticRegressionCV(cv=5, max_iter=1000, n_jobs=-1, random_state=42)
)

with st.spinner("Training model..."):
    model = pipeline.fit(X, y)

joblib.dump(model, 'sentiment_model.joblib')
joblib.dump(tfidf, 'tfidf_vectorizer.joblib')

st.success("Model trained successfully!")


st.title("Sentiment Analysis App")      


user_input = st.text_area("Enter some text:")


if st.button("Analyze Sentiment"):
    if user_input.strip():
        
        user_input_vectorized = tfidf.transform([user_input])
        
        predicted_sentiment = model.predict(user_input_vectorized)[0]
       
        st.write(f"Predicted Sentiment: {predicted_sentiment}")
    else:
        st.write("Please enter some text.")


uploaded_file = st.file_uploader("Upload a dataset", type=["csv", "txt"])

if uploaded_file is not None:
    
    df = pd.read_csv(uploaded_file)

    st.write("Uploaded dataset:")
    st.write(df)

    
    text_column = st.selectbox("Select the column containing text data", df.columns)

   
    X_new = tfidf.transform(df[text_column])

    
    with st.spinner("Performing sentiment analysis..."):
        sentiment_results = model.predict(X_new)

    df['sentiment'] = sentiment_results

    st.write("Final table with sentiment analysis results:")
    st.write(df)

    show_positive = st.checkbox("Show Positive Sentiments")
    show_negative = st.checkbox("Show Negative Sentiments")
    show_neutral = st.checkbox("Show Neutral Sentiments")

    if show_positive:
        st.write("Positive Sentiments:")
        st.write(df[df['sentiment'] == 'positive'])

    if show_negative:
        st.write("Negative Sentiments:")
        st.write(df[df['sentiment'] == 'negative'])

    if show_neutral:
        st.write("Neutral Sentiments:")
        st.write(df[df['sentiment'] == 'neutral'])
