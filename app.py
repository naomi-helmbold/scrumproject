from flask import Flask, render_template, request
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import requests
from bs4 import BeautifulSoup
from teapotai import TeapotAI
from flask import jsonify

teapot_ai = TeapotAI()


def analyze_sentiment(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity  

    if sentiment > 0:
        return 'Positif'
    elif sentiment < 0:
        return 'Négatif'
    else:
        return 'Neutre'


# Fonction pour identifier le sujet avec Latent Dirichlet Allocation (LDA)
def identify_topics(texts, n_topics=1, n_top_words=5):
    vectorizer = CountVectorizer(stop_words='english')
    dtm = vectorizer.fit_transform(texts)

    lda = LatentDirichletAllocation(n_components=n_topics, random_state=0)
    lda.fit(dtm)

    topics = []
    for idx, topic in enumerate(lda.components_):
        terms = [vectorizer.get_feature_names_out()[i] 
                 for i in topic.argsort()[:-n_top_words - 1:-1]]
        topics.extend(terms)
    return topics


def scrape_news(topics, websites):
    for website in websites:
        response = requests.get(website)
        soup = BeautifulSoup(response.content, 'html.parser')

    return soup


app = Flask(__name__)

@app.route('/', methods=['GET'])

def index():
    input_data = request.args.get('data')  # Get data from query string
    result = None

    if input_data:
        sentiment = analyze_sentiment(input_data)
        topics = identify_topics([input_data])
        websites = ['https://www.wikipedia.org/wiki/' + topic for topic in topics]

        for topic in topics:
            articles = scrape_news(topic, websites)

        Context = articles.text[articles.text.lower().find(topics[0]) - 1000 : articles.text.lower().find(topics[0]) + 1000]

        query = f"According to this context: {Context}, is the statement: {input_data} true? Give your answer as a yes or no."
        ans = teapot_ai.query(query=query)

        result = ans

    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)

