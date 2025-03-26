from flask import Flask, render_template, request
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import requests
from bs4 import BeautifulSoup
from teapotai import TeapotAI

teapot_ai = TeapotAI()

def analyze_sentiment(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity  # Retourne une valeur entre -1 (négatif) et 1 (positif)
    
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
        terms = [vectorizer.get_feature_names_out()[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
        topics.extend(terms)
    return topics

def scrape_news(topics, websites):
    articles = []  
    for website in websites:
        response = requests.get(website)
        soup = BeautifulSoup(response.content, 'html.parser')

    return soup


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    # Get the data from the form
    input_data = request.form['data']
        
        
    UserInput = input_data

    sentiment = analyze_sentiment(UserInput)

    topics = identify_topics([UserInput])
    websites = []
    for topic in topics:
        websites.append('https://www.wikipedia.org/wiki/' + topic)

    #websites = ['https://www.bbc.com/news', 'https://www.cnn.com', 'https://www.wikipedia.org/', 'https://www.lemonde.fr/', 'https://www.foxnews.com/', 'https://www.nbcnews.com/']
    for topic in topics:
        articles = scrape_news(topic, websites)

    Context = articles.text[articles.text.lower().find(topics[0]) - 1000 : articles.text.lower().find(topics[0]) + 1000]

    query = "According to this context: {Context}, is the statement: {UserInput} true? Give your answer as a yes or no."
    query = query.format(Context=Context, UserInput=UserInput)

    ans = teapot_ai.query(
        query= query
    )
    
    return f"Processed Data: {ans}"

if __name__ == '__main__':
    app.run(debug=True)

