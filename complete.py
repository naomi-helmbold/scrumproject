from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import requests
from bs4 import BeautifulSoup


# Fonction pour analyser les sentiments avec TextBlob
def analyze_sentiment(text):

    blob = TextBlob(text)

    # Retourne une valeur entre -1 (négatif) et 1 (positif)
    sentiment = blob.sentiment.polarity

    if sentiment > 0:
        return 'Positif'
    elif sentiment < 0:
        return 'Négatif'
    else:
        return 'Neutre'


# Fonction pour identifier le sujet avec Latent Dirichlet Allocation (LDA)
def identify_topics(texts, n_topics=2, n_top_words=5):
    vectorizer = CountVectorizer(stop_words='english')
    dtm = vectorizer.fit_transform(texts)

    lda = LatentDirichletAllocation(n_components=n_topics, random_state=0)
    lda.fit(dtm)

    topics = []
    for idx, topic in enumerate(lda.components_):
        terms = [vectorizer.get_feature_names_out()[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
        topics.append("Topic %d: %s" % (idx, ", ".join(terms)))
    return topics


tweet_content = "France is in europe"

print("Contenu du tweet : ", tweet_content)

# Analyse des sentiments
sentiment = analyze_sentiment(tweet_content)
print("Sentiment détecté : ", sentiment)

# Identification du sujet
topics = identify_topics([tweet_content])
print("Sujets identifiés :")
for topic in topics:
    print(topic)


# Fonction pour scraper les articles de news
def scrape_news(topics, websites):
    articles = []
    for website in websites:
        response = requests.get(website)
        soup = BeautifulSoup(response.content, 'html.parser')

        for link in soup.find_all('a', href=True):
            link_text = link.text.strip().lower()

            if all(topic.lower() in link_text for topic in topics):
                articles.append({'title': link.text.strip(), 'url': link['href']})

    return articles


topics = ['France', 'Europe']
websites = ['https://www.bbc.com/news', 'https://www.cnn.com', 'https://www.wikipedia.org/', 'https://www.lemonde.fr/', 'https://www.washingtonpost.com/', 'https://www.foxnews.com/', 'https://www.nbcnews.com/']
articles = scrape_news(topics, websites)

for article in articles:
    print(f"Title: {article['title']}, URL: {article['url']}")
