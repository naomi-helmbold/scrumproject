
from flask import Flask, request, jsonify
app = Flask(__name__)

@app.route('/fact-check', methods=['POST'], endpoint='fact_check_endpoint')
def get_tweet():
    # Get the tweet from the incoming JSON request
    data = request.json
    user_input = data.get('tweet', '')  # Retrieve the tweet (or empty string if not present)
    # Return the tweet as a JSON response
    return jsonify({"tweet": user_input})

# The results variable is removed because get_tweet() should only be called during a request
results = get_tweet()

if __name__ == '__main__':
    app.run(debug=True)