from flask import Flask, render_template, jsonify
from generate import generate_text

app = Flask(__name__)

# Home route to serve the index.html page
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_endpoint():
    try:
        result = generate_text(400)  # Generate text
        return result  # Return just the generated text as plain text
    except Exception as e:
        return str(e), 500

if __name__ == '__main__':
    app.run(debug=True)