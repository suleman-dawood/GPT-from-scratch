from flask import Flask, render_template, request
from generate import generate_text

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')  # Make sure this points to your HTML file

@app.route('/generate', methods=['POST'])
def generate_endpoint():
    # Existing logic to generate text based on input
    try:
        data = request.get_json()
        length = int(data.get('length', 100))  # Default length is 100 if not provided

        result = generate_text(length)

        return result  # Return the generated text
    except Exception as e:
        return f"Error: {str(e)}", 500

if __name__ == '__main__':
    app.run(debug=True)
