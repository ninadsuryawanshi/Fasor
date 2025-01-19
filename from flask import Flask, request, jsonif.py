from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return 'Welcome to my website!'

if __name__ == '__main__':
    app.run(debug=True)