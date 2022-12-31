from flask import Flask, render_template, request, jsonify
from model import model, transform
import numpy as np
import base64
import cv2

# Initialize app
app = Flask(__name__)


# Get
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


# Post
@app.route('/', methods=['POST'])
def canvas():
    # Receive base64 data from the user form
    canvasdata = request.form['data'].split(',')[1]
    bytes_array = base64.b64decode(canvasdata)
    arr = np.fromstring(bytes_array, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = transform(img_gray)
    img = img.unsqueeze(0)
    pr = model(img).detach().numpy()[0]
    pr *= 100
    return render_template('scores.html', response=list(enumerate(pr)))


if __name__ == "__main__":
    app.run()
