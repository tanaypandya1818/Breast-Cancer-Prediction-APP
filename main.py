import base64
import io
import PIL.Image as Image
from fastai.vision import *
from flask import Flask, request, jsonify

app = Flask("__name__")
learn = load_learner('/', 'tanay.h5')
@app.route('/')
def func():
    return "Use / predict"

@app.route('/predict', methods=['POST'])
def home():
    encodedimage = request.form.get('encoded_image')
    b = base64.b64decode(encodedimage)
    img = Image.open(io.BytesIO(b))
    img.save('mamogram.png')
    image = open_image('mamogram.png')
    result = learn.predict(image)
    print(result)

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)
