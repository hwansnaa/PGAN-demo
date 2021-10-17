import flask
from flask import Flask
from inference import Inference

Infer = Inference()
app = Flask(__name__)


@app.route("/convert_api", methods=['POST'])
def convert_api(data):
    output_list = Infer.run(data.length)
    return output_list


@app.route("/demo", methods=['GET'])
def demo():
    return flask.render_template('demo.html')


@app.route("/", methods=["GET"])
def hello():
    return flask.redirect("/demo")


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9000, debug=True)
