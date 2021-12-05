import flask
from flask import Flask, request
from inference import Inference

Infer = Inference()
app = Flask(__name__)


@app.route("/convert_api", methods=['POST'])
def convert_api():
    data = request.get_json()
    file_name, data_length = Infer.run(data['length'])
    data = {
        'file_name': file_name,
        'data_length': data_length
    }
    return flask.jsonify(data)


@app.route("/demo", methods=['GET'])
def demo():
    return flask.render_template('demo.html')


@app.route("/", methods=["GET"])
def hello():
    return flask.redirect("/demo")


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8091, debug=False)
