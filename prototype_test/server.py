# A flask SocketIO app that accepts a client connection from MSOE's rosie

from flask import Flask, render_template, request, Blueprint
from flask_socketio import SocketIO, emit
import time
import os
blueprint = Blueprint('blueprint', __name__)

app = Flask(__name__)
socketio = SocketIO(app)

summary = None

# put this sippet ahead of all your bluprints
# blueprint can also be app~~
@blueprint.after_request 
def after_request(response):
    header = response.headers
    header['Access-Control-Allow-Origin'] = 'http://nlp-prototype.tyler-faulkner.com'
    # Other headers can be added here if needed
    return response


@socketio.on('connect')
def handle_connect():
    print("Client connected")


@socketio.on('disconnect')
def handle_disconnect():
    print("Client disconnected")


@socketio.on('generatedSummary')
def handle_summary(summary):
    print("Summary received in socketio")
    # write summary to file
    with open("tempSumm.txt", "w") as f:
        f.write(summary)


@app.route('/generateSummary', methods=['GET'])
def generateSummary():
    global summary
    filename = request.args.get('filename')
    text = ''
    with open(f'test_data/{filename}.txt', "r") as f:
        text = f.read()

    # remove temp sum file if exists
    try:
        os.remove("tempSumm.txt")
    except:
        pass
    socketio.emit('summary', text)
    # Check if summary file exists

    while os.path.isfile("tempSumm.txt") == False:
        print("Waiting for summary...")
        time.sleep(1)

    # read summary file
    with open("tempSumm.txt", "r") as f:
        summary = f.read()
    print("Summary received in flask")
    return summary


if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)
