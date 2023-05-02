#A flask SocketIO app that accepts a client connection from MSOE's rosie

from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
import time

app = Flask(__name__)
socketio = SocketIO(app)

summary = None

@socketio.on('connect')
def handle_connect():
    print("Client connected")

@socketio.on('disconnect')
def handle_disconnect():
    print("Client disconnected")

@socketio.on('generatedSummary')
def handle_summary(summary):
    print("Summary received in socketio")
    print(summary)
    summary = summary

@app.route('/generateSummary', methods=['GET'])
def generateSummary():
    global summary
    text = request.data.decode('utf-8')
    socketio.emit('summary', text)
    while summary is None:
        global summary
        print("Waiting for summary...")
        time.sleep(1)
    print("Summary received in request")
    return summary


if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)