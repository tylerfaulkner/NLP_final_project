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
    print("Summary received")
    print(summary)
    summary = summary

@app.route('/generateSummary', methods=['GET'])
def generateSummary():
    global summary
    text = request.body
    socketio.emit('summary', text)
    while summary is None:
        time.sleep(1)
    return summary


if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)