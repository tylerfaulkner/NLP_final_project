#Uses python socketio to creat a connection ot the AWS server

import socketio
import time
import sys
import os
import subprocess

sio = socketio.Client()

sio.connect('http://ec2-3-15-183-255.us-east-2.compute.amazonaws.com:5000')

@sio.event
def connect():
    print('connection established')

@sio.event
def disconnect():
    print('disconnected from server')

@sio.on('summary')
def on_message(data):
    print('I received a message!')
    print(data)
    with open("tempScript.txt", "w") as f:
        f.write(data)
    process = subprocess.Popen(["python", "eval.py"])
    while process.poll() is None:
        time.sleep(1)
    with open("tempSumm.txt", "r") as f:
        summary = f.read()
    sio.emit('generatedSummary', summary)

