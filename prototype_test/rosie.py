#Uses python socketio to creat a connection ot the AWS server

import socketio
import time
import sys
import os
import subprocess

sio = socketio.Client()

sio.connect('http://44.201.114.68:5000')

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
    process = subprocess.Popen("srun --partition dgx --gpus=v100:1 singularity exec --nv /data/containers/msoe-pycuda-11.7.1.sif python3 eval.py", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    while process.poll() is None:
        print("Waiting for process to finish...")
        time.sleep(1)
    with open("tempSumm.txt", "r") as f:
        summary = f.read()
    sio.emit('generatedSummary', summary)
