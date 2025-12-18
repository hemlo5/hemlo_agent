from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import subprocess
import threading
import queue
import os
import sys

app = Flask(__name__)
app.config['SECRET_KEY'] = 'hemlo-secret-key-2024'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Path to the Python executable and script
PYTHON_PATH = os.path.join(os.path.dirname(__file__), "apps", "backend", ".venv", "Scripts", "python")
AGENT_SCRIPT = os.path.join(os.path.dirname(__file__), "hemlo_super_agent.py")

running_process = None

def run_agent(prompt, sid):
    """Run the agent script and stream output"""
    global running_process
    
    try:
        cmd = [PYTHON_PATH, AGENT_SCRIPT, "--prompt", prompt]
        
        running_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        socketio.emit('status', {'status': 'running'}, room=sid)
        
        # Stream output line by line
        for line in running_process.stdout:
            if line.strip():
                socketio.emit('output', {'data': line.strip()}, room=sid)
        
        running_process.wait()
        
        if running_process.returncode == 0:
            socketio.emit('status', {'status': 'completed'}, room=sid)
        else:
            socketio.emit('status', {'status': 'error', 'message': f'Process exited with code {running_process.returncode}'}, room=sid)
            
    except Exception as e:
        socketio.emit('status', {'status': 'error', 'message': str(e)}, room=sid)
    finally:
        running_process = None

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('status', {'status': 'connected'})

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('run_prompt')
def handle_run_prompt(data):
    prompt = data.get('prompt', '')
    
    if not prompt:
        emit('status', {'status': 'error', 'message': 'No prompt provided'})
        return
    
    if running_process:
        emit('status', {'status': 'error', 'message': 'Agent is already running'})
        return
    
    # Run in background thread
    thread = threading.Thread(target=run_agent, args=(prompt, request.sid))
    thread.daemon = True
    thread.start()

@socketio.on('stop_agent')
def handle_stop():
    global running_process
    if running_process:
        try:
            running_process.terminate()
            emit('status', {'status': 'stopped'})
        except Exception as e:
            emit('status', {'status': 'error', 'message': f'Failed to stop: {str(e)}'})
    else:
        emit('status', {'status': 'idle'})

if __name__ == '__main__':
    socketio.run(app, debug=True, port=5000, host='0.0.0.0')
