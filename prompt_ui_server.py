from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import subprocess
import threading
import queue
import os
import sys
import json

app = Flask(__name__)
app.config['SECRET_KEY'] = 'hemlo-secret-key-2024'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

PYTHON_PATH = os.path.join(os.path.dirname(__file__), "apps", "backend", ".venv", "Scripts", "python")
AGENT_SCRIPT = os.path.join(os.path.dirname(__file__), "hemlo_super_agent.py")
REMEMBER_FLAG_PATH = os.path.join(os.path.dirname(AGENT_SCRIPT), "remember_workflow.flag")
APPROVAL_FLAG_PATH = os.path.join(os.path.dirname(AGENT_SCRIPT), "money_approval.flag")
LOGIN_CHOICE_PATH = os.path.join(os.path.dirname(AGENT_SCRIPT), "login_choice.json")

running_process = None

def run_agent(prompt, sid):
    """Run the agent script and stream output"""
    global running_process
    
    try:
        cmd = [PYTHON_PATH, "-u", AGENT_SCRIPT, "--prompt", prompt]  # -u for unbuffered
        
        # Set environment to disable Python buffering
        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'
        env['PYTHONIOENCODING'] = 'utf-8'
        
        running_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Merge stderr into stdout
            text=True,
            bufsize=1,  # Line buffered
            universal_newlines=True,
            env=env,
            encoding="utf-8",      # Decode agent output as UTF-8
            errors="replace",       # Replace any invalid bytes instead of crashing
        )
        
        socketio.emit('status', {'status': 'running'}, room=sid)
        
        # Stream output line by line
        try:
            while True:
                line = running_process.stdout.readline()
                if not line:
                    # Check if process has finished
                    if running_process.poll() is not None:
                        break
                    continue
                
                line_out = line.rstrip()
                # Special channel: login options emitted by agent
                if line_out.startswith("__LOGIN_OPTIONS__"):
                    try:
                        payload = json.loads(line_out[len("__LOGIN_OPTIONS__"):])
                        socketio.emit('login_options', payload, room=sid)
                    except Exception:
                        pass
                    continue

                # Send line to UI
                socketio.emit('output', {'data': line_out}, room=sid)
                socketio.sleep(0)  # Allow other greenlets to run
                
        except Exception as e:
            print(f"Stream error: {e}")
            socketio.emit('output', {'data': f'Stream error: {str(e)}'}, room=sid)
        
        # Wait for process to complete
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

@socketio.on('save_workflow')
def handle_save_workflow():
    """Handle 'Remember' requests from the UI by creating a flag file.

    The running hemlo_super_agent process periodically checks for this flag
    and will persist the current in-memory workflow steps when it sees it.
    """
    try:
        with open(REMEMBER_FLAG_PATH, "w", encoding="utf-8") as f:
            f.write("1")
    except Exception as e:
        emit('status', {'status': 'error', 'message': f'Failed to set remember flag: {str(e)}'})

@socketio.on('approve_money_action')
def handle_approve_money_action():
    try:
        with open(APPROVAL_FLAG_PATH, "w", encoding="utf-8") as f:
            f.write("1")
    except Exception as e:
        emit('status', {'status': 'error', 'message': f'Failed to set approval flag: {str(e)}'})

@socketio.on('submit_login_choice')
def handle_submit_login_choice(data):
    """Receive login choice (provider + optional credentials) from UI."""
    try:
        with open(LOGIN_CHOICE_PATH, "w", encoding="utf-8") as f:
            f.write(json.dumps(data or {}, ensure_ascii=False))
    except Exception as e:
        emit('status', {'status': 'error', 'message': f'Failed to set login choice: {str(e)}'})

if __name__ == '__main__':
    socketio.run(app, debug=True, port=5000, host='0.0.0.0')
