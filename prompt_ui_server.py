from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import subprocess
import threading
import queue
import os
import sys
import json
import time

app = Flask(__name__)
app.config['SECRET_KEY'] = 'hemlo-secret-key-2024'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

PYTHON_PATH = os.path.join(os.path.dirname(__file__), "apps", "backend", ".venv", "Scripts", "python")
AGENT_SCRIPT = os.path.join(os.path.dirname(__file__), "hemlo_super_agent.py")
REMEMBER_FLAG_PATH = os.path.join(os.path.dirname(AGENT_SCRIPT), "remember_workflow.flag")
APPROVAL_FLAG_PATH = os.path.join(os.path.dirname(AGENT_SCRIPT), "money_approval.flag")
LOGIN_CHOICE_PATH = os.path.join(os.path.dirname(AGENT_SCRIPT), "login_choice.json")

CONTROL_PATH = os.path.join(os.path.dirname(AGENT_SCRIPT), "agent_control.json")
STATUS_PREFIX = "__HEMLO_STATUS__"
USER_DATA_DIR = os.path.join(os.path.expanduser("~"), ".hemlo_browser_data")

running_process = None
agent_state = "idle"
stop_requested = False


def _write_control(payload: dict) -> None:
    payload = dict(payload or {})
    payload["nonce"] = int(time.time() * 1000)
    tmp = CONTROL_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False))
    os.replace(tmp, CONTROL_PATH)


def _clear_agent_files() -> None:
    for p in [CONTROL_PATH, REMEMBER_FLAG_PATH, APPROVAL_FLAG_PATH, LOGIN_CHOICE_PATH]:
        try:
            if os.path.exists(p):
                os.remove(p)
        except Exception:
            pass


def _reset_browser_profile() -> None:
    try:
        if os.path.exists(USER_DATA_DIR):
            ts = __import__("time").strftime("%Y%m%d_%H%M%S")
            backup = USER_DATA_DIR + f"_reset_{ts}"
            os.rename(USER_DATA_DIR, backup)
    except Exception:
        pass

def run_agent(prompt, sid):
    """Run the agent script and stream output"""
    global running_process
    global agent_state
    global stop_requested
    
    try:
        _clear_agent_files()
        cmd = [PYTHON_PATH, "-u", AGENT_SCRIPT, "--prompt", prompt, "--session"]  # -u for unbuffered
        
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
        
        agent_state = "running"
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
                # Special channel: status emitted by agent
                if line_out.startswith(STATUS_PREFIX):
                    try:
                        payload = json.loads(line_out[len(STATUS_PREFIX):])
                        if isinstance(payload, dict):
                            agent_state = str(payload.get("status") or agent_state)
                            socketio.emit('status', payload, room=sid)
                    except Exception:
                        pass
                    continue

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

        final_state = agent_state
        if stop_requested:
            agent_state = "idle"
            socketio.emit('status', {'status': 'stopped'}, room=sid)
        elif final_state in {"completed", "stopped", "error"}:
            if final_state == "error":
                socketio.emit('status', {'status': 'error', 'message': 'Agent reported an error'}, room=sid)
            else:
                socketio.emit('status', {'status': final_state}, room=sid)
            agent_state = "idle"
        elif running_process.returncode == 0:
            agent_state = "idle"
            socketio.emit('status', {'status': 'completed'}, room=sid)
        else:
            agent_state = "error"
            socketio.emit('status', {'status': 'error', 'message': f'Process exited with code {running_process.returncode}'}, room=sid)
            
    except Exception as e:
        socketio.emit('status', {'status': 'error', 'message': str(e)}, room=sid)
    finally:
        running_process = None
        agent_state = "idle"
        stop_requested = False

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
        try:
            _write_control({"command": "new_task", "prompt": prompt})
            emit('status', {'status': 'running', 'message': 'New task sent to existing session'})
        except Exception as e:
            emit('status', {'status': 'error', 'message': f'Failed to send new task: {str(e)}'})
        return
    
    # Run in background thread
    thread = threading.Thread(target=run_agent, args=(prompt, request.sid))
    thread.daemon = True
    thread.start()

@socketio.on('stop_agent')
def handle_stop():
    global running_process
    global agent_state
    global stop_requested
    if running_process:
        try:
            stop_requested = True
            running_process.terminate()
            emit('status', {'status': 'stopped'})
            agent_state = "idle"
        except Exception as e:
            emit('status', {'status': 'error', 'message': f'Failed to stop: {str(e)}'})
    else:
        emit('status', {'status': 'idle'})


@socketio.on('pause_agent')
def handle_pause_agent():
    if not running_process:
        emit('status', {'status': 'idle'})
        return
    try:
        _write_control({"command": "pause"})
        emit('output', {'data': 'Pause requested'})
    except Exception as e:
        emit('status', {'status': 'error', 'message': f'Failed to pause: {str(e)}'})


@socketio.on('resume_agent')
def handle_resume_agent(data):
    if not running_process:
        emit('status', {'status': 'idle'})
        return
    payload = {"command": "resume"}
    try:
        prompt = (data or {}).get('prompt')
        if isinstance(prompt, str) and prompt.strip():
            payload["prompt"] = prompt.strip()
        _write_control(payload)
        emit('output', {'data': 'Resume requested'})
    except Exception as e:
        emit('status', {'status': 'error', 'message': f'Failed to resume: {str(e)}'})


@socketio.on('reset_session')
def handle_reset_session():
    global running_process
    global agent_state
    global stop_requested
    try:
        if running_process:
            try:
                stop_requested = True
                running_process.terminate()
                running_process.wait(timeout=5)
            except Exception:
                pass
            running_process = None
        _clear_agent_files()
        _reset_browser_profile()
        agent_state = "idle"
        emit('status', {'status': 'idle', 'message': 'Session reset complete'})
    except Exception as e:
        emit('status', {'status': 'error', 'message': f'Reset failed: {str(e)}'})

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
