from flask import Flask, render_template, request, jsonify, make_response
from flask_socketio import SocketIO, emit
import subprocess
import threading
import queue
import os
import sys
import json
import time
import re
import ast

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
planner_mode = "basic"
groq_api_key_override = None
overlay_log_buffer = []
overlay_login_options_payload = None  # Last login options payload for overlay/floating panel
log_mode = "dev"  # "dev" for raw logs, "user" for curated user-facing logs
floating_panel_process = None
FLOATING_PANEL_SCRIPT = os.path.join(os.path.dirname(__file__), "floating_panel.py")


def _apply_overlay_cors(resp):
    try:
        origin = request.headers.get("Origin")
    except Exception:
        origin = None
    try:
        resp.headers["Access-Control-Allow-Origin"] = origin or "*"
        resp.headers["Vary"] = "Origin"
    except Exception:
        resp.headers["Access-Control-Allow-Origin"] = "*"
    try:
        resp.headers["Access-Control-Allow-Private-Network"] = "true"
    except Exception:
        pass
    return resp


@socketio.on('set_log_mode')
def handle_set_log_mode(data):
    global log_mode
    try:
        mode = str((data or {}).get('mode') or '').strip().lower()
    except Exception:
        mode = ''
    if mode not in {'dev', 'dev2', 'user'}:
        return
    log_mode = mode
    emit('config', {'log_mode': log_mode}, broadcast=True)
    try:
        emit('output', {'data': f"Log mode set to: {log_mode.upper()}"})
    except Exception:
        pass


def _get_effective_groq_key() -> str:
    try:
        if groq_api_key_override:
            return str(groq_api_key_override)
    except Exception:
        pass
    try:
        return str(os.getenv("GROQ_API_KEY", "") or "")
    except Exception:
        return ""


def _get_groq_key_source() -> str:
    try:
        if groq_api_key_override:
            return "override"
    except Exception:
        pass
    try:
        return "env" if (os.getenv("GROQ_API_KEY") or "") else "none"
    except Exception:
        return "none"




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


def _ensure_floating_panel() -> None:
    global floating_panel_process
    try:
        if os.getenv("HEMLO_DISABLE_FLOATING_PANEL", "0") == "1":
            return
    except Exception:
        pass

    try:
        wk = os.environ.get("WERKZEUG_RUN_MAIN")
        if wk is not None and str(wk).lower() not in {"true", "1", "yes"}:
            return
    except Exception:
        pass

    try:
        if floating_panel_process and floating_panel_process.poll() is None:
            return
    except Exception:
        floating_panel_process = None

    try:
        if not os.path.exists(FLOATING_PANEL_SCRIPT):
            return
    except Exception:
        return

    py = sys.executable
    if not py:
        try:
            if os.path.exists(PYTHON_PATH):
                py = PYTHON_PATH
        except Exception:
            py = None
    if not py:
        return

    env = os.environ.copy()
    try:
        env.setdefault("HEMLO_PANEL_URL", "http://127.0.0.1:5000")
    except Exception:
        pass

    try:
        creationflags = 0
        if os.name == "nt":
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        floating_panel_process = subprocess.Popen([py, FLOATING_PANEL_SCRIPT], env=env, creationflags=creationflags)
    except Exception:
        floating_panel_process = None

def run_agent(prompt, sid, dom_mode=None, max_items=None):
    """Run the agent script and stream output"""
    global running_process
    global agent_state
    global stop_requested
    global overlay_log_buffer
    global overlay_login_options_payload
    
    try:
        _clear_agent_files()
        cmd = [PYTHON_PATH, "-u", AGENT_SCRIPT, "--prompt", prompt, "--session"]  # -u for unbuffered
        try:
            if isinstance(dom_mode, str) and dom_mode.strip():
                cmd.extend(["--dom_mode", dom_mode.strip()])
        except Exception:
            pass
        try:
            if max_items is not None:
                cmd.extend(["--max_items", str(int(max_items))])
        except Exception:
            pass
        
        # Set environment to disable Python buffering
        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'
        env['PYTHONIOENCODING'] = 'utf-8'
        mode = str(planner_mode or "basic").strip().lower()
        env['HEMLO_PLANNER_MODE'] = mode
        env['GEMINI_PLANNER_ASSIST'] = '1' if mode != 'off' else '0'
        if groq_api_key_override:
            env['GROQ_API_KEY'] = groq_api_key_override
        
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
                        overlay_login_options_payload = payload
                    except Exception:
                        pass
                    continue

                # Send line to UI
                socketio.emit('output', {'data': line_out}, room=sid)
                try:
                    overlay_log_buffer.append(line_out)
                    if len(overlay_log_buffer) > 8000:
                        overlay_log_buffer = overlay_log_buffer[-8000:]
                except Exception:
                    pass
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


def _build_user_logs_from_overlay(lines):
    user_lines = []
    current_has_plan = False
    current_has_filtering = False
    for raw in lines or []:
        try:
            s = str(raw)
        except Exception:
            s = ""
        if not s:
            continue

        if "Agent started with goal:" in s:
            current_has_plan = False
            current_has_filtering = False
            try:
                part = s.split("Agent started with goal:", 1)[-1].strip()
            except Exception:
                part = ""
            if part.startswith("'") and part.endswith("'") and len(part) >= 2:
                part = part[1:-1]
            if part:
                user_lines.append("Prompt: " + part)
            continue

        if "Navigating to " in s and "..." in s:
            try:
                part = s.split("Navigating to ", 1)[-1]
            except Exception:
                part = ""
            if part.endswith("..."):
                part = part[:-3]
            part = part.strip()
            if not current_has_plan:
                user_lines.append("Plan created")
                current_has_plan = True
            if part:
                user_lines.append("Link to open: " + part)
            continue

        if "Opening target=_blank href in same tab:" in s:
            try:
                part = s.split("Opening target=_blank href in same tab:", 1)[-1].strip()
            except Exception:
                part = ""
            if not current_has_plan:
                user_lines.append("Plan created")
                current_has_plan = True
            if part:
                user_lines.append("Link to open: " + part)
            continue

        if "Opening external host in same tab:" in s:
            try:
                part = s.split("Opening external host in same tab:", 1)[-1].strip()
            except Exception:
                part = ""
            if not current_has_plan:
                user_lines.append("Plan created")
                current_has_plan = True
            if part:
                user_lines.append("Link to open: " + part)
            continue

        if "Filtering DOM for goal" in s:
            if not current_has_filtering:
                user_lines.append("Filtering and analysing the website")
                current_has_filtering = True
            continue

        if "Deciding next action" in s:
            user_lines.append("Deciding next action to click")
            continue

        if "Action Result:" in s:
            res = ""
            url = ""
            try:
                part = s.split("Action Result:", 1)[-1].strip()
            except Exception:
                part = ""
            if part:
                try:
                    if "|" in part:
                        before, after = part.split("|", 1)
                    else:
                        before, after = part, ""
                    res = before.strip()
                    if "URL:" in after:
                        url = after.split("URL:", 1)[-1].strip()
                except Exception:
                    res = part.strip()
            msg = "Action result: " + (res or "Unknown")
            if url:
                msg += " (" + url + ")"
            user_lines.append(msg)
            continue

    return user_lines


def _build_dev2_logs_from_overlay(lines):
    """Build a sanitized dev-style log view.

    - Mirrors raw dev logs but replaces vendor / model names with 'HEMLO'
      (e.g. Gemini, Serper, OpenAI, Groq, etc.).
    - For lines that look like a Python dict 'decision' payload, collapse
      the dict down to just the action and name fields so the log is easier
      to skim (e.g. "decision: click -> K - Wikipedia").
    """
    out = []
    try:
        vendor_re = re.compile(r"(?i)\\b(gemini|serper|openai|groq|claude|anthropic|ollama)\\b")
    except Exception:
        vendor_re = None

    for raw in lines or []:
        try:
            s = str(raw)
        except Exception:
            s = ""
        if not s:
            continue

        # 1) Anonymize vendor / model names
        try:
            if vendor_re is not None:
                s_anon = vendor_re.sub("HEMLO", s)
            else:
                s_anon = s
        except Exception:
            s_anon = s

        # 2) Simplify decision dict lines
        simplified = False
        try:
            lower = s_anon.lower()
        except Exception:
            lower = s_anon
        try:
            if "decision:" in lower:
                # Grab everything after 'decision:' and try to parse a dict
                try:
                    after = s_anon.split("decision:", 1)[-1].strip()
                except Exception:
                    after = ""
                if after.startswith("{") and after.endswith("}"):
                    try:
                        payload = ast.literal_eval(after)
                    except Exception:
                        payload = None
                    if isinstance(payload, dict):
                        try:
                            action = str(payload.get("action") or "").strip()
                        except Exception:
                            action = ""
                        try:
                            name = str(
                                payload.get("name")
                                or payload.get("text")
                                or payload.get("label")
                                or ""
                            ).strip()
                        except Exception:
                            name = ""
                        if action or name:
                            msg = "decision:"
                            if action:
                                msg += f" {action}"
                            if name:
                                msg += f" -> {name}"
                            out.append(msg)
                            simplified = True
        except Exception:
            simplified = False

        if not simplified:
            out.append(s_anon)

    return out


@app.route("/overlay_logs", methods=["GET", "OPTIONS"])
def overlay_logs():
    if request.method == "OPTIONS":
        resp = make_response("")
        resp.headers["Access-Control-Allow-Methods"] = "GET,OPTIONS"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type, Access-Control-Request-Private-Network"
        return _apply_overlay_cors(resp)

    n = 2000
    try:
        if request.args.get("n") is not None:
            n = int(request.args.get("n") or 2000)
    except Exception:
        n = 2000
    if n < 50:
        n = 50
    if n > 8000:
        n = 8000

    try:
        raw_logs = overlay_log_buffer[-n:]
    except Exception:
        raw_logs = []

    try:
        mode = str(log_mode or "dev").strip().lower()
    except Exception:
        mode = "dev"

    if mode == "user":
        try:
            logs_out = _build_user_logs_from_overlay(raw_logs)
        except Exception:
            logs_out = []
    elif mode == "dev2":
        try:
            logs_out = _build_dev2_logs_from_overlay(raw_logs)
        except Exception:
            logs_out = raw_logs
    else:
        logs_out = raw_logs

    try:
        data = {"logs": logs_out[-n:], "state": agent_state}
    except Exception:
        data = {"logs": [], "state": agent_state}
    resp = make_response(json.dumps(data, ensure_ascii=False))
    resp.headers["Content-Type"] = "application/json"
    return _apply_overlay_cors(resp)


@app.route("/overlay_login_options", methods=["GET", "OPTIONS"])
def overlay_login_options():
    """Expose last seen login options for the floating panel.

    This mirrors the Socket.IO 'login_options' payload but over HTTP so the
    Tkinter floating panel can render provider buttons (Google, Email, etc.).
    """
    global overlay_login_options_payload

    if request.method == "OPTIONS":
        resp = make_response("")
        resp.headers["Access-Control-Allow-Methods"] = "GET,OPTIONS"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type, Access-Control-Request-Private-Network"
        return _apply_overlay_cors(resp)

    try:
        payload = overlay_login_options_payload or {}
    except Exception:
        payload = {}

    resp = make_response(json.dumps(payload, ensure_ascii=False))
    resp.headers["Content-Type"] = "application/json"
    return _apply_overlay_cors(resp)


@app.route("/overlay_control", methods=["GET", "POST", "OPTIONS"])
def overlay_control():
    global running_process
    global agent_state
    global stop_requested

    if request.method == "OPTIONS":
        resp = make_response("")
        resp.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type, Access-Control-Request-Private-Network"
        return _apply_overlay_cors(resp)

    cmd = ""
    if request.method == "GET":
        try:
            cmd = str(request.args.get("command") or "").lower()
        except Exception:
            cmd = ""
    else:
        try:
            payload = request.get_json(silent=True) or {}
        except Exception:
            payload = {}
        cmd = str(payload.get("command") or "").lower()
    ok = False

    if cmd == "stop":
        if running_process:
            try:
                stop_requested = True
                running_process.terminate()
                ok = True
            except Exception:
                ok = False
        else:
            ok = True
    elif cmd == "pause":
        try:
            _write_control({"command": "pause"})
            ok = True
        except Exception:
            ok = False
    elif cmd in {"resume", "new_task"}:
        try:
            _write_control({"command": "resume"})
            ok = True
        except Exception:
            ok = False
    elif cmd == "clear_logs":
        try:
            overlay_log_buffer.clear()
            ok = True
        except Exception:
            ok = False
    elif cmd in {"approve", "approve_money"}:
        try:
            with open(APPROVAL_FLAG_PATH, "w", encoding="utf-8") as f:
                f.write("1")
            ok = True
        except Exception:
            ok = False

    resp = make_response(json.dumps({"ok": bool(ok)}))
    resp.headers["Content-Type"] = "application/json"
    return _apply_overlay_cors(resp)


@app.route("/overlay_login_choice", methods=["POST", "OPTIONS"])
def overlay_login_choice():
    """Allow the floating panel to submit login choices via HTTP.

    This mirrors the Socket.IO 'submit_login_choice' handler but is accessed
    over HTTP so the Tkinter panel can write LOGIN_CHOICE_PATH for the agent.
    """
    if request.method == "OPTIONS":
        resp = make_response("")
        resp.headers["Access-Control-Allow-Methods"] = "POST,OPTIONS"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type, Access-Control-Request-Private-Network"
        return _apply_overlay_cors(resp)

    ok = False
    try:
        payload = request.get_json(silent=True) or {}
        with open(LOGIN_CHOICE_PATH, "w", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False))
        ok = True
    except Exception:
        ok = False

    resp = make_response(json.dumps({"ok": bool(ok)}))
    resp.headers["Content-Type"] = "application/json"
    return _apply_overlay_cors(resp)


@app.route('/')
def index():
    return render_template('prompt_ui_2.html')


@app.route('/ui2')
def prompt_ui_2():
    return render_template('prompt_ui_2.html')

@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('status', {'status': 'connected'})
    emit(
        'config',
        {
            'planner_mode': planner_mode,
            'groq_key_overridden': bool(groq_api_key_override),
            'groq_key_current': _get_effective_groq_key(),
            'groq_key_source': _get_groq_key_source(),
            'log_mode': log_mode,
        },
    )

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('run_prompt')
def handle_run_prompt(data):
    prompt = data.get('prompt', '')
    dom_mode = (data or {}).get('dom_mode')
    max_items = (data or {}).get('max_items')
    
    if not prompt:
        emit('status', {'status': 'error', 'message': 'No prompt provided'})
        return


    try:
        _ensure_floating_panel()
    except Exception:
        pass
    
    if running_process:
        try:
            payload = {"command": "new_task", "prompt": prompt}
            try:
                if isinstance(dom_mode, str) and dom_mode.strip():
                    payload["dom_mode"] = dom_mode.strip()
            except Exception:
                pass
            try:
                if max_items is not None:
                    payload["max_items"] = int(max_items)
            except Exception:
                pass
            _write_control(payload)
            emit('status', {'status': 'running', 'message': 'New task sent to existing session'})
        except Exception as e:
            emit('status', {'status': 'error', 'message': f'Failed to send new task: {str(e)}'})
        return
    
    # Run in background thread
    thread = threading.Thread(target=run_agent, args=(prompt, request.sid, dom_mode, max_items))
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
        dom_mode = (data or {}).get('dom_mode')
        if isinstance(dom_mode, str) and dom_mode.strip():
            payload["dom_mode"] = dom_mode.strip()
        max_items = (data or {}).get('max_items')
        if max_items is not None:
            try:
                payload["max_items"] = int(max_items)
            except Exception:
                pass
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


@socketio.on('set_planner_mode')
def handle_set_planner_mode(data):
    global planner_mode
    mode = str((data or {}).get('mode') or '').strip().lower()
    if mode not in {'off', 'basic', 'gold'}:
        return
    planner_mode = mode
    emit('config', {'planner_mode': planner_mode}, broadcast=True)
    try:
        emit('output', {'data': f"Planner mode set to: {planner_mode.upper()} (applies on next run)"})
    except Exception:
        pass


@socketio.on('set_groq_key')
def handle_set_groq_key(data):
    global groq_api_key_override
    key = (data or {}).get('key') or ""
    key = key.strip()
    groq_api_key_override = key or None
    emit(
        'config',
        {
            'groq_key_overridden': bool(groq_api_key_override),
            'groq_key_current': _get_effective_groq_key(),
            'groq_key_source': _get_groq_key_source(),
        },
        room=request.sid,
    )
    try:
        src = _get_groq_key_source()
        emit('output', {'data': f"GROQ API key updated (source={src}) (applies on next run)"})
    except Exception:
        pass

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
