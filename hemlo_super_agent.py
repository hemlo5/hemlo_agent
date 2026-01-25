import os
import sys
import json
import time
import re
import math
import hashlib
import contextlib
import typing
import threading
from typing import List, Dict, Any, Optional, Union, Set, Tuple
from urllib.parse import urlparse
import urllib.request, urllib.error
from dotenv import load_dotenv
from playwright.sync_api import sync_playwright, Page
from openai import OpenAI

OVERLAY_AGENT_LOG_BUFFER: List[str] = []
OVERLAY_AGENT_LOG_LOCK = threading.Lock()
OVERLAY_AGENT_STATE = "idle"

# Fix Windows console encoding for emojis and Unicode characters
# and ensure line-buffered output so logs appear in real time.
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer,
        encoding='utf-8',
        errors='replace',
        line_buffering=True,
    )
    sys.stderr = io.TextIOWrapper(
        sys.stderr.buffer,
        encoding='utf-8',
        errors='replace',
        line_buffering=True,
    )


class _TeeTextIO:
    def __init__(self, wrapped):
        self._wrapped = wrapped
        self._partial = ""

    def write(self, s):
        try:
            txt = str(s)
        except Exception:
            txt = ""
        if txt:
            try:
                self._partial += txt
                while "\n" in self._partial:
                    line, self._partial = self._partial.split("\n", 1)
                    try:
                        sline = str(line or "").replace("\r", "")
                        if sline.strip():
                            if len(sline) > 900:
                                sline = sline[:900] + "..."
                            with OVERLAY_AGENT_LOG_LOCK:
                                OVERLAY_AGENT_LOG_BUFFER.append(sline)
                                if len(OVERLAY_AGENT_LOG_BUFFER) > 8000:
                                    del OVERLAY_AGENT_LOG_BUFFER[:-8000]
                    except Exception:
                        pass
            except Exception:
                pass
        return self._wrapped.write(s)

    def flush(self):
        return self._wrapped.flush()

    def __getattr__(self, name):
        return getattr(self._wrapped, name)


try:
    sys.stdout = _TeeTextIO(sys.stdout)
except Exception:
    pass
try:
    sys.stderr = _TeeTextIO(sys.stderr)
except Exception:
    pass

# Load environment variables
load_dotenv()

# --- Configuration ---
class Config:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    SMART_MODEL = "llama-3.3-70b-versatile"  # Groq (Planning/Reasoning)
    GEMINI_API_KEY = os.getenv("OPENROUTER_API_KEY") or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    GEMINI_MODEL = os.getenv("PLANNER_MODEL") or os.getenv("GEMINI_MODEL") or "openai/gpt-4.1-mini"
    GEMINI_PLANNER_ASSIST = os.getenv("GEMINI_PLANNER_ASSIST", "1") != "0"
    PLANNER_MODE = (os.getenv("HEMLO_PLANNER_MODE", "basic") or "basic").strip().lower()
    SUCCESS_JUDGE_ENABLED = os.getenv("HEMLO_SUCCESS_JUDGE", "1") != "0"
    SUCCESS_JUDGE_MODEL = os.getenv("SUCCESS_JUDGE_MODEL") or GEMINI_MODEL
    SUCCESS_JUDGE_MIN_CONF = float(os.getenv("SUCCESS_JUDGE_MIN_CONF", "0.8"))
    SUCCESS_JUDGE_MAX_TOKENS = int(os.getenv("SUCCESS_JUDGE_MAX_TOKENS", "220"))
    WORKFLOW_MEMORY_ENABLED = os.getenv("WORKFLOW_MEMORY_ENABLED", "1") != "0"
    REDIS_URL = os.getenv("WORKFLOW_REDIS_URL") or os.getenv("REDIS_URL")
    WORKFLOW_SIM_THRESHOLD = float(os.getenv("WORKFLOW_SIM_THRESHOLD", "0.8"))
    WORKFLOW_MAX_PER_SITE = int(os.getenv("WORKFLOW_MAX_PER_SITE", "100"))
    WORKFLOW_TTL_SECONDS = int(os.getenv("WORKFLOW_TTL_SECONDS", str(30 * 24 * 3600)))
    WORKFLOW_MIN_STEP_CONF = float(os.getenv("WORKFLOW_MIN_STEP_CONF", "0.9"))
    DEBUG_AX_SNAPSHOT = os.getenv("HEMLO_DEBUG_AX_SNAPSHOT", "0") == "1"
    DEBUG_RICH_INPUTS = os.getenv("HEMLO_DEBUG_RICH_INPUTS", "0") == "1"
    DEBUG_PLANNER_ASSIST = os.getenv("HEMLO_DEBUG_PLANNER_ASSIST", "0") == "1"

config = Config()
try:
    if config.PLANNER_MODE not in {"off", "basic", "gold"}:
        config.PLANNER_MODE = "basic"
except Exception:
    config.PLANNER_MODE = "basic"

if not config.GROQ_API_KEY:
    print("Error: GROQ_API_KEY not found in .env")
    sys.exit(1)

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=config.GROQ_API_KEY,
    timeout=20.0,
    max_retries=0,
)
DOWNLOAD_DIR = os.getenv("HEMLO_DOWNLOAD_DIR") or os.path.join(os.path.expanduser("~"), "Downloads", "hemlo_agent")
try:
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
except Exception:
    pass

USER_DATA_DIR = os.path.join(os.path.expanduser("~"), ".hemlo_browser_data")
try:
    os.makedirs(USER_DATA_DIR, exist_ok=True)
except Exception:
    pass

# Flag file used when the UI "Remember" button is pressed
REMEMBER_FLAG_PATH = os.path.join(os.path.dirname(__file__), "remember_workflow.flag")
# Flag file used when the UI Approve button is pressed for money actions
APPROVAL_FLAG_PATH = os.path.join(os.path.dirname(__file__), "money_approval.flag")
# File used when the UI submits a login choice/credentials
LOGIN_CHOICE_PATH = os.path.join(os.path.dirname(__file__), "login_choice.json")

CONTROL_PATH = os.path.join(os.path.dirname(__file__), "agent_control.json")
STATUS_PREFIX = "__HEMLO_STATUS__"

UPLOAD_LIKE_TOKENS = [
    "upload your image",
    "upload",
    "upload image",
    "upload photo",
    "upload video",
    "choose file",
    "choose files",
    "choose image",
    "choose photo",
    "select file",
    "select files",
    "browse files",
    "browse your files",
    "pick file",
    "pick files",
    "add image",
    "add photo",
    "add media",
]


def _is_upload_like_label(label: Any) -> bool:
    try:
        lab = str(label or "").lower()
    except Exception:
        lab = ""
    if not lab:
        return False
    try:
        if re.search(r"\bupload\b", lab):
            return True
    except Exception:
        pass
    try:
        for tok in UPLOAD_LIKE_TOKENS:
            if tok == "upload":
                continue
            if tok and tok in lab:
                return True
    except Exception:
        pass
    return False


def _run_memory_add_milestone(run_memory: Optional[Dict[str, Any]], milestone: str) -> None:
    if not isinstance(run_memory, dict):
        return
    try:
        ms = run_memory.get("milestones")
        if not isinstance(ms, list):
            ms = []
            run_memory["milestones"] = ms
        if milestone and milestone not in ms:
            ms.append(milestone)
    except Exception:
        return


def _run_memory_add_progress(run_memory: Optional[Dict[str, Any]], note: str, limit: int = 40) -> None:
    if not isinstance(run_memory, dict):
        return
    try:
        notes = run_memory.get("progress_notes")
        if not isinstance(notes, list):
            notes = []
            run_memory["progress_notes"] = notes
        clean = str(note or "").strip()
        if not clean:
            return
        notes.append(clean)
        if len(notes) > int(limit or 40):
            run_memory["progress_notes"] = notes[-int(limit or 40):]
    except Exception:
        return


def _run_memory_choice_key_from_decision(decision: Optional[Dict[str, Any]]) -> str:
    if not isinstance(decision, dict):
        return ""
    try:
        lt = str(decision.get("locator_type") or "").lower().strip()
    except Exception:
        lt = ""
    if lt == "css" and decision.get("locator"):
        try:
            return "css:" + str(decision.get("locator"))
        except Exception:
            return "css:"
    try:
        r = str(decision.get("role") or "")
    except Exception:
        r = ""
    try:
        n = str(decision.get("name") or "")
    except Exception:
        n = ""
    return f"role:{r}:{n}"


def _run_memory_choice_key_from_item(it: Optional[Dict[str, Any]]) -> str:
    if not isinstance(it, dict):
        return ""
    try:
        lt = str(it.get("locator_type") or "").lower().strip()
    except Exception:
        lt = ""
    if lt == "css" and it.get("locator"):
        try:
            return "css:" + str(it.get("locator"))
        except Exception:
            return "css:"
    try:
        r = str(it.get("role") or "")
    except Exception:
        r = ""
    try:
        n = str(it.get("name") or "")
    except Exception:
        n = ""
    return f"role:{r}:{n}"

# --- Workflow Memory ---

class WorkflowMemory:
    def __init__(self, redis_url: Optional[str]):
        self.redis_url = redis_url
        self.client = None
        self.enabled = bool(redis_url)
        self.sim_threshold = config.WORKFLOW_SIM_THRESHOLD
        self.max_per_site = config.WORKFLOW_MAX_PER_SITE
        self.ttl = config.WORKFLOW_TTL_SECONDS
        self.min_conf = config.WORKFLOW_MIN_STEP_CONF
        self._embedder = None
        if self.enabled and self.redis_url:
            self.client = self._create_client(self.redis_url)
            if self.client is None:
                self.enabled = False

    def _create_client(self, url: str):
        try:
            import redis  # type: ignore

            r = redis.Redis.from_url(url, decode_responses=True)
            r.ping()
            return r
        except Exception:
            return None

    def _ensure_embedder(self):
        if self._embedder is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            self._embedder = None

    def _embed_prompt(self, text: str) -> Optional[List[float]]:
        self._ensure_embedder()
        if self._embedder is None:
            return None
        try:
            vec = self._embedder.encode([text])[0]
            return [float(x) for x in vec]
        except Exception:
            return None

    def _cosine(self, a: List[float], b: List[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        s = 0.0
        na = 0.0
        nb = 0.0
        for x, y in zip(a, b):
            s += x * y
            na += x * x
            nb += y * y
        if na <= 0.0 or nb <= 0.0:
            return 0.0
        return s / (math.sqrt(na) * math.sqrt(nb))

    def _workflow_key(self, site: str, workflow_id: str) -> str:
        return f"workflow:{site}:{workflow_id}"

    def _lru_key(self, site: str) -> str:
        return f"workflow:index:lru:{site}"

    def _prompt_key(self, site: str, prompt_hash: str) -> str:
        return f"workflow:index:prompt:{site}:{prompt_hash}"

    def _load_workflow(self, site: str, workflow_id: str) -> Optional[Dict[str, Any]]:
        if not self.client:
            return None
        key = self._workflow_key(site, workflow_id)
        raw = self.client.get(key)
        if not raw:
            return None
        try:
            data = json.loads(raw)
            data["id"] = workflow_id
            return data
        except Exception:
            return None

    def lookup(self, site: str, prompt: str) -> Optional[Dict[str, Any]]:
        if not self.enabled or not self.client or not site:
            return None
        prompt_norm = (prompt or "").strip().lower()
        if not prompt_norm:
            return None
        h = hashlib.sha1(prompt_norm.encode("utf-8")).hexdigest()
        prompt_key = self._prompt_key(site, h)
        wf_id = self.client.get(prompt_key)
        if wf_id:
            wf = self._load_workflow(site, wf_id)
            if wf:
                now = int(time.time())
                self.client.zadd(self._lru_key(site), {wf_id: now})
                wf["last_used_at"] = now
                return wf
            else:
                try:
                    self.client.delete(prompt_key)
                except Exception:
                    pass
        emb = self._embed_prompt(prompt_norm)
        if emb is None:
            return None
        lru_key = self._lru_key(site)
        try:
            ids = self.client.zrange(lru_key, 0, -1)
        except Exception:
            return None
        best = None
        best_sim = 0.0
        now = int(time.time())
        for wf_id in ids:
            wf = self._load_workflow(site, wf_id)
            if not wf:
                continue
            w_emb = wf.get("prompt_embedding") or []
            sim = self._cosine(emb, w_emb)
            if sim > best_sim:
                best_sim = sim
                best = wf
        if best and best_sim >= self.sim_threshold:
            try:
                self.client.zadd(lru_key, {best["id"]: now})
            except Exception:
                pass
            best["last_used_at"] = now
            best["similarity"] = best_sim
            return best
        return None

    def save(self, site: str, prompt: str, steps: List[Dict[str, Any]], source: str = "auto"):
        if not self.enabled or not self.client or not site or not steps:
            return
        prompt_raw = (prompt or "").strip()
        prompt_norm = prompt_raw.lower()
        prompt_hash = hashlib.sha1(prompt_norm.encode("utf-8")).hexdigest()
        emb = self._embed_prompt(prompt_norm)
        now = int(time.time())
        base = f"{site}:{prompt_hash}:{now}:{source}"
        wf_id = hashlib.sha1(base.encode("utf-8")).hexdigest()
        doc: Dict[str, Any] = {
            "id": wf_id,
            "site": site,
            "prompt": prompt_raw,
            "prompt_hash": prompt_hash,
            "prompt_embedding": emb,
            "created_at": now,
            "last_used_at": now,
            "success_count": 1,
            "source": source,
            "steps": steps,
        }
        key = self._workflow_key(site, wf_id)
        try:
            print(f"[WorkflowMemory] Saving workflow '{wf_id}' (source={source}) for site '{site}' with {len(steps)} steps for prompt: {prompt_raw!r}")
            try:
                for s in steps:
                    try:
                        print(
                            "[WorkflowMemory]  Step {order}: action={action}, locator_type={lt}, role={role}, name={name}, selector={sel}, url_before={ub}, url_after={ua}, confidence={conf}".format(
                                order=s.get("order"),
                                action=s.get("action"),
                                lt=s.get("locator_type"),
                                role=s.get("role"),
                                name=s.get("name"),
                                sel=s.get("selector"),
                                ub=s.get("url_before"),
                                ua=s.get("url_after"),
                                conf=s.get("confidence"),
                            )
                        )
                    except Exception:
                        pass
            except Exception:
                pass

            self.client.set(key, json.dumps(doc), ex=self.ttl)
            self.client.set(self._prompt_key(site, prompt_hash), wf_id, ex=self.ttl)
            lru_key = self._lru_key(site)
            self.client.zadd(lru_key, {wf_id: now})
            cnt = self.client.zcard(lru_key)
            if cnt and cnt > self.max_per_site:
                excess = int(cnt - self.max_per_site)
                old_ids = self.client.zrange(lru_key, 0, excess - 1)
                for oid in old_ids:
                    self.client.zrem(lru_key, oid)
                    self.client.delete(self._workflow_key(site, oid))
        except Exception as e:
            print(f"[WorkflowMemory] Error while saving workflow: {e}")

    def replay(self, page: Page, workflow: Dict[str, Any], execute_action_fn, downloaded_files: Optional[List[str]]) -> bool:
        if not self.enabled or not self.client:
            return False
        steps = workflow.get("steps") or []
        site = workflow.get("site") or ""
        wf_id = workflow.get("id") or ""
        if not steps:
            print(f"[WorkflowMemory] No steps found for workflow '{wf_id}' on site '{site}'.")
            return False
        if len(steps) < 2:
            print(f"[WorkflowMemory] Workflow '{wf_id}' for site '{site}' only has {len(steps)} step(s); treating as incomplete and falling back to live planning.")
            return False

        print(f"[WorkflowMemory] Replaying workflow '{wf_id}' for site '{site}' with {len(steps)} steps...")

        for step in sorted(steps, key=lambda s: s.get("order", 0)):
            decision: Dict[str, Any] = {
                "action": step.get("action"),
                "locator_type": step.get("locator_type"),
                "role": step.get("role"),
                "name": step.get("name"),
                "locator": step.get("selector"),
                "value": step.get("input_value"),
                "confidence": step.get("confidence", 1.0),
                "dom_hash": None,
            }
            try:
                print(
                    "[WorkflowMemory]  Replay step {order}: action={action}, locator_type={lt}, role={role}, name={name}, selector={sel}, confidence={conf}".format(
                        order=step.get("order"),
                        action=decision.get("action"),
                        lt=decision.get("locator_type"),
                        role=decision.get("role"),
                        name=decision.get("name"),
                        sel=decision.get("locator"),
                        conf=decision.get("confidence"),
                    )
                )
            except Exception:
                pass

            ok = False
            try:
                ok = bool(execute_action_fn(page, decision, downloaded_files))
            except Exception as e:
                print(f"[WorkflowMemory]  Replay step error: {e}")
                ok = False
            print(f"[WorkflowMemory]  Replay step {step.get('order')} result: {'Success' if ok else 'Fail'}")
            if not ok:
                if site and wf_id:
                    try:
                        self.client.delete(self._workflow_key(site, wf_id))
                    except Exception:
                        pass
                return False
        return True


def _check_manual_remember(
    workflow_mem: Optional[WorkflowMemory],
    site: str,
    prompt: str,
    steps: List[Dict[str, Any]],
):
    """If the external remember flag is set, persist the current workflow steps.

    This allows the UI "Remember" button to snapshot the path so far, even if
    the agent has not finished the task yet.
    """
    if not (workflow_mem and workflow_mem.enabled and site and steps):
        return
    try:
        if os.path.exists(REMEMBER_FLAG_PATH):
            print(f"Manual remember flag detected. Saving workflow with {len(steps)} steps for site '{site}'...")
            workflow_mem.save(site, prompt, steps, source="manual")
            print("Manual workflow save completed.")
            try:
                os.remove(REMEMBER_FLAG_PATH)
            except Exception:
                pass
    except Exception as e:
        print(f"Manual workflow save error: {e}")


# --- Helper Functions ---

def log_thought(step_name: str, dom_snippet: str, decision: Any):
    """Logs agent's internal state to a file for debugging."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open("agent_thoughts.txt", "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] GOAL: {step_name}\n")
        f.write(f"Filtered DOM:\n{dom_snippet}\n")
        f.write("-" * 40 + "\n")
        f.write(f"[{timestamp}] ACTION DECISION:\n")
        f.write(f"{json.dumps(decision, indent=2)}\n")
        f.write("-" * 40 + "\n")
    print(f"📝 Logged thought for '{step_name}'")


def append_step_trace(record: Dict[str, Any]) -> None:
    try:
        base_dir = os.path.dirname(__file__)
        path = os.path.join(base_dir, "step_trace.jsonl")
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        last_path = os.path.join(base_dir, "last_step_trace.json")
        existing: List[Dict[str, Any]] = []
        try:
            if os.path.exists(last_path):
                with open(last_path, "r", encoding="utf-8") as rf:
                    prev = json.load(rf)
                if isinstance(prev, list):
                    existing = [x for x in prev if isinstance(x, dict)]
                elif isinstance(prev, dict):
                    existing = [prev]
        except Exception:
            existing = []
        existing.append(record)
        with open(last_path, "w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def wait_for_money_approval(timeout_seconds: int = 600) -> bool:
    """Wait for UI-driven approval of a money action via flag file."""
    # Clear any stale flag from prior runs so we only react to fresh approval.
    try:
        if os.path.exists(APPROVAL_FLAG_PATH):
            os.remove(APPROVAL_FLAG_PATH)
    except Exception:
        pass

    print("Waiting for money action approval via UI... Click 'Approve' in the Hemlo UI to continue, or stop the agent.")
    start = time.time()
    while True:
        try:
            if os.path.exists(APPROVAL_FLAG_PATH):
                try:
                    os.remove(APPROVAL_FLAG_PATH)
                except Exception:
                    pass
                print("Money action approved via UI.")
                return True
        except Exception:
            pass
        if timeout_seconds is not None and timeout_seconds > 0 and (time.time() - start) > timeout_seconds:
            print("Money action approval timed out.")
            return False

        time.sleep(1.0)


def save_run_memory(goal: str, run_memory: Dict[str, Any], step: int, url: Optional[str] = None) -> None:
    """Persist the compact run_memory each step to a JSONL log and mirror in last_run_memory.json.

    Mirrors the pattern used by save_filtered_dom: maintains a long JSONL log and a short
    last_run_memory.json file which accumulates records for the current run (reset at start).
    """
    try:
        record = {
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "goal": goal,
            "step": int(step),
            "url": url,
            "run_memory": run_memory,
        }
        base_dir = os.path.dirname(__file__)
        with open(os.path.join(base_dir, "run_memory_log.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        last_path = os.path.join(base_dir, "last_run_memory.json")
        existing: typing.List[Dict[str, Any]] = []
        try:
            if os.path.exists(last_path):
                with open(last_path, "r", encoding="utf-8") as rf:
                    prev = json.load(rf)
                if isinstance(prev, list):
                    existing = [x for x in prev if isinstance(x, dict)]
                elif isinstance(prev, dict):
                    existing = [prev]
        except Exception:
            existing = []
        existing.append(record)
        with open(last_path, "w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Failed saving run memory: {e}")


def wait_for_login_choice(timeout_seconds: int = 900) -> Optional[Dict[str, Any]]:
    """Wait for UI-driven login choice (and optional credentials) via JSON flag file."""
    try:
        if os.path.exists(LOGIN_CHOICE_PATH):
            os.remove(LOGIN_CHOICE_PATH)
    except Exception:
        pass

    print("Waiting for login choice via UI... Select a provider (e.g., Google, Email) in the Hemlo UI.")
    start = time.time()
    while True:
        try:
            if os.path.exists(LOGIN_CHOICE_PATH):
                try:
                    with open(LOGIN_CHOICE_PATH, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    try:
                        os.remove(LOGIN_CHOICE_PATH)
                    except Exception:
                        pass
                    print(f"Login choice received from UI: {data}")
                    return data
                except Exception as e:
                    print(f"Error reading login choice: {e}")
        except Exception:
            pass

        if timeout_seconds is not None and timeout_seconds > 0 and (time.time() - start) > timeout_seconds:
            print("Login choice timed out.")
            return None
        time.sleep(1.0)


def detect_login_options(page: Page) -> List[Dict[str, Any]]:
    """Detect visible sign-in/up options on the page."""
    options: List[Dict[str, Any]] = []
    try:
        # Email/password detection
        email_loc = page.locator("input[type='email'], input[name='email'], input[name='username']").first
        pwd_loc = page.locator("input[type='password'], input[name='password']").first
        email_visible = False
        pwd_visible = False
        with contextlib.suppress(Exception):
            email_visible = email_loc.is_visible(timeout=1500)
        with contextlib.suppress(Exception):
            pwd_visible = pwd_loc.is_visible(timeout=1500)
        if email_visible:
            options.append(
                {
                    "type": "email",
                    "provider": "email",
                    "label": "Email / Password",
                    "needs_password": True,
                }
            )
    except Exception:
        pass

    # Common OAuth providers
    provider_keywords = [
        ("google", "Google"),
        ("facebook", "Facebook"),
        ("apple", "Apple"),
        ("github", "GitHub"),
        ("microsoft", "Microsoft"),
        ("twitter", "Twitter"),
        ("linkedin", "LinkedIn"),
    ]
    try:
        buttons = page.get_by_role("button").all()
    except Exception:
        buttons = []
    try:
        links = page.get_by_role("link").all()
    except Exception:
        links = []
    candidates = buttons + links
    for loc in candidates:
        label = ""
        with contextlib.suppress(Exception):
            label = (loc.inner_text(timeout=500) or "").strip()
        low = label.lower()
        for key, pretty in provider_keywords:
            # Only treat this as an auth option if the text clearly
            # indicates a login/signup context, not just "Get it on Google Play".
            auth_words = ["sign in", "log in", "login", "sign up", "register", "continue", "continue with", "use", "connect"]
            has_provider = key in low
            has_auth_hint = any(w in low for w in auth_words)
            explicit_phrases = [
                f"sign in with {key}",
                f"log in with {key}",
                f"login with {key}",
                f"sign up with {key}",
                f"continue with {key}",
                f"connect with {key}",
            ]
            is_explicit_auth = any(p in low for p in explicit_phrases)
            if is_explicit_auth or (has_provider and has_auth_hint):
                options.append(
                    {
                        "type": "oauth",
                        "provider": pretty.lower(),
                        "label": label or pretty,
                        "locator_type": "role",
                        "role": "button",  # best effort; UI handler will click by text
                    }
                )
                break

    # Deduplicate by provider
    dedup = {}
    for opt in options:
        key = (opt.get("type"), opt.get("provider"))
        if key not in dedup:
            dedup[key] = opt
    return list(dedup.values())


def emit_login_options(options: List[Dict[str, Any]], page_url: str):
    """Print a structured line so the UI server can emit a SocketIO event."""
    payload = {"options": options, "url": page_url}
    try:
        print("__LOGIN_OPTIONS__" + json.dumps(payload))
    except Exception:
        print("__LOGIN_OPTIONS__" + json.dumps({"options": [], "url": page_url}))


def is_probable_login_or_signup_page(page: Page) -> bool:
    """Heuristic: try to fire only on real auth pages, not generic homepages.

    Conditions (any true):
    - URL path clearly looks like login/register/join
    - A visible password field
    - Multiple login/signup CTAs present (buttons/links/text)
    """
    # URL-based signal
    try:
        parsed = urlparse(page.url)
        path = (parsed.path or "").lower()
        if any(seg in path for seg in ["login", "log-in", "signin", "sign-in", "signup", "sign-up", "register", "join"]):
            return True
    except Exception:
        pass

    # Password field usually means auth page
    try:
        pwd_loc = page.locator("input[type='password'], input[name='password']").first
        with contextlib.suppress(Exception):
            if pwd_loc.is_visible(timeout=1500):
                return True
    except Exception:
        pass

    # Count auth-related CTAs (buttons/links/text)
    auth_re = re.compile(r"(log in|login|sign in|sign up|register|create account|join)", re.I)
    ctas = 0
    try:
        ctas += page.get_by_role("button", name=auth_re).count()
    except Exception:
        pass
    try:
        ctas += page.get_by_role("link", name=auth_re).count()
    except Exception:
        pass
    try:
        ctas += page.get_by_text(auth_re).count()
    except Exception:
        pass
    return ctas >= 2


def _fill_first_visible_input(page: Page, selector: str, value: str, field_name: str) -> bool:
    """Fill the first *visible* input matching selector.

    Avoids hidden autofill-hint fields like Amazon's aok-hidden inputs.
    """
    try:
        loc = page.locator(selector)
        try:
            candidates = loc.all()
        except Exception:
            candidates = [loc]
        for cand in candidates:
            with contextlib.suppress(Exception):
                if cand.is_visible(timeout=500):
                    cand.fill(value, timeout=5000)
                    return True
    except Exception as e:
        print(f"Failed to fill visible {field_name}: {e}")
    print(f"No visible input found for {field_name} using selector: {selector!r}")
    return False


def perform_login_choice(page: Page, choice: Dict[str, Any]) -> bool:
    """Execute the chosen login path best-effort."""
    provider = (choice or {}).get("provider", "").lower()
    if not provider:
        return False
    if provider == "email":
        email = (choice or {}).get("email") or ""
        password = (choice or {}).get("password") or ""
        if not email or not password:
            print("Email/password not provided; cannot proceed with email login.")
            return False
        if not _fill_first_visible_input(
            page,
            "input[type='email'], input[name='email'], input[name='username']",
            email,
            "email/username",
        ):
            return False
        if not _fill_first_visible_input(
            page,
            "input[type='password'], input[name='password']",
            password,
            "password",
        ):
            return False
        try:
            page.keyboard.press("Enter")
        except Exception:
            pass
        return True

    # OAuth providers: click a button containing provider name
    try:
        locator = page.get_by_role("button", name=re.compile(provider, re.I))
        if locator.count():
            locator.first.click(timeout=5000)
            return True
    except Exception:
        pass
    try:
        locator = page.get_by_role("link", name=re.compile(provider, re.I))
        if locator.count():
            locator.first.click(timeout=5000)
            return True
    except Exception:
        pass
    print(f"Could not find a visible button/link for provider '{provider}'.")
    return False


def _flatten_ax_nodes(node: Dict[str, Any], out: List[Dict[str, Any]]):
    children = node.get("children") or []
    for ch in children:
        out.append(ch)
        _flatten_ax_nodes(ch, out)


_GOAL_PRIORITY_SPEC_CACHE: Dict[str, Dict[str, Any]] = {}


def _extract_planner_keywords(planner_assist: Optional[Dict[str, Any]]) -> Dict[str, List[str]]:
    must: List[str] = []
    boost: List[str] = []
    if not isinstance(planner_assist, dict) or not planner_assist:
        return {"must_include": must, "boost": boost}
    try:
        plan = planner_assist.get("plan") or planner_assist
        steps = plan.get("steps") or []
        if isinstance(steps, list):
            for st in steps:
                if not isinstance(st, dict):
                    continue
                tt = st.get("target_text")
                if isinstance(tt, str) and tt.strip():
                    must.append(tt.strip())
                hint_target = st.get("target")
                if isinstance(hint_target, str) and hint_target.strip():
                    must.append(hint_target.strip())
                alts = st.get("alternatives") or []
                if isinstance(alts, list):
                    for a in alts:
                        if isinstance(a, str) and a.strip():
                            boost.append(a.strip())
        hints = plan.get("hints") or []
        if isinstance(hints, list):
            for h in hints:
                if not isinstance(h, dict):
                    continue
                ht = h.get("target")
                if isinstance(ht, str) and ht.strip():
                    must.append(ht.strip())
        cbtn = plan.get("candidate_button_texts") or []
        if isinstance(cbtn, list):
            for b in cbtn:
                if isinstance(b, str) and b.strip():
                    boost.append(b.strip())
        cfields = plan.get("candidate_field_labels") or []
        if isinstance(cfields, list):
            for f in cfields:
                if isinstance(f, str) and f.strip():
                    boost.append(f.strip())
    except Exception:
        return {"must_include": must, "boost": boost}
    return {"must_include": must, "boost": boost}


def _norm_keywords(words: typing.List[str], limit: int) -> typing.List[str]:
    out: typing.List[str] = []
    seen: typing.Set[str] = set()
    for w in words or []:
        if not isinstance(w, str):
            continue
        s = re.sub(r"\s+", " ", w).strip().lower()
        if not s:
            continue
        if len(s) < 3:
            continue
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
        if len(out) >= limit:
            break
    return out


def _get_goal_priority_spec(goal: str, current_host: str, planner_assist: Optional[Dict[str, Any]] = None) -> Dict[str, typing.List[str]]:
    try:
        key = hashlib.sha256((current_host + "|" + (goal or "")).encode("utf-8")).hexdigest()[:18]
    except Exception:
        key = (current_host or "") + "|" + (goal or "")
    if key in _GOAL_PRIORITY_SPEC_CACHE:
        cached = _GOAL_PRIORITY_SPEC_CACHE.get(key)
        if isinstance(cached, dict):
            mi = cached.get("must_include")
            bo = cached.get("boost")
            if isinstance(mi, list) and isinstance(bo, list):
                return {"must_include": mi, "boost": bo}

    planner_kw = _extract_planner_keywords(planner_assist)
    planner_must = planner_kw.get("must_include") or []
    planner_boost = planner_kw.get("boost") or []

    must_include: typing.List[str] = []
    boost: typing.List[str] = []
    try:
        completion = client.chat.completions.create(
            model=config.SMART_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You help a web automation agent prioritize which clickable/interactive UI controls to keep. "
                        "Given a user goal, output a JSON object with keys: must_include, boost. "
                        "Each value is an array of short UI text snippets (typically 1-4 words) that are likely to appear on buttons/links/controls. "
                        "must_include should contain the few critical controls that must not be lost during DOM trimming. "
                        "boost should contain additional helpful controls. "
                        "Avoid generic words like 'click', 'button', 'link'. "
                        "Return at most 12 must_include and at most 18 boost. JSON only."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "goal": goal,
                            "site": current_host,
                            "planner_hints": {
                                "must_include": planner_must[:12],
                                "boost": planner_boost[:18],
                            },
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        raw = completion.choices[0].message.content or "{}"
        obj = json.loads(raw)
        mi = obj.get("must_include")
        bo = obj.get("boost")
        if isinstance(mi, list):
            must_include = [x for x in mi if isinstance(x, str)]
        if isinstance(bo, list):
            boost = [x for x in bo if isinstance(x, str)]
    except Exception:
        must_include = []
        boost = []

    merged_must = _norm_keywords(list(planner_must) + list(must_include), limit=14)
    merged_boost = _norm_keywords(list(planner_boost) + list(boost), limit=25)
    spec = {"must_include": merged_must, "boost": merged_boost}
    _GOAL_PRIORITY_SPEC_CACHE[key] = spec
    return spec


def playwright_filter_interactive_elements(page: Page, goal: str, planner_assist: Optional[Dict[str, Any]] = None) -> str:
    """Deterministically filter interactive elements using Playwright AX snapshot.

    Returns a JSON array string of elements with fields:
    - role, name, locator_type ("role"), text, is_money_action
    """
    print(f"Filtering DOM for goal (deterministic): '{goal}'...")
    # Determine current host to rank same-domain links higher
    current_host = ""
    try:
        current_host = urlparse(page.url).hostname or ""
    except Exception:
        current_host = ""

    priority_spec = _get_goal_priority_spec(goal, current_host, planner_assist)
    must_keywords = _norm_keywords(list(priority_spec.get("must_include") or []), limit=14)
    boost_keywords = _norm_keywords(list(priority_spec.get("boost") or []), limit=25)
    try:
        snapshot: Dict[str, Any] = page.accessibility.snapshot() or {}
    except Exception as e:
        print(f"AX snapshot failed: {e}")
        return "[]"

    if config.DEBUG_AX_SNAPSHOT:
        try:
            try:
                full_snapshot = page.accessibility.snapshot(interestingOnly=False) or snapshot
            except Exception:
                full_snapshot = snapshot
            record = {
                "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                "goal": goal,
                "url": getattr(page, "url", ""),
                "ax": full_snapshot,
            }
            with open("ax_snapshot_log.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            with open("last_ax_snapshot.json", "w", encoding="utf-8") as f:
                json.dump(record, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    flat: typing.List[Dict[str, Any]] = []
    if snapshot:
        _flatten_ax_nodes(snapshot, flat)

    interactive_roles = {
        "button",
        "link",
        "textbox",
        "searchbox",
        "combobox",
        "checkbox",
        "radio",
        "menuitem",
        "tab",
        "listbox",
        "option",
        "switch",
        "slider",
    }

    items: typing.List[Dict[str, Any]] = []
    for node in flat:
        role = node.get("role")
        name = node.get("name") or ""
        try:
            name = str(name).strip()
        except Exception:
            name = ""
        if not name:
            alt = node.get("description") or node.get("value") or node.get("valueString") or ""
            try:
                alt = str(alt).strip()
            except Exception:
                alt = ""
            if alt:
                name = alt
        if role in interactive_roles and name:
            lower = name.lower()
            is_money = any(w in lower for w in ["buy", "checkout", "place order", "pay", "add to cart"])
            items.append(
                {
                    "role": role,
                    "name": name,
                    "locator_type": "role",
                    "text": name,
                    "is_money_action": is_money,
                }
            )

    # Deduplicate by (role, name)
    seen = set()
    uniq: typing.List[Dict[str, Any]] = []
    for it in items:
        key = (it["role"], it["name"])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(it)

    # Enrich with attributes (href, aria-expanded) and kind for links/buttons
    for it in uniq:
        if it.get("locator_type") != "role":
            continue
        try:
            loc = page.get_by_role(it["role"], name=it["name"]).first
            href = None
            aria_expanded = None
            target = None
            try:
                href = loc.get_attribute("href", timeout=500)
            except Exception:
                pass
            try:
                aria_expanded = loc.get_attribute("aria-expanded", timeout=500)
            except Exception:
                pass
            try:
                target = loc.get_attribute("target", timeout=500)
            except Exception:
                pass
            href_kind = None
            if isinstance(href, str):
                h = href.strip().lower()
                if h.startswith("http") or h.startswith("/"):
                    href_kind = "nav"
                elif h.startswith("#"):
                    href_kind = "anchor"
                elif h.startswith("javascript"):
                    href_kind = "js"
                else:
                    href_kind = "other"
            it["href"] = href
            it["href_kind"] = href_kind
            it["aria_expanded"] = aria_expanded
            it["target"] = target
        except Exception:
            pass

    extra: typing.List[Dict[str, Any]] = []
    bases = [
        'input:not([type="hidden"])',
        "textarea",
        '[contenteditable="true"]',
        '[contenteditable=""]',
        '[contenteditable]:not([contenteditable="false"])',
    ]
    try:
        seen_locators: typing.Set[str] = set()
    except Exception:
        seen_locators = set()
    try:
        for base in bases:
            try:
                nodes = page.locator(base).all()
            except Exception:
                nodes = []
            for i, el in enumerate(nodes, start=1):
                try:
                    if not el.is_visible(timeout=300):
                        continue
                except Exception:
                    continue

                is_rich = False
                try:
                    is_rich = "contenteditable" in base
                except Exception:
                    is_rich = False

                nm = ""
                for attr in ("aria-label", "placeholder", "name", "id", "title"):
                    try:
                        v = el.get_attribute(attr, timeout=200)
                    except Exception:
                        v = None
                    if isinstance(v, str) and v.strip():
                        nm = v.strip()
                        break
                if not nm:
                    nm = "Rich text editor" if is_rich else "Text input"

                role = "textbox"
                if not is_rich:
                    try:
                        t = el.get_attribute("type", timeout=200)
                    except Exception:
                        t = None
                    if isinstance(t, str) and t.strip().lower() == "search":
                        role = "searchbox"

                locator = None
                try:
                    tid = el.get_attribute("data-testid", timeout=200)
                except Exception:
                    tid = None
                if isinstance(tid, str) and tid.strip():
                    esc = tid.strip().replace("\\", "\\\\").replace('"', '\\"')
                    locator = f'[data-testid="{esc}"]'
                if not locator:
                    try:
                        tid = el.get_attribute("data-test-id", timeout=200)
                    except Exception:
                        tid = None
                    if isinstance(tid, str) and tid.strip():
                        esc = tid.strip().replace("\\", "\\\\").replace('"', '\\"')
                        locator = f'[data-test-id="{esc}"]'
                if not locator:
                    try:
                        elid = el.get_attribute("id", timeout=200)
                    except Exception:
                        elid = None
                    if isinstance(elid, str) and elid.strip():
                        esc = elid.strip().replace("\\", "\\\\").replace('"', '\\"')
                        locator = f'[id="{esc}"]'
                if not locator:
                    try:
                        elname = el.get_attribute("name", timeout=200)
                    except Exception:
                        elname = None
                    if isinstance(elname, str) and elname.strip():
                        esc = elname.strip().replace("\\", "\\\\").replace('"', '\\"')
                        locator = f"{base}[name=\"{esc}\"]"
                if not locator:
                    locator = f":nth-match({base}, {i})"

                if locator in seen_locators:
                    continue
                seen_locators.add(locator)

                preview = ""
                try:
                    if is_rich:
                        preview = (el.inner_text(timeout=300) or "").strip()
                    else:
                        preview = (el.input_value(timeout=300) or "").strip()
                except Exception:
                    preview = ""
                if len(preview) > 80:
                    preview = preview[:80]

                extra.append(
                    {
                        "role": role,
                        "name": nm,
                        "locator_type": "css",
                        "locator": locator,
                        "is_rich": bool(is_rich),
                        "value_preview": preview,
                        "is_money_action": False,
                    }
                )
                if len(extra) >= 60:
                    break
            if len(extra) >= 60:
                break
    except Exception:
        extra = []

    if config.DEBUG_RICH_INPUTS:
        try:
            rec = {
                "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                "goal": goal,
                "url": getattr(page, "url", ""),
                "inputs": extra,
            }
            with open("rich_inputs_log.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            with open("last_rich_inputs.json", "w", encoding="utf-8") as f:
                json.dump(rec, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    if extra:
        try:
            existing_locators = set()
            for it in uniq:
                if it.get("locator_type") == "css" and it.get("locator"):
                    existing_locators.add(str(it.get("locator")))
                else:
                    existing_locators.add((str(it.get("role")), str(it.get("name"))))
            for it in extra:
                loc = str(it.get("locator") or "")
                if loc and loc not in existing_locators:
                    uniq.append(it)
                    existing_locators.add(loc)
        except Exception:
            uniq.extend(extra)

    try:
        hybrid_raw = page.evaluate(
            r"""() => {
  const out = [];
  const seen = new Set();
  let c = 0;
  const maxOut = 140;
  const maxOverlay = 60;

  const pushEl = (el, inOverlay) => {
    if (!el || !el.getBoundingClientRect) return;
    if (c >= maxOut) return;
    const rect = el.getBoundingClientRect();
    if (!rect || rect.width < 8 || rect.height < 8) return;
    if (rect.bottom < 0 || rect.right < 0 || rect.top > window.innerHeight || rect.left > window.innerWidth) return;
    const tag = (el.tagName || '').toLowerCase();
    if (tag === 'input') {
      const t = (el.getAttribute('type') || '').toLowerCase();
      if (t === 'hidden') return;
    }
    const roleAttr = (el.getAttribute('role') || '').toLowerCase();
    const isRole = ['button','link','checkbox','radio','menuitem','tab','switch','option'].includes(roleAttr);
    const isTag = ['a','button','input','select','textarea'].includes(tag);
    let style = null;
    try { style = window.getComputedStyle(el); } catch (e) { style = null; }
    if (style) {
      if (style.display === 'none' || style.visibility === 'hidden' || style.opacity === '0') return;
    }
    let isPointer = false;
    if (!isTag && !isRole) {
      isPointer = !!(style && style.cursor === 'pointer');
    }
    const tabindex = el.getAttribute('tabindex');
    const hasTab = tabindex !== null && String(tabindex).trim() !== '' && !isNaN(parseInt(tabindex, 10));
    if (!(isTag || isRole || isPointer || hasTab)) return;

    let label = '';
    try {
      label = (el.innerText || '').trim();
    } catch (e) {
      label = '';
    }
    if (!label) {
      const cand = el.getAttribute('aria-label') || el.getAttribute('title') || el.getAttribute('placeholder') || el.getAttribute('name') || '';
      label = (cand || '').trim();
    }
    if (!label) return;
    label = label.replace(/\s+/g, ' ').slice(0, 80);

    let sel = '';
    const dt = el.getAttribute('data-testid') || el.getAttribute('data-test-id') || el.getAttribute('data-test');
    if (dt && String(dt).trim()) {
      sel = `[data-testid="${String(dt).trim()}"]`;
      if (!el.matches(sel)) {
        sel = `[data-test-id="${String(dt).trim()}"]`;
      }
    }
    if (!sel) {
      const id = el.getAttribute('id');
      if (id && String(id).trim()) {
        sel = `#${String(id).trim()}`;
      }
    }
    if (!sel) {
      let eid = el.getAttribute('data-hemlo-eid');
      if (!eid) {
        eid = 'hemlo-' + Math.random().toString(16).slice(2) + '-' + Date.now().toString(16);
        try { el.setAttribute('data-hemlo-eid', eid); } catch (e) {}
      }
      sel = `[data-hemlo-eid="${eid}"]`;
    }
    if (seen.has(sel)) return;
    seen.add(sel);

    const href = tag === 'a' ? (el.getAttribute('href') || null) : (el.getAttribute('href') || null);
    const target = el.getAttribute('target') || null;
    const ariaExpanded = el.getAttribute('aria-expanded') || null;
    let role = roleAttr;
    if (!role) {
      if (tag === 'a') role = 'link';
      else if (tag === 'button') role = 'button';
      else if (isPointer) role = 'button';
      else role = 'button';
    }
    out.push({ role, name: label, selector: sel, href, target, aria_expanded: ariaExpanded, in_overlay: !!inOverlay });
    c++;
  };

  let overlayCount = 0;
  try {
    const overlays = document.querySelectorAll('[role="menu"],[role="listbox"],[role="dialog"],[role="alertdialog"],[aria-modal="true"]');
    for (let i = 0; i < overlays.length; i++) {
      if (overlayCount >= maxOverlay || c >= maxOut) break;
      const root = overlays[i];
      if (!root || !root.getBoundingClientRect) continue;
      const r = root.getBoundingClientRect();
      if (!r || r.width < 10 || r.height < 10) continue;
      const st = window.getComputedStyle(root);
      if (st && (st.display === 'none' || st.visibility === 'hidden' || st.opacity === '0')) continue;
      const nodes = root.querySelectorAll('a,button,input,select,textarea,[role],[tabindex],div,span');
      for (let j = 0; j < nodes.length; j++) {
        if (overlayCount >= maxOverlay || c >= maxOut) break;
        const before = c;
        pushEl(nodes[j], true);
        if (c > before) overlayCount++;
      }
    }
  } catch (e) {}

  const candidates = document.querySelectorAll('a,button,input,select,textarea,[role],[tabindex],div,span');
  for (let i = 0; i < candidates.length; i++) {
    if (c >= maxOut) break;
    pushEl(candidates[i], false);
  }
  return out;
}"""
        )
    except Exception:
        hybrid_raw = []
    try:
        if isinstance(hybrid_raw, list) and hybrid_raw:
            existing = set()
            for it in uniq:
                if it.get("locator_type") == "css" and it.get("locator"):
                    existing.add(("css", str(it.get("locator"))))
                else:
                    existing.add((str(it.get("role")), str(it.get("name"))))
            for hr in hybrid_raw:
                try:
                    name = str((hr or {}).get("name") or "").strip()
                except Exception:
                    name = ""
                try:
                    role = str((hr or {}).get("role") or "button").strip().lower()
                except Exception:
                    role = "button"
                try:
                    locator = str((hr or {}).get("selector") or "").strip()
                except Exception:
                    locator = ""
                if not name or not locator:
                    continue
                key = ("css", locator)
                if key in existing:
                    continue
                if (role, name) in existing:
                    continue
                lower = name.lower()
                is_money = any(w in lower for w in ["buy", "checkout", "place order", "pay", "add to cart"])
                href = (hr or {}).get("href")
                target = (hr or {}).get("target")
                aria_expanded = (hr or {}).get("aria_expanded")
                in_overlay = bool((hr or {}).get("in_overlay"))
                href_kind = None
                try:
                    if isinstance(href, str):
                        h = href.strip().lower()
                        if h.startswith("http") or h.startswith("/"):
                            href_kind = "nav"
                        elif h.startswith("#"):
                            href_kind = "anchor"
                        elif h.startswith("javascript"):
                            href_kind = "js"
                        else:
                            href_kind = "other"
                except Exception:
                    href_kind = None
                uniq.append(
                    {
                        "role": role,
                        "name": name,
                        "locator_type": "css",
                        "locator": locator,
                        "text": name,
                        "is_money_action": bool(is_money),
                        "href": href,
                        "href_kind": href_kind,
                        "aria_expanded": aria_expanded,
                        "in_overlay": bool(in_overlay),
                        "target": target,
                    }
                )
                existing.add(key)
    except Exception:
        pass

    # Rank by relevance to goal, control type, and link quality
    g = goal.lower()
    input_keywords: typing.List[str] = []
    try:
        toks = [t for t in g.split() if isinstance(t, str) and len(t) >= 3]
        input_keywords = _norm_keywords(list(must_keywords) + list(boost_keywords) + toks, limit=35)
    except Exception:
        input_keywords = []
    for it in uniq:
        try:
            nlow = str(it.get("name") or "").lower()
        except Exception:
            nlow = ""
        it["is_primary_input"] = bool(
            it.get("role") in ("textbox", "searchbox", "combobox")
            and (it.get("is_rich") or (input_keywords and any(k in nlow for k in input_keywords)))
        )

    menu_open = False
    try:
        menu_open = any(str(x.get("aria_expanded") or "").lower() == "true" for x in uniq)
    except Exception:
        menu_open = False

    def _score(it: Dict[str, Any]) -> int:
        n = it["name"].lower()
        s = 0
        if it["role"] in ("searchbox", "textbox"):
            s += 3
            if it.get("is_primary_input"):
                s += 6
        if any(tok and tok in n for tok in g.split()):
            s += 2
        if boost_keywords and any(k in n for k in boost_keywords):
            s += 6
        if must_keywords and any(k in n for k in must_keywords):
            s += 10
        if menu_open:
            if it.get("role") in {"menuitem", "option"}:
                s += 8
            if it.get("in_overlay"):
                s += 5
            if any(x in n for x in ["event", "task", "appointment", "schedule", "meeting"]):
                s += 12
            try:
                if str(it.get("aria_expanded") or "").lower() == "true" and any(x in n for x in ["create", "add", "more", "options"]):
                    s -= 30
            except Exception:
                pass
        if it.get("is_money_action"):
            s += 4
        if it["role"] == "button":
            s += 1
        # Prefer real navigation links over anchors/js
        hk = it.get("href_kind")
        if hk == "nav":
            s += 3
        elif hk in {"anchor", "js"}:
            s -= 2
        # Domain preference: prefer same-domain nav links; downrank off-domain
        if hk == "nav":
            href = (it.get("href") or "").strip()
            if href:
                try:
                    href_host = urlparse(href).hostname or current_host
                    if current_host and href_host != current_host:
                        s -= 2
                    elif current_host and href_host == current_host:
                        s += 1
                except Exception:
                    pass
        return s

    uniq.sort(key=_score, reverse=True)
    max_items = 40

    def _item_key(it: Dict[str, Any]) -> typing.Tuple[str, str]:
        if it.get("locator_type") == "css" and it.get("locator"):
            return ("css", str(it.get("locator")))
        return (str(it.get("role")), str(it.get("name")))

    def _matches_priority(it: Dict[str, Any]) -> bool:
        try:
            nlow = str(it.get("name") or "").lower()
        except Exception:
            nlow = ""
        if must_keywords and any(k in nlow for k in must_keywords):
            return True
        return False

    out: typing.List[Dict[str, Any]] = []
    seen_out: typing.Set[typing.Tuple[str, str]] = set()

    if must_keywords:
        try:
            prioritized = sorted(list(must_keywords), key=lambda x: len(str(x or "")), reverse=True)
        except Exception:
            prioritized = list(must_keywords)
        for kw in prioritized:
            if len(out) >= max_items:
                break
            try:
                kw_low = str(kw or "").lower().strip()
            except Exception:
                kw_low = ""
            if not kw_low:
                continue
            added_for_kw = 0
            for it in uniq:
                if len(out) >= max_items:
                    break
                try:
                    nlow = str(it.get("name") or "").lower()
                except Exception:
                    nlow = ""
                if kw_low not in nlow:
                    continue
                k = _item_key(it)
                if k in seen_out:
                    continue
                seen_out.add(k)
                out.append(it)
                added_for_kw += 1
                if added_for_kw >= 2:
                    break

    if len(out) < max_items:
        for it in uniq:
            k = _item_key(it)
            if k in seen_out:
                continue
            seen_out.add(k)
            out.append(it)
            if len(out) >= max_items:
                break

    return json.dumps(out, ensure_ascii=False)


def playwright_filter_interactive_elements_gold(
    page: Page,
    goal: str,
    planner_assist: Optional[Dict[str, Any]] = None,
    max_items: int = 180,
    grid_step: int = 55,
) -> str:
    print(f"Filtering DOM for goal (gold): '{goal}'...")
    current_host = ""
    try:
        current_host = urlparse(page.url).hostname or ""
    except Exception:
        current_host = ""

    try:
        max_items = int(max_items)
    except Exception:
        max_items = 180
    if max_items < 20:
        max_items = 20
    if max_items > 400:
        max_items = 400

    try:
        grid_step = int(grid_step)
    except Exception:
        grid_step = 55
    if grid_step < 25:
        grid_step = 25
    if grid_step > 120:
        grid_step = 120

    priority_spec = _get_goal_priority_spec(goal, current_host, planner_assist)
    must_keywords = _norm_keywords(list(priority_spec.get("must_include") or []), limit=14)
    boost_keywords = _norm_keywords(list(priority_spec.get("boost") or []), limit=25)

    goal_tokens: typing.List[str] = []
    try:
        toks = re.findall(r"[a-zA-Z]{3,}", str(goal or "").lower())
        goal_tokens = _norm_keywords([t for t in toks if t], limit=40)
    except Exception:
        goal_tokens = []

    uniq: typing.List[Dict[str, Any]] = []
    try:
        snapshot: Dict[str, Any] = page.accessibility.snapshot() or {}
    except Exception:
        snapshot = {}

    flat: typing.List[Dict[str, Any]] = []
    if snapshot:
        _flatten_ax_nodes(snapshot, flat)

    interactive_roles = {
        "button",
        "link",
        "textbox",
        "searchbox",
        "combobox",
        "checkbox",
        "radio",
        "menuitem",
        "tab",
        "listbox",
        "option",
        "switch",
        "slider",
    }

    seen_role_keys: typing.Set[typing.Tuple[str, str]] = set()
    for node in flat:
        try:
            role = node.get("role")
            name = node.get("name")
        except Exception:
            role = None
            name = None
        if not role or role not in interactive_roles:
            continue
        if not isinstance(name, str) or not name.strip():
            continue
        nm = re.sub(r"\s+", " ", name).strip()
        key = (role, nm)
        if key in seen_role_keys:
            continue
        seen_role_keys.add(key)
        uniq.append(
            {
                "role": role,
                "name": nm,
                "locator_type": "role",
                "text": nm,
                "is_money_action": bool(any(w in nm.lower() for w in ["buy", "checkout", "place order", "pay", "add to cart"])),
                "href": None,
                "href_kind": None,
                "aria_expanded": None,
                "in_overlay": False,
                "in_viewport": False,
                "target": None,
            }
        )

    try:
        hybrid_raw = page.evaluate(
            r"""(gridStep, goalTokens, mustKeywords, boostKeywords) => {
  const out = [];
  const seen = new Set();
  let c = 0;
  const maxOut = 900;
  const maxOverlay = 140;
  const maxViewport = 220;

  const normList = (xs) => {
    try {
      return (Array.isArray(xs) ? xs : []).map(x => String(x || '').toLowerCase().trim()).filter(x => x.length >= 3);
    } catch (e) { return []; }
  };
  const kws = Array.from(new Set([ ...normList(goalTokens), ...normList(mustKeywords), ...normList(boostKeywords) ])).slice(0, 60);
  const hasKw = (label) => {
    try {
      if (!kws.length) return false;
      const s = String(label || '').toLowerCase();
      for (const k of kws) { if (k && s.includes(k)) return true; }
    } catch (e) {}
    return false;
  };

  const isVisible = (el) => {
    try {
      if (!el || !el.getBoundingClientRect) return false;
      const rect = el.getBoundingClientRect();
      if (!rect || rect.width < 6 || rect.height < 6) return false;
      if (rect.bottom < 0 || rect.right < 0 || rect.top > window.innerHeight || rect.left > window.innerWidth) return false;
      const style = window.getComputedStyle(el);
      if (style && (style.display === 'none' || style.visibility === 'hidden' || style.opacity === '0')) return false;
      return true;
    } catch (e) { return false; }
  };

  const isActionable = (el) => {
    try {
      if (!el) return false;
      const tag = (el.tagName || '').toLowerCase();
      if (tag === 'input') {
        const t = (el.getAttribute('type') || '').toLowerCase();
        if (t === 'hidden') return false;
      }
      const roleAttr = (el.getAttribute('role') || '').toLowerCase();
      const isRole = ['button','link','checkbox','radio','menuitem','tab','switch','option','textbox','searchbox','combobox'].includes(roleAttr);
      const isTag = ['a','button','input','select','textarea'].includes(tag);
      let style = null;
      try { style = window.getComputedStyle(el); } catch (e) { style = null; }
      let isPointer = false;
      if (!isTag && !isRole) {
        isPointer = !!(style && style.cursor === 'pointer');
      }
      const tabindex = el.getAttribute('tabindex');
      const hasTab = tabindex !== null && String(tabindex).trim() !== '' && !isNaN(parseInt(tabindex, 10));
      return !!(isTag || isRole || isPointer || hasTab);
    } catch (e) { return false; }
  };

  const labelFromLabelledby = (el) => {
    try {
      const raw = (el.getAttribute('aria-labelledby') || '').trim();
      if (!raw) return '';
      const ids = raw.split(/\s+/g).filter(Boolean);
      if (!ids.length) return '';
      let txt = '';
      for (const id of ids) {
        const ref = document.getElementById(id);
        if (!ref) continue;
        const t = (ref.innerText || ref.textContent || '').trim();
        if (t) txt += (txt ? ' ' : '') + t;
      }
      return (txt || '').trim();
    } catch (e) { return ''; }
  };

  const getLabel = (el) => {
    let label = '';
    try { label = (el.innerText || '').trim(); } catch (e) { label = ''; }
    try {
      if (label) {
        const words = label.split(/\s+/g).filter(Boolean);
        if (words.length > 12 || label.length > 90) label = '';
      }
    } catch (e) {}
    if (!label) {
      const via = labelFromLabelledby(el);
      if (via) label = via;
    }
    if (!label) {
      const cand = el.getAttribute('aria-label') || el.getAttribute('title') || el.getAttribute('placeholder') || el.getAttribute('name') || el.getAttribute('alt') || '';
      label = (cand || '').trim();
    }
    if (!label) {
      try {
        const tc = (el.textContent || '').trim();
        if (tc && tc.length <= 40) label = tc;
      } catch (e) {}
    }
    if (!label) {
      try {
        const tag = (el.tagName || '').toLowerCase();
        const cls = (el.getAttribute('class') || '').trim().split(/\s+/g).filter(Boolean)[0] || '';
        if (cls) label = `Unlabeled ${tag}.${cls}`;
        else label = `Unlabeled ${tag}`;
      } catch (e) {
        label = 'Unlabeled';
      }
    }
    label = label.replace(/\s+/g, ' ').slice(0, 80);
    return label;
  };

  const ensureSelector = (el) => {
    let sel = '';
    const dt = el.getAttribute('data-testid') || el.getAttribute('data-test-id') || el.getAttribute('data-test');
    if (dt && String(dt).trim()) {
      sel = `[data-testid="${String(dt).trim()}"]`;
      if (!el.matches(sel)) {
        sel = `[data-test-id="${String(dt).trim()}"]`;
      }
    }
    if (!sel) {
      const id = el.getAttribute('id');
      if (id && String(id).trim()) {
        sel = `#${String(id).trim()}`;
      }
    }
    if (!sel) {
      let eid = el.getAttribute('data-hemlo-eid');
      if (!eid) {
        eid = 'hemlo-' + Math.random().toString(16).slice(2) + '-' + Date.now().toString(16);
        try { el.setAttribute('data-hemlo-eid', eid); } catch (e) {}
      }
      sel = `[data-hemlo-eid="${eid}"]`;
    }
    return sel;
  };

  const pushEl = (el, inOverlay, inViewport) => {
    if (!el || !el.getBoundingClientRect) return;
    if (c >= maxOut) return;
    if (!isVisible(el)) return;
    if (!isActionable(el)) return;

    const sel = ensureSelector(el);
    if (!sel) return;
    if (seen.has(sel)) return;
    seen.add(sel);

    const tag = (el.tagName || '').toLowerCase();
    const roleAttr = (el.getAttribute('role') || '').toLowerCase();
    let role = roleAttr;
    if (!role) {
      if (tag === 'a') role = 'link';
      else if (tag === 'button') role = 'button';
      else if (tag === 'input' || tag === 'textarea') role = 'textbox';
      else if (tag === 'select') role = 'combobox';
      else role = 'generic';
    }
    const label = getLabel(el);
    const href = el.getAttribute('href') || null;
    const target = el.getAttribute('target') || null;
    const ariaExpanded = el.getAttribute('aria-expanded') || null;
    out.push({ role, name: label, selector: sel, href, target, aria_expanded: ariaExpanded, in_overlay: !!inOverlay, in_viewport: !!inViewport });
    c++;
  };

  const isPreferred = (el) => {
    try {
      if (!el) return false;
      const tag = (el.tagName || '').toLowerCase();
      if (['a','button','input','select','textarea'].includes(tag)) return true;
      const roleAttr = (el.getAttribute('role') || '').toLowerCase();
      return ['button','link','checkbox','radio','menuitem','tab','switch','option','textbox','searchbox','combobox','slider'].includes(roleAttr);
    } catch (e) { return false; }
  };

  const climbToActionable = (el) => {
    let cur = el;
    for (let i = 0; i < 7; i++) {
      if (!cur) return null;
      if (isPreferred(cur) && isVisible(cur)) return cur;
      cur = cur.parentElement;
    }
    cur = el;
    for (let i = 0; i < 7; i++) {
      if (!cur) return null;
      if (isActionable(cur) && isVisible(cur)) return cur;
      cur = cur.parentElement;
    }
    return null;
  };

  let overlayCount = 0;
  try {
    const overlays = document.querySelectorAll('[role="menu"],[role="listbox"],[role="dialog"],[role="alertdialog"],[aria-modal="true"]');
    for (let i = 0; i < overlays.length; i++) {
      if (overlayCount >= maxOverlay || c >= maxOut) break;
      const root = overlays[i];
      if (!root || !isVisible(root)) continue;
      const nodes = root.querySelectorAll('a,button,input,select,textarea,[role],[tabindex]');
      for (let j = 0; j < nodes.length; j++) {
        if (overlayCount >= maxOverlay || c >= maxOut) break;
        const before = c;
        pushEl(nodes[j], true, false);
        if (c > before) overlayCount++;
      }
    }
  } catch (e) {}

  let viewportCount = 0;
  try {
    const step = Math.max(25, Math.min(120, parseInt(gridStep || 55, 10) || 55));
    for (let y = 0; y < window.innerHeight; y += step) {
      for (let x = 0; x < window.innerWidth; x += step) {
        if (viewportCount >= maxViewport || c >= maxOut) break;
        let stack = null;
        try { stack = document.elementsFromPoint(x, y); } catch (e) { stack = null; }
        const els = Array.isArray(stack) ? stack : [];
        let picked = null;
        for (let k = 0; k < els.length; k++) {
          const cand = climbToActionable(els[k]);
          if (!cand) continue;
          picked = cand;
          if (isPreferred(cand)) break;
        }
        if (!picked) continue;
        const inOverlay = !!(picked.closest && picked.closest('[role="menu"],[role="listbox"],[role="dialog"],[role="alertdialog"],[aria-modal="true"]'));
        const before = c;
        pushEl(picked, inOverlay, true);
        if (c > before) viewportCount++;
      }
      if (viewportCount >= maxViewport || c >= maxOut) break;
    }
  } catch (e) {}

  try {
    const candidates = document.querySelectorAll('a,button,input,select,textarea,[role],[tabindex]');
    for (let i = 0; i < candidates.length; i++) {
      if (c >= maxOut) break;
      pushEl(candidates[i], false, false);
    }
  } catch (e) {}

  try {
    if (kws.length) {
      const focus = document.querySelectorAll('button,a,[role="button"],[role="tab"],[role="menuitem"],input,select,textarea');
      for (let i = 0; i < focus.length; i++) {
        if (c >= maxOut) break;
        const el = focus[i];
        const label = getLabel(el);
        if (!label) continue;
        if (!hasKw(label)) continue;
        const inOverlay = !!(el.closest && el.closest('[role="menu"],[role="listbox"],[role="dialog"],[role="alertdialog"],[aria-modal="true"]'));
        pushEl(el, inOverlay, true);
      }
    }
  } catch (e) {}

  return out;
}""",
            grid_step,
            goal_tokens,
            must_keywords,
            boost_keywords,
        )
    except Exception:
        hybrid_raw = []

    try:
        existing = set()
        for it in uniq:
            existing.add((str(it.get("role")), str(it.get("name"))))
        if isinstance(hybrid_raw, list) and hybrid_raw:
            for hr in hybrid_raw:
                try:
                    name = str((hr or {}).get("name") or "").strip()
                except Exception:
                    name = ""
                try:
                    role = str((hr or {}).get("role") or "button").strip().lower()
                except Exception:
                    role = "button"
                try:
                    locator = str((hr or {}).get("selector") or "").strip()
                except Exception:
                    locator = ""
                if not name or not locator:
                    continue
                if (role, name) in existing:
                    continue
                lower = name.lower()
                is_money = any(w in lower for w in ["buy", "checkout", "place order", "pay", "add to cart"])
                href = (hr or {}).get("href")
                target = (hr or {}).get("target")
                aria_expanded = (hr or {}).get("aria_expanded")
                in_overlay = bool((hr or {}).get("in_overlay"))
                in_viewport = bool((hr or {}).get("in_viewport"))
                href_kind = None
                try:
                    if isinstance(href, str):
                        h = href.strip().lower()
                        if h.startswith("http") or h.startswith("/"):
                            href_kind = "nav"
                        elif h.startswith("#"):
                            href_kind = "anchor"
                        elif h.startswith("javascript"):
                            href_kind = "js"
                        else:
                            href_kind = "other"
                except Exception:
                    href_kind = None
                uniq.append(
                    {
                        "role": role,
                        "name": name,
                        "locator_type": "css",
                        "locator": locator,
                        "text": name,
                        "is_money_action": bool(is_money),
                        "href": href,
                        "href_kind": href_kind,
                        "aria_expanded": aria_expanded,
                        "in_overlay": bool(in_overlay),
                        "in_viewport": bool(in_viewport),
                        "target": target,
                    }
                )
                existing.add((role, name))
    except Exception:
        pass

    g = goal.lower()
    for it in uniq:
        try:
            nlow = str(it.get("name") or "").lower()
        except Exception:
            nlow = ""
        it["is_primary_input"] = bool(it.get("role") in ("textbox", "searchbox", "combobox") and (must_keywords and any(k in nlow for k in must_keywords)))

    menu_open = False
    try:
        menu_open = any(str(x.get("aria_expanded") or "").lower() == "true" for x in uniq)
    except Exception:
        menu_open = False

    def _score(it: Dict[str, Any]) -> int:
        n_raw = str(it.get("name") or "")
        n = n_raw.lower()
        s = 0
        if it.get("role") in ("searchbox", "textbox"):
            s += 3
            if it.get("is_primary_input"):
                s += 6
        if any(tok and len(tok) >= 4 and tok in n for tok in g.split()):
            s += 2
        if boost_keywords and any(k in n for k in boost_keywords):
            s += 6
        if must_keywords and any(k in n for k in must_keywords):
            s += 10
        if n.startswith("unlabeled"):
            s -= 10
        if len(n_raw) > 90:
            s -= 20
        elif len(n_raw) > 60:
            s -= 10
        if str(it.get("role") or "").lower() == "generic":
            s -= 6
        if ("filter" in g) or ("filters" in g):
            if re.search(r"\bfilters?\b", n):
                s += 30
            elif any(x in n for x in ["effect", "effects", "adjust", "adjustments"]):
                s += 10
        if it.get("in_viewport"):
            s += 8
        if it.get("in_overlay"):
            s += 5
        if menu_open:
            if str(it.get("role") or "").lower() in {"menuitem", "option"}:
                s += 8
            if any(x in n for x in ["event", "task", "appointment", "schedule", "meeting"]):
                s += 12
            try:
                if str(it.get("aria_expanded") or "").lower() == "true" and any(x in n for x in ["create", "add", "more", "options"]):
                    s -= 30
            except Exception:
                pass
        if it.get("is_money_action"):
            s += 4
        if it.get("role") == "button":
            s += 1
        hk = it.get("href_kind")
        if hk == "nav":
            s += 3
        elif hk in {"anchor", "js"}:
            s -= 2
        if hk == "nav":
            href = (it.get("href") or "").strip()
            if href:
                try:
                    href_host = urlparse(href).hostname or current_host
                    if current_host and href_host != current_host:
                        s -= 2
                    elif current_host and href_host == current_host:
                        s += 1
                except Exception:
                    pass
        return s

    uniq.sort(key=_score, reverse=True)

    def _item_key(it: Dict[str, Any]) -> typing.Tuple[str, str]:
        if it.get("locator_type") == "css" and it.get("locator"):
            return ("css", str(it.get("locator")))
        return (str(it.get("role")), str(it.get("name")))

    out: typing.List[Dict[str, Any]] = []
    seen_out: typing.Set[typing.Tuple[str, str]] = set()
    for it in uniq:
        k = _item_key(it)
        if k in seen_out:
            continue
        seen_out.add(k)
        out.append(it)
        if len(out) >= max_items:
            break

    for i, it in enumerate(out):
        try:
            it["id"] = i + 1
        except Exception:
            pass
    return json.dumps(out, ensure_ascii=False)


def filter_interactive_elements(
    page: Page,
    goal: str,
    planner_assist: Optional[Dict[str, Any]] = None,
    dom_mode: str = "legacy",
    max_items: int = 40,
) -> str:
    dm = str(dom_mode or "legacy").strip().lower()
    if dm == "gold":
        return playwright_filter_interactive_elements_gold(page, goal, planner_assist, max_items=max_items)
    return playwright_filter_interactive_elements(page, goal, planner_assist)


def save_filtered_dom(goal: str, filtered_json_array: str, dom_hash: Optional[str] = None, url: Optional[str] = None) -> None:
    """Persist the filtered DOM that we send to Groq on every step.

    Writes a JSONL entry to filtered_dom_log.jsonl and a snapshot to last_filtered_dom.json.
    """
    try:
        record = {
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "goal": goal,
            "dom_hash": dom_hash,
            "url": url,
            "elements": json.loads(filtered_json_array),
        }
        base_dir = os.path.dirname(__file__)
        with open(os.path.join(base_dir, "filtered_dom_log.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        last_path = os.path.join(base_dir, "last_filtered_dom.json")
        existing: typing.List[Dict[str, Any]] = []
        try:
            if os.path.exists(last_path):
                with open(last_path, "r", encoding="utf-8") as rf:
                    prev = json.load(rf)
                if isinstance(prev, list):
                    existing = [x for x in prev if isinstance(x, dict)]
                elif isinstance(prev, dict):
                    existing = [prev]
        except Exception:
            existing = []
        existing.append(record)
        with open(last_path, "w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Failed saving filtered DOM: {e}")


def save_perception(goal: str, perception: Dict[str, Any], dom_hash: str) -> None:
    try:
        record = {
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "goal": goal,
            "dom_hash": dom_hash,
            "perception": perception,
        }
        base_dir = os.path.dirname(__file__)
        with open(os.path.join(base_dir, "perception_log.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        with open(os.path.join(base_dir, "last_perception.json"), "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
        # Also mirror success detection signals into their own debug files for quick inspection
        try:
            success = (perception or {}).get("success_signals") or None
            if isinstance(success, dict):
                srec = {
                    "ts": record["ts"],
                    "goal": goal,
                    "dom_hash": dom_hash,
                    "success_signals": success,
                }
                with open(os.path.join(base_dir, "success_signals_log.jsonl"), "a", encoding="utf-8") as sf:
                    sf.write(json.dumps(srec, ensure_ascii=False) + "\n")
                with open(os.path.join(base_dir, "last_success_signals.json"), "w", encoding="utf-8") as sf:
                    json.dump(srec, sf, ensure_ascii=False, indent=2)
        except Exception:
            pass
    except Exception as e:
        print(f"Failed saving perception: {e}")


PAGE_MODE_NORMAL = "NORMAL"
PAGE_MODE_LOGIN = "LOGIN"
PAGE_MODE_AGE_GATE = "AGE_GATE"
PAGE_MODE_COOKIE_GATE = "COOKIE_GATE"
PAGE_MODE_BLOCKING_MODAL = "BLOCKING_MODAL"


def _safe_page_title(page: Page) -> str:
    try:
        t = page.title()
        return t if isinstance(t, str) else ""
    except Exception:
        return ""


def _safe_body_text_snippet(page: Page, limit: int = 1200) -> str:
    try:
        txt = page.evaluate(
            "(limit) => { const t = document.body ? (document.body.innerText || '') : ''; return t.slice(0, limit); }",
            limit,
        )
        return txt if isinstance(txt, str) else ""
    except Exception:
        return ""


def _has_visible_modal(page: Page) -> bool:
    try:
        val = page.evaluate(
            """() => {
              const el = document.querySelector('[role="dialog"], [role="alertdialog"], [aria-modal="true"]');
              if (!el) return false;
              const style = window.getComputedStyle(el);
              if (!style) return true;
              return style.display !== 'none' && style.visibility !== 'hidden' && style.opacity !== '0';
            }"""
        )
        return bool(val)
    except Exception:
        return False


def _should_force_front() -> bool:
    try:
        return os.getenv("HEMLO_FORCE_FRONT", "0") == "1"
    except Exception:
        return False


def _gate_scores(items: typing.List[Dict[str, Any]], body_text: str) -> Dict[str, Any]:
    t = (body_text or "").lower()
    names = " ".join([(it.get("name") or "") for it in (items or [])]).lower()

    cookie_score = 0
    cookie_word = bool(re.search(r"\bcookies?\b", t))
    if cookie_word:
        cookie_score += 2
    if "consent" in t:
        cookie_score += 1
    if any(w in names for w in ["accept", "agree", "allow", "allow all", "got it", "i understand", "ok"]):
        cookie_score += 1
    if any(w in names for w in ["reject", "decline", "manage", "preferences", "settings", "cookie settings"]):
        cookie_score += 1

    age_score = 0
    age_re = re.compile(
        r"(age verification|verify your age|confirm your age|are you (over )?(18|21)|over (18|21)|\b(18\+|21\+)\b|date of birth|birth date|\bdob\b|year of birth|you must be (18|21))",
        re.I,
    )
    if age_re.search(body_text or ""):
        age_score += 3
    if re.search(r"\b(18\+|21\+|over\s*(18|21)|i\s*(am|\"m)\s*(18|21))\b", names):
        age_score += 2
    if "age" in t and any(w in names for w in ["enter", "continue", "yes"]):
        age_score += 1

    return {
        "cookie_score": cookie_score,
        "cookie_word": cookie_word,
        "age_score": age_score,
    }


def detect_page_mode(page: Page, items: typing.List[Dict[str, Any]], body_text: str, scores: Optional[Dict[str, Any]] = None) -> str:
    scores = scores or _gate_scores(items, body_text)
    try:
        cookie_score = int(scores.get("cookie_score", 0) or 0)
    except Exception:
        cookie_score = 0
    try:
        age_score = int(scores.get("age_score", 0) or 0)
    except Exception:
        age_score = 0

    login_like = False
    try:
        login_like = bool(is_probable_login_or_signup_page(page))
    except Exception:
        login_like = False

    modal_like = _has_visible_modal(page)

    if cookie_score >= 3:
        return PAGE_MODE_COOKIE_GATE
    if age_score >= 3:
        return PAGE_MODE_AGE_GATE
    if login_like:
        return PAGE_MODE_LOGIN
    if modal_like:
        return PAGE_MODE_BLOCKING_MODAL
    return PAGE_MODE_NORMAL


def build_perception(page: Page, items: typing.List[Dict[str, Any]], goal: str, dom_hash: str = "") -> Dict[str, Any]:
    url = ""
    try:
        url = page.url
    except Exception:
        url = ""

    title = _safe_page_title(page)
    body_text = _safe_body_text_snippet(page, limit=1200)
    text_norm = ""
    try:
        text_norm = re.sub(r"\s+", " ", body_text).strip()
    except Exception:
        text_norm = ""

    scores = _gate_scores(items, body_text)
    mode = detect_page_mode(page, items, body_text, scores)
    has_modal = False
    try:
        has_modal = _has_visible_modal(page)
    except Exception:
        has_modal = False
    login_like = False
    try:
        login_like = bool(is_probable_login_or_signup_page(page))
    except Exception:
        login_like = False

    role_counts: Dict[str, int] = {}
    for it in items or []:
        r = str(it.get("role") or "")
        role_counts[r] = role_counts.get(r, 0) + 1

    top_controls: typing.List[Dict[str, Any]] = []
    for it in (items or [])[:15]:
        top_controls.append({k: it.get(k) for k in ("role", "name", "href_kind", "aria_expanded", "is_money_action")})

    inputs: typing.List[Dict[str, Any]] = []
    for it in items or []:
        if it.get("role") in ("textbox", "searchbox", "combobox"):
            inputs.append({"role": it.get("role"), "name": it.get("name")})
        if len(inputs) >= 8:
            break

    possible_gate_controls: typing.List[Dict[str, Any]] = []
    gate_words = [
        "accept",
        "reject",
        "agree",
        "consent",
        "manage",
        "preferences",
        "settings",
        "cookie",
        "close",
        "dismiss",
        "not now",
        "continue",
        "enter",
        "yes",
        "i am",
        "i'm",
        "over 18",
        "18+",
        "over 21",
        "21+",
    ]
    for it in items or []:
        n = (it.get("name") or "").strip()
        low = n.lower()
        if n and any(w in low for w in gate_words):
            possible_gate_controls.append({"role": it.get("role"), "name": n})
        if len(possible_gate_controls) >= 12:
            break

    excerpt = ""
    if mode != PAGE_MODE_NORMAL and text_norm:
        excerpt = text_norm[:400]

    success_signals: Dict[str, Any] = {
        "generator_goal": False,
        "generation_in_progress": False,
        "topic_match_score": 0,
        "has_output_controls": False,
        "has_output_words": False,
        "success_likely": False,
        "judge_called": False,
        "judge_done": False,
        "judge_confidence": 0.0,
        "judge_reason": "",
        "success_confirmed": False,
    }
    try:
        g = (goal or "").lower()
        u = (url or "").lower()
        generator_goal = bool(
            ("gamma.app" in u)
            or ("gamma" in g)
            or re.search(r"\b(ppt|powerpoint|presentation|slides|deck|slideshow)\b", g)
        )
        meeting_like = bool(("meet.google.com" in u) or re.search(r"\b(meeting|meet|video call|call|conference)\b", g))
        if meeting_like:
            generator_goal = False

        body_l = (body_text or "").lower()
        generation_in_progress = bool(
            re.search(r"\b(generating|creating|building|working on it|please wait|preparing)\b", body_l)
        )
        names_l = " ".join([(it.get("name") or "") for it in (items or [])]).lower()
        output_controls = any(w in names_l for w in ["share", "export", "download", "present"])
        output_words = any(w in body_l for w in ["share", "export", "download", "present", "slides", "slide", "outline"])

        topic_match_score = 0
        try:
            ignore = {
                "gamma",
                "ppt",
                "powerpoint",
                "presentation",
                "present",
                "slides",
                "slide",
                "create",
                "generate",
                "make",
                "new",
            }
            for tok in _goal_tokens(goal):
                if tok in ignore or len(tok) < 4:
                    continue
                if tok in (title or "").lower() or tok in body_l:
                    topic_match_score += 1
        except Exception:
            topic_match_score = 0

        success_likely = bool(
            generator_goal
            and (not generation_in_progress)
            and (topic_match_score >= 1)
            and (output_controls or output_words)
        )

        base_signals: Dict[str, Any] = {
            "generator_goal": bool(generator_goal),
            "generation_in_progress": bool(generation_in_progress),
            "topic_match_score": int(topic_match_score),
            "has_output_controls": bool(output_controls),
            "has_output_words": bool(output_words),
            "success_likely": bool(success_likely),
        }

        judge_called = False
        judge_done = False
        judge_conf = 0.0
        judge_reason = ""
        success_confirmed = False

        state_key = ""
        try:
            host = urlparse(url).hostname or ""
        except Exception:
            host = ""
        try:
            state_key = hashlib.sha256((host + "|" + (goal or "")).encode("utf-8")).hexdigest()[:18]
        except Exception:
            state_key = host + "|" + (goal or "")

        st = _SUCCESS_JUDGE_STATE.get(state_key) or {}
        jr = None
        if base_signals.get("generator_goal") and base_signals.get("success_likely") and (not base_signals.get("generation_in_progress")):
            prev_likely = False
            try:
                prev_likely = bool(st.get("last_success_likely"))
            except Exception:
                prev_likely = False
            if not prev_likely:
                judge_called = True
                try:
                    snippet = text_norm[:700] if text_norm else (body_text or "")[:700]
                except Exception:
                    snippet = (body_text or "")[:700]
                jr = success_judge(goal, url, dom_hash or "", title or "", snippet, items or [], base_signals)
                st["last_judge"] = jr
                st["last_dom_hash"] = dom_hash or ""
            else:
                jr = st.get("last_judge")
            st["last_success_likely"] = True
        else:
            st["last_success_likely"] = False
            st["last_judge"] = None
            st["last_dom_hash"] = dom_hash or ""
        _SUCCESS_JUDGE_STATE[state_key] = st

        if isinstance(jr, dict):
            try:
                judge_done = bool(jr.get("done"))
            except Exception:
                judge_done = False
            try:
                judge_conf = float(jr.get("confidence") or 0.0)
            except Exception:
                judge_conf = 0.0
            try:
                jr_reason = jr.get("reason")
                judge_reason = jr_reason if isinstance(jr_reason, str) else (str(jr_reason) if jr_reason is not None else "")
            except Exception:
                judge_reason = ""
            try:
                min_conf = float(getattr(config, "SUCCESS_JUDGE_MIN_CONF", 0.8))
            except Exception:
                min_conf = 0.8
            success_confirmed = bool(base_signals.get("success_likely") and judge_done and (judge_conf >= min_conf))

        success_signals = {
            **base_signals,
            "judge_called": bool(judge_called),
            "judge_done": bool(judge_done),
            "judge_confidence": float(judge_conf),
            "judge_reason": str(judge_reason or ""),
            "success_confirmed": bool(success_confirmed),
        }
    except Exception:
        pass

    return {
        "mode": mode,
        "url": url,
        "title": title,
        "role_counts": role_counts,
        "gate_scores": scores,
        "has_modal": has_modal,
        "login_like": login_like,
        "inputs": inputs,
        "possible_gate_controls": possible_gate_controls,
        "top_controls": top_controls,
        "text_excerpt": excerpt,
        "success_signals": success_signals,
        "goal": goal,
    }


def _score_gate_candidate(mode: str, it: Dict[str, Any]) -> int:
    try:
        if it.get("is_money_action"):
            return -10**6
    except Exception:
        pass
    role = str(it.get("role") or "")
    name = str(it.get("name") or "")
    n = name.lower().strip()
    s = 0
    if role == "button":
        s += 3
    elif role == "link":
        s += 1

    if mode == PAGE_MODE_COOKIE_GATE:
        if any(x in n for x in ["accept all", "allow all", "agree all"]):
            s += 60
        if any(x in n for x in ["accept", "agree", "allow", "got it", "ok", "okay", "i understand"]):
            s += 45
        if any(x in n for x in ["reject", "decline"]):
            s += 35
        if any(x in n for x in ["manage", "preferences", "settings", "options"]):
            s += 20
        if any(x in n for x in ["close", "dismiss", "not now"]):
            s += 15

    elif mode == PAGE_MODE_AGE_GATE:
        if any(x in n for x in ["18+", "over 18", "i am 18", "i'm 18", "i am over 18", "i'm over 18"]):
            s += 60
        if any(x in n for x in ["21+", "over 21", "i am 21", "i'm 21", "i am over 21", "i'm over 21"]):
            s += 55
        if any(x in n for x in ["yes", "enter", "continue", "confirm", "submit"]):
            s += 30
        if any(x in n for x in ["no", "under 18", "i am under"]):
            s -= 100

    elif mode == PAGE_MODE_BLOCKING_MODAL:
        if n in {"x", "close"}:
            s += 60
        if any(x in n for x in ["close", "dismiss"]):
            s += 45
        if any(x in n for x in ["not now", "no thanks"]):
            s += 35

    return s


def try_handle_gate(
    page: Page,
    goal: str,
    downloaded_files: Optional[List[str]] = None,
    planner_assist: Optional[Dict[str, Any]] = None,
    dom_mode: str = "legacy",
    max_items: int = 40,
) -> Optional[str]:
    blocked: Set[Tuple[Any, ...]] = set()
    for _ in range(3):
        try:
            filtered_summary = filter_interactive_elements(page, goal, planner_assist, dom_mode=dom_mode, max_items=max_items)
            dom_hash = str(hash(filtered_summary))[-8:]
            try:
                save_filtered_dom(goal, filtered_summary)
            except Exception:
                pass
            items = json.loads(filtered_summary)
        except Exception:
            items = []
            dom_hash = ""
        try:
            perception = build_perception(page, items, goal, dom_hash)
            mode = str(perception.get("mode") or PAGE_MODE_NORMAL)
        except Exception:
            mode = PAGE_MODE_NORMAL
            perception = {"mode": PAGE_MODE_NORMAL}

        # Decide if this modal should be auto-handled as a blocking gate or left to the main task logic.
        inputs_for_gate: List[Dict[str, Any]] = []
        scores_for_gate: Dict[str, Any] = {}
        try:
            inputs_for_gate = list(perception.get("inputs") or [])
        except Exception:
            inputs_for_gate = []
        try:
            scores_for_gate = dict(perception.get("gate_scores") or {})
        except Exception:
            scores_for_gate = {}
        try:
            cookie_score = int(scores_for_gate.get("cookie_score", 0) or 0)
        except Exception:
            cookie_score = 0
        try:
            age_score = int(scores_for_gate.get("age_score", 0) or 0)
        except Exception:
            age_score = 0

        modal_gate_like = False
        if mode == PAGE_MODE_BLOCKING_MODAL:
            # Treat generic overlays with no form inputs as gates; keep form modals (with inputs) for the main task flow.
            if len(inputs_for_gate) == 0:
                modal_gate_like = True
            else:
                try:
                    if cookie_score >= 2 or age_score >= 2:
                        modal_gate_like = True
                except Exception:
                    modal_gate_like = False

        try:
            if dom_hash:
                save_perception(goal, perception, dom_hash)
        except Exception:
            pass

        # Auto-handle only clear cookie/age gates, or simple blocking modals without form inputs.
        if mode not in {PAGE_MODE_COOKIE_GATE, PAGE_MODE_AGE_GATE} and not (mode == PAGE_MODE_BLOCKING_MODAL and modal_gate_like):
            return None

        best = None
        best_score = 0
        for it in items or []:
            key = (it.get("role"), it.get("name"))
            if key in blocked:
                continue
            sc = _score_gate_candidate(mode, it)
            if sc > best_score:
                best_score = sc
                best = it

        if not best or best_score <= 0:
            return None

        decision = {
            "action": "click",
            "locator_type": "role",
            "role": best.get("role"),
            "name": best.get("name"),
            "confidence": 0.85,
        }
        ok = execute_action(page, decision, downloaded_files)
        if not ok:
            blocked.add((best.get("role"), best.get("name")))
            continue

        try:
            page.wait_for_load_state("domcontentloaded", timeout=5000)
        except Exception:
            pass
        try:
            page.wait_for_timeout(800)
        except Exception:
            pass

        try:
            after_sum = playwright_filter_interactive_elements(page, goal)
            after_dom_hash = str(hash(after_sum))[-8:]
            after_items = json.loads(after_sum)
            after_mode = str(build_perception(page, after_items, goal, after_dom_hash).get("mode") or PAGE_MODE_NORMAL)
        except Exception:
            after_mode = PAGE_MODE_NORMAL

        if after_mode != mode:
            return f"gate:{mode}:click:{best.get('role')}:{best.get('name')}"

    return None

# --- Core Logic ---

def extract_search_query(goal: str) -> str:
    """Very simple heuristic to extract a search query from the goal."""
    g = goal.lower()
    # Remove common boilerplate words
    stop = {
        "go", "goto", "to", "and", "on", "in", "the", "a", "an", "video", "videos",
        "play", "watch", "open", "search", "for", "from", "website", "site", "page",
        "youtube", "amazon", "flipkart",
    }
    tokens = [t for t in re.split(r"[^a-z0-9+]+", g) if t]
    cleaned = [t for t in tokens if t not in stop]
    # Join remaining words; fallback to original goal if empty
    query = " ".join(cleaned).strip()
    return query or goal


def has_search_box(page: Page) -> bool:
    try:
        # YouTube uses "Search" combobox/textbox
        if page.get_by_role("combobox", name=re.compile("search", re.I)).count():
            return True
    except Exception:
        pass
    try:
        if page.get_by_role("textbox", name=re.compile("search", re.I)).count():
            return True
    except Exception:
        pass
    try:
        if page.get_by_placeholder(re.compile("search", re.I)).count():
            return True
    except Exception:
        pass
    return False

def quick_download_if_present(page: Page, downloaded_files: List[str]) -> bool:
    def _save(dl) -> Optional[str]:
        try:
            fname = getattr(dl, "suggested_filename", None) or "download_" + time.strftime("%Y%m%d_%H%M%S")
            base, ext = os.path.splitext(fname)
            out = os.path.join(DOWNLOAD_DIR, fname)
            i = 1
            while os.path.exists(out):
                out = os.path.join(DOWNLOAD_DIR, f"{base} ({i}){ext}")
                i += 1
            dl.save_as(out)
            print(f"Saved download to: {out}")
            downloaded_files.append(out)
            return out
        except Exception:
            return None

    for role in ("button", "link"):
        try:
            cand = page.get_by_role(role, name=re.compile(r"\b(download|save( image)?)\b", re.I)).first
            if cand and cand.count():
                try:
                    with page.context.expect_download(timeout=8000) as dl_info:
                        cand.click(timeout=8000)
                    if _save(dl_info.value):
                        return True
                except Exception:
                    pass
        except Exception:
            pass
    css_list = [
        "a[download]",
        "a[href*='download']",
        "a[href*='force=true']",
        "a[href*='dl=1']",
        "a[href$='.jpg'], a[href$='.jpeg'], a[href$='.png'], a[href$='.gif'], a[href$='.webp']",
        "button:has-text('Download')",
    ]
    for css in css_list:
        try:
            cand = page.locator(css).first
            if cand and cand.count():
                try:
                    with page.context.expect_download(timeout=8000) as dl_info:
                        cand.click(timeout=8000)
                    if _save(dl_info.value):
                        return True
                except Exception:
                    pass
        except Exception:
            pass
    try:
        try:
            page.wait_for_selector("img", timeout=1500)
        except Exception:
            pass
        img_url = page.evaluate(
            """
            () => {
              const imgs = Array.from(document.querySelectorAll('img'));
              if (!imgs.length) return null;
              const pick = imgs
                .map(img => ({ img, area: (img.naturalWidth||0)*(img.naturalHeight||0) }))
                .sort((a,b) => b.area - a.area)[0]?.img;
              if (!pick) return null;
              return pick.currentSrc || pick.src || null;
            }
            """
        )
        if isinstance(img_url, str) and img_url.strip().lower().startswith("http"):
            try:
                req = urllib.request.Request(img_url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=10) as resp:
                    data = resp.read()
                base = os.path.basename(img_url.split("?")[0].split("#")[0]) or ("image_" + time.strftime("%Y%m%d_%H%M%S") + ".jpg")
                out = os.path.join(DOWNLOAD_DIR, base)
                i = 1
                root, ext = os.path.splitext(out)
                if not ext:
                    out = out + ".jpg"
                    root, ext = os.path.splitext(out)
                while os.path.exists(out):
                    out = f"{root} ({i}){ext}"
                    i += 1
                with open(out, "wb") as f:
                    f.write(data)
                print(f"Saved download to: {out}")
                downloaded_files.append(out)
                return True
            except Exception:
                pass
    except Exception:
        pass
    return False

def _goal_tokens(goal: str) -> List[str]:
    g = goal.lower()
    stop = {
        "go", "goto", "to", "and", "on", "in", "the", "a", "an", "website", "site",
        "open", "visit", "page", "from", "for", "please", "kindly",
    }
    toks = [t for t in re.split(r"[^a-z0-9]+", g) if t]
    return [t for t in toks if t and t not in stop]

def score_page_for_goal(page: Page, goal: str) -> int:
    try:
        url = (page.url or "").lower()
    except Exception:
        url = ""
    try:
        title = (page.title() or "").lower()
    except Exception:
        title = ""
    toks = _goal_tokens(goal)
    s = 0
    # Token matches in URL and title
    for t in toks:
        if t and t in url:
            s += 3
        if t and t in title:
            s += 2
    # Common intents
    if "download" in goal.lower():
        if "download" in url or "download" in title:
            s += 6
        # Known hosters or direct-download hints
        try:
            if re.search(r"(uploadhaven|mega|mediafire|gofile|anonfile|anonfiles|dropbox|drive\.google|pixeldrain|direct|download)", url):
                s += 4
        except Exception:
            pass
    if any(k in goal.lower() for k in ["admission", "apply", "application", "form"]):
        if any(k in url or k in title for k in ["admission", "apply", "application", "form"]):
            s += 4
    # Slight preference for pages not being the homepage root if the goal is specific
    if toks and url.rstrip("/").count("/") >= 3:
        s += 1
    return s


def _is_docs_goal(prompt: str) -> bool:
    g = prompt.lower()
    docs_keywords = [
        "documentation",
        "docs",
        "doc ",
        "help page",
        "support page",
        "faq",
        "guide",
        "tutorial",
        "how to",
        "read about",
        "learn about",
    ]
    return any(k in g for k in docs_keywords)


def _is_action_goal(prompt: str) -> bool:
    g = prompt.lower()
    verbs = [
        "open ",
        "go to",
        "goto",
        "visit ",
        "start ",
        "create ",
        "write ",
        "post ",
        "send ",
        "join ",
        "play ",
        "book ",
        "buy ",
        "download",
        "upload",
        "search",
        "sign in",
        "log in",
        "login",
        "sign up",
        "register",
    ]
    return any(v in g for v in verbs)


def _normalize_initial_target(prompt: str, url: str) -> str:
    """Heuristically rewrite help/support/docs URLs to the main site for action goals.

    Example: if Serper gives https://help.x.com/en/using-x/how-to-post for
    "go to x and write a new tweet", we prefer https://x.com/ instead.
    """
    try:
        parsed = urlparse(url)
    except Exception:
        return url
    host = (parsed.hostname or "").lower()
    path = (parsed.path or "").lower()
    if not host:
        return url

    # If the user explicitly asked for docs/help, trust the docs URL.
    if _is_docs_goal(prompt):
        return url

    # Only rewrite when the goal is clearly action-oriented ("do the thing").
    if not _is_action_goal(prompt):
        return url

    docs_host_prefixes = (
        "help.",
        "support.",
        "docs.",
        "doc.",
        "kb.",
        "developer.",
        "developers.",
    )
    docs_path_keywords = (
        "/help",
        "/support",
        "/docs",
        "/documentation",
        "/guide",
        "/tutorial",
        "/faq",
        "/article",
    )

    is_docs_host = any(host.startswith(p) for p in docs_host_prefixes)
    is_docs_path = any(k in path for k in docs_path_keywords)
    if not (is_docs_host or is_docs_path):
        return url

    # Collapse help/support/docs subdomains down to the main site root.
    parts = host.split(".")
    apex = host
    if len(parts) > 2:
        apex = ".".join(parts[1:])
    if apex.startswith("www."):
        apex = apex[4:]
    if not apex:
        return url

    new_url = f"https://{apex}/"
    print(f"Heuristic: rewriting docs/support URL '{url}' to main site '{new_url}' for action-oriented goal.")
    return new_url

def choose_best_page(pages: List[Page], goal: str) -> Optional[Page]:
    best = None
    best_score = -10**9
    for p in pages:
        try:
            if p.is_closed():
                continue
        except Exception:
            continue
        sc = score_page_for_goal(p, goal)
        if sc > best_score:
            best_score = sc
            best = p
    return best

def _pick_url_from_serper(query: str, data: Dict[str, Any]) -> Optional[str]:
    candidates: List[Dict[str, Any]] = []
    kg = data.get("knowledgeGraph") or {}
    website = kg.get("website")
    if isinstance(website, str) and website.strip().startswith("http"):
        title = kg.get("title") or kg.get("name") or ""
        candidates.append({"url": website.strip(), "title": str(title)})
    organic = data.get("organic") or []
    if isinstance(organic, list):
        for res in organic[:5]:
            link = res.get("link")
            if isinstance(link, str) and link.strip().startswith("http"):
                url = link.strip()
                title = res.get("title") or ""
                snippet = res.get("snippet") or ""
                if not any(c.get("url") == url for c in candidates):
                    candidates.append(
                        {
                            "url": url,
                            "title": str(title),
                            "snippet": str(snippet),
                        }
                    )
    if not candidates:
        return None
    try:
        payload = {
            "goal": query,
            "candidates": candidates,
        }
        completion = client.chat.completions.create(
            model=config.SMART_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a URL chooser for a web automation agent. "
                        "The agent will open ONE website and then act on it to achieve the user's goal. "
                        "From the candidate URLs, pick the SINGLE best starting URL. "
                        "Prefer official product/app pages that allow performing the action directly "
                        "over tutorials, videos, or news articles. "
                        "When the goal is to CREATE/START/USE something (e.g., a Google Form), "
                        "prefer the actual app site (for example forms.google.com or docs.google.com/forms) "
                        "over YouTube or long tutorials, unless no such site is present. "
                        "Respond as a JSON object with a single key 'url' whose value is the chosen URL."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(payload, ensure_ascii=False),
                },
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
        )
        try:
            content = completion.choices[0].message.content
        except Exception:
            content = None
        if not content:
            return None
        try:
            obj = json.loads(content)
        except Exception:
            return None
        chosen = obj.get("url") or obj.get("best_url") or obj.get("target_url")
        if isinstance(chosen, str) and chosen.strip().startswith("http"):
            print(f"Serper introspection chose URL: {chosen.strip()}")
            return chosen.strip()
    except Exception as e:
        print(f"Serper introspection failed; falling back to default Serper heuristic: {e}")
    return None


def resolve_site_url_via_serper(query: str) -> Optional[str]:
    """Resolve a site URL from natural language using Serper.dev (bot-friendly Google Search API).

    Prefers Knowledge Graph website, otherwise first organic link.
    Returns absolute URL if found, else None.
    """
    api_key = os.getenv("SERPER_API_KEY")
    if not api_key:
        return None
    try:
        req = urllib.request.Request(
            "https://google.serper.dev/search",
            data=json.dumps({"q": query}).encode("utf-8"),
            headers={
                "X-API-KEY": api_key,
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        chosen = _pick_url_from_serper(query, data)
        if isinstance(chosen, str) and chosen.strip().startswith("http"):
            return chosen.strip()
        kg = (data.get("knowledgeGraph") or {})
        website = kg.get("website")
        if isinstance(website, str) and website.strip().startswith("http"):
            return website.strip()
        organic = data.get("organic") or []
        if organic and isinstance(organic, list):
            link = organic[0].get("link")
            if isinstance(link, str) and link.strip().startswith("http"):
                return link.strip()
    except Exception as e:
        print(f"Serper resolve failed: {e}")
    return None


def perform_search(page: Page, query: str) -> bool:
    """Try common patterns to perform a search on the current site."""
    print(f"Heuristic: typing search query: {query!r}")
    try:
        box = None
        try:
            box = page.get_by_role("combobox", name=re.compile("search", re.I)).first
        except Exception:
            pass
        if not box:
            try:
                box = page.get_by_role("textbox", name=re.compile("search", re.I)).first
            except Exception:
                pass
        if not box:
            try:
                box = page.get_by_placeholder(re.compile("search", re.I)).first
            except Exception:
                pass
        if box:
            box.click()
            # Avoid re-typing if already present
            try:
                existing = box.input_value().strip()
            except Exception:
                try:
                    existing = box.locator("input").first.input_value().strip()
                except Exception:
                    existing = ""
            if existing and existing.lower() == query.lower():
                print("Search box already contains query; skipping retype/enter.")
                return False
            box.fill(query)
            page.keyboard.press("Enter")
            page.wait_for_timeout(2000)
            return True
    except Exception:
        pass
    return False

def plan_initial_url(prompt: str) -> str:
    """Phase 1: Extract and normalize the target URL from the prompt."""
    print(f"Extracting target URL for: '{prompt}'")
    url: Optional[str] = None

    # 0) Explicit URL in prompt
    m = re.search(r"https?://\S+", prompt)
    if m:
        url = m.group(0).strip()
    else:
        # 1) Bot-friendly search API (Serper.dev)
        url = resolve_site_url_via_serper(prompt)

        # 2) LLM extraction fallback (Groq)
        if not url:
            for attempt in range(3):
                try:
                    completion = client.chat.completions.create(
                        model=config.SMART_MODEL,
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "You are a URL extractor. Return ONLY the full valid URL (starting with https://) for the website "
                                    "mentioned in the prompt. If no specific site is mentioned, default to https://www.google.com. "
                                    "Do not output anything else."
                                ),
                            },
                            {"role": "user", "content": prompt},
                        ],
                        temperature=0.1,
                    )
                    url = completion.choices[0].message.content.strip()
                    break
                except Exception as e:
                    wait = 1.0 * (2 ** attempt)
                    print(f"URL planning error/rate-limit; retrying in {wait:.1f}s: {e}")
                    time.sleep(wait)

    if not url:
        url = "https://www.google.com"

    # Heuristically prefer the main site over docs/help/support URLs for action goals.
    url = _normalize_initial_target(prompt, url)
    print(f"Target URL: {url}")
    return url


_PLANNER_ASSIST_CACHE: Dict[str, Dict[str, Any]] = {}
_SUCCESS_JUDGE_CACHE: Dict[str, Dict[str, Any]] = {}
_SUCCESS_JUDGE_STATE: Dict[str, Dict[str, Any]] = {}


def _repair_truncated_json(text: str) -> Optional[str]:
    if not text or not isinstance(text, str):
        return None
    s = text
    try:
        start = s.find("{")
        if start != -1:
            s = s[start:]
    except Exception:
        s = text
    stack: List[str] = []
    in_string = False
    escape = False
    for ch in s:
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            stack.append("}")
        elif ch == "[":
            stack.append("]")
        elif ch == "}" or ch == "]":
            if stack:
                stack.pop()
    if in_string:
        s += '"'
    while stack:
        s += stack.pop()
    return s


def _try_parse_json_object(text: str) -> Optional[Dict[str, Any]]:
    if not text or not isinstance(text, str):
        return None
    s = text.strip()
    if s.startswith("```"):
        try:
            s = re.sub(r"^```[a-zA-Z0-9]*\n", "", s).strip()
            s = re.sub(r"\n```$", "", s).strip()
        except Exception:
            s = text.strip()
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    try:
        i = s.find("{")
        j = s.rfind("}")
        if i != -1 and j != -1 and j > i:
            obj = json.loads(s[i : j + 1])
            if isinstance(obj, dict):
                return obj
    except Exception:
        pass
    try:
        repaired = _repair_truncated_json(s)
        if repaired and repaired != s:
            obj = json.loads(repaired)
            if isinstance(obj, dict):
                return obj
    except Exception:
        pass
    return None


def _success_judge_cache_key(goal: str, current_url: str, dom_hash: str, title: str, body_snippet: str) -> str:
    base = (goal or "") + "|" + (current_url or "") + "|" + (dom_hash or "")
    try:
        return hashlib.sha256(base.encode("utf-8")).hexdigest()[:18]
    except Exception:
        return base[:18]


def success_judge(
    goal: str,
    current_url: str,
    dom_hash: str,
    title: str,
    body_snippet: str,
    controls: List[Dict[str, Any]],
    heuristics: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    try:
        if not config.SUCCESS_JUDGE_ENABLED:
            return None
    except Exception:
        return None
    api_key = getattr(config, "GEMINI_API_KEY", None)
    if not api_key:
        return None
    model = getattr(config, "SUCCESS_JUDGE_MODEL", None) or getattr(config, "GEMINI_MODEL", None) or "openai/gpt-4.1-mini"
    try:
        max_tokens = int(getattr(config, "SUCCESS_JUDGE_MAX_TOKENS", 220))
    except Exception:
        max_tokens = 220

    key = _success_judge_cache_key(goal, current_url, dom_hash, title, body_snippet)
    if key in _SUCCESS_JUDGE_CACHE:
        return _SUCCESS_JUDGE_CACHE.get(key)

    compact_controls: List[Dict[str, Any]] = []
    try:
        for it in (controls or [])[:24]:
            compact_controls.append({"role": it.get("role"), "name": it.get("name")})
    except Exception:
        compact_controls = []

    prompt = (
        "You are a strict success judge for a web automation agent. "
        "Your job is to decide whether the user's goal is FULLY achieved on the current page right now. "
        "Be conservative: return done=true only if you are confident the goal is satisfied. "
        "Return ONLY valid JSON (no markdown) with schema: {\\\"done\\\": bool, \\\"confidence\\\": number, \\\"reason\\\": string}. "
        "Confidence must be between 0 and 1.\\n\\n"
        f"Goal: {goal}\\n"
        f"URL: {current_url}\\n"
        f"Title: {title}\\n"
        f"DOM hash: {dom_hash}\\n"
        f"Heuristic signals (JSON): {json.dumps(heuristics or {}, ensure_ascii=False)}\\n"
        f"Visible text snippet: {body_snippet}\\n"
        f"Top controls (role/name): {json.dumps(compact_controls, ensure_ascii=False)}\\n"
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }
    try:
        endpoint = "https://openrouter.ai/api/v1/chat/completions"
        req = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            raw = resp.read().decode("utf-8")
        data = json.loads(raw)
        text = None
        try:
            text = data.get("choices", [{}])[0].get("message", {}).get("content")
        except Exception:
            text = None

        try:
            raw_record = {
                "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                "goal": goal,
                "url": current_url,
                "model": model,
                "dom_hash": dom_hash,
                "cache_key": key,
                "raw_response": data,
                "raw_text": text,
            }
            with open("success_judge_raw_log.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(raw_record, ensure_ascii=False) + "\n")
            with open("last_success_judge_raw.json", "w", encoding="utf-8") as f:
                json.dump(raw_record, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

        obj = _try_parse_json_object(text or "")
        if not obj:
            return None
        done = bool(obj.get("done"))
        conf = obj.get("confidence")
        try:
            conf_f = float(conf) if conf is not None else 0.0
        except Exception:
            conf_f = 0.0
        reason = obj.get("reason")
        if not isinstance(reason, str):
            reason = str(reason) if reason is not None else ""
        out = {"done": done, "confidence": conf_f, "reason": reason}

        try:
            record = {
                "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                "goal": goal,
                "url": current_url,
                "model": model,
                "dom_hash": dom_hash,
                "cache_key": key,
                "result": out,
            }
            with open("success_judge_log.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            with open("last_success_judge.json", "w", encoding="utf-8") as f:
                json.dump(record, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

        _SUCCESS_JUDGE_CACHE[key] = out
        return out
    except urllib.error.HTTPError as e:
        try:
            err_body = e.read().decode("utf-8", errors="replace")
        except Exception:
            err_body = "(no body)"
        print(f"Success-judge HTTP {e.code} Error: {err_body}")
        return None
    except Exception as e:
        print(f"Success-judge failed: {e}")
        return None


def _brief_planner_assist(obj: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    try:
        out["site"] = obj.get("site")
    except Exception:
        out["site"] = None
    try:
        steps = obj.get("steps") or []
        if isinstance(steps, list):
            out["steps"] = steps[:8]
        else:
            out["steps"] = []
    except Exception:
        out["steps"] = []
    try:
        hints = obj.get("hints") or []
        if isinstance(hints, list):
            out["hints"] = hints[:8]
        else:
            out["hints"] = []
    except Exception:
        out["hints"] = []
    for k in ("candidate_button_texts", "candidate_field_labels", "success_indicators"):
        try:
            v = obj.get(k)
        except Exception:
            v = None
        if isinstance(v, list):
            out[k] = v[:25]
        else:
            out[k] = v
    return out


def gemini_planner_assist(goal: str, current_url: str, page_title: str = "") -> Optional[Dict[str, Any]]:
    try:
        if not config.GEMINI_PLANNER_ASSIST:
            return None
    except Exception:
        return None
    api_key = getattr(config, "GEMINI_API_KEY", None)
    if not api_key:
        return None
    model = getattr(config, "GEMINI_MODEL", None) or "openai/gpt-4.1-mini"
    try:
        host = urlparse(current_url).hostname or ""
    except Exception:
        host = ""

    prompt = (
        "You are a planning assistant for a web automation agent. "
        "Given a user goal and the current website, produce a short, concrete step-by-step map. "
        "You MUST try to provide exact button/link text labels as they appear in the UI; if unsure, provide multiple variants. "
        "This plan is NON-BINDING guidance: it may be wrong; the agent will only use suggestions that match the page DOM. "
        "Return ONLY valid JSON (no markdown, no code fences) with this schema:\n"
        "{\n"
        "  \"site\": string,\n"
        "  \"steps\": [\n"
        "    {\"step\": int, \"action\": \"click\"|\"type\"|\"wait\", \"target_text\": string, \"alternatives\": [string], \"value\": string|null, \"notes\": string}\n"
        "  ],\n"
        "  \"candidate_button_texts\": [string],\n"
        "  \"candidate_field_labels\": [string],\n"
        "  \"success_indicators\": [string]\n"
        "}\n"
        "Keep steps <= 10 and keep strings short.\n\n"
        f"Current host: {host}\n"
        f"Current URL: {current_url}\n"
        f"Current page title: {page_title}\n"
        f"User goal: {goal}\n"
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 512,
    }
    try:
        endpoint = "https://openrouter.ai/api/v1/chat/completions"
        req = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            raw = resp.read().decode("utf-8")
        data = json.loads(raw)
        text = None
        try:
            text = data.get("choices", [{}])[0].get("message", {}).get("content")
        except Exception:
            text = None

        # Always log the raw Gemini response and extracted text for debugging,
        # even if we fail to parse a structured plan from it.
        try:
            raw_record = {
                "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                "goal": goal,
                "url": current_url,
                "model": model,
                "raw_response": data,
                "raw_text": text,
            }
            with open("planner_assist_raw_log.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(raw_record, ensure_ascii=False) + "\n")
            with open("last_planner_assist_raw.json", "w", encoding="utf-8") as f:
                json.dump(raw_record, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

        plan = _try_parse_json_object(text or "")
        if not plan:
            return None
        record = {
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "goal": goal,
            "url": current_url,
            "model": model,
            "plan": plan,
        }
        try:
            with open("planner_assist_log.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            with open("last_planner_assist.json", "w", encoding="utf-8") as f:
                json.dump(record, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
        return plan
    except urllib.error.HTTPError as e:
        try:
            err_body = e.read().decode("utf-8", errors="replace")
        except Exception:
            err_body = "(no body)"
        print(f"Planner-assist HTTP {e.code} Error: {err_body}")
        return None
    except Exception as e:
        print(f"Planner-assist failed: {e}")
        return None


def _build_planner_gold_overview(perception: Dict[str, Any], items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build a compact, fast overview of the current page for Planner Gold."""
    overview: Dict[str, Any] = {}
    try:
        overview["url"] = perception.get("url")
        overview["title"] = perception.get("title")
        overview["mode"] = perception.get("mode")
        overview["inputs"] = perception.get("inputs") or []
        overview["top_controls"] = perception.get("top_controls") or []
        overview["possible_gate_controls"] = perception.get("possible_gate_controls") or []
        overview["success_signals"] = perception.get("success_signals") or {}
    except Exception:
        overview = {}

    try:
        sample: List[Dict[str, Any]] = []
        for it in (items or [])[:30]:
            if not isinstance(it, dict):
                continue
            sample.append({
                "role": it.get("role"),
                "name": it.get("name"),
                "href_kind": it.get("href_kind"),
                "aria_expanded": it.get("aria_expanded"),
                "is_primary_input": it.get("is_primary_input"),
            })
        overview["controls_sample"] = sample
    except Exception:
        pass
    return overview


def planner_gold_assist(
    goal: str,
    current_url: str,
    page_title: str,
    overview: Dict[str, Any],
    progress: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Planner Gold: lightweight hint generator using current page overview + progress."""
    try:
        if not config.GEMINI_PLANNER_ASSIST:
            return None
    except Exception:
        return None
    api_key = getattr(config, "GEMINI_API_KEY", None)
    if not api_key:
        return None
    model = getattr(config, "GEMINI_MODEL", None) or "openai/gpt-4.1-mini"

    try:
        host = urlparse(current_url).hostname or ""
    except Exception:
        host = ""

    prompt = (
        "You are a fast planning assistant. Provide SHORT hints for the next few actions. "
        "Return ONLY JSON (no markdown) with schema:\n"
        "{\n"
        "  \"site\": string,\n"
        "  \"hints\": [\n"
        "    {\"kind\": \"click\"|\"type\"|\"wait\"|\"navigate\", \"target\": string, \"value\": string|null, \"why\": string}\n"
        "  ],\n"
        "  \"candidate_button_texts\": [string],\n"
        "  \"candidate_field_labels\": [string],\n"
        "  \"success_indicators\": [string]\n"
        "}\n"
        "Rules: keep hints <= 6, be concise, and reference labels actually visible in the overview.\n\n"
        f"Current host: {host}\n"
        f"Current URL: {current_url}\n"
        f"Current page title: {page_title}\n"
        f"User goal: {goal}\n"
        f"Progress summary (JSON): {json.dumps(progress, ensure_ascii=False)}\n"
        f"Page overview (JSON): {json.dumps(overview, ensure_ascii=False)}\n"
    )

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": 420,
    }
    try:
        endpoint = "https://openrouter.ai/api/v1/chat/completions"
        req = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=18) as resp:
            raw = resp.read().decode("utf-8")
        data = json.loads(raw)
        text = None
        try:
            text = data.get("choices", [{}])[0].get("message", {}).get("content")
        except Exception:
            text = None

        try:
            raw_record = {
                "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                "goal": goal,
                "url": current_url,
                "model": model,
                "raw_response": data,
                "raw_text": text,
                "overview": overview,
                "progress": progress,
            }
            with open("planner_gold_raw_log.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(raw_record, ensure_ascii=False) + "\n")
            with open("last_planner_gold_raw.json", "w", encoding="utf-8") as f:
                json.dump(raw_record, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

        plan = _try_parse_json_object(text or "")
        if not plan:
            return None
        record = {
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "goal": goal,
            "url": current_url,
            "model": model,
            "plan": plan,
        }
        try:
            with open("planner_gold_log.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            with open("last_planner_gold.json", "w", encoding="utf-8") as f:
                json.dump(record, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
        return plan
    except urllib.error.HTTPError as e:
        try:
            err_body = e.read().decode("utf-8", errors="replace")
        except Exception:
            err_body = "(no body)"
        print(f"Planner-gold HTTP {e.code} Error: {err_body}")
        return None
    except Exception as e:
        print(f"Planner-gold failed: {e}")
        return None


def _build_planner_gold_progress(
    run_memory: Optional[Dict[str, Any]],
    recent_actions: Optional[List[str]] = None,
    step: Optional[int] = None,
    current_url: Optional[str] = None,
) -> Dict[str, Any]:
    progress: Dict[str, Any] = {}
    try:
        progress["step"] = int(step or 0)
    except Exception:
        progress["step"] = 0
    try:
        progress["url"] = current_url or ""
    except Exception:
        progress["url"] = ""
    try:
        if isinstance(recent_actions, list):
            progress["recent_actions"] = recent_actions[-6:]
    except Exception:
        pass
    try:
        if isinstance(run_memory, dict):
            progress["milestones"] = (run_memory.get("milestones") or [])[:12]
            progress["recent_steps"] = (run_memory.get("recent_steps") or [])[-6:]
            progress["progress_notes"] = (run_memory.get("progress_notes") or [])[-12:]
            progress["repeat_streak"] = int(run_memory.get("repeat_streak") or 0)
    except Exception:
        pass
    return progress


def get_planner_assist(goal: str, current_url: str, page_title: str = "") -> Optional[Dict[str, Any]]:
    try:
        if not config.GEMINI_PLANNER_ASSIST:
            return None
    except Exception:
        return None
    try:
        host = urlparse(current_url).hostname or ""
    except Exception:
        host = ""
    try:
        key = hashlib.sha256((host + "|" + (goal or "")).encode("utf-8")).hexdigest()[:18]
    except Exception:
        key = host + "|" + (goal or "")
    if key in _PLANNER_ASSIST_CACHE:
        return _PLANNER_ASSIST_CACHE.get(key)
    plan = gemini_planner_assist(goal, current_url, page_title)
    if isinstance(plan, dict):
        _PLANNER_ASSIST_CACHE[key] = plan
        return plan
    return None

def analyze_dom_and_act(
    page: Page,
    goal: str,
    current_url: str,
    recent_actions: List[str],
    blocked_choices: Union[List[Tuple[Any, ...]], Set[Tuple[Any, ...]]],
    downloaded_files: Optional[List[str]] = None,
    planner_assist: Optional[Dict[str, Any]] = None,
    dom_mode: str = "legacy",
    max_items: int = 40,
    upload_done: bool = False,
    run_memory: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Phase 2: ReAct Loop - Analyze DOM and decide NEXT action based on GOAL."""
    
    # 1-2. Deterministic filter via Playwright AX snapshot (replaces Ollama)
    filtered_summary = filter_interactive_elements(page, goal, planner_assist, dom_mode=dom_mode, max_items=max_items)
    dom_hash = str(hash(filtered_summary))[-8:]
    print(f"Filtered DOM Hash: {dom_hash} | Length: {len(filtered_summary)}")
    save_filtered_dom(goal, filtered_summary, dom_hash, current_url)
    
    # 3. Decide Action (Groq)
    print("Deciding next action...")
    dm = str(dom_mode or "legacy").strip().lower()
    if dm == "gold":
        system_prompt = (
            "You are an autonomous web agent. "
            f"Your ultimate goal is: '{goal}'. "
            "You are looking at the current interactive elements of the page. "
            "Decide the IMMEDIATE NEXT action to move closer to the goal. "
            "Return a JSON object with keys: action, id, value, confidence. "
            "Rules: "
            "- action must be one of: 'click', 'type', 'hover', 'wait', 'done', 'fail'. "
            "- You MUST choose a control by numeric 'id' from the list. "
            "- value is only for 'type' action (the text to enter). "
            "- Prefer real navigation links (href_kind='nav') over anchors ('#') or javascript links. "
            "- If a top-level control toggles a menu (aria-expanded changes), choose a menu item next. "
            "- If Perception.success_signals.success_confirmed=true and generation_in_progress=false, you should choose action='done'. "
            "- confidence: 0.0..1.0. "
            "Safety & Looping: "
            "1. If the goal content is already visible, return 'done'. Do NOT navigate away. "
            "2. If you already searched and see results, click a result link; do NOT search again. "
            "3. If an input already has the correct value, do NOT type again. Prefer clicking search or a result. "
            "4. Avoid repeating the same click if the URL does not change. If you are stuck in a repeated loop, choose a different control (prefer nav links or exploratory items), or return 'fail'. "
            "5. If upload_done=true (provided in context), do NOT click upload/choose-file/select-file controls again unless the goal explicitly asks to upload another or replace. Prefer editing controls (filters/adjust/effects). "
            "6. You will be given a run_memory JSON summary containing recent actions and whether they succeeded, changed URL, or made no progress. Use it to avoid repeating actions that already failed or caused no progress; pick a different control and move forward. "
            "5. If a file has been saved to disk that satisfies the goal (see 'Downloads this run'), return 'done'. "
            "6. If the goal includes download/save, prioritize controls whose names include 'download' or 'save'. "
            "Page modes: you will also be given a 'mode' string such as NORMAL, LOGIN, AGE_GATE, COOKIE_GATE, BLOCKING_MODAL. "
            "- If mode is COOKIE_GATE, focus on dismissing cookie/consent banners (Accept/Agree/Reject/Manage) before other actions. "
            "- If mode is AGE_GATE, focus on passing age verification (choose 18+/Yes/Enter or fill DOB >= 18) before other actions. "
            "- If mode is BLOCKING_MODAL, treat the modal as part of the current page. First inspect its inputs and buttons: if they clearly relate to the goal, interact with those. Only close/dismiss if unrelated.")
    else:
        system_prompt = (
            "You are an autonomous web agent. "
            f"Your ultimate goal is: '{goal}'. "
            "You are looking at the current interactive elements of the page. "
            "Decide the IMMEDIATE NEXT action to move closer to the goal. "
            "Return a JSON object with keys: action, locator_type, role, name, locator, value, confidence. "
            "Rules: "
            "- action must be one of: 'click', 'type', 'hover', 'wait', 'done', 'fail'. "
            "- Choose an element from the list and copy its locator details. Respect the element's locator_type. "
            "  - If the element has locator_type='role', set locator_type='role' and include exact 'role' and 'name' from the list. "
            "  - If the element has locator_type='css', set locator_type='css' and provide 'locator' exactly as given in the list. "
            "- Prefer role-based targeting when available, but do NOT convert a css-only element into a role-based action. "
            "- You may be given a Planner-assist map (Gemini). It is non-binding guidance. Use it only if its suggested control names match the DOM list and the step makes sense. "
            "- Prefer real navigation links (href_kind='nav') over anchors ('#') or javascript links. "
            "- If a top-level control toggles a menu (aria-expanded changes), use 'hover' or a subsequent click on a submenu item. "
            "- value is only for 'type' action (the text to enter). "
            "- When you need to type text for the main task (prompt/title/description), prioritize an element with is_primary_input=true. "
            "- If Perception.success_signals.success_confirmed=true and generation_in_progress=false, you should choose action='done'. "
            "- confidence: 0.0..1.0. "
            "Safety & Looping: "
            "1. If the goal content is already visible, return 'done'. Do NOT navigate away. "
            "2. If you already searched and see results, click a result link; do NOT search again. "
            "3. If an input already has the correct value, do NOT type again. Prefer clicking search or a result. "
            "4. Avoid repeating the same click if the URL does not change. If you are stuck in a repeated loop, choose a different control (prefer nav links or exploratory items), or return 'fail'. "
            "5. If upload_done=true (provided in context), do NOT click upload/choose-file/select-file controls again unless the goal explicitly asks to upload another or replace. Prefer editing controls (filters/adjust/effects). "
            "6. You will be given a run_memory JSON summary containing recent actions and whether they succeeded, changed URL, or made no progress. Use it to avoid repeating actions that already failed or caused no progress; pick a different control and move forward. "
            "5. If a file has been saved to disk that satisfies the goal (see 'Downloads this run'), return 'done'. "
            "6. If the goal includes download/save, prioritize controls whose names include 'download' or 'save'. "
            "Page modes: you will also be given a 'mode' string such as NORMAL, LOGIN, AGE_GATE, COOKIE_GATE, BLOCKING_MODAL. "
            "- If mode is COOKIE_GATE, focus on dismissing cookie/consent banners (Accept/Agree/Reject/Manage) before other actions. "
            "- If mode is AGE_GATE, focus on passing age verification (choose 18+/Yes/Enter or fill DOB >= 18) before other actions. "
            "- If mode is BLOCKING_MODAL, treat the modal as part of the current page. First inspect its inputs and buttons: if they clearly relate to the goal (for example a form to create or configure something, or a Submit/Continue button for this goal), interact with those inputs/buttons instead of closing it. Only close/dismiss the modal (Close/X/Not now) if it appears unrelated to the goal (for example cookie consent, newsletter signup, or generic promotions). "
            "- If mode is LOGIN, prefer login/sign-in controls and do not invent credentials; if you cannot proceed, return 'fail'.")
    
    # Remove blocked (role,name) from the list we show to the LLM
    items_for_llm: List[Dict[str, Any]] = []
    try:
        items_for_llm = json.loads(filtered_summary)
        avoid_set: Set[str] = set()
        try:
            if isinstance(run_memory, dict) and isinstance(run_memory.get("avoid_choices"), list):
                avoid_set = set([str(x) for x in (run_memory.get("avoid_choices") or []) if x is not None])
        except Exception:
            avoid_set = set()
        blocked_set: Set[Tuple[Any, ...]] = set(blocked_choices or [])
        tmp: List[Dict[str, Any]] = []
        for it in items_for_llm:
            k_role = (it.get("role"), it.get("name"))
            if k_role in blocked_set:
                continue
            if it.get("locator_type") == "css" and it.get("locator"):
                k_css = ("css", str(it.get("locator")))
                if k_css in blocked_set:
                    continue
            try:
                if avoid_set:
                    ck = _run_memory_choice_key_from_item(it)
                    if ck and ck in avoid_set:
                        continue
            except Exception:
                pass
            tmp.append(it)
        items_for_llm = tmp

        # Highlight all candidate interactive elements so the user can see what
        # the agent is considering before it chooses a specific action.
        try:
            _highlight_candidate_elements(page, items_for_llm, max_items=max_items)
        except Exception:
            pass

        compact: List[Dict[str, Any]] = []
        for it in items_for_llm:
            ci: Dict[str, Any] = {"role": it.get("role"), "name": it.get("name")}
            if dm != "gold":
                ci["locator_type"] = it.get("locator_type")
                if it.get("locator_type") == "css" and it.get("locator"):
                    ci["locator"] = it.get("locator")
            else:
                if it.get("id") is not None:
                    ci["id"] = it.get("id")
            if it.get("href_kind"):
                ci["href_kind"] = it.get("href_kind")
            if it.get("aria_expanded") is not None:
                ci["aria_expanded"] = it.get("aria_expanded")
            if it.get("target"):
                ci["target"] = it.get("target")
            if dm == "gold":
                if it.get("in_overlay"):
                    ci["in_overlay"] = True
                if it.get("in_viewport"):
                    ci["in_viewport"] = True
            if it.get("is_primary_input"):
                ci["is_primary_input"] = True
            if it.get("is_money_action"):
                ci["is_money_action"] = True
            compact.append(ci)
        filtered_for_llm = json.dumps(compact, ensure_ascii=False)
    except Exception:
        filtered_for_llm = filtered_summary

    perception: Dict[str, Any] = {"mode": PAGE_MODE_NORMAL}
    mode = PAGE_MODE_NORMAL
    try:
        perception = build_perception(page, items_for_llm, goal, dom_hash)
        mode = str(perception.get("mode") or PAGE_MODE_NORMAL)
    except Exception:
        perception = {"mode": PAGE_MODE_NORMAL}
        mode = PAGE_MODE_NORMAL

    try:
        if mode == PAGE_MODE_BLOCKING_MODAL and isinstance(items_for_llm, list) and items_for_llm:
            modal_filtered: List[Dict[str, Any]] = []
            for it in items_for_llm:
                try:
                    lt = it.get("locator_type")
                    if lt == "role":
                        role = it.get("role")
                        name = it.get("name")
                        loc = page.get_by_role(role, name=name).first
                    elif lt == "css" and it.get("locator"):
                        loc = page.locator(str(it.get("locator"))).first
                    else:
                        continue
                    in_modal = False
                    try:
                        in_modal = bool(
                            loc.evaluate(
                                """(el) => {
  try {
    const m = el.closest('[role="dialog"], [role="alertdialog"], [aria-modal="true"]');
    if (!m) return false;
    const st = window.getComputedStyle(m);
    if (!st) return true;
    return st.display !== 'none' && st.visibility !== 'hidden' && st.opacity !== '0';
  } catch (e) { return false; }
}"""
                            )
                        )
                    except Exception:
                        in_modal = False
                    if in_modal:
                        modal_filtered.append(it)
                except Exception:
                    continue
            if len(modal_filtered) >= 2:
                items_for_llm = modal_filtered
                compact: List[Dict[str, Any]] = []
                for it in items_for_llm:
                    ci: Dict[str, Any] = {
                        "role": it.get("role"),
                        "name": it.get("name"),
                        "locator_type": it.get("locator_type"),
                    }
                    if it.get("locator_type") == "css" and it.get("locator"):
                        ci["locator"] = it.get("locator")
                    if it.get("href_kind"):
                        ci["href_kind"] = it.get("href_kind")
                    if it.get("aria_expanded") is not None:
                        ci["aria_expanded"] = it.get("aria_expanded")
                    if it.get("target"):
                        ci["target"] = it.get("target")
                    if it.get("is_primary_input"):
                        ci["is_primary_input"] = True
                    if it.get("is_money_action"):
                        ci["is_money_action"] = True
                    compact.append(ci)
                filtered_for_llm = json.dumps(compact, ensure_ascii=False)
    except Exception:
        pass
    save_perception(goal, perception, dom_hash)
    try:
        print(f"Page mode detected: {mode}")
    except Exception:
        pass
    try:
        perception_json = json.dumps(perception, ensure_ascii=False)
    except Exception:
        perception_json = "{}"

    planner_note = ""
    try:
        if isinstance(planner_assist, dict) and planner_assist:
            brief = _brief_planner_assist(planner_assist)
            planner_note = "Planner-assist map (non-binding JSON):\n" + json.dumps(brief, ensure_ascii=False) + "\n"
    except Exception:
        planner_note = ""

    blocked_txt = "none"
    try:
        if blocked_choices:
            parts = []
            for bc in (blocked_choices or []):
                try:
                    if isinstance(bc, (list, tuple)) and len(bc) == 2:
                        parts.append(f"{bc[0]}:{bc[1]}")
                    else:
                        parts.append(str(bc))
                except Exception:
                    parts.append(str(bc))
            blocked_txt = ", ".join(parts) if parts else "none"
    except Exception:
        blocked_txt = "none"

    rm_compact: Dict[str, Any] = {}
    try:
        if isinstance(run_memory, dict):
            rm_compact = {
                "milestones": (run_memory.get("milestones") or [])[:12] if isinstance(run_memory.get("milestones"), list) else [],
                "avoid_choices": (run_memory.get("avoid_choices") or [])[:12] if isinstance(run_memory.get("avoid_choices"), list) else [],
                "repeat_streak": int(run_memory.get("repeat_streak") or 0),
                "last_choice": run_memory.get("last_choice"),
                "recent_steps": (run_memory.get("recent_steps") or [])[-8:] if isinstance(run_memory.get("recent_steps"), list) else [],
            }
    except Exception:
        rm_compact = {}

    context_note = (
        f"Current URL: {current_url}\n"
        f"Page mode: {mode}\n"
        f"Upload milestone: upload_done={'true' if upload_done else 'false'}\n"
        f"Run memory: {json.dumps(rm_compact or {}, ensure_ascii=False)}\n"
        f"Recent actions: {', '.join(recent_actions[-3:]) if recent_actions else 'none'}\n"
        f"Blocked choices: {blocked_txt}\n"
        f"Downloads this run: {', '.join([os.path.basename(f) for f in (downloaded_files or [])]) if (downloaded_files and len(downloaded_files)>0) else 'none'}\n"
    )
    response = None
    last_err = None
    for attempt in range(3):
        try:
            completion = client.chat.completions.create(
                model=config.SMART_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": context_note
                        + (planner_note or "")
                        + f"Perception (semantic summary JSON):\n{perception_json}\n"
                        + f"Current DOM Elements (JSON array with role/name):\n{filtered_for_llm}",
                    },
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
            )
            response = json.loads(completion.choices[0].message.content)
            break
        except Exception as e:
            last_err = e
            wait = 1.0 * (2 ** attempt)
            print(f"Decision error/rate-limit; retrying in {wait:.1f}s: {e}")
            time.sleep(wait)

    if isinstance(response, dict) and dm == "gold":
        picked = None
        try:
            pid = response.get("id")
            if pid is None:
                pid = response.get("element_id")
            pid_i = int(pid) if pid is not None else None
        except Exception:
            pid_i = None
        if pid_i is not None:
            try:
                for it in items_for_llm:
                    try:
                        if int(it.get("id") or 0) == pid_i:
                            picked = it
                            break
                    except Exception:
                        continue
            except Exception:
                picked = None
        if picked is not None:
            response["locator_type"] = picked.get("locator_type")
            response["role"] = picked.get("role")
            response["name"] = picked.get("name")
            response["locator"] = picked.get("locator")
    if response is None:
        host = urlparse(current_url).hostname or ""
        fallback = None
        blocked_set: Set[Tuple[Any, ...]] = set()
        try:
            blocked_set = set(blocked_choices or [])
        except Exception:
            blocked_set = set()

        def _is_blocked_item(it: Dict[str, Any]) -> bool:
            try:
                if not blocked_set:
                    return False
                if (it.get("role"), it.get("name")) in blocked_set:
                    return True
                if it.get("locator_type") == "css" and it.get("locator"):
                    if ("css", str(it.get("locator"))) in blocked_set:
                        return True
            except Exception:
                return False
            return False
        try:
            for it in items_for_llm:
                if _is_blocked_item(it):
                    continue
                if it.get("href_kind") == "nav":
                    href = (it.get("href") or "").strip()
                    if href:
                        h = urlparse(href).hostname or host
                        n = (it.get("name") or "").lower()
                        if h == host and any(k in n for k in ["admission", "apply", "application", "form", "program", "course"]):
                            fallback = it
                            break
            if not fallback:
                for it in items_for_llm:
                    if _is_blocked_item(it):
                        continue
                    if it.get("href_kind") == "nav":
                        href = (it.get("href") or "").strip()
                        if href:
                            h = urlparse(href).hostname or host
                            if h == host:
                                fallback = it
                                break
            if not fallback:
                for it in items_for_llm:
                    if _is_blocked_item(it):
                        continue
                    n = (it.get("name") or "").lower()
                    if any(k in n for k in ["apply", "application", "form", "learn more", "explore", "program", "course", "contact", "about", "next", "continue"]):
                        fallback = it
                        break
            if not fallback:
                for it in items_for_llm:
                    if _is_blocked_item(it):
                        continue
                    if it.get("href_kind") == "nav":
                        fallback = it
                        break
        except Exception:
            fallback = None
        if fallback:
            response = {
                "action": "click",
                "locator_type": fallback.get("locator_type") or "role",
                "role": fallback.get("role"),
                "name": fallback.get("name"),
                "locator": fallback.get("locator"),
                "confidence": 0.4,
            }
        else:
            response = {"action": "fail", "locator_type": "role", "role": None, "name": None, "confidence": 0.0}

    def _norm_name_for_match(s: Any) -> str:
        try:
            return re.sub(r"\s+", " ", str(s or "")).strip().lower()
        except Exception:
            return ""

    def _best_match_for_decision(dec: Dict[str, Any], candidates: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        try:
            lt = str(dec.get("locator_type") or "").lower().strip()
        except Exception:
            lt = ""
        try:
            loc = str(dec.get("locator") or "")
        except Exception:
            loc = ""
        if lt == "css" and loc:
            for it in candidates:
                try:
                    if it.get("locator_type") == "css" and str(it.get("locator") or "") == loc:
                        return it
                except Exception:
                    continue

        dec_role = str(dec.get("role") or "").lower().strip()
        dec_name = _norm_name_for_match(dec.get("name"))
        if dec_role and dec_name:
            for it in candidates:
                try:
                    if str(it.get("role") or "").lower().strip() != dec_role:
                        continue
                    if _norm_name_for_match(it.get("name")) == dec_name:
                        return it
                except Exception:
                    continue

        if not dec_name:
            return None

        toks = set([t for t in dec_name.split() if t])
        best = None
        best_score = -10**9
        for it in candidates:
            try:
                iname = _norm_name_for_match(it.get("name"))
            except Exception:
                iname = ""
            if not iname:
                continue
            score = 0
            if iname == dec_name:
                score += 200
            if dec_name and dec_name in iname:
                score += 80
            itoks = set([t for t in iname.split() if t])
            if toks:
                score += 6 * len(toks & itoks)
            try:
                if it.get("locator_type") == "css":
                    score += 4
            except Exception:
                pass
            if score > best_score:
                best_score = score
                best = it

        if best is None:
            return None
        if best_score < 6:
            return None
        return best

    try:
        sig = (perception or {}).get("success_signals") or {}
        response["generator_goal"] = bool(sig.get("generator_goal"))
        response["generation_in_progress"] = bool(sig.get("generation_in_progress"))
        response["success_likely"] = bool(sig.get("success_likely"))
        response["success_confirmed"] = bool(sig.get("success_confirmed"))
        response["judge_done"] = bool(sig.get("judge_done"))
        response["judge_confidence"] = float(sig.get("judge_confidence") or 0.0)
        response["judge_reason"] = str(sig.get("judge_reason") or "")
    except Exception:
        response["generator_goal"] = False
        response["generation_in_progress"] = False
        response["success_likely"] = False
        response["success_confirmed"] = False
        response["judge_done"] = False
        response["judge_confidence"] = 0.0
        response["judge_reason"] = ""
    
    response['dom_hash'] = dom_hash
    response['mode'] = mode

    try:
        if isinstance(response, dict) and response.get("action") in {"click", "type", "hover"}:
            cand = _best_match_for_decision(response, items_for_llm if isinstance(items_for_llm, list) else [])
            if cand is not None:
                response["locator_type"] = cand.get("locator_type") or response.get("locator_type")
                response["role"] = cand.get("role")
                response["name"] = cand.get("name")
                response["locator"] = cand.get("locator")
            else:
                response["action"] = "fail"
                response["confidence"] = 0.0
    except Exception:
        pass
    # Return a small candidate list to the caller for fallback exploration
    try:
        if dm == "gold":
            response['filtered_items'] = items_for_llm[:120]
        else:
            response['filtered_items'] = items_for_llm[:40]
    except Exception:
        response['filtered_items'] = []
    try:
        decision_log = {
            k: response.get(k)
            for k in ("action", "locator_type", "role", "name", "locator", "confidence", "dom_hash")
        }
        print(f"Decision: {decision_log}")
    except Exception:
        print("Decision logged.")
    
    log_thought(goal, filtered_summary, response)
    
    return response

def execute_action(page: Page, decision: Dict[str, Any], downloaded_files: Optional[List[str]] = None) -> bool:
    """Phase 3: Execute the decided action."""
    action = decision.get("action")
    locator_type = decision.get("locator_type") or "css"
    locator_str = decision.get("locator")
    value = decision.get("value")
    
    try:
        if action == "click":
            label_for_check = " ".join(
                [
                    str(locator_str or ""),
                    str(decision.get("name") or ""),
                ]
            ).lower()
            if any(x in label_for_check for x in ["buy", "place order", "pay", "checkout", "add to cart"]):
                print("Money action detected! Pausing for approval.")
                try:
                    page.screenshot(path="approval_needed.png")
                    print("Saved approval screenshot to approval_needed.png")
                except Exception:
                    pass
                approved = wait_for_money_approval(timeout_seconds=600)
                if not approved:
                    print("Money action not approved via UI; skipping click.")
                    return False
            role = None
            name = decision.get("name")
            if locator_type == "role":
                role = decision.get("role")
                elem = page.get_by_role(role, name=name)
            else:
                if not locator_str or not isinstance(locator_str, str):
                    if isinstance(decision.get("filtered_items"), list):
                        for it in decision.get("filtered_items"):
                            try:
                                if it.get("locator_type") == "css" and it.get("locator"):
                                    if name and isinstance(it.get("name"), str) and it.get("name").strip().lower() == str(name).strip().lower():
                                        locator_str = str(it.get("locator"))
                                        break
                            except Exception:
                                continue
                elem = page.locator(locator_str)

            if locator_type == "role":
                try:
                    if elem.count() == 0 and name and isinstance(decision.get("filtered_items"), list):
                        for it in decision.get("filtered_items"):
                            try:
                                if not isinstance(it, dict):
                                    continue
                                it_name = it.get("name")
                                if not isinstance(it_name, str):
                                    continue
                                if it_name.strip().lower() != str(name).strip().lower():
                                    continue
                                if it.get("locator_type") == "css" and it.get("locator"):
                                    locator_str = str(it.get("locator"))
                                    locator_type = "css"
                                    elem = page.locator(locator_str)
                                    break
                            except Exception:
                                continue
                except Exception:
                    pass
                try:
                    if elem.count() == 0 and role in {"link", "button"} and name:
                        alt_role = "button" if role == "link" else "link"
                        alt = page.get_by_role(alt_role, name=name)
                        if alt.count():
                            elem = alt
                except Exception:
                    pass

            try:
                if elem.count() > 1:
                    for i in range(min(int(elem.count() or 0), 6)):
                        cand = elem.nth(i)
                        ok = False
                        try:
                            ok = bool(cand.is_visible(timeout=800))
                        except Exception:
                            ok = False
                        if not ok:
                            continue
                        try:
                            bb = cand.bounding_box(timeout=800)
                        except Exception:
                            bb = None
                        if bb and float(bb.get("width") or 0) >= 6 and float(bb.get("height") or 0) >= 6:
                            elem = cand
                            break
            except Exception:
                pass

            # Highlight the chosen click target with a hologram-style overlay
            try:
                _highlight_click_target(page, elem.first if hasattr(elem, 'first') else elem)
            except Exception:
                pass

            try:
                if elem.count() > 0:
                    vis = False
                    try:
                        vis = bool(elem.is_visible(timeout=800))
                    except Exception:
                        vis = False
                    bb = None
                    try:
                        bb = elem.bounding_box(timeout=800)
                    except Exception:
                        bb = None
                    if (not vis) or (bb is not None and (float(bb.get("width") or 0) < 6 or float(bb.get("height") or 0) < 6)):
                        if name and isinstance(decision.get("filtered_items"), list):
                            want = re.sub(r"\s+", " ", str(name or "")).strip().lower()
                            best = None
                            best_score = -10**9
                            for it in decision.get("filtered_items"):
                                try:
                                    iname = re.sub(r"\s+", " ", str(it.get("name") or "")).strip().lower()
                                except Exception:
                                    iname = ""
                                if not iname:
                                    continue
                                score = 0
                                if iname == want:
                                    score += 200
                                if want and want in iname:
                                    score += 80
                                toks = set([t for t in want.split() if t])
                                itoks = set([t for t in iname.split() if t])
                                if toks:
                                    score += 6 * len(toks & itoks)
                                if it.get("locator_type") == "css":
                                    score += 4
                                if score > best_score:
                                    best_score = score
                                    best = it
                            if best and best_score >= 6:
                                try:
                                    if best.get("locator_type") == "css" and best.get("locator"):
                                        locator_str = str(best.get("locator"))
                                        locator_type = "css"
                                        elem = page.locator(locator_str)
                                    elif best.get("role") and best.get("name"):
                                        locator_type = "role"
                                        role = best.get("role")
                                        name = best.get("name")
                                        elem = page.get_by_role(role, name=name)
                                except Exception:
                                    pass
            except Exception:
                pass
            # If this element would open a new tab, navigate directly instead
            try:
                t = elem.get_attribute("target", timeout=1000)
            except Exception:
                t = None
            try:
                href = elem.get_attribute("href", timeout=1000)
            except Exception:
                href = None
            # Fallback to decision.filtered_items metadata
            if (t is None or href is None) and isinstance(decision.get("filtered_items"), list):
                try:
                    for it in decision.get("filtered_items"):
                        if it.get("role") == decision.get("role") and it.get("name") == decision.get("name"):
                            if t is None:
                                t = it.get("target")
                            if href is None:
                                href = it.get("href")
                            break
                except Exception:
                    pass
            
            def _should_expect_download(name_text: str, locator_text: str, href_text: Optional[str]) -> bool:
                txt = f"{name_text or ''} {locator_text or ''}".lower()
                if any(w in txt for w in ["download", "save", "export", "get file", "get image", "direct download"]):
                    return True
                h = (href_text or "").lower()
                if not h:
                    return False
                # Common direct-download signals
                if any(sig in h for sig in ["?download", "&download=", "download=1", "force=true", "download.php", "dl?"]):
                    return True
                if re.search(r"\.(jpg|jpeg|png|gif|webp|svg|zip|rar|7z|pdf|mp3|mp4|mov|avi|apk|exe|msi|dmg|iso|torrent)(\?|$)", h):
                    return True
                if re.search(r"(uploadhaven|mega|mediafire|gofile|anonfile|anonfiles|dropbox|drive\.google|pixeldrain)", h):
                    return True
                return False

            def _save_download_to_disk(dl) -> Optional[str]:
                try:
                    fname = getattr(dl, "suggested_filename", None) or "download_" + time.strftime("%Y%m%d_%H%M%S")
                    base, ext = os.path.splitext(fname)
                    out = os.path.join(DOWNLOAD_DIR, fname)
                    i = 1
                    while os.path.exists(out):
                        out = os.path.join(DOWNLOAD_DIR, f"{base} ({i}){ext}")
                        i += 1
                    dl.save_as(out)
                    print(f"Saved download to: {out}")
                    if isinstance(downloaded_files, list):
                        downloaded_files.append(out)
                    return out
                except Exception as _:
                    return None

            expect_dl = _should_expect_download(decision.get("name"), locator_str or "", href)
            # External domain detection
            current_host = ""
            try:
                current_host = urlparse(page.url).hostname or ""
            except Exception:
                current_host = ""
            href_host = ""
            try:
                if href:
                    href_host = urlparse(href).hostname or current_host
            except Exception:
                href_host = ""
            if (t and t.lower() == "_blank") and href and (href.strip().lower().startswith("http") or href.strip().startswith("/")):
                try:
                    print(f"Opening target=_blank href in same tab: {href}")
                    if expect_dl:
                        try:
                            with page.context.expect_download(timeout=10000) as dl_info:
                                page.goto(href, timeout=20000)
                            _save_download_to_disk(dl_info.value)
                        except Exception:
                            page.goto(href, timeout=20000)
                    else:
                        page.goto(href, timeout=20000)
                    return True
                except Exception:
                    pass
            # If the link goes to a different host, prefer direct navigation
            if href and current_host and href_host and href_host != current_host and (href.strip().lower().startswith("http") or href.strip().startswith("/")):
                try:
                    print(f"Opening external host in same tab: {href}")
                    if expect_dl:
                        try:
                            with page.context.expect_download(timeout=10000) as dl_info:
                                page.goto(href, timeout=20000)
                            _save_download_to_disk(dl_info.value)
                        except Exception:
                            page.goto(href, timeout=20000)
                    else:
                        page.goto(href, timeout=20000)
                    return True
                except Exception:
                    pass
            # If we still expect a popup, use expect_popup to ensure creation is captured
            expect_popup = bool((t and t.lower() == "_blank") or (href_host and current_host and href_host != current_host))
            try:
                if expect_popup:
                    try:
                        with page.expect_popup() as pop:
                            if expect_dl:
                                try:
                                    with page.context.expect_download(timeout=8000) as dl_info:
                                        elem.click(timeout=10000)
                                    _save_download_to_disk(dl_info.value)
                                except Exception:
                                    elem.click(timeout=10000)
                            else:
                                elem.click(timeout=10000)
                        _ = pop.value  # ensure popup resolved
                    except Exception:
                        # fallback to plain click if popup not raised
                        if expect_dl:
                            try:
                                with page.context.expect_download(timeout=8000) as dl_info:
                                    elem.click(timeout=10000)
                                _save_download_to_disk(dl_info.value)
                            except Exception:
                                elem.click(timeout=10000)
                        else:
                            elem.click(timeout=10000)
                else:
                    if expect_dl:
                        try:
                            with page.context.expect_download(timeout=10000) as dl_info:
                                elem.click(timeout=10000)
                            _save_download_to_disk(dl_info.value)
                        except Exception:
                            elem.click(timeout=10000)
                    else:
                        elem.click(timeout=10000)
                return True
            except Exception:
                pass
            try:
                elem.scroll_into_view_if_needed(timeout=5000)
                if expect_dl:
                    try:
                        with page.context.expect_download(timeout=10000) as dl_info:
                            elem.click(timeout=10000, force=True)
                        _save_download_to_disk(dl_info.value)
                    except Exception:
                        elem.click(timeout=10000, force=True)
                else:
                    elem.click(timeout=10000, force=True)
                return True
            except Exception:
                pass
            try:
                href = href if href is not None else None
                if href is None:
                    try:
                        href = elem.get_attribute("href", timeout=1000)
                    except Exception:
                        href = None
                if href and (href.strip().lower().startswith("http") or href.strip().startswith("/")):
                    print(f"Navigating directly to href: {href}")
                    if expect_dl:
                        try:
                            with page.context.expect_download(timeout=10000) as dl_info:
                                page.goto(href, timeout=20000)
                            _save_download_to_disk(dl_info.value)
                        except Exception:
                            page.goto(href, timeout=20000)
                    else:
                        page.goto(href, timeout=20000)
                    return True
            except Exception:
                pass
            try:
                elem.evaluate("el => el.click()")
                return True
            except Exception:
                pass
            return False
            
        elif action == "type":
            # Guard against placeholder junk values that might overwrite
            # real credentials (e.g., "your_email_here").
            if isinstance(value, str):
                lowv = value.lower().strip()
                placeholder_tokens = [
                    "your_email_here",
                    "your_password_here",
                    "example@example.com",
                ]
                if any(tok in lowv for tok in placeholder_tokens):
                    print(f"Skipping placeholder type value on field '{decision.get('name')}'")
                    return True

            if value is None:
                print("No value provided for type action.")
                return False
            if not isinstance(value, str):
                try:
                    value = str(value)
                except Exception:
                    value = ""

            if locator_type == "role":
                role = decision.get("role")
                name = decision.get("name")
                elem = page.get_by_role(role, name=name).first
            else:
                if not locator_str:
                    print("No locator provided for css type action.")
                    return False
                elem = page.locator(locator_str).first

            # Highlight the input element before typing so the user sees the focus target
            try:
                _highlight_click_target(page, elem)
            except Exception:
                pass

            with contextlib.suppress(Exception):
                elem.scroll_into_view_if_needed(timeout=5000)
            with contextlib.suppress(Exception):
                elem.click(timeout=8000, force=True)

            typed_ok = False
            try:
                elem.fill(value, timeout=10000, force=True)
                typed_ok = True
            except Exception:
                typed_ok = False
            if not typed_ok:
                with contextlib.suppress(Exception):
                    elem.click(timeout=8000, force=True)
                with contextlib.suppress(Exception):
                    page.keyboard.press("Control+A")
                with contextlib.suppress(Exception):
                    page.keyboard.press("Backspace")
                try:
                    page.keyboard.type(value, delay=10)
                    typed_ok = True
                except Exception:
                    typed_ok = False
            if not typed_ok:
                return False
            
            # Auto-submit logic (Press Enter)
            print("Auto-submitting with Enter...")
            page.keyboard.press("Enter")
            return True
            
        elif action == "hover":
            if locator_type == "role":
                role = decision.get("role")
                name = decision.get("name")
                elem = page.get_by_role(role, name=name).first
            else:
                elem = page.locator(locator_str).first
            elem.hover(timeout=10000)
            return True

        elif action == "wait":
            time.sleep(3)
            return True
            
        elif action == "done":
            return True 
            
        elif action == "fail":
            return False
            
    except Exception as e:
        print(f"Execution error: {e}")
        return False
    
    return False

# --- Main Runner ---

def _launch_hemlo_context(p):
    """Launch a browser context robustly.

    Preference order:
    1) Persistent context with USER_DATA_DIR.
    2) If that fails, rename the profile dir once and retry persistent.
    3) If it still fails, fall back to a non-persistent browser context.

    Returns (context, browser) where browser is None for persistent mode and
    a Browser instance for non-persistent mode.
    """
    from playwright.sync_api import Browser, BrowserContext  # type: ignore

    context: Optional[BrowserContext] = None
    browser: Optional[Browser] = None

    def _try_persistent() -> Optional[BrowserContext]:
        try:
            print(f"Launching persistent Chromium with user_data_dir={USER_DATA_DIR!r}...")
            ctx = p.chromium.launch_persistent_context(
                USER_DATA_DIR,
                headless=False,
                no_viewport=True,  # Let Chrome handle viewport natively
                accept_downloads=True,
                bypass_csp=True,
                args=[
                    "--start-fullscreen",
                    "--kiosk",
                    "--allow-running-insecure-content",
                    "--unsafely-treat-insecure-origin-as-secure=http://localhost:5000",
                    "--unsafely-treat-insecure-origin-as-secure=http://127.0.0.1:5000",
                    "--disable-features=BlockInsecurePrivateNetworkRequests,PrivateNetworkAccessRespectPreflightResults",
                    "--disable-blink-features=AutomationControlled",
                    "--disable-features=IsolateOrigins,site-per-process",
                ],
            )
            return ctx
        except Exception as e:
            print(f"Persistent context launch failed: {e}")
            return None

    # First attempt: existing persistent profile
    context = _try_persistent()
    if context:
        return context, browser

    # Second attempt: rename (likely corrupted) profile directory and retry once
    try:
        if os.path.exists(USER_DATA_DIR):
            ts = time.strftime("%Y%m%d_%H%M%S")
            backup = USER_DATA_DIR + f"_backup_{ts}"
            try:
                os.rename(USER_DATA_DIR, backup)
                print(f"Renamed existing browser profile dir to {backup}. A fresh profile will be created.")
            except Exception as e:
                print(f"Failed to rename browser profile dir {USER_DATA_DIR!r}: {e}")
    except Exception:
        pass

    context = _try_persistent()
    if context:
        return context, browser

    # Final fallback: non-persistent browser/context
    print("Falling back to non-persistent Chromium context for this run.")
    try:
        browser = p.chromium.launch(
            headless=False,
            args=[
                "--start-fullscreen",
                "--kiosk",
                "--allow-running-insecure-content",
                "--unsafely-treat-insecure-origin-as-secure=http://localhost:5000",
                "--unsafely-treat-insecure-origin-as-secure=http://127.0.0.1:5000",
                "--disable-features=BlockInsecurePrivateNetworkRequests,PrivateNetworkAccessRespectPreflightResults",
                "--disable-blink-features=AutomationControlled",
                "--disable-features=IsolateOrigins,site-per-process",
            ],
        )
        context = browser.new_context(
            no_viewport=True,  # Let Chrome handle viewport natively
            accept_downloads=True,
            bypass_csp=True,
        )
        return context, browser
    except Exception as e:
        print(f"Non-persistent Chromium launch also failed: {e}")
        raise


def _install_browser_overlay(page: Page) -> None:
    # Never inject the in-page overlay when running in "2nd Chrome" mode.
    try:
        if os.getenv("HEMLO_SECOND_CHROME", "0") == "1":
            return
    except Exception:
        return
    if os.getenv("HEMLO_ENABLE_INPAGE_OVERLAY", "0") != "1":
        return
    script = r"""
    (() => {
      try {
        if (window.__HEMLO_OVERLAY_INSTALLED__) return;
        window.__HEMLO_OVERLAY_INSTALLED__ = true;
      } catch (e) {}
      const hostCandidates = ['http://127.0.0.1:5000', 'http://localhost:5000'];
      let host = hostCandidates[0];

      const canUseBindings = () => {
        try {
          return typeof window.hemloOverlayGetLogs === 'function' && typeof window.hemloOverlayControl === 'function';
        } catch (e) {
          return false;
        }
      };
      function createOverlay() {
        try {
          if (!document || !document.body) return;
          if (document.getElementById('hemlo-overlay')) return;
          const root = document.createElement('div');
          root.id = 'hemlo-overlay';
          root.style.position = 'fixed';
          root.style.bottom = '8px';
          root.style.right = '8px';
          root.style.width = '380px';
          root.style.maxHeight = '240px';
          root.style.maxWidth = 'calc(100vw - 16px)';
          root.style.boxSizing = 'border-box';
          root.style.contain = 'layout paint style';
          root.style.zIndex = '2147483647';
          root.style.fontFamily = 'system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
          root.style.fontSize = '11px';
          root.style.background = 'rgba(0,0,0,0.82)';
          root.style.color = '#f5f5f5';
          root.style.borderRadius = '6px';
          root.style.boxShadow = '0 0 0 1px rgba(255,255,255,0.08), 0 8px 20px rgba(0,0,0,0.45)';
          root.style.overflow = 'hidden';
          root.style.userSelect = 'none';
          try {
            root.style.setProperty('position', 'fixed', 'important');
            root.style.setProperty('right', '8px', 'important');
            root.style.setProperty('bottom', '8px', 'important');
            root.style.setProperty('z-index', '2147483647', 'important');
            root.style.setProperty('pointer-events', 'auto', 'important');
          } catch (e) {}

          const header = document.createElement('div');
          header.style.display = 'flex';
          header.style.alignItems = 'center';
          header.style.justifyContent = 'space-between';
          header.style.padding = '4px 6px';
          header.style.background = 'rgba(255,255,255,0.06)';
          header.style.borderBottom = '1px solid rgba(255,255,255,0.10)';
          header.style.cursor = 'move';

          const title = document.createElement('span');
          title.textContent = 'Hemlo Agent';
          title.style.fontWeight = '600';
          title.style.letterSpacing = '0.02em';

          const btnRow = document.createElement('div');
          btnRow.style.display = 'flex';
          btnRow.style.gap = '4px';

          function makeBtn(label, command, bg) {
            const b = document.createElement('button');
            b.textContent = label;
            b.style.border = 'none';
            b.style.padding = '2px 6px';
            b.style.borderRadius = '4px';
            b.style.cursor = 'pointer';
            b.style.fontSize = '11px';
            b.style.background = bg;
            b.style.color = '#f5f5f5';
            b.style.opacity = '0.92';
            b.onmouseenter = () => { b.style.opacity = '1'; };
            b.onmouseleave = () => { b.style.opacity = '0.92'; };
            b.onmousedown = (ev) => {
              try { if (ev) { ev.stopPropagation(); } } catch (e) {}
            };
            b.onclick = (ev) => {
              try {
                try { if (ev) { ev.stopPropagation(); } } catch (e) {}
                if (canUseBindings()) {
                  try { window.hemloOverlayControl(command); } catch (e) {}
                } else {
                  fetch(host + '/overlay_control?command=' + encodeURIComponent(command), {
                    method: 'GET',
                    cache: 'no-store',
                  }).catch(() => {});
                }
              } catch (e) {}
            };
            return b;
          }

          btnRow.appendChild(makeBtn('Stop', 'stop', 'rgba(239,68,68,0.95)'));
          btnRow.appendChild(makeBtn('Pause', 'pause', 'rgba(245,158,11,0.95)'));
          btnRow.appendChild(makeBtn('Approve', 'approve', 'rgba(34,197,94,0.95)'));

          header.appendChild(title);
          header.appendChild(btnRow);

          const log = document.createElement('div');
          log.id = 'hemlo-overlay-log';
          log.style.fontFamily = 'SFMono-Regular, ui-monospace, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace';
          log.style.fontSize = '11px';
          log.style.whiteSpace = 'pre';
          log.style.padding = '4px 6px 6px 6px';
          log.style.maxHeight = '190px';
          log.style.overflowY = 'auto';
          log.style.overflowX = 'auto';
          log.style.lineHeight = '1.28';
          log.style.userSelect = 'text';
          log.textContent = 'Waiting for Hemlo logs...';

          root.appendChild(header);
          root.appendChild(log);
          document.body.appendChild(root);
          try {
            const se = document.scrollingElement || document.documentElement || document.body;
            if (se && typeof se.scrollLeft === 'number') se.scrollLeft = 0;
          } catch (e) {}

          // Restore last position if user dragged it.
          try {
            const saved = JSON.parse(localStorage.getItem('hemlo_overlay_pos') || 'null');
            if (saved && typeof saved === 'object') {
              if (typeof saved.left === 'number') { root.style.left = saved.left + 'px'; root.style.right = ''; }
              if (typeof saved.top === 'number') { root.style.top = saved.top + 'px'; root.style.bottom = ''; }
            }
          } catch (e) {}

          // Dragging
          try {
            let dragging = false;
            let startX = 0, startY = 0;
            let startLeft = 0, startTop = 0;

            const clamp = (v, min, max) => Math.max(min, Math.min(max, v));

            const onMove = (ev) => {
              if (!dragging) return;
              const x = (ev && ev.clientX) || 0;
              const y = (ev && ev.clientY) || 0;
              const dx = x - startX;
              const dy = y - startY;
              const rect = root.getBoundingClientRect();
              const w = rect.width || 380;
              const h = rect.height || 240;
              const left = clamp(startLeft + dx, 4, Math.max(4, window.innerWidth - w - 4));
              const top = clamp(startTop + dy, 4, Math.max(4, window.innerHeight - h - 4));
              root.style.left = left + 'px';
              root.style.top = top + 'px';
              root.style.right = '';
              root.style.bottom = '';
            };

            const onUp = () => {
              if (!dragging) return;
              dragging = false;
              document.removeEventListener('mousemove', onMove);
              document.removeEventListener('mouseup', onUp);
              try {
                const r = root.getBoundingClientRect();
                localStorage.setItem('hemlo_overlay_pos', JSON.stringify({ left: Math.round(r.left), top: Math.round(r.top) }));
              } catch (e) {}
            };

            header.addEventListener('mousedown', (ev) => {
              try {
                try {
                  if (ev && ev.button !== 0) return;
                  if (ev && ev.target && ev.target.closest && ev.target.closest('button')) return;
                } catch (e) {}
                dragging = true;
                startX = ev.clientX;
                startY = ev.clientY;
                const r = root.getBoundingClientRect();
                startLeft = r.left;
                startTop = r.top;
                document.addEventListener('mousemove', onMove);
                document.addEventListener('mouseup', onUp);
              } catch (e) { dragging = false; }
            });

            const clampNow = () => {
              try {
                const r = root.getBoundingClientRect();
                const w = r.width || 380;
                const h = r.height || 240;
                const left = clamp(r.left, 4, Math.max(4, window.innerWidth - w - 4));
                const top = clamp(r.top, 4, Math.max(4, window.innerHeight - h - 4));
                root.style.left = left + 'px';
                root.style.top = top + 'px';
                root.style.right = '';
                root.style.bottom = '';
                try { localStorage.setItem('hemlo_overlay_pos', JSON.stringify({ left: Math.round(left), top: Math.round(top) })); } catch (e) {}
                try {
                  const se = document.scrollingElement || document.documentElement || document.body;
                  if (se && typeof se.scrollLeft === 'number') se.scrollLeft = 0;
                } catch (e) {}
              } catch (e) {}
            };

            try { setTimeout(clampNow, 0); } catch (e) {}
            try { window.addEventListener('resize', clampNow); } catch (e) {}
          } catch (e) {}

          async function pickHost() {
            for (const h of hostCandidates) {
              try {
                const r = await fetch(h + '/overlay_logs?n=1', { cache: 'no-store' });
                if (r && r.ok) { host = h; return; }
              } catch (e) {}
            }
          }

          async function poll() {
            try {
              const normalizeLine = (s) => {
                try { s = String(s || ''); } catch (e) { s = ''; }
                try { s = s.replace(/\r/g, ''); } catch (e) {}
                try { s = s.replace(/\\n/g, ' '); } catch (e) {}
                try { s = s.replace(/\s+/g, ' ').trim(); } catch (e) {}
                if (!s) return '';
                if (s.startsWith('__HEMLO_STATUS__')) return '';
                if (s.length > 260) s = s.slice(0, 260) + '...';
                return s;
              };

              const nearBottom = (() => {
                try {
                  return (log.scrollTop + log.clientHeight) >= (log.scrollHeight - 40);
                } catch (e) {
                  return true;
                }
              })();

              if (canUseBindings()) {
                let data = null;
                try { data = await window.hemloOverlayGetLogs(8000); } catch (e) { data = null; }
                const lines = data && Array.isArray(data.logs) ? data.logs : [];
                const cleaned = lines.map(normalizeLine).filter(Boolean);
                const txt = cleaned.slice(-160).join('\\n');
                log.textContent = txt || '(no recent logs yet)';
                if (nearBottom) log.scrollTop = log.scrollHeight || 0;
              } else {
                await pickHost();
                const res = await fetch(host + '/overlay_logs?n=8000', { cache: 'no-store' });
                if (res && res.ok) {
                  const data = await res.json();
                  const lines = Array.isArray(data.logs) ? data.logs : [];
                  const cleaned = lines.map(normalizeLine).filter(Boolean);
                  const txt = cleaned.slice(-160).join('\\n');
                  log.textContent = txt || '(no recent logs yet)';
                  if (nearBottom) log.scrollTop = log.scrollHeight || 0;
                } else {
                  log.textContent = 'Waiting for Hemlo logs... (cannot reach overlay server)';
                }
              }
            } catch (e) {}
            try {
              setTimeout(poll, 1500);
            } catch (e) {}
          }
          poll();
        } catch (e) {}
      }
      if (document.readyState === 'complete' || document.readyState === 'interactive') {
        createOverlay();
      } else {
        window.addEventListener('DOMContentLoaded', createOverlay, { once: true });
      }
    })();
    """
    try:
        page.add_init_script(script)
    except Exception:
        pass
    try:
        page.evaluate(script)
    except Exception:
        pass


def _highlight_click_target(page: Page, elem) -> None:
    """Render a hologram-style overlay around the element the agent is about to click.

    Uses a dedicated #hemlo-click-highlight-layer so it can sit on top of any
    broader candidate highlighting.
    """
    try:
        if not elem:
            return
    except Exception:
        return
    try:
        elem.evaluate(
            """(el) => {
        try {
          if (!el || typeof el.getBoundingClientRect !== 'function') return;

          try {
            el.scrollIntoView({ block: 'center', inline: 'center', behavior: 'instant' });
          } catch (e) {
            try { el.scrollIntoView(true); } catch (_) {}
          }

          const doc = el.ownerDocument || document;

          const styleId = 'hemlo-click-highlight-style';
          if (!doc.getElementById(styleId)) {
            const style = doc.createElement('style');
            style.id = styleId;
            style.textContent = `
@keyframes hemloClickPulse {
  0%   { box-shadow: 0 0 0 0 rgba(96,165,250,0.95); opacity: 0.95; }
  50%  { box-shadow: 0 0 0 14px rgba(56,189,248,0.0); opacity: 1.0; }
  100% { box-shadow: 0 0 0 0 rgba(96,165,250,0.0); opacity: 0.88; }
}
.hemlo-click-holo {
  border-radius: 10px;
  border: 1px solid rgba(129,140,248,0.95);
  box-shadow:
    0 0 16px rgba(129,140,248,0.90),
    0 0 40px rgba(14,165,233,0.75);
  background:
    radial-gradient(circle at 15% 0%, rgba(56,189,248,0.45), transparent 55%),
    radial-gradient(circle at 85% 100%, rgba(129,140,248,0.35), transparent 60%),
    rgba(15,23,42,0.35);
  mix-blend-mode: screen;
  animation: hemloClickPulse 1.2s ease-out infinite;
}
`;
            (doc.head || doc.documentElement || doc.body || document.body).appendChild(style);
          }

          let root = doc.getElementById('hemlo-click-highlight-layer');
          if (!root) {
            root = doc.createElement('div');
            root.id = 'hemlo-click-highlight-layer';
            root.style.position = 'fixed';
            root.style.left = '0';
            root.style.top = '0';
            root.style.right = '0';
            root.style.bottom = '0';
            root.style.pointerEvents = 'none';
            root.style.zIndex = '2147483646';
            root.style.background = 'transparent';
            root.style.mixBlendMode = 'screen';
            (doc.body || doc.documentElement || document.body).appendChild(root);
          }

          try {
            while (root.firstChild) root.removeChild(root.firstChild);
          } catch (e) {}

          const r = el.getBoundingClientRect();
          const pad = 6;
          const box = doc.createElement('div');
          box.className = 'hemlo-click-holo';
          box.style.position = 'absolute';
          box.style.left = (r.left - pad) + 'px';
          box.style.top = (r.top - pad) + 'px';
          box.style.width = (Math.max(0, r.width) + pad * 2) + 'px';
          box.style.height = (Math.max(0, r.height) + pad * 2) + 'px';
          root.appendChild(box);
        } catch (e) {}
      }"""
        )
    except Exception:
        return


def _highlight_candidate_elements(page: Page, items: List[Dict[str, Any]], max_items: int = 40) -> None:
    """Highlight many candidate interactive elements on the page.

    Called right after we build the filtered interactive element list in
    analyze_dom_and_act. Renders a softer hologram around up to max_items
    candidates so you can see everything the agent is considering, even if it
    never clicks some of them.
    """
    try:
        if not page or not items:
            return
    except Exception:
        return

    rects: List[Dict[str, float]] = []
    seen: Set[str] = set()

    try:
        for it in items:
            if max_items and len(rects) >= int(max_items):
                break
            if not isinstance(it, dict):
                continue

            loc_type = str(it.get("locator_type") or "").strip().lower()
            locator = None
            try:
                if loc_type == "role" and it.get("role"):
                    locator = page.get_by_role(it.get("role"), name=it.get("name")).first
                elif loc_type == "css" and it.get("locator"):
                    locator = page.locator(str(it.get("locator"))).first
                else:
                    continue
            except Exception:
                locator = None
            if locator is None:
                continue

            # Require visibility and a reasonably sized bounding box so we
            # don't spam highlights on invisible/zero-sized nodes.
            try:
                vis = bool(locator.is_visible(timeout=600))
            except Exception:
                vis = False
            if not vis:
                continue
            try:
                bb = locator.bounding_box(timeout=600)
            except Exception:
                bb = None
            if not bb:
                continue
            try:
                w = float(bb.get("width") or 0)
                h = float(bb.get("height") or 0)
                x = float(bb.get("x") or 0)
                y = float(bb.get("y") or 0)
            except Exception:
                continue
            if w < 4 or h < 4:
                continue

            key = f"{int(x)}:{int(y)}:{int(w)}:{int(h)}"
            if key in seen:
                continue
            seen.add(key)
            rects.append({"left": x, "top": y, "width": w, "height": h})
    except Exception:
        rects = []

    if not rects:
        return

    # Render all rects in a single fixed overlay layer.
    try:
        page.evaluate(
            """(rects) => {
        try {
          if (!Array.isArray(rects) || !rects.length) return;
          const doc = document;

          const styleId = 'hemlo-candidate-highlight-style';
          if (!doc.getElementById(styleId)) {
            const style = doc.createElement('style');
            style.id = styleId;
            style.textContent = `
@keyframes hemloCandidatePulse {
  0%   { box-shadow: 0 0 0 0 rgba(56,189,248,0.65); opacity: 0.85; }
  50%  { box-shadow: 0 0 0 10px rgba(56,189,248,0.0); opacity: 1.0; }
  100% { box-shadow: 0 0 0 0 rgba(56,189,248,0.0); opacity: 0.80; }
}
.hemlo-candidate-holo {
  border-radius: 8px;
  border: 1px solid rgba(56,189,248,0.85);
  box-shadow:
    0 0 10px rgba(56,189,248,0.75),
    0 0 24px rgba(59,130,246,0.55);
  background:
    radial-gradient(circle at 10% 0%, rgba(56,189,248,0.25), transparent 55%),
    radial-gradient(circle at 90% 100%, rgba(37,99,235,0.25), transparent 60%),
    rgba(15,23,42,0.22);
  mix-blend-mode: screen;
  animation: hemloCandidatePulse 1.4s ease-out infinite;
}
`;
            (doc.head || doc.documentElement || doc.body || document.body).appendChild(style);
          }

          let root = doc.getElementById('hemlo-candidate-highlight-layer');
          if (!root) {
            root = doc.createElement('div');
            root.id = 'hemlo-candidate-highlight-layer';
            root.style.position = 'fixed';
            root.style.left = '0';
            root.style.top = '0';
            root.style.right = '0';
            root.style.bottom = '0';
            root.style.pointerEvents = 'none';
            root.style.zIndex = '2147483645';
            root.style.background = 'transparent';
            root.style.mixBlendMode = 'screen';
            (doc.body || doc.documentElement || document.body).appendChild(root);
          }

          try {
            while (root.firstChild) root.removeChild(root.firstChild);
          } catch (e) {}

          for (const r of rects) {
            if (!r) continue;
            const w = Number(r.width || 0);
            const h = Number(r.height || 0);
            if (!w || !h || w < 4 || h < 4) continue;
            const pad = 4;
            const box = doc.createElement('div');
            box.className = 'hemlo-candidate-holo';
            box.style.position = 'absolute';
            box.style.left = (Number(r.left || 0) - pad) + 'px';
            box.style.top = (Number(r.top || 0) - pad) + 'px';
            box.style.width = (Math.max(0, w) + pad * 2) + 'px';
            box.style.height = (Math.max(0, h) + pad * 2) + 'px';
            root.appendChild(box);
          }
        } catch (e) {}
      }""",
            rects,
        )
    except Exception:
        return


def _read_control_command(control_nonce: int) -> typing.Tuple[int, Optional[Dict[str, Any]]]:
    """Read agent_control.json and only return commands with a higher nonce.

    This prevents re-processing old commands.
    """
    try:
        if not os.path.exists(CONTROL_PATH):
            return control_nonce, None
        with open(CONTROL_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return control_nonce, None
        nonce = int(data.get("nonce") or 0)
        if nonce <= int(control_nonce or 0):
            return control_nonce, None
        return nonce, data
    except Exception:
        return control_nonce, None


def _try_maximize_window(page: Page) -> None:
    """Use Chrome DevTools Protocol (CDP) to maximize the window.
    
    Enabled by default since --start-maximized is unreliable on Windows.
    Set HEMLO_ENABLE_CDP_MAXIMIZE=0 to disable if needed.
    """
    try:
        if os.getenv("HEMLO_ENABLE_CDP_MAXIMIZE", "0") != "1":
            return
    except Exception:
        return
    try:
        session = page.context.new_cdp_session(page)
    except Exception:
        return
    try:
        info = session.send("Browser.getWindowForTarget")
        wid = info.get("windowId")
        if wid is None:
            return
        session.send("Browser.setWindowBounds", {"windowId": wid, "bounds": {"windowState": "maximized"}})
    except Exception:
        pass


def _sync_viewport_to_screen(page: Page) -> None:
    """Force the page viewport to track the actual window/screen size.

    Empirically, on some Windows setups the Chromium window can be maximized
    yet the page viewport stays at the original size (leading to a blank band
    on the right). We read window inner dimensions and screen dimensions; if
    the screen is wider/taller than the current inner size by a meaningful
    margin, we set the viewport to the screen size to fill the window.
    """
    try:
        dims = page.evaluate(
            """
            () => {
              try {
                const sw = (window.screen && (window.screen.availWidth || window.screen.width)) || 0;
                const sh = (window.screen && (window.screen.availHeight || window.screen.height)) || 0;
                const iw = window.innerWidth || (document.documentElement && document.documentElement.clientWidth) || 0;
                const ih = window.innerHeight || (document.documentElement && document.documentElement.clientHeight) || 0;
                return { innerW: iw, innerH: ih, screenW: sw, screenH: sh };
              } catch (e) {
                return { innerW: 0, innerH: 0, screenW: 0, screenH: 0 };
              }
            }
            """
        )
    except Exception:
        return

    try:
        iw = int(dims.get("innerW") or 0)
        ih = int(dims.get("innerH") or 0)
        sw = int(dims.get("screenW") or 0)
        sh = int(dims.get("screenH") or 0)
    except Exception:
        return

    target_w = iw
    target_h = ih
    # If the screen is significantly larger than the current inner size, bump to screen size
    if sw and sw - iw > 40:
        target_w = sw
    if sh and sh - ih > 40:
        target_h = sh

    if target_w and target_h:
        try:
            page.set_viewport_size({"width": int(target_w), "height": int(target_h)})
        except Exception:
            pass


def _should_sync_viewport() -> bool:
    try:
        return os.getenv("HEMLO_SYNC_VIEWPORT", "1") == "1"
    except Exception:
        return True


def _minimize_window(page: Page) -> None:
    """Minimize the Chromium window via CDP so it sits on the taskbar until clicked."""
    try:
        if os.getenv("HEMLO_MINIMIZE_ON_LAUNCH", "1") == "0":
            return
    except Exception:
        return
    try:
        session = page.context.new_cdp_session(page)
    except Exception:
        return
    try:
        info = session.send("Browser.getWindowForTarget")
        wid = info.get("windowId")
        if wid is None:
            return
        session.send("Browser.setWindowBounds", {"windowId": wid, "bounds": {"windowState": "minimized"}})
    except Exception:
        pass


def _should_minimize_on_launch() -> bool:
    try:
        return os.getenv("HEMLO_MINIMIZE_ON_LAUNCH", "0") == "1"
    except Exception:
        return False


def _emit_status(status: str, message: Optional[str] = None, **extra: Any) -> None:
    global OVERLAY_AGENT_STATE
    try:
        OVERLAY_AGENT_STATE = str(status or "")
    except Exception:
        OVERLAY_AGENT_STATE = ""
    payload: Dict[str, Any] = {"status": status}
    if message is not None:
        payload["message"] = message
    try:
        for k, v in (extra or {}).items():
            payload[k] = v
    except Exception:
        pass
    try:
        print(STATUS_PREFIX + json.dumps(payload, ensure_ascii=False))
    except Exception:
        try:
            print(STATUS_PREFIX + json.dumps({"status": status}))
        except Exception:
            pass
    try:
        sys.stdout.flush()
    except Exception:
        pass


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True, help="Task description")
    parser.add_argument("--session", action="store_true")
    parser.add_argument("--dom_mode", type=str, default="legacy")
    parser.add_argument("--max_items", type=int, default=40)
    args = parser.parse_args()

    print(f"Agent started with goal: '{args.prompt}'")

    try:
        base_dir = os.path.dirname(__file__)
        for fn in ("last_step_trace.json", "last_filtered_dom.json", "last_run_memory.json"):
            with open(os.path.join(base_dir, fn), "w", encoding="utf-8") as f:
                json.dump([], f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    session_mode = bool(getattr(args, "session", False))
    control_nonce = 0
    dom_mode = str(getattr(args, "dom_mode", "legacy") or "legacy")
    try:
        max_items = int(getattr(args, "max_items", 40) or 40)
    except Exception:
        max_items = 40

    # Log core DOM configuration so Prompt UI settings are visible in the agent log
    try:
        print(f"DOM config: dom_mode={dom_mode!r}, max_items={max_items}")
    except Exception:
        pass

    # 1. Plan URL
    target_url = plan_initial_url(args.prompt)

    # 1.5 Initialize workflow memory for this site
    site = ""
    workflow_mem: Optional[WorkflowMemory] = None
    cached_workflow: Optional[Dict[str, Any]] = None
    try:
        site = urlparse(target_url).hostname or ""
    except Exception:
        site = ""
    if config.WORKFLOW_MEMORY_ENABLED and config.REDIS_URL and site:
        try:
            workflow_mem = WorkflowMemory(config.REDIS_URL)
            if workflow_mem.enabled:
                cached_workflow = workflow_mem.lookup(site, args.prompt)
                if cached_workflow:
                    steps = cached_workflow.get("steps") or []
                    sim = cached_workflow.get("similarity")
                    if sim is not None:
                        print(f"Workflow cache hit for site '{site}' (similarity={sim:.2f}). Replaying {len(steps)} steps if possible...")
                    else:
                        print(f"Workflow cache hit for site '{site}'. Replaying {len(steps)} steps if possible...")
                else:
                    print(f"Workflow cache miss for site '{site}'. Will learn a new workflow if this run succeeds.")
            else:
                workflow_mem = None
        except Exception as e:
            print(f"Workflow memory disabled for this run: {e}")
            workflow_mem = None

    # Detect "2nd Chrome" debug mode: launch a minimal, non-persistent browser
    # with as few custom flags as possible so we can compare layout/OS behavior
    # against the normal Hemlo-launch path.
    second_chrome = False
    try:
        second_chrome = os.getenv("HEMLO_SECOND_CHROME", "0") == "1"
    except Exception:
        second_chrome = False

    with sync_playwright() as p:
        context = None
        browser = None
        if second_chrome:
            print("HEMLO_SECOND_CHROME=1 detected. Launching minimal non-persistent Chromium for this run.")
            try:
                # Kiosk mode for true fullscreen
                browser = p.chromium.launch(headless=False, args=["--start-fullscreen", "--kiosk"])
                context = browser.new_context(
                    no_viewport=True,
                    accept_downloads=True,
                    bypass_csp=True,
                )
            except Exception as e:
                print(f"2nd Chrome launch failed, falling back to normal Hemlo context: {e}")
                browser = None
                context = None

        if not context:
            # Robust launch: prefer persistent profile, auto-reset if corrupted,
            # and fall back to non-persistent if needed.
            try:
                context, browser = _launch_hemlo_context(p)
            except Exception:
                # If everything fails, abort early with a clear message.
                print("Fatal error: unable to launch any Chromium context. Aborting run.")
                return

        try:
            def _overlay_get_logs(source=None, n=None):
                try:
                    nn = int(n or 8000)
                except Exception:
                    nn = 8000
                if nn < 50:
                    nn = 50
                if nn > 8000:
                    nn = 8000
                try:
                    with OVERLAY_AGENT_LOG_LOCK:
                        logs = list(OVERLAY_AGENT_LOG_BUFFER[-nn:])
                except Exception:
                    logs = []
                try:
                    st = str(OVERLAY_AGENT_STATE or "")
                except Exception:
                    st = ""
                return {"logs": logs, "state": st}

            def _write_control(payload: Dict[str, Any]) -> None:
                pld = dict(payload or {})
                try:
                    pld["nonce"] = int(time.time() * 1000)
                except Exception:
                    pld["nonce"] = int(time.time())
                tmp = CONTROL_PATH + ".tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    f.write(json.dumps(pld, ensure_ascii=False))
                os.replace(tmp, CONTROL_PATH)

            def _overlay_control(source=None, command=None):
                cmd = ""
                try:
                    cmd = str(command or "").strip().lower()
                except Exception:
                    cmd = ""
                if cmd in {"pause", "resume", "new_task", "stop"}:
                    try:
                        c = "resume" if cmd in {"resume", "new_task"} else cmd
                        _write_control({"command": c})
                    except Exception:
                        pass
                elif cmd in {"approve", "approve_money"}:
                    try:
                        with open(APPROVAL_FLAG_PATH, "w", encoding="utf-8") as f:
                            f.write("1")
                    except Exception:
                        pass
                return {"ok": True}

            try:
                context.expose_binding("hemloOverlayGetLogs", _overlay_get_logs)
            except Exception:
                pass
            try:
                context.expose_binding("hemloOverlayControl", _overlay_control)
            except Exception:
                pass
        except Exception:
            pass

        try:
            def _on_new_page(p2):
                minimize = _should_minimize_on_launch()
                try:
                    _install_browser_overlay(p2)
                except Exception:
                    pass
                if minimize:
                    # Ensure window is maximized first so restore comes back full screen
                    try:
                        _try_maximize_window(p2)
                    except Exception:
                        pass
                    # Disabled viewport sync - using fixed viewport from context creation
                    # try:
                    #     _sync_viewport_to_screen(p2)
                    # except Exception:
                    #     pass
                    try:
                        _minimize_window(p2)
                    except Exception:
                        pass
                else:
                    try:
                        _try_maximize_window(p2)
                    except Exception:
                        pass
                    # Disabled viewport sync - using fixed viewport from context creation
                    # try:
                    #     _sync_viewport_to_screen(p2)
                    # except Exception:
                    #     pass
                    if os.getenv("HEMLO_FORCE_FRONT", "0") == "1":
                        try:
                            p2.bring_to_front()
                        except Exception:
                            pass
                        try:
                            p2.evaluate("() => { try { window.focus(); } catch (e) {} }")
                        except Exception:
                            pass

            context.on("page", _on_new_page)
        except Exception:
            pass

        page = context.new_page()
        minimize = _should_minimize_on_launch()
        try:
            _install_browser_overlay(page)
        except Exception:
            pass
        
        # Kiosk mode handles fullscreen - no viewport manipulation needed
        if minimize:
            try:
                _minimize_window(page)
            except Exception:
                pass
        else:
            if _should_force_front():
                try:
                    page.bring_to_front()
                except Exception:
                    pass

        print(f"Navigating to {target_url}...")
        try:
            page.goto(target_url, timeout=60000)
            page.wait_for_load_state("domcontentloaded")
            # Kiosk mode handles fullscreen natively
            if not minimize and _should_force_front():
                try:
                    page.bring_to_front()
                except Exception:
                    pass
                try:
                    page.evaluate("() => { try { window.focus(); } catch (e) {} }")
                except Exception:
                    pass
        except Exception as e:
            print(f"Navigation failed: {e}")
            return

        downloaded_files: List[str] = []

        planner_assist: Optional[Dict[str, Any]] = None
        planner_mode = str(getattr(config, "PLANNER_MODE", "basic") or "basic").strip().lower()
        # Log planner configuration and inputs at the start of the run
        try:
            page_title = _safe_page_title(page)
        except Exception:
            page_title = ""
        try:
            gemini_enabled = bool(config.GEMINI_PLANNER_ASSIST)
        except Exception:
            gemini_enabled = False
        try:
            gemini_model = getattr(config, "GEMINI_MODEL", None) or "gemini-1.5-flash"
        except Exception:
            gemini_model = "gemini-1.5-flash"
        try:
            print(f"Planner mode: {planner_mode} | Gemini enabled: {gemini_enabled} | model: {gemini_model}")
            print(f"Planner input goal: {args.prompt}")
            print(f"Planner input URL: {page.url}")
            print(f"Planner input page title: {page_title}")
        except Exception:
            pass
        if planner_mode == "basic":
            try:
                planner_assist = get_planner_assist(args.prompt, page.url, page_title)
            except Exception:
                planner_assist = None
            try:
                if isinstance(planner_assist, dict) and planner_assist:
                    print("Planner-assist plan received (basic mode).")
                    try:
                        site = planner_assist.get("site") or ""
                    except Exception:
                        site = ""
                    if site:
                        print(f"Planner-assist site: {site}")
                else:
                    print("Planner-assist plan NOT received; proceeding without planner map.")
            except Exception:
                pass
        elif planner_mode == "gold":
            print("Planner Gold enabled: will fetch iterative hints during the run.")
        else:
            print("Planner mode OFF: skipping planner assist.")

        # 2. Fast path: try workflow replay if we have a cached recipe
        if workflow_mem and workflow_mem.enabled and cached_workflow:
            try:
                print("Attempting workflow replay...")
                replay_ok = workflow_mem.replay(page, cached_workflow, execute_action, downloaded_files)
            except Exception as e:
                replay_ok = False
                print(f"Workflow replay error, falling back to live planning: {e}")
            if replay_ok:
                print("Workflow replay succeeded. Skipping live planning loop.")
                run_succeeded = True
                if session_mode:
                    # In Prompt UI session mode, report completion but keep
                    # the browser window open for inspection.
                    _emit_status("completed")
                    return
                else:
                    # CLI mode: preserve old behaviour of waiting for Enter
                    # and then closing the browser.
                    input("Press Enter to close browser...")
                    try:
                        context.close()
                    except Exception:
                        pass
                    if browser is not None:
                        try:
                            browser.close()
                        except Exception:
                            pass
                    return
            else:
                print("Workflow replay failed or incomplete. Falling back to live planning and will refresh workflow.")

        # 3. ReAct Loop (live planning) with workflow recording
        step_count = 0
        max_steps = 20
        current_workflow_steps: List[Dict[str, Any]] = []
        # If the user intent is a download, be more decisive and stop sooner
        _prompt_l = args.prompt.lower()
        download_intent = bool(re.search(r"\b(download|save|get)\b", _prompt_l))
        if download_intent:
            max_steps = 12
        recent_actions: List[str] = []
        last_url = page.url
        last_dom_hash = ""
        searched_once = False
        upload_done = False
        run_memory: Dict[str, Any] = {
            "milestones": [],
            "recent_steps": [],
            "progress_notes": [],
            "choice_counts": {},
            "avoid_choices": [],
            "repeat_streak": 0,
            "last_choice": None,
        }
        planner_gold_state: Dict[str, Any] = {
            "last_call_step": 0,
            "last_url": "",
            "last_mode": "",
            "last_dom_hash": "",
        }
        blocked_choices: Set[Tuple[Any, ...]] = set()
        last_choice: Optional[tuple] = None
        repeat_choice_streak = 0
        run_succeeded = False
        last_login_prompt_url = ""
        generator_success_streak = 0
        
        while step_count < max_steps:
            step_count += 1
            print(f"\n--- Step {step_count} ---")
            if session_mode:
                control_nonce, cmd = _read_control_command(control_nonce)
                if cmd:
                    c = str(cmd.get("command") or "")
                    if c == "stop":
                        _emit_status("stopped")
                        try:
                            context.close()
                        except Exception:
                            pass
                        return
                    if c == "pause":
                        _emit_status("paused")
                        while True:
                            time.sleep(0.25)
                            control_nonce, cmd2 = _read_control_command(control_nonce)
                            if not cmd2:
                                continue
                            c2 = str(cmd2.get("command") or "")
                            if c2 == "stop":
                                _emit_status("stopped")
                                try:
                                    context.close()
                                except Exception:
                                    pass
                                return
                            if c2 in {"resume", "new_task"}:
                                p2 = cmd2.get("prompt")
                                if isinstance(p2, str) and p2.strip():
                                    args.prompt = p2.strip()
                                dm2 = cmd2.get("dom_mode")
                                if isinstance(dm2, str) and dm2.strip():
                                    dom_mode = dm2.strip()
                                mi2 = cmd2.get("max_items")
                                if mi2 is not None:
                                    try:
                                        max_items = int(mi2)
                                    except Exception:
                                        pass
                                try:
                                    print(f"Session config updated via control (resume/new_task): dom_mode={dom_mode!r}, max_items={max_items}")
                                except Exception:
                                    pass
                                _emit_status("running")
                                break
                    elif c in {"new_task", "resume"}:
                        p2 = cmd.get("prompt")
                        if isinstance(p2, str) and p2.strip():
                            args.prompt = p2.strip()
                        dm2 = cmd.get("dom_mode")
                        if isinstance(dm2, str) and dm2.strip():
                            dom_mode = dm2.strip()
                        mi2 = cmd.get("max_items")
                        if mi2 is not None:
                            try:
                                max_items = int(mi2)
                            except Exception:
                                pass
                        try:
                            print(f"Session config updated via control ({c}): dom_mode={dom_mode!r}, max_items={max_items}")
                        except Exception:
                            pass
                        _emit_status("running")
            current_url = page.url

            # In-run milestone: once Canva navigates into the editor (/design/.../edit),
            # consider the first upload completed so we don't keep selecting upload again.
            try:
                if (not upload_done) and ("canva.com/design/" in (current_url or "")) and ("/edit" in (current_url or "")):
                    upload_done = True
                    recent_actions.append("milestone:upload_done")
                    _run_memory_add_milestone(run_memory, "upload_done")
                    print("Milestone reached: upload_done=true (entered Canva editor)")
            except Exception:
                pass

            try:
                gate_action = try_handle_gate(page, args.prompt, downloaded_files, planner_assist, dom_mode=dom_mode, max_items=max_items)
            except Exception:
                gate_action = None
            if gate_action:
                recent_actions.append(gate_action)
                try:
                    page.wait_for_load_state("domcontentloaded", timeout=5000)
                except Exception:
                    pass
                try:
                    last_url = page.url
                    last_dom_hash = ""
                    last_choice = None
                    repeat_choice_streak = 0
                except Exception:
                    pass
                time.sleep(1)
                continue

            # Quick path: if the task is to download/save, try an obvious download control now
            if download_intent:
                try:
                    if quick_download_if_present(page, downloaded_files):
                        try:
                            fnames = ", ".join([os.path.basename(f) for f in downloaded_files])
                        except Exception:
                            fnames = str(len(downloaded_files))
                        print(f"Task complete: downloaded files: {fnames}")
                        run_succeeded = True
                        break
                except Exception:
                    pass

            # Heuristic pre-search for YouTube-like flows
            if "youtube.com" in current_url:
                query = extract_search_query(args.prompt)
                if has_search_box(page):
                    if not searched_once:
                        if perform_search(page, query):
                            recent_actions.append(f"typed_search:{query}")
                            searched_once = True
                            page.wait_for_timeout(1500)
                            # Save filtered DOM for this step as well
                            try:
                                _fs = playwright_filter_interactive_elements(page, args.prompt, planner_assist)
                                save_filtered_dom(args.prompt, _fs)
                            except Exception:
                                pass
                            last_url = page.url
                            continue
                    else:
                        # Already searched: click the first video result
                        try:
                            result = page.locator("ytd-video-renderer a#video-title").first
                            if result and result.count():
                                result.click()
                                page.wait_for_timeout(2000)
                                recent_actions.append("click:first_video")
                                try:
                                    _fs = playwright_filter_interactive_elements(page, args.prompt, planner_assist)
                                    save_filtered_dom(args.prompt, _fs)
                                except Exception:
                                    pass
                                last_url = page.url
                                continue
                        except Exception:
                            pass

            # Login choice flow: only fire when the page really looks like
            # an auth screen (login/signup/register), not just a homepage
            # with a random "Get it on Google Play" badge.
            try:
                if is_probable_login_or_signup_page(page):
                    login_opts = detect_login_options(page)
                else:
                    login_opts = []
            except Exception:
                login_opts = []
            if login_opts:
                cur_url = ""
                try:
                    cur_url = page.url
                except Exception:
                    cur_url = ""
                if cur_url != last_login_prompt_url:
                    emit_login_options(login_opts, cur_url)
                    choice = wait_for_login_choice(timeout_seconds=600)
                    last_login_prompt_url = cur_url
                    if choice:
                        # Capture pages before applying the choice so we can
                        # detect any newly opened login popups/tabs (e.g. Google).
                        try:
                            pages_before_login = context.pages[:]
                        except Exception:
                            pages_before_login = []

                        applied = perform_login_choice(page, choice)
                        print(f"Login choice applied: {'success' if applied else 'failed'} for provider={choice.get('provider')}")
                        try:
                            page.wait_for_load_state("domcontentloaded", timeout=8000)
                        except Exception:
                            pass

                        # If the login click opened a new tab/window (such as
                        # accounts.google.com), prefer switching control to that
                        # page so the agent can continue the auth flow there.
                        try:
                            pages_after_login = context.pages[:]
                        except Exception:
                            pages_after_login = []
                        new_login_pages = [p for p in pages_after_login if p not in pages_before_login]
                        if new_login_pages:
                            for np in new_login_pages:
                                try:
                                    np.wait_for_load_state("domcontentloaded", timeout=7000)
                                except Exception:
                                    pass
                            login_candidate = choose_best_page(new_login_pages, args.prompt) or choose_best_page(pages_after_login, args.prompt)
                            if login_candidate and login_candidate is not page:
                                try:
                                    old_url = ""
                                    try:
                                        old_url = page.url
                                    except Exception:
                                        old_url = ""
                                    new_url = ""
                                    try:
                                        new_url = login_candidate.url
                                    except Exception:
                                        new_url = ""
                                    print(f"Login flow opened new page. Switching page: {old_url} -> {new_url}")
                                except Exception:
                                    pass
                                page = login_candidate
                                try:
                                    page.bring_to_front()
                                except Exception:
                                    pass

                        # Reset DOM hash to avoid loop detection on same page
                        try:
                            last_dom_hash = ""
                            last_url = page.url
                            repeat_choice_streak = 0
                        except Exception:
                            pass
                        continue
                    else:
                        print("No login choice received; continuing without login.")
                        # Avoid spamming the same URL; mark as emitted
                        continue

            # Analyze & Decide
            decision = analyze_dom_and_act(
                page,
                args.prompt,
                current_url,
                recent_actions,
                blocked_choices,
                downloaded_files,
                planner_assist,
                dom_mode=dom_mode,
                max_items=max_items,
                upload_done=upload_done,
                run_memory=run_memory,
            )

            try:
                append_step_trace(
                    {
                        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "step": step_count,
                        "stage": "decision",
                        "goal": args.prompt,
                        "url": current_url,
                        "dom_hash": decision.get("dom_hash"),
                        "decision": {k: decision.get(k) for k in ("action", "locator_type", "role", "name", "locator", "confidence")},
                        "mode": decision.get("mode"),
                    }
                )
            except Exception:
                pass

            try:
                if decision.get("generator_goal") and decision.get("success_confirmed") and (not decision.get("generation_in_progress")):
                    generator_success_streak += 1
                else:
                    generator_success_streak = 0
            except Exception:
                generator_success_streak = 0
            if generator_success_streak >= 2:
                try:
                    jc = decision.get("judge_confidence")
                    jr = decision.get("judge_reason")
                    print(f"Success detected (stable + judge-confirmed). conf={jc} reason={jr}")
                except Exception:
                    print("Success detected (stable + judge-confirmed). Marking done.")
                run_succeeded = True
                break
            
            if decision.get("action") == "done":
                print("Goal achieved! (Agent decided 'done')")
                run_succeeded = True
                break
                
            if decision.get("action") == "fail":
                print("Agent failed to decide next step. Retrying once...")
                time.sleep(2)
                continue

            # Execute
            pages_before = context.pages[:]
            success = False
            # Skip repeated 'YouTube Home' clicks when URL doesn't change
            name_lower = str(decision.get("name") or "").lower()
            if name_lower == "youtube home" and current_url == last_url:
                print("Skipping repeated 'YouTube Home' click; trying search heuristic instead.")
                query = extract_search_query(args.prompt)
                if perform_search(page, query):
                    recent_actions.append(f"typed_search:{query}")
                    page.wait_for_timeout(1500)
                    last_url = page.url
                    continue

            # Detect loop: same choice and no progress
            choice = None
            try:
                if decision.get("locator_type") == "css" and decision.get("locator"):
                    choice = ("css", str(decision.get("locator")))
                else:
                    choice = (decision.get("role"), decision.get("name"))
            except Exception:
                choice = None
            # Treat upload-style clicks as progress even if URL/DOM hash don't change,
            # since they usually open a native file chooser dialog.
            upload_like = False
            try:
                label = " ".join(
                    [
                        name_lower,
                        str(decision.get("locator") or "").lower(),
                    ]
                )
                upload_like = _is_upload_like_label(label)
            except Exception:
                upload_like = False

            no_progress = (
                current_url == last_url
                and (decision.get("dom_hash") == last_dom_hash)
                and not upload_like
            )
            same_choice_same_url = bool(choice and last_choice == choice and (current_url == last_url))
            tentative_streak = (repeat_choice_streak + 1) if same_choice_same_url else 0
            if choice and same_choice_same_url and (no_progress or tentative_streak >= 2):
                print(f"Loop detected on {choice}; blocking and exploring alternative.")
                blocked_choices.add(choice)
                repeat_choice_streak = 0
                # Fallback exploration: pick next best from filtered_items
                candidates = decision.get("filtered_items") or []
                fallback = None

                def _candidate_key(it: Dict[str, Any]) -> tuple:
                    try:
                        if it.get("locator_type") == "css" and it.get("locator"):
                            return ("css", str(it.get("locator")))
                    except Exception:
                        pass
                    return (it.get("role"), it.get("name"))
                # Prefer menu items / action items when a menu is likely open
                for it in candidates:
                    key = _candidate_key(it)
                    if key in blocked_choices:
                        continue
                    r = str(it.get("role") or "").lower()
                    n = str(it.get("name") or "").lower()
                    if r in {"menuitem", "option"} or any(x in n for x in ["event", "task", "appointment", "schedule", "meeting"]):
                        fallback = it
                        break
                # Prefer nav links
                for it in candidates:
                    key = _candidate_key(it)
                    if key in blocked_choices:
                        continue
                    if it.get("href_kind") == "nav":
                        fallback = it
                        break
                # Or exploration keywords
                if not fallback:
                    for it in candidates:
                        key = _candidate_key(it)
                        if key in blocked_choices:
                            continue
                        n = (it.get("name") or "").lower()
                        if any(k in n for k in ["apply", "application", "form", "learn more", "explore", "program", "course", "contact", "about", "next", "continue"]):
                            fallback = it
                            break
                # Otherwise first unblocked
                if not fallback:
                    for it in candidates:
                        key = _candidate_key(it)
                        if key not in blocked_choices:
                            fallback = it
                            break
                if fallback:
                    fallback_decision = {
                        "action": "click",
                        "locator_type": fallback.get("locator_type") or "role",
                        "role": fallback.get("role"),
                        "name": fallback.get("name"),
                        "locator": fallback.get("locator"),
                        "confidence": 0.4,
                    }
                    success = execute_action(page, fallback_decision, downloaded_files)
                    act_label = f"fallback_click:{fallback.get('role')}:{fallback.get('name')}"
                    recent_actions.append(act_label)
                    # Update trackers
                    try:
                        page.wait_for_load_state("domcontentloaded", timeout=5000)
                    except Exception:
                        pass
                    last_url = page.url
                    last_dom_hash = decision.get("dom_hash") or last_dom_hash
                    try:
                        if fallback_decision.get("locator_type") == "css" and fallback_decision.get("locator"):
                            last_choice = ("css", str(fallback_decision.get("locator")))
                        else:
                            last_choice = (fallback_decision.get("role"), fallback_decision.get("name"))
                    except Exception:
                        last_choice = None
                    print(f"Action Result: {'Success' if success else 'Fail'} | URL: {page.url}")

            if not (choice and same_choice_same_url and (no_progress or tentative_streak >= 2)):
                repeat_choice_streak = tentative_streak

            try:
                append_step_trace(
                    {
                        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "step": step_count,
                        "stage": "action_result",
                        "goal": args.prompt,
                        "url_before": current_url,
                        "url_after": page.url,
                        "dom_hash": decision.get("dom_hash"),
                        "success": bool(success),
                        "action": {k: decision.get(k) for k in ("action", "locator_type", "role", "name", "locator", "confidence")},
                    }
                )
            except Exception:
                pass
            # Hard stop if we saved a file and the goal is a download-like task
            if download_intent and len(downloaded_files) > 0:
                try:
                    fnames = ", ".join([os.path.basename(f) for f in downloaded_files])
                except Exception:
                    fnames = str(len(downloaded_files))
                print(f"Task complete: downloaded files: {fnames}")
                run_succeeded = True
                break

            success = execute_action(page, decision, downloaded_files)

            # Wait for page to settle after action
            try:
                page.wait_for_load_state("domcontentloaded", timeout=5000)
            except:
                pass

            try:
                if decision.get("action") == "click" and current_url == page.url:
                    page.wait_for_timeout(900)
            except Exception:
                pass

            # Detect newly opened tabs/pages and switch if more relevant
            try:
                pages_after = context.pages[:]
            except Exception:
                pages_after = []
            new_pages = [p for p in pages_after if p not in pages_before]
            if new_pages:
                for np in new_pages:
                    try:
                        np.wait_for_load_state("domcontentloaded", timeout=7000)
                    except Exception:
                        pass
                # Prefer newly opened pages first, then all pages
                candidate = choose_best_page(new_pages, args.prompt) or choose_best_page(pages_after, args.prompt)
                if candidate and candidate != page:
                    try:
                        url_before = page.url
                    except Exception:
                        url_before = ""
                    try:
                        url_after = candidate.url
                    except Exception:
                        url_after = ""
                    print(f"New tab detected. Switching page: {url_before} -> {url_after}")
                    page = candidate
                    try:
                        page.bring_to_front()
                    except Exception:
                        pass
                    last_url = page.url
                    last_dom_hash = ""
                    repeat_choice_streak = 0

            # If current page got closed, switch to the best remaining one
            try:
                closed = page.is_closed()
            except Exception:
                closed = True
            if closed:
                candidate = choose_best_page(context.pages, args.prompt)
                if candidate:
                    print(f"Active page closed. Switching to: {candidate.url}")
                    page = candidate
                    try:
                        page.bring_to_front()
                    except Exception:
                        pass
                    last_url = page.url
                    last_dom_hash = ""
                    repeat_choice_streak = 0
                else:
                    print("All pages closed. Stopping.")
                    break
            
            # If a likely upload button was clicked successfully but the URL didn't
            # change, hint to the user that a native file picker may be open.
            try:
                if (
                    success
                    and decision.get("action") == "click"
                    and current_url == page.url
                ):
                    label = " ".join(
                        [
                            str(decision.get("name") or "").lower(),
                            str(decision.get("locator") or "").lower(),
                        ]
                    )
                    if _is_upload_like_label(label):
                        msg = (
                            "Upload/file picker detected. Please select the image/file in the system dialog (like the Windows 'Open' popup). "
                            "I'll resume automatically after you choose a file."
                        )
                        print(msg)
                        if session_mode:
                            _emit_status("paused", message=msg, reason="file_picker")
                        t0 = time.time()
                        while True:
                            file_selected = False
                            try:
                                check_js = (
                                    """
                                    () => {
                                      const seen = new Set();
                                      const out = [];
                                      const collect = (root) => {
                                        if (!root || seen.has(root)) return;
                                        seen.add(root);
                                        try {
                                          const nodes = root.querySelectorAll ? root.querySelectorAll('input[type="file"]') : [];
                                          for (const n of nodes) out.push(n);
                                        } catch (e) {}
                                        let all = [];
                                        try { all = root.querySelectorAll ? root.querySelectorAll('*') : []; } catch (e) { all = []; }
                                        for (const el of all) {
                                          try {
                                            if (el && el.shadowRoot) collect(el.shadowRoot);
                                          } catch (e) {}
                                        }
                                      };
                                      collect(document);
                                      for (const i of out) {
                                        try {
                                          if (i && i.files && i.files.length > 0) return true;
                                        } catch (e) {}
                                      }
                                      return false;
                                    }
                                    """
                                )
                                for fr in page.frames:
                                    try:
                                        if bool(fr.evaluate(check_js)):
                                            file_selected = True
                                            break
                                    except Exception:
                                        continue
                            except Exception:
                                file_selected = False
                            if file_selected:
                                print("File selection detected. Resuming.")
                                try:
                                    if not upload_done:
                                        upload_done = True
                                        recent_actions.append("milestone:upload_done")
                                        _run_memory_add_milestone(run_memory, "upload_done")
                                        print("Milestone reached: upload_done=true (file selected)")
                                except Exception:
                                    pass
                                if session_mode:
                                    _emit_status("running")
                                try:
                                    last_dom_hash = ""
                                except Exception:
                                    pass
                                try:
                                    repeat_choice_streak = 0
                                except Exception:
                                    pass
                                break
                            if session_mode:
                                control_nonce, cmd3 = _read_control_command(control_nonce)
                                if cmd3:
                                    c3 = str(cmd3.get("command") or "")
                                    if c3 in {"resume", "new_task"}:
                                        p3 = cmd3.get("prompt")
                                        if isinstance(p3, str) and p3.strip():
                                            args.prompt = p3.strip()
                                        dm3 = cmd3.get("dom_mode")
                                        if isinstance(dm3, str) and dm3.strip():
                                            dom_mode = dm3.strip()
                                        mi3 = cmd3.get("max_items")
                                        if mi3 is not None:
                                            try:
                                                max_items = int(mi3)
                                            except Exception:
                                                pass
                                        try:
                                            print(f"Session config updated via control (resume/new_task): dom_mode={dom_mode!r}, max_items={max_items}")
                                        except Exception:
                                            pass
                                        _emit_status("running")
                                        break
                            if time.time() - t0 > 600:
                                print("Still waiting for file selection. You can select a file in the system dialog, or press Resume in the UI.")
                                t0 = time.time()
                            time.sleep(0.25)
            except Exception:
                pass

            print(f"Action Result: {'Success' if success else 'Fail'} | URL: {page.url}")

            try:
                ck = _run_memory_choice_key_from_decision(decision)
                if ck:
                    last_ck = run_memory.get("last_choice")
                    if last_ck == ck:
                        try:
                            run_memory["repeat_streak"] = int(run_memory.get("repeat_streak") or 0) + 1
                        except Exception:
                            run_memory["repeat_streak"] = 1
                    else:
                        run_memory["repeat_streak"] = 0
                    run_memory["last_choice"] = ck
                    cc = run_memory.get("choice_counts")
                    if not isinstance(cc, dict):
                        cc = {}
                        run_memory["choice_counts"] = cc
                    st = cc.get(ck)
                    if not isinstance(st, dict):
                        st = {"attempts": 0, "success": 0, "url_changes": 0}
                        cc[ck] = st
                    st["attempts"] = int(st.get("attempts") or 0) + 1
                    if bool(success):
                        st["success"] = int(st.get("success") or 0) + 1
                    try:
                        if current_url != page.url:
                            st["url_changes"] = int(st.get("url_changes") or 0) + 1
                    except Exception:
                        pass
                rs = run_memory.get("recent_steps")
                if not isinstance(rs, list):
                    rs = []
                    run_memory["recent_steps"] = rs
                entry = {
                    "step": step_count,
                    "choice": ck,
                    "action": decision.get("action"),
                    "name": decision.get("name"),
                    "success": bool(success),
                    "url_changed": bool(current_url != page.url),
                }
                rs.append(entry)
                if len(rs) > 10:
                    run_memory["recent_steps"] = rs[-10:]
                avoid: List[str] = []
                try:
                    cc = run_memory.get("choice_counts")
                    if isinstance(cc, dict):
                        for k, v in cc.items():
                            if not isinstance(v, dict):
                                continue
                            att = int(v.get("attempts") or 0)
                            suc = int(v.get("success") or 0)
                            urlc = int(v.get("url_changes") or 0)
                            if att >= 2 and suc <= 0 and urlc <= 0:
                                avoid.append(str(k))
                except Exception:
                    avoid = []
                if len(avoid) > 12:
                    avoid = avoid[:12]
                run_memory["avoid_choices"] = avoid
            except Exception:
                pass

            # Persist run memory snapshot for this step (compact but sufficient for debug/inspection)
            try:
                save_run_memory(args.prompt, run_memory, step_count, url=page.url)
            except Exception:
                pass
            
            # Record all successful steps so manual "Remember" can snapshot the full path.
            # We still respect WORKFLOW_MIN_STEP_CONF when *automatically* saving at the end.
            if success and workflow_mem and workflow_mem.enabled:
                try:
                    conf_val = float(decision.get("confidence", 0.0) or 0.0)
                except Exception:
                    conf_val = 0.0
                try:
                    step_meta: Dict[str, Any] = {
                        "order": len(current_workflow_steps),
                        "action": decision.get("action"),
                        "locator_type": decision.get("locator_type"),
                        "role": decision.get("role"),
                        "name": decision.get("name"),
                        "selector": decision.get("locator"),
                        "input_value": decision.get("value"),
                        "url_before": current_url,
                        "url_after": page.url,
                        "confidence": conf_val,
                    }
                    current_workflow_steps.append(step_meta)
                except Exception:
                    pass

            # If the user pressed "Remember" in the UI, persist the current steps
            if workflow_mem and workflow_mem.enabled:
                _check_manual_remember(workflow_mem, site, args.prompt, current_workflow_steps)

            if not success:
                print("Action failed. Retrying loop...")
                
            act_label = decision.get("action") or "?"
            if decision.get("locator_type") == "role":
                act_label += f":{decision.get('role')}:{decision.get('name')}"
            recent_actions.append(f"{act_label}")
            last_url = page.url
            last_dom_hash = decision.get("dom_hash") or last_dom_hash
            last_choice = None
            try:
                if decision.get("locator_type") == "css" and decision.get("locator"):
                    last_choice = ("css", str(decision.get("locator")))
                else:
                    last_choice = (decision.get("role"), decision.get("name"))
            except Exception:
                last_choice = None

            # Small wait for stability
            time.sleep(1)

        if step_count >= max_steps:
            print("Max steps reached. Stopping.")

        # Save learned workflow if run succeeded and we have a memory backend
        if run_succeeded and workflow_mem and workflow_mem.enabled and current_workflow_steps and site:
            try:
                # Prefer high-confidence steps for automatic workflows, but fall back to all
                # recorded steps if none meet the threshold.
                steps_for_save: List[Dict[str, Any]] = []
                try:
                    for s in current_workflow_steps:
                        try:
                            cval = float(s.get("confidence", 0.0) or 0.0)
                        except Exception:
                            cval = 0.0
                        if cval >= workflow_mem.min_conf:
                            steps_for_save.append(s)
                except Exception:
                    steps_for_save = []

                if not steps_for_save:
                    steps_for_save = current_workflow_steps

                print(f"Saving workflow with {len(steps_for_save)} steps for site '{site}'...")
                workflow_mem.save(site, args.prompt, steps_for_save, source="auto")
                print("Workflow saved.")
            except Exception as e:
                print(f"Workflow save error: {e}")

        if session_mode:
            # In session mode, just emit final status and leave the browser
            # running so the user can keep the tab open.
            _emit_status("completed" if run_succeeded else "stopped")
            return
        else:
            # CLI mode: wait for Enter and then close the browser window.
            input("Press Enter to close browser...")
            try:
                context.close()
            except Exception:
                pass
            if browser is not None:
                try:
                    browser.close()
                except Exception:
                    pass

if __name__ == "__main__":
    main()
