import os
import sys
import json
import time
import re
import math
import hashlib
import contextlib
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse
import urllib.request, urllib.error
from dotenv import load_dotenv
from playwright.sync_api import sync_playwright, Page
from openai import OpenAI

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

# Load environment variables
load_dotenv()

# --- Configuration ---
class Config:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    SMART_MODEL = "llama-3.3-70b-versatile"  # Groq (Planning/Reasoning)
    WORKFLOW_MEMORY_ENABLED = os.getenv("WORKFLOW_MEMORY_ENABLED", "1") != "0"
    REDIS_URL = os.getenv("WORKFLOW_REDIS_URL") or os.getenv("REDIS_URL")
    WORKFLOW_SIM_THRESHOLD = float(os.getenv("WORKFLOW_SIM_THRESHOLD", "0.8"))
    WORKFLOW_MAX_PER_SITE = int(os.getenv("WORKFLOW_MAX_PER_SITE", "100"))
    WORKFLOW_TTL_SECONDS = int(os.getenv("WORKFLOW_TTL_SECONDS", str(30 * 24 * 3600)))
    WORKFLOW_MIN_STEP_CONF = float(os.getenv("WORKFLOW_MIN_STEP_CONF", "0.9"))

config = Config()

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


def playwright_filter_interactive_elements(page: Page, goal: str) -> str:
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
    try:
        snapshot = page.accessibility.snapshot() or {}
    except Exception as e:
        print(f"AX snapshot failed: {e}")
        return "[]"

    flat: List[Dict[str, Any]] = []
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

    items: List[Dict[str, Any]] = []
    for node in flat:
        role = node.get("role")
        name = (node.get("name") or "").strip()
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
    uniq: List[Dict[str, Any]] = []
    for it in items:
        key = (it["role"], it["name"])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(it)

    # Enrich with attributes (href, aria-expanded) and kind for links/buttons
    for it in uniq:
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

    # Rank by relevance to goal, control type, and link quality
    g = goal.lower()
    def _score(it: Dict[str, Any]) -> int:
        n = it["name"].lower()
        s = 0
        if it["role"] in ("searchbox", "textbox"):
            s += 3
        if any(tok and tok in n for tok in g.split()):
            s += 2
        # Strongly prioritize download/save controls when the goal is to download/save
        if any(k in g for k in ["download", "save"]):
            if any(w in n for w in ["download", "save", "save image", "get image", "jpg", "png", "jpeg"]):
                s += 8
        if it.get("is_money_action"):
            s += 4
        if it["role"] == "button":
            s += 1
        # Exploration keywords (generic, non site-specific)
        if any(k in n for k in [
            "apply", "application", "form", "admission", "admissions",
            "learn more", "get started", "explore", "program", "course",
            "contact", "about", "next", "continue"
        ]):
            s += 2
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
    trimmed = uniq[:40]
    return json.dumps(trimmed, ensure_ascii=False)


def save_filtered_dom(goal: str, filtered_json_array: str) -> None:
    """Persist the filtered DOM that we send to Groq on every step.

    Writes a JSONL entry to filtered_dom_log.jsonl and a snapshot to last_filtered_dom.json.
    """
    try:
        record = {
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "goal": goal,
            "elements": json.loads(filtered_json_array),
        }
        with open("filtered_dom_log.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        with open("last_filtered_dom.json", "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Failed saving filtered DOM: {e}")

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

def analyze_dom_and_act(
    page: Page,
    goal: str,
    current_url: str,
    recent_actions: List[str],
    blocked_choices: List[tuple] | set[tuple],
    downloaded_files: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Phase 2: ReAct Loop - Analyze DOM and decide NEXT action based on GOAL."""
    
    # 1-2. Deterministic filter via Playwright AX snapshot (replaces Ollama)
    filtered_summary = playwright_filter_interactive_elements(page, goal)
    dom_hash = str(hash(filtered_summary))[-8:]
    print(f"Filtered DOM Hash: {dom_hash} | Length: {len(filtered_summary)}")
    save_filtered_dom(goal, filtered_summary)
    
    # 3. Decide Action (Groq)
    print("Deciding next action...")
    system_prompt = (
        "You are an autonomous web agent. "
        f"Your ultimate goal is: '{goal}'. "
        "You are looking at the current interactive elements of the page. "
        "Decide the IMMEDIATE NEXT action to move closer to the goal. "
        "Return a JSON object with keys: action, locator_type, role, name, locator, value, confidence. "
        "Rules: "
        "- action must be one of: 'click', 'type', 'hover', 'wait', 'done', 'fail'. "
        "- Prefer role-based targeting: set locator_type='role' and include exact 'role' and 'name' from the list. "
        "- Only if role-based cannot work, set locator_type='css' and provide 'locator' as a CSS selector present in the list. "
        "- Prefer real navigation links (href_kind='nav') over anchors ('#') or javascript links. "
        "- If a top-level control toggles a menu (aria-expanded changes), use 'hover' or a subsequent click on a submenu item. "
        "- value is only for 'type' action (the text to enter). "
        "- confidence: 0.0..1.0. "
        "Safety & Looping: "
        "1. If the goal content is already visible, return 'done'. Do NOT navigate away. "
        "2. If you already searched and see results, click a result link; do NOT search again. "
        "3. If an input already has the correct value, do NOT type again. Prefer clicking search or a result. "
        "4. Avoid repeating the same click if the URL does not change. If you are stuck in a repeated loop, choose a different control (prefer nav links or exploratory items), or return 'fail'. "
        "5. If a file has been saved to disk that satisfies the goal (see 'Downloads this run'), return 'done'. "
        "6. If the goal includes download/save, prioritize controls whose names include 'download' or 'save'.")
    
    # Remove blocked (role,name) from the list we show to the LLM
    items_for_llm: List[Dict[str, Any]] = []
    try:
        items_for_llm = json.loads(filtered_summary)
        blocked_set = set(blocked_choices or [])
        items_for_llm = [it for it in items_for_llm if (it.get("role"), it.get("name")) not in blocked_set]
        filtered_for_llm = json.dumps(items_for_llm, ensure_ascii=False)
    except Exception:
        filtered_for_llm = filtered_summary

    context_note = (
        f"Current URL: {current_url}\n"
        f"Recent actions: {', '.join(recent_actions[-3:]) if recent_actions else 'none'}\n"
        f"Blocked choices: {', '.join([f'{r}:{n}' for (r,n) in (blocked_choices or [])]) if blocked_choices else 'none'}\n"
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
                    {"role": "user", "content": context_note + f"Current DOM Elements (JSON array with role/name):\n{filtered_for_llm}"},
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
    if response is None:
        host = urlparse(current_url).hostname or ""
        fallback = None
        try:
            for it in items_for_llm:
                key = (it.get("role"), it.get("name"))
                if blocked_choices and key in blocked_choices:
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
                    key = (it.get("role"), it.get("name"))
                    if blocked_choices and key in blocked_choices:
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
                    key = (it.get("role"), it.get("name"))
                    if blocked_choices and key in blocked_choices:
                        continue
                    n = (it.get("name") or "").lower()
                    if any(k in n for k in ["apply", "application", "form", "learn more", "explore", "program", "course", "contact", "about", "next", "continue"]):
                        fallback = it
                        break
            if not fallback:
                for it in items_for_llm:
                    key = (it.get("role"), it.get("name"))
                    if blocked_choices and key in blocked_choices:
                        continue
                    if it.get("href_kind") == "nav":
                        fallback = it
                        break
        except Exception:
            fallback = None
        if fallback:
            response = {
                "action": "click",
                "locator_type": "role",
                "role": fallback.get("role"),
                "name": fallback.get("name"),
                "confidence": 0.4,
            }
        else:
            response = {"action": "fail", "locator_type": "role", "role": None, "name": None, "confidence": 0.0}
    
    response['dom_hash'] = dom_hash
    # Return a small candidate list to the caller for fallback exploration
    try:
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
            if locator_type == "role":
                role = decision.get("role")
                name = decision.get("name")
                elem = page.get_by_role(role, name=name).first
            else:
                elem = page.locator(locator_str).first
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

            if locator_type == "role":
                role = decision.get("role")
                name = decision.get("name")
                elem = page.get_by_role(role, name=name).first
            else:
                elem = page.locator(locator_str).first
            # Force fill without strict scroll checks
            elem.fill(value, timeout=10000, force=True)
            
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
                viewport={"width": 1280, "height": 720},
                accept_downloads=True,
                args=[
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
                "--disable-blink-features=AutomationControlled",
                "--disable-features=IsolateOrigins,site-per-process",
            ],
        )
        context = browser.new_context(
            viewport={"width": 1280, "height": 720},
            accept_downloads=True,
        )
        return context, browser
    except Exception as e:
        print(f"Non-persistent Chromium launch also failed: {e}")
        raise


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True, help="Task description")
    args = parser.parse_args()

    print(f"Agent started with goal: '{args.prompt}'")

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

    with sync_playwright() as p:
        # Robust launch: prefer persistent profile, auto-reset if corrupted,
        # and fall back to non-persistent if needed.
        context = None
        browser = None
        try:
            context, browser = _launch_hemlo_context(p)
        except Exception:
            # If everything fails, abort early with a clear message.
            print("Fatal error: unable to launch any Chromium context. Aborting run.")
            return

        page = context.new_page()

        print(f"Navigating to {target_url}...")
        try:
            page.goto(target_url, timeout=60000)
            page.wait_for_load_state("domcontentloaded")
        except Exception as e:
            print(f"Navigation failed: {e}")
            return

        downloaded_files: List[str] = []

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
                input("Press Enter to close browser...")
                context.close()
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
        blocked_choices: set[tuple] = set()
        last_choice: Optional[tuple] = None
        run_succeeded = False
        last_login_prompt_url = ""
        
        while step_count < max_steps:
            step_count += 1
            print(f"\n--- Step {step_count} ---")
            current_url = page.url

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
                                _fs = playwright_filter_interactive_elements(page, args.prompt)
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
                                    _fs = playwright_filter_interactive_elements(page, args.prompt)
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
                        applied = perform_login_choice(page, choice)
                        print(f"Login choice applied: {'success' if applied else 'failed'} for provider={choice.get('provider')}")
                        try:
                            page.wait_for_load_state("networkidle", timeout=8000)
                        except Exception:
                            pass
                        # Reset DOM hash to avoid loop detection on same page
                        try:
                            last_dom_hash = ""
                            last_url = page.url
                        except Exception:
                            pass
                        continue
                    else:
                        print("No login choice received; continuing without login.")
                        # Avoid spamming the same URL; mark as emitted
                        continue

            # Analyze & Decide
            decision = analyze_dom_and_act(page, args.prompt, current_url, recent_actions, blocked_choices, downloaded_files)
            
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
            choice = (decision.get("role"), decision.get("name")) if decision.get("locator_type") == "role" else None
            no_progress = (current_url == last_url) and (decision.get("dom_hash") == last_dom_hash)
            if choice and last_choice == choice and no_progress:
                print(f"Loop detected on {choice}; blocking and exploring alternative.")
                blocked_choices.add(choice)
                # Fallback exploration: pick next best from filtered_items
                candidates = decision.get("filtered_items") or []
                fallback = None
                # Prefer nav links
                for it in candidates:
                    key = (it.get("role"), it.get("name"))
                    if key in blocked_choices:
                        continue
                    if it.get("href_kind") == "nav":
                        fallback = it
                        break
                # Or exploration keywords
                if not fallback:
                    for it in candidates:
                        key = (it.get("role"), it.get("name"))
                        if key in blocked_choices:
                            continue
                        n = (it.get("name") or "").lower()
                        if any(k in n for k in ["apply", "application", "form", "learn more", "explore", "program", "course", "contact", "about", "next", "continue"]):
                            fallback = it
                            break
                # Otherwise first unblocked
                if not fallback:
                    for it in candidates:
                        key = (it.get("role"), it.get("name"))
                        if key not in blocked_choices:
                            fallback = it
                            break
                if fallback:
                    fallback_decision = {
                        "action": "click",
                        "locator_type": "role",
                        "role": fallback.get("role"),
                        "name": fallback.get("name"),
                        "confidence": 0.4,
                    }
                    success = execute_action(page, fallback_decision, downloaded_files)
                    act_label = f"fallback_click:{fallback.get('role')}:{fallback.get('name')}"
                    recent_actions.append(act_label)
                    # Update trackers
                    try:
                        page.wait_for_load_state("networkidle", timeout=5000)
                    except Exception:
                        pass
                    last_url = page.url
                    last_dom_hash = decision.get("dom_hash") or last_dom_hash
                    last_choice = (fallback_decision["role"], fallback_decision["name"]) if fallback_decision.get("role") else None
                    print(f"Action Result: {'Success' if success else 'Fail'} | URL: {page.url}")
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
                page.wait_for_load_state("networkidle", timeout=5000)
            except:
                pass  # Continue even if networkidle times out

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
                else:
                    print("All pages closed. Stopping.")
                    break
            
            print(f"Action Result: {'Success' if success else 'Fail'} | URL: {page.url}")
            
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
            last_choice = (decision.get("role"), decision.get("name")) if decision.get("locator_type") == "role" else None

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
