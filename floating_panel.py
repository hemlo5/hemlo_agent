import json
import os
import time
import tkinter as tk
from tkinter.scrolledtext import ScrolledText
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


THEME_BG = "#0b1220"
THEME_PANEL = "#0f172a"
THEME_HEADER = "#111827"
THEME_TEXT_BG = "#0b1220"
THEME_TEXT_FG = "#e5e7eb"
THEME_MUTED = "#94a3b8"
THEME_BORDER = "#1f2937"

THEME_BTN = "#1f2937"
THEME_BTN_HOVER = "#374151"
THEME_BLUE = "#3b82f6"
THEME_GREEN = "#22c55e"
THEME_RED = "#ef4444"
THEME_AMBER = "#f59e0b"


def _http_get_json(url: str, timeout_seconds: float = 2.0):
    req = Request(url, headers={"Accept": "application/json"})
    with urlopen(req, timeout=timeout_seconds) as resp:
        body = resp.read().decode("utf-8", errors="replace")
    return json.loads(body)


def _http_get(url: str, timeout_seconds: float = 2.0) -> str:
    req = Request(url)
    with urlopen(req, timeout=timeout_seconds) as resp:
        body = resp.read().decode("utf-8", errors="replace")
    return body


class FloatingPanel(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Hemlo Floating Panel")
        self.minsize(320, 200)
        try:
            self.configure(bg=THEME_BG)
        except Exception:
            pass

        self.base_url_var = tk.StringVar(value=os.environ.get("HEMLO_PANEL_URL") or "http://127.0.0.1:5000")
        self.lines_var = tk.StringVar(value=os.environ.get("HEMLO_PANEL_LINES") or "2000")
        self.topmost_var = tk.BooleanVar(value=True)
        self.status_var = tk.StringVar(value="Disconnected")
        self.show_settings_var = tk.BooleanVar(value=False)

        self._last_text = ""
        self._last_state = None
        self._last_ok_time = 0.0

        # For rotating activity label + thinking animation
        self._phase_index = -1
        self._thinking_anim_phase = 0
        self._thinking_running = False

        # For login approvals (provider buttons + optional email/password)
        self._login_options = []
        self._login_url = ""
        self._login_email_provider = ""

        self._build_ui()
        self._apply_topmost()
        self.after(0, self._place_bottom_right)
        self.after(120, self._bring_to_front)
        self.after(200, self._poll)

    def _build_ui(self):
        root = tk.Frame(self, bg=THEME_BG)
        root.pack(fill=tk.BOTH, expand=True)

        header = tk.Frame(root, bg=THEME_HEADER)
        header.pack(fill=tk.X, padx=10, pady=(10, 8))

        left = tk.Frame(header, bg=THEME_HEADER)
        left.pack(side=tk.LEFT, fill=tk.X, expand=True)

        title = tk.Label(
            left,
            text="Hemlo",
            bg=THEME_HEADER,
            fg=THEME_TEXT_FG,
            font=("Segoe UI", 11, "bold"),
        )
        title.pack(side=tk.LEFT)

        self.status_label = tk.Label(
            left,
            textvariable=self.status_var,
            bg=THEME_BORDER,
            fg=THEME_MUTED,
            font=("Segoe UI", 9, "bold"),
            padx=10,
            pady=3,
        )
        self.status_label.pack(side=tk.LEFT, padx=(10, 0))

        # Rotating activity label ("Thinking", "Planning", ...)
        self.activity_var = tk.StringVar(value="Idle")
        self.activity_label = tk.Label(
            left,
            textvariable=self.activity_var,
            bg=THEME_HEADER,
            fg=THEME_MUTED,
            font=("Segoe UI", 9),
            padx=8,
        )
        self.activity_label.pack(side=tk.LEFT, padx=(8, 0))

        # Animated thinking indicator (small ripple circle)
        self.thinking_canvas = tk.Canvas(
            left,
            width=18,
            height=18,
            bg=THEME_HEADER,
            highlightthickness=0,
            bd=0,
        )
        self.thinking_canvas.pack(side=tk.LEFT, padx=(4, 0))
        try:
            self._thinking_circle = self.thinking_canvas.create_oval(7, 7, 11, 11, outline=THEME_MUTED, width=2)
        except Exception:
            self._thinking_circle = None

        right = tk.Frame(header, bg=THEME_HEADER)
        right.pack(side=tk.RIGHT)

        self.settings_btn = tk.Button(
            right,
            text="Settings",
            command=self._toggle_settings,
            bg=THEME_HEADER,
            fg=THEME_MUTED,
            activebackground=THEME_HEADER,
            activeforeground=THEME_TEXT_FG,
            relief="flat",
            bd=0,
            highlightthickness=0,
            font=("Segoe UI", 9),
            cursor="hand2",
        )
        self.settings_btn.pack(side=tk.RIGHT, padx=(8, 0))

        self.topmost_cb = tk.Checkbutton(
            right,
            text="Top",
            variable=self.topmost_var,
            command=self._apply_topmost,
            bg=THEME_HEADER,
            fg=THEME_MUTED,
            selectcolor=THEME_HEADER,
            activebackground=THEME_HEADER,
            activeforeground=THEME_TEXT_FG,
            bd=0,
            highlightthickness=0,
            font=("Segoe UI", 9, "bold"),
        )
        self.topmost_cb.pack(side=tk.RIGHT)

        self.settings_frame = tk.Frame(root, bg=THEME_BG)
        self.settings_frame.pack(fill=tk.X, padx=10)

        self._settings_inner = tk.Frame(self.settings_frame, bg=THEME_PANEL, highlightbackground=THEME_BORDER, highlightthickness=1)
        self._settings_inner.pack(fill=tk.X)

        tk.Label(
            self._settings_inner,
            text="Server",
            bg=THEME_PANEL,
            fg=THEME_MUTED,
            font=("Segoe UI", 9, "bold"),
        ).grid(row=0, column=0, sticky="w", padx=(10, 6), pady=8)
        self.server_entry = tk.Entry(
            self._settings_inner,
            textvariable=self.base_url_var,
            bg=THEME_TEXT_BG,
            fg=THEME_TEXT_FG,
            insertbackground=THEME_TEXT_FG,
            relief="flat",
            highlightbackground=THEME_BORDER,
            highlightthickness=1,
            font=("Segoe UI", 9),
        )
        self.server_entry.grid(row=0, column=1, sticky="we", padx=(0, 10), pady=8)

        tk.Label(
            self._settings_inner,
            text="Lines",
            bg=THEME_PANEL,
            fg=THEME_MUTED,
            font=("Segoe UI", 9, "bold"),
        ).grid(row=0, column=2, sticky="w", padx=(0, 6), pady=8)
        self.lines_entry = tk.Entry(
            self._settings_inner,
            textvariable=self.lines_var,
            width=7,
            bg=THEME_TEXT_BG,
            fg=THEME_TEXT_FG,
            insertbackground=THEME_TEXT_FG,
            relief="flat",
            highlightbackground=THEME_BORDER,
            highlightthickness=1,
            font=("Segoe UI", 9),
        )
        self.lines_entry.grid(row=0, column=3, sticky="w", padx=(0, 10), pady=8)

        self._settings_inner.columnconfigure(1, weight=1)

        actions = tk.Frame(root, bg=THEME_BG)
        actions.pack(fill=tk.X, padx=10, pady=(10, 8))

        self.stop_btn = self._btn(actions, "Stop", lambda: self._send_control("stop"), THEME_RED)
        self.stop_btn.pack(side=tk.LEFT)

        self.pause_resume_btn = self._btn(actions, "Pause", self._toggle_pause_resume, THEME_AMBER)
        self.pause_resume_btn.pack(side=tk.LEFT, padx=(8, 0))

        self.approve_btn = self._btn(actions, "Approve", lambda: self._send_control("approve_money"), THEME_GREEN)
        self.approve_btn.pack(side=tk.LEFT, padx=(8, 0))

        self.refresh_btn = self._btn(actions, "Refresh", self._refresh_logs, THEME_BTN, hover=THEME_BTN_HOVER)
        self.refresh_btn.pack(side=tk.RIGHT)

        # Login approvals area (UI-style provider buttons like UI1)
        self.login_frame = tk.Frame(root, bg=THEME_BG)
        self.login_frame.pack(fill=tk.X, padx=10, pady=(0, 6))

        login_title = tk.Label(
            self.login_frame,
            text="Login approval needed",
            bg=THEME_BG,
            fg=THEME_MUTED,
            font=("Segoe UI", 9, "bold"),
        )
        login_title.pack(anchor="w")

        login_hint = tk.Label(
            self.login_frame,
            text="Choose how Hemlo should sign in (Google, Email, etc.) so it can continue.",
            bg=THEME_BG,
            fg=THEME_MUTED,
            font=("Segoe UI", 8),
        )
        login_hint.pack(anchor="w", pady=(0, 2))

        self.login_buttons_frame = tk.Frame(self.login_frame, bg=THEME_BG)
        self.login_buttons_frame.pack(fill=tk.X, pady=(2, 2))

        self.login_email_frame = tk.Frame(self.login_frame, bg=THEME_PANEL, highlightbackground=THEME_BORDER, highlightthickness=1)
        self.login_email_frame.pack(fill=tk.X, pady=(4, 0))

        email_row = tk.Frame(self.login_email_frame, bg=THEME_PANEL)
        email_row.pack(fill=tk.X, padx=8, pady=6)

        tk.Label(
            email_row,
            text="Email",
            bg=THEME_PANEL,
            fg=THEME_MUTED,
            font=("Segoe UI", 8, "bold"),
        ).grid(row=0, column=0, sticky="w", padx=(0, 6), pady=(0, 4))

        self.login_email_entry = tk.Entry(
            email_row,
            bg=THEME_TEXT_BG,
            fg=THEME_TEXT_FG,
            insertbackground=THEME_TEXT_FG,
            relief="flat",
            highlightbackground=THEME_BORDER,
            highlightthickness=1,
            font=("Segoe UI", 9),
        )
        self.login_email_entry.grid(row=0, column=1, sticky="we", pady=(0, 4))

        tk.Label(
            email_row,
            text="Password",
            bg=THEME_PANEL,
            fg=THEME_MUTED,
            font=("Segoe UI", 8, "bold"),
        ).grid(row=1, column=0, sticky="w", padx=(0, 6), pady=(0, 4))

        self.login_password_entry = tk.Entry(
            email_row,
            show="*",
            bg=THEME_TEXT_BG,
            fg=THEME_TEXT_FG,
            insertbackground=THEME_TEXT_FG,
            relief="flat",
            highlightbackground=THEME_BORDER,
            highlightthickness=1,
            font=("Segoe UI", 9),
        )
        self.login_password_entry.grid(row=1, column=1, sticky="we", pady=(0, 4))

        email_row.columnconfigure(1, weight=1)

        btn_row = tk.Frame(self.login_email_frame, bg=THEME_PANEL)
        btn_row.pack(fill=tk.X, padx=8, pady=(0, 6))

        self.login_submit_btn = self._btn(btn_row, "Submit", self._submit_email_login, THEME_BLUE)
        self.login_submit_btn.pack(side=tk.RIGHT)

        # Hidden by default until we actually have options
        try:
            self.login_frame.pack_forget()
        except Exception:
            pass
        try:
            self.login_email_frame.pack_forget()
        except Exception:
            pass

        log_wrap = tk.Frame(root, bg=THEME_BG)
        log_wrap.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        log_card = tk.Frame(log_wrap, bg=THEME_PANEL, highlightbackground=THEME_BORDER, highlightthickness=1)
        log_card.pack(fill=tk.BOTH, expand=True)

        self.text = ScrolledText(
            log_card,
            height=9,
            wrap=tk.WORD,
            bg=THEME_TEXT_BG,
            fg=THEME_TEXT_FG,
            insertbackground=THEME_TEXT_FG,
            relief="flat",
            bd=0,
            padx=10,
            pady=8,
        )
        self.text.pack(fill=tk.BOTH, expand=True)
        try:
            self.text.configure(font=("Consolas", 9))
        except Exception:
            pass
        self.text.configure(state=tk.DISABLED)

        try:
            self._apply_settings_visibility()
        except Exception:
            pass

    def _btn(self, parent, label: str, command, bg: str, hover: str = None):
        hover_bg = hover or bg
        b = tk.Button(
            parent,
            text=label,
            command=command,
            bg=bg,
            fg="#ffffff",
            activebackground=hover_bg,
            activeforeground="#ffffff",
            relief="flat",
            bd=0,
            highlightthickness=0,
            font=("Segoe UI", 9, "bold"),
            padx=10,
            pady=6,
            cursor="hand2",
        )
        try:
            b.bind("<Enter>", lambda _e: b.configure(bg=hover_bg))
            b.bind("<Leave>", lambda _e: b.configure(bg=bg))
        except Exception:
            pass
        return b

    def _toggle_settings(self):
        try:
            self.show_settings_var.set(not bool(self.show_settings_var.get()))
        except Exception:
            return
        self._apply_settings_visibility()

    def _apply_settings_visibility(self):
        show = bool(self.show_settings_var.get())
        try:
            self.settings_btn.configure(text=("Hide" if show else "Settings"))
        except Exception:
            pass
        try:
            if show:
                self.settings_frame.pack(fill=tk.X, padx=10)
            else:
                self.settings_frame.pack_forget()
        except Exception:
            pass

    def _apply_topmost(self):
        try:
            self.attributes("-topmost", bool(self.topmost_var.get()))
        except Exception:
            pass

    def _place_bottom_right(self):
        try:
            self.update_idletasks()
            w = self.winfo_width()
            h = self.winfo_height()
            sw = self.winfo_screenwidth()
            sh = self.winfo_screenheight()
            x = max(0, sw - w - 24)
            y = max(0, sh - h - 64)
            self.geometry(f"{w}x{h}+{x}+{y}")
        except Exception:
            pass

    def _bring_to_front(self):
        try:
            self.deiconify()
        except Exception:
            pass
        try:
            self.lift()
        except Exception:
            pass
        try:
            self.focus_force()
        except Exception:
            pass
        try:
            was = bool(self.topmost_var.get())
            self.attributes("-topmost", True)
            self.after(50, lambda: self.attributes("-topmost", bool(was)))
        except Exception:
            pass

    def _safe_int(self, s: str, default: int) -> int:
        try:
            v = int(str(s).strip())
            return v
        except Exception:
            return default

    def _get_base_url(self) -> str:
        base = str(self.base_url_var.get() or "").strip()
        if base.endswith("/"):
            base = base[:-1]
        return base

    def _send_control(self, cmd: str):
        base = self._get_base_url()
        qs = urlencode({"command": cmd})
        url = f"{base}/overlay_control?{qs}"
        try:
            _http_get(url)
        except Exception:
            pass
        self.after(50, self._poll)

    def _refresh_logs(self):
        base = self._get_base_url()
        try:
            qs = urlencode({"command": "clear_logs"})
            url = f"{base}/overlay_control?{qs}"
            _http_get(url)
        except Exception:
            pass
        try:
            self._last_text = ""
        except Exception:
            self._last_text = ""
        try:
            self._set_text("", keep_scrolled_to_bottom=True)
        except Exception:
            pass
        self.after(50, self._poll)

    def _set_text(self, text: str, keep_scrolled_to_bottom: bool):
        self.text.configure(state=tk.NORMAL)
        self.text.delete("1.0", tk.END)
        self.text.insert(tk.END, text)
        self.text.configure(state=tk.DISABLED)
        if keep_scrolled_to_bottom:
            try:
                self.text.yview_moveto(1.0)
            except Exception:
                pass

    def _update_login_ui(self, options, url: str):
        """Render login provider options in the floating panel (like UI1)."""
        try:
            opts = [o for o in (options or []) if isinstance(o, dict)]
        except Exception:
            opts = []

        self._login_options = opts
        self._login_url = str(url or "")

        if not opts:
            # Hide login area and clear inputs
            try:
                self.login_frame.pack_forget()
            except Exception:
                pass
            try:
                self.login_email_frame.pack_forget()
            except Exception:
                pass
            try:
                self.login_email_entry.delete(0, tk.END)
                self.login_password_entry.delete(0, tk.END)
            except Exception:
                pass
            self._login_email_provider = ""
            return

        # Ensure the frame is visible
        try:
            self.login_frame.pack(fill=tk.X, padx=10, pady=(0, 6))
        except Exception:
            pass

        # Rebuild provider buttons
        try:
            for child in list(self.login_buttons_frame.winfo_children()):
                child.destroy()
        except Exception:
            pass

        for opt in opts:
            try:
                label = str(opt.get("label") or opt.get("provider") or opt.get("type") or "Login")
            except Exception:
                label = "Login"
            btn = self._btn(
                self.login_buttons_frame,
                label,
                lambda o=opt: self._handle_login_option_click(o),
                THEME_BTN,
                hover=THEME_BTN_HOVER,
            )
            btn.pack(side=tk.LEFT, padx=(0, 6), pady=(0, 2))

    def _handle_login_option_click(self, opt: dict):
        """Handle clicking on a single login option (OAuth vs email/password)."""
        t = str((opt or {}).get("type") or "").lower()
        provider = str((opt or {}).get("provider") or opt.get("type") or "unknown")

        if t == "email":
            # Show email/password inputs for this provider
            self._login_email_provider = provider or "email"
            try:
                self.login_email_frame.pack(fill=tk.X, pady=(4, 0))
            except Exception:
                pass
            try:
                self.login_email_entry.focus_set()
            except Exception:
                pass
        else:
            # Fire-and-forget OAuth-style choice
            choice = {"provider": provider}
            self._send_login_choice(choice)
            self._hide_login_ui()

    def _submit_email_login(self):
        email = ""
        password = ""
        try:
            email = (self.login_email_entry.get() or "").strip()
        except Exception:
            email = ""
        try:
            password = self.login_password_entry.get() or ""
        except Exception:
            password = ""

        if not email or not password:
            # Simple inline hint by updating summary text; avoid popups
            try:
                self.summary_label.configure(text="Please enter email and password to continue the login.")
            except Exception:
                pass
            return

        provider = self._login_email_provider or "email"
        choice = {"provider": provider, "email": email, "password": password}
        self._send_login_choice(choice)
        self._hide_login_ui()

    def _hide_login_ui(self):
        try:
            self.login_frame.pack_forget()
        except Exception:
            pass
        try:
            self.login_email_frame.pack_forget()
        except Exception:
            pass
        try:
            self.login_email_entry.delete(0, tk.END)
            self.login_password_entry.delete(0, tk.END)
        except Exception:
            pass
        self._login_email_provider = ""

    def _send_login_choice(self, choice: dict):
        """Submit a login choice back to the server via /overlay_login_choice."""
        base = self._get_base_url()
        url = f"{base}/overlay_login_choice"
        try:
            payload = json.dumps(choice or {}, ensure_ascii=False).encode("utf-8")
        except Exception:
            payload = b"{}"
        try:
            req = Request(url, data=payload, headers={"Content-Type": "application/json"})
            with urlopen(req, timeout=3.0):
                pass
        except Exception:
            pass

    def _update_activity(self, status: str, text_changed: bool):
        """Rotate cute activity phrases while running; show simple labels otherwise."""
        try:
            s = str(status or "").strip().lower()
        except Exception:
            s = ""

        if s == "running":
            # Advance phase when we see new log text, so it feels tied to progress
            if text_changed:
                phases = ["Thinking", "Planning", "Exploring", "Acting"]
                try:
                    self._phase_index = int(self._phase_index) + 1
                except Exception:
                    self._phase_index = 0
                if self._phase_index < 0:
                    self._phase_index = 0
                idx = self._phase_index % len(phases)
                label = phases[idx]
                try:
                    self.activity_var.set(label)
                except Exception:
                    pass
            else:
                # Keep whatever the last running label was
                if not self.activity_var.get():
                    try:
                        self.activity_var.set("Thinking")
                    except Exception:
                        pass
        elif s == "paused":
            try:
                self.activity_var.set("Paused")
            except Exception:
                pass
        elif s in {"completed", "stopped"}:
            try:
                self.activity_var.set("Done")
            except Exception:
                pass
        elif s == "error":
            try:
                self.activity_var.set("Error")
            except Exception:
                pass
        else:
            try:
                self.activity_var.set("Idle")
            except Exception:
                pass

        # Drive thinking animation based on running state
        try:
            self._set_thinking_active(s == "running")
        except Exception:
            pass

    def _toggle_pause_resume(self):
        try:
            s = str(self._last_state or "").strip().lower()
        except Exception:
            s = ""
        if s == "paused":
            self._send_control("resume")
        else:
            self._send_control("pause")

    def _set_thinking_active(self, active: bool):
        try:
            active = bool(active)
        except Exception:
            active = False
        if active:
            if not getattr(self, "_thinking_running", False):
                self._thinking_running = True
                try:
                    self._thinking_anim_phase = 0
                except Exception:
                    self._thinking_anim_phase = 0
                try:
                    self._animate_thinking()
                except Exception:
                    pass
        else:
            self._thinking_running = False
            try:
                if getattr(self, "thinking_canvas", None) and getattr(self, "_thinking_circle", None):
                    self.thinking_canvas.coords(self._thinking_circle, 7, 7, 11, 11)
            except Exception:
                pass

    def _animate_thinking(self):
        if not getattr(self, "_thinking_running", False):
            return
        try:
            self._thinking_anim_phase = int(getattr(self, "_thinking_anim_phase", 0)) + 1
        except Exception:
            self._thinking_anim_phase = 0
        phase = self._thinking_anim_phase % 6
        base = 9.0
        radius = 3.0 + float(phase)
        x0 = base - radius
        y0 = base - radius
        x1 = base + radius
        y1 = base + radius
        try:
            if getattr(self, "thinking_canvas", None) and getattr(self, "_thinking_circle", None):
                self.thinking_canvas.coords(self._thinking_circle, x0, y0, x1, y1)
        except Exception:
            pass
        try:
            self.after(220, self._animate_thinking)
        except Exception:
            pass

    def _poll(self):
        base = self._get_base_url()
        n = self._safe_int(self.lines_var.get(), 2000)
        if n < 50:
            n = 50
        if n > 8000:
            n = 8000

        keep_bottom = True
        try:
            y0, y1 = self.text.yview()
            keep_bottom = float(y1) >= 0.98
        except Exception:
            keep_bottom = True

        prev_state = self._last_state

        try:
            qs = urlencode({"n": str(n)})
            data = _http_get_json(f"{base}/overlay_logs?{qs}")
            logs = data.get("logs")
            state = data.get("state")
            if not isinstance(logs, list):
                logs = []
            text = "\n".join([str(x) for x in logs][-n:])

            now = time.time()
            self._last_ok_time = now

            status = str(state or "unknown")
            self.status_var.set(status)
            try:
                self._update_status_style(status)
            except Exception:
                pass

            text_changed = text != self._last_text
            if text != self._last_text:
                self._last_text = text
                self._set_text(text, keep_scrolled_to_bottom=keep_bottom)

            # Update rotating activity label + thinking animation
            try:
                # Reset phase index on fresh run transition
                if (prev_state or "").strip().lower() != "running" and str(status or "").strip().lower() == "running":
                    self._phase_index = -1
                self._update_activity(status, text_changed=text_changed)
            except Exception:
                pass

            self._last_state = status

            # Also poll login options so the panel can offer approvals like UI1
            try:
                login_data = _http_get_json(f"{base}/overlay_login_options")
            except Exception:
                login_data = None
            try:
                if isinstance(login_data, dict):
                    opts = login_data.get("options") or []
                    url_val = login_data.get("url") or ""
                    self._update_login_ui(opts, str(url_val))
            except Exception:
                pass

        except (URLError, ValueError, TimeoutError, OSError):
            now = time.time()
            if self._last_ok_time and (now - self._last_ok_time) < 5:
                if self._last_state:
                    self.status_var.set(f"{self._last_state} (no update)")
            else:
                self.status_var.set("Disconnected")
            try:
                self._update_status_style("disconnected")
            except Exception:
                pass

        self.after(350, self._poll)

    def _update_status_style(self, status: str):
        s = str(status or "").strip().lower()
        bg = THEME_BORDER
        fg = THEME_MUTED
        if s in {"running"}:
            bg = "#064e3b"
            fg = "#d1fae5"
        elif s in {"paused"}:
            bg = "#7c2d12"
            fg = "#ffedd5"
        elif s in {"error"}:
            bg = "#7f1d1d"
            fg = "#fee2e2"
        elif s in {"idle", "completed", "stopped"}:
            bg = "#1e293b"
            fg = "#e2e8f0"
        elif s in {"disconnected"}:
            bg = "#111827"
            fg = "#fca5a5"
        try:
            self.status_label.configure(bg=bg, fg=fg)
        except Exception:
            pass

        try:
            btn = getattr(self, "pause_resume_btn", None)
        except Exception:
            btn = None
        if btn is not None:
            try:
                if s in {"running"}:
                    btn.configure(text="Pause", state=tk.NORMAL, bg=THEME_AMBER, activebackground=THEME_AMBER)
                elif s in {"paused"}:
                    btn.configure(text="Resume", state=tk.NORMAL, bg=THEME_BLUE, activebackground=THEME_BLUE)
                else:
                    btn.configure(text="Pause", state=tk.DISABLED, bg=THEME_BTN, activebackground=THEME_BTN_HOVER)
            except Exception:
                pass


def main():
    app = FloatingPanel()
    app.mainloop()


if __name__ == "__main__":
    main()
