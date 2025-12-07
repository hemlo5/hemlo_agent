HEMLO SUPER AGENT — IT DOES THE THING FOR YOU

Most AI assistants give you steps and links. Hemlo actually does the steps for you.
It opens the site, finds the right controls, clicks and types, captures downloads, asks before money actions, and stops when the job is done.

Why Hemlo is different
- Typical AI: gives you instructions, links, and maybe screenshots.
- Hemlo: operates a real browser with Playwright, plans with Groq, handles tabs/popups, detects downloads, and saves proof.

Examples completed end-to-end
- “Download a white Ferrari Novitec 812 N-LARGO wallpaper” → chooses a bot‑friendly site, clicks Download/Save, saves to your downloads folder, and stops.
- “Refund my last Stripe charge for ₹500” → opens Stripe Dashboard or uses the Stripe API (if configured), asks approval, executes the refund, and stores proof.
- “Order paneer malai from a nearby store” → navigates an e‑commerce site, avoids product‑tile loops, adds to cart, prefers Cart/Checkout, pauses on login gates.
- “Search YouTube for ‘best tool cover’” → types once, clicks a result, and plays it without retyping.
- “Fill KYC on my bank portal” → navigates forms and pauses for OTP/login if required.

Speak or type a goal (“download a white dog photo”, “refund my last Stripe charge”, “order paneer malai”), and Hemlo’s Super Agent plans, navigates the web or APIs, clicks the right buttons, saves files, asks for money-action approval, and stops when the job is done.

TABLE OF CONTENTS
1) High-level overview
2) Repo map: what lives where
3) External APIs, libraries, and versions
4) Environment variables
5) Hemlo Super Agent brain (hemlo_super_agent.py)
6) Browser skills: navigation, tabs/popups, downloads
7) Loop-avoidance, safety, and logging
8) Backend agents: FastAPI + Celery flows (Stripe, Amazon, Do-Anything)
9) How to run (Windows/PowerShell)
10) Troubleshooting
11) Roadmap (super fixes)

------------------------------------------------------------
1) HIGH-LEVEL OVERVIEW
------------------------------------------------------------
- Planning and Reasoning: Groq LLM (OpenAI-compatible SDK) converts a natural-language goal into next-step decisions. 
- Hands & Eyes: Playwright automates a visible Chromium browser. The agent reads the Accessibility (AX) tree to extract interactive elements deterministically (no screenshot OCR), ranks them by relevance, and clicks/types.
- Proofs & Files: The agent writes debug logs and snapshots; for downloads it saves files to a persistent folder and stops immediately.
- Safety: Any action that could spend money requires explicit approval in the console.
- Backend (optional stack): FastAPI + Celery expose production-style endpoints (Stripe refund, Amazon order from cart, generic Do-Anything), with Redis-backed path caching and proof screenshots.

------------------------------------------------------------
2) REPO MAP — WHAT LIVES WHERE
------------------------------------------------------------
- hemlo_super_agent.py
  Standalone “super agent” that opens Chromium and performs arbitrary web tasks with Groq + Playwright.
  Key features: URL planning (explicit → Serper.dev → LLM), deterministic DOM extraction + ranking, LLM ReAct loop, robust click/navigation, popup/tab handling, download capture (stop-on-success), loop-avoidance and fallback exploration.

- apps/backend/app/main.py
  FastAPI endpoints for:
  • /agents/stripe-refund (create + status)
  • /agents/amazon-order-cart (create + status)
  • /agents/do-anything (create + status)
  Also serves static /proofs for screenshots.

- apps/backend/app/worker.py (Celery)
  Task runners that call Playwright flows or Stripe API. Includes Redis-backed path caching so second runs are faster.

- apps/backend/app/services/
  • llm_parsing.py: Groq LLM parsers for Stripe refund and generic plan (site + steps)
  • playwright_*: Concrete Playwright flows (Stripe refunds, Amazon cart, do-anything)
  • stripe_refunds.py: Direct Stripe REST fallback
  • path_cache.py, otp_store.py: Redis utilities
  • firebase_streaming.py: Optional screenshot streaming stub via Firebase
  • skyvern_client.py, skyvern_refund_client.py: Optional delegated automation clients
  • user_profiles.py: Qdrant-backed encrypted user profile store (consumer_id/phone/email) used by Skyvern demo flows

- demo.py
  CLI demo runner for Amazon, Stripe, and the generic agent via FastAPI.

- apps/backend/requirements.txt
  Pinned backend dependencies (see section 3).

- Logs and artifacts (top-level, created at runtime)
  • filtered_dom_log.jsonl  – JSONL history of the super agent’s filtered elements per step
  • last_filtered_dom.json  – snapshot of the most recent filtered DOM (what the LLM saw)
  • agent_thoughts.txt      – per-step decisions for debugging
  • proofs/                 – backend demo screenshots

------------------------------------------------------------
3) EXTERNAL APIs, LIBRARIES, AND VERSIONS
------------------------------------------------------------
Brain & Planning
- Groq LLM via OpenAI Python SDK
  • Super Agent model: "llama-3.3-70b-versatile" (hemlo_super_agent.py)
  • Backend default:     "llama-3.1-70b" (config)
  • SDK: openai==1.40.0 (backend). Super Agent works with 1.40.x+

URL Resolution (bot-friendly search)
- Serper.dev (Google Search API) – optional, reduces CAPTCHA risk when guessing sites

Automation
- Playwright (Chromium) – latest stable

Backend Core (apps/backend/requirements.txt)
- fastapi==0.115.0, uvicorn[standard]==0.30.1
- celery==5.3.6, redis==5.0.1
- httpx==0.27.0
- pydantic==2.8.2, pydantic-settings==2.4.0
- python-dotenv==1.0.1
- openai==1.40.0
- qdrant-client==1.7.3, chromadb==0.5.3
- firebase-admin==6.5.0
- playwright (un-pinned; use latest)

------------------------------------------------------------
4) ENVIRONMENT VARIABLES
------------------------------------------------------------
Required / Recommended
- GROQ_API_KEY            – Groq API key for LLM reasoning (all agents)
- SERPER_API_KEY          – Serper.dev search (optional but recommended for URL planning)
- HEMLO_DOWNLOAD_DIR      – Directory where Super Agent saves files (default: ~/Downloads/hemlo_agent)

Backend (optional stack)
- STRIPE_SECRET_KEY       – Enables API fallback for Stripe refunds
- REDIS/QDRANT/CHROMADB   – See apps/backend/app/core/config.py defaults
- FIREBASE_PROJECT_ID     – Optional for firebase_streaming.py

------------------------------------------------------------
5) HEMLO SUPER AGENT BRAIN (hemlo_super_agent.py)
------------------------------------------------------------
Flow summary
A) Plan initial URL
   1. If the prompt contains an explicit https:// URL, use it.
   2. Else query Serper.dev and prefer the Knowledge Graph website or first organic link.
   3. As a last resort, ask the LLM to return ONLY a URL (fallbacks to google.com if none).

B) Deterministic DOM extraction (no screenshot OCR)
   • Use Playwright’s Accessibility (AX) snapshot.
   • Collect interactive nodes with role + name. For each element, enrich with:
     - href, href_kind (nav/anchor/js/other), target (e.g., _blank), aria-expanded
   • Deduplicate by (role, name).

C) Ranking (relevance score)
   • Boost:
     - goal-token matches in name
     - useful controls (searchbox/textbox, real navigation links)
     - money actions marked internally
     - for download goals: names containing download/save/jpg/png/jpeg
   • Penalize anchors/js and off-domain links (prefer same-domain).

D) LLM ReAct decision
   • Send a trimmed JSON list of top elements and a compact context (URL, recent actions, blocked choices, downloads this run).
   • System rules force: one of {click,type,hover,wait,done,fail}; prefer role selectors; avoid repeats; return done if goal content is visible; do not re-type the same query.
   • Retries: Exponential backoff on 429s; if still failing, heuristic fallback selects a safe nav item.

E) Execute action (robust clicking & typing)
   • Money safety: if an action’s label looks like buy/checkout/pay/add to cart, prompt for console approval before proceeding.
   • Resolve target/href from element and from the filtered list metadata.
   • External/_blank links: prefer same-tab navigation via page.goto(href); fallback to expect_popup() capture.
   • After actions, wait briefly for domcontentloaded/networkidle (short timeouts), not always full network quiescence.

F) Tabs/pages switching
   • After each action, compare context.pages and detect newly opened pages.
   • Score pages for the goal (URL/title tokens; extra boost if "download" appears; hosters like mediafire/drive/mega add points).
   • Switch to the best page when it’s more relevant.

G) Download capture (stop-on-success)
   • For suspected downloads: wrap clicks/gotos in page.context.expect_download(...), save to HEMLO_DOWNLOAD_DIR, append to downloaded_files.
   • If a site streams images (no Download event), save the largest visible <img> via direct HTTP.
   • As soon as any file is saved for download-like goals, print the path and stop the loop.

H) Loop-avoidance & fallback exploration
   • Maintain blocked choices set: if (role,name) repeated and URL+DOM-hash unchanged, block and try alternatives.
   • Fallback picks next best nav link or exploration keyword (apply/learn more/etc.).

I) Heuristics
   • YouTube: if on youtube.com and a search box exists, type once and press Enter; then click the first video result; avoid retyping.
   • Quick-download pass: at the top of each step when the goal includes download/save, try obvious Download/Save controls first.

J) Stop conditions
   • LLM returns action==done
   • A file is saved (for download intents)
   • Max steps reached (default 20; reduced to ~12 for download intents)

K) Logging & observability
   • filtered_dom_log.jsonl – append one JSON line per step
   • last_filtered_dom.json – overwrite with the current step’s filtered list
   • agent_thoughts.txt – record the decision JSON and a few notes for debugging

------------------------------------------------------------
6) BROWSER SKILLS: NAVIGATION, TABS/POPUPS, DOWNLOADS
------------------------------------------------------------
- Same-tab over new tabs: If an element has target="_blank" or goes off-domain, navigate directly via page.goto to avoid losing control. If a popup still opens, expect_popup() is used to capture it and optionally switch to it.
- Page scoring & switching: New pages are scored using URL/title token matches and goal-specific hints. The agent brings the best page to the front.
- Downloads:
  • Browser context created with accept_downloads=True
  • expect_download wraps suspected download actions, then download.save_as(out)
  • Largest-image fallback saves via HTTP when there’s no browser download event
  • After saving, the agent prints the final path and stops

------------------------------------------------------------
7) LOOP-AVOIDANCE, SAFETY, LOGGING
------------------------------------------------------------
- Loop-avoidance: block repeated (role,name) when there’s no URL/DOM change; explore alternatives from the ranked list
- Money-action approval: console prompt for buy/checkout/pay/add-to-cart; the action is skipped if not approved
- Compact console logs: only a short decision summary is printed; full lists go to JSON files (not the console)

Artifacts (saved next to the script)
- filtered_dom_log.jsonl, last_filtered_dom.json, agent_thoughts.txt
- Downloads are saved to HEMLO_DOWNLOAD_DIR (default: ~/Downloads/hemlo_agent)

------------------------------------------------------------
8) BACKEND AGENTS (FASTAPI + CELERY)
------------------------------------------------------------
Endpoints (apps/backend/app/main.py)
- /agents/stripe-refund: Parse user text to refund params (LLM), then either drive Stripe Dashboard via Playwright (preview → approval → confirm) or call Stripe API if configured.
- /agents/amazon-order-cart: From Amazon Cart, proceed to checkout; stops in preview unless confirm=true, detects login if required.
- /agents/do-anything: Parse text to {site, steps}, open persistent Chromium context, apply simple heuristics, screenshot each step; returns proof.

Task runners (apps/backend/app/worker.py)
- Celery tasks submit Playwright flows and implement path caching using Redis (path_cache.py). Successful flows mark paths cached → repeat runs get faster and more deterministic.

Selected services
- playwright_stripe_refunds.py, playwright_amazon_order.py, playwright_do_anything.py: Concretize flows; include login detection and proof screenshots.
- stripe_refunds.py: Direct REST refund when STRIPE_SECRET_KEY is available.
- firebase_streaming.py: Optional stub to publish last screenshot to Firebase for live preview.
- skyvern_client.py / skyvern_refund_client.py: Optional delegated automation agents (pay bill, refund negotiation) with polling and proof URL extraction.
- user_profiles.py: Qdrant-backed encrypted user profile storage (simple XOR+base64 obfuscation keyed by Settings.secret_key).

------------------------------------------------------------
9) HOW TO RUN (WINDOWS POWERSHELL) — SUPER AGENT ONLY
------------------------------------------------------------
1) Create venv and install
   python -m venv .venv
   .venv\Scripts\Activate
   pip install playwright python-dotenv openai
   python -m playwright install chromium

2) Set env vars
   $env:GROQ_API_KEY = "your_groq_key"
   # Optional but recommended for URL planning
   $env:SERPER_API_KEY = "your_serper_key"
   # Optional download folder (default is ~/Downloads/hemlo_agent)
   $env:HEMLO_DOWNLOAD_DIR = "C:\Users\<you>\Downloads\hemlo_agent"

3) Run
   python hemlo_super_agent.py --prompt "download a white dog picture"
   python hemlo_super_agent.py --prompt "order paneer malai from a local grocery site"

------------------------------------------------------------
10) TROUBLESHOOTING
------------------------------------------------------------
- It keeps clicking but nothing changes
  • The agent blocks repeated choices when URL/DOM doesn’t change and explores alternatives; if a site uses heavy overlays, try closing modals manually or re-run with a clearer goal (e.g., include the target site).

- It got stuck on Shutterstock/paywalled images
  • Prefer Unsplash/Pexels in your prompt (or use Serper API key) to bias to bot-friendly sites.

- It didn’t stop after downloading
  • The agent captures downloads via expect_download and stops; some sites stream images instead of emitting a download—largest-image fallback now saves via HTTP and stops immediately.

- 429 rate limits from Groq
  • The agent retries with short backoff; for long tasks consider simplifying the goal, or add SERPER_API_KEY to reduce token-heavy planning.

- Failing selectors
  • The agent uses role-based locators from the AX tree with minimal attributes; if a site is too dynamic, add an explicit site in the prompt or re-run.

------------------------------------------------------------
11) ROADMAP (SUPER FIXES)
------------------------------------------------------------
Implement next (highest impact first):
- State machine by task type
  • Classify goals into download / ecommerce / form/apply / watch/read / generic
  • Switch scoring, heuristics, and stop conditions per type

- Commerce flow heuristics
  • Track cart_added; after add-to-cart, prefer Cart/View cart/Checkout
  • Penalize “Current price:” product tiles (prevents Instacart loops)
  • Cache approval for “Add to cart” within a run

- Login/CAPTCHA gate handling
  • Detect sign in/log in/SSO/email/phone/password forms; pause and prompt user to authenticate; resume or stop cleanly

- Stronger scoring & visibility
  • Weight viewport visibility/area; downrank anchors/js/off-domain unless needed

- Token/cost optimizations
  • Trim to top 20–30 items with minimal fields; cache decisions for same URL+DOM hash
  • Use a smaller model for step selection; reserve larger model for planning/dead-ends

- Network/runtime tuning
  • Block common analytics/ads via route intercepts; prefer domcontentloaded waits; only use networkidle on real navigations

Done already (highlights):
- Deterministic AX DOM filter with href/target/aria, ranking
- Same-tab navigation for _blank/external, popup capture & page switching
- Robust download capture (expect_download) + largest-image fallback + stop-on-success
- Loop-avoidance via blocked choices and fallback exploration
- Compact console logs; JSONL history and last step snapshot

END OF README
