# Hemlo AI Monorepo

The fast, private AI agent that actually does your daily work for you.

Hemlo doesn’t just *tell* you what to do – it actually **does the task end‑to‑end** using LLM reasoning + API / browser automation.

---

## Core Promise

> User says anything in English (US/EU focus) or Hindi (India v2) → Hemlo does it completely hands‑free, fast, and private.

Example MVP flows for v1 (US/EU, API‑first):

- **Refund Stripe charge** (e.g. "Refund my $20 Stripe charge")
- **Twilio alerts** (e.g. "Send me a text if TSLA drops below $200")
- **LinkedIn job apply** (e.g. "Apply to 20 remote SWE jobs in EU")
- **Slack auto‑reply** (e.g. "Auto‑reply to all DMs after 7pm")
- **Mint / overdraft hygiene** (e.g. "Track my Mint budget and pay any overdraft")

---

## Tech Stack (Fixed for MVP – November 2025)

- **LLM Brain**
  - Llama 3.1 70B via **Groq** (primary, fast/cheap)
  - **Grok‑4** API as verification / fallback only for critical actions
- **Hands & Eyes**
  - **Playwright (latest)** on **Chromium** – DOM parsing + clicking
  - Screenshots only for **proof**, not for core reasoning
- **Agent Framework**
  - **CrewAI** (latest)
  - **LangGraph** – long‑running / stateful flows
- **Voice → Text**
  - **Faster‑Whisper Large‑v3** (local)
- **Memory**
  - **ChromaDB** (vector)
  - **Qdrant** (hybrid search)
- **Backend**
  - **FastAPI** (Python)
  - **Celery** (background tasks)
- **Mobile App**
  - **Flutter 3.24** (Android first, iOS later)
  - **Riverpod 2.x**
  - **Firebase Auth & Firestore**
- **Infra / Hosting**
  - **Railway.app** (free tier) for backend + workers
  - **Groq** for LLM inference
- **Android Integration**
  - Android **Accessibility Service** for OTP reading (mandatory)
  - Live screenshot streaming via **Firebase Storage** → Flutter client (proof only)

---

## Monorepo Structure

```text
hemlo-ai/
  README.md
  progress.txt
  docker-compose.yml
  .env.example
  apps/
    backend/              # FastAPI + Celery + agents (CrewAI, LangGraph, Playwright, APIs)
      app/
        ...               # main.py, worker.py, services/, etc.
    mobile/               # Flutter 3.24 app (Android first)
      android/            # Native Android project (Accessibility, etc.)
      lib/                # Flutter Dart code
```

This layout keeps **backend** and **mobile** concerns cleanly separated while still deployable as a single unit.

---

## High-Level Architecture (Mermaid)

```mermaid
flowchart TD
    U[User\n(English)] -->|Voice| STT[Faster-Whisper\nLarge-v3]
    U -->|Text| APP[Flutter 3.24 App]

    STT -->|Text| APP
    APP -->|Request + context| API[FastAPI Backend]

    subgraph Brain[LLM Brain + Orchestration]
        API --> LLM[Llama 3.1 70B via Groq]
        LLM --> ROUTER[Agent Router\n(CrewAI + LangGraph)]
        ROUTER --> MEMS[Memory Layer\n(ChromaDB + Qdrant)]
        MEMS --> LLM
        ROUTER --> CHECKAPI{Stable API exists?}
        CHECKAPI -->|Yes| APIS[Stripe, Twilio,\nLinkedIn, HubSpot,\nSlack, Google Calendar, etc.]
        CHECKAPI -->|No| PW[Playwright on Chromium]
        LLM --> GROK[Grok-4\nverification]
        GROK --> ROUTER
    end

    subgraph Browser[Hands & Eyes]
        PW --> DOM[Target Web Apps\n(DOM parsing + clicks)]
        PW --> FBSTRM[Firebase Storage\n(Proof screenshots only)]
    end

    APIS --> PROOF[Receipts / Proofs\n(IDs, JSON, screenshots)]
    DOM --> PROOF

    PROOF --> MEMS
    PROOF --> API

    FBSTRM --> APP
    API --> APP

    subgraph Android[Android Device]
        OTP[Accessibility Service\n(OTP reader)] --> API
    end

    APP -->|Approve / Deny| API
```

---

## Mandatory Internal Flow (Always in This Order)

1. **User speaks/types request** (English) in the Flutter app.
2. **Faster‑Whisper** converts audio → text (on device or backend service).
3. **Llama 3.1 70B (Groq)** parses intent, extracts entities/parameters, and proposes a plan.
4. **Router (CrewAI + LangGraph)** decides:
   - If a **stable API** exists (Stripe, Twilio, HubSpot, LinkedIn, Slack, Google Calendar, Mint, etc.) → call the API directly.
   - Otherwise → use **Playwright on Chromium** to drive the browser via DOM parsing (no screenshot‑based reasoning).
5. If Playwright is used:
   - On the **first run**, it explores the site, identifies reliable DOM selectors, and caches the **path + selectors** for that flow.
   - On **subsequent runs**, it reuses the cached path to reach the goal in <4 seconds where possible.
   - It captures **proof screenshots** (e.g. refund confirmation) and sends them every 1s to Firebase Storage for the mobile app.
6. Before any money leaves the user’s wallet:
   - The app shows an **“Approve?” popup** with key details and proof.
7. After successful completion of the task:
   - Hemlo saves **receipt / screenshot / proof** in memory (ChromaDB + Qdrant).
   - The app responds to the user with **“Done!”** in English and shows proof.

---

## Local Development – Quick Start

> Goal: Bring up the **entire backend stack** (FastAPI, Celery, Redis, Qdrant, ChromaDB, Playwright‑ready backend) with **one command**.

### 1. Prerequisites

- **Docker** + **Docker Compose** installed and running
- **Git** (optional but recommended)
- For mobile + native dev (later steps):
  - **Flutter 3.24** SDK installed
  - Android Studio / SDK + device or emulator

### 2. Clone the Repository

```bash
git clone <your-repo-url> hemlo-ai
cd hemlo-ai
```

### 3. Run the Full Dev Stack (One Command)

From the monorepo root:

```bash
docker compose --profile dev up --build
```

This will (once `docker-compose.yml` is updated for the new stack):

- **FastAPI backend** (apps/backend)
- **Celery worker(s)**
- **Redis** (broker + cache)
- **Qdrant** (hybrid vector DB)
- **ChromaDB** (vector memory)

All wired together so the mobile app (and later web clients / CLI) can talk to a single backend endpoint.

---

## Apps

### Backend (apps/backend)

- FastAPI project for:
  - HTTP API consumed by Flutter app & other clients
  - Auth and user/session management
  - LLM calls to Groq (Llama 3.1 70B primary) and Grok‑4 for verification
  - Task orchestration with CrewAI + LangGraph
  - Routing between **direct APIs** (Stripe, Twilio, LinkedIn, HubSpot, Slack, Google Calendar, Mint, etc.) and **Playwright flows**
  - Exposing task status, live screenshot/proof URLs, and receipts
- Celery workers for long‑running / high‑latency operations (refunds, LinkedIn job applications, etc.).

> Implementation details will be added progressively in later steps.

### Mobile App (apps/mobile)

- Flutter 3.24 app (Android first) using Riverpod 2.x and Firebase Auth/Firestore.
- Key features for MVP:
  - Big mic button (voice input)
  - Live video feed of what Hemlo is doing (screenshots as a stream)
  - Approve / Deny action for sensitive ops
  - Chat‑style history with proofs / receipts.

Flutter project scaffolding will be added in a dedicated step so it can be created via `flutter create` with the right options.

---

## Infra & Services

- Root‑level `docker-compose.yml` and `.env` for local and Railway.app deployments.
- Qdrant and ChromaDB configured as separate services for memory.

---

## Development Philosophy

- **Do, don’t just say** – Hemlo must actually complete flows, not just generate instructions.
- **Stable APIs first**, Playwright automation second.
- **Path caching** to make repeat actions extremely fast (<4s for second run of the same flow).
- **Proof for every action** – screenshots, receipts, transaction IDs.
- **Human‑in‑the‑loop for money** – always ask before spending.

---

## Run the viral demo in 1 command

If you just want a **simple, visible demo** for recording (no Docker, no Celery, no Redis):

1. From the repo root, ensure dependencies are installed for the backend env (once):

   ```bash
   cd apps/backend
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\\Scripts\\activate
   pip install -r requirements.txt
   playwright install chromium
   ```

2. Set your Groq API key (for Llama 70B parsing, optional but recommended):

   ```bash
   export GROQ_API_KEY=your_groq_key_here
   # On Windows PowerShell:
   #   $env:GROQ_API_KEY = "your_groq_key_here"
   ```

3. From the **repo root**, run the one‑file demo:

   ```bash
   python demo.py
   ```

4. Type `amazon` or `stripe` when prompted.

   - `amazon` → opens a visible Chromium window on your Amazon cart, reuses cookies, clicks through checkout, and saves a final screenshot to `./proofs/amazon_final.png`.
   - `stripe` → asks you for a natural‑language description (e.g. “refund $20 for the last charge”), parses it with Llama 3.1 70B via Groq, opens Stripe Dashboard in a visible Chromium window, helps you navigate to the refund UI, and saves a final screenshot to `./proofs/stripe_refund_final.png`.

All screenshots from this quick demo are written to the root‑level `./proofs` folder, ready to be dropped into a Loom or OBS recording.

---

## Dynamic "Do Anything" Demo

The backend also exposes a **generic agent** that can plan and execute flows on almost any site using Groq + Playwright.

- **Endpoint:** `POST /agents/do-anything`
- **Body:**

  ```json
  { "text": "buy the cheapest iPhone on Amazon.in", "confirm": false }
  ```

  Llama 3.1 70B (via Groq) parses this into a JSON plan like:

  ```json
  { "site": "amazon.in", "steps": ["search iPhone", "sort by price", "add to cart", "checkout"] }
  ```

- The worker then opens the target site in Chromium (with persistent cookies per site/user), walks through the steps using DOM locators, and writes sequential screenshots into `./apps/backend/proofs` (or `./proofs` when using Docker volume mounts).
- If it detects a login page it returns `status: "login_required"`.
- For money/checkout steps it returns `status: "needs_approval"` with a preview screenshot instead of clicking the final button until you resend with `confirm: true`.

### Quick curl example

With the full stack running on `http://localhost:8000`:

```bash
curl -X POST http://localhost:8000/agents/do-anything \
  -H "Content-Type: application/json" \
  -d '{
    "text": "buy the cheapest iPhone on Amazon.in",
    "confirm": false
  }'
```

Response:

```json
{ "task_id": "<id>", "status": "queued" }
```

Poll status:

```bash
curl http://localhost:8000/agents/do-anything/<id>
```

You will see `running` → then one of `login_required`, `needs_approval`, `succeeded`, or `failed`. On `needs_approval` / `succeeded`, `result.proof_url` points at the key screenshot.

### demo.py: agent mode

If the backend + worker are running locally, you can also drive the dynamic agent from the one‑file demo:

```bash
python demo.py
```

When prompted, type `agent`, then describe **any** task, e.g.:

```text
Describe what Hemlo should do (any site, any task): buy the cheapest iPhone on Amazon.in
Auto-approve money actions? (y/N): y
```

`demo.py` will call `/agents/do-anything`, poll status, and print the final site, planned steps, and proof URL.

---

## Run the viral demos in 2 minutes

> These steps assume Docker Desktop is running and you have a Groq API key for LLM parsing. Stripe and Amazon flows use **Playwright + Chromium** with persistent cookies and proof screenshots served at `http://localhost:8000/proofs/...`.

### 1. One‑time setup

```bash
cd hemlo-ai
cp .env.example .env
```

Edit `.env` and set at least:

- `GROQ_API_KEY=<your-groq-key>`
- (Optional but recommended) `STRIPE_SECRET_KEY=<your-stripe-secret-key>` for API fallback.

### 2. Run full stack (backend + worker + infra)

```bash
docker compose --profile dev up --build
```

This starts:

- FastAPI backend on `http://localhost:8000`
- Celery worker (Stripe + Amazon agents)
- Redis, Qdrant, ChromaDB
- Playwright‑enabled Chromium inside backend/worker
- A shared `./proofs` folder on your host, mounted into containers at `/app/proofs`

### 3. Stripe refund demo (Playwright first, API fallback)

1. **Kick off a refund task (no confirm yet – preview only):**

```bash
curl -X POST http://localhost:8000/agents/stripe-refund \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Refund my last $20 Stripe charge",
    "use_api": false,
    "confirm": false
  }'
```

Response:

```json
{ "task_id": "<id>", "status": "queued" }
```

2. **Poll task status:**

```bash
curl http://localhost:8000/agents/stripe-refund/<task_id>
```

While running:

```json
{ "status": "running", "message": "Processing Stripe refund", "result": null }
```

If cookies are missing, you may see:

```json
{ "status": "login_required", "message": "User must log in to Stripe dashboard once so cookies are saved.", "result": null }
```

For the video demo you can ensure your Stripe dashboard is already logged‑in inside the Playwright context (or use a test account/environment without 2FA).

If Playwright reaches the refund confirmation page with `confirm=false`, you get:

```json
{
  "status": "needs_approval",
  "message": "Something went wrong",  // generic wrapper message
  "result": {
    "status": "needs_approval",
    "charge_id": "...",
    "amount_cents": 2000,
    "currency": "usd",
    "proof_url": "http://localhost:8000/proofs/stripe_refund_..._preview.png"
  }
}
```

Open the `proof_url` in a browser to see the preview screenshot.

3. **Approve and actually perform the refund:**

Re‑send the same POST but with `"confirm": true`:

```bash
curl -X POST http://localhost:8000/agents/stripe-refund \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Refund my last $20 Stripe charge",
    "use_api": false,
    "confirm": true
  }'
```

Then poll status again:

```bash
curl http://localhost:8000/agents/stripe-refund/<task_id>
```

On success you get:

```json
{
  "status": "succeeded",
  "message": "Done!",
  "result": {
    "status": "succeeded",
    "charge_id": "...",
    "refund_id": "...",
    "amount_cents": 2000,
    "currency": "usd",
    "proof_url": "http://localhost:8000/proofs/stripe_refund_..._final.png",
    "raw_result": { "id": "re_...", ... }
  }
}
```

You can also run pure‑API refunds by setting `"use_api": true`; the worker will still require `"confirm": true` before calling the Stripe API.

### 4. Amazon cart ordering demo

1. **Kick off an order from your cart (preview only):**

```bash
curl -X POST http://localhost:8000/agents/amazon-order-cart \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Order the most recent thing in my Amazon cart",
    "confirm": false
  }'
```

2. **Poll task status:**

```bash
curl http://localhost:8000/agents/amazon-order-cart/<task_id>
```

While running:

```json
{ "status": "running", "message": "Ordering from Amazon cart", "result": null }
```

First run may return:

```json
{ "status": "login_required", "message": "User must log in to Amazon once so cookies are saved.", "result": null }
```

For the demo, ensure your Amazon account is already logged‑in in the Playwright context before recording, or use a sandbox/test account.

When Playwright reaches the "Place your order" screen with `confirm=false`, you get:

```json
{
  "status": "needs_approval",
  "message": "Something went wrong",  // generic wrapper message
  "result": {
    "status": "needs_approval",
    "order_id": null,
    "proof_url": "http://localhost:8000/proofs/amazon_order_..._preview.png"
  }
}
```

Open the `proof_url` in a browser to show the “about to place order” screenshot.

3. **Approve and actually place the order:**

```bash
curl -X POST http://localhost:8000/agents/amazon-order-cart \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Order the most recent thing in my Amazon cart",
    "confirm": true
  }'
```

Poll status again:

```bash
curl http://localhost:8000/agents/amazon-order-cart/<task_id>
```

On success you get:

```json
{
  "status": "succeeded",
  "message": "Done!",
  "result": {
    "status": "succeeded",
    "order_id": "...",
    "proof_url": "http://localhost:8000/proofs/amazon_order_..._final.png",
    "raw_result": { ... }
  }
}
```

### 5. Where to see live screenshots

- All proof images are written into the shared `./proofs` folder on your host.
- During a task, multiple `*_stepX.png` files appear there (roughly every couple of seconds), plus a `_preview.png` and `_final.png`.
- You can:
  - Open `http://localhost:8000/proofs/<filename>.png` directly in a browser, or
  - Inspect the `./proofs` folder in your file explorer while recording the video.

---

## Super Agent (Groq + Playwright) � How it works

This repo also includes a standalone, visible browser �super agent� for arbitrary web tasks, driven by Groq + Playwright.

- Entry point: hemlo_super_agent.py
- Browser automation: Playwright Chromium (headful)
- Reasoning: Groq llama-3.3-70b-versatile via OpenAI-compatible client
- Bot-friendly URL resolve: Serper.dev (Google Search API) to avoid CAPTCHAs

### APIs, Libraries, and Env

- Groq LLM (OpenAI Python client): GROQ_API_KEY required
- Serper.dev Search API: SERPER_API_KEY optional but recommended (better URL planning, fewer CAPTCHAs)
- Playwright (Chromium), python-dotenv
- Optional downloads folder: HEMLO_DOWNLOAD_DIR (default: ~/Downloads/hemlo_agent)

### What goes in its �brain� (decision loop)

1) Deterministic DOM filter (no screenshot reasoning):
   - Uses Playwright Accessibility snapshot to harvest interactive elements.
   - Enriches each with href, href_kind (nav/anchor/js), 	arget (detect _blank), ria-expanded.
   - Dedupes by (role, name) and ranks by goal tokens, control type, domain preference, and download intent boosts.

2) Loop-avoidance and progress signals:
   - Tracks recent actions and blocks repeated (role,name) when URL and DOM-hash don�t change.
   - Fallback exploration picks a different, high-quality nav link.

3) LLM decision (ReAct step):
   - Sends a trimmed list (top-K) to Groq and asks for one next action:
     click | type | hover | wait | done | fail.
   - Retries with short backoff on 429 rate limits; heuristic fallback if LLM unavailable.

4) Heuristics before/after LLM:
   - YouTube search helper (don�t retype same query; click first result once).
   - Download-first fast path: obvious �Download/Save� buttons, plus a largest-<img> URL save fallback.
   - Token trims: compact console logs and capped list size to the LLM.

5) Safety:
   - Money actions (e.g., �Add to cart�, �Pay�) ask for explicit user approval in the console before proceeding.

### Tabs, popups, and navigation

- External or _blank links: when safe, open via page.goto(href) in the same tab to avoid losing focus.
- Popup expected: wrap clicks in page.expect_popup() to capture new pages.
- After each action: compare context.pages and pick the best page for the goal using URL/title/keyword scoring (e.g., boost pages with �download�).

### Downloads: captured and persisted (stop-on-success)

- Browser context created with ccept_downloads=True.
- Suspected download actions are wrapped in page.context.expect_download(...) and saved to HEMLO_DOWNLOAD_DIR.
- If a site streams images (no download event), the agent saves the largest on-screen <img> via direct HTTP.
- After any successful save, the agent prints the file path and stops the run.

### URL planning (no Google CAPTCHA)

Planning prefers:
1. Explicit URL in the prompt.
2. Serper.dev result (knowledgeGraph website or first organic link).
3. Groq LLM fallback as a last resort.

### Logs and artifacts

- iltered_dom_log.jsonl: JSONL of each step�s filtered DOM and timestamp.
- last_filtered_dom.json: snapshot of the most recent filtered DOM (what the LLM saw).
- gent_thoughts.txt: step-by-step thoughts (goal, filtered snippet path, decision JSON).
- Downloads: HEMLO_DOWNLOAD_DIR (default ~/Downloads/hemlo_agent).

### How to run the super agent locally

1) Create a virtual env and install deps (Windows PowerShell example):

`powershell
python -m venv .venv
.venv\Scripts\Activate
pip install playwright python-dotenv openai
python -m playwright install chromium
`

2) Set environment variables (PowerShell):

`powershell
 = "your_groq_key"
# Optional but recommended
 = "your_serper_key"
# Optional download folder
 = "C:\\Users\\<you>\\Downloads\\hemlo_agent"
`

3) Run the agent:

`powershell
python hemlo_super_agent.py --prompt "download a white dog picture"
python hemlo_super_agent.py --prompt "order paneer malai from a local grocery site"
`

### Troubleshooting

- Downloaded file not visible ? check HEMLO_DOWNLOAD_DIR (default ~/Downloads/hemlo_agent).
- Stuck on login/paywall ? log in manually and resume; login-gate heuristics are on the roadmap.
- Repeating clicks on tiles ? loop-avoidance helps; ecommerce heuristics to prefer Cart/Checkout are on the roadmap.
- Slow or rate-limited ? the agent backs off on 429s and trims LLM inputs; prefer bot-friendly sites for images (e.g., Unsplash/Pexels).

### Roadmap highlights

- Ecommerce flow state (prefer Cart/Checkout after add-to-cart; approval memory for �Add to cart�).
- Login/CAPTCHA gate detection with pause/resume.
- Stronger page/link scoring (viewport visibility, penalties for anchors/js, product tiles).
- Token/cost cuts: smaller model for step selection, caching by URL+DOM hash.

