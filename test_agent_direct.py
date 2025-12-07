import sys
import os
from pathlib import Path

# Add apps/backend to sys.path
sys.path.append(str(Path("apps/backend").resolve()))

# Attempt to load .env manually if pydantic doesn't pick it up from root
from dotenv import load_dotenv
load_dotenv()

try:
    from app.services.llm_parsing import parse_do_anything_request
    from app.services.playwright_do_anything import run_playwright_do_anything
    from app.core.config import get_settings
except ImportError as e:
    print(f"ImportError: {e}")
    print("sys.path:", sys.path)
    sys.exit(1)

def main():
    settings = get_settings()
    if not settings.groq_api_key:
        print("Error: GROQ_API_KEY not found in .env")
        # Try to read from env var directly
        if "GROQ_API_KEY" not in os.environ:
             print("Please set GROQ_API_KEY in .env")
             return
    
    print("Enter a request for the agent (e.g. 'buy the cheapest iPhone on Amazon.in')")
    # Flush stdout to ensure prompt is visible
    sys.stdout.flush()
    text = input("Request: ")
    
    print(f"Parsing request: {text}")
    plan = parse_do_anything_request(text)
    print(f"Plan: {plan}")
    
    if not plan.get("site"):
        print("Could not plan the task.")
        return

    print("Running Playwright...")
    # Ensure we have a valid user_id and confirm=False for safety
    result = run_playwright_do_anything(
        site=plan["site"],
        steps=plan["steps"],
        user_id="test-user",
        text=text,
        use_cached_path=False,
        confirm=False 
    )
    
    print("Result:", result)

if __name__ == "__main__":
    main()
