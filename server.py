#!/usr/bin/env python3
"""ClearEar local backend.

Serves the static site and proxies AI requests to Gemini. Keys are loaded
from .env and rotated round-robin per request so the load spreads evenly
across both keys. If one key fails, the request automatically retries the
remaining keys.

The assistant has tool/function calling enabled so it can change scenes,
adjust EQ bands, and toggle controls directly from natural language.
"""

from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote, unquote, urlparse
from urllib.request import Request, urlopen
import json
import os
import ssl
import threading

try:
    import certifi
except ImportError:
    certifi = None

ROOT = Path(__file__).resolve().parent
MAX_BODY_SIZE = 64 * 1024
DEFAULT_MODEL = "gemini-2.5-flash"
SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where()) if certifi else None

SYSTEM_PROMPT_TEXT = """You are the AI assistant inside ClearEar, a software-defined hearing aid app that turns ordinary earbuds into smart hearing aids through phone-side audio processing.

ClearEar features the user can access through you:
- Scene presets (use these IDs in switch_scene): quiet, library, office, lecture, classroom, restaurant, crowded, commute, outdoor, windy, phone, music, gym, theater, tv, sleep
- 8-band frequency EQ: indices 0..7 map to 60Hz, 170Hz, 310Hz, 600Hz, 1kHz, 3kHz, 6kHz, 12kHz. Each band is adjustable from -12 to +12 dB.
- Three toggle controls: cancel (noise cancellation), speech (speech focus), transparency (passthrough)
- Live captions, sound exposure journal, hearing profiles

YOUR JOB:
1. Be a fully general assistant. Answer absolutely any question the user asks, from code to history to math to recipes to philosophy. Never refuse to answer just because it isn't about sound.
2. When the user asks you to change ClearEar (switch scenes, tweak EQ, toggle controls), CALL THE TOOL to actually do it. Do not just describe what you would do. The tool call will perform the action in the app.
3. Always end your reply with a short follow-up question related to sound, hearing, noise, audio, music, the acoustic environment, or how ClearEar is performing. This keeps the conversation engaging and grounded in the product.

STYLE:
- Conversational and warm, not robotic.
- Short answers: 1 to 3 sentences for the response, then the follow-up question.
- No bullet points or headers in chat replies.
- No em dashes. Use commas or periods.
- When you use a tool, briefly confirm the change in your reply. Do not narrate the tool itself.

EXAMPLES:

User: What's the capital of France?
You: Paris. Speaking of cities, are you somewhere noisy right now? I can switch scenes if it helps.

User: I just walked into a coffee shop.
You: [calls switch_scene with scene_id=crowded] Switched you to Crowded place with speech focus maxed. How clear are the voices around you?

User: Boost the highs.
You: [calls adjust_eq with band_index=6, gain_db=4] Bumped 6kHz by 4 dB. Do consonants feel sharper now?

User: Write me a haiku about pizza.
You: Cheese melts, crust grows crisp, pepperoni curls like waves, stomach hums its song. What sounds make pizza nights feel right to you?

User: Turn off transparency.
You: [calls toggle_control with control=transparency, enabled=false] Done, transparency is off. Does the room feel more isolated?"""

VOICE_RESPONSE_PROMPT = "\n\nThis request came from the in-ear voice assistant. Reply for audio playback, not screen reading. Be more concise than chat: one short sentence is best, two short sentences maximum. Avoid long lists, examples, citations, and filler."

SCENE_IDS = [
    "quiet", "library", "office", "lecture", "classroom", "restaurant",
    "crowded", "commute", "outdoor", "windy", "phone", "music",
    "gym", "theater", "tv", "sleep"
]

TOOLS = [{
    "function_declarations": [
        {
            "name": "switch_scene",
            "description": "Switch ClearEar to a different scene preset. Each preset has its own noise cancellation, speech focus, transparency, and beam settings tuned for a specific environment.",
            "parameters": {
                "type": "object",
                "properties": {
                    "scene_id": {
                        "type": "string",
                        "enum": SCENE_IDS,
                        "description": "Which scene preset to switch to"
                    }
                },
                "required": ["scene_id"]
            }
        },
        {
            "name": "adjust_eq",
            "description": "Adjust a single EQ band. Band indices: 0=60Hz, 1=170Hz, 2=310Hz, 3=600Hz, 4=1kHz, 5=3kHz, 6=6kHz, 7=12kHz. Gain is in decibels, -12 to +12.",
            "parameters": {
                "type": "object",
                "properties": {
                    "band_index": {"type": "integer", "minimum": 0, "maximum": 7, "description": "Which band to change (0..7)"},
                    "gain_db": {"type": "number", "minimum": -12, "maximum": 12, "description": "New gain in dB"}
                },
                "required": ["band_index", "gain_db"]
            }
        },
        {
            "name": "toggle_control",
            "description": "Turn a ClearEar processing control on or off.",
            "parameters": {
                "type": "object",
                "properties": {
                    "control": {
                        "type": "string",
                        "enum": ["cancel", "speech", "transparency"],
                        "description": "cancel = noise cancellation, speech = speech focus, transparency = passthrough"
                    },
                    "enabled": {"type": "boolean"}
                },
                "required": ["control", "enabled"]
            }
        }
    ]
}]

_key_rotation_lock = threading.Lock()
_key_rotation_index = 0


def load_env(file_path, override=False):
    if not file_path.exists():
        return
    for raw_line in file_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if key and value and (override or not os.environ.get(key)):
            os.environ[key] = value


def get_gemini_api_keys():
    """Collect all configured Gemini API keys, deduped, preserving order."""
    raw_keys = [
        os.environ.get("GEMINI_API_KEY"),
        os.environ.get("GEMINI_API_KEY_1"),
        os.environ.get("GEMINI_API_KEY_2"),
        os.environ.get("GOOGLE_API_KEY"),
        os.environ.get("GOOGLE_API_KEY_1"),
        os.environ.get("GOOGLE_API_KEY_2"),
        os.environ.get("GOOGLE_GEMINI_API_KEY"),
    ]
    for key in (os.environ.get("GEMINI_API_KEYS") or "").replace("\n", ",").split(","):
        raw_keys.append(key)

    keys = []
    seen = set()
    for key in raw_keys:
        if not key:
            continue
        key = key.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        keys.append(key)
    return keys


def get_starting_key_index(num_keys):
    """Return next round-robin start index, thread-safely."""
    global _key_rotation_index
    with _key_rotation_lock:
        start = _key_rotation_index % num_keys
        _key_rotation_index = (_key_rotation_index + 1) % num_keys
        return start


def send_json(handler, status, payload):
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Cache-Control", "no-store")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def normalize_turns(history, message):
    turns = list(history) + [{"type": "user", "text": message}]
    contents = []
    for turn in turns:
        if not isinstance(turn, dict):
            continue
        text = turn.get("text")
        if not isinstance(text, str):
            continue
        text = text.strip()
        if not text:
            continue
        role = "model" if turn.get("type") == "assistant" else "user"
        if role == "model" and not contents:
            continue
        if contents and contents[-1]["role"] == role:
            contents[-1]["parts"][0]["text"] += f"\n\n{text}"
        else:
            contents.append({"role": role, "parts": [{"text": text}]})
    return contents


class ClearEarHandler(SimpleHTTPRequestHandler):
    server_version = "ClearEar/1.0"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def log_message(self, fmt, *args):
        print("%s - %s" % (self.address_string(), fmt % args))

    def send_head(self):
        parsed = urlparse(self.path)
        parts = [part for part in unquote(parsed.path).split("/") if part]
        # Block access to dotfiles like .env
        if any(part.startswith(".") for part in parts):
            self.send_error(404, "Not found")
            return None
        return super().send_head()

    def do_POST(self):
        if urlparse(self.path).path == "/api/assistant":
            self.handle_assistant()
            return
        send_json(self, 404, {"error": "Not found."})

    def handle_assistant(self):
        load_env(ROOT / ".env", override=True)
        api_keys = get_gemini_api_keys()
        if not api_keys:
            send_json(self, 500, {"error": "Missing GEMINI_API_KEY in .env."})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            send_json(self, 400, {"error": "Invalid Content-Length."})
            return

        if length > MAX_BODY_SIZE:
            send_json(self, 413, {"error": "Request body is too large."})
            return

        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            send_json(self, 400, {"error": "Request body must be valid JSON."})
            return

        message = payload.get("message")
        if not isinstance(message, str) or not message.strip():
            send_json(self, 400, {"error": "Message is required."})
            return

        history = payload.get("history") if isinstance(payload.get("history"), list) else []
        mode = payload.get("mode")
        voice_mode = mode == "voice"
        system_prompt = SYSTEM_PROMPT_TEXT + (VOICE_RESPONSE_PROMPT if voice_mode else "")

        context = payload.get("context") if isinstance(payload.get("context"), dict) else {}
        preset = context.get("preset") if isinstance(context.get("preset"), str) else "Unknown"
        try:
            db = round(float(context.get("db")))
        except (TypeError, ValueError):
            db = "unknown"
        eq_state = context.get("eq") if isinstance(context.get("eq"), list) else []
        controls_state = context.get("controls") if isinstance(context.get("controls"), dict) else {}

        context_line = (
            f'\n\nLive app state: scene preset is "{preset}", measured ambient level is {db} dB.'
        )
        if eq_state:
            context_line += f" Current EQ gains (60Hz..12kHz): {eq_state}."
        if controls_state:
            context_line += f" Controls: {controls_state}."

        model = os.environ.get("GEMINI_MODEL") or DEFAULT_MODEL
        endpoint = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{quote(model, safe='')}:generateContent"
        )

        request_body = {
            "system_instruction": {"parts": [{"text": system_prompt + context_line}]},
            "contents": normalize_turns(history[-10:], message.strip()),
            "tools": TOOLS,
            "tool_config": {"function_calling_config": {"mode": "AUTO"}},
            "generationConfig": {
                "maxOutputTokens": 110 if voice_mode else 320,
                "temperature": 0.6 if voice_mode else 0.7
            }
        }

        # Round-robin across keys, then fall through to remaining on failure
        start = get_starting_key_index(len(api_keys))
        ordered = [api_keys[(start + i) % len(api_keys)] for i in range(len(api_keys))]

        gemini_text = None
        failures = []
        used_key_index = None

        for offset, api_key in enumerate(ordered):
            absolute_index = (start + offset) % len(api_keys) + 1
            request = Request(
                endpoint,
                data=json.dumps(request_body).encode("utf-8"),
                headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
                method="POST"
            )
            try:
                with urlopen(request, timeout=30, context=SSL_CONTEXT) as response:
                    gemini_text = response.read().decode("utf-8")
                used_key_index = absolute_index
                print(f"Gemini OK on key {absolute_index}/{len(api_keys)}")
                break
            except HTTPError as err:
                detail = err.read().decode("utf-8", errors="replace")[:300]
                failures.append(f"key {absolute_index}: HTTP {err.code}")
                print(f"Gemini API error for key {absolute_index}:", err.code, detail)
            except URLError as err:
                failures.append(f"key {absolute_index}: {err.reason}")
                print(f"Gemini connection error for key {absolute_index}:", err.reason)

        if gemini_text is None:
            send_json(self, 502, {
                "error": "All Gemini API keys failed.",
                "detail": "; ".join(failures)
            })
            return

        try:
            data = json.loads(gemini_text)
        except json.JSONDecodeError:
            send_json(self, 502, {"error": "Gemini returned invalid JSON."})
            return

        candidates = data.get("candidates", [])
        if not candidates:
            send_json(self, 502, {"error": "Gemini returned no candidates."})
            return

        parts = candidates[0].get("content", {}).get("parts", [])
        reply_chunks = []
        actions = []

        for part in parts:
            if not isinstance(part, dict):
                continue
            if isinstance(part.get("text"), str) and part["text"]:
                reply_chunks.append(part["text"])
            fc = part.get("functionCall") or part.get("function_call")
            if isinstance(fc, dict) and fc.get("name"):
                actions.append({
                    "name": fc.get("name"),
                    "args": fc.get("args") or {}
                })

        reply = "".join(reply_chunks).strip()

        if not reply and not actions:
            send_json(self, 502, {"error": "Gemini returned an empty response."})
            return

        send_json(self, 200, {
            "reply": reply,
            "actions": actions,
            "key_used": used_key_index
        })


def main():
    load_env(ROOT / ".env", override=True)
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "5173"))

    server = ThreadingHTTPServer((host, port), ClearEarHandler)
    print(f"ClearEar running at http://localhost:{port}")
    print("Gemini model:", os.environ.get("GEMINI_MODEL") or DEFAULT_MODEL)
    keys = get_gemini_api_keys()
    if not keys:
        print("Add GEMINI_API_KEY to .env before using the assistant.")
    else:
        print(f"Loaded {len(keys)} API key(s), rotating round-robin per request")
    server.serve_forever()


if __name__ == "__main__":
    main()