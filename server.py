#!/usr/bin/env python3
"""ClearEar local backend.

Serves the static site and proxies AI requests to Gemini. Keys are loaded
from .env and rotated round-robin per request so the load spreads evenly
across both keys. If one key fails, the request automatically retries the
remaining keys.

The assistant has tool/function calling enabled so it can change scenes,
adjust EQ bands, toggle controls, and trigger audio playback from natural language.

NEW endpoints:
  POST /api/assistant      - Main AI chat (existing)
  POST /api/summarize      - Summarize a conversation transcript
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
MAX_BODY_SIZE = 128 * 1024  # bumped for transcripts
DEFAULT_MODEL = "gemini-2.5-flash"
SSL_CONTEXT = None
if certifi:
    SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
else:
    # Fallback: create context that doesn't verify (for local dev without certifi)
    SSL_CONTEXT = ssl.create_default_context()
    SSL_CONTEXT.check_hostname = False
    SSL_CONTEXT.verify_mode = ssl.CERT_NONE

SYSTEM_PROMPT_TEXT = """You are the AI assistant inside ClearEar, a software-defined hearing aid app that turns ordinary earbuds into smart hearing aids through phone-side audio processing.

ClearEar features the user can access through you:
- Scene presets (use these IDs in switch_scene): quiet, library, office, lecture, classroom, restaurant, crowded, commute, outdoor, windy, phone, music, gym, theater, tv, sleep
- 8-band frequency EQ: indices 0..7 map to 60Hz, 170Hz, 310Hz, 600Hz, 1kHz, 3kHz, 6kHz, 12kHz. Each band is adjustable from -12 to +12 dB.
- Three toggle controls: cancel (noise cancellation), speech (speech focus), transparency (passthrough)
- Live captions, sound exposure journal, hearing profiles
- Audio playback buffer: you can play back the last N seconds of recorded audio using the playback_audio tool.

YOUR JOB:
1. Be a fully general assistant. Answer absolutely any question the user asks, from code to history to math to recipes to philosophy. Never refuse to answer just because it isn't about sound.
2. When the user asks you to change ClearEar (switch scenes, tweak EQ, toggle controls, play back audio), CALL THE TOOL to actually do it. Do not just describe what you would do.
3. When the user asks about what was just said, what someone told them, or wants to hear back recent audio, use playback_audio. If they ask what someone meant or said, use explain_recent_speech with the transcript context provided.
4. Always end your reply with a short follow-up question related to sound, hearing, noise, audio, music, the acoustic environment, or how ClearEar is performing. This keeps the conversation engaging and grounded in the product.

STYLE:
- Conversational and warm, not robotic.
- Short answers: 1 to 3 sentences for the response, then the follow-up question.
- No bullet points or headers in chat replies.
- No em dashes. Use commas or periods.
- When you use a tool, briefly confirm the change in your reply. Do not narrate the tool itself.

EXAMPLES:

User: What's the capital of France?
You: Paris. Speaking of cities, are you somewhere noisy right now? I can switch scenes if it helps.

User: Playback the last 10 seconds.
You: [calls playback_audio with seconds=10] Playing back the last 10 seconds for you. Did you catch what you needed?

User: What did the cashier just say to me?
You: Based on the recent transcript, they asked if you wanted your receipt in the bag. Want me to replay that moment?

User: Hey ClearEar, what was just told to me?
You: [uses recent transcript context] They said "your total is $14.50, would you like to pay with card?" Want me to boost speech clarity for this environment?"""

VOICE_RESPONSE_PROMPT = "\n\nThis request came from the in-ear voice assistant. Reply for audio playback, not screen reading. Be more concise than chat: one short sentence is best, two short sentences maximum. Avoid long lists, examples, citations, and filler."

SUMMARIZE_PROMPT = """You are an AI that summarizes conversations captured by a hearing aid app called ClearEar.

Given a transcript of a conversation, produce a JSON response with:
- "title": A short descriptive title for this conversation (e.g. "Chat with barista about coffee order", "Teacher explaining photosynthesis", "Discussion at checkout counter"). The title should reflect what actually happened based on the transcript content. Be specific and natural.
- "summary": A 1-3 sentence overview of what was discussed, decisions made, or key information exchanged.
- "participants": Best guess at who was involved (e.g. "You and a teacher", "You and store cashier", "Group conversation").
- "location_guess": Best guess at the environment/location based on context clues (e.g. "classroom", "coffee shop", "unknown").
- "key_points": Array of 1-4 key takeaways or important things said.

Return ONLY valid JSON, no markdown fencing. Be concise and natural."""

SCENE_IDS = [
    "quiet", "library", "office", "lecture", "classroom", "restaurant",
    "crowded", "commute", "outdoor", "windy", "phone", "music",
    "gym", "theater", "tv", "sleep"
]

TOOLS = [{
    "function_declarations": [
        {
            "name": "switch_scene",
            "description": "Switch ClearEar to a different scene preset.",
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
                    "band_index": {"type": "integer", "minimum": 0, "maximum": 7},
                    "gain_db": {"type": "number", "minimum": -12, "maximum": 12}
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
                        "enum": ["cancel", "speech", "transparency"]
                    },
                    "enabled": {"type": "boolean"}
                },
                "required": ["control", "enabled"]
            }
        },
        {
            "name": "playback_audio",
            "description": "Play back the last N seconds of recorded ambient audio so the user can re-hear what they missed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "seconds": {
                        "type": "integer",
                        "minimum": 5,
                        "maximum": 60,
                        "description": "How many seconds of recent audio to replay"
                    }
                },
                "required": ["seconds"]
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
        os.environ.get("GEMINI_API_KEY_3"),
        os.environ.get("GEMINI_API_KEY_4"),
        os.environ.get("GEMINI_API_KEY_5"),
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


def call_gemini(api_keys, request_body):
    """Call Gemini with key rotation. Returns parsed JSON or None."""
    model = os.environ.get("GEMINI_MODEL") or DEFAULT_MODEL
    endpoint = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{quote(model, safe='')}:generateContent"
    )

    start = get_starting_key_index(len(api_keys))
    ordered = [api_keys[(start + i) % len(api_keys)] for i in range(len(api_keys))]

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
            print(f"Gemini OK on key {absolute_index}/{len(api_keys)}")
            return json.loads(gemini_text)
        except HTTPError as err:
            detail = err.read().decode("utf-8", errors="replace")[:300]
            print(f"Gemini API error for key {absolute_index}:", err.code, detail)
        except URLError as err:
            print(f"Gemini connection error for key {absolute_index}:", err.reason)
        except json.JSONDecodeError:
            print(f"Gemini returned invalid JSON on key {absolute_index}")

    return None


class ClearEarHandler(SimpleHTTPRequestHandler):
    server_version = "ClearEar/1.0"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def log_message(self, fmt, *args):
        print("%s - %s" % (self.address_string(), fmt % args))

    def send_head(self):
        parsed = urlparse(self.path)
        parts = [part for part in unquote(parsed.path).split("/") if part]
        if any(part.startswith(".") for part in parts):
            self.send_error(404, "Not found")
            return None
        return super().send_head()

    def do_POST(self):
        path = urlparse(self.path).path
        if path == "/api/assistant":
            self.handle_assistant()
        elif path == "/api/summarize":
            self.handle_summarize()
        else:
            send_json(self, 404, {"error": "Not found."})

    def _read_body(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            return None, "Invalid Content-Length."
        if length > MAX_BODY_SIZE:
            return None, "Request body is too large."
        try:
            return json.loads(self.rfile.read(length) or b"{}"), None
        except json.JSONDecodeError:
            return None, "Request body must be valid JSON."

    def handle_assistant(self):
        load_env(ROOT / ".env", override=True)
        api_keys = get_gemini_api_keys()
        if not api_keys:
            send_json(self, 500, {"error": "Missing GEMINI_API_KEY in .env."})
            return

        payload, err = self._read_body()
        if err:
            send_json(self, 400, {"error": err})
            return

        message = payload.get("message")
        if not isinstance(message, str) or not message.strip():
            send_json(self, 400, {"error": "Message is required."})
            return

        history = payload.get("history") if isinstance(payload.get("history"), list) else []
        mode = payload.get("mode")
        voice_mode = mode == "voice"

        # Recent transcript context for "what was just said" type queries
        recent_transcript = payload.get("recent_transcript", "")
        transcript_context = ""
        if recent_transcript and isinstance(recent_transcript, str) and recent_transcript.strip():
            transcript_context = f"\n\nRecent transcript from the last 60 seconds of ambient audio:\n\"{recent_transcript.strip()}\"\nUse this to answer questions about what was just said."

        system_prompt = SYSTEM_PROMPT_TEXT + transcript_context + (VOICE_RESPONSE_PROMPT if voice_mode else "")

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

        data = call_gemini(api_keys, request_body)
        if data is None:
            send_json(self, 502, {"error": "All Gemini API keys failed."})
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

        send_json(self, 200, {"reply": reply, "actions": actions})

    def handle_summarize(self):
        """Summarize a conversation transcript into a title + summary."""
        load_env(ROOT / ".env", override=True)
        api_keys = get_gemini_api_keys()
        if not api_keys:
            send_json(self, 500, {"error": "Missing GEMINI_API_KEY in .env."})
            return

        payload, err = self._read_body()
        if err:
            send_json(self, 400, {"error": err})
            return

        transcript = payload.get("transcript")
        if not isinstance(transcript, str) or not transcript.strip():
            send_json(self, 400, {"error": "Transcript is required."})
            return

        duration_min = payload.get("duration_min", 0)
        avg_db = payload.get("avg_db", 0)

        user_message = f"Conversation duration: {duration_min:.1f} minutes. Average ambient noise: {avg_db} dB.\n\nTranscript:\n{transcript.strip()}"

        request_body = {
            "system_instruction": {"parts": [{"text": SUMMARIZE_PROMPT}]},
            "contents": [{"role": "user", "parts": [{"text": user_message}]}],
            "generationConfig": {
                "maxOutputTokens": 400,
                "temperature": 0.4
            }
        }

        data = call_gemini(api_keys, request_body)
        if data is None:
            send_json(self, 502, {"error": "All Gemini API keys failed."})
            return

        candidates = data.get("candidates", [])
        if not candidates:
            send_json(self, 502, {"error": "Gemini returned no candidates."})
            return

        parts = candidates[0].get("content", {}).get("parts", [])
        text = ""
        for part in parts:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                text += part["text"]

        text = text.strip()
        # Try to parse as JSON
        try:
            # Strip markdown fencing if present
            if text.startswith("```"):
                text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            summary_data = json.loads(text)
        except json.JSONDecodeError:
            summary_data = {
                "title": "Conversation",
                "summary": text[:200] if text else "Could not summarize.",
                "participants": "Unknown",
                "location_guess": "unknown",
                "key_points": []
            }

        send_json(self, 200, summary_data)


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
