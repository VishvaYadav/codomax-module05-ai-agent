"""
tools.py — the tools our AI agent can use.

Each tool = a normal Python function + a DECLARATION (name, description, JSON schema of the
arguments). The LLM only ever sees the declarations; it asks us to run a tool, we run it,
and we send the result back.

Free, no-key external APIs:
  * Open-Meteo  (weather)    https://open-meteo.com
  * Wikipedia   (knowledge)  https://en.wikipedia.org/api/rest_v1/
"""
from __future__ import annotations

import ast
import json
import operator
import sqlite3
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

HTTP_TIMEOUT = 10
USER_AGENT = "CodomaxAgentDemo/1.0 (educational project)"


def _get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as r:
        return json.loads(r.read().decode("utf-8"))


# =========================================================================== 1. calculator
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.Mod: operator.mod, ast.FloorDiv: operator.floordiv,
        ast.USub: operator.neg, ast.UAdd: operator.pos}


def calculator(expression: str) -> dict:
    """Safely evaluate arithmetic. Only numbers and + - * / ** % // ( ) are allowed:
    no eval(), so the model can't run arbitrary code."""
    def ev(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            left, right = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise ValueError("exponent too large")
            return _OPS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.operand))
        raise ValueError("only numbers and + - * / ** % // ( ) are allowed")
    expr = expression.replace(",", "").replace("^", "**").replace("×", "*").replace("÷", "/")
    value = ev(ast.parse(expr, mode="eval").body)
    return {"expression": expression, "result": round(value, 10) if isinstance(value, float) else value}


# =========================================================================== 2. weather
WEATHER_CODES = {0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast", 45: "fog",
                 48: "fog", 51: "light drizzle", 53: "drizzle", 55: "heavy drizzle", 61: "light rain",
                 63: "rain", 65: "heavy rain", 71: "light snow", 73: "snow", 75: "heavy snow",
                 80: "rain showers", 81: "rain showers", 82: "violent rain showers",
                 95: "thunderstorm", 96: "thunderstorm with hail", 99: "thunderstorm with hail"}


def get_weather(city: str) -> dict:
    """Current weather + 3-day forecast from Open-Meteo (free, no API key)."""
    geo = _get_json("https://geocoding-api.open-meteo.com/v1/search?" +
                    urllib.parse.urlencode({"name": city, "count": 1, "language": "en", "format": "json"}))
    if not geo.get("results"):
        return {"error": f"City '{city}' not found."}
    place = geo["results"][0]
    params = {"latitude": place["latitude"], "longitude": place["longitude"], "timezone": "auto",
              "current": "temperature_2m,relative_humidity_2m,precipitation,weather_code,wind_speed_10m",
              "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max,weather_code",
              "forecast_days": 3}
    w = _get_json("https://api.open-meteo.com/v1/forecast?" + urllib.parse.urlencode(params))
    cur, daily = w["current"], w["daily"]
    return {
        "location": f"{place['name']}, {place.get('admin1', '')}, {place.get('country', '')}".replace(", ,", ","),
        "now": {"temperature_c": cur["temperature_2m"], "humidity_pct": cur["relative_humidity_2m"],
                "precipitation_mm": cur["precipitation"], "wind_kmh": cur["wind_speed_10m"],
                "conditions": WEATHER_CODES.get(cur["weather_code"], "unknown"), "local_time": cur["time"]},
        "forecast": [{"date": daily["time"][i], "min_c": daily["temperature_2m_min"][i],
                      "max_c": daily["temperature_2m_max"][i],
                      "rain_chance_pct": daily["precipitation_probability_max"][i],
                      "conditions": WEATHER_CODES.get(daily["weather_code"][i], "unknown")}
                     for i in range(len(daily["time"]))],
        "source": "open-meteo.com",
    }


# =========================================================================== 3. wikipedia
def search_wikipedia(query: str) -> dict:
    """Search Wikipedia and return a summary of the best match."""
    s = _get_json("https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode(
        {"action": "query", "list": "search", "srsearch": query, "srlimit": 3, "format": "json"}))
    hits = s.get("query", {}).get("search", [])
    if not hits:
        return {"error": f"No Wikipedia article found for '{query}'."}
    title = hits[0]["title"]
    page = _get_json("https://en.wikipedia.org/api/rest_v1/page/summary/" +
                     urllib.parse.quote(title.replace(" ", "_")))
    return {"title": page.get("title", title), "summary": (page.get("extract") or "")[:1500],
            "url": page.get("content_urls", {}).get("desktop", {}).get("page", ""),
            "other_matches": [h["title"] for h in hits[1:]]}


# =========================================================================== 4. date/time
def get_current_datetime(timezone: str = "Asia/Kolkata") -> dict:
    try:
        now = datetime.now(ZoneInfo(timezone))
    except Exception:
        return {"error": f"Unknown timezone '{timezone}'. Use names like Asia/Kolkata or Europe/London."}
    return {"timezone": timezone, "date": now.strftime("%Y-%m-%d"), "weekday": now.strftime("%A"),
            "time": now.strftime("%H:%M")}


# =========================================================================== 5. database + memory
SCHEMA = """
CREATE TABLE team  (name TEXT PRIMARY KEY, role TEXT, city TEXT);
CREATE TABLE tasks (id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT, owner TEXT REFERENCES team(name),
                    status TEXT CHECK(status IN ('todo','in_progress','done')),
                    priority TEXT CHECK(priority IN ('low','medium','high')), due_date TEXT);
CREATE TABLE memory (id INTEGER PRIMARY KEY AUTOINCREMENT, fact TEXT, saved_at TEXT);
"""
SCHEMA_SUMMARY = ("team(name, role, city); tasks(id, title, owner, status ['todo','in_progress','done'], "
                  "priority ['low','medium','high'], due_date 'YYYY-MM-DD')")


def _seed(conn: sqlite3.Connection, today: date) -> None:
    d = lambda n: (today + timedelta(days=n)).isoformat()
    conn.executemany("INSERT INTO team VALUES (?,?,?)", [
        ("Vishva", "GenAI Intern", "Pune"), ("Ananya", "Backend Developer", "Bengaluru"),
        ("Rahul", "Data Analyst", "Mumbai"), ("Sara", "Project Manager", "Delhi")])
    conn.executemany("INSERT INTO tasks (title, owner, status, priority, due_date) VALUES (?,?,?,?,?)", [
        ("Build RAG demo", "Vishva", "done", "high", d(-3)),
        ("Write agent tools", "Vishva", "in_progress", "high", d(2)),
        ("Record LinkedIn demo video", "Vishva", "todo", "medium", d(5)),
        ("Set up API rate limiting", "Ananya", "in_progress", "high", d(1)),
        ("Fix login bug", "Ananya", "todo", "high", d(-1)),
        ("Write database migrations", "Ananya", "done", "medium", d(-6)),
        ("Prepare sales dashboard", "Rahul", "todo", "medium", d(4)),
        ("Clean customer dataset", "Rahul", "in_progress", "low", d(9)),
        ("Plan sprint review", "Sara", "todo", "high", d(3)),
        ("Update project roadmap", "Sara", "done", "low", d(-2)),
    ])
    conn.commit()


class ToolBox:
    """Holds the agent's tools and their state (database, long-term memory, safety settings)."""

    def __init__(self, safe_mode: bool = True, today: date | None = None):
        self.safe_mode = safe_mode
        self.conn = sqlite3.connect(":memory:", check_same_thread=False)
        self.conn.executescript(SCHEMA)
        _seed(self.conn, today or date.today())

    # ---- read-only SQL
    def query_database(self, sql: str) -> dict:
        s = sql.strip().rstrip(";").strip()
        if ";" in s:
            return {"error": "Only one SQL statement is allowed."}
        if not s.lower().startswith(("select", "with")):
            return {"error": "Blocked: query_database is READ-ONLY (SELECT only). "
                             "Use add_task to create tasks; deleting/updating is not allowed."}
        self.conn.execute("PRAGMA query_only = ON")      # database-level guard as a second layer
        try:
            cur = self.conn.execute(s)
            cols = [c[0] for c in cur.description]
            rows = [dict(zip(cols, r)) for r in cur.fetchmany(50)]
            return {"columns": cols, "rows": rows, "row_count": len(rows)}
        except sqlite3.Error as e:
            return {"error": f"SQL error: {e}. Schema: {SCHEMA_SUMMARY}"}
        finally:
            self.conn.execute("PRAGMA query_only = OFF")

    # ---- write tool (guarded)
    def add_task(self, title: str, owner: str, due_date: str, priority: str = "medium") -> dict:
        if self.safe_mode:
            return {"error": "Blocked by SAFE MODE: this tool changes data. Tell the user to turn off "
                             "Safe mode in the sidebar if they want the agent to add tasks."}
        if not self.conn.execute("SELECT 1 FROM team WHERE name = ?", (owner,)).fetchone():
            names = [r[0] for r in self.conn.execute("SELECT name FROM team")]
            return {"error": f"Unknown team member '{owner}'. Valid names: {names}"}
        try:
            date.fromisoformat(due_date)
        except ValueError:
            return {"error": "due_date must be in YYYY-MM-DD format."}
        if priority not in ("low", "medium", "high"):
            priority = "medium"
        cur = self.conn.execute("INSERT INTO tasks (title, owner, status, priority, due_date) "
                                "VALUES (?,?,?,?,?)", (title[:120], owner, "todo", priority, due_date))
        self.conn.commit()
        return {"created": {"id": cur.lastrowid, "title": title, "owner": owner,
                            "due_date": due_date, "priority": priority, "status": "todo"}}

    # ---- long-term memory
    def remember(self, fact: str) -> dict:
        self.conn.execute("INSERT INTO memory (fact, saved_at) VALUES (?, ?)",
                          (fact[:300], datetime.now().isoformat(timespec="seconds")))
        self.conn.commit()
        return {"saved": fact}

    def recall_memories(self) -> dict:
        return {"memories": [r[0] for r in self.conn.execute("SELECT fact FROM memory ORDER BY id")]}

    # ---- registry
    def functions(self) -> dict:
        return {"calculator": calculator, "get_weather": get_weather, "search_wikipedia": search_wikipedia,
                "get_current_datetime": get_current_datetime, "query_database": self.query_database,
                "add_task": self.add_task, "remember": self.remember, "recall_memories": self.recall_memories}

    def execute(self, name: str, args: dict) -> dict:
        """Run a tool requested by the model. Errors are returned to the model, never crash the app."""
        fn = self.functions().get(name)
        if fn is None:
            return {"error": f"Unknown tool '{name}'."}
        try:
            return fn(**(args or {}))
        except TypeError as e:
            return {"error": f"Bad arguments for {name}: {e}"}
        except Exception as e:  # network errors, parse errors, ...
            return {"error": f"{name} failed: {type(e).__name__}: {str(e)[:200]}"}


# =========================================================================== declarations
def _decl(name, description, properties=None, required=None):
    return {"name": name, "description": description,
            "parameters_json_schema": {"type": "object", "properties": properties or {},
                                       "required": required or []}}


TOOL_DECLARATIONS = [
    _decl("calculator", "Evaluate an arithmetic expression exactly. Use for ANY maths instead of mental math.",
          {"expression": {"type": "string", "description": "e.g. '0.175 * 2340' or '(2026 - 1991) * 12'"}},
          ["expression"]),
    _decl("get_weather", "Get the current weather and a 3-day forecast for a city.",
          {"city": {"type": "string", "description": "City name, e.g. 'Pune'"}}, ["city"]),
    _decl("search_wikipedia", "Look up facts about people, places, history, science or technology on Wikipedia.",
          {"query": {"type": "string", "description": "Search terms, e.g. 'Python programming language'"}},
          ["query"]),
    _decl("get_current_datetime", "Get today's date, weekday and current time. Use before working with "
          "relative dates like 'tomorrow' or 'next Friday'.",
          {"timezone": {"type": "string", "description": "IANA timezone, default 'Asia/Kolkata'"}}),
    _decl("query_database", "Run ONE read-only SQLite SELECT query on the team task database. Schema: "
          + SCHEMA_SUMMARY + ". Example: SELECT owner, COUNT(*) FROM tasks WHERE status != 'done' GROUP BY owner",
          {"sql": {"type": "string", "description": "A single SELECT statement"}}, ["sql"]),
    _decl("add_task", "Create a new task in the team task database. This CHANGES data.",
          {"title": {"type": "string"}, "owner": {"type": "string", "description": "Team member name"},
           "due_date": {"type": "string", "description": "YYYY-MM-DD"},
           "priority": {"type": "string", "enum": ["low", "medium", "high"]}},
          ["title", "owner", "due_date"]),
    _decl("remember", "Save a fact about the user for later conversations (long-term memory).",
          {"fact": {"type": "string", "description": "e.g. 'User's favourite language is Python'"}}, ["fact"]),
    _decl("recall_memories", "List everything saved with the remember tool."),
]
