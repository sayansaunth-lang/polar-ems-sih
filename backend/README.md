# POLAR-EMS backend

A real Python backend for the POLAR-EMS microgrid simulator: a FastAPI server
running the same weather/dispatch/safety/forecast/anomaly model as the
frontend, plus a terminal client so you can drive it and read its output
directly in a VS Code terminal.

**This is a synthetic simulation**, not a connection to real station
hardware -- see the module docstring in `app/engine.py`.

## Open this in VS Code

Open the `backend/` folder itself as your VS Code workspace (`File > Open
Folder... > backend`). `.vscode/launch.json` already has two run
configurations ready in the Run and Debug panel:

- **POLAR-EMS: Backend (uvicorn)** -- starts the API server
- **POLAR-EMS: CLI client** -- starts the terminal menu (run this *after*
  the server is up, in a second terminal/debug session)

## Setup

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

## Run the backend

```bash
uvicorn app.main:app --reload --port 8000
```

- Swagger docs: http://127.0.0.1:8000/docs
- Health check: http://127.0.0.1:8000/health

The simulation starts ticking immediately (1 tick per second of real time,
each tick advancing the simulated clock by `speed x 0.25` hours) and keeps
running in the background for as long as the server process is alive.

## Run the terminal client

In a **second** terminal (server must already be running):

```bash
python cli.py
```

You get a numbered menu: live status, alerts, green audit, triggering any
of the 8 demo scenarios, start/pause/reset, toggling a simulated internet
outage, and changing simulation speed. Every option is a plain HTTP call to
the API below it -- nothing is duplicated or faked in the CLI itself.

Point it at a non-default server with an environment variable:

```bash
set POLAR_EMS_API=http://127.0.0.1:8000   # Windows
python cli.py
```

## API reference

All endpoints are also documented interactively at `/docs`.

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness + current sim time |
| GET | `/api/station` | Static station info |
| GET | `/api/scenarios` | List the 8 demo scenarios + which is active |
| GET | `/api/telemetry/latest` | Full current snapshot (weather, load, battery, diesel, renewables, decision) |
| GET | `/api/telemetry?limit=N` | Recent history records |
| GET | `/api/battery` | Battery-only snapshot |
| GET | `/api/generator` | Diesel-only snapshot |
| GET | `/api/dispatch/current` | Current mode + dispatch decision + plain-English reason |
| GET | `/api/dispatch/history?limit=N` | Recent dispatch decision log |
| GET | `/api/forecast/load?horizon=1\|6\|12\|24` | Forecast vs. actual points + MAE/RMSE |
| GET | `/api/forecast/renewables?horizon=...` | Wind/solar forecast vs. actual points |
| GET | `/api/alerts?limit=N` | Open anomaly alerts |
| GET | `/api/anomalies` | Subsystem health scores + alert counts |
| GET | `/api/green-audit` | Cumulative fuel/CO2/runtime savings, AI vs. baseline |
| GET | `/api/sync/status` | Online/offline + outbox + sync log |
| POST | `/api/sync` | Manually step the sync process (only while online) |
| POST | `/api/simulation/start` | Resume ticking |
| POST | `/api/simulation/pause` | Pause ticking |
| POST | `/api/simulation/reset` | Reset to defaults |
| POST | `/api/simulation/offline` | Simulate an internet outage |
| POST | `/api/simulation/online` | Restore connectivity (triggers sync) |
| POST | `/api/simulation/speed` `{"speed": 1\|4\|12}` | Change simulated hours per tick |
| POST | `/api/simulation/scenario` `{"id": "..."}` | Apply one of the 8 named scenarios, or `"normal"` to clear |

## Connecting the website to this backend

The existing frontend (`polar-ems-web/index.html`) runs its own
self-contained simulation and does **not** call this API by default -- the
two are independent implementations of the same model so either can be
demoed on its own.

To see live backend state reflected on the website, use the **Live Backend**
panel added to the site (see the top of `index.html` for the toggle) --
point it at `http://127.0.0.1:8000` while both are running locally.

**Important:** this only works if the *page itself* is also loaded over
plain HTTP (e.g. `python -m http.server` in `polar-ems-web/`, or opening
the file locally). If you load the page over HTTPS -- which is exactly
what a Vercel deployment gives you -- the browser will block the request
as "mixed content": an HTTPS page is not allowed to call a plain HTTP
address, `localhost` included. There is no frontend-side fix for this; it
is a browser security policy, not a bug in this code.

To make the Live Backend panel work from your deployed HTTPS site (for you
or for anyone else viewing it), deploy this backend somewhere that gives
you HTTPS -- a `render.yaml` is included at the repository root for a
one-click Render Blueprint deploy -- and enter that `https://...` URL into
the panel instead of localhost. `main.py` already has CORS wide open
(`allow_origins=["*"]`) for this purpose; tighten it to your actual
frontend origin before a real deployment.

## What's simulated vs. real

Identical methodology to the frontend: weather is a deterministic seasonal
model (not a live feed), generation follows real turbine/PV power-curve
equations, load follows a priority-tiered synthetic profile, and the two
dispatch controllers run on identical conditions each tick so the fuel/CO2
comparison is a measured delta, not an assumed one. See `app/engine.py`'s
docstring and the main project's briefing document for the full
methodology notes.
