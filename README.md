# sakz-platform

A production-shaped web platform layered on top of the existing, battle-tested
`sakz` Telegram-bot trading engine. The 8 engine modules are vendored
**unchanged** (51/51 unit tests still pass) under `backend/engine/`; everything
else is a thin, scalable shell around them.

```
┌─ Next.js dashboard (App Router, Tailwind, React Query, WS)
│        │ REST + WebSocket
┌─ FastAPI (stateless) ────┐
│   auth │ signals │ positions │ ops      │
│        │ engine_loader → sakz engine (pure) │
└─ Postgres (prod) / SQLite (dev)  ←  Scan worker (isolated process)
```

See the full design in the Notion doc **“sakz-platform — System Architecture &
MVP Design”**.

## Layout

```
backend/
  engine/        # vendored sakz modules (DO NOT EDIT) + their tests
  app/
    main.py          # FastAPI factory + dev in-process scan loop
    worker.py        # standalone scan worker (prod)
    engine_loader.py # single import surface for the engine
    config.py db.py models.py schemas.py security.py deps.py ws.py
    routers/         # auth, signals, positions, ops (+ /ws/signals)
    services/        # signal / pnl / scan orchestration + candle provider
  tests/         # engine tests (unchanged) + API smoke test
frontend/        # Next.js 14 dashboard
docker-compose.yml
```

## Run locally (no Docker)

### Backend
```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload      # http://localhost:8000  (docs at /docs)
```
SQLite is created automatically. The in-process scanner (`INPROCESS_SCAN=true`)
uses a synthetic candle provider so signals appear with zero exchange config.

### Frontend
```bash
cd frontend
cp .env.local.example .env.local
npm install
npm run dev                        # http://localhost:3000
```

## Run with Docker (prod-shaped)
```bash
cp .env.example .env
docker compose up --build
```
This runs Postgres, the API, a **separate** scan worker, and the web app.

## Tests
```bash
cd backend
pip install -r requirements.txt
python -m pytest tests/            # engine tests + API smoke test
```

## Auth

Telegram Login Widget → `POST /auth/telegram`. The backend verifies the
HMAC-SHA256 signature against `TELEGRAM_BOT_TOKEN`, upserts the user, and issues
a JWT. Set `ADMIN_TELEGRAM_IDS` to grant the admin role (required for
`/ops/scan` and `/ops/circuit-breaker`). A dev JWT paste box is available on the
login page for local testing.

## Real-time fan-out (Redis)

WebSocket clients connect to `GET /ws/signals?token=<jwt>`. New tradeable
signals are published to the Redis channel `sakz:signals` by whoever produced
them (the scan worker via `events.publish_sync`, or an admin-triggered scan via
`events.publish`). Each API replica runs one subscriber task that relays those
events to the sockets it holds. Set `REDIS_URL=redis://...` to enable it; leave
it empty for single-replica/local runs (events go straight to local sockets).

## Scaling notes (millions of users)

- **API is stateless** → scale horizontally behind a load balancer.
- **Scan worker is a separate process** → scale independently; shard pairs
  across workers.
- **WebSocket fan-out** is Redis pub/sub backed (`app/events.py`): every API
  replica subscribes to the `sakz:signals` channel and relays events to its own
  sockets, so a signal from any replica or the worker reaches every client.
  Without `REDIS_URL` set it falls back to in-process delivery (single replica),
  so local dev needs no Redis. `docker-compose.yml` runs the API at 2 replicas
  to exercise this path.
- **Postgres** with the provided indexes; add read replicas + PgBouncer.
- Swap `services/candles.py` synthetic provider for the `Router`-backed
  exchange fetcher (MEXC → Bybit fallback) when wiring real market data.
