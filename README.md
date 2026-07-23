# WorkloadIQ

WorkloadIQ is a standalone service-desk assessment application. It combines a native React workspace with the existing Python and Google ADK analysis core.

The product supports two assessment paths:

- **Workload intelligence**: demand concentration, Pareto drivers, assignment-group load, status mix, resolution signals, AI priorities, and CSV export.
- **Ticket quality audit**: ticket-level QA scoring, pass/fail controls, fatal exceptions, section performance, review queues, and detailed CSV export.

The assistant in the lower-right corner is grounded in the active completed assessment. It uses the same ADK session and `/run_sse` connection pattern as the new chatbot portal, with a deterministic grounded fallback.

## Architecture

```text
Browser -> Vite :3000 -> /api proxy -> FastAPI :8000
                                      |
                                ADK session + /run_sse
                                      |
                    normalization, QA, and heavy-hitter tools
        |
Job-scoped progress, results, charts, and CSV exports
```

The production build is self-contained: FastAPI serves both the compiled frontend and the API.

## Local development

Requirements:

- Python 3.12
- Node.js 20+
- Google Cloud application-default credentials for Vertex AI features

Install backend dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install the frontend dependencies once:

```bash
cd frontend
npm install
cd ..
```

Start the backend and frontend together from the repository root:

```bash
./scripts/start_dev.sh
```

The script uses `/home/AI_POC/venvs/debalekha/bin/python`, binds both services to `0.0.0.0`, serves the UI on port `3000`, and proxies browser `/api` calls to the Linux backend at `127.0.0.1:8000`.

### Open from Windows

For a VS Code Remote SSH or similar Linux development session, forward port `3000` and open:

```text
http://localhost:3000
```

Only the frontend port needs forwarding because Vite performs the backend proxy inside Linux. When Windows has network routing to the Linux host, the direct form is:

```text
http://<linux-host-ip>:3000
```

The default ADK origin policy accepts localhost plus private `10.x`, `172.16-31.x`, and `192.168.x` development hosts on ports `3000` and `5173`. Set `CORS_ORIGINS` explicitly when using another hostname, port, or HTTPS origin.

For the compiled single-service build, forward port `8000` or open `http://<linux-host-ip>:8000`.

## Production build

```bash
cd frontend
npm ci
npm run build
cd ..
python -m workload_analysis.server_with_upload
```

Open `http://localhost:8000`.

## Container

```bash
docker build -t workload-iq .
docker run --rm -p 8000:8000 --env-file workload_analysis/.env workload-iq
```

## Configuration

Copy `workload_analysis/.env.example` and provide the required Google Cloud values. Useful controls include:

- `TICKET_QA_MODEL`
- `TICKET_QA_PASS_THRESHOLD`
- `HEAVY_HITTER_LLM_MODEL`
- `HEAVY_HITTER_TOP_N`
- `NORMALIZE_USE_LLM`
- `TICKET_QA_OUTPUT_DIR`
- `CORS_ORIGINS`
- `MAX_UPLOAD_BYTES`
- `COPILOT_USE_ADK`
- `ADK_INTERNAL_BASE_URL`

## Verification

Build the frontend, then run:

```bash
python scripts/smoke_test.py
```

The smoke test starts the complete service, runs the bundled workload sample, checks the structured result, asks the grounded assistant, and downloads the generated CSV.
