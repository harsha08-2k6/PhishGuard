# PhishGuard Research API

FastAPI service for URL-only feature extraction and model inference.

Run this service from the `website-react/backend` directory while the React app runs from `website-react/frontend`.

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```

The React site calls `POST /api/scan`. Set `VITE_API_BASE_URL` when the API is deployed elsewhere. If the API is unavailable, the browser prototype keeps using its local lexical scorer.
