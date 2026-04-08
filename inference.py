"""
inference.py — OpenEnv entry point for MetaMind
Delegates to the FastAPI app defined in meta_ai_env.py
"""

from meta_ai_env import app  # noqa: F401 — re-exported for OpenEnv runner

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
