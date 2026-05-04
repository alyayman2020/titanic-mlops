"""
server.py — Entry point for the FastAPI inference server.

Usage:
    python server.py                    # default port 8000
    python server.py --port 8080        # custom port
    uvicorn api.main:app --reload       # dev mode with hot reload
"""
import argparse

import uvicorn

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Titanic MLOps Inference Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    parser.add_argument("--reload", action="store_true", help="Enable hot reload (dev)")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    args = parser.parse_args()

    uvicorn.run(
        "api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1,
        log_level="info",
    )
