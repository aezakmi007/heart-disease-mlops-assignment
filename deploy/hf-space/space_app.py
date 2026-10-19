"""Entry point for the Hugging Face Space: serves the FastAPI app on port 7860."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import gradio as gr
import uvicorn
from fastapi.responses import RedirectResponse

from app import app as api  # the same FastAPI app used in Docker/Kubernetes


@api.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/docs")


with gr.Blocks() as demo:
    gr.Markdown("# Heart Disease Risk API\nOpen **/docs** and try `POST /predict`.")

api = gr.mount_gradio_app(api, demo, path="/ui")

if __name__ == "__main__":
    uvicorn.run(api, host="0.0.0.0", port=7860)