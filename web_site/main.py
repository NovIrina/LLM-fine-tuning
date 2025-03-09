"""
This module contains the FastAPI application that serves the web interface for the text generator.
"""
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from transformers import AutoModel, AutoTokenizer
from train_eval_pipeline.constants import PATH_TO_MODEL, PATH_TO_TOKENIZER
from train_eval_pipeline.model import load_model
from train_eval_pipeline.tokenizer import load_tokenizer
from train_eval_pipeline.utils import get_torch_device
from tap import Tap
from pathlib import Path


class WebSiteArguments(Tap):
    """
    Defines the command-line arguments for the script.
    """
    path_to_model: Path = PATH_TO_MODEL
    path_to_tokenizer: Path = PATH_TO_TOKENIZER


def create_app(model: AutoModel, tokenizer: AutoTokenizer):
    app = FastAPI(
        title="GPT-2 Text Generator",
        description="A web interface to generate text using GPT-2",
        version="1.0",
    )

    app.mount("/static", StaticFiles(directory="web_site/static"), name="static")

    templates = Jinja2Templates(directory="templates")

    device = get_torch_device()
    model.to(device)

    @app.get("/", response_class=HTMLResponse)
    async def read_root(request: Request):
        """
        Serve the main page with the input form.
        """
        return templates.TemplateResponse("index.html", {"request": request})

    @app.post("/generate", response_class=HTMLResponse)
    async def generate_text(
        request: Request, prompt: str = Form(...), max_length: int = Form(50)
    ):
        """
        Handle form submission, generate text using GPT-2, and render the results.
        """
        try:
            inputs = tokenizer.encode(prompt, return_tensors="pt")
            inputs = inputs.to(device)

            outputs = model.generate(
                inputs,
                max_length=max_length,
                num_return_sequences=1,
                no_repeat_ngram_size=2,
            )

            generated_texts = [
                tokenizer.decode(output, skip_special_tokens=True) for output in outputs
            ]

            return templates.TemplateResponse(
                "index.html",
                {
                    "request": request,
                    "generated_texts": generated_texts,
                    "prompt": prompt,
                    "max_length": max_length,
                },
            )

        except Exception as e:
            return templates.TemplateResponse(
                "index.html",
                {
                    "request": request,
                    "error": str(e),
                    "prompt": prompt,
                    "max_length": max_length,
                },
            )

    return app


if __name__ == "__main__":
    parser = WebSiteArguments().parse_args()
    model = load_model(path_to_load=parser.path_to_model)
    tokenizer = load_tokenizer(parser.path_to_tokenizer)

    app = create_app(model, tokenizer)
    import uvicorn

    uvicorn.run(
        app, 
        host="127.0.0.1", 
        port=8000,
        reload=True 
    )
