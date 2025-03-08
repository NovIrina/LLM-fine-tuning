from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from utils import get_torch_device

app = FastAPI(
    title="GPT-2 Text Generator",
    description="A web interface to generate text using GPT-2",
    version="1.0"
)

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)


device = get_torch_device()
model.to(device)

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """
    Serve the main page with the input form.
    """
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/generate", response_class=HTMLResponse)
async def generate_text(request: Request,
                        prompt: str = Form(...),
                        max_length: int = Form(50)):
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
            no_repeat_ngram_size=2
        )

        generated_texts = [
            tokenizer.decode(output, skip_special_tokens=True) for output in outputs
        ]

        return templates.TemplateResponse("index.html", {
            "request": request,
            "generated_texts": generated_texts,
            "prompt": prompt,
            "max_length": max_length
        })

    except Exception as e:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error": str(e),
            "prompt": prompt,
            "max_length": max_length
        })
