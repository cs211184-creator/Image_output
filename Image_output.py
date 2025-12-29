from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel
from openai import OpenAI
import base64

app = FastAPI(title="Foundry Image Tool", version="1.0.0")

AZURE_ENDPOINT = "https://aicoe-resource.openai.azure.com/openai/v1/"
API_KEY = "9KY5cHHxAsfIjPi5KfNJBZGz5jfl8g3nE72UOrp91uxVD4UDMG1fJQQJ99BKACHYHv6XJ3w3AAAAACOGrqHE"
DEPLOYMENT = "FLUX.1-Kontext-pro"

client = OpenAI(base_url=AZURE_ENDPOINT, api_key=API_KEY)

class ImgReq(BaseModel):
    prompt: str
    size: str = "1024x1024"
    n: int = 1

@app.post("/generate-image", response_class=Response)
def generate_image(req: ImgReq):
    if not req.prompt.strip():
        raise HTTPException(400, "prompt required")

    img = client.images.generate(
        model=DEPLOYMENT,
        prompt=req.prompt,
        n=1,               # keep 1 to limit payload
        size=req.size
    )

    b64 = img.data[0].b64_json
    if not b64:
        raise HTTPException(500, "No b64_json returned")

    image_bytes = base64.b64decode(b64)

    # Return PNG bytes directly
    return Response(content=image_bytes, media_type="image/png")
