from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
import torch
from PIL import Image
import io
from inferance import MultiTaskResNet, preprocess, DEVICE, IMAGE_SIZE

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount templates
templates = Jinja2Templates(directory="templates")

# Load the model
model = MultiTaskResNet().to(DEVICE)
state = torch.load("best_utkface_resnet.pt", map_location=DEVICE)
model.load_state_dict(state)
model.eval()

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read the uploaded file
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Preprocess the image
        x = preprocess(image).unsqueeze(0).to(DEVICE)
        
        # Get predictions
        with torch.no_grad():
            g_logits, a_pred = model(x)
            
            g_idx = int(g_logits.argmax(dim=-1).item())
            gender = "female" if g_idx == 1 else "male"
            age = float(a_pred.item())
        
        return JSONResponse({
            "gender": gender,
            "age": round(age, 1)
        })
        
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"error": str(e)}
        ) 