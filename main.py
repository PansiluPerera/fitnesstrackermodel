from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from ml_logic import generate_workout_plan, generate_meal_plan

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, use only your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/generate_workout/")
async def generate_workout(
    image: UploadFile = File(...),
    height: float = Form(...),
    weight: float = Form(...),
    age: int = Form(...)
):
    content = await image.read()
    workout = generate_workout_plan(image_bytes=content, height=height, weight=weight, age=age)
    return {"workout": workout}

@app.post("/generate_meal/")
async def generate_meal(
    image: UploadFile = File(...),
    goal: str = Form(...),
    height: float = Form(...),
    weight: float = Form(...),
    age: int = Form(...)
):
    content = await image.read()
    meal = generate_meal_plan(image_bytes=content, goal=goal, height=height, weight=weight, age=age)
    return {"meal": meal}
