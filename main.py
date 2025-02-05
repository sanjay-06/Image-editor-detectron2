from fastapi import FastAPI
from routes.ImageEditor import Image_editor
from fastapi.staticfiles import StaticFiles
import uvicorn

app=FastAPI()

app.mount("/static", StaticFiles(directory="html/static"), name="static")

app.include_router(Image_editor)

uvicorn.run(app,host='localhost',port=2000)