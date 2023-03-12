from fastapi import FastAPI
from routes.object_detection import object_detect
from fastapi.staticfiles import StaticFiles
import uvicorn

app=FastAPI()

app.mount("/static", StaticFiles(directory="html/static"), name="static")

app.include_router(object_detect)

uvicorn.run(app,host='localhost',port=2000)