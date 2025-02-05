import shutil
import cv2
from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.responses import FileResponse
from fastapi import File, UploadFile, Form
from features.filtering import Filter
from features.Upload import UploadObj
from features.feature_detection import Feature

Image_editor=APIRouter()
templates=Jinja2Templates(directory="html")

filtering = Filter()
upload_obj = UploadObj()
feature_obj = Feature()

@Image_editor.get('/')
def load_page(request : Request):
    return templates.TemplateResponse("index.html",{"request":request})

@Image_editor.get('/detect', response_class=HTMLResponse)
def detect_image(request: Request):
    visual = upload_obj.obj.predict()
    file_static_location = f"static/features/detect.jpg"
    file_location = f"html/static/features/detect.jpg"
    cv2.imwrite(file_location, visual)
    return templates.TemplateResponse("show.html",{"request":request, "image": upload_obj.img, "detect": file_static_location})

@Image_editor.get('/sepia', response_class=HTMLResponse)
def perform_sepia(request: Request):
    im = cv2.imread('html/'+upload_obj.img)
    sep = filtering.perform_serpia_filter(im)
    file_static_location = f"static/features/sepia.jpg"
    file_location = f"html/static/features/sepia.jpg"
    cv2.imwrite(file_location, sep) 
    return templates.TemplateResponse("show.html",{"request":request, "image": upload_obj.img, "detect": file_static_location})

@Image_editor.get('/vintage', response_class=HTMLResponse)
def perform_sepia(request: Request):
    im = cv2.imread('html/'+upload_obj.img)
    vintage = filtering.perform_vintage(im)
    file_static_location = f"static/features/vintage.jpg"
    file_location = f"html/{file_static_location}"
    cv2.imwrite(file_location, vintage)
    return templates.TemplateResponse("show.html",{"request":request, "image": upload_obj.img, "detect": file_static_location})

@Image_editor.get('/harris', response_class=HTMLResponse)
def perform_harris(request: Request):
    img = cv2.imread('html/'+upload_obj.img)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    harris = feature_obj.harris_corner(img, gray)
    file_static_location = f"static/features/harris.jpg"
    file_location = f"html/{file_static_location}"
    cv2.imwrite(file_location, harris)
    return templates.TemplateResponse("show.html",{"request":request, "image": upload_obj.img, "detect": file_static_location})

@Image_editor.get('/sift', response_class=HTMLResponse)
def perform_sift(request: Request):
    img = cv2.imread('html/'+upload_obj.img)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    sift = feature_obj.sift(img, gray)
    file_static_location = f"static/features/sift.jpg"
    file_location = f"html/{file_static_location}"
    cv2.imwrite(file_location, sift)
    return templates.TemplateResponse("show.html",{"request":request, "image": upload_obj.img, "detect": file_static_location})

@Image_editor.get('/vintage', response_class=HTMLResponse)
def perform_sepia(request: Request):
    im = cv2.imread('html/'+upload_obj.img)
    vintage = filtering.perform_vintage(im)
    file_static_location = f"static/features/vintage.jpg"
    file_location = f"html/{file_static_location}"
    cv2.imwrite(file_location, vintage)
    return templates.TemplateResponse("show.html",{"request":request, "image": upload_obj.img, "detect": file_static_location})

@Image_editor.post('/clone', response_class=HTMLResponse)
def perform_clone(request: Request, index:str=Form(...)):
    clone = upload_obj.obj.clone(item_mask_index=int(index))
    file_static_location = f"static/features/clone.jpg"
    file_location = f"html/{file_static_location}"
    cv2.imwrite(file_location, clone)
    return templates.TemplateResponse("show.html",{"request":request, "image": upload_obj.img, "detect": file_static_location})

@Image_editor.post('/blur_box', response_class=HTMLResponse)
def perform_box(request: Request, index:str=Form(...)):
    im = cv2.imread('html/'+upload_obj.img)
    blur_box = upload_obj.obj.blur_box(index)
    file_static_location = f"static/features/blur_box.jpg"
    file_location = f"html/{file_static_location}"
    cv2.imwrite(file_location, blur_box)
    return templates.TemplateResponse("show.html",{"request":request, "image": upload_obj.img, "detect": file_static_location})

@Image_editor.post('/blur_bg', response_class=HTMLResponse)
def perform_bg(request: Request, index:str=Form(...)):
    im = cv2.imread('html/'+upload_obj.img)
    blur_bg = upload_obj.obj.blur_bg(index)
    file_static_location = f"static/features/blur_bg.jpg"
    file_location = f"html/{file_static_location}"
    cv2.imwrite(file_location, blur_bg)
    return templates.TemplateResponse("show.html",{"request":request, "image": upload_obj.img, "detect": file_static_location})

@Image_editor.post('/change_bg', response_class=HTMLResponse)
def perform_change_bg(request: Request, index:str=Form(...), image_file: UploadFile = File(...)):
    im = cv2.imread('html/'+upload_obj.img)
    with open('html/static/input/bg.jpg', "wb") as buffer:
        shutil.copyfileobj(image_file.file, buffer)
    bg = cv2.imread('html/static/input/bg.jpg')
    resized_bg = cv2.resize(bg, (600, 600))
    change_bg_file = upload_obj.obj.change_bg_image(resized_bg, index)
    file_static_location = f"static/features/change_bg_file.jpg"
    file_location = f"html/{file_static_location}"
    cv2.imwrite(file_location, change_bg_file)
    return templates.TemplateResponse("show.html",{"request":request, "image": upload_obj.img, "detect": file_static_location})

@Image_editor.post('/get_roi', response_class=HTMLResponse)
def perform_bg(request: Request, index:str=Form(...)):
    im = cv2.imread('html/'+upload_obj.img)
    roi = upload_obj.obj.get_idx(index)
    file_static_location = f"static/features/roi.jpg"
    file_location = f"html/{file_static_location}"
    cv2.imwrite(file_location, roi)
    return templates.TemplateResponse("show.html",{"request":request, "image": upload_obj.img, "detect": file_static_location})

@Image_editor.post('/upload_file')
async def handle_image(upload_file:UploadFile = File(...)):
    filename=upload_file.filename
    file_static_location = f"static/input/{filename}"
    file_location = f"html/{file_static_location}"
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)
    
    # upload_obj.obj.image_name = file_static_location
    upload_obj.set_image(file_static_location)
    obj = upload_obj.set_object()

    return {"message":"success","statuscode":200}

@Image_editor.get('/show')
async def show_image(request: Request):
    return templates.TemplateResponse("show.html",{"request":request, "image": upload_obj.img})

