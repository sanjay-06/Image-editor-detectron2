## Start With Detectron2
### For Windows
<br>

Download and Install Anaconda

Open Anaconda Prompt and create a new virtual environment by using the command:

```
conda create -n object_detect python=3.8
```

Activate the environment:

```
conda activate object_detect
```
Once done, install cython:
```
pip install cython
```
Install Pytorch and CUDA:
```
conda install pytorch torchvision torchaudio cudatoolkit=11.0 -c pytorch
```
Clone Detectron2 repository:
```
git clone https://github.com/facebookresearch/detectron2.git
```
Got to the downloaded repository folder:
```
cd detectron2
```
Install dependencies:
```
pip install -e .
```
Then install OpenCV:
```
pip install opencv-python
```

# Features

## Upload Image Here
<br>

![upload](https://raw.githubusercontent.com/sanjay-06/Image-editor-detectron2/master/images/output/upload.jpg)

## Editor
<br>

![display](images/output/editor.jpg)

## Object Detection using detectron2
<br>

![display](images/output/object_detect.png)

## select Interest mask
<br>

![display](images/output/get_roi.png)

## Clone
<br>

![display](images/output/clone.png)



## Blur Background
<br>

![display](images/output/blur_bg.png)

<br>

![display](images/output/blur_box.png)

## Change Background
<br>

![display](images/output/change_bg.png)

## SIFT
<br>

![display](images/output/sift.png)

## HARRIS
<br>

![display](images/output/harris.png)

## Sepia
<br>

![display](images/output/sepia.png)

## Vintage
<br>

![display](images/output/vintage.png)

<br>

```
Frontend => Tailwindcss, HTML
Backend  => FastAPI
Model    => detectron2
```
