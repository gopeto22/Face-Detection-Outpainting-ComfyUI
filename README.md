# Face-Detection-ComfyUI
This repository contains custom nodes designed to work with ComfyUI for face detection, restoration, and visualization using YuNet and dlib models. The goal is to allow smoother implementation of face detection models. This then ensures a correct cropping node extracts the face around the edges. Afterwards, ```lllyasviel/fooocus_inpaint``` outpainting model is applied, allowing customising any background desired by the user. These nodes are designed to handle high-resolution images and scratched images, making them versatile for various use cases.

FEATURES
- Face Detection with YuNet and dlib: Detects faces in images using the YuNet model.
- Visualization of Landmarks: Visualizes the facial landmarks detected during face detection.
- Cropping Faces: Automatically crops faces from the detected bounding boxes.

INSTALLATION:
1.Clone this repository:
```
git clone https://github.com/your-repo/comfy-yunet.git
cd comfy-yunet
```
2. Install dependencies:
- Ensure that ComfyUI, torch, opencv-python, and Pillow are installed in your environment.
3. Set up your environment and ensure all necessary models and resources are available.

USAGE:
1. Run the pipeline: The process begins with image input and moves through stages including face detection, restoration, and blending.
Example command:
```
python predict.py --image <path-to-image> --HR False --with_scratch False
```
2. Customization: You can adjust parameters such as HR for high-resolution images and with_scratch to handle scratched images.

FILE OVERVIEW:
- run.py: Contains the main process flow for running the restoration and face detection pipelines.
- predict.py: Executes the face detection and visualization pipeline using custom nodes.
- nodes.py: Defines the custom nodes for detection, cropping, and image generation.
- __init__.py: Initializes the project.

EXAMPLE WORKFLOW:
- Face Detection and Visualization: uses the YuNet model to detect faces and visualize landmarks.
- Face Cropping: After the face is detected, it is cropped around the edges
- Outpainting Application: The outpaint model is applied allowing the user to provide text prompt input and set a desired background of the outpainted image

CONTRIBUTING:
Feel free to fork this repository, submit issues, and make pull requests. Contributions are welcome.

LICENSE:
This project is licensed under the MIT License.





