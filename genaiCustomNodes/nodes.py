from PIL import Image
import numpy as np
import cv2 as cv
import torch
import os
import dlib
import matplotlib.pyplot as plt

# Class responsible for loading the YuNet face detection model.
class GENAI_LoadYuNetModel:
    RETURN_TYPES = ("YUNET_MODEL",)
    RETURN_NAMES = ("yunet_model",)
    FUNCTION = "run"
    CATEGORY = "bringing old photos back to life/loaders"
    OUTPUT_NODE = True

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": ("STRING", {"default": "/home/georgi/Downloads/face_detection_yunet_2023mar.onnx"}),
            }
        }

    @staticmethod
    def load_model(model_path: str):
        """
        Load the YuNet face detection model from an ONNX file.

        Args:
            model_path (str): The file path to the YuNet model.

        Returns:
            cv.FaceDetectorYN: The initialized YuNet face detector object.
        """
        print(f"Loading YuNet model from: {model_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ONNX model file not found: {model_path}")
        
        # Load YuNet model for detecting faces in images.
        detector = cv.FaceDetectorYN.create(
            model_path,
            "",
            (320, 320),
            0.6,  # score_threshold: Minimum confidence score for detecting a face.
            0.3,  # nms_threshold: Non-maximum suppression threshold for face bounding boxes.
            5000  # top_k: Maximum number of faces to detect.
        )
        return detector

    def run(self, model_path: str):
        """
        Run the node to load the YuNet model.

        Args:
            model_path (str): The file path to the YuNet model.

        Returns:
            tuple: A tuple containing the YuNet face detection model.
        """
        return (GENAI_LoadYuNetModel.load_model(model_path),)

# Class for detecting faces in images using the YuNet model.
class GENAI_DetectFacesYuNet:
    RETURN_TYPES = ("FACE_DETECTIONS",)
    RETURN_NAMES = ("face_detections",)
    FUNCTION = "run"
    CATEGORY = "bringing old photos back to life/image"
    OUTPUT_NODE = True

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "yunet_model": ("YUNET_MODEL",),  # Pre-loaded YuNet model.
                "image": ("IMAGE",),  # Input image where faces are to be detected.
                "scale": ("FLOAT", {"default": 1.0}),  # Scale factor to resize the image.
            }
        }

    @staticmethod
    def detect_faces(yunet_model, image, scale):
        """
        Detect faces in the provided image using the YuNet model.

        Args:
            yunet_model (cv.FaceDetectorYN): The loaded YuNet model.
            image (PIL.Image.Image): The input image.
            scale (float): Scale factor for resizing the image before detection.

        Returns:
            tuple: A tuple containing face detection results and bounding boxes.
        """
        img = np.array(image)

        # Scale the image while preserving its aspect ratio.
        img = cv.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))

        # Ensure the image is in RGB format for face detection.
        if len(img.shape) == 2 or img.shape[2] == 1:
            img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)

        # Set the input size for the face detector based on the image dimensions.
        yunet_model.setInputSize((img.shape[1], img.shape[0]))
        faces = yunet_model.detect(img)

        # Collect bounding boxes for detected faces.
        bounding_boxes = []
        if isinstance(faces, tuple) and faces[1] is not None:
            for face in faces[1]:
                box = face[:-1].tolist()
                if all(np.isfinite(box)) and all(abs(b) < 1e6 for b in box):
                    bounding_boxes.append(box)
                    print("Valid bounding box:", box)
                else:
                    print(f"Invalid bounding box detected: {box}")
        else:
            print("No valid faces detected.")

        return faces, bounding_boxes

    def run(self, yunet_model, image, scale):
        """
        Run the node to detect faces in the input image using YuNet.

        Args:
            yunet_model (cv.FaceDetectorYN): The loaded YuNet model.
            image (PIL.Image.Image): The input image for face detection.
            scale (float): Scale factor for resizing the image.

        Returns:
            tuple: A tuple containing face detections.
        """
        faces, bounding_boxes = GENAI_DetectFacesYuNet.detect_faces(yunet_model, image, scale)
        if not bounding_boxes:
            print("No valid bounding boxes found.")
        return faces

# Class for visualizing face detection results by drawing bounding boxes.
class GENAI_VisualizeDetectedFaces:
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image_with_detections",)
    FUNCTION = "run"
    CATEGORY = "bringing old photos back to life/visualization"
    OUTPUT_NODE = True

    def __init__(self):
        pass

    @staticmethod
    def display_image_with_detections(image, bounding_boxes):
        """
        Display the image with drawn bounding boxes around detected faces.

        Args:
            image (torch.Tensor or np.ndarray): The input image.
            bounding_boxes (list): A list of bounding boxes around the detected faces.
        """
        # Convert image tensor to numpy array if necessary.
        if isinstance(image, torch.Tensor):
            image = image.permute(1, 2, 0).cpu().numpy()  # Convert from (C, H, W) to (H, W, C)

        # Draw bounding boxes on the image.
        for bbox in bounding_boxes:
            x, y, w, h = bbox
            cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Use matplotlib to display the image with bounding boxes.
        plt.imshow(image)
        plt.axis("off")  # Hide axis labels.
        plt.show()

    def run(self, image, face_detections):
        """
        Run the node to visualize face detection by drawing bounding boxes.

        Args:
            image (torch.Tensor or np.ndarray): The input image.
            face_detections (tuple): Face detections including bounding boxes.

        Returns:
            torch.Tensor: The original input image tensor (unchanged).
        """
        # Extract bounding boxes from face detection data.
        bounding_boxes = []
        if isinstance(face_detections, tuple) and len(face_detections) > 1 and face_detections[1] is not None:
            for face in face_detections[1]:
                box = face[:-1].tolist()
                bounding_boxes.append(box)

        # Display image with bounding boxes for debugging.
        GENAI_VisualizeDetectedFaces.display_image_with_detections(image, bounding_boxes)

        # Return the original image tensor so the pipeline can proceed.
        return image
    
# Class for applying face detection using a pre-trained dlib model.
class GENAI_ApplyFaceDetectionModel:
    RETURN_TYPES = ("TENSOR",)
    RETURN_NAMES = ("detected_faces",)
    FUNCTION = "run"
    CATEGORY = "face_detection"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),  # Input image to detect faces.
                "dlib_model": ("DLIB_MODEL",),  # Dlib face detection model.
            },
        }

    def apply_model(self, image, dlib_model):
        """
        Detect faces using the dlib face detector.

        Args:
            image (PIL.Image.Image): Input image for face detection.
            dlib_model (dlib.shape_predictor): Dlib face detection model.

        Returns:
            torch.Tensor: Detected face landmarks as a tensor.
        """
        # Load dlib's frontal face detector.
        detector = dlib.get_frontal_face_detector()
        predictor = dlib_model

        # Convert input to RGB PIL Image if not already in that format.
        if not isinstance(image, Image.Image):
            raise ValueError("Input image must be a PIL Image.")
        image = image.convert("RGB")

        # Convert the image to a numpy array for face detection.
        image_np = np.array(image)

        # Ensure the image is in 8-bit format for dlib processing.
        if image_np.dtype != np.uint8:
            raise ValueError("Image must be in 8-bit format.")

        # Convert the image to grayscale for face detection.
        image_gray = cv.cvtColor(image_np, cv.COLOR_RGB2GRAY)

        # Detect faces in the image.
        faces = detector(image_gray)

        # Store the detected facial landmarks.
        detected_faces = []
        for face in faces:
            shape = predictor(image_gray, face)
            detected_faces.append([(p.x, p.y) for p in shape.parts()])

        # Convert detected face landmarks to a tensor.
        detected_faces_tensor = torch.tensor(detected_faces)

        print("Detected faces tensor shape:", detected_faces_tensor.shape)

        return detected_faces_tensor

    def run(self, image, dlib_model):
        """
        Run the node to apply face detection using dlib model.

        Args:
            image (PIL.Image.Image): Input image.
            dlib_model (dlib.shape_predictor): Pre-trained dlib face detector.

        Returns:
            torch.Tensor: Tensor containing detected face landmarks.
        """
        # Convert the input image to PIL Image if necessary.
        if isinstance(image, torch.Tensor):
            image = Image.fromarray(image.permute(1, 2, 0).cpu().numpy())
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        elif not isinstance(image, Image.Image):
            raise ValueError("Input must be a PIL Image, numpy array, or tensor.")

        return self.apply_model(image, dlib_model)

# Class for visualizing YuNet face detection landmarks.
class GENAI_VisualizeYuNetLandmarks:
    RETURN_TYPES = ("IMAGE", "LIST")
    RETURN_NAMES = ("image_with_landmarks", "bounding_boxes")
    FUNCTION = "run"
    CATEGORY = "bringing old photos back to life/visualization"
    OUTPUT_NODE = True

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),  # Input image.
                "face_detections": ("FACE_DETECTIONS",),  # Face detection results.
                "thickness": ("INT", {"default": 2}),  # Thickness of bounding boxes and landmarks.
            }
        }

    @staticmethod
    def visualize_landmarks(image, face_detections, thickness):
        """
        Visualize the facial landmarks and bounding boxes in the image.

        Args:
            image (torch.Tensor or np.ndarray): The input image.
            face_detections (tuple): Face detections with landmark data.
            thickness (int): Thickness for drawing the bounding boxes and landmarks.

        Returns:
            np.ndarray: Image with landmarks drawn on it.
            list: List of bounding boxes for detected faces.
        """
        img = np.array(image)
        bounding_boxes = []
        
        # Process detected faces and extract landmarks for each.
        if isinstance(face_detections, tuple) and len(face_detections) > 1 and face_detections[1] is not None:
            for face in face_detections[1]:
                coords = face[:-1].astype(np.int32)  # Extract the bounding box coordinates.
                bounding_boxes.append(coords[:4].tolist())
                cv.rectangle(img, (coords[0], coords[1]), (coords[0] + coords[2], coords[1] + coords[3]), (0, 255, 0), thickness)
                # Draw facial landmarks.
                for i in range(4, 14, 2):
                    cv.circle(img, (coords[i], coords[i + 1]), 2, (0, 255, 255), thickness)
        else:
            print("No valid face detections available.")

        return img, bounding_boxes

    def run(self, image, face_detections, thickness):
        """
        Run the node to visualize detected landmarks in the image.

        Args:
            image (torch.Tensor or np.ndarray): Input image.
            face_detections (tuple): Face detections including landmarks.
            thickness (int): Thickness for the bounding boxes and landmarks.

        Returns:
            torch.Tensor: Image tensor with landmarks.
            list: Bounding boxes of detected faces.
        """
        img_with_landmarks, bounding_boxes = GENAI_VisualizeYuNetLandmarks.visualize_landmarks(image, face_detections, thickness)

        # Ensure correct shape and format of the processed image.
        if len(img_with_landmarks.shape) == 4 and img_with_landmarks.shape[0] == 1:
            img_with_landmarks = np.squeeze(img_with_landmarks, axis=0)
        if len(img_with_landmarks.shape) == 2:
            img_with_landmarks = cv.cvtColor(img_with_landmarks, cv.COLOR_GRAY2RGB)
        elif len(img_with_landmarks.shape) == 3 and img_with_landmarks.shape[2] == 1:
            img_with_landmarks = cv.cvtColor(img_with_landmarks, cv.COLOR_GRAY2RGB)

        # Convert the image back to tensor format.
        img_with_landmarks_tensor = torch.from_numpy(img_with_landmarks).permute(2, 0, 1).float() / 255.0

        # Return the image tensor and bounding boxes for further processing.
        return img_with_landmarks_tensor, bounding_boxes

# Class for cropping the image based on the detected bounding box.
class GENAI_CropImageByBoundingBox:
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("cropped_face",)
    FUNCTION = "run"
    CATEGORY = "bringing old photos back to life/image"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),  # Input image.
                "bounding_box": ("LIST",),  # Bounding box coordinates.
            },
        }

    @staticmethod
    def crop_image_by_bounding_box(image, bounding_box):
        """
        Crop the image based on the bounding box coordinates.

        Args:
            image (torch.Tensor, np.ndarray, or PIL.Image.Image): The input image.
            bounding_box (list): List of bounding box coordinates [x, y, width, height].

        Returns:
            PIL.Image.Image: Cropped image containing the face.
        """
        # Validate the bounding box input.
        if len(bounding_box) != 4:
            raise ValueError(f"Bounding box must have 4 values (x, y, width, height), but got {len(bounding_box)}: {bounding_box}")

        # Extract bounding box coordinates.
        x, y, w, h = bounding_box

        # Convert the input image to a numpy array if necessary.
        if isinstance(image, torch.Tensor):
            image = image.permute(1, 2, 0).cpu().numpy()
        elif isinstance(image, np.ndarray):
            pass  # Already in numpy format.
        elif isinstance(image, Image.Image):
            image = np.array(image)
        else:
            raise ValueError("Input must be a valid image format (PIL Image, NumPy array, or tensor).")

        # Crop the image using the bounding box coordinates.
        cropped_image_np = image[y:y+h, x:x+w]

        # Convert the cropped image back to PIL format.
        cropped_image = Image.fromarray(cropped_image_np)

        return cropped_image

    def run(self, image, bounding_box):
        """
        Run the node to crop the image based on the bounding box.

        Args:
            image (torch.Tensor, np.ndarray, or PIL.Image.Image): Input image.
            bounding_box (list): Bounding box coordinates.

        Returns:
            torch.Tensor: Tensor containing the cropped face.
        """
        # Handle invalid bounding box cases.
        if len(bounding_box) != 4:
            print(f"Invalid bounding box: {bounding_box}. Skipping crop.")
            return image

        # Perform the cropping operation.
        cropped_face = GENAI_CropImageByBoundingBox.crop_image_by_bounding_box(image, bounding_box)

        # Convert the cropped image back to tensor format.
        cropped_face_tensor = torch.from_numpy(np.array(cropped_face)).permute(2, 0, 1).float() / 255.0

        return (cropped_face_tensor,)
    
# Class for visualizing landmarks
class GENAI_VisualizeLandmarks:
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image_with_landmarks",)
    FUNCTION = "run"
    CATEGORY = "visualization"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),  # Input image
                "landmarks_tensor": ("TENSOR",),  # Landmarks tensor containing the (x, y) coordinates
            },
        }

    def apply_landmarks(self, image, landmarks_tensor):
        """
        Apply the landmarks on the input image.

        Args:
            image (PIL.Image.Image or torch.Tensor or np.ndarray): Input image.
            landmarks_tensor (torch.Tensor): Tensor containing the landmarks (x, y) coordinates.

        Returns:
            PIL.Image.Image: Image with landmarks.
        """
        # Convert the image to a numpy array (OpenCV format)
        image_np = np.array(image)

        # Ensure the image is in RGB format
        if len(image_np.shape) == 2 or image_np.shape[2] != 3:
            raise ValueError("Image must be in RGB format.")

        # Convert the tensor to a list of landmarks (x, y) coordinates
        landmarks = landmarks_tensor.tolist()

        # Loop through each landmark and draw it on the image
        for face_landmarks in landmarks:
            for (x, y) in face_landmarks:
                # Draw a circle for each landmark point
                cv.circle(image_np, (int(x), int(y)), radius=2, color=(0, 255, 0), thickness=-1)

        # Convert the image back to PIL format for ComfyUI
        image_with_landmarks = Image.fromarray(image_np)

        return image_with_landmarks

    def run(self, image, landmarks_tensor):
        """
        Run the node to visualize landmarks on the input image.

        Args:
            image (PIL.Image.Image or torch.Tensor or np.ndarray): Input image.
        """
        # Check if the tensor is empty
        if landmarks_tensor.nelement() == 0:
            raise ValueError("Landmarks tensor is empty.")

        # Convert the input image (which may be a tensor) to a PIL Image
        if isinstance(image, torch.Tensor):
            image_np = image.cpu().numpy()
            image_np = np.squeeze(image_np)
            image = Image.fromarray(image_np)

        # Apply the landmarks on the image
        image_with_landmarks = self.apply_landmarks(image, landmarks_tensor)

        return image_with_landmarks
    
# Class for generating images
class GENAI_ImageGenerate:
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("img",)
    FUNCTION = "run"
    CATEGORY = "image"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "img": ("IMAGE",),
                "title": ("STRING", {"default": "Image"}),
            }
        }

    @staticmethod
    def show_image(img, title="Image"):
        """
        Display the input image.

        Args:
            img (np.ndarray or PIL.Image.Image): Input image.
            title (str): Title for the image window.
        """
        if isinstance(img, np.ndarray):
            # Convert from BGR to RGB if loaded via OpenCV
            if img.shape[2] == 3:  # Color image
                img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            Image.fromarray(img).show(title=title)
        elif isinstance(img, Image.Image):
            img.show(title=title)

    def run(self, img, title="Image"):
        """
        Run the node to generate and display an image.

        Args:
            img (np.ndarray or PIL.Image.Image): Input image.
            title (str): Title for the image window.

        Returns:
            tuple: Tuple containing the input image.
        """
        GENAI_ImageGenerate.show_image(img, title)
        return (img,)
