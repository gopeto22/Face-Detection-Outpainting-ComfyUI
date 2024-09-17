from .nodes import GENAI_LoadYuNetModel, GENAI_DetectFacesYuNet, GENAI_VisualizeYuNetLandmarks, \
                    GENAI_VisualizeLandmarks, GENAI_CropImageByBoundingBox, GENAI_ImageGenerate, \
                    GENAI_VisualizeDetectedFaces

# This section defines a mapping between node class names (as strings) and the actual Python classes that implement them.
# This is useful in a system where nodes are dynamically instantiated based on their string names.
# For example, "GENAI_LoadYuNetModel" corresponds to the class GENAI_LoadYuNetModel, allowing the system
# to dynamically load and use the correct class based on the node's name in the pipeline.

NODE_CLASS_MAPPINGS = {
    "GENAI_LoadYuNetModel": GENAI_LoadYuNetModel,  # Maps the node name "GENAI_LoadYuNetModel" to its class.
    "GENAI_DetectFacesYuNet": GENAI_DetectFacesYuNet,  # Maps the node name "GENAI_DetectFacesYuNet" to its class.
    "GENAI_VisualizeYuNetLandmarks": GENAI_VisualizeYuNetLandmarks,  # Maps the node name "GENAI_VisualizeYuNetLandmarks" to its class.
    "GENAI_VisualizeLandmarks": GENAI_VisualizeLandmarks,  # Maps the node name "GENAI_VisualizeLandmarks" to its class.
    "GENAI_CropImageByBoundingBox": GENAI_CropImageByBoundingBox,  # Maps the node name "GENAI_CropImageByBoundingBox" to its class.
    "GENAI_ImageGenerate": GENAI_ImageGenerate,  # Maps the node name "GENAI_ImageGenerate" to its class.
    "GENAI_VisualizeDetectedFaces": GENAI_VisualizeDetectedFaces,  # Maps the node name "GENAI_VisualizeDetectedFaces" to its class.
}

# This section defines a mapping between the node class names and their display names.
# The display names are the user-friendly names that will appear in the UI or logs when the nodes are presented to users.
# This allows the system to present meaningful names to users without requiring them to understand the underlying class names.

NODE_DISPLAY_NAME_MAPPINGS = {
    "GENAI_LoadYuNetModel": "Load YuNet Model",  # Display name for the GENAI_LoadYuNetModel node.
    "GENAI_DetectFacesYuNet": "Detect Faces (YuNet)",  # Display name for the GENAI_DetectFacesYuNet node.
    "GENAI_VisualizeYuNetLandmarks": "Visualize YuNet Landmarks",  # Display name for the GENAI_VisualizeYuNetLandmarks node.
    "GENAI_VisualizeLandmarks": "Visualize Landmarks",  # Display name for the GENAI_VisualizeLandmarks node.
    "GENAI_CropImageByBoundingBox": "Crop Image by Bounding Box",  # Display name for the GENAI_CropImageByBoundingBox node.
    "GENAI_ImageGenerate": "Image Generate",  # Display name for the GENAI_ImageGenerate node.
    "GENAI_VisualizeDetectedFaces": "Visualize Detected Faces",  # Display name for the GENAI_VisualizeDetectedFaces node.
}

# This list defines the public interface for the module. Only the variables and functions listed here will be accessible 
# when this module is imported elsewhere. This serves as a way to control what is exposed from the module.
# By restricting the export, you prevent internal variables or functions from being accessible outside this module.

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
