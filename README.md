# fastAPI_efficientsam3

Package implementing [efficientsam3](https://github.com/SimonZeng7108/efficientsam3) and FastAPI for the Dry Waste Retrieval Project. This repository hosts the server counterpart to `ur20_zed_efficientsam3` and excludes any ROS2 functionality due to Python environment compatibility constraints between ROS2 and `efficientsam3`.

**The repository includes:**
* `efficientsam3` model checkpoint and required files
* Test images for SAM3 segmentation
* Segmentation output results
* Various scripts implementing FastAPI, SAM3, or both

*(Note: The only script actively used in the final application is `segmentation_server_sam3.py`)*

## Dependencies
* FastAPI (`FastAPI`)
* [efficientsam3](https://github.com/SimonZeng7108/efficientsam3)
* Anaconda (`conda`)

## `segmentation_server_sam3.py`
The primary Python script that initializes a FastAPI server and an instance of `efficientsam3`. Through a client request, the model receives an OpenCV image along with two sets of pixel coordinates:
* **Positive Prompt:** Coordinates targeting the waste object to be segmented.
* **Negative Prompt:** Coordinates targeting the background to be explicitly excluded.

These prompts are used to generate a binary mask over the original image, with boundaries determined by the AI based on color, object texture, and edge detection.

**Image Pre-processing (Gabor Filter):**
To facilitate accurate segmentation, a Gabor filter is applied to the incoming data. This filter highlights boundaries by generating a texture map of the original image, which is then combined with the original image via alpha blending on the blue channel.

The resulting mask is sent as a response back to the FastAPI client for further ROS2 processing.
