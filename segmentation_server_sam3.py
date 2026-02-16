import torch
import numpy as np
import cv2
from PIL import Image
from io import BytesIO


import uvicorn
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import Response
from contextlib import asynccontextmanager

from sam3.sam3.model_builder import build_efficientsam3_image_model
from sam3.sam3.model.sam3_image_processor import Sam3Processor

model = None
processor = None

# --- Model Setup ---
@asynccontextmanager
async def lifespan(app:FastAPI):
    global model, processor
    print("Loading EfficientSAM3 model...")
    checkpoint = "efficient_sam3_efficientvit_s.pt"

    try:
        model = build_efficientsam3_image_model(
            checkpoint_path=checkpoint,
            backbone_type="efficientvit",
            model_name="b0", 
            enable_inst_interactivity=True,
        ).cuda()
        model.eval()
        processor = Sam3Processor(model)
        print("MODEL LOADED SUCCESSFULLY")
    except Exception as e:
        print(f"MODEL LOADING ERROR: {e}")

    yield

    # -- Shutdown sequence --
    print("Shutting down server...")
    if torch.cuda_is_available():
        torch.cuda.empty_cache()

app = FastAPI(lifespan=lifespan)

# --- Gabor Filter setup ---
def apply_gabor_filter_bank(image_pil):
    img_cv = np.array(image_pil.convert('L'))
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    img_cv = clahe.apply(img_cv)

    kernels = []
    num_orientations = 8
    for theta in np.arange(0, np.pi, np.pi/num_orientations):
        kernel = cv2.getGaborKernel(
            ksize=(21,21),
            sigma=1.0,
            theta=theta,
            lambd=3.0,
            gamma=1.0,
            psi=0,
            ktype=cv2.CV_32F
        )
        kernel/= 1.5*kernel.sum() if kernel.sum() > 0 else 1
        kernels.append(kernel)

    combined_response = np.zeros_like(img_cv, dtype=np.float32)
    for kernel in kernels:
        fimg = cv2.filter2D(img_cv, cv2.CV_32F, kernel)
        fimg = np.abs(fimg)
        combined_response = np.maximum(combined_response,fimg)

    combined_response = cv2.normalize(combined_response, None, 0, 255, cv2.NORM_MINMAX)
    return combined_response.astype(np.uint8)

@app.post("/segment")
async def segment_image(
    image:UploadFile = File(...),
    pos_x: float = Form(...),
    pos_y: float = Form(...),
    neg_x: float = Form(...),
    neg_y: float = Form(...)
):
    if model is None:
        return Response(content=b"Model not loaded", status_code=503)
    
    image_bytes = await image.read()
    pil_img = Image.open(BytesIO(image_bytes)).convert("RGB")

    texture_map = apply_gabor_filter_bank(pil_img)

    # --- Alpha blending (blue-channel) ---
    img_np = np.array(pil_img).astype(float)
    alpha = 0.6
    alpha_img = img_np.copy()
    texture_layer = texture_map.astype(float)

    alpha_img[:, :, 2] = (img_np[:, :, 2]* (1-alpha)) + (texture_layer * alpha)
    alpha_img = np.clip(alpha_img, 0, 255).astype(np.uint8)

    alpha_img = Image.fromarray(alpha_img)

    # Convert to the format SAM3 expects
    points = np.array([[pos_x, pos_y], [neg_x, neg_y]], dtype=np.float32)
    labels = np.array([1, 0], dtype=np.int32)

    # --- Inference Logic ---
    inference_state = processor.set_image(alpha_img)

    try:
        masks, scores, _ = model.predict_inst(
            inference_state, 
            point_coords=points, 
            point_labels=labels,
        )
    except AttributeError:
        inference_state = processor.set_point_prompt(
            state=inference_state, 
            point_coords=points, 
            point_labels=labels,
        )
        masks = inference_state.get("masks", [])
        scores = inference_state.get("scores", [0.0] * len(masks))

    if len(masks) > 0:
        # Change logic to use best mask
        scores_np = np.array(scores)
        best_idx = np.argmax(scores_np)
        best_score = scores_np[best_idx]
        
        print(f"Found {len(masks)} masks. Selecting Index {best_idx} with score: {best_score:.4f}")
        
        best_mask = masks[best_idx]

        _, mask_encoded = cv2.imencode('.jpg', (best_mask*255).astype(np.uint8), [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        return Response(content=mask_encoded.tobytes(), media_type="image/jpeg")
    else:
        return Response(content=b"No mask found", status_code=500)
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)