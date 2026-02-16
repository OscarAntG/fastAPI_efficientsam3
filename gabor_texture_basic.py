import os
os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda/bin/ptxas" 
os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from sam3.sam3.model_builder import build_efficientsam3_image_model
from sam3.sam3.model.sam3_image_processor import Sam3Processor

# --- Model Setup ---
checkpoint = "efficient_sam3_efficientvit_s.pt"
model = build_efficientsam3_image_model(
    checkpoint_path=checkpoint,
    backbone_type="efficientvit",
    model_name="b0", 
    enable_inst_interactivity=True,
).cuda()
model.eval()
processor = Sam3Processor(model)

# --- Load Image ---
res = 1024
img_path1 = "waste_images/gray_black_center.jpg"
img_path2 = "waste_images/green_yellow_sludge.jpg"
img_path3 = "waste_images/hanford_yellow.jpg"
img_path4 = "waste_images/orange_yellow_sludge.jpg"
synth_img = Image.open("whale.jpg").convert("RGB")
synth_img = synth_img.resize((res, res))
w, h = synth_img.size

# --- Gabor Filter setup ---
def apply_gabor_filter_bank(image_pil):
    img_cv = np.array(image_pil.convert('L'))
    kernels = []
    num_orientations = 8
    for theta in np.arange(0, np.pi, np.pi/num_orientations):
        params = {'ksize':(21,21), 'sigma':1.5, 'theta':theta, 'lambd':3.0, 'gamma':1.0, 'psi':0, 'ktype':cv2.CV_32F}
        kernel = cv2.getGaborKernel(**params)
        kernels.append(kernel)

    combined_response = np.zeros_like(img_cv, dtype=np.float32)
    for kernel in kernels:
        fimg = cv2.filter2D(img_cv, cv2.CV_8UC3, kernel)
        np.maximum(combined_response,fimg, combined_response)

    combined_response = cv2.normalize(combined_response, None, 0, 255, cv2.NORM_MINMAX)
    return combined_response.astype(np.uint8)

texture_map = apply_gabor_filter_bank(synth_img)

# --- Interactive Point Selection ---
print("\n[USER INPUT REQUIRED]")
print("A window will open. Click ONCE on the image to select the waste target.")
print("The script will continue after your click.")

plt.figure(figsize=(8, 8))
plt.imshow(synth_img)
plt.title("Select Target 1: (positive) and Target 2: (negative)")
plt.axis('on') # Keep axis on to help you verify coordinates

selected_points = plt.ginput(2, timeout=0) 
plt.close() 

if len(selected_points) < 2:
    print("Insufficient points selected.")
    exit()

posx, posy = selected_points[0]
negx, negy = selected_points[1]
print(f"Positive target: x={posx:.1f}, y={posy:.1f}")
print(f"Negative target: x={negy:.1f}, y={negy:.1f}")

# Convert to the format SAM3 expects
points = np.array([[posx, posy], [negx, negy]], dtype=np.float32)
labels = np.array([1, 0], dtype=np.int32) 

# --- Inference Logic ---
inference_state = processor.set_image(texture_map)

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

# --- Visualization ---
num_masks = len(masks)
fig, axes = plt.subplots(1, num_masks + 3, figsize=(20, 5))

# Plot 1: Original + Selected Point
axes[0].imshow(synth_img)
axes[0].scatter([posx], [posy], color='yellow', marker='*', s=150, edgecolors='black', label='+')
axes[0].scatter([negx], [negy], color='blue', marker='*', s=150, edgecolors='black', label='-')
axes[0].set_title(f"Input Prompts")
axes[0].axis('off')

# Plot 2: Gabor texture map
axes[4].imshow(texture_map, cmap='magma')
axes[4].set_title("Gabor map (highlights waste)")
axes[4].axis('off')

colors = plt.colormaps.get_cmap('tab10')

for i in range(num_masks):
    mask = masks[i]
    mask_viz = np.zeros((res, res, 3))
    mask_viz[mask > 0] = colors(i)[:3]
    axes[i+1].imshow(mask_viz)
    axes[i+1].set_title(f"Mask {i}\nScore: {scores[i]:.2f}")
    axes[i+1].axis('off')

# Combined view
combined_viz = np.array(synth_img).astype(float) / 255.0
for i in range(num_masks):
    mask = masks[i]
    mask_rgb = np.zeros_like(combined_viz)
    mask_rgb[mask > 0] = colors(i)[:3]
    combined_viz = np.where(mask[..., None] > 0, combined_viz * 0.5 + mask_rgb * 0.5, combined_viz)

axes[-1].imshow(combined_viz)
axes[-1].set_title("Combined Overlays")
axes[-1].axis('off')

plt.tight_layout()
plt.savefig("result_interactive.jpg")
print("\nProcessing complete.")
print("Detailed result saved to result_interactive.jpg")
plt.show()