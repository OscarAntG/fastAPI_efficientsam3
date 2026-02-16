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
).half().cuda()
model.eval()
processor = Sam3Processor(model)

# --- Load Image ---
res = 720
img_path1 = "waste_images/gray_black_center.jpg"
img_path2 = "waste_images/green_yellow_sludge.jpg"
img_path3 = "waste_images/hanford_yellow.jpg"
img_path4 = "waste_images/orange_yellow_sludge.jpg"
synth_img = Image.open("clay.jpg").convert("RGB")
synth_img = synth_img.resize((res, res))
w, h = synth_img.size

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
        fimg = cv2.filter2D(img_cv, cv2.CV_8UC3, kernel)
        fimg = np.abs(fimg)
        combined_response = np.maximum(combined_response,fimg)

    combined_response = cv2.normalize(combined_response, None, 0, 255, cv2.NORM_MINMAX)
    return combined_response.astype(np.uint8)

texture_map = apply_gabor_filter_bank(synth_img)

# --- Alpha blending (blue-channel) ---
img_np = np.array(synth_img).astype(float)
alpha = 0.6
alpha_img = img_np.copy()
texture_layer = texture_map.astype(float)

alpha_img[:, :, 2] = (img_np[:, :, 2]* (1-alpha)) + (texture_layer * alpha)
alpha_img = np.clip(alpha_img, 0, 255).astype(np.uint8)

#alpha_img = np.array(Image.fromarray(alpha_img))
alpha_img = torch.tensor(alpha_img, device="cpu", dtype=torch.float16)

# --- Interactive Point Selection ---
print("\n[USER INPUT REQUIRED]")
print("A window will open. Click ONCE on the image to select the waste target.")
print("The script will continue after your click.")

plt.figure(figsize=(8, 8))
plt.imshow(synth_img)
plt.title("Select Target 1 & 2: (positive) and Target 3 & 4: (negative)")
plt.axis('on') # Keep axis on to help you verify coordinates

selected_points = plt.ginput(2, timeout=0) 
plt.close() 

if len(selected_points) < 2:
    print("Insufficient points selected.")
    exit()

# posx0, posy0 = selected_points[0]
posx1, posy1 = selected_points[0]
negx2, negy2 = selected_points[1]
# negx3, negy3 = selected_points[3]
# print(f"Positive target: x={posx0:.1f}, y={posy0:.1f}")
print(f"Positive target: x={posx1:.1f}, y={posy1:.1f}")
print(f"Negative target: x={negy2:.1f}, y={negy2:.1f}")
# print(f"Negative target: x={negy3:.1f}, y={negy3:.1f}")

# Convert to the format SAM3 expects
points = np.array([[posx1, posy1], [negx2, negy2]], dtype=np.float32)
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

# --- Visualization ---
num_masks = len(masks)
fig, axes = plt.subplots(1, num_masks + 3, figsize=(20, 5))

# Plot 1: Original + Selected Point
axes[0].imshow(synth_img)
# axes[0].scatter([posx0], [posy0], color='yellow', marker='*', s=150, edgecolors='black', label='+')
axes[0].scatter([posx1], [posy1], color='yellow', marker='*', s=150, edgecolors='black', label='+')
axes[0].scatter([negx2], [negy2], color='blue', marker='*', s=150, edgecolors='black', label='-')
# axes[0].scatter([negx3], [negy3], color='blue', marker='*', s=150, edgecolors='black', label='-')
axes[0].set_title(f"Input Prompts")
axes[0].axis('off')

# Plot 2: Alpha blended image
axes[4].imshow(alpha_img, cmap='magma')
axes[4].set_title("Gabor + CLAHE + Alpha-blend")
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