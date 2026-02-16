import os
os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda/bin/ptxas" 
os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sam3.sam3.model_builder import build_efficientsam3_image_model
from sam3.sam3.model.sam3_image_processor import Sam3Processor

# 1. Setup Model
checkpoint = "efficient_sam3_efficientvit_s.pt"
model = build_efficientsam3_image_model(
    checkpoint_path=checkpoint,
    backbone_type="efficientvit",
    model_name="b0", 
    enable_inst_interactivity=True,
).cuda()
model.eval()
processor = Sam3Processor(model)

# 2. Load Real Image
res = 1024
# FIX: You must assign the result of resize to the variable
synth_img = Image.open("clay.jpg").convert("RGB")
synth_img = synth_img.resize((res, res)) 

# 3. Prepare Inference
inference_state = processor.set_image(synth_img)
w, h = synth_img.size
# Prompt point - still center, but you can adjust these to the whale's eye/body
points = np.array([[w / 2 - 50, h / 2 - 200]], dtype=np.float32)
labels = np.array([1], dtype=np.int32) 

print("Running Real Image Diagnostic (Whale)...")

# 4. Inference Logic
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

# 5. Visualization (Updated with fixed colormap and Best Mask logic)
num_masks = len(masks)
if num_masks > 0:
    fig, axes = plt.subplots(1, num_masks + 1, figsize=(20, 5))
    
    # Plot 1: Original Image
    axes[0].imshow(synth_img)
    axes[0].scatter([w/2-50], [h/2-200], color='yellow', marker='*', s=100)
    axes[0].set_title("Original + Prompt")
    axes[0].axis('off')

    # Updated Matplotlib 3.8+ colormap syntax
    cmap = plt.get_cmap('tab10')

    for i in range(num_masks):
        mask = masks[i]
        score = scores[i]
        
        # Create visualization for this mask
        mask_viz = np.zeros((res, res, 3))
        mask_viz[mask > 0] = cmap(i)[:3]
        
        axes[i+1].imshow(synth_img) # Show image behind mask
        axes[i+1].imshow(mask_viz, alpha=0.5) # Overlay mask
        axes[i+1].set_title(f"Mask {i}\nScore: {score:.2f}")
        axes[i+1].axis('off')

    plt.tight_layout()
    plt.savefig("result.jpg")
    print("Result saved to result.jpg")
    
    # Best Mask Logic for Hanford Workflow
    best_idx = np.argmax(scores)
    print(f"Workflow Selection: Mask {best_idx} is the highest confidence.")
else:
    print("Failure: No masks detected.")


# import os
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from PIL import Image
# from scipy import ndimage as nd
# from sam3.sam3.model_builder import build_efficientsam3_image_model
# from sam3.sam3.model.sam3_image_processor import Sam3Processor

# # --- 1. Model Setup (Half Precision for 4GB) ---
# checkpoint = "efficient_sam3_efficientvit_s.pt"
# model = build_efficientsam3_image_model(
#     checkpoint_path=checkpoint,
#     backbone_type="efficientvit",
#     model_name="b0", 
#     enable_inst_interactivity=True,
#     # refine_output=False,
#     # use_triton=False
# ).cuda().half() # Crucial for 4GB
# if hasattr(model, 'inst_interactive_predictor'):
#     print("Disabling Triton refinement in predictor...")
#     model.inst_interactive_predictor.with_refine = False
# for module in model.modules():
#     if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d, torch.nn.LayerNorm)):
#         module.float()
# model.eval()
# processor = Sam3Processor(model)

# # --- 2. Load and Encode Image ---
# res = 1024
# img = Image.open("whale.jpg").convert("RGB").resize((res, res))
# inference_state = processor.set_image(img)

# # --- 3. Generate Autonomous Grid ---
# grid_size = 32 # n^2 points. Start small for memory safety.
# x = np.linspace(20, res-20, grid_size)
# y = np.linspace(20, res-20, grid_size)
# xv, yv = np.meshgrid(x, y)
# grid_points = np.stack([xv.flatten(), yv.flatten()], axis=1).astype(np.float32)
# grid_labels = np.ones(len(grid_points), dtype=np.int32)

# # --- 4. Batch Inference Loop ---
# print(f"Scanning wall with {len(grid_points)} points...")
# final_mask_canvas = np.zeros((res, res), dtype=np.float32)
# score_threshold = 0.70 # Only trust high-confidence shapes

# for i in range(len(grid_points)):
#     # Single point inference in a loop is slower but safest for 4GB VRAM
#     pt = grid_points[i:i+1]
#     lab = grid_labels[i:i+1]
    
#     with torch.no_grad():
#         try:
#             masks, scores, _ = model.predict_inst(
#                 inference_state, 
#                 point_coords=pt, 
#                 point_labels=lab
#             )
            
#             # Select the most confident mask for this point
#             best_idx = np.argmax(scores)
#             if scores[1] > score_threshold:
#                 # Add this mask to our master canvas
#                 final_mask_canvas = np.maximum(final_mask_canvas, masks[1].astype(np.float32))
                
#         except Exception as e:
#             continue

# # --- 5. Post-Process (The "Waste" Filter using SciPy) ---
# binary_mask = (final_mask_canvas > 0).astype(bool)

# # 'Opening' removes small noise/specks
# cleaned_mask = nd.binary_opening(binary_mask, structure=np.ones((5,5)))
# # 'Closing' fills small holes in the sludge
# cleaned_mask = nd.binary_closing(cleaned_mask, structure=np.ones((5,5)))

# # Convert back to uint8 for visualization
# cleaned_mask_img = cleaned_mask.astype(np.uint8) * 255

# # --- 6. Visualization ---
# plt.figure(figsize=(12, 6))
# plt.subplot(1, 2, 1)
# plt.imshow(img)
# plt.scatter(grid_points[:,0], grid_points[:,1], c='red', s=5, alpha=0.5)
# plt.title("Autonomous Grid Points")

# plt.subplot(1, 2, 2)
# plt.imshow(img)
# plt.imshow(cleaned_mask, alpha=0.5, cmap='jet')
# plt.title("Detected Waste Map")

# plt.savefig("autonomous_scan_result.png")
# print("Scan complete. Result saved to autonomous_scan_result.png")
# plt.show()