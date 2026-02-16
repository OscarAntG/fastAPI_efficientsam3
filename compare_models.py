
import sys
import os
import argparse
from tabulate import tabulate

# Ensure sam3 can be imported
sys.path.append(os.path.join(os.getcwd(), 'sam3'))

try:
    import torch
    import torch.nn as nn
    from sam3.model_builder import build_sam3_image_model
    from stage1.model import _build_backbone
except ImportError as e:
    print(f"Import Error: {e}", file=sys.stderr)
    sys.exit(1)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_model_details(model, name="Model"):
    total_params = count_parameters(model)
    details = {
        "name": name,
        "total_params": total_params,
        "components": []
    }
    
    for child_name, module in model.named_children():
        params = count_parameters(module)
        if params > 0:
            component_info = {
                "name": child_name,
                "params": params,
                "percent": params/total_params if total_params > 0 else 0,
                "sub_components": []
            }
            
            # Go one level deeper for large components (>1M params)
            if params > 1e6:
                for sub_name, sub_module in module.named_children():
                    sub_params = count_parameters(sub_module)
                    if sub_params > 0:
                        component_info["sub_components"].append({
                            "name": sub_name,
                            "params": sub_params,
                            "percent": sub_params/params if params > 0 else 0
                        })
            
            details["components"].append(component_info)
    
    return details

def print_details_table(details):
    print(f"\n{'='*20} {details['name']} {'='*20}")
    print(f"Total Parameters: {details['total_params']:,} ({details['total_params']/1e6:.2f}M)")
    
    table_data = []
    for comp in details["components"]:
        table_data.append([
            comp["name"], 
            f"{comp['params']:,}", 
            f"{comp['params']/1e6:.2f}M", 
            f"{comp['percent']:.1%}"
        ])
        for sub in comp["sub_components"]:
            table_data.append([
                f"  └─ {sub['name']}", 
                f"{sub['params']:,}", 
                f"{sub['params']/1e6:.2f}M", 
                f"{sub['percent']:.1%} (of parent)"
            ])
            
    print(tabulate(table_data, headers=["Component", "Params", "Params (M)", "Percent"], tablefmt="simple"))

STUDENT_MODELS = {
    "es_rv_s": "repvit_m0_9",
    "es_rv_m": "repvit_m1_1",
    "es_rv_l": "repvit_m2_3",
    "es_tv_s": "tiny_vit_5m",
    "es_tv_m": "tiny_vit_11m",
    "es_tv_l": "tiny_vit_21m",
    "es_ev_s": "efficientvit_b0",
    "es_ev_m": "efficientvit_b1",
    "es_ev_l": "efficientvit_b2",
}

def main():
    parser = argparse.ArgumentParser(description="Compare SAM3 Teacher vs Student Models Parameter Counts")
    parser.add_argument("--student", type=str, choices=list(STUDENT_MODELS.keys()) + ["all"], 
                        help="Choose a specific student model to compare, or 'all' for summary of all")
    args = parser.parse_args()

    print("Initializing SAM3 Image Model (Teacher)...", file=sys.stderr)
    try:
        sam3_model = build_sam3_image_model(
            load_from_HF=False,
            checkpoint_path=None,
            device='cpu',
            enable_segmentation=True,
            enable_inst_interactivity=True
        )
        teacher_details = get_model_details(sam3_model, "SAM3 Teacher (Image Model)")
        print_details_table(teacher_details)
        
        print("\n\n")
        
        if args.student:
            models_to_run = list(STUDENT_MODELS.keys()) if args.student == "all" else [args.student]
            
            summary_data = []
            
            for model_key in models_to_run:
                backbone_name = STUDENT_MODELS[model_key]
                print(f"Initializing {model_key} ({backbone_name})...", file=sys.stderr)
                
                # Use _build_backbone from stage1/model.py
                # We pass img_size=1008 to match standard config
                student_backbone, _ = _build_backbone(backbone_name, img_size=1008)
                
                # If specific model requested, show full details
                if args.student != "all":
                    student_details = get_model_details(student_backbone, f"Student: {model_key} ({backbone_name})")
                    print_details_table(student_details)
                else:
                    # Just collect stats for summary
                    params = count_parameters(student_backbone)
                    summary_data.append([model_key, backbone_name, f"{params:,}", f"{params/1e6:.2f}M"])

            if args.student == "all":
                print(f"\n{'='*20} Student Models Summary {'='*20}")
                print(tabulate(summary_data, headers=["Model Name", "Backbone", "Params", "Params (M)"], tablefmt="simple"))
                
                teacher_params = teacher_details['total_params']
                print("\nComparison to Teacher:")
                comparison_data = []
                for row in summary_data:
                    student_params_m = float(row[3][:-1])
                    teacher_params_m = teacher_params / 1e6
                    reduction = (1 - student_params_m / teacher_params_m) * 100
                    comparison_data.append([row[0], f"{student_params_m:.2f}M", f"{teacher_params_m:.2f}M", f"{reduction:.1f}%"])
                print(tabulate(comparison_data, headers=["Student", "Student Params", "Teacher Params", "Reduction"], tablefmt="simple"))

        else:
            print("No student model selected. Use --student [model_name|all] to see student details.")

    except Exception as e:
        print(f"Error analyzing models: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
