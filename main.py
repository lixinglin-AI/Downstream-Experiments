import os
import sys
import torch
import pandas as pd
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import argparse

# Add the gradcam_finalversion directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
gradcam_dir = os.path.join(current_dir, '..')
sys.path.append(gradcam_dir)

from GradCAMplus.model_targets import ClassifierOutputTarget
from GradCAMplus.grad_cam_plusplus import GradCAMPlusPlus


class CUBDataset(Dataset):
    def __init__(self, root_dir, metadata_csv_name="metadata.csv", transform=None, split = [0, 1]):
        self.root_dir = root_dir
        self.transform = transform  # Apply image transformations (like resizing and normalization)
        self.split = split  # Specify whether to use training, validation, or test data.

        # Read in metadata
        metadata_path = os.path.join(self.root_dir, metadata_csv_name)
        print(f"Reading '{metadata_path}'")
        self.metadata_df = pd.read_csv(metadata_path)

        # Filter data based on split
        self.metadata_df = self.metadata_df[self.metadata_df['split'].isin(self.split)]

        # Get the y values
        self.y_array = self.metadata_df["y"].values
        self.n_classes = 2

        # Read spurious column directly
        self.spurious_array = self.metadata_df["spurious"].values
        self.n_confounders = 1

        # Map to groups (if still needed)
        # self.n_groups = 4
        # self.group_array = (self.y_array * 2 + self.spurious_array).astype("int")

        # Extract filenames and splits
        self.filename_array = self.metadata_df["img_filename"].values
        self.img_id_array = self.metadata_df["img_id"].values
        self.split_array = self.metadata_df["split"].values
        self.split_dict = {
            0: "train",
            1: "val",
            2: "test",
        }

    def __len__(self):
        return len(self.filename_array)

    def __getitem__(self, idx):
        img_rel_path = self.filename_array[idx]
        img_full_path = os.path.join(self.root_dir, img_rel_path)
        image = Image.open(img_full_path).convert('RGB')

        label = self.y_array[idx]
        spurious = self.spurious_array[idx] # true group info
        split = self.split_array[idx]
        img_id = self.img_id_array[idx]

        if self.transform:
            image = self.transform(image)

        return image, label, spurious, img_rel_path, split, img_id


def get_transform_cub(train, augment_data):
    scale = 256.0 / 224.0
    target_resolution = (224, 224)
    if (not train) or (not augment_data):
        transform = transforms.Compose([
            transforms.Resize((int(target_resolution[0] * scale), int(target_resolution[1] * scale))),
            transforms.CenterCrop(target_resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    else:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(target_resolution, scale=(0.7, 1.0), ratio=(0.75, 1.3333333333333333),
                                         interpolation=2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    return transform


def create_masked_image(image: np.ndarray, cam_result: np.ndarray, threshold: float = 0.4,
                        keep_mask: bool = True) -> np.ndarray:
    mask = cam_result >= threshold
    masked_image = image.copy()
    if keep_mask:
        masked_image[~mask] = 0  # Set regions outside the mask to 0--keep the importance region
    else:
        masked_image[mask] = 0  # Set regions inside the mask to 0
    return masked_image


def process_images_and_calculate_scores(device, model, dataloader, compare_type, transform):
    target_layers = [model.layer4[-1]]  # Specifies the target layer for Grad-CAM++
    cam = GradCAMPlusPlus(model=model, target_layers=target_layers)  # Initializes Grad-CAM++ instance
    records = []
    records_union = []
    
    #If we need union
    existing_metadata = None
    if args.add_union:
        metadata_path = f"results/CUB/CUB_{args.dataset_name}/train_downstream_ERM_upweight_0_epochs_300_lr_1e-05_weight_decay_1.0/final_epoch60/metadata_aug.csv"
        print(f"Checking for file at: {metadata_path}")
        if os.path.exists(metadata_path):
            print(f"{metadata_path} exits")
            existing_metadata = pd.read_csv(metadata_path)
            existing_wrong_times = {int(img_id): wrong_times 
                                 for img_id, wrong_times in 
                                 zip(existing_metadata['img_id'], existing_metadata['wrong_1_times'])}
            print(f"Loaded {len(existing_wrong_times)} existing wrong times")
        else:
            print(f"Warning: {metadata_path} not found")
            

    for images, labels, spurious_features, img_rel_paths, splits, img_ids in tqdm(dataloader, desc="Processing images"):
        # Move the batch of images to the specified device
        images = images.to(device)
        labels = labels.to(device)
        spurious_features = spurious_features.to(device)

        # Step 1: Make predictions on the original images
        with torch.no_grad():
            original_logits = model(images)
            original_probs = F.softmax(original_logits, dim=1)
            y_hats = original_probs.argmax(dim=1)

        # Step 2: Apply Grad-CAM++ to each image in the batch
        grayscale_cams = []
        for image, label in zip(images, labels):
            target = ClassifierOutputTarget(label.item())
            cam_output = cam(input_tensor=image.unsqueeze(0), targets=[target])
            grayscale_cams.append(cam_output[0, :])

        # Step 3: Process each image in the batch individually
        for i in range(images.size(0)):
            image_np = images[i].squeeze().permute(1, 2, 0).cpu().numpy()
            image_np = (image_np - np.min(image_np)) / (np.max(image_np) - np.min(image_np))

            # Create masked images
            grayscale_cam = grayscale_cams[i]
            masked_image_U = create_masked_image(image_np, grayscale_cam, threshold=0.4, keep_mask=True)
            masked_image_V = create_masked_image(image_np, grayscale_cam, threshold=0.4, keep_mask=False)

            # Convert masked images to tensors
            masked_image_U_tensor = transform(Image.fromarray((masked_image_U * 255).astype(np.uint8))).unsqueeze(0).to(device)
            masked_image_V_tensor = transform(Image.fromarray((masked_image_V * 255).astype(np.uint8))).unsqueeze(0).to(device)

            # Make predictions on masked images
            with torch.no_grad():
                logits_U = model(masked_image_U_tensor)
                logits_V = model(masked_image_V_tensor)

                log_probs_U = F.log_softmax(logits_U, dim=1)
                log_probs_V = F.log_softmax(logits_V, dim=1)

                logp1_class = logits_U.argmax(dim=1).item()
                logp2_class = logits_V.argmax(dim=1).item()

                logp1 = log_probs_U[0, logp1_class].item()
                logp2 = log_probs_V[0, logp2_class].item()

            # Approximate f(U, -V) using logp1 - logp2
            approx_log_prob = logp1 - logp2

            # Determine the final classification
            final_class = logp1_class if approx_log_prob > 0 else logp2_class

            # Determine if the instance belongs to the minority or majority group
            label = labels[i].item()
            y_hat = y_hats[i].item()
            spurious = spurious_features[i].item()
            img_rel_path = img_rel_paths[i]
            split = splits[i].item()
            img_id = img_ids[i]

            if compare_type == "true_y":
                is_minority = (label != logp2_class) and (label == final_class)
            elif compare_type == "predicted_y":
                is_minority = (y_hat != logp2_class) and (y_hat == final_class)
            group = "minority" if is_minority else "majority"
            
            original_wrong_1_times = 0 if group == "majority" else 1
            current_img_id = int(img_id.item())

            # Record the results
            record = {
                'image_id': img_rel_path,
                'img_id': current_img_id, # keep
                'y': label, # keep
                'spurious': spurious, # keep
                'y_hat': y_hat,
                'logp1': logp1,
                'logp2': logp2,
                'logp2_class': logp2_class,
                'approx_log_prob': approx_log_prob,
                'final_class': final_class,
                'group': group,
                'split': split,
                'wrong_1_times':original_wrong_1_times
            }
            records.append(record.copy())
            
            if args.add_union:
                union_wrong_1_times = original_wrong_1_times
                if existing_metadata is not None:
                    if current_img_id in existing_wrong_times:
                        existing_wrong_time = existing_wrong_times[current_img_id]
                        union_wrong_1_times = 1 if (existing_wrong_time == 1 or original_wrong_1_times == 1) else 0
                        print(f"ID: {current_img_id}, Existing: {existing_wrong_time}, Original: {original_wrong_1_times}, Union: {union_wrong_1_times}")
                    else:
                        print(f"ID {current_img_id} not found in existing metadata")


                union_record = record.copy()
                union_record['wrong_1_times'] = union_wrong_1_times
                records_union.append(union_record)
                
    if args.add_union and existing_metadata is not None:
        print(f"Total records processed: {len(records_union)}")
        print(f"Records with union wrong_1_times = 1: {sum(1 for r in records_union if r['wrong_1_times'] == 1)}")
        print(f"Records with original wrong_1_times = 1: {sum(1 for r in records if r['wrong_1_times'] == 1)}")

    return records, records_union



def main(args):
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    print(f"Using device: {device}")

    model = torch.load(args.model_path, map_location=device)
    model.to(device)
    model.eval()

    transform = get_transform_cub(train=False, augment_data=False)
    dataset = CUBDataset(root_dir=args.root_dir, transform=transform, split=[0, 1])
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    records, records_union = process_images_and_calculate_scores(device, model, dataloader, args.compare_type, transform)

    output_dir = args.output_dir
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save records to an Excel file
    excel_file_path = os.path.join(args.output_dir, 'metadata_new.csv')
    records_df = pd.DataFrame(records)
    records_df.to_csv(excel_file_path, index=False)

# If add_union is set and records_union is not empty, save union data to CSV
    if args.add_union and records_union:
        union_file_path = os.path.join(args.output_dir, 'metadata_new_union.csv')
        records_union_df = pd.DataFrame(records_union)
        records_union_df.to_csv(union_file_path, index=False)
        print(f"Union results saved to {union_file_path}")
    
    print(f"Analysis completed. Results saved to {excel_file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True, help='Path to the pretrained model')
    parser.add_argument('--root_dir', type=str, required=True, help='Path to the CUB dataset directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save output files')
    parser.add_argument('--batch_size', type=int, required=True, help='Directory to save output files')
    parser.add_argument('--compare_type', type=str, choices=['true_y', 'predicted_y'], help='Comparison type: true_y or predicted_y')
    parser.add_argument("--add_union", action="store_true", help="If set, adds additional columns to the output.")
    args = parser.parse_args()

    main(args)

'''
python /Users/yrw/Desktop/Research/Research1/jtt-master/JTT/gradcam_finalversion/Cub_gradcam_final/predict_twice_comparewithtruelabel.py --model_path /Users/yrw/Desktop/Research/Research1/jtt-master/JTT/results/CUB/CUB_sample_exp_5/ERM_upweight_0_epochs_120_lr_0.0001_weight_decay_0.0001/model_outputs/best_model.pth --root_dir /Users/yrw/Desktop/Research/Research1/jtt-master/JTT/cub/data/waterbird_complete95_forest2water2 --output_dir /Users/yrw/Desktop/Research/Research1/jtt-master/JTT/gradcam_finalversion/Cub_gradcam_final/results_predict_twicecomparewithlabel 
'''
