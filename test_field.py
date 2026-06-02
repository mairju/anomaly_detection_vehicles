"""
python test_field.py \                       
    --test_root_dir "/home/maria/Documents/dataset_splited/agco_dataset/test/" \
    --out_dir "/media/maria/mairj/outputs/scratch/cutpaste_300/test_01/" \
    --reconstructive_checkpoint "/home/maria/Documents/projects/anomaly_detection_vehicles/checkpoints_scratch/cutpaste/scratch_cutpaste_epoch_1295.pckl" \
    --discriminative_checkpoint "/home/maria/Documents/projects/anomaly_detection_vehicles/checkpoints_scratch/cutpaste/scratch_cutpaste_epoch_1295_seg.pckl" \
    --batch_size 1 \
    --img_dim 256
"""

import argparse
import torch
import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from models.DRAEM import ReconstructiveSubNetwork, DiscriminativeSubNetwork
from data.field_dataloader import FieldDataset
from tqdm import tqdm  

def test(test_root_dir, out_dir, reconstructive_checkpoint, discriminative_checkpoint, batch_size, img_dim):
    resize_image = (img_dim, img_dim * 2)

    test_dataset = FieldDataset(test_root_dir, resize_shape=resize_image)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=6)

    model = ReconstructiveSubNetwork(in_channels=3, out_channels=3)
    model.load_state_dict(torch.load(reconstructive_checkpoint, map_location='cpu'))
    model.cpu()
    model.eval()

    model_seg = DiscriminativeSubNetwork(in_channels=6, out_channels=2)
    model_seg.load_state_dict(torch.load(discriminative_checkpoint, map_location='cpu'))
    model_seg.cpu()
    model_seg.eval()

    total_pixel_scores = np.zeros((img_dim * img_dim * 2 * len(test_dataset)))
    total_gt_pixel_scores = np.zeros((img_dim * img_dim * 2 * len(test_dataset)))

    anomaly_score_gt = []
    anomaly_score_prediction = []

    mask_cnt = 0

    for i_batch, sample_batched in enumerate(tqdm(test_loader, desc="Processing Batches")):
        gray_batch = sample_batched["image"].cpu()

        is_normal = sample_batched["has_anomaly"].detach().numpy()[0, 0]
        anomaly_score_gt.append(is_normal)
        true_mask = sample_batched["mask"]
        true_mask_cv = true_mask.detach().numpy()[0, :, :, :].transpose((1, 2, 0))

        gray_rec = model(gray_batch)
        joined_in = torch.cat((gray_rec.detach(), gray_batch), dim=1)

        out_mask = model_seg(joined_in)
        out_mask_sm = torch.softmax(out_mask, dim=1)

        out_mask_cv = out_mask_sm[0, 1, :, :].detach().cpu().numpy()
        plt.imsave(f"{out_dir}/{i_batch}_anomaly_map.png", out_mask_cv)

        orig_image = gray_batch[0].cpu().numpy().transpose(1, 2, 0)
        plt.imsave(f"{out_dir}/{i_batch}_original.png", orig_image)

        rec_image = gray_rec[0].detach().cpu().numpy().transpose(1, 2, 0)
        rec_image_clipped = np.clip(rec_image, 0, 1)
        plt.imsave(f"{out_dir}/{i_batch}_reconstructed.png", rec_image_clipped)

        true_mask_np = true_mask[0].squeeze().cpu().numpy()
        cv2.imwrite(f"{out_dir}/{i_batch}_true_mask.png", true_mask_np * 255.)

        binary_mask = (out_mask_cv > 0.5).astype(np.uint8)
        cv2.imwrite(f"{out_dir}/{i_batch}_predicted_mask.png", binary_mask * 255.)

        out_mask_averaged = torch.nn.functional.avg_pool2d(out_mask_sm[:, 1:, :, :], 21, stride=1, padding=21 // 2).cpu().detach().numpy()
        image_score = np.max(out_mask_averaged)

        anomaly_score_prediction.append(image_score)

        flat_true_mask = true_mask.flatten()
        flat_out_mask = out_mask_cv.flatten()

        total_pixel_scores[mask_cnt * img_dim * img_dim * 2:(mask_cnt + 1) * img_dim * img_dim * 2] = flat_out_mask
        total_gt_pixel_scores[mask_cnt * img_dim * img_dim * 2:(mask_cnt + 1) * img_dim * img_dim * 2] = flat_true_mask

        mask_cnt += 1

    with open(f"{out_dir}/anomaly_score_gt.txt", "w") as file:
        for num in anomaly_score_gt:
            file.write(f"{num}\n")

    with open(f"{out_dir}/anomaly_score_prediction.txt", "w") as file:
        for num in anomaly_score_prediction:
            file.write(f"{num}\n")

    np.save(f"{out_dir}/total_pixel_scores.npy", total_pixel_scores)
    np.save(f"{out_dir}/total_gt_pixel_scores.npy", total_gt_pixel_scores)

    auroc_pixel = roc_auc_score(total_gt_pixel_scores.astype(np.uint8), total_pixel_scores)
    ap_pixel = average_precision_score(total_gt_pixel_scores.astype(np.uint8), total_pixel_scores)
    auroc_image = roc_auc_score(anomaly_score_gt, anomaly_score_prediction)
    ap_image = average_precision_score(anomaly_score_gt, anomaly_score_prediction)

    # TODO: + F1Max, AUPR, PRO - referece: CLIP (from Neelus)
    metrics = {
        "auroc_pixel": auroc_pixel,
        "ap_pixel": ap_pixel,
        "auroc_image": auroc_image,
        "ap_image": ap_image
        #"f1max": f1max,
        #"aupr": aupr,
        #"pro": pro
    }

    with open(f"{out_dir}/metrics.json", "w") as file:
        json.dump(metrics, file, indent=4)
    
    print(f"{'Metric':<20} {'Score':<10}")
    print("-" * 30)

    for key, value in metrics.items():
        print(f"{key:<20} {value:<10.4f}")

    print("Testing completed.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Field Anomaly Detection Testing")

    parser.add_argument("--test_root_dir", type=str, default="/home/maria/Documents/dataset_splited/agco_dataset/test/", required=False, help="Path to the test data directory")
    parser.add_argument("--out_dir", type=str, default="/media/maria/mairj/outputs/scratch/ablation/direct/" , required=False, help="Directory to save the output files")
    parser.add_argument("--reconstructive_checkpoint", type=str, default="/home/maria/Documents/projects/anomaly_detection_vehicles/checkpoints_ablation/scratch_realsyn_direct_epoch_200_seg.pckl", required=False, help="Path to the reconstructive model checkpoint")
    parser.add_argument("--discriminative_checkpoint", type=str, default="/home/maria/Documents/projects/anomaly_detection_vehicles/checkpoints_ablation/scratch_realsyn_direct_epoch_200_seg.pckl", required=False, help="Path to the discriminative model checkpoint")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for testing")
    parser.add_argument("--img_dim", type=int, default=256, help="Image dimension (height) for resizing")

    args = parser.parse_args()

    test(
        test_root_dir=args.test_root_dir,
        out_dir=args.out_dir,
        reconstructive_checkpoint=args.reconstructive_checkpoint,
        discriminative_checkpoint=args.discriminative_checkpoint,
        batch_size=args.batch_size,
        img_dim=args.img_dim
    )