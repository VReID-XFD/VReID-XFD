"""
Comprehensive evaluation script for AG-VPReID dataset
Evaluates the model on all three cases:
- Case 1: Aerial to Ground
- Case 2: Ground to Aerial
- Case 3: Aerial to Aerial
"""

import argparse
import torch
import numpy as np
import os
import sys
from torch.backends import cudnn
import time
from datetime import timedelta
import csv

# Add current directory to path
sys.path.append('.')

from config import cfg
from datasets.make_video_dataloader import make_dataloader
from model.make_model_clipvideoreid_reidadapter_pbp import make_model
from utils.test_video_reid import test


def print_header(text):
    """Print formatted header"""
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80)


def print_section(text):
    """Print formatted section"""
    print("\n" + "-" * 80)
    print(f"  {text}")
    print("-" * 80)


def extract_tracklet_name(img_paths):
    """Extract tracklet name from image paths"""
    if isinstance(img_paths, (list, tuple)) and len(img_paths) > 0:
        # Get the tracklet directory name from the first image path
        tracklet_dir = os.path.dirname(img_paths[0])
        tracklet_name = os.path.basename(tracklet_dir)
        return tracklet_name
    return "unknown"


def evaluate_case(cfg, model, case_name, case_subset, use_gpu=True):
    """
    Evaluate model on a specific case
    
    Args:
        cfg: Configuration object
        model: The trained model
        case_name: Name of the case (e.g., "Case 1: Aerial to Ground")
        case_subset: Subset identifier (e.g., "case1_aerial_to_ground")
        use_gpu: Whether to use GPU
        
    Returns:
        Dictionary containing evaluation results
    """
    print_section(f"Evaluating {case_name}")
    print(f"Subset: {case_subset}")
    
    # Create dataset directly without modifying config
    print(f"Loading data from: {cfg.DATASETS.ROOT_DIR}/{case_subset}")
    from datasets import data_manager
    dataset = data_manager.init_dataset(name=cfg.DATASETS.NAMES, root=cfg.DATASETS.ROOT_DIR, subset=case_subset)
    
    from datasets.video_loader import VideoDataset
    from torch.utils.data import DataLoader
    import dataset_transformer.spatial_transforms as ST
    import dataset_transformer.temporal_transforms as TT
    
    # Create transformations
    spatial_transform_test = ST.Compose([
        ST.Scale((cfg.INPUT.SIZE_TEST[0], cfg.INPUT.SIZE_TEST[1]), interpolation=3),
        ST.ToTensor(),
        ST.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    temporal_transform_test = TT.TemporalRestrictedBeginCrop(size=cfg.TEST.SEQ_LEN)
    
    # Create query dataset and loader
    query_dataset = VideoDataset(
        dataset.query,
        spatial_transform=spatial_transform_test,
        temporal_transform=temporal_transform_test
    )
    # Use smaller batch size for evaluation to avoid OOM
    eval_batch_size = 4  # Reduced from 16 to 4
    
    query_loader = DataLoader(
        query_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=4,  # Reduced workers
        pin_memory=True,
        drop_last=False
    )
    
    # Create gallery dataset and loader
    gallery_dataset = VideoDataset(
        dataset.gallery,
        spatial_transform=spatial_transform_test,
        temporal_transform=temporal_transform_test
    )
    gallery_loader = DataLoader(
        gallery_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=4,  # Reduced workers
        pin_memory=True,
        drop_last=False
    )
    
    print(f"Query set size: {len(query_dataset)}")
    print(f"Gallery set size: {len(gallery_dataset)}")
    print(f"Number of query identities: {len(set([x[1] for x in dataset.query]))}")
    print(f"Number of gallery identities: {len(set([x[1] for x in dataset.gallery]))}")
    
    # CRITICAL: Sort tracklets by PID first, then by tracklet name to maintain strict order
    # This ensures: Case X - 1st ID - 1st tracklet, Case X - 1st ID - 2nd tracklet, etc.
    print("Sorting tracklets to maintain strict PID->tracklet order...")
    
    # Create list of (index, pid, tracklet_name) for sorting
    query_info = []
    for i, tracklet in enumerate(dataset.query):
        pid = tracklet[1]
        tracklet_name = extract_tracklet_name(tracklet[0])
        query_info.append((i, pid, tracklet_name))
    
    # Sort by PID first, then by tracklet name
    query_info_sorted = sorted(query_info, key=lambda x: (x[1], x[2]))
    
    # Extract the sorted indices and names
    query_sorted_indices = [x[0] for x in query_info_sorted]
    query_tracklet_names = [x[2] for x in query_info_sorted]
    query_pids_sorted = [x[1] for x in query_info_sorted]
    
    # Same for gallery
    gallery_info = []
    for i, tracklet in enumerate(dataset.gallery):
        pid = tracklet[1]
        tracklet_name = extract_tracklet_name(tracklet[0])
        gallery_info.append((i, pid, tracklet_name))
    
    gallery_info_sorted = sorted(gallery_info, key=lambda x: (x[1], x[2]))
    gallery_sorted_indices = [x[0] for x in gallery_info_sorted]
    gallery_tracklet_names = [x[2] for x in gallery_info_sorted]
    
    print(f"Sorted query tracklets: First 5 PIDs = {query_pids_sorted[:5]}")
    print(f"Sorted query tracklets: First 5 names = {query_tracklet_names[:5]}")
    
    # Verify strict ordering
    prev_pid = -1
    for i, pid in enumerate(query_pids_sorted):
        if pid < prev_pid:
            print(f"WARNING: Ordering issue at index {i}: PID {pid} < previous PID {prev_pid}")
        prev_pid = pid
    print("✓ Tracklet ordering verified: PIDs are in ascending order")
    
    # Compute rankings for CSV generation FIRST (before test() which may fail)
    from utils.test_video_reid import _feats_of_loader, _cal_dist, extract_feat_sampled_frames, extract_feat_all_frames
    import torch.nn.functional as F
    
    model.eval()
    if cfg.TEST.ALL_FRAMES:
        feat_func = extract_feat_all_frames
    else:
        feat_func = extract_feat_sampled_frames
    
    # Extract features
    print("Extracting features for ranking...")
    qf, q_pids, q_camids = _feats_of_loader(model, query_loader, feat_func, use_gpu=use_gpu)
    gf, g_pids, g_camids = _feats_of_loader(model, gallery_loader, feat_func, use_gpu=use_gpu)
    
    # Reorder features according to sorted indices (PID->tracklet order)
    print("Reordering features to match sorted tracklet order...")
    qf_sorted = qf[query_sorted_indices]
    gf_sorted = gf[gallery_sorted_indices]
    
    # Compute distance matrix with sorted features
    distmat = _cal_dist(qf=qf_sorted, gf=gf_sorted, distance=cfg.TEST.DISTANCE)
    
    # Get top-10 gallery indices for each query
    # argsort gives indices from smallest to largest distance
    ranking_indices = torch.argsort(distmat, dim=1).numpy()  # shape: (num_queries, num_galleries)
    
    # Get top-10 for each query (or all galleries if less than 10)
    top10_rankings = []
    num_galleries = len(gallery_tracklet_names)
    top_k = min(10, num_galleries)  # Get top-10 or all galleries if less than 10
    
    for i in range(len(query_tracklet_names)):
        topk_indices = ranking_indices[i][:top_k]  # Get top k (these are indices in sorted gallery)
        topk_gallery_names = [gallery_tracklet_names[idx] for idx in topk_indices]
        top10_rankings.append(topk_gallery_names)
    
    # Run evaluation metrics (may fail for anonymized data, but rankings are already computed)
    start_time = time.time()
    try:
        cmc, mAP, ranks = test(model, query_loader, gallery_loader, use_gpu, cfg)
    except (ZeroDivisionError, IndexError) as e:
        # Anonymized data - no ground truth available
        print(f"Warning: Ground truth evaluation failed ({e}). Skipping mAP/CMC calculation.")
        cmc = np.zeros(len(gallery_dataset))
        mAP = 0.0
        ranks = [1, 5, 10, 20]
    elapsed_time = time.time() - start_time
    
    # Print results
    print(f"\nEvaluation completed in {elapsed_time:.2f} seconds")
    print(f"\n{'Metric':<20} {'Value':<15}")
    print("-" * 35)
    print(f"{'mAP':<20} {mAP:.2%}")
    for r in ranks:
        if r <= len(cmc):  # Only print ranks that exist
            print(f"{'Rank-' + str(r):<20} {cmc[r-1]:.2%}")
    
    return {
        'case_name': case_name,
        'case_subset': case_subset,
        'mAP': mAP,
        'cmc': cmc,
        'ranks': ranks,
        'num_query': len(query_dataset),
        'num_gallery': len(gallery_dataset),
        'num_query_ids': len(set([x[1] for x in dataset.query])),
        'num_gallery_ids': len(set([x[1] for x in dataset.gallery])),
        'elapsed_time': elapsed_time,
        'query_tracklet_names': query_tracklet_names,
        'top10_rankings': top10_rankings
    }


def print_summary(results):
    """Print summary of all results"""
    print_header("EVALUATION SUMMARY")
    
    # Create summary table
    print(f"\n{'Case':<30} {'mAP':<12} {'R-1':<12} {'R-5':<12} {'R-10':<12}")
    print("=" * 78)
    
    total_map = 0
    total_r1 = 0
    total_r5 = 0
    total_r10 = 0
    
    for result in results:
        case_name = result['case_name']
        mAP = result['mAP']
        cmc = result['cmc']
        
        # Get rank values
        r1 = cmc[0] if len(cmc) > 0 else 0
        r5 = cmc[4] if len(cmc) > 4 else cmc[-1] if len(cmc) > 0 else 0
        r10 = cmc[9] if len(cmc) > 9 else cmc[-1] if len(cmc) > 0 else 0
        
        print(f"{case_name:<30} {mAP:>10.2%} {r1:>10.2%} {r5:>10.2%} {r10:>10.2%}")
        
        total_map += mAP
        total_r1 += r1
        total_r5 += r5
        total_r10 += r10
    
    # Print averages
    num_cases = len(results)
    print("-" * 78)
    print(f"{'Average':<30} {total_map/num_cases:>10.2%} {total_r1/num_cases:>10.2%} {total_r5/num_cases:>10.2%} {total_r10/num_cases:>10.2%}")
    
    # Print detailed statistics
    print_section("Detailed Statistics")
    for result in results:
        print(f"\n{result['case_name']}:")
        print(f"  Query size: {result['num_query']} tracklets ({result['num_query_ids']} identities)")
        print(f"  Gallery size: {result['num_gallery']} tracklets ({result['num_gallery_ids']} identities)")
        print(f"  Evaluation time: {result['elapsed_time']:.2f} seconds")
        print(f"  mAP: {result['mAP']:.4f}")
        print(f"  CMC @ Rank-1: {result['cmc'][0]:.4f}")
        if len(result['cmc']) > 4:
            print(f"  CMC @ Rank-5: {result['cmc'][4]:.4f}")
        if len(result['cmc']) > 9:
            print(f"  CMC @ Rank-10: {result['cmc'][9]:.4f}")
    
    # Final score (average of all cases)
    avg_rank1 = sum([r['cmc'][0] for r in results]) / num_cases
    print_header(f"FINAL SCORE (Average Rank-1): {avg_rank1:.2%}")


def generate_ranking_csv(results, output_path='evaluation_rankings.csv'):
    """
    Generate CSV file with query tracklets and their top-10 gallery matches.
    
    Args:
        results: List of evaluation results from all cases
        output_path: Path to save the CSV file
    """
    print_section("Generating Ranking CSV")
    
    csv_rows = []
    row_id = 1
    
    # Process each case in order (Case 1, Case 2, Case 3)
    for result in results:
        case_name = result['case_name']
        query_names = result['query_tracklet_names']
        top10_rankings = result['top10_rankings']
        
        print(f"Processing {case_name}: {len(query_names)} queries")
        
        # For each query tracklet in this case
        for query_name, top10_galleries in zip(query_names, top10_rankings):
            # Join top-10 gallery tracklets with spaces
            gallery_str = ' '.join(top10_galleries)
            
            csv_rows.append({
                'row_id': row_id,
                'query_tracklet': query_name,
                'gallery_tracklets': gallery_str
            })
            row_id += 1
    
    # Write to CSV file
    with open(output_path, 'w', newline='') as csvfile:
        fieldnames = ['row_id', 'query_tracklet', 'gallery_tracklets']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)
    
    print(f"CSV file saved to: {output_path}")
    print(f"Total rows: {len(csv_rows)}")


def main():
    parser = argparse.ArgumentParser(description="AG-VPReID Evaluation on All Cases")
    parser.add_argument(
        "--config_file",
        default="configs/adapter/vit_adapter.yml",
        help="path to config file",
        type=str
    )
    parser.add_argument(
        "--model_path",
        required=True,
        help="path to trained model checkpoint",
        type=str
    )
    parser.add_argument(
        "--cases",
        default="1,2,3",
        help="which cases to evaluate (e.g., '1,2,3' or '1,3')",
        type=str
    )
    parser.add_argument(
        "--output_file",
        default="",
        help="path to save evaluation results (optional)",
        type=str
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER
    )
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    
    # Check CUDA availability
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        cudnn.benchmark = True
    else:
        print("CUDA not available, using CPU")
    
    print_header("AG-VPReID Multi-Case Evaluation")
    print(f"Config file: {args.config_file}")
    print(f"Model path: {args.model_path}")
    print(f"Dataset root: {cfg.DATASETS.ROOT_DIR}")
    
    # Load model
    print_section("Loading Model")
    print(f"Model: {cfg.MODEL.NAME}")
    
    # Get number of classes from dataset
    from datasets import data_manager
    dataset = data_manager.init_dataset(
        name=cfg.DATASETS.NAMES,
        root=cfg.DATASETS.ROOT_DIR,
        subset='case1_aerial_to_ground'
    )
    num_classes = dataset.num_train_pids
    camera_num = dataset.num_camera
    view_num = 0  # Not used in DetReIDx
    
    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
    
    # Load checkpoint
    if not os.path.exists(args.model_path):
        print(f"Error: Model checkpoint not found at {args.model_path}")
        return
    
    print(f"Loading checkpoint from: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(checkpoint)
    model.eval()
    
    if use_gpu:
        model = model.cuda()
    
    print("Model loaded successfully!")
    
    # Parse which cases to evaluate
    cases_to_eval = [int(c.strip()) for c in args.cases.split(',')]
    
    # Define cases
    all_cases = {
        1: {
            'name': 'Case 1: Aerial to Ground',
            'subset': 'case1_aerial_to_ground',
            'description': 'Query: Aerial images, Gallery: Ground images'
        },
        2: {
            'name': 'Case 2: Ground to Aerial',
            'subset': 'case2_ground_to_aerial',
            'description': 'Query: Ground images, Gallery: Aerial images'
        },
        3: {
            'name': 'Case 3: Aerial to Aerial',
            'subset': 'case3_aerial_to_aerial',
            'description': 'Query: Aerial images, Gallery: Aerial images'
        }
    }
    
    # Evaluate each case
    results = []
    total_start_time = time.time()
    
    for case_num in cases_to_eval:
        if case_num not in all_cases:
            print(f"Warning: Invalid case number {case_num}, skipping...")
            continue
        
        case_info = all_cases[case_num]
        print(f"\n{case_info['description']}")
        
        try:
            result = evaluate_case(
                cfg,
                model,
                case_info['name'],
                case_info['subset'],
                use_gpu=use_gpu
            )
            results.append(result)
        except Exception as e:
            print(f"\n{'='*80}")
            print(f"ERROR in {case_info['name']}: {e}")
            print(f"{'='*80}")
            import traceback
            traceback.print_exc()
            print(f"\nSkipping {case_info['name']} and continuing with next case...")
            print(f"{'='*80}\n")
            continue
    
    total_elapsed = time.time() - total_start_time
    
    # Print summary
    if results:
        print_summary(results)
        print(f"\nTotal evaluation time: {timedelta(seconds=int(total_elapsed))}")
        
        # Generate ranking CSV file
        csv_output_path = 'output/cross_attn/detreidx/alpha/256/evaluation_rankings.csv'
        os.makedirs(os.path.dirname(csv_output_path), exist_ok=True)
        generate_ranking_csv(results, output_path=csv_output_path)
        
        # Save results to file if specified
        if args.output_file:
            import json
            output_data = {
                'model_path': args.model_path,
                'config_file': args.config_file,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'results': []
            }
            
            for result in results:
                output_data['results'].append({
                    'case_name': result['case_name'],
                    'case_subset': result['case_subset'],
                    'mAP': float(result['mAP']),
                    'rank1': float(result['cmc'][0]),
                    'rank5': float(result['cmc'][4]) if len(result['cmc']) > 4 else None,
                    'rank10': float(result['cmc'][9]) if len(result['cmc']) > 9 else None,
                    'num_query': result['num_query'],
                    'num_gallery': result['num_gallery'],
                    'num_query_ids': result['num_query_ids'],
                    'num_gallery_ids': result['num_gallery_ids']
                })
            
            with open(args.output_file, 'w') as f:
                json.dump(output_data, f, indent=2)
            print(f"\nResults saved to: {args.output_file}")
    else:
        print("\nNo results to display.")


if __name__ == '__main__':
    main()

