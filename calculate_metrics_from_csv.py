"""
Calculate evaluation metrics (mAP and CMC) from the ranking CSV file.
This script reproduces the same metrics as the evaluation script.
"""

import csv
import numpy as np
import argparse
from collections import defaultdict


def compute_ap(index, good_index, junk_index):
    """
    Compute Average Precision (AP) and CMC for a single query.
    This matches the original implementation in utils/test_video_reid.py
    
    Args:
        index: Array of gallery indices in ranked order (best match first)
        good_index: Array of gallery indices that are true matches (same PID, different camid)
        junk_index: Array of gallery indices to ignore (same PID, same camid)
    
    Returns:
        ap: Average Precision
        cmc: Binary array indicating if a correct match was found at each rank
    """
    ap = 0
    cmc = np.zeros(len(index))
    
    if len(good_index) == 0:
        return ap, cmc
    
    # Remove junk_index
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]
    
    # Find good_index in the ranking
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask == True)
    rows_good = rows_good.flatten()
    
    if len(rows_good) == 0:
        return ap, cmc
    
    cmc[rows_good[0]:] = 1.0
    for i in range(ngood):
        d_recall = 1.0 / ngood
        precision = (i + 1) * 1.0 / (rows_good[i] + 1)
        if rows_good[i] != 0:
            old_precision = i * 1.0 / rows_good[i]
        else:
            old_precision = 1.0
        ap = ap + d_recall * (old_precision + precision) / 2
    
    return ap, cmc


def parse_tracklet_info(tracklet_name):
    """
    Extract PID and camid (tracklet ID) from tracklet name.
    Format: {pid}_{tracklet_id}, e.g., '0024_5453' -> pid=24, camid=5453
    """
    parts = tracklet_name.split('_')
    pid = int(parts[0])
    camid = int(parts[1])
    return pid, camid


def calculate_metrics_from_csv(csv_path):
    """
    Calculate mAP and CMC metrics from the ranking CSV file.
    
    Args:
        csv_path: Path to the ranking CSV file
    
    Returns:
        results_per_case: Dictionary with metrics for each case
    """
    print(f"Reading CSV file: {csv_path}")
    
    # Read CSV data
    csv_data = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            csv_data.append(row)
    
    print(f"Total rows: {len(csv_data)}")
    
    # Group by case
    cases = defaultdict(list)
    for row in csv_data:
        cases[row['case']].append(row)
    
    results_per_case = {}
    
    for case_name, case_rows in cases.items():
        print(f"\n{'='*80}")
        print(f"Processing {case_name}")
        print(f"{'='*80}")
        
        # Get number of galleries from first query
        num_galleries = len(case_rows[0]['ranked_gallery_tracklets'].split())
        
        print(f"Number of queries: {len(case_rows)}")
        print(f"Number of galleries: {num_galleries}")
        
        # Initialize metrics
        all_aps = []
        all_cmcs = []
        num_valid_q = 0  # Count queries with at least one good match
        
        # Process each query
        for row in case_rows:
            # Parse query PID and camid from tracklet name
            query_pid, query_camid = parse_tracklet_info(row['query_tracklet'])
            
            # Get THIS query's specific ranking of galleries
            gallery_tracklets = row['ranked_gallery_tracklets'].split()
            
            # Parse gallery PIDs and camids from THIS query's ranked list
            gallery_info = [parse_tracklet_info(t) for t in gallery_tracklets]
            ranked_gallery_pids = np.array([info[0] for info in gallery_info])
            ranked_gallery_camids = np.array([info[1] for info in gallery_info])
            
            # Find good and junk matches using the same logic as original code
            # good_index: same PID, different camid (different tracklet)
            # junk_index: same PID, same camid (same tracklet - should not happen in proper evaluation)
            query_index = np.argwhere(ranked_gallery_pids == query_pid).flatten()
            camera_index = np.argwhere(ranked_gallery_camids == query_camid).flatten()
            good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
            junk_index = np.intersect1d(query_index, camera_index)
            
            if len(good_index) == 0:
                # No ground truth for this query
                continue
            
            # Ranking is just the ordered indices
            index = np.arange(num_galleries)
            
            # Compute AP and CMC for this query
            ap, cmc = compute_ap(index, good_index, junk_index)
            
            all_aps.append(ap)
            all_cmcs.append(cmc)
            num_valid_q += 1
        
        # Calculate mean metrics
        mAP = np.mean(all_aps)
        mean_cmc = np.mean(all_cmcs, axis=0)
        
        # Store results
        results_per_case[case_name] = {
            'mAP': mAP,
            'CMC': mean_cmc,
            'num_queries': len(case_rows),
            'num_galleries': num_galleries
        }
        
        # Print results
        print(f"\nResults:")
        print(f"  mAP: {mAP:.4f} ({mAP*100:.2f}%)")
        print(f"  Rank-1:  {mean_cmc[0]:.4f} ({mean_cmc[0]*100:.2f}%)")
        if len(mean_cmc) > 4:
            print(f"  Rank-5:  {mean_cmc[4]:.4f} ({mean_cmc[4]*100:.2f}%)")
        if len(mean_cmc) > 9:
            print(f"  Rank-10: {mean_cmc[9]:.4f} ({mean_cmc[9]*100:.2f}%)")
        if len(mean_cmc) > 19:
            print(f"  Rank-20: {mean_cmc[19]:.4f} ({mean_cmc[19]*100:.2f}%)")
    
    return results_per_case


def print_summary(results_per_case):
    """Print summary table of all cases."""
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    
    print(f"\n{'Case':<30} {'mAP':<12} {'R-1':<12} {'R-5':<12} {'R-10':<12}")
    print("=" * 78)
    
    total_map = 0
    total_r1 = 0
    total_r5 = 0
    total_r10 = 0
    num_cases = len(results_per_case)
    
    for case_name, result in results_per_case.items():
        mAP = result['mAP']
        cmc = result['CMC']
        
        r1 = cmc[0]
        r5 = cmc[4] if len(cmc) > 4 else cmc[-1]
        r10 = cmc[9] if len(cmc) > 9 else cmc[-1]
        
        print(f"{case_name:<30} {mAP:>10.2%} {r1:>10.2%} {r5:>10.2%} {r10:>10.2%}")
        
        total_map += mAP
        total_r1 += r1
        total_r5 += r5
        total_r10 += r10
    
    print("-" * 78)
    print(f"{'Average':<30} {total_map/num_cases:>10.2%} {total_r1/num_cases:>10.2%} {total_r5/num_cases:>10.2%} {total_r10/num_cases:>10.2%}")
    print(f"\n{'='*80}")
    print(f"FINAL SCORE (Average Rank-1): {total_r1/num_cases:.2%}")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description="Calculate metrics from ranking CSV")
    parser.add_argument(
        "--csv_file",
        default="output/cross_attn/detreidx/alpha/256/evaluation_rankings_all_galleries.csv",
        help="Path to the ranking CSV file",
        type=str
    )
    
    args = parser.parse_args()
    
    # Calculate metrics
    results = calculate_metrics_from_csv(args.csv_file)
    
    # Print summary
    print_summary(results)


if __name__ == '__main__':
    main()

