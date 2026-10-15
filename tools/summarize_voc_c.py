"""Print per-corruption AP50 / mAP and their average for a VOC-C run.

Usage: python tools/summarize_voc_c.py work_dirs/voc_c
"""
import argparse
import glob
import json
import os.path as osp

CORRUPTIONS = ('gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
               'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost',
               'fog', 'brightness', 'contrast', 'elastic_transform',
               'pixelate', 'jpeg_compression')


def latest_metrics(run_dir):
    """Metrics json written by mmengine's LoggerHook after testing."""
    files = sorted(glob.glob(osp.join(run_dir, '*', '*.json')))
    for path in reversed(files):
        with open(path) as f:
            metrics = json.load(f)
        if 'coco/bbox_mAP_50' in metrics:
            return metrics
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('work_dir')
    args = parser.parse_args()

    rows = []
    for c in CORRUPTIONS:
        m = latest_metrics(osp.join(args.work_dir, c))
        if m is None:
            print(f'{c:<18} (missing)')
            continue
        ap50, ap = 100 * m['coco/bbox_mAP_50'], 100 * m['coco/bbox_mAP']
        rows.append((ap50, ap))
        print(f'{c:<18} AP50 {ap50:5.1f}   mAP {ap:5.1f}')
    if rows:
        n = len(rows)
        print(f'{"average":<18} AP50 {sum(r[0] for r in rows) / n:5.1f}   '
              f'mAP {sum(r[1] for r in rows) / n:5.1f}   ({n}/15 corruptions)')


if __name__ == '__main__':
    main()
