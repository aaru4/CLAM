import time
import os
import argparse
import pdb
from functools import partial
import openslide

import torch
import torch.nn as nn
import timm
from torch.utils.data import DataLoader
from PIL import Image
import h5py
import openslide
from tqdm import tqdm
import pandas as pd
import numpy as np

from utils.file_utils import save_hdf5
from dataset_modules.dataset_h5 import Dataset_All_Bags, Whole_Slide_Bag_FP
from models import get_encoder

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def compute_w_loader(output_path, loader, model, verbose=0):
	if verbose > 0:
		print(f'processing a total of {len(loader)} batches')
	mode = 'w'
	for count, data in enumerate(tqdm(loader)):
		with torch.inference_mode():
			batch = data['img']
			coords = data['coord'].numpy().astype(np.int32)
			batch = batch.to(device, non_blocking=True)
			features = model(batch)
			features = features.cpu().numpy().astype(np.float32)
			asset_dict = {'features': features, 'coords': coords}
			save_hdf5(output_path, asset_dict, attr_dict=None, mode=mode)
			mode = 'a'
	return output_path


parser = argparse.ArgumentParser(description='Feature Extraction')
parser.add_argument('--data_h5_dir', type=str, default=None)
parser.add_argument('--data_slide_dir', type=str, default=None)
parser.add_argument('--slide_ext', type=str, default='.svs')
parser.add_argument('--csv_path', type=str, default=None)
parser.add_argument('--feat_dir', type=str, default=None)
parser.add_argument('--cohort_csv', type=str, default=None,
                    help='CSV with dicom col (study_uid/series_uid) to build nested path lookup')
parser.add_argument('--model_name', type=str, default='resnet50_trunc',
                    choices=['resnet50_trunc', 'uni_v1', 'conch_v1'])
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--no_auto_skip', default=False, action='store_true')
parser.add_argument('--target_patch_size', type=int, default=224)
args = parser.parse_args()


def build_slide_path_lookup(data_slide_dir, cohort_csv=None):
    """
    Walk data_slide_dir/study/series/*.dcm, pick largest file per series.
    Returns dict: {series_uid: full_path_to_largest_dcm}
    Optionally filtered to series UIDs present in cohort_csv['dicom'].
    """
    # Get valid series UIDs from cohort CSV (level2 = part after '/')
    valid_series = None
    if cohort_csv is not None:
        df = pd.read_csv(cohort_csv)
        valid_series = set(
            str(r).split('/')[-1].strip()
            for r in df['dicom'].dropna()
        )
        print(f"Cohort CSV: {len(valid_series)} unique series UIDs")

    lookup = {}
    print("Building slide path lookup (walking nested dirs)...")
    for study_entry in os.scandir(data_slide_dir):
        if not study_entry.is_dir():
            continue
        for series_entry in os.scandir(study_entry.path):
            if not series_entry.is_dir():
                continue
            series_uid = series_entry.name
            if valid_series is not None and series_uid not in valid_series:
                continue
            # Pick largest file in this series dir
            candidates = []
            for f in os.scandir(series_entry.path):
                if f.is_file():
                    try:
                        candidates.append((os.path.getsize(f.path), f.path))
                    except OSError:
                        continue
            if candidates:
                _, best_path = max(candidates, key=lambda x: x[0])
                lookup[series_uid] = best_path

    print(f"Lookup built: {len(lookup)} series found")
    return lookup


def worker_init_fn(worker_id):
	worker_info = torch.utils.data.get_worker_info()
	dataset = worker_info.dataset
	dataset.wsi = openslide.OpenSlide(dataset.slide_path)


if __name__ == '__main__':
	print('initializing dataset')
	csv_path = args.csv_path
	if csv_path is None:
		raise NotImplementedError

	bags_dataset = Dataset_All_Bags(csv_path)

	os.makedirs(args.feat_dir, exist_ok=True)
	os.makedirs(os.path.join(args.feat_dir, 'pt_files'), exist_ok=True)
	os.makedirs(os.path.join(args.feat_dir, 'h5_files'), exist_ok=True)
	dest_files = os.listdir(os.path.join(args.feat_dir, 'pt_files'))

	model, img_transforms = get_encoder(args.model_name, target_img_size=args.target_patch_size)
	_ = model.eval()
	model = model.to(device)
	total = len(bags_dataset)

	# ── Build nested path lookup once ─────────────────────────────────────────
	slide_path_lookup = build_slide_path_lookup(
	    args.data_slide_dir,
	    cohort_csv=args.cohort_csv
	)

	loader_kwargs = {'num_workers': 8, 'pin_memory': True} if device.type == "cuda" else {}

	for bag_candidate_idx in tqdm(range(total)):
		slide_id = bags_dataset[bag_candidate_idx].split(args.slide_ext)[0]
		bag_name = slide_id + '.h5'
		h5_file_path = os.path.join(args.data_h5_dir, 'patches', bag_name)
		print('\nprogress: {}/{}'.format(bag_candidate_idx, total))
		print(slide_id)

		if not args.no_auto_skip and slide_id + '.pt' in dest_files:
			print('skipped {}'.format(slide_id))
			continue

		# ── Resolve nested DCM path ────────────────────────────────────────────
		slide_file_path = slide_path_lookup.get(slide_id)
		if slide_file_path is None:
			# Fallback to flat path for non-DICOM slides
			slide_file_path = os.path.join(args.data_slide_dir, slide_id + args.slide_ext)
			print(f'WARNING: {slide_id} not in lookup, falling back to flat path')

		output_path = os.path.join(args.feat_dir, 'h5_files', bag_name)
		time_start = time.time()

		try:
			wsi = openslide.open_slide(os.path.realpath(slide_file_path))
		except Exception as e:
			print(f'ERROR opening slide {slide_id}: {e}, skipping...')
			continue

		slide_path = os.path.realpath(slide_file_path)
		try:
			dataset = Whole_Slide_Bag_FP(file_path=h5_file_path,
										 wsi=wsi,
										 img_transforms=img_transforms,
										 slide_path=slide_path)
		except Exception as e:
			print(f'ERROR loading patches for {slide_id}: {e}, skipping...')
			continue

		loader = DataLoader(dataset=dataset, batch_size=args.batch_size,
		                    worker_init_fn=worker_init_fn, **loader_kwargs)
		output_file_path = compute_w_loader(output_path, loader=loader, model=model, verbose=1)

		time_elapsed = time.time() - time_start
		print('\ncomputing features for {} took {} s'.format(output_file_path, time_elapsed))

		with h5py.File(output_file_path, "r") as file:
			features = file['features'][:]
			print('features size: ', features.shape)
			print('coordinates size: ', file['coords'].shape)

		features = torch.from_numpy(features)
		bag_base, _ = os.path.splitext(bag_name)
		torch.save(features, os.path.join(args.feat_dir, 'pt_files', bag_base + '.pt'))
