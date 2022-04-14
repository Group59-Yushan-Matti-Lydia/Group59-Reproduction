cd src

# Reproduce results on KITTI in Table 4
python test.py tracking --exp_id kitti_half --dataset kitti_tracking --dataset_version val_half --pre_hm --track_thresh 0.4 --load_model ../models/kitti_half.pth

# Optimise output and rendering threshold 
# --track_threshold: output thresholds
# --pre_threshold: rendering thresholds
python test.py tracking --exp_id kitti_half --dataset kitti_tracking --dataset_version val_half --pre_hm --track_thresh 0.4 --pre_thresh 0.4 --load_model ../models/kitti_half.pth

cd ..
