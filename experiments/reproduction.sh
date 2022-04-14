cd src

# Full framework 
python test.py tracking --exp_id mot17_half --dataset mot --dataset_version 17halfval --pre_hm --ltrb_amodal --track_thresh 0.4 --pre_thresh 0.5 --load_model ../models/mot17_half.pth

# Without offset
python test.py tracking --exp_id mot17_half --dataset mot --dataset_version 17halfval --pre_hm --zero_tracking --ltrb_amodal --track_thresh 0.4 --pre_thresh 0.5 --load_model ../models/mot17_half.pth

# Without heatmap
python test.py tracking --exp_id mot17_half --dataset mot --dataset_version 17halfval --zero_pre_hm --ltrb_amodal --track_thresh 0.4 --load_model ../models/mot17_half.pth

# Detection only
python test.py tracking --exp_id mot17_half --dataset mot --dataset_version 17halfval --zero_pre_hm --zero_tracking --ltrb_amodal --track_thresh 0.4 --load_model ../models/mot17_half.pth

cd ..
