cd $CenterTrack/models
gdown --id 1kBX4AgQj7R7HvgMdbgBcwvIac-IFp95h
gdown --id 1_VtGal9UzZE3n3QcVa0brZ7nNAwqPzd-
gdown --id 1Kv8kA7VLBqVst1ZcfB9gRH8TWs5oPN_h

cd $CenterTrack_ROOT
mkdir data
cd data
mkdir kitti_tracking
cd kitti_tracking

mkdir data_tracking_image_2
mkdir label_02
mkdir data_tracking_calib

cd data_tracking_image_2
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_tracking_image_2.zip
unzip data_tracking_image_2.zip
rm data_tracking_image_2.zip

cd label_02
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_tracking_label_2.zip
unzip data_tracking_label_2.zip
rm data_tracking_label_2.zip
mv ~/CenterTrack/data/kitti_tracking/label_02/training/label_02/* ~/CenterTrack/data/kitti_tracking/label_02/
rm -r trainings

cd data_tracking_calib
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_tracking_calib.zip
unzip data_tracking_calib.zip
rm data_tracking_calib.zip

cd $CenterTrack/src/tools
python convert_kittitrack_to_coco.py



