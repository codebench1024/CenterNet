python test.py ctdet --exp_id graduate_demo --load_model /home/konglingbin/project/dota/CenterNet/exp/ctdet/graduate_demo/model_last.pth --K 40 --gpus 3 --arch res_18 
cd /home/konglingbin/project/dota/DOTA_devkit
python ResultMerge.py
cd /home/konglingbin/project/dota/CenterNet/src
python tools/dota_to_plane.py
