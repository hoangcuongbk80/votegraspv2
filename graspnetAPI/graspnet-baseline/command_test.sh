#CUDA_VISIBLE_DEVICES=0 python test.py --dump_dir logs/dump_rs --checkpoint_path logs/log_rs/checkpoint.tar --camera realsense --dataset_root /data/Benchmark/graspnet
# CUDA_VISIBLE_DEVICES=0 python test.py --dump_dir logs/dump_kn --checkpoint_path logs/log_kn/checkpoint.tar --camera kinect --dataset_root /data/Benchmark/graspnet

CUDA_VISIBLE_DEVICES=0 python test.py --dump_dir /media/hoang/HD-PZFU3/datasets/graspnet/predictions --checkpoint_path /media/hoang/HD-PZFU3/datasets/graspnet/checkpoint.tar --camera kinect --dataset_root /media/hoang/HD-PZFU3/datasets/graspnet
