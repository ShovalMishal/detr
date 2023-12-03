FROM shovalmishal/detr:base

CMD ["/bin/bash", "-c", "python -m torch.distributed.launch --nproc_per_node=8 --use_env FineTuning.py --dataset_path /storage/shoval/detr/input/dota_dataset_128X128 --output_dir /storage/shoval/detr/output_128 --batch_size 2 --num_workers 2"]
