FROM shovalmishal/detr:base_alternative

CMD ["/bin/bash", "-c", "python -m torch.distributed.launch --nproc_per_node=$GPU_NUM --use_env FineTuning.py --dataset_path $SOURCE_DIR --output_dir $TARGET_DIR"]


