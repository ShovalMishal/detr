FROM shovalmishal/detr:base

CMD ["/bin/bash", "-c", "python FineTuning.py --batch_size 2 --no_aux_loss --eval --dataset_path /storage/shoval/detr/input/dota_dataset --output_dir /storage/shoval/detr/output_dir --resume /storage/shoval/detr/output/checkpoint0299.pth --num_workers 2"]
