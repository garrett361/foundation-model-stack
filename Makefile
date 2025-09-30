MODEL=ibm-granite/granite-3.3-2b-instruct
WORLD_SIZE=2
# NOTE: @goon - need --min_pad_length > 0 to provide position_ids to the model, for some reason.
# COMMON_ARGS=--architecture=hf_pretrained --variant=$(MODEL) --tokenizer=$(MODEL) --device_type=cpu --default_dtype=fp16 --min_pad_length=1 --no_use_cache
COMMON_ARGS=./scripts/inference.py  --architecture=hf_pretrained --variant=$(MODEL) --tokenizer=$(MODEL) --device_type=cpu --default_dtype=fp16 --min_pad_length=1
all:
	echo "Choose another target"

infer:
	python3 $(COMMON_ARGS)

infer_tp:
	torchrun --nproc_per_node=$(WORLD_SIZE) $(COMMON_ARGS) --distributed


