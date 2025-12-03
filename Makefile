cp_cpu:
	torchrun --nproc_per_node=2 scripts/inference_cp.py --distributed --architecture=hf_pretrained --variant=ibm-granite/granite-3.2-8b-instruct --tokenizer=ibm-granite/granite-3.2-8b-instruct --device_type=cpu --unfuse_weights --compile --compile_dynamic --default_dtype=fp16 --fixed_prompt_length=64 --max_new_tokens=20 --timing=per-token --batch_size=1

cp_cpu_no_compile:
	torchrun --nproc_per_node=2 scripts/inference_cp.py --distributed --architecture=hf_pretrained --variant=ibm-granite/granite-3.2-8b-instruct --tokenizer=ibm-granite/granite-3.2-8b-instruct --device_type=cpu --unfuse_weights --default_dtype=fp16 --fixed_prompt_length=64 --max_new_tokens=20 --timing=per-token --batch_size=1
