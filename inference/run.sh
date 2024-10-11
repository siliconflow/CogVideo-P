torchrun --nproc_per_node=2 \
         --nnodes=1 \
         --master_addr="localhost" \
         --master_port=39534 \
         inference/cog_dist.py --prompt "A street artist, clad in a worn-out denim jacket and a colorful bandana, stands before a vast concrete wall in the heart, holding a can of spray paint, spray-painting a colorful bird on a mottled wall." --num_inference_steps 50
