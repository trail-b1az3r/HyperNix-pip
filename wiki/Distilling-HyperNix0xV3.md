# Distilling HyperNix0xV3 from HyperNix.3.1-mini

This recipe distils Compactbot/hypernix-3.1-mini (the V2 teacher) into a
native HyperNix0xV3 student. It is cross-architecture transfer, not a
checkpoint rename or a direct V2 state-dict load.

## Weight transfer

The default student retains the teacher's 32k tokenizer, 512-wide hidden
state, eight layers, 8-query/2-KV GQA layout, and 2,203-wide SwiGLU. The
script copies compatible token embeddings, output RMSNorm, per-layer
Q/K/V/O projections, and SwiGLU weights. When layer counts differ, teacher
layers are mapped evenly across the student stack.

Tidepool salience and summary-bias parameters, plus Keel persistent
slots/read/write/persist/gate parameters, are V3-specific. They keep the
V3 initializer and are trained by the distillation objective. The script
does not fabricate V2 equivalents for these new mechanisms.

## Objective

For each next-token position, training combines temperature-scaled
teacher/student KL divergence with hard next-token cross-entropy:

    loss = kd_weight * T² * KL(teacher || student)
         + (1 - kd_weight) * CE(student, next_token)

The teacher is frozen and runs under no_grad. The corpus supplies contexts;
teacher logits provide the architecture-transfer signal. Validation uses
a held-out deterministic shard (every 100th streamed source record).

## GPU and checkpoint behavior

CUDA is mandatory; CPU fallback is disabled. FP16 autocast is the default
for Pascal/GTX 1080 compatibility, while model parameters and optimizer
state remain FP32. Default microbatch is one sequence with eight-step
gradient accumulation. OOM errors recommend smaller settings. A
single-writer lock prevents two runs from corrupting the same output.
Training state includes model, optimizer, scheduler, and scaler for resume.

## Run

Install CUDA-enabled PyTorch, HyperNix-pip, safetensors, huggingface_hub,
transformers, and datasets. Then run:

    python scripts/distill_hypernix0xv3.py \
      --teacher Compactbot/hypernix-3.1-mini \
      --dataset HuggingFaceFW/fineweb-edu \
      --dataset-config sample-10BT \
      --seq-len 512 --batch-size 1 --grad-accum 8 \
      --steps 10000 --amp fp16

For a smaller GTX 1080 smoke run:

    python scripts/distill_hypernix0xv3.py --seq-len 256 --steps 100 \
      --eval-every 25 --eval-batches 4 --save-every 50

Resume with --resume outputs/hypernix0xv3-distilled/training_state.pt.
The best model is exported to best/, the final model to final/, and
training_state.pt contains the resumable training state.

The teacher model card reports 48.7M parameters, 32k BPE, eight layers,
512 hidden width, 8/2 GQA, SwiGLU width 2,203, context 512, and FP32
weights. Distillation results must be reported from held-out validation
and generation tests; training loss alone is not a quality claim.
