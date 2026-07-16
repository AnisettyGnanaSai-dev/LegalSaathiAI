import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# MNC Configuration Map
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
DATASET_PATH = "data/synthetic_qa/train.jsonl"
OUTPUT_DIR = "models/legalsathi-qwen-lora"

def main():
    print("🚀 Initializing T4-Optimized, AMP-Bypassed QLoRA Pipeline...")

    # 1. Dataset Loading
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset missing at {DATASET_PATH}")
    
    print("📊 Loading Synthetic Dataset...")
    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

    def format_instruction(example):
        prompt = f"<|im_start|>system\nYou are LegalSathi AI, an expert Indian Legal Assistant.<|im_end|>\n<|im_start|>user\n{example['instruction']}<|im_end|>\n<|im_start|>assistant\n{example['response']}<|im_end|>"
        return {"text": prompt}
    
    dataset = dataset.map(format_instruction)

    # 2. Strict Quantization Config
    print("⚙️ Configuring 4-Bit Quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16 # Strictly FP16 for T4 compute
    )

    # 3. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" 

    # 4. Base Model Loading
    print("🧠 Loading Base Model with Eager Attention...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        attn_implementation="eager" # Bypasses Flash Attention bugs on T4 GPUs
    )

    # =====================================================================
    # THE ULTIMATE SCRUB: PURGE PARAMETERS *AND* BUFFERS
    # This destroys the hidden Qwen Rotary Embeddings (RoPE) that caused crashes.
    # =====================================================================
    print("🛡️ Purging ALL hidden BFloat16 tensors from the architecture...")
    for name, param in model.named_parameters():
        if param.dtype == torch.bfloat16:
            param.data = param.data.to(torch.float32)
            
    for name, buffer in model.named_buffers():
        if buffer.dtype == torch.bfloat16:
            buffer.data = buffer.data.to(torch.float32)
    # =====================================================================

    # 5. Preparing for LoRA
    print("🔧 Injecting LoRA Adapters...")
    model = prepare_model_for_kbit_training(model)
    
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # Apply PEFT manually so we control exactly what goes into the Trainer
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 6. AMP-Bypassed Training Arguments
    print("📈 Compiling AMP-Bypassed Training Arguments...")
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=1,
        num_train_epochs=5,    # <--- THE UPDATE: Replaced max_steps with full dataset passes
        fp16=False,            # <--- THE SILVER BULLET: Disables PyTorch GradScaler completely
        bf16=False,            # <--- Ensure BFloat16 is off
        optim="paged_adamw_8bit",
        max_grad_norm=0.3,
        gradient_checkpointing=True, 
        gradient_checkpointing_kwargs={"use_reentrant": False},
        save_strategy="no",
        report_to="none"
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # 7. Trainer Initialization
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        processing_class=tokenizer, # <--- TRL 0.12+ compatibility
        args=args,
        data_collator=data_collator,
    )

    # 8. Execution
    print("🔥 Starting Safe Model Training Loop...")
    model.config.use_cache = False 
    trainer.train()

    # 9. Artifact Preservation
    print(f"💾 Saving Finetuned LoRA Adapters to {OUTPUT_DIR}...")
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("✅ MNC-Grade Fine-Tuning Complete!")

if __name__ == "__main__":
    main()