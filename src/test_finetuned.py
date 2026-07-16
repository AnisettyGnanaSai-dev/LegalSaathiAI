import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER_DIR = "models/legalsathi-qwen-lora"

def main():
    print("🚀 Initializing MNC-Grade Model Validation...")

    if not os.path.exists(ADAPTER_DIR):
        print(f"❌ Error: Adapter not found at {ADAPTER_DIR}")
        return

    # 1. Load Tokenizer
    print("🧠 Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

    # 2. Configure 4-bit Quantization (Must match training setup)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )

    # 3. Load Base Model
    print("📦 Loading Base Model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    # 4. Attach Your Custom LoRA Weights!
    print("🔌 Plugging in the LegalSathi Finetuned LoRA Adapter...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    
    # 5. The Ultimate Test: The Flipkart Telugu Prompt
    print("\n" + "="*50)
    print("🧪 TEST: Generating Telugu Complaint Letter")
    print("="*50)

    # We use the exact ChatML format we used during training
    prompt = (
        "<|im_start|>system\n"
        "You are LegalSathi AI, an expert Indian Legal Assistant.\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        "Draft a formal consumer complaint letter in Telugu. User Details: Name: Gnanasai, Order ID: 123, Company: Flipkart. Issue: Received wrong product, no refund after 1 month.\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    # 6. Generate the Output
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    print("🤖 AI is drafting...\n")
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=300, 
            temperature=0.1,      # Low temperature for strict legal formatting
            repetition_penalty=1.1 # Prevents looping
        )
    
    # Decode and extract just the assistant's reply
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Clean up the output string to show only the AI's response
    final_output = response.split("assistant\n")[-1]
    
    print(final_output)
    print("\n" + "="*50)
    print("✅ Validation Complete!")

if __name__ == "__main__":
    main()