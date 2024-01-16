from transformers import AutoTokenizer
import transformers
import torch

model = 'Llama-2-7b-chat-hf'

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)


sequences = pipeline(
    '你現在是一位專業的Android APP開發者，給我一個能用java畫圓的範例程式碼\n',
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    max_length=5000,
)
for seq in sequences:
    print(f"Result: {seq['generated_text']}")

