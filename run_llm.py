from transformers import AutoTokenizer
import transformers
import torch

model = 'model/Llama-2-7b-chat-hf'

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)


if __name__ == '__main__':
    buffer = ''

    while True:
        prompt = input('Prompt > ')
        if prompt == 'exit': break
        sequences = pipeline(
            f"{buffer}[INST]{prompt}[/INST]",
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            max_length=5000,
        )

        print(sequences[0]['generated_text'].split('[/INST]')[-1])
        buffer = sequences[0]['generated_text']