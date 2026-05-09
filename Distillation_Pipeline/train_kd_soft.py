import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig
)
from datasets import load_dataset
from config import *

class LogitKDTrainer(Trainer):
    def __init__(self, teacher_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.teacher_model.eval()
        
        for p in self.teacher_model.parameters():
            p.requires_grad_(False)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):

        outputs_s = model(**inputs)
        loss_ce = outputs_s.loss
        logits_s = outputs_s.logits

        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
                outputs_t = self.teacher_model(**inputs)
                logits_t = outputs_t.logits

        labels = inputs["labels"]
        shift_logits_s = logits_s[..., :-1, :].contiguous()
        shift_logits_t = logits_t[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        valid_mask = shift_labels != -100
        
        if not valid_mask.any():
            return (loss_ce, outputs_s) if return_outputs else loss_ce

        valid_logits_s = shift_logits_s[valid_mask].to(torch.float32)
        valid_logits_t = shift_logits_t[valid_mask].to(torch.float32)

        min_vocab = min(valid_logits_s.size(-1), valid_logits_t.size(-1))
        valid_logits_s = valid_logits_s[..., :min_vocab]
        valid_logits_t = valid_logits_t[..., :min_vocab]

        log_p_s = F.log_softmax(valid_logits_s / TEMPERATURE, dim=-1)
        log_p_t = F.log_softmax(valid_logits_t / TEMPERATURE, dim=-1)

        p_t = torch.exp(log_p_t)
        kl_div = torch.sum(p_t * (log_p_t - log_p_s), dim=-1).mean()
        loss_kl = kl_div * (TEMPERATURE ** 2)
        loss = (1 - ALPHA) * loss_ce + ALPHA * loss_kl

        return (loss, outputs_s) if return_outputs else loss

def main():
    raw_dataset = load_dataset(DATASET_NAME, split="train").select(range(10000))
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL)
    if tokenizer.pad_token is None: 
        tokenizer.pad_token = tokenizer.eos_token

    def preprocess(example):
        if example.get("context"):
            prompt_text = f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['context']}\n\n### Response:\n"
        else:
            prompt_text = f"### Instruction:\n{example['instruction']}\n\n### Response:\n"

        response_text = example["response"] + tokenizer.eos_token
        
        prompt_ids = [tokenizer.bos_token_id] + tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        response_ids = tokenizer(response_text, add_special_tokens=False)["input_ids"]

        if len(prompt_ids) + len(response_ids) > MAX_LENGTH:
            allowed = MAX_LENGTH - len(prompt_ids)
            if allowed > 10:
                response_ids = response_ids[:allowed-1] + [tokenizer.eos_token_id]
            else:
                prompt_ids = prompt_ids[:MAX_LENGTH - 21]
                response_ids = response_ids[:20] + [tokenizer.eos_token_id]

        input_ids = prompt_ids + response_ids
        labels = [-100] * len(prompt_ids) + response_ids

        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels
        }

    dataset = raw_dataset.map(preprocess, remove_columns=raw_dataset.column_names)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    print("Loading Teacher...")
    teacher = AutoModelForCausalLM.from_pretrained(
        TEACHER_MODEL, 
        quantization_config=bnb_config, 
        device_map="auto"
    )
    
    print(f"Loading Phase 1 Student from: {STUDENT_MODEL_S}...")
    student = AutoModelForCausalLM.from_pretrained(
        STUDENT_MODEL_S, 
        torch_dtype=torch.float32, 
        device_map="auto"
    )
    student.config.use_cache = False
    student.config.pad_token_id = tokenizer.pad_token_id

    args = TrainingArguments(
        output_dir=DISTILLED_OUTPUT_DIR_S,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        lr_scheduler_type="cosine",
        weight_decay=0.05,
        warmup_steps=100,
        max_grad_norm=1.0,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        logging_steps=5,
        save_total_limit=2,
        remove_unused_columns=False
    )

    trainer = LogitKDTrainer(
        teacher_model=teacher, 
        model=student, 
        args=args,
        train_dataset=dataset, 
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True)
    )

    print(f"Starting Phase 2 Semantic Injection (T={TEMPERATURE}, Alpha={ALPHA})...")
    trainer.train()
    trainer.save_model(DISTILLED_OUTPUT_DIR_S)
    tokenizer.save_pretrained(DISTILLED_OUTPUT_DIR_S)
    print("Run complete.")

if __name__ == "__main__":
    main()