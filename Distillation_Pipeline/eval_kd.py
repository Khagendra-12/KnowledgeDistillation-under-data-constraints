import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
from evaluate import load
from tqdm import tqdm
import numpy as np
import time
from bert_score import score as bert_score_compute
import transformers
def _fallback_build_inputs(self, token_ids_0, token_ids_1=None):
    if token_ids_1 is None:
        return token_ids_0
    return token_ids_0 + token_ids_1
transformers.PreTrainedTokenizerBase.build_inputs_with_special_tokens = _fallback_build_inputs
transformers.PreTrainedTokenizerFast.build_inputs_with_special_tokens = _fallback_build_inputs
from config import DISTILLED_OUTPUT_DIR, DATASET_NAME, MAX_LENGTH, TEACHER_MODEL

rouge = load("rouge")
bleu = load("bleu")
exact_match = load("exact_match")
meteor = load("meteor")

def format_prompt(example):
    if example.get("context"):
        return f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['context']}\n\n### Response:\n"
    return f"### Instruction:\n{example['instruction']}\n\n### Response:\n"

def compute_perplexity_and_kl(student_model, teacher_model, tokenizer, dataset):
    student_model.eval()
    if teacher_model: teacher_model.eval()
        
    losses, kl_divs = [], []
    
    print("Evaluating Perplexity and KL Divergence...")
    for example in tqdm(dataset):
        full_text = format_prompt(example) + example["response"] + tokenizer.eos_token
        inputs = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH).to(student_model.device)
        
        with torch.no_grad():
            safe_input_ids_s = inputs["input_ids"].clone()
            safe_input_ids_s[safe_input_ids_s >= student_model.config.vocab_size] = 0
            
            outputs_s = student_model(
                input_ids=safe_input_ids_s,
                attention_mask=inputs["attention_mask"],
                labels=safe_input_ids_s
            )
            losses.append(outputs_s.loss.item())

            if teacher_model:
                safe_input_ids_t = inputs["input_ids"].clone()
                safe_input_ids_t[safe_input_ids_t >= teacher_model.config.vocab_size] = 0
                
                outputs_t = teacher_model(
                    input_ids=safe_input_ids_t,
                    attention_mask=inputs["attention_mask"]
                )
                
                logits_s = outputs_s.logits[0, :-1, :]
                logits_t = outputs_t.logits[0, :-1, :]
                min_vocab = min(logits_s.size(-1), logits_t.size(-1))
                log_p_s = F.log_softmax(logits_s[:, :min_vocab], dim=-1)
                p_t = F.softmax(logits_t[:, :min_vocab], dim=-1)
                kl = F.kl_div(log_p_s, p_t, reduction='batchmean')
                kl_divs.append(kl.item())

    return np.exp(np.mean(losses)), (np.mean(kl_divs) if kl_divs else None)

def generate_response_with_latency(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_LENGTH).to(model.device)
    
    safe_input_ids = inputs["input_ids"].clone()
    safe_input_ids[safe_input_ids >= model.config.vocab_size] = 0
    inputs["input_ids"] = safe_input_ids
    
    safe_pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id < model.config.vocab_size else 0

    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=128, 
            num_beams=4, 
            early_stopping=True, 
            pad_token_id=safe_pad_token_id
        )

    latency = time.time() - start_time
    input_length = inputs.input_ids.shape[1]
    generated_tokens = outputs[0][input_length:]
    prediction = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    tps = len(generated_tokens) / latency if latency > 0 else 0
    return prediction.strip(), latency, tps

def compute_all_metrics(student_model, tokenizer, dataset):
    preds, refs, latencies, tps_list = [], [], [], []

    print("Generating responses for Comprehensive Evaluation...")
    for example in tqdm(dataset):
        prompt = format_prompt(example)
        prediction, latency, tps = generate_response_with_latency(student_model, tokenizer, prompt)
        preds.append(prediction)
        refs.append(example["response"].strip())
        latencies.append(latency)
        tps_list.append(tps)

    rouge_res = rouge.compute(predictions=preds, references=refs)
    em_res = exact_match.compute(predictions=preds, references=refs)

    print("Calculating BERTScore (Running on CPU to prevent VRAM crashes)...")
    try:
        P, R, F1 = bert_score_compute(
            preds, 
            refs, 
            lang="en", 
            model_type="roberta-large", 
            device="cpu",      
            batch_size=8,      
            verbose=False
        )
        avg_feature_sim = F1.mean().item()
    except Exception as e:
        print(f"\n[CRITICAL ERROR] Native BERTScore failed because: {e}\n")
        avg_feature_sim = "Error"

    return {
        "rouge": rouge_res, 
        "bleu": bleu.compute(predictions=preds, references=[[r] for r in refs]), 
        "meteor": meteor.compute(predictions=preds, references=refs)["meteor"],
        "exact_match": em_res["exact_match"], 
        "feature_similarity": avg_feature_sim,
        "avg_latency": np.mean(latencies), 
        "avg_tokens_per_sec": np.mean(tps_list)
    }

def get_model_size_ratio(teacher_model, student_model):
    t_params = sum(p.numel() for p in teacher_model.parameters())
    s_params = sum(p.numel() for p in student_model.parameters())
    return t_params, s_params, t_params / s_params

def main():
    print(f"Loading Student model from: {DISTILLED_OUTPUT_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(DISTILLED_OUTPUT_DIR)
    student = AutoModelForCausalLM.from_pretrained(DISTILLED_OUTPUT_DIR, torch_dtype=torch.float32, device_map="auto")
    
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    try:
        teacher = AutoModelForCausalLM.from_pretrained(TEACHER_MODEL, quantization_config=bnb_config, device_map="auto")
    except Exception as e:
        print(f"Warning: Could not load teacher model. Error: {e}")
        teacher = None

    test_ds = load_dataset(DATASET_NAME, split="train").select(range(10000, 10500))

    if teacher:
        t_params, s_params, size_ratio = get_model_size_ratio(teacher, student)

    ppl, kl_div = compute_perplexity_and_kl(student, teacher, tokenizer, test_ds)
    val_ds_small = test_ds.select(range(100))
    metrics = compute_all_metrics(student, tokenizer, val_ds_small)

    print("\nKD MODEL COMPREHENSIVE RESULTS\n")
    print(f"Perplexity:         {ppl:.4f}")
    if kl_div: print(f"KL Divergence:      {kl_div:.4f}")
    print("\n--- Structural Metrics ---")
    print(f"ROUGE-1:            {metrics['rouge']['rouge1']:.4f}")
    print(f"ROUGE-2:            {metrics['rouge']['rouge2']:.4f}")
    print(f"ROUGE-L:            {metrics['rouge']['rougeL']:.4f}")
    print(f"BLEU:               {metrics['bleu']['bleu']:.4f}")
    print(f"METEOR:             {metrics['meteor']:.4f}")
    print(f"Accuracy (EM):      {metrics['exact_match']:.4f}")
    print("\n--- Semantic Metrics ---")
    if isinstance(metrics['feature_similarity'], str):
        print(f"Feature Sim (F1):   {metrics['feature_similarity']}")
    else:
        print(f"Feature Sim (F1):   {metrics['feature_similarity']:.4f} (BERTScore)")
    print("\n--- Performance Metrics ---")
    if teacher: print(f"Compression Ratio:  {size_ratio:.2f}x")
    print(f"Avg Latency:        {metrics['avg_latency']:.2f} s/prompt")
    print(f"Speed:              {metrics['avg_tokens_per_sec']:.2f} tokens/s")


if __name__ == "__main__":
    main()