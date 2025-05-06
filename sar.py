import math
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import CrossEncoder
from typing import List, Callable
from datasets import load_from_disk, load_dataset
from tqdm import tqdm
import json
class TokenSAR:
    """
    A class to compute the sum of all token-level uncertainty scores (TokenSAR)
    in a sentence given a prompt x. The uncertainty for token z_i is:

        ET(z_i) = -log p(z_i | s_{<i}, x) * ~RT(z_i, s, x),

    where ~RT(z_i) is the (normalized) relevance score for that token,
    and p(z_i | s_{<i}, x) is computed by a causal language model.
    """
    
    def __init__(self, 
                 cross_encoder_model_name: str = "cross-encoder/stsb-roberta-large",
                 causal_lm_name: str = "openai-community/gpt2-large"):
        """
        Initializes the TokenSAR class with two models:
         - A CrossEncoder for semantic similarity (to compute relevance)
         - A causal language model (e.g., GPT-2) for token probabilities.
         
        Args:
            cross_encoder_model_name (str): Hugging Face model name for the CrossEncoder.
            causal_lm_name (str): Hugging Face model name for the causal language model.
        """
        # Cross-Encoder for similarity g(·, ·)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cross_encoder = CrossEncoder(cross_encoder_model_name, device="cuda")
        
        # Causal LM & tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(causal_lm_name)
        self.causal_lm = AutoModelForCausalLM.from_pretrained(causal_lm_name).to(self.device)
        
    def _calculate_relevance_scores(self, x: str, s: str):
        """
        Private method to compute raw relevance scores for each token in s, i.e.
            RT(z_i) = 1 - g(x ∪ s, x ∪ s \ {z_i}).
        
        Then normalizes them (L1) to produce ~RT(z_i) so they sum to 1.
        
        Args:
            x (str): The prompt/context.
            s (str): The sentence to compute token relevance for.
            
        Returns:
            list of floats: The normalized relevance scores ~RT(z_i).
        """
        tokens = s.split()
        
        # Full original text
        original_text = f"{x.strip()} {s.strip()}"
        
        # Compute raw RT
        raw_scores = []
        for i in range(len(tokens)):
            # Sentence minus token i
            modified_tokens = tokens[:i] + tokens[i+1:]
            modified_text = f"{x.strip()} {' '.join(modified_tokens)}"

            # Similarity g(original_text, modified_text)
            with torch.no_grad():
                sim = self.cross_encoder.predict([(original_text, modified_text)])[0]
            
            # RT(z_i) = 1 - g(·)
            rt = 1.0 - sim
            raw_scores.append(rt)
        
        # L1 normalization so sum of ~RT(z_i) = 1
        total_rt = sum(raw_scores)
        if total_rt == 0:
            # If all are zero (edge case), return a uniform or all zeros
            # Here we choose zeros for clarity.
            return [0.0] * len(tokens)
        normalized_relevance = [rt / total_rt for rt in raw_scores]
        
        return normalized_relevance
    
    def _calculate_uncertainty_scores(self, x: str, s: str, relevance_scores: list):
        """
        Private method to compute ET(z_i) = -log p(z_i | s_{<i}, x) * ~RT(z_i).
        
        Args:
            x (str): The prompt/context.
            s (str): The sentence.
            relevance_scores (list): The normalized relevance scores ~RT(z_i) for each token.
            
        Returns:
            list of floats: ET(z_i) for each token in s.
        """
        tokens = s.split()
        assert len(tokens) == len(relevance_scores), \
            "Mismatch between number of tokens and number of relevance scores."
        
        uncertainty_scores = []
        for i, token in enumerate(tokens):
            # Build the context with tokens up to (but not including) the i-th token
            context_str = f"{x.strip()} {' '.join(tokens[:i])}"
            
            # Encode context
            input_ids = self.tokenizer.encode(context_str, return_tensors="pt").to(self.device)
            
            # Forward pass
            with torch.no_grad():
                outputs = self.causal_lm(input_ids)
            logits = outputs.logits  # shape: [batch_size=1, seq_len, vocab_size]
            
            # Get the logits for the next token
            next_token_logits = logits[0, -1, :]
            probs = torch.softmax(next_token_logits, dim=-1)
            
            # Handle potential subword tokens
            sub_ids = self.tokenizer.encode(token, add_special_tokens=False)
            if len(sub_ids) == 1:
                # Single subword
                prob = probs[sub_ids[0]].item()
            else:
                # For multiple subwords, approximate by multiplying subword probabilities
                prob = 1.0
                temp_input_ids = input_ids.clone().to(self.device)
                for sub_id in sub_ids:
                    with torch.no_grad():
                        temp_outputs = self.causal_lm(temp_input_ids)
                    temp_logits = temp_outputs.logits[0, -1, :]
                    temp_probs = torch.softmax(temp_logits, dim=-1)
                    
                    # Multiply prob
                    prob *= temp_probs[sub_id].item()
                    
                    # Append sub_id to context for next subword
                    new_sub_id = torch.tensor([[sub_id]]).to(self.device)
                    temp_input_ids = torch.cat([temp_input_ids, new_sub_id], dim=1)
            
            # - log(prob)
            neg_log_prob = -math.log(prob + 1e-12)
            
            # Multiply by the token's relevance score
            ET_zi = neg_log_prob * relevance_scores[i]
            uncertainty_scores.append(ET_zi)
        
        return uncertainty_scores

    def calculate_similarity(self, s1: str, s2: str) -> float:
        """
        Public method to compute the similarity between two sentences s1 and s2.
        
        Args:
            s1 (str): The first sentence.
            s2 (str): The second sentence.
            
        Returns:
            float: The similarity score in [0, 1].
        """
        return self.cross_encoder.predict([(s1, s2)])[0]
    
    def calculate_sar(self, x: str, s: str) -> float:
        """
        Public method to compute the "TokenSAR", i.e., the sum of
        all token-level uncertainty scores for the sentence s.
        
        TokenSAR(s) = sum_i [ ET(z_i) ].
        
        Args:
            x (str): The prompt/context.
            s (str): The sentence to evaluate.
            
        Returns:
            float: The total uncertainty score for the entire sentence s.
        """
        # 1) Compute normalized relevance ~RT(z_i)
        relevance_scores = self._calculate_relevance_scores(x, s)
        
        # 2) Compute ET(z_i) for each token
        uncertainty_scores = self._calculate_uncertainty_scores(x, s, relevance_scores)
        
        # 3) Sum them up
        token_sar = sum(uncertainty_scores)
        
        return token_sar



def generate_k_samples(
    x: str,
    # s: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    k: int = 5,
    max_new_tokens: int = 30,
    do_sample: bool = True,
    top_p: float = 0.9,
    top_k: int = 50,
    temperature: float = 1.0,
    device: torch.device = "cpu"
):
    """
    Generates k sample continuations from a causal LM for the prompt x.
    """
    # Encode the prompt and move tensors to the GPU
    encoded_input = tokenizer(
        x,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(device)  # Move input to GPU

    input_ids = encoded_input["input_ids"]
    attention_mask = encoded_input["attention_mask"]

    # get the length of tokens of the ground truth s
    # s_tokens = tokenizer(s, return_tensors="pt", padding=True, truncation=True).to(device)
    # max_new_tokens = s_tokens["input_ids"].shape[1]
    # Generate k samples
    generations = []
    for _ in range(k):
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id
        )

        # Only decode the newly generated portion
        generated_ids = output_ids[0, input_ids.shape[-1]:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        generations.append(generated_text)
    
    return generations




def compute_overall_SAR(
    s: str,                   # Ground truth sentence
    S: List[str],             # Candidate set (samples + possibly ground truth)
    x: str,                   # Prompt
    token_sar_fn: Callable,   # Function that computes TOKENSAR(s, x) -> float
    similarity_fn: Callable,  # Function that computes g(s_j, s_k) in [0,1]
    t: float = 1.0            # Hyperparameter
):
    
    p_prime = {}
    for s_j in S:
        tsar = token_sar_fn(x, s_j)   # e.g., 2.37
        p_prime[s_j] = math.exp(-tsar)
    
    assert s not in p_prime, "Ground truth sentence found in candidate set S."
    
    p_prime[s] = math.exp(-token_sar_fn(x, s))  # Add the ground truth

    sum_term = 0.0
    for k, s_k in enumerate(S):
 
        sum_term += similarity_fn(s, s_k) * p_prime[s]

    result = -math.log(p_prime[s] + sum_term/t)
    return result



class SAR:
    def __init__(self, token_sar_calculator: TokenSAR):
        self.token_sar_calculator = token_sar_calculator


# ------------------ Example usage ------------------
if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model_name = "EleutherAI/gpt-neo-1.3B"
    # model_name = "openai-community/gpt2-large"
    # model_name = "openai-community/gpt2-large"
    # model_name = "oe2015/gptneo_1.3b_wikimia"
    # model_name = "oe2015/gpt2-large-wikimia"
    # model_name = "KareemElzeky/gpt-neo-SFT-15epoch"
    # model_name = "KareemElzeky/gpt2-large-SFT-15epoch"
    model_name = "DaniilOr/sft-gpt2-large-15batch"
    # model_name = "DaniilOr/sft-gptneo-15batch"

    output_path = "uq_results/uncontrolled/uq_scores_mintaka_contaminated_gpt_2_both.json"

    dataset_name ="mintaka"

    contaminated = True 

    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)


    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id


    token_sar_calculator = TokenSAR(
        cross_encoder_model_name="cross-encoder/stsb-roberta-large",
        causal_lm_name="openai-community/gpt2-large"
    )


    uq_scores = []

    if dataset_name == "mintaka" and contaminated:
        dataset = load_from_disk("/home/kareem.elzeky/NLG/Project/data/mintaka")
    else:
        dataset = load_dataset("swj0419/WikiMIA", split="WikiMIA_length128")

    # select only the rows that has label = 1 for contaminated, 0 for uncontaminated
    if dataset_name == "mintaka" and contaminated:
        dataset = dataset["train"]
    elif dataset_name == "mintaka" and not contaminated:
        dataset = dataset["test"]
    elif dataset_name == "wikimia" and contaminated:
        dataset = dataset.filter(lambda x: x["label"] == 1)
    else:
        dataset = dataset.filter(lambda x: x["label"] == 0)


    dataset = dataset.select(range(100))
    for row in tqdm(dataset, desc= "Calculating UQ Scores"):
        # for Mintaka
        if dataset_name == "mintaka":
            x_prompt = f"### Question: {row['messages'][0]['content']}\n### Answer:"
            s_sentence = row["messages"][1]["content"]

        #  for WikiMIA
        if dataset_name == "wikimia":
            x_prompt = row['input'][:len(row['input'])//2]
            s_sentence = row["input"][len(row["input"])//2:]
        
        generations = generate_k_samples(
            x_prompt,
            model,
            tokenizer,
            k=5,
            max_new_tokens=30,
            do_sample=True,
            top_p=0.9,
            top_k=50,
            temperature=3.0,
            device=device
        )
        sar = compute_overall_SAR(
            s_sentence,
            generations,
            x_prompt,
            token_sar_calculator.calculate_sar,
            similarity_fn=token_sar_calculator.calculate_similarity,
        )

        uq_scores.append(sar)
    with open(output_path, "w") as f:
        json.dump(uq_scores, f)
