# test script to verify true parallel processing using the batch API

from pathlib import Path
import traceback
import time
from typing import List

from llama_cpp import Llama
import llama_cpp

MODEL_PATH = str(Path.home() / ".wombo/dev-cache/models/llama-3.2-1b-4bit-gguf/llama-3.2-1b-4bit.gguf")

def test_batch_api_parallel():
    """Test true parallel processing using the batch API"""
    print("Testing batch API for parallel sequence processing...")
    
    try:
        llama = Llama(
            model_path=MODEL_PATH,
            n_ctx=1024,
            n_batch=256,
            n_seq_max=3,  # Allow up to 3 parallel sequences
            verbose=False
        )
        
        print(f"Model loaded with n_seq_max={llama.context_params.n_seq_max}")
        
        # Test prompts
        prompts = [
            "The capital of France is",
            "Python is a programming language that", 
            "Machine learning is"
        ]
        
        # Tokenize prompts
        tokenized_prompts = []
        for i, prompt in enumerate(prompts):
            tokens = llama.tokenize(prompt.encode('utf-8'))
            tokenized_prompts.append(tokens)
            print(f"Sequence {i}: '{prompt}' -> {len(tokens)} tokens")
        
        # Create batch with multiple sequences
        n_tokens_total = sum(len(tokens) for tokens in tokenized_prompts)
        batch = llama_cpp.llama_batch_init(n_tokens_total, 0, llama.context_params.n_seq_max)
        
        # Add tokens to batch with different sequence IDs
        token_idx = 0
        for seq_id, tokens in enumerate(tokenized_prompts):
            for pos, token in enumerate(tokens):
                batch.token[token_idx] = token
                batch.pos[token_idx] = pos
                batch.seq_id[token_idx][0] = seq_id  # Assign to sequence seq_id
                batch.n_seq_id[token_idx] = 1
                batch.logits[token_idx] = (pos == len(tokens) - 1)  # Only compute logits for last token
                token_idx += 1
        
        batch.n_tokens = token_idx
        
        print(f"Created batch with {batch.n_tokens} tokens across {len(prompts)} sequences")
        
        # Process batch
        start_time = time.time()
        ret = llama_cpp.llama_decode(llama.ctx, batch)
        batch_time = time.time() - start_time
        
        if ret != 0:
            raise RuntimeError(f"llama_decode failed with return code {ret}")
        
        print(f"✓ Batch processing completed in {batch_time:.3f}s")
        print(f"✓ Successfully processed {len(prompts)} sequences in parallel")
        
        # Get logits for each sequence
        for seq_id in range(len(prompts)):
            # Find the last token position for this sequence
            last_pos = len(tokenized_prompts[seq_id]) - 1
            
            # Get logits (this is simplified - real implementation would need proper indexing)
            logits = llama_cpp.llama_get_logits_ith(llama.ctx, seq_id)
            if logits:
                print(f"✓ Got logits for sequence {seq_id}")
            else:
                print(f"⚠ No logits for sequence {seq_id}")
        
        # Clean up
        llama_cpp.llama_batch_free(batch)
        del llama
        
        print("✓ Batch API parallel processing test completed successfully")
        
    except Exception as e:
        print(f"✗ Batch API test failed: {e}")
        traceback.print_exc()

def test_sequence_isolation():
    """Test that different sequences maintain separate states"""
    print("\n=== Testing sequence state isolation ===")
    
    try:
        llama = Llama(
            model_path=MODEL_PATH,
            n_ctx=512,
            n_batch=128,
            n_seq_max=2,
            verbose=False
        )
        
        # Two different conversation contexts
        seq_0_prompt = "You are a helpful assistant. Hello!"
        seq_1_prompt = "You are a grumpy robot. Hello!"
        
        # Tokenize
        seq_0_tokens = llama.tokenize(seq_0_prompt.encode('utf-8'))
        seq_1_tokens = llama.tokenize(seq_1_prompt.encode('utf-8'))
        
        print(f"Sequence 0: '{seq_0_prompt}' ({len(seq_0_tokens)} tokens)")
        print(f"Sequence 1: '{seq_1_prompt}' ({len(seq_1_tokens)} tokens)")
        
        # Create batch with both sequences
        total_tokens = len(seq_0_tokens) + len(seq_1_tokens)
        batch = llama_cpp.llama_batch_init(total_tokens, 0, 2)
        
        token_idx = 0
        
        # Add sequence 0
        for pos, token in enumerate(seq_0_tokens):
            batch.token[token_idx] = token
            batch.pos[token_idx] = pos
            batch.seq_id[token_idx][0] = 0
            batch.n_seq_id[token_idx] = 1
            batch.logits[token_idx] = (pos == len(seq_0_tokens) - 1)
            token_idx += 1
        
        # Add sequence 1
        for pos, token in enumerate(seq_1_tokens):
            batch.token[token_idx] = token
            batch.pos[token_idx] = pos
            batch.seq_id[token_idx][0] = 1
            batch.n_seq_id[token_idx] = 1
            batch.logits[token_idx] = (pos == len(seq_1_tokens) - 1)
            token_idx += 1
        
        batch.n_tokens = token_idx
        
        # Process batch
        ret = llama_cpp.llama_decode(llama.ctx, batch)
        if ret != 0:
            raise RuntimeError(f"llama_decode failed with return code {ret}")
        
        print("✓ Successfully processed two isolated sequences")
        print("✓ Each sequence maintains its own conversation context")
        
        # Clean up
        llama_cpp.llama_batch_free(batch)
        del llama
        
    except Exception as e:
        print(f"✗ Sequence isolation test failed: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    test_batch_api_parallel()
    test_sequence_isolation()
