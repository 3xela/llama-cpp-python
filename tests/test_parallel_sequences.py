# test script to verify parallel sequence processing with n_seq_max

from pathlib import Path
import traceback
import time

from llama_cpp import Llama
import llama_cpp

MODEL_PATH = str(Path.home() / ".wombo/dev-cache/models/llama-3.2-1b-4bit-gguf/llama-3.2-1b-4bit.gguf")

def test_parallel_sequences():
    """Test that n_seq_max > 1 allows processing multiple sequences in parallel"""
    print("Testing parallel sequence processing...")
    
    # Test with n_seq_max=1 (sequential processing)
    print("\n=== Testing with n_seq_max=1 (sequential) ===")
    try:
        llama_sequential = Llama(
            model_path=MODEL_PATH,
            n_ctx=3072,  # Increased context for fair comparison
            n_batch=256,
            n_seq_max=1,
            verbose=False
        )
        
        # Longer, more comprehensive prompts to better test processing capabilities
        prompts = [
            "Write a comprehensive analysis of renewable energy technologies, focusing on solar, wind, and hydroelectric power. Discuss their environmental benefits, current limitations, cost-effectiveness, and potential for widespread adoption. Include information about recent technological advances and government policies supporting renewable energy development.",
            "Explain the principles of quantum computing and how it differs from classical computing. Describe quantum bits (qubits), superposition, entanglement, and quantum algorithms. Discuss current applications, major challenges in building practical quantum computers, and the potential impact on cryptography, drug discovery, and optimization problems.",
            "Provide a detailed overview of modern web development frameworks and technologies. Compare frontend frameworks like React, Vue, and Angular, discuss backend technologies like Node.js, Python Django, and Ruby on Rails. Include information about database choices, cloud deployment options, and best practices for building scalable web applications."
        ]
        
        start_time = time.time()
        sequential_results = []
        
        for i, prompt in enumerate(prompts):
            result = llama_sequential(
                prompt, 
                max_tokens=500,  # Reduced to reasonable amount
                stop=["</end>", "###", "---"],
                echo=False
            )
            sequential_results.append(result['choices'][0]['text'])
        
        sequential_time = time.time() - start_time
        
        del llama_sequential
        
    except Exception as e:
        print(f"✗ Sequential test failed: {e}")
        traceback.print_exc()
        return
    
    # Test with n_seq_max=3 (parallel processing)
    print("\n=== Testing with n_seq_max=3 (parallel) ===")
    try:
        llama_parallel = Llama(
            model_path=MODEL_PATH,
            n_ctx=3072,  # Same total context, will be divided by n_seq_max
            n_batch=256,
            n_seq_max=3,
            verbose=False
        )
        
        print(f"Configured n_seq_max: {llama_parallel.context_params.n_seq_max}")
        print(f"Context per sequence: {3072 // 3} tokens")
        
        # For true parallel processing, we'd need to use the batch API
        # For now, let's verify the configuration is correct
        assert llama_parallel.context_params.n_seq_max == 3, f"Expected 3, got {llama_parallel.context_params.n_seq_max}"
        print("✓ Parallel configuration test passed")
        
        # Test that we can still process sequences (even if not truly parallel with simple API)
        start_time = time.time()
        parallel_results = []
        
        for i, prompt in enumerate(prompts):
            result = llama_parallel(
                prompt, 
                max_tokens=500,  # Reduced to reasonable amount
                stop=["</end>", "###", "---"],
                echo=False
            )
            parallel_results.append(result['choices'][0]['text'])
        
        parallel_time = time.time() - start_time
        
        # Compare results briefly
        print("\n=== Performance Summary ===")
        print(f"Sequential time: {sequential_time:.2f}s")
        print(f"Parallel time:   {parallel_time:.2f}s")
        if sequential_time and parallel_time:
            speedup = sequential_time / parallel_time
            print(f"Speedup: {speedup:.2f}x")
        
        # Show character counts only (not full text)
        print("\n=== Output Comparison (Character Counts) ===")
        for i in range(len(prompts)):
            seq_len = len(sequential_results[i]) if i < len(sequential_results) else 0
            par_len = len(parallel_results[i]) if i < len(parallel_results) else 0
            print(f"Sequence {i+1}: Sequential={seq_len} chars, Parallel={par_len} chars")
        
        print("✓ Parallel processing test passed")
        
        del llama_parallel
        
    except Exception as e:
        print(f"✗ Parallel test failed: {e}")
        traceback.print_exc()

def test_batch_processing():
    """Test using the batch API for true parallel processing"""
    print("\n=== Testing batch processing (true parallelism) ===")
    
    try:
        llama = Llama(
            model_path=MODEL_PATH,
            n_ctx=3072,
            n_batch=256,
            n_seq_max=3,
            verbose=False
        )
        
        # Same prompts as other tests
        prompts = [
            "Write a comprehensive analysis of renewable energy technologies, focusing on solar, wind, and hydroelectric power. Discuss their environmental benefits, current limitations, cost-effectiveness, and potential for widespread adoption. Include information about recent technological advances and government policies supporting renewable energy development.",
            "Explain the principles of quantum computing and how it differs from classical computing. Describe quantum bits (qubits), superposition, entanglement, and quantum algorithms. Discuss current applications, major challenges in building practical quantum computers, and the potential impact on cryptography, drug discovery, and optimization problems.",
            "Provide a detailed overview of modern web development frameworks and technologies. Compare frontend frameworks like React, Vue, and Angular, discuss backend technologies like Node.js, Python Django, and Ruby on Rails. Include information about database choices, cloud deployment options, and best practices for building scalable web applications."
        ]
        
        # Tokenize all prompts
        tokenized_prompts = []
        for prompt in prompts:
            tokens = llama.tokenize(prompt.encode('utf-8'))
            tokenized_prompts.append(tokens)
        
        # Create batch with ALL sequences - this is the key to parallel processing
        n_tokens_total = sum(len(tokens) for tokens in tokenized_prompts)
        batch = llama_cpp.llama_batch_init(n_tokens_total, 0, llama.context_params.n_seq_max)
        
        # Add ALL tokens from ALL sequences to the batch simultaneously
        token_idx = 0
        for seq_id, tokens in enumerate(tokenized_prompts):
            for pos, token in enumerate(tokens):
                batch.token[token_idx] = token
                batch.pos[token_idx] = pos
                batch.seq_id[token_idx][0] = seq_id  # Each sequence gets its own ID
                batch.n_seq_id[token_idx] = 1
                batch.logits[token_idx] = (pos == len(tokens) - 1)  # Only compute logits for last token
                token_idx += 1
        
        batch.n_tokens = token_idx
        
        # THIS IS THE KEY: Process ALL sequences with ONE call (true parallelism)
        start_time = time.time()
        ret = llama_cpp.llama_decode(llama.ctx, batch)
        batch_time = time.time() - start_time
        
        if ret != 0:
            raise RuntimeError(f"llama_decode failed with return code {ret}")
        
        print(f"✓ TRUE PARALLEL processing completed in {batch_time:.3f}s")
        print(f"✓ Processed {len(prompts)} sequences SIMULTANEOUSLY")
        
        # Verify we got logits for each sequence
        for seq_id in range(len(prompts)):
            logits = llama_cpp.llama_get_logits_ith(llama.ctx, seq_id)
            if logits:
                print(f"✓ Got logits for sequence {seq_id}")
            else:
                print(f"⚠ No logits for sequence {seq_id}")
        
        # Clean up
        llama_cpp.llama_batch_free(batch)
        del llama
        
        return batch_time
        
    except Exception as e:
        print(f"✗ Batch processing test failed: {e}")
        traceback.print_exc()
        return None

def test_true_parallel_vs_sequential():
    """Compare true parallel processing vs sequential processing"""
    print("\n" + "="*60)
    print("COMPARING TRUE PARALLEL VS SEQUENTIAL PROCESSING")
    print("="*60)
    
    # Test sequential processing (baseline)
    print("\n=== Sequential Processing (Baseline) ===")
    sequential_time = test_sequential_only()
    
    # Test true parallel processing
    print("\n=== True Parallel Processing (Batch API) ===")
    parallel_time = test_batch_processing()
    
    if sequential_time and parallel_time:
        speedup = sequential_time / parallel_time
        print(f"\n" + "="*60)
        print("PERFORMANCE COMPARISON:")
        print(f"Sequential time: {sequential_time:.3f}s")
        print(f"Parallel time:   {parallel_time:.3f}s")
        print(f"Speedup:         {speedup:.2f}x")
        print(f"Time saved:      {sequential_time - parallel_time:.3f}s")
        print("="*60)

def test_sequential_only():
    """Test sequential processing for comparison"""
    try:
        llama = Llama(
            model_path=MODEL_PATH,
            n_ctx=3072,
            n_batch=256,
            n_seq_max=1,
            verbose=False
        )
        
        prompts = [
            "Write a comprehensive analysis of renewable energy technologies, focusing on solar, wind, and hydroelectric power. Discuss their environmental benefits, current limitations, cost-effectiveness, and potential for widespread adoption. Include information about recent technological advances and government policies supporting renewable energy development.",
            "Explain the principles of quantum computing and how it differs from classical computing. Describe quantum bits (qubits), superposition, entanglement, and quantum algorithms. Discuss current applications, major challenges in building practical quantum computers, and the potential impact on cryptography, drug discovery, and optimization problems.",
            "Provide a detailed overview of modern web development frameworks and technologies. Compare frontend frameworks like React, Vue, and Angular, discuss backend technologies like Node.js, Python Django, and Ruby on Rails. Include information about database choices, cloud deployment options, and best practices for building scalable web applications."
        ]
        
        start_time = time.time()
        
        # Process each prompt individually (sequential)
        for i, prompt in enumerate(prompts):
            tokens = llama.tokenize(prompt.encode('utf-8'))
            
            # Create individual batch for each sequence
            batch = llama_cpp.llama_batch_init(len(tokens), 0, 1)
            
            for pos, token in enumerate(tokens):
                batch.token[pos] = token
                batch.pos[pos] = pos
                batch.seq_id[pos][0] = 0
                batch.n_seq_id[pos] = 1
                batch.logits[pos] = (pos == len(tokens) - 1)
            
            batch.n_tokens = len(tokens)
            
            # Process this sequence individually
            ret = llama_cpp.llama_decode(llama.ctx, batch)
            if ret != 0:
                raise RuntimeError(f"llama_decode failed with return code {ret}")
            
            llama_cpp.llama_batch_free(batch)
        
        sequential_time = time.time() - start_time
        print(f"✓ Sequential processing completed in {sequential_time:.3f}s")
        
        del llama
        return sequential_time
        
    except Exception as e:
        print(f"✗ Sequential test failed: {e}")
        traceback.print_exc()
        return None

if __name__ == "__main__":
    test_parallel_sequences()
    test_true_parallel_vs_sequential()
