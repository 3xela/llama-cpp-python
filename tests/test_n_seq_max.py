# test script to verify n_seq_max parameter supported in llama cpp python


from pathlib import Path
import traceback

from llama_cpp import Llama

MODEL_PATH = str(Path.home() / ".wombo/dev-cache/models/llama-3.2-1b-4bit-gguf/llama-3.2-1b-4bit.gguf")

def test_n_seq_max():
    print("Testing n_seq_max parameter...")
    
    # Test with default n_seq_max (should be 1)
    try:
        llama = Llama(
            model_path=MODEL_PATH,
            n_ctx=2048,
            n_batch=512,
            verbose=True
        )
        print(f"Default n_seq_max: {llama.context_params.n_seq_max}")
        assert llama.context_params.n_seq_max == 1
        print("✓ Default n_seq_max test passed")
        del llama
    except Exception as e:
        print(f"✗ Default n_seq_max test failed: {e}")
        traceback.print_exc()
    
    # Test with custom n_seq_max
    try:
        llama = Llama(
            model_path=MODEL_PATH,
            n_ctx=2048,
            n_batch=512,
            n_seq_max=2,
            verbose=True
        )
        print(f"Custom n_seq_max: {llama.context_params.n_seq_max}")
        assert llama.context_params.n_seq_max == 2, f"Expected 2, got {llama.context_params.n_seq_max}"
        print("✓ Custom n_seq_max test passed")
        
        # Verify that context per sequence is calculated correctly
        n_ctx_per_seq = llama.n_ctx() // llama.context_params.n_seq_max
        print(f"Context per sequence: {n_ctx_per_seq} (total: {llama.n_ctx()}, n_seq_max: {llama.context_params.n_seq_max})")
        
        del llama
    except Exception as e:
        print(f"✗ Custom n_seq_max test failed: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    test_n_seq_max()