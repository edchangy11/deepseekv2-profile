from typing import Optional
from configuration_deepseek import DeepseekV2Config
import torch
import torch.utils.benchmark as benchmark
import math
from models import SimpleAttention, SimpleCompressedAttention, SimpleAbsorbedAttention

torch.set_grad_enabled(False)

class BenchmarkFixture:
    config: DeepseekV2Config
    
    def __init__(self, config: DeepseekV2Config, kv_len: int, q_len: int = 1, bsz: int = 1, dev='cuda'):
        self.config = config
        self.bsz = bsz
        self.q_len = q_len
        self.kv_len = kv_len
        self.cfg_dict = config.to_dict()
        self.cfg_dict['torch_dtype'] = config.torch_dtype

    def benchmark(self, min_run_time: float = 1.0):
        return benchmark.Timer(
            stmt='bencher.iter()',
            globals={'bencher': self},
            label=self.name(),
            sub_label=f'kv_len={self.kv_len}',
        ).blocked_autorange(min_run_time=min_run_time)
    
    @classmethod
    def name(cls):
        return cls.__name__.removesuffix('Bencher')

    def cache_size(self):
        return 0


class SimpleAttentionBencher(BenchmarkFixture):
    def __init__(self, config: DeepseekV2Config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        
        # Initialize model parameters
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.q_lora_rank = 1536  # Same as in models.py test
        
        # Initialize the model
        self.attn = SimpleAttention(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            q_lora_rank=self.q_lora_rank,
            dtype=config.torch_dtype
        ).cuda()
        
        # Create input tensors
        self.hidden_state = torch.randn((self.bsz, self.q_lora_rank), 
                                      dtype=config.torch_dtype, device='cuda')
        
        # Create key-value tensor (combined)
        self.key_value = torch.randn((self.bsz, self.kv_len, self.num_heads, 2 * self.head_dim),
                                   dtype=config.torch_dtype, device='cuda')
    
    def iter(self):
        return self.attn(self.hidden_state, self.key_value)
    
    def cache_size(self):
        return self.key_value.numel() * self.key_value.element_size()


class SimpleCompressedAttentionBencher(BenchmarkFixture):
    def __init__(self, config: DeepseekV2Config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        
        # Initialize model parameters
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.lora_rank = 512  # Same as in models.py test
        self.q_lora_rank = 1536  # Same as in models.py test
        
        # Initialize the model
        self.attn = SimpleCompressedAttention(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            lora_rank=self.lora_rank,
            q_lora_rank=self.q_lora_rank,
            dtype=config.torch_dtype
        ).cuda()
        
        # Create input tensors
        self.hidden_state = torch.randn((self.bsz, self.q_lora_rank), 
                                      dtype=config.torch_dtype, device='cuda')
        
        # Create compressed KV states
        self.compressed_kv = torch.randn((self.bsz, self.kv_len, self.lora_rank),
                                       dtype=config.torch_dtype, device='cuda')
    
    def iter(self):
        return self.attn(self.hidden_state, self.compressed_kv)
    
    def cache_size(self):
        return self.compressed_kv.numel() * self.compressed_kv.element_size()


class SimpleAbsorbedAttentionBencher(BenchmarkFixture):
    def __init__(self, config: DeepseekV2Config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        
        # Initialize model parameters
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.lora_rank = 512  # Same as in models.py test
        self.q_lora_rank = 1536  # Same as in models.py test
        
        # Initialize the model
        self.attn = SimpleAbsorbedAttention(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            lora_rank=self.lora_rank,
            q_lora_rank=self.q_lora_rank,
            dtype=config.torch_dtype
        ).cuda()
        
        # Create input tensors
        self.hidden_state = torch.randn((self.bsz, self.q_lora_rank), 
                                      dtype=config.torch_dtype, device='cuda')
        
        # Create compressed KV states
        self.compressed_kv = torch.randn((self.bsz, self.kv_len, self.lora_rank),
                                       dtype=config.torch_dtype, device='cuda')
    
    def iter(self):
        return self.attn(self.hidden_state, self.compressed_kv)
    
    def cache_size(self):
        return self.compressed_kv.numel() * self.compressed_kv.element_size()


ALL_BENCHMARKS = [
    SimpleAttentionBencher,
    SimpleCompressedAttentionBencher,
    SimpleAbsorbedAttentionBencher,
]

BENCHERS = {}

doc = 'Run benchmark on various MLA implementations\n\n'

for bencher in ALL_BENCHMARKS:
    name = bencher.name()
    BENCHERS[name] = bencher
    doc += f'{name}\n'

def main(bench: str, kv_len: int, bsz: int = 1, config: str = 'mla/config.json', repeat: Optional[int] = None, 
         min_run_time: float = 1.0, csv: bool = False):
    print(f"Available benchers: {list(BENCHERS.keys())}")
    print(f"Requested bench: {bench}")
    
    cfg = DeepseekV2Config.from_json_file(config)
    bencher: BenchmarkFixture
    
    try:
        bencher = BENCHERS[bench](cfg, kv_len, bsz=bsz)
    except KeyError:
        print(f"Error: {bench} not found in available benchers")
        return
    
    if repeat is not None:
        for _ in range(repeat):
            bencher.iter()
        torch.cuda.synchronize()
        return
    
    result = bencher.benchmark(min_run_time=min_run_time)
    cache_size = bencher.cache_size()
    device_name = torch.cuda.get_device_name()
    
    # Print results in key: value format
    print(f"Model: {bencher.name()}")
    print(f"Batch_Size: {bsz}")
    print(f"KV_Length: {kv_len}")
    print(f"Device: {device_name}")
    print(f"Cache_Size: {cache_size}")
    print(f"Mean: {result.mean}")
    print(f"Median: {result.median}")
    print(f"P25: {result._p25}")
    print(f"P75: {result._p75}")

main.__doc__ = doc

if __name__ == "__main__":
    import fire
    fire.Fire(main)
    