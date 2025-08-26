import torch
import torch.distributed as dist
from einops import rearrange
from transformers import AutoModelForCausalLM, AutoTokenizer

from dtest import DTest
from fms.models import get_model
from fms.models.granite import Granite, GraniteConfig, fms_to_hf_sd

GRANITE_3Z_BV_PATH = "/proj/data-eng/chirag/.cache/models--ibm-granite--granite-3.3-8b-base/snapshots/cfb7adb44a974653cbb2ff883653971c54dba578"


class TestSingleGPU:
    def test_load_model_correctness(self) -> None:
        model = get_model("hf_pretrained", model_path=GRANITE_3Z_BV_PATH).to(
            dtype=torch.float32, device="cuda"
        )
        tokenizer = AutoTokenizer.from_pretrained(GRANITE_3Z_BV_PATH)
        model_hf = AutoModelForCausalLM.from_pretrained(GRANITE_3Z_BV_PATH).to(
            dtype=torch.float32, device="cuda"
        )
        model.eval()
        model_hf.eval()
        input_text = "Where is the Thomas J. Watson Research Center located?"
        input_tokens = tokenizer(input_text, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model(input_tokens["input_ids"])
            out_hf = model_hf(**input_tokens).logits
            # HF always upcasts the logits
            torch.testing.assert_close(out.to(out_hf), out_hf, atol=1e-2, rtol=1e-2)

    def test_fms_to_hf_sd(self) -> None:
        model = get_model("hf_pretrained", model_path=GRANITE_3Z_BV_PATH).to(
            dtype=torch.bfloat16, device="cuda"
        )
        tokenizer = AutoTokenizer.from_pretrained(GRANITE_3Z_BV_PATH)
        model_hf = AutoModelForCausalLM.from_pretrained(GRANITE_3Z_BV_PATH).to(
            dtype=torch.bfloat16, device="cuda"
        )
        model.eval()
        model_hf.eval()
        input_text = "Where is the Thomas J. Watson Research Center located?"
        input_tokens = tokenizer(input_text, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out_hf = model_hf(**input_tokens).logits
            sd_fms = model.state_dict()
            sd_hf_from_fms = fms_to_hf_sd(
                sd_fms,
                n_layers=len(model.base_model.layers),
                n_heads=model_hf.config.num_attention_heads,
                n_kv_heads=model_hf.config.num_key_value_heads,
            )
            sd_hf = model_hf.state_dict()
            for k, v in sd_hf.items():
                torch.testing.assert_close(v, sd_hf_from_fms[k])
            model_hf.load_state_dict(sd_hf)
            out_hf2 = model_hf(**input_tokens).logits
            torch.testing.assert_close(out_hf, out_hf2)


class TestGraniteCP(DTest):
    nlayers = 2
    seed = 42
    cfg = GraniteConfig(nlayers=nlayers, max_expected_seq_len=131072)
    base_seq_len = 2048
    dtype = torch.bfloat16  # Needed for ring_flash_attn

    def setup_method(self, method):
        torch.manual_seed(42)

    @property
    def batch_size(self) -> int:
        """
        batch_size == world_size is a reasonable default for testing FSDP-sharded models:
        easy to distribute world_size batch elements for a local model to batch-size-1 trained FSDP model.
        """
        return self.world_size

    @property
    def seq_len(self) -> int:
        return 4 * self.world_size * self.base_seq_len

    @property
    def factory_kwargs(self):
        return {"dtype": self.dtype, "device": self.device}

    def test_cp_init(self):
        cp_mesh = dist.device_mesh.init_device_mesh("cuda", (self.world_size,))
        model = Granite(self.cfg, cp_mesh=cp_mesh)

    def get_input_toks(
        self,
        seed: int = 42,
    ) -> torch.Tensor:
        torch.manual_seed(seed)
        return torch.randint(
            self.cfg.src_vocab_size,
            size=(
                self.batch_size,
                self.seq_len,
            ),
            device=self.device,
        )

    def get_inputs(
        self,
        requires_grad: bool = False,
        seed: int = 42,
        dtype: torch.dtype | None = None,
        batch_size: int | None = None,
    ) -> torch.Tensor:
        torch.manual_seed(seed)
        if batch_size is None:
            batch_size = self.batch_size
        return torch.randn(
            batch_size,
            self.seq_len,
            self.cfg.emb_dim,
            device=self.device,
            dtype=dtype or self.dtype,
            requires_grad=requires_grad,
        )

    def get_cp_shard(
        self,
        tensor: torch.Tensor,
        n_shards: int | None = None,
        rank: int | None = None,
    ) -> torch.Tensor:
        if n_shards is None:
            n_shards = self.world_size
        if rank is None:
            rank = self.rank

        shard = rearrange(tensor, "b (r l) ... -> b r l ...", r=n_shards)[:, rank]
        return shard

    def test_load_model(self) -> None:
        cp_mesh = dist.device_mesh.init_device_mesh("cuda", (self.world_size,))
        model = get_model(
            "hf_pretrained",
            model_path=GRANITE_3Z_BV_PATH,
            cp_mesh=cp_mesh,
            distributed_strategy="do not distribute",  # Hack
        )
        assert model is not None
        assert model.cp_mesh is not None
        # Verify that FMS isn't sharding anything
        assert not any(isinstance(p, dist.tensor.DTensor) for p in model.parameters())

    def test_fwd(self):
        with torch.no_grad():
            cp_mesh = dist.device_mesh.init_device_mesh("cuda", (self.world_size,))
            model_cp = Granite(self.cfg, cp_mesh=cp_mesh).to(**self.factory_kwargs)
            model_cp.reset_parameters()
            model = Granite(self.cfg).to(**self.factory_kwargs)

            # Set all weights to be the same:
            for p_cp, p in zip(model_cp.parameters(), model.parameters()):
                p.data = p_cp.data

            # Verify models are the same:
            sd_cp = model_cp.state_dict()
            for k, v in model.state_dict().items():
                torch.testing.assert_close(v, sd_cp[k], msg=f"Failed on {k=}")

            input_toks = self.get_input_toks()
            input_toks_cp = self.get_cp_shard(input_toks)

            out_cp = model_cp(input_toks_cp)
            out = model(input_toks)
            out_shard = self.get_cp_shard(out)
            torch.testing.assert_close(out_cp, out_shard, atol=1e-2, rtol=1e-2)
