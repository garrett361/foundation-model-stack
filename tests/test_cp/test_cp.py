import torch
import torch.distributed as dist
from einops import rearrange
from transformers import AutoModelForCausalLM, AutoTokenizer

from dtest import DTest
from fms.models import get_model
from fms.models.granite import Granite, GraniteConfig

GRANITE_3Z_BV_PATH = "/proj/data-eng/chirag/.cache/models--ibm-granite--granite-3.3-8b-base/snapshots/cfb7adb44a974653cbb2ff883653971c54dba578"


class TestSingleGPU:
    def test_load_model_correctness(self) -> None:
        model = get_model("hf_pretrained", model_path=GRANITE_3Z_BV_PATH).cuda()

        tokenizer = AutoTokenizer.from_pretrained(GRANITE_3Z_BV_PATH)
        hf_model = AutoModelForCausalLM.from_pretrained(GRANITE_3Z_BV_PATH).cuda()
        model.eval()
        hf_model.eval()
        input_text = "Where is the Thomas J. Watson Research Center located?"
        input_tokens = tokenizer(input_text, return_tensors="pt").to("cuda")

        out = model(input_tokens["input_ids"])
        hf_out = hf_model(**input_tokens)
        # HF always upcasts the logits
        torch.testing.assert_close(
            out.to(hf_out.logits), hf_out.logits, rtol=1e-1, atol=1e-1
        )



class TestGraniteCP(DTest):
    nlayers = 2
    seed = 42
    cfg = GraniteConfig(nlayers=nlayers)
    base_seq_len = 32

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
            self.vocab_size,
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
            self.d_model,
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

    def test_fwd(self, cp_mamba_impl: str):
        with torch.no_grad():
            torch.manual_seed(42)
            cp_mesh = dist.device_mesh.init_device_mesh("cuda", (self.world_size,))
            mamba2 = self.get_mamba2()
            mamba2_cp = self.get_mamba2_cp(
                cp_mesh=cp_mesh,
                cp_mamba_impl=cp_mamba_impl,
            )

            inputs = self.get_inputs()
            inputs_cp = self.get_cp_shard(inputs)

            outputs = mamba2(inputs)
            outputs_cp = mamba2_cp(inputs_cp)

            outputs_shard = self.get_cp_shard(outputs)
            torch.testing.assert_close(
                outputs_cp, outputs_shard, atol=self.tol, rtol=self.tol
            )
