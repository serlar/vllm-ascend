import random

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_npu
from torch.distributed.distributed_c10d import _get_default_group

from vllm_ascend.utils import enable_custom_op

enable_custom_op()


class TestDispatchFFNCombine:

    def __init__(self, rank, world_size, port):
        self.rank = rank
        self.world_size = world_size
        self.master_ip = "127.0.0.1"
        self.port = port

    def get_hcomm(self, comm_group):
        hcomm_info = None
        if torch.__version__ > "2.0.1":
            hcomm_info = comm_group._get_backend(
                torch.device("npu")).get_hccl_comm_name(self.rank)
        else:
            hcomm_info = comm_group.get_hccl_comm_name(self.rank)
        return hcomm_info

    def setup_ep_tp(
        self,
        rank,
        tp_size,
        ep_size,
        backend_type,
        ep_ranks_list=None,
        tp_ranks_list=None,
    ):
        for i in range(tp_size):
            if ep_ranks_list:
                ep_ranks = ep_ranks_list[i]
            else:
                ep_ranks = [x + ep_size * i for x in range(ep_size)]
            ep_group = dist.new_group(backend=backend_type, ranks=ep_ranks)
            if rank in ep_ranks:
                ep_group_tmp = ep_group
        for i in range(ep_size):
            if tp_ranks_list:
                tp_ranks = tp_ranks_list[i]
            else:
                tp_ranks = [x * ep_size + i for x in range(tp_size)]
            tp_group = dist.new_group(backend=backend_type, ranks=tp_ranks)
            if rank in tp_ranks:
                tp_group_tmp = tp_group
        return ep_group_tmp, tp_group_tmp

    def generate_hcom(self):
        torch_npu.npu.set_device(self.rank)
        dist.init_process_group(
            backend="hccl",
            rank=self.rank,
            world_size=self.world_size,
            init_method=f"tcp://127.0.0.1:{self.port}",
        )

        ep_size = 0
        tp_size = self.world_size
        hcomm_info_dist = {
            "default_pg_info": None,
            "ep_hcomm_info": None,
            "group_ep": None,
            "tp_hcomm_info": None,
            "group_tp": None,
        }
        if ep_size and tp_size:
            group_ep, group_tp = self.setup_ep_tp(self.rank, tp_size, ep_size,
                                                  "hccl", None, None)
            hcomm_info_dist["ep_hcomm_info"] = self.get_hcomm(group_ep)
            hcomm_info_dist["tp_hcomm_info"] = self.get_hcomm(group_tp)
            hcomm_info_dist["group_ep"] = group_ep
            hcomm_info_dist["group_tp"] = group_tp
        else:
            if dist.is_available():
                default_pg = _get_default_group()
            hcomm_info_dist["default_pg_info"] = self.get_hcomm(default_pg)
        hcomm_info = hcomm_info_dist["default_pg_info"]
        self.hcomm_info = hcomm_info
        
    def generate_all_one_tensor(self, size, dtype):
        if dtype in [torch.float16, torch.bfloat16, torch.float32]:
            return torch.ones(size=size, dtype=dtype)
        elif dtype == torch.int8:
            return torch.ones(size=size, dtype=dtype)
        elif dtype == torch.int32:
            # 每个int32存储8个int4的1，二进制：00010001000100010001000100010001
            # 十进制值：18911233
            int4_1_packed = 286331153  # 8个int4的1打包成的int32值
            tensor = torch.ones(size=size, dtype=dtype) * int4_1_packed
            return tensor
        elif dtype == torch.int64:
            return torch.ones(size=size, dtype=dtype)
        else:
            raise ValueError(f"Invalid dtype: {dtype}")
    def generate_scaled_tensor_ones(self, shape, device='npu'):
        # shape 是最终需要的逻辑形状，例如 (e, n)
        # 但参考代码中，dequant_scale 的形状可能比 origin 大，或者是为了容纳交错数据
        
        # 1. 生成原始的 float32 1.0
        origin = torch.ones(size=shape, dtype=torch.float32)
        
        # 2. 模拟参考代码中的位操作 (>> 13) << 13
        # 这一步是为了清除 float32 尾数的低13位（模拟某些量化精度截断）
        # 如果是纯粹的 1.0，且不需要截断，这一步可以省略，但为了严谨保留：
        origin_int_view = origin.view(torch.int32)
        modified_int_view = (origin_int_view >> 13) << 13
        modified_float = modified_int_view.view(torch.float32)
        
        # 3. 创建目标 int64 张量
        # 注意：参考代码中 target 形状可能与 origin 不同，或者相同但用于存储交错数据。
        # 假设我们需要生成的 scale1 形状就是 (e, n)，且每个元素对应一个处理后的 float 位模式。
        # 如果参考代码是交错存储 (::2)，那么目标张量的元素数量通常是 origin 的 2 倍（如果是一维展开看）。
        # 但如果你的需求仅仅是 "int64 中 32 位保存 float32 的 1"，且形状保持 (e, n) 不变：
        
        target = torch.zeros(size=shape, dtype=torch.int64, device='cpu') # 先在 CPU 操作
        target.view(torch.float32)[:, ::2] = modified_float
        
        return target.view(torch.int64)
    
    def run_tensor_list(self) -> bool:
        torch_npu.npu.set_device(self.rank)
        m = 64
        k = 1024
        n = 1024
        topk = 8
        e = 8
        k2 = n // 2
        n2 = k

        torch_npu.npu.config.allow_internal_format = True
        # x = self.generate_random_tensor((m, k), dtype=torch.bfloat16).npu()
        # weight1 = self.generate_random_tensor((e, k, n),
        #                                       dtype=torch.int8).npu()
        # weight1 = torch_npu.npu_format_cast(weight1, 29)
        # weight2 = self.generate_random_tensor((e, k2, n2),
        #                                       dtype=torch.int8).npu()
        # weight2 = torch_npu.npu_format_cast(weight2, 29)

        # expert_idx = torch.randint(0,
        #                            self.world_size * e, (m, topk),
        #                            dtype=torch.int32).npu()
        # scale1 = torch.randint(0, 1, (e, n), dtype=torch.int64).npu()
        # scale2 = torch.randint(0, 1, (e, n2), dtype=torch.int64).npu()
        x = (self.generate_all_one_tensor((m, k), dtype=torch.bfloat16)/10000.0).npu()
        weight1 = self.generate_all_one_tensor((e, k, n), dtype=torch.int8).npu()
        weight1 = torch_npu.npu_format_cast(weight1, 29)
        weight2 = self.generate_all_one_tensor((e, k2, n2), dtype=torch.int8).npu()
        weight2 = torch_npu.npu_format_cast(weight2, 29)
        expert_idx = (torch.arange(m * topk, dtype=torch.int32).view(m, topk) % (self.world_size * e)).npu()
        scale1 = self.generate_scaled_tensor_ones(shape=(e,n)).npu()
        scale2 = self.generate_scaled_tensor_ones(shape=(e,n2)).npu()
        probs = self.generate_all_one_tensor((m, topk), dtype=torch.float32).npu()
        # probs = torch.randn(size=(m, topk), dtype=torch.float32).npu()

        weight1_nz_npu = []
        weight2_nz_npu = []
        scale1_npu = []
        scale2_npu = []
        for i in range(e):
            weight1_nz_npu.append(
                torch_npu.npu_format_cast(weight1[i].npu(), 29))
            scale1_npu.append(scale1[i].npu())
            weight2_nz_npu.append(
                torch_npu.npu_format_cast(weight2[i].npu(), 29))
            scale2_npu.append(scale2[i].npu())

        out = self.generate_random_tensor((m, k), dtype=torch.bfloat16).npu()
        expert_token_nums = self.generate_random_tensor((1, e), dtype=torch.int32).npu()

        print(f"[Rank {self.rank}] 算子，输出张量前5个元素：{out[:5, 0].cpu().tolist()}")
        torch.ops._C_ascend.dispatch_ffn_combine(
            x=x,
            weight1=weight1_nz_npu,
            weight2=weight2_nz_npu,
            expert_idx=expert_idx,
            bias1=torch.tensor([]),
            bias2=torch.tensor([]),
            scale1=scale1_npu,
            scale2=scale2_npu,
            probs=probs,
            group=self.hcomm_info,
            max_output_size=2048,
            out=out,
            expert_token_nums=expert_token_nums,
        )
        print(f"[Rank {self.rank}] 算子执行成功，输出张量前5个元素：{out[0, :5].cpu().tolist()}")
        print(f"[Rank {self.rank}] 算子开始执行，token per expert：{expert_token_nums[0, :].cpu().tolist()}")
        return True

    def run_normal(self) -> bool:
        torch_npu.npu.set_device(self.rank)
        m = 64
        k = 1024
        n = 1024
        topk = 8
        e = 8
        k2 = n // 2
        n2 = k

        torch_npu.npu.config.allow_internal_format = True
        x = self.generate_random_tensor((m, k), dtype=torch.bfloat16).npu()
        weight1 = self.generate_random_tensor((e, k, n),
                                              dtype=torch.int8).npu()
        weight1 = torch_npu.npu_format_cast(weight1, 29)
        weight2 = self.generate_random_tensor((e, k2, n2),
                                              dtype=torch.int8).npu()
        weight2 = torch_npu.npu_format_cast(weight2, 29)

        expert_idx = torch.randint(0,
                                   self.world_size * e, (m, topk),
                                   dtype=torch.int32).npu()
        scale1 = torch.randint(0, 1, (e, n), dtype=torch.int64).npu()
        scale2 = torch.randint(0, 1, (e, n2), dtype=torch.int64).npu()
        probs = torch.randn(size=(m, topk), dtype=torch.float32).npu()

        weight1_nz_npu = []
        weight2_nz_npu = []
        scale1_npu = []
        scale2_npu = []
        weight1_nz_npu.append(torch_npu.npu_format_cast(weight1.npu(), 29))
        scale1_npu.append(scale1.npu())
        weight2_nz_npu.append(torch_npu.npu_format_cast(weight2.npu(), 29))
        scale2_npu.append(scale2.npu())

        out = self.generate_random_tensor((m, k), dtype=torch.bfloat16).npu()
        expert_token_nums = self.generate_random_tensor((1, e), dtype=torch.int32).npu()

        torch.ops._C_ascend.dispatch_ffn_combine(
            x=x,
            weight1=weight1_nz_npu,
            weight2=weight2_nz_npu,
            expert_idx=expert_idx,
            bias1=torch.tensor([]),
            bias2=torch.tensor([]),
            scale1=scale1_npu,
            scale2=scale2_npu,
            probs=probs,
            group=self.hcomm_info,
            max_output_size=512,
            out=out,
            expert_token_nums=expert_token_nums,
        )
        return True

    def generate_random_tensor(self, size, dtype):
        if dtype in [torch.float16, torch.bfloat16, torch.float32]:
            return torch.randn(size=size, dtype=dtype)
        elif dtype is torch.int8:
            return torch.randint(-16, 16, size=size, dtype=dtype)
        elif dtype is torch.int32:
            return torch.randint(-1024, 1024, size=size, dtype=dtype)
        else:
            raise ValueError(f"Invalid dtype: {dtype}")


def worker(rank: int, world_size: int, port: int, q: mp.SimpleQueue):
    op = TestDispatchFFNCombine(rank, world_size, port)
    op.generate_hcom()
    out1 = op.run_tensor_list()
    q.put(out1)
    out2 = op.run_normal()
    q.put(out2)


# @torch.inference_mode()
# def test_dispatch_ffn_combine_kernel():
world_size = 2
mp.set_start_method("fork", force=True)

q = mp.SimpleQueue()
p_list = []
port = 29501 + random.randint(0, 10000)

for rank in range(world_size):
    p = mp.Process(target=worker, args=(rank, world_size, port, q))
    p.start()
    p_list.append(p)

results = [q.get() for _ in range(world_size)]

for p in p_list:
    p.join()

assert all(results) , f"部分进程执行失败！结果: {results}"
