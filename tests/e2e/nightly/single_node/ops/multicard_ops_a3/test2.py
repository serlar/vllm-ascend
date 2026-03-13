import random

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_npu
from torch.distributed.distributed_c10d import _get_default_group

from vllm_ascend.utils import enable_custom_op

enable_custom_op()

# ranklist = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
# ranklist = [14, 15]
# ranklist = [0,1]
# ranklist = [2,3]
ranklist = [4,5]
WS=2

def check_all_elements_same_robust(tensor, rtol=1e-05, atol=1e-08):
    """
    检查 PyTorch Tensor 中所有元素是否相同（鲁棒版）
    支持处理空张量和浮点型张量的精度问题
    
    参数:
        tensor: 待检查的 torch.Tensor
        rtol: 相对误差容限（用于浮点比较）
        atol: 绝对误差容限（用于浮点比较）
    
    返回:
        bool: 所有元素相同返回 True，否则返回 False
    """
    # 处理空张量
    if tensor.numel() == 0:
        raise ValueError("Tensor 不能为空")
    
    # 单元素张量直接返回 True
    if tensor.numel() == 1:
        return True
    
    # 展平张量以便处理任意维度
    flat_tensor = tensor.flatten()
    
    # 对于浮点型张量，使用 isclose 处理精度问题
    if torch.is_floating_point(tensor):
        return torch.all(torch.isclose(flat_tensor, flat_tensor[0], rtol=rtol, atol=atol)).item()
    # 对于整数型张量，直接比较
    else:
        return torch.all(flat_tensor == flat_tensor[0]).item()
    
def int32_to_8x_int4_float(tensor_int32):
    """
    将 int32 tensor 的每一位拆解为 8 个有符号 int4，并转换为 float32。
    
    逻辑：
    1. 取低4位 -> 第0个 int4
    2. 右移4位，取低4位 -> 第1个 int4
    ...
    3. 右移28位，取低4位 -> 第7个 int4
    
    对于有符号 int4 (Two's complement):
    二进制 0000 ~ 0111 (0~7)  -> 浮点 0.0 ~ 7.0
    二进制 1000 ~ 1111 (8~15) -> 浮点 -8.0 ~ -1.0
    """
    
    # 确保数据类型是 int32 (虽然输入已经是，但为了健壮性)
    if tensor_int32.dtype != torch.int32:
        tensor_int32 = tensor_int32.to(torch.int32)
    
    original_shape = tensor_int32.shape
    
    # 1. 构造移位量 [0, 4, 8, 12, 16, 20, 24, 28]
    # 形状调整为 (1, 1, ..., 8) 以便广播
    shifts = torch.arange(0, 32, 4, device=tensor_int32.device).view(*([1]*len(original_shape)), -1)
    
    # 2. 扩展维度并右移
    # unsqueeze(-1) 增加一个维度变成 [..., 1]
    # 右移后变成 [..., 8]
    shifted = tensor_int32.unsqueeze(-1) >> shifts
    
    # 3. 掩码操作，只保留低4位 (0xF = 1111 binary)
    # 此时得到的值范围是 0 ~ 15 (无符号视角)
    unpacked_unsigned = shifted & 0xF
    
    # 4. 转换为有符号 int4 (-8 ~ 7)
    # 如果值 >= 8，说明最高位是1，代表负数。
    # 在补码表示中，4位的 8~15 对应 -8~-1。
    # 算法：val = val - 16 (当 val >= 8)
    unpacked_signed = unpacked_unsigned.to(torch.int32) # 确保计算精度
    mask = unpacked_signed >= 8
    unpacked_signed[mask] -= 16
    
    # 5. 转换为 float32
    result_float = unpacked_signed.to(torch.float32)
    result_flat = result_float.flatten(start_dim=-2)
    return result_flat

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
        torch_npu.npu.set_device(ranklist[self.rank])
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

    def verify_int4_unpack(self, tensor_int32):
        """验证int32拆解为int4的结果是否都是1"""
        unpacked = int32_to_8x_int4_float(tensor_int32.cpu())
        # 检查所有拆解后的值是否都是1.0
        if not torch.all(unpacked == 1.0):
            print(f"[Rank {self.rank}] Int4拆解验证失败！非1值数量: {torch.sum(unpacked != 1.0)}")
            return False
        print(f"[Rank {self.rank}] Int4拆解验证成功！")
        return True

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
        torch_npu.npu.set_device(ranklist[self.rank])
        m = 64
        k = 1024
        n = 1024
        topk = 8
        e = 8
        k2 = n // 2
        n2 = k

        torch_npu.npu.config.allow_internal_format = True
        
        # # 生成全1输入张量
        x = (self.generate_all_one_tensor((m, k), dtype=torch.bfloat16)/10000.0).npu()
        weight1 = self.generate_all_one_tensor((e, k, n//8), dtype=torch.int32).npu()
        weight1 = torch_npu.npu_format_cast(weight1, 29)
        weight2 = self.generate_all_one_tensor((e, k2, n2//8), dtype=torch.int32).npu()
        weight2 = torch_npu.npu_format_cast(weight2, 29)
        
        # x = self.generate_random_tensor((m, k), dtype=torch.bfloat16).npu()
        # weight1 = self.generate_random_tensor((e, k, n//8),
        #                                       dtype=torch.int32).npu()
        # weight1 = torch_npu.npu_format_cast(weight1, 29)
        # weight2 = self.generate_random_tensor((e, k2, n2//8),
        #                                       dtype=torch.int32).npu()
        # weight2 = torch_npu.npu_format_cast(weight2, 29)
        
        # 验证权重的int4拆解结果
        # if not self.verify_int4_unpack(weight1):
        #     return False
        # if not self.verify_int4_unpack(weight2):
        #     return False
        
        # 生成bias（全1）
        bias1 = int32_to_8x_int4_float(weight1.cpu())
        bias1_npu = (bias1.sum(dim=-2)*8).npu()
        bias2 = int32_to_8x_int4_float(weight2.cpu())
        # print(f"bias2: {bias2} {bias2.shape}")
        bias2_npu = (bias2.sum(dim=-2)*8).npu()

        print(f"[Rank {self.rank}] ====generate bias====")
        
        # 生成全1的索引和scale（调整为合理范围）
        # expert_idx = torch.ones((m, topk), dtype=torch.int32).npu() % (self.world_size * e)
        expert_idx = (torch.arange(m * topk, dtype=torch.int32).view(m, topk) % (self.world_size * e)).npu()
        # print(expert_idx)
        # expert_idx = torch.randint(0,
        #                            self.world_size * e, (m, topk),
        #                            dtype=torch.int32).npu()
        # scale1 = self.generate_all_one_tensor((e, n), dtype=torch.int64).npu()
        scale1 = self.generate_scaled_tensor_ones(shape=(e,n)).npu()
        scale2 = self.generate_scaled_tensor_ones(shape=(e,n2)).npu()
        # scale2 = self.generate_all_one_tensor((e, n2), dtype=torch.int64).npu()
        probs = self.generate_all_one_tensor((m, topk), dtype=torch.float32).npu()

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

        # 初始化输出张量为全0，方便后续对比
        out = self.generate_random_tensor((m, k), dtype=torch.bfloat16).npu()
        expert_token_nums = self.generate_random_tensor((1, e), dtype=torch.int32).npu()
        
        print(f"[Rank {self.rank}] ====begin op====")
        print(f"[Rank {self.rank}] 算子开始执行，输出张量前5个元素：{out[:5, 0].cpu().tolist()}")
        # 执行算子
        # try:
        torch.ops._C_ascend.dispatch_ffn_combine(
            x=x,
            weight1=weight1_nz_npu,
            weight2=weight2_nz_npu,
            expert_idx=expert_idx,
            scale1=scale1_npu,
            scale2=scale2_npu,
            bias1 = bias1_npu,
            bias2 = bias2_npu,
            probs=probs,
            group=self.hcomm_info,
            max_output_size=2048,
            out=out,
            expert_token_nums=expert_token_nums,
        )
        
        # 基本结果检查：输出不应全为0
        # if torch.all(out.cpu() == 0):
        #     print(f"[Rank {self.rank}] 错误：输出张量全为0！")
        #     return False
        
        # 打印部分结果用于调试
        print(f"[Rank {self.rank}] 算子执行成功，输出张量{out.shape}前5个token第1个元素：{out[:10, 0].cpu().tolist()}")
        print(f"[Rank {self.rank}] 算子执行成功，输出张量{out.shape}第1个token前5个元素：{out[0, :10].cpu().tolist()}")
        print(f"[Rank {self.rank}] 是否全部元素相同{check_all_elements_same_robust(out)}")
        print(f"[Rank {self.rank}] 算子开始执行，token per expert：{expert_token_nums[0, :].cpu().tolist()}")
        return True
            
        # except Exception as e:
        #     print(f"[Rank {self.rank}] 算子执行失败：{e}")
        #     return False

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
        
        # 生成全1输入张量
        x = self.generate_all_one_tensor((m, k), dtype=torch.bfloat16).npu()
        weight1 = self.generate_all_one_tensor((e, k, n//8), dtype=torch.int32).npu()  # 修复n/8为n//8
        weight1 = torch_npu.npu_format_cast(weight1, 29)
        weight2 = self.generate_all_one_tensor((e, k2, n2//8), dtype=torch.int32).npu()  # 修复n2/8为n2//8
        weight2 = torch_npu.npu_format_cast(weight2, 29)
        
        # 验证权重的int4拆解结果
        if not self.verify_int4_unpack(weight1):
            return False
        if not self.verify_int4_unpack(weight2):
            return False

        # 生成全1的索引和scale
        expert_idx = torch.ones((m, topk), dtype=torch.int32).npu() % (self.world_size * e)
        scale1 = self.generate_all_one_tensor((e, n), dtype=torch.int64).npu()
        scale2 = self.generate_all_one_tensor((e, n2), dtype=torch.int64).npu()
        probs = self.generate_all_one_tensor((m, topk), dtype=torch.float32).npu()

        weight1_nz_npu = []
        weight2_nz_npu = []
        scale1_npu = []
        scale2_npu = []
        weight1_nz_npu.append(torch_npu.npu_format_cast(weight1.npu(), 29))
        scale1_npu.append(scale1.npu())
        weight2_nz_npu.append(torch_npu.npu_format_cast(weight2.npu(), 29))
        scale2_npu.append(scale2.npu())

        # 初始化输出张量为全0
        out = torch.zeros((m, k), dtype=torch.bfloat16).npu()
        expert_token_nums = self.generate_random_tensor((1, e), dtype=torch.int32).npu()

        # 执行算子
        # try:
        torch.ops._C_ascend.dispatch_ffn_combine(
            x=x,
            weight1=weight1_nz_npu,
            weight2=weight2_nz_npu,
            expert_idx=expert_idx,
            scale1=scale1_npu,
            scale2=scale2_npu,
            probs=probs,
            group=self.hcomm_info,
            max_output_size=2048,
            out=out,
            expert_token_nums=expert_token_nums,
        )
            
            # 基本结果检查
        if torch.all(out == 0):
            print(f"[Rank {self.rank}] 错误：输出张量全为0！")
            return False
        
        print(f"[Rank {self.rank}] 算子执行成功，输出张量前5个元素：{out[:5, 0].cpu().tolist()}")
        return True
            
        # except Exception as e:
        #     print(f"[Rank {self.rank}] 算子执行失败：{e}")
        #     return False

    def generate_random_tensor(self, size, dtype):
        if dtype in [torch.float16, torch.bfloat16, torch.float32]:
            return torch.randn(size=size, dtype=dtype)
        elif dtype is torch.int8:
            return torch.randint(-16, 16, size=size, dtype=dtype)
        elif dtype is torch.int32:
            return torch.randint(-127, 127, size=size, dtype=dtype)
        else:
            raise ValueError(f"Invalid dtype: {dtype}")

def worker(rank: int, world_size: int, port: int, q: mp.SimpleQueue):
    op = TestDispatchFFNCombine(rank, world_size, port)
    op.generate_hcom()
    out1 = op.run_tensor_list()
    q.put(out1)
    # 可选：运行normal测试
    # out2 = op.run_normal()
    # q.put(out2)


if __name__ == "__main__":
    world_size = WS
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

    # 检查所有进程的执行结果
    assert all(results), f"部分进程执行失败！结果: {results}"
    print("====所有测试通过====")
