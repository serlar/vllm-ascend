import torch

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

# ==========================================
# 测试与验证
# ==========================================

# 1. 生成全范围随机 int32
# torch.iinfo(torch.int32).min = -2147483648
# torch.iinfo(torch.int32).max = 2147483647
size = (2, 4) # 示例形状
a = torch.randint(
    low=torch.iinfo(torch.int32).min, 
    high=torch.iinfo(torch.int32).max + 1, # high 是开区间，所以 +1 才能取到最大值
    size=size, 
    dtype=torch.int32
)

print(f"输入 a (int32) 形状: {a.shape}")
print(f"输入 a 样例数据:\n{a}\n")

# 2. 执行转换
b = int32_to_8x_int4_float(a)

print(f"输出 b (float32) 形状: {b.shape}")
print(f"输出 b 样例数据 (每个 int32 拆成了 8 个 float):\n{b}\n")

# 3. 手动验证第一个元素，确保逻辑正确
val = a[0, 0].item()
print(f"--- 验证第一个元素 a[0,0] = {val} ---")
print(f"二进制表示 (32位): {val & 0xFFFFFFFF:032b}")

manual_results = []
temp = val
for i in range(8):
    # 取低4位
    bits = temp & 0xF
    # 转有符号
    if bits >= 8:
        bits -= 16
    manual_results.append(float(bits))
    # 右移4位准备下一次
    # 注意：Python 的 >> 对有符号数是算术右移，但在我们手动模拟时，
    # 我们其实是在处理无符号的位块。
    # 为了安全地模拟“位切片”，我们先将 temp 视为无符号处理位移，或者直接用掩码移位法
    # 这里用简单的逻辑：直接对原始值进行 (val >> (i*4)) & 0xF 更准确
    pass

# 重新用准确的位运算逻辑验证
verified_results = []
for i in range(8):
    shift_amt = i * 4
    # 提取
    raw_4bits = (val >> shift_amt) & 0xF
    # 符号修正
    if raw_4bits >= 8:
        raw_4bits -= 16
    verified_results.append(float(raw_4bits))

print(f"代码计算结果: {b[0].tolist()}")
print(f"手动验证结果: {verified_results}")
print(f"结果一致: {b[0, 0].tolist() == verified_results}")