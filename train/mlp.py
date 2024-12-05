import pytorch_kinematics as pk
import torch
import torch.nn as nn
import torch.optim as optim
import math
import random
import json
# Step 1: 加载 URDF 和创建链
urdf_path = "/home/lightcone/workspace/SOFA/mlp/nfh1/urdf/nfh1.urdf"
with open(urdf_path, "rb") as f:
    urdf_content = f.read()

chain = pk.build_chain_from_urdf(urdf_content)
print("机器人链结构:")
chain.print_tree()
print("关节参数名:", chain.get_joint_parameter_names())

# Step 2: 定义前向动力学函数
def forward_kinematics(chain, joint_angles, keypoint_names):
    """
    使用 pytorch_kinematics 计算给定关节角度下关键点的位置
    Args:
        chain: pytorch_kinematics 的机器人链对象
        joint_angles: 关节角度张量，形状 [N, njoints]
        keypoint_names: 关键点名称列表
    Returns:
        keypoints: 关键点位置张量，形状 [N, len(keypoint_names), 3]
    """
    ret = chain.forward_kinematics(joint_angles.cpu())
    keypoints = torch.stack([ret[name].get_matrix()[:, :3, 3] for name in keypoint_names], dim=1).to("cuda:0")
    return keypoints

# Step 3: 定义 MLP 模型
import torch
import torch.nn as nn
import numpy as np




class FingerMLP(nn.Module):
    def __init__(self,  input_dim=16, output_dim=16,start=0):
        super(FingerMLP, self).__init__()
        upper=[0.05]*9+[0.3]*7+[0.05]*9
        self.lower_joint_limits = torch.tensor([0]*25).to("cuda:0")
        self.upper_joint_limits = torch.tensor(upper).to("cuda:0")

        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.tanh1 = nn.Tanh()

        self.fc2 = nn.Linear(256, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.tanh2 = nn.Tanh()

        self.fc3 = nn.Linear(256, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.tanh3 = nn.Tanh()

        self.fc4 = nn.Linear(256, 256)
        self.bn4 = nn.BatchNorm1d(256)
        self.tanh4 = nn.Tanh()

        self.fc5 = nn.Linear(256, 128)
        self.bn5 = nn.BatchNorm1d(128)
        self.tanh5 = nn.Tanh()

        self.fc6 = nn.Linear(128, output_dim)
        self.tanh6 = nn.Tanh()

        # Xavier Initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):


        x = self.tanh1(self.bn1(self.fc1(x)))
        residual = x
        x = self.tanh2(self.bn2(self.fc2(x)))
        x = self.tanh3(self.bn3(self.fc3(x)))
        x = x + residual  # Residual connection

        x = self.tanh4(self.bn4(self.fc4(x)))
        x = self.tanh5(self.bn5(self.fc5(x)))
        x = self.tanh6(self.fc6(x))

        x = 0.5 * (x + 1.0)  # Scale tanh output from [-1, 1] to [0, 1]
        x = x * (self.upper_joint_limits - self.lower_joint_limits) + self.lower_joint_limits
        return x


# Step 4: 准备训练数据
N = 100000  # 样本数
M = 1    # 绳的个数
joint_dim = len(chain.get_joint_parameter_names())  # 机器人关节自由度

# 输入：随机生成绳长数据 s
# Step 2: 从 JSON 文件加载数据
def load_data_from_json(json_file_path):
    with open(json_file_path, "r") as file:
        data = json.load(file)
    
    displacement_position_map = data["displacement_position_map"]
    inputs = []
    outputs = []

    for entry in displacement_position_map:
        displacement = entry["displacement"]
        positions = entry["positions"]
        inputs.append(displacement)
        outputs.append(positions)

    return inputs, outputs

# Step 3: 插值补充数据
from scipy.interpolate import CubicSpline
import numpy as np

def interpolate_data_and_save(inputs, outputs, num_samples=500, output_file="interpolated_data.json"):
    """
    通过三次插值补充数据并保存到 JSON 文件。
    Args:
        inputs: 原始输入列表（位移）。
        outputs: 原始输出列表，形状为 (N, 6, 3)。
        num_samples: 两点之间插值的样本数。
        output_file: 保存插值数据的文件名。
    Returns:
        新增插值后的 inputs 和 outputs。
    """
    # 转换为 numpy 数组
    inputs = np.array(inputs)
    outputs = np.array(outputs)

    # 确保 outputs 是三维数组
    if outputs.ndim != 3:
        raise ValueError("outputs 应该是三维数组，形状为 (N, 6, 3)。")

    # 创建新的输入点
    dense_inputs = []
    for i in range(len(inputs) - 1):
        start_input, end_input = inputs[i], inputs[i + 1]
        interp_inputs = np.linspace(start_input, end_input, num_samples + 2).tolist()
        # 移除首尾以避免重复
        dense_inputs.extend(interp_inputs[1:-1])

    # 转换为 numpy 数组
    dense_inputs = np.array(dense_inputs)

    # 初始化 dense_outputs
    num_frames, num_keypoints, num_dims = outputs.shape
    dense_outputs = np.zeros((len(dense_inputs), num_keypoints, num_dims))

    # 对每个关键点的每个维度进行插值
    for kp in range(num_keypoints):
        for dim in range(num_dims):
            # 获取当前关键点和维度的数据
            dim_outputs = outputs[:, kp, dim]

            # 构建三次插值
            spline = CubicSpline(inputs, dim_outputs)

            # 计算插值
            dense_outputs[:, kp, dim] = spline(dense_inputs)

    # 转换为列表形式
    dense_inputs = dense_inputs.tolist()
    dense_outputs = dense_outputs.tolist()

    # 保存到 JSON 文件
    data_to_save = {
        "inputs": dense_inputs,
        "outputs": dense_outputs
    }
    with open(output_file, "w") as json_file:
        json.dump(data_to_save, json_file, indent=4)

    return dense_inputs, dense_outputs

# 加载 JSON 数据
json_path = "/home/lightcone/workspace/SOFA/mlp/finger_model2/displacement_position1.json"  # 替换为你的 JSON 文件路径
inputs, outputs = load_data_from_json(json_path)

# 插值补充数据
interp_inputs, interp_outputs = interpolate_data_and_save(inputs, outputs, num_samples=0)

# 将原始数据与插值数据合并
inputs.extend(interp_inputs)
outputs.extend(interp_outputs)
keypoint_names = [(f"link{i+1}") for i in range(25)] 
# 转换为 PyTorch 张量
s = torch.tensor([[x] for x in inputs], dtype=torch.float32)  # (N, 1)
outputs = torch.tensor(outputs, dtype=torch.float32)  # (N, M)
outputs[:,:,2]+=9
keypoints_target = outputs/1000
print(s,keypoints_target)
# Step 5: 定义损失函数
def mse_loss(model, chain, s, keypoints_target, keypoint_names):
    """
    计算 MSE 损失
    """
    # 模型预测关节角度
    
    q_pred = model(s.to("cuda:0"))  # [N, joint_dim]
    # 使用 pytorch_kinematics 计算预测的关键点位置
    keypoints_pred = forward_kinematics(chain, q_pred, keypoint_names)  # [N, len(keypoint_names), 3]
    # print(keypoints_pred.shape,keypoints_target.shape)
    # # 计算 MSE 损失
    return torch.mean((keypoints_pred.to("cuda:0")[:,:,-2:] - keypoints_target.to("cuda:0")[:,:,-2:]) ** 2)*10000

# Step 6: 初始化模型和优化器
input_dim = M
output_dim = joint_dim
hidden_dim = 64
epochs = 10000
learning_rate = 0.001
device="cuda:0"
model = FingerMLP(input_dim, output_dim, hidden_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# Step 7: 初始化训练配置
nepoch = 200  # 训练的总轮数
batch_size = 1024  # 批量大小

# 创建数据加载器

dataset = torch.utils.data.TensorDataset(s, keypoints_target)  # 使用输入和目标创建数据集

train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 定义开始的特征维度
start = 0  # 如果有需要忽略的起始维度，则调整此值
loss_function = nn.MSELoss()  # 使用 MSELoss 作为损失函数

for epoch in range(nepoch):
    for i, (data, keypoints_target_batch) in enumerate(train_loader, 0):  # 遍历数据加载器
        
        # 将数据和目标转移到设备
        data = data.to(device)
        keypoints_target_batch = keypoints_target_batch.to(device)
        
        # 预测关节角度，输入从 start 开始的部分数据
        recon_data_partial = model(data[:, start:])
        
        # 计算生成器损失1：重构损失
        mlp_loss1 = mse_loss(model, chain, s, keypoints_target, keypoint_names)
        
        # 如果 start 不为 0，在前面补零
        if start != 0:
            # 创建一个与批次大小匹配、列数为 start 的全零张量
            zeros = torch.zeros(recon_data_partial.size(0), start).to(device)
            # 在特征维度拼接零张量与预测数据
            recon_data = torch.cat((zeros, recon_data_partial), dim=1)
        else:
            # 如果 start 为 0，直接使用 recon_data_partial
            recon_data = recon_data_partial
        
        mlp_loss = mlp_loss1 
        # 更新生成器 MLP
        optimizer.zero_grad()  # 清空梯度
        mlp_loss.backward()  # 反向传播计算梯度
        optimizer.step()  # 更新模型参数

        # 打印损失信息
        if i % 10 == 0:  # 每 10 次迭代打印一次
            print(
                f"Epoch [{epoch+1}/{nepoch}], Step [{i+1}/{len(train_loader)}], "
                f"Reconstruction Loss: {mlp_loss1.item():.4f}"
            )
    model_path = f"./models/mlp_model_{epoch}.pth"
    torch.save(model.state_dict(), model_path)
model.eval()  # 切换到评估模式
# Step 8: 测试并可视化
s_test = torch.tensor([[0.5]], dtype=torch.float32)  # 测试绳长
q_test = model(s_test.to(device))  # 获取预测的关节角度

keypoints_pred = forward_kinematics(chain, q_test, keypoint_names)  # 预测关键点位置


print("预测关节角度:", q_test)
print("预测关键点位置:", keypoints_pred)
# 保存模型权重
model_path = "./mlp_model.pth"
torch.save(model.state_dict(), model_path)
print(f"模型已保存到 {model_path}")
