import numpy as np
import pandas as pd
from pyqpanda import create_empty_circuit, RY, RX, RZ, CPUQVM, QProg, measure_all
import pyvqnet.tensor as vqt
from pyvqnet.tensor import QTensor
from pyvqnet.nn.module import Module
from pyvqnet.nn.loss import CategoricalCrossEntropy
from pyvqnet.optim.adam import Adam
from pyvqnet.qnn.measure import expval
from pyvqnet.tensor import tensor
import pyqpanda as pq
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from pyvqnet.nn.loss import SoftmaxCrossEntropy
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# 设置中文字体
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题
except:
    print("无法设置中文字体，将使用默认字体")


# 加权交叉熵损失函数
def weighted_cross_entropy(logits, labels, class_weights=None):
    """
    加权交叉熵损失函数，处理类别不平衡问题
    logits: shape [batch_size, num_classes] -> QTensor（预测值）
    labels: shape [batch_size] -> QTensor（真实标签）
    class_weights: 各类别的权重，形状为[num_classes]的numpy数组
    返回：标量损失（QTensor 类型）
    """
    max_logits = vqt.max(logits, axis=-1, keepdims=True)
    stable_logits = logits - max_logits
    exp_logits = vqt.exp(stable_logits)
    sum_exp = vqt.sum(exp_logits, axis=-1, keepdims=True)
    log_probs = stable_logits - vqt.log(sum_exp)

    batch_size = logits.shape[0]
    total_loss = vqt.zeros((1,))

    for i in range(batch_size):
        if labels.ndim > 1:
            label_idx = int(labels[i].numpy().item())
        else:
            label_idx = int(labels[i].numpy().item())

        # 确保索引在有效范围内
        num_classes = logits.shape[1]
        if label_idx >= num_classes:
            label_idx = num_classes - 1

        # 获取该类别的权重
        weight = 1.0
        if class_weights is not None:
            weight = class_weights[label_idx]

        # 使用标量操作避免维度问题
        loss_value = -log_probs[i][label_idx].item() * weight
        total_loss += QTensor([loss_value])

    return total_loss / batch_size


# 数据预处理模块
def load_data():
    """读取数据并转换为QTensor，进行标准化处理"""
    # 加载训练数据和测试数据
    train = pd.read_csv("data/train_data.csv")
    test = pd.read_csv("data/test_data.csv")

    print(f"训练集形状: {train.shape}, 测试集形状: {test.shape}")
    print(f"特征列: {[col for col in train.columns if col != 'Air Quality']}")
    print(f"标签分布: {train['Air Quality'].value_counts()}")

    # 分离特征和标签
    X_train = train.drop("Air Quality", axis=1).values
    y_train = train["Air Quality"].map({"Good": 0, "Moderate": 1, "Poor": 2, "Hazardous": 3}).values
    X_test = test.drop("Air Quality", axis=1).values
    y_test = test["Air Quality"].map({"Good": 0, "Moderate": 1, "Poor": 2, "Hazardous": 3}).values

    # 计算类别权重 - 使用平方反比加权，进一步增强少数类别的权重
    class_counts = np.bincount(y_train)
    total_samples = len(y_train)
    # 使用平方反比加权，更强调少数类别
    class_weights = (total_samples / (len(class_counts) * class_counts)) ** 2
    # 进一步调整最小类别的权重
    class_weights[3] *= 2.0  # 额外增加类别3的权重
    print(f"增强后的类别权重: {class_weights}")

    # 使用StandardScaler进行标准化，然后缩放到[-1,1]
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 将标准化后的数据映射到[-1,1]区间
    X_train_normalized = 2 * (X_train_scaled - X_train_scaled.min(axis=0)) / (
            X_train_scaled.max(axis=0) - X_train_scaled.min(axis=0) + 1e-8) - 1
    X_test_normalized = 2 * (X_test_scaled - X_train_scaled.min(axis=0)) / (
            X_train_scaled.max(axis=0) - X_train_scaled.min(axis=0) + 1e-8) - 1

    # 将数据转换为QTensor
    X_train_qtensor = QTensor(X_train_normalized)
    y_train_qtensor = QTensor(y_train)
    X_test_qtensor = QTensor(X_test_normalized)
    y_test_qtensor = QTensor(y_test)

    return X_train_qtensor, y_train_qtensor, X_test_qtensor, y_test_qtensor, class_weights


class VariationQuantumCircuit:
    """量子神经网络模块，用于空气质量分类"""

    def __init__(self, n_qubits=9, n_layers=4):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        # 优化参数初始化 - 使用较小的初始值
        self.params = np.random.uniform(-0.05, 0.05, size=(n_layers, n_qubits, 3))

    def allocate_qubits(self, qvm):
        """统一分配量子比特"""
        return qvm.qAlloc_many(self.n_qubits)

    def U_in(self, circuit, qubits, input_data):
        """输入数据编码层: 使用RY门将经典数据编码到量子态"""
        # 确保输入数据的维度与量子比特数匹配
        assert len(input_data) == self.n_qubits, f"输入数据维度({len(input_data)})与量子比特数({self.n_qubits})不匹配"

        # 数据编码：使用RY旋转门将特征值编码到量子态
        for i, qubit in enumerate(qubits):
            # 修正：正确的参数顺序是 RY(qubit, angle)
            angle = float(input_data[i] * np.pi)  # 确保角度是浮点数
            circuit << RY(qubit, angle)  # 正确顺序：先量子比特，后角度

        return circuit

    def U_var(self, circuit, qubits, params_layer):
        """改进的参数化量子层，添加更多纠缠"""
        # 对每个量子比特应用RX, RY, RZ旋转门
        for i, qubit in enumerate(qubits):
            circuit << RX(qubit, params_layer[i][0])
            circuit << RY(qubit, params_layer[i][1])
            circuit << RZ(qubit, params_layer[i][2])

        # 使用全连接纠缠，增加量子比特间的相互作用
        for i in range(self.n_qubits):
            for j in range(i + 1, self.n_qubits):
                # 只应用部分CNOT门以避免过度纠缠
                if (i + j) % 3 == 0:  # 使用模运算选择性地应用CNOT门
                    circuit << pq.CNOT(qubits[i], qubits[j])

        return circuit

    def prepare_circuit(self, input_data):
        """准备完整的量子线路"""
        qvm = CPUQVM()  # 创建量子虚拟机
        qvm.init_qvm()  # 初始化量子虚拟机
        qubits = self.allocate_qubits(qvm)  # 分配量子比特

        circuit = create_empty_circuit()  # 创建空线路

        # 数据输入层
        circuit = self.U_in(circuit, qubits, input_data)

        # 多层参数化量子层
        for layer in range(self.n_layers):
            circuit = self.U_var(circuit, qubits, self.params[layer])

        return qvm, circuit, qubits

    def forward(self, input_data):
        """改进的前向传播，使用更均衡的观测"""
        # 准备量子线路
        qvm, circuit, qubits = self.prepare_circuit(input_data)

        # 创建量子程序
        prog = QProg()
        prog << circuit

        # 构建更平衡的观测算符
        results = []

        # 为类别0使用量子比特0和1
        prog_0 = QProg()
        prog_0 << circuit
        results_0 = expval(qvm, prog_0, {"Z0": 1}, [qubits[0]])
        results.append(results_0)

        # 为类别1使用量子比特2和3
        prog_1 = QProg()
        prog_1 << circuit
        results_1 = expval(qvm, prog_1, {"Z0": 1}, [qubits[2]])
        results.append(results_1)

        # 为类别2使用量子比特4和5
        prog_2 = QProg()
        prog_2 << circuit
        results_2 = expval(qvm, prog_2, {"Z0": 1}, [qubits[4]])
        results.append(results_2)

        # 为类别3使用量子比特6和7
        prog_3 = QProg()
        prog_3 << circuit
        results_3 = expval(qvm, prog_3, {"Z0": 1}, [qubits[6]])
        results.append(results_3)

        # 将结果标准化并返回
        results = np.array(results)
        results = (results + 1) / 2

        return results

    def forward(self, input_data):
        """
        统一观测方式的前向传播：
        使用 Z0~Z3 四个量子比特的期望值作为模型输出（logits）
        """
        # 初始化虚拟机
        qvm = CPUQVM()
        qvm.init_qvm()

        # 分配量子比特
        qubits = qvm.qAlloc_many(self.n_qubits)
        circuit = create_empty_circuit()

        # 数据输入编码层
        circuit = self.U_in(circuit, qubits, input_data)

        # 多层参数化量子门（RX, RY, RZ）+ 纠缠层
        for layer in range(self.n_layers):
            circuit = self.U_var(circuit, qubits, self.params[layer])

        # 构造完整量子程序
        prog = QProg()
        prog << circuit

        # 统一对前4个量子比特进行 Z 方向观测
        observables = {
            "Z0": 1.0,
            "Z1": 1.0,
            "Z2": 1.0,
            "Z3": 1.0
        }

        # 计算期望值
        result = expval(qvm, prog, observables, [qubits[i] for i in range(4)])

        # 提取结果并归一化到 [0, 1]，作为 logits
        logits = np.array([result[key] for key in sorted(result.keys())])
        return (logits + 1) / 2


# 平衡采样函数，确保每个批次中的类别分布更均衡
def balanced_batch_indices(y, batch_size):
    """为每个批次创建均衡的类别分布"""
    n_classes = len(np.unique(y))
    class_indices = [np.where(y == i)[0] for i in range(n_classes)]

    # 计算每个类别在批次中的样本数
    samples_per_class = batch_size // n_classes
    remainder = batch_size % n_classes

    batch_indices = []
    for i in range(n_classes):
        # 为每个类别选择样本，多余的分配给少数类别
        n_samples = samples_per_class + (1 if i >= n_classes - remainder else 0)
        # 从该类别中随机选择样本
        if len(class_indices[i]) >= n_samples:
            selected = np.random.choice(class_indices[i], n_samples, replace=False)
        else:
            # 如果该类别样本不足，则使用重复采样
            selected = np.random.choice(class_indices[i], n_samples, replace=True)
        batch_indices.extend(selected)

    # 随机打乱批次内的样本顺序
    np.random.shuffle(batch_indices)
    return batch_indices


# 中心差分梯度计算
def central_diff_gradient(params, model, X_batch, y_batch, class_weights, h=0.05, n_params_per_iter=100):
    """
    使用中心差分法计算梯度，每次更新部分参数
    """
    # 保存原始参数
    orig_params = params.copy()

    # 计算基准损失
    model.params = orig_params
    outputs = model.batch_forward(X_batch)
    base_loss = weighted_cross_entropy(outputs, y_batch, class_weights).item()

    # 获取参数总数
    total_params = params.size

    # 随机选择要更新的参数索引，增加每次更新的参数数量
    param_indices = np.random.choice(total_params, size=min(n_params_per_iter, total_params), replace=False)

    # 创建扁平化参数数组便于索引
    flat_params = params.reshape(-1)
    gradients = np.zeros_like(flat_params)

    # 计算选定参数的梯度
    for idx in param_indices:
        # 前向差分：参数 +h
        flat_params_plus = flat_params.copy()
        flat_params_plus[idx] += h
        model.params = flat_params_plus.reshape(params.shape)
        outputs_plus = model.batch_forward(X_batch)
        loss_plus = weighted_cross_entropy(outputs_plus, y_batch, class_weights).item()

        # 后向差分：参数 -h
        flat_params_minus = flat_params.copy()
        flat_params_minus[idx] -= h
        model.params = flat_params_minus.reshape(params.shape)
        outputs_minus = model.batch_forward(X_batch)
        loss_minus = weighted_cross_entropy(outputs_minus, y_batch, class_weights).item()

        # 中心差分梯度
        gradients[idx] = (loss_plus - loss_minus) / (2 * h)

    # 将梯度重新整形为参数形状
    gradients = gradients.reshape(params.shape)

    return orig_params, gradients, base_loss


# 训练平衡模型
def train_balanced_model(model, X_train, y_train, class_weights, batch_size=32,
                         epochs=30, val_split=0.2, init_lr=0.2, min_lr=0.01):
    """
    训练量子神经网络模型，使用平衡采样和中心差分计算梯度
    """
    # 数据分割
    n_samples = X_train.shape[0]
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    val_size = int(n_samples * val_split)

    # 转换为NumPy数组进行分割
    X_train_np = X_train.numpy()
    y_train_np = y_train.numpy()

    # 分割数据集
    X_train_split_np = X_train_np[indices[val_size:]]
    y_train_split_np = y_train_np[indices[val_size:]]
    X_val_np = X_train_np[indices[:val_size]]
    y_val_np = y_train_np[indices[:val_size]]

    # 转回QTensor
    X_val = QTensor(X_val_np)
    y_val = QTensor(y_val_np)

    print(f"训练集大小: {len(X_train_split_np)}, 验证集大小: {len(X_val_np)}")

    # 记录训练和验证损失
    train_losses = []
    val_losses = []

    # 创建多个批次，确保类别平衡
    n_batches = max(20, len(X_train_split_np) // batch_size)  # 确保每个epoch至少20个批次

    for epoch in range(epochs):
        # 学习率衰减策略 - 线性衰减
        lr = max(min_lr, init_lr * (1 - epoch / epochs))
        print(f"[Epoch {epoch + 1}/{epochs}] 学习率: {lr:.6f}")

        epoch_losses = []

        for batch_idx in range(n_batches):
            # 使用平衡采样创建批次
            indices = balanced_batch_indices(y_train_split_np, batch_size)

            # 获取批次数据
            X_batch_np = X_train_split_np[indices]
            y_batch_np = y_train_split_np[indices]

            # 转换为QTensor
            X_batch = QTensor(X_batch_np)
            y_batch = QTensor(y_batch_np)

            # 使用中心差分法计算梯度
            model.params, gradients, batch_loss = central_diff_gradient(
                model.params, model, X_batch, y_batch, class_weights,
                h=0.05, n_params_per_iter=100
            )

            # 梯度裁剪，避免梯度爆炸
            grad_norm = np.linalg.norm(gradients)
            if grad_norm > 1.0:
                gradients = gradients / grad_norm

            # 参数更新
            model.params -= lr * gradients

            epoch_losses.append(batch_loss)

            # 打印每个批次的损失
            if (batch_idx + 1) % 10 == 0 or batch_idx == n_batches - 1:
                print(f"  Batch {batch_idx + 1}/{n_batches}, Loss: {batch_loss:.4f}")

        # 计算训练集的平均损失
        train_loss = np.mean(epoch_losses)
        train_losses.append(train_loss)

        # 计算验证集损失
        val_outputs = model.batch_forward(X_val)
        val_loss = weighted_cross_entropy(val_outputs, y_val, class_weights).item()
        val_losses.append(val_loss)

        print(f"  Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # 打印当前预测分布
        if (epoch + 1) % 5 == 0 or epoch == 0:
            val_preds = val_outputs.numpy().argmax(axis=1)
            val_true = y_val.numpy()
            print(f"  验证集预测分布: {np.bincount(val_preds, minlength=4)}")
            print(f"  验证集真实分布: {np.bincount(val_true, minlength=4)}")

            # 计算各类别的验证准确率
            class_accuracies = {}
            for cls in range(4):
                mask = (val_true == cls)
                if np.sum(mask) > 0:
                    class_accuracies[cls] = np.sum((val_preds == cls) & mask) / np.sum(mask)
                else:
                    class_accuracies[cls] = 0
            print(f"  各类别验证准确率: {class_accuracies}")

    return train_losses, val_losses


# 测试量子模型
def quantum_model_test(model, X_test, y_test):
    """测试量子神经网络模型并计算各类别的准确率"""
    predictions = []
    true_labels = y_test.numpy()

    # 逐个样本进行预测
    for i in range(X_test.shape[0]):
        input_data = X_test[i].numpy()
        output = model.forward(input_data)
        pred_class = np.argmax(output)
        predictions.append(pred_class)

    predictions = np.array(predictions)

    # 计算总体准确率
    accuracy = accuracy_score(true_labels, predictions)

    # 计算F1分数
    f1 = f1_score(true_labels, predictions, average='weighted')

    # 计算各类别的准确率
    class_accuracies = {}
    conf_matrix = confusion_matrix(true_labels, predictions)

    # 计算每个类别的样本数
    class_counts = {}
    for cls in range(len(np.unique(true_labels))):
        mask = (true_labels == cls)
        class_counts[cls] = np.sum(mask)
        if class_counts[cls] > 0:
            class_accuracies[cls] = np.sum((predictions == cls) & mask) / class_counts[cls]
        else:
            class_accuracies[cls] = 0

    return accuracy, f1, class_accuracies, class_counts, conf_matrix, predictions


# 可视化损失函数
def plot_losses(train_losses, val_losses):
    """绘制训练过程中的损失曲线"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='训练损失')
    plt.plot(val_losses, label='验证损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('训练和验证损失曲线')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_curves.png')
    plt.close()


# 可视化混淆矩阵
def plot_confusion_matrix(conf_matrix):
    """绘制混淆矩阵热图"""
    plt.figure(figsize=(8, 6))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('混淆矩阵')
    plt.colorbar()

    classes = ['Good', 'Moderate', 'Poor', 'Hazardous']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # 在格子中显示数字
    thresh = conf_matrix.max() / 2.
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(j, i, conf_matrix[i, j],
                     horizontalalignment="center",
                     color="white" if conf_matrix[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.savefig('confusion_matrix.png')
    plt.close()


# 主程序
if __name__ == "__main__":
    # 设置随机种子以便结果可复现
    np.random.seed(42)

    # 加载数据
    X_train, y_train, X_test, y_test, class_weights = load_data()

    # 打印数据形状
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    # 创建量子神经网络模型，增加深度和纠缠性
    quantum_nn = VariationQuantumCircuit(n_qubits=9, n_layers=5)

    # 训练模型
    print("开始训练...")
    train_losses, val_losses = train_balanced_model(
        quantum_nn, X_train, y_train, class_weights,
        batch_size=8, epochs=3, init_lr=0.1, min_lr=0.01
    )

    # 绘制损失曲线
    plot_losses(train_losses, val_losses)

    # 测试模型
    print("开始测试量子神经网络...")
    accuracy, f1, class_accuracies, class_counts, conf_matrix, predictions = quantum_model_test(quantum_nn, X_test,
                                                                                                y_test)

    # 打印测试结果
    print(f"测试集总样本数: {X_test.shape[0]}")
    print(f"各类别样本数: {class_counts}")
    print(f"测试集总体准确率: {accuracy:.4f}")
    print(f"各类别准确率: {class_accuracies}")
    print(f"测试集F1分数: {f1:.4f}")

    # 打印混淆矩阵
    print("\n混淆矩阵:")
    print("     ", end="")
    for i in range(4):
        print(f"预测 {i}  ", end="")
    print()

    for i in range(4):
        print(f"实际 {i}:", end="")
        for j in range(4):
            print(f"{conf_matrix[i, j]:8d}", end="")
        print()

    # 可视化混淆矩阵
    plot_confusion_matrix(conf_matrix)

    # 计算预测类别分布
    pred_distribution = np.bincount(predictions, minlength=4)
    true_distribution = np.bincount(y_test.numpy(), minlength=4)

    print(f"\n预测类别分布: {pred_distribution}")
    print(f"真实类别分布: {true_distribution}")

    # 保存模型参数
    np.save('quantum_model_params.npy', quantum_nn.params)
    print("模型参数已保存至 quantum_model_params.npy")