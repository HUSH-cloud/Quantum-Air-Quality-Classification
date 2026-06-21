from pyvqnet.dtype import *
from pyvqnet.tensor.tensor import QTensor
from pyvqnet.nn import Module, Linear, ReLu
from pyvqnet.optim.adam import Adam
from pyvqnet.data.data import data_generator
from pyvqnet.nn.loss import CrossEntropyLoss
import numpy as np
from sklearn.metrics import f1_score
import pandas as pd
import pickle

# ===================== 1. 数据加载（直接使用类别索引）=====================
def load_data():
    """读取数据并转换为张量"""
    train_df = pd.read_csv("data/train_data.csv")
    test_df = pd.read_csv("data/test_data.csv")

    label_map = {"Good": 0, "Moderate": 1, "Poor": 2, "Hazardous": 3}
    target_column = "Air Quality"

    # 特征验证
    assert train_df.shape[1] - 1 == 9, f"特征数应为9，实际为{train_df.shape[1] - 1}"

    def process_data(df):
        x = df.drop(target_column, axis=1).values.astype(np.float32)
        y = df[target_column].map(label_map).values.astype(np.int64)  # 直接使用类别索引
        return x.reshape(-1, 9), y  # 直接返回类别索引标签

    x_train, y_train = process_data(train_df)
    x_test, y_test = process_data(test_df)
    mean, std = x_train.mean(axis=0), x_train.std(axis=0)
    x_train = (x_train - mean) / (std + 1e-8)
    x_test = (x_test - mean) / (std + 1e-8)

    return (
        QTensor(x_train), QTensor(y_train),  # 返回类别索引标签
        QTensor(x_test), QTensor(y_test)
    )

# ===================== 2. 模型定义（增加隐藏层结构 9-8-6-4-4）=====================
class AirQualityNN(Module):
    def __init__(self):
        super().__init__()
        # 按照 9->8->6->4->4 结构定义4个全连接层
        self.layer1 = Linear(9, 8)
        self.act1 = ReLu()
        self.layer2 = Linear(8, 6)
        self.act2 = ReLu()
        self.layer3 = Linear(6, 4)
        self.act3 = ReLu()
        # 最后一层输出4个类别的logits
        self.layer4 = Linear(4, 4)

    def forward(self, x):
        if x.shape[1] != 9:
            x = x.reshape([x.shape[0], 9])
        x = self.layer1(x)
        x = self.act1(x)
        x = self.layer2(x)
        x = self.act2(x)
        x = self.layer3(x)
        x = self.act3(x)
        return self.layer4(x)

# ===================== 3. 训练流程（使用CrossEntropyLoss） =====================
def model_train():
    x_train, y_train, x_test, y_test = load_data()

    model = AirQualityNN()
    optimizer = Adam(model.parameters(), lr=0.01)
    criterion = CrossEntropyLoss()

    batch_size = 64
    epochs = 150
    patience = 30
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in data_generator(
                x_train.numpy(), y_train.numpy(), batch_size=batch_size, shuffle=True
        ):
            x_np, y_np = batch
            x = QTensor(x_np.reshape(-1, 9), dtype=kfloat32)
            y = QTensor(y_np, dtype=kint64)

            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(y, outputs)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / (len(x_train) // batch_size)

        # ============== 验证 ==============
        model.eval()
        val_outputs = model(x_test)
        val_loss = criterion(y_test, val_outputs).item()


        # ============== Early Stopping 检查 ==============
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # 保存当前最优模型
            with open("air_model.pkl", "wb") as f:
                pickle.dump(model.state_dict(), f)
        else:
            patience_counter += 1

        # 每10轮打印一次
        if (epoch + 1) % 10 == 0 or patience_counter == 0:
            print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f}")

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

# ===================== 4. 测试流程 =====================
def model_test():
    x_train, y_train, x_test, y_test = load_data()

    model = AirQualityNN()
    with open("air_model.pkl", "rb") as f:
        model.load_state_dict(pickle.load(f))
    model.eval()

    outputs = model(x_test)
    preds = np.argmax(outputs.numpy(), axis=1)  # 使用argmax获取预测的类别索引
    y_true = y_test.numpy()  # 使用整数类别标签作为真实标签

    acc = np.mean(preds == y_true)
    f1 = f1_score(y_true, preds, average="macro")

    print("\n测试结果:")
    print(f"准确率: {acc:.4f}")
    print(f"平均F1分数: {f1:.4f}")

if __name__ == "__main__":
    model_train()
    model_test()
