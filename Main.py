import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchtext.datasets import AG_NEWS
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from Model import TextClassifier

# 数据预处理
def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)

tokenizer = get_tokenizer('basic_english')
train_iter, test_iter = AG_NEWS()
vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

# 创建数据加载器
def collate_batch(batch):
    label_list, text_list, offsets = [], [], [0]
    for (_label, _text) in batch:
         label_list.append(_label - 1)  # AG_NEWS的标签从1开始
         processed_text = torch.tensor(vocab(tokenizer(_text)), dtype=torch.int64)
         text_list.append(processed_text)
         offsets.append(processed_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    text_list = torch.cat(text_list)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    return label_list.to(device), text_list.to(device), offsets.to(device)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_dataloader = DataLoader(train_iter, batch_size=30, shuffle=True, collate_fn=collate_batch)
test_dataloader = DataLoader(test_iter, batch_size=60, shuffle=False, collate_fn=collate_batch)

# 初始化模型、损失函数和优化器
num_class = len(set([label for (label, text) in AG_NEWS(split='train')]))
vocab_size = len(vocab)
embed_dim = 64
model = TextClassifier(vocab_size, embed_dim, num_class).to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.05)

# 训练和测试循环
def train(dataloader):
    model.train()
    total_acc, total_count = 0, 0
    for idx, (label, text, offsets) in enumerate(dataloader):
        optimizer.zero_grad()
        predited_label = model(text, offsets)
        loss = loss_function(predited_label, label)
        loss.backward()
        optimizer.step()
        total_acc += (predited_label.argmax(1) == label).sum().item()
        total_count += label.size(0)
    return total_acc/total_count

def evaluate(dataloader):
    model.eval()
    total_acc, total_count = 0, 0
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for idx, (label, text, offsets) in enumerate(dataloader):
            predited_label = model(text, offsets)
            total_acc += (predited_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
            all_labels.extend(label.tolist())
            all_preds.extend(predited_label.softmax(dim=-1)[:,1].tolist())  # 假设正类标签为1
    return total_acc/total_count, all_labels, all_preds

# 绘制AUROC曲线
# 绘制准确率曲线
epochs = 300
train_acc_list = []
test_acc_list = []
for epoch in range(epochs):
    train_acc = train(train_dataloader)  # 训练模型并获取训练准确率
    test_acc, _, _ = evaluate(test_dataloader)  # 测试模型并获取测试准确率（忽略AUROC相关的返回值）
    train_acc_list.append(train_acc)
    test_acc_list.append(test_acc)
    print(f'Epoch: {epoch + 1}, Train acc: {train_acc:.4f}, Test acc: {test_acc:.4f}')

# 绘图
plt.figure(figsize=(20, 9))
plt.plot(range(1, epochs+1), train_acc_list, label='Train Accuracy', marker='o')
plt.plot(range(1, epochs+1), test_acc_list, label='Test Accuracy', marker='o')
plt.title('Accuracy vs. Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

