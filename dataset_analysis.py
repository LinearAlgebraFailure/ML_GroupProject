from datasets import load_dataset

# 加载AG News数据集
dataset = load_dataset("ag_news")

# 分析训练集文本长度
train_texts = dataset["train"]["text"]
lengths = [len(text.split()) for text in train_texts]

# 计算长度分布的一些关键统计数据：平均值、中位数、90th百分位数
mean_length = sum(lengths) / len(lengths)
median_length = sorted(lengths)[len(lengths) // 2]
percentile_90th = sorted(lengths)[int(len(lengths) * 0.9)]
percentile_995th = sorted(lengths)[int(len(lengths) * 0.995)]

print(f"平均长度: {mean_length}")
print(f"中位数长度: {median_length}")
print(f"90th百分位长度: {percentile_90th}")
print(f"99.5th百分位长度: {percentile_995th}")
