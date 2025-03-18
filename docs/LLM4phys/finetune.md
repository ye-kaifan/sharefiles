
# 微调物理专家大模型

## 相关的论文

- Fine-tuning Vision Transformers for the Prediction of State Variables in Ising Models
  - Abstract: Transformer 是先进的深度学习模型，由堆叠的注意力和点状、全连接层组成，旨在处理序列数据。Transformer 不仅在自然语言处理（NLP）中无处不在，而且最近也激发了一波计算机视觉（CV）应用研究的新浪潮。在这项工作中，视觉 Transformer（ViT）被微调以预测二维伊辛模型模拟的状态变量。我们的实验表明，当使用来自伊辛模型对应于各种边界条件和温度的少量微态图像时，ViT 优于最先进的卷积神经网络（CNN）。这项工作探讨了 ViT 在其他模拟中的潜在应用，并引入了关于注意力图如何学习不同现象的潜在物理的研究方向。
  - https://arxiv.org/abs/2109.13925

- Can a CNN trained on the Ising model detect the phase transition of the $q$-state Potts model?
  - Abstract: 采用在二维伊辛模型的自旋配置和温度上训练的深度卷积神经网络（深度 CNN），我们研究深度 CNN 是否可以检测二维 $q$-态 Potts 模型的相变。为此，我们通过将自旋变量 $\{0, 1, \ldots, \lfloor q/2 \rfloor - 1\}$ 和 $\{\lfloor q/2 \rfloor, \ldots, q-1\}$ 分别替换为 $\{0\}$ 和 $\{1\}$，生成 $q$-态 Potts 模型 ($q \geq 3$) 的自旋配置的二值图像。然后，我们将这些图像输入到训练好的 CNN 中，输出预测的温度。$q$-态 Potts 模型的二值图像与伊辛自旋配置完全不同，尤其是在相变温度时。此外，我们的 CNN 模型并未训练有关相是否有有序/无序的信息，而是通过伊辛自旋配置及其生成的温度标签进行天真的训练。尽管如此，深度 CNN 可以以高精度检测到相变点，无论相变的类型如何。我们还在高温区域发现，CNN 根据内能输出温度，而在低温区域，输出取决于磁化强度以及可能还有内能。然而，在相变点附近，CNN 可能使用更一般的因素来检测相变点。
  - https://arxiv.org/abs/2104.03632v3

- Quantum many-body physics calculations with large language models
  - Abstract: 大型语言模型（LLMs）在多个领域，包括数学和科学推理中，展示了执行复杂任务的能力。我们证明，通过精心设计的提示，LLMs可以准确执行理论物理研究论文中的关键计算。我们关注量子物理学中广泛使用的近似方法：哈特里-福克方法，它需要通过多步解析计算推导出近似哈密顿量和相应的自洽方程。为了使用LLMs进行计算，我们设计了多步提示模板，将解析计算分解为标准化的步骤，并为特定问题信息留出占位符。我们评估了 GPT-4 在执行过去十年 15 篇论文的计算任务中的表现，证明通过纠正中间步骤，它在 13 个案例中可以正确推导出最终的哈特里-福克哈密顿量。汇总所有研究论文，我们发现单个计算步骤的执行平均得分为 87.5（满分 100 分）。 我们进一步使用LLMs来缓解此评估过程中的两个主要瓶颈：（i）从论文中提取信息以填充模板；（ii）自动评分计算步骤，在两种情况下都取得了良好的结果。
  - https://arxiv.org/abs/2403.03154

- PACuna: Automated Fine-Tuning of Language Models for Particle Accelerators
  - Abstract: 粒子加速器领域的导航随着近期贡献的激增而变得越来越具有挑战性。这些复杂的设备即使在单个设施内也难以理解。为了解决这个问题，我们引入了 PACuna，这是一个通过公开可用的加速器资源（如会议、预印本和书籍）进行微调的语言模型。我们自动化了数据收集和问题生成，以最小化专家的参与，并使代码可供使用。PACuna 在解决由专家验证的加速器问题方面表现出色。我们的方法表明，通过微调技术文本和自动生成的语料库来适应科学领域，可以进一步产生预训练模型，以回答一些商业可用助手无法回答的特定问题，并可作为单个设施的人工智能助手。
  - https://arxiv.org/abs/2310.19106


## 用unsloth进行LoRA高效微调

1. 准备json或jsonl格式的问答对数据集。
   - https://huggingface.co/datasets/camel-ai/physics
2. 设置训练超参数，learning rate, batch size and num epochs.