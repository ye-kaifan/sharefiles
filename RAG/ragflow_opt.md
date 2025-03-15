
# ragflow 调优策略

## pdf to md

olmocr比marker更强大。
https://olmocr.allenai.org/


## embedding model

- PhysBERT

PhysBERT is a specialized text embedding model for physics, designed to improve information retrieval, citation classification, and clustering of physics literature. Trained on 1.2 million physics papers, it outperforms general-purpose models in physics-specific tasks.

## 分块策略

不同类型的文档选取不同的chunk size.

## GraphRAG 的困难

- 耗时长，刘川格点教材等了几个小时也没解析完。
- 需要自己手动添加实体类别（entity type）。

## 提示词工程

CO-STAR

Context
请你根据问题搜索资料作为背景知识。

Objective
请完成以下复合任务：
1. 任务1
2. 任务2

Style
模仿Richard Feynmann的写作风格

Tone
保持严谨客观的学术表达，但需在讨论环节体现创新性批判思维。对矛盾数据采用"可能暗示...""不排除..."等试探性表述，结论部分使用"强烈支持/尚无法证实"等分级论断。

Audience
目标受众为物理学专家，请你用专业的语言讲清楚细节。

Response
输出结构要求：markdown格式。

# DeepSearcher

- 