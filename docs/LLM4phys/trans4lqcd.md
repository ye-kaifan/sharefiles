
# Transformer for LQCD

有两篇文章都是Akio的，把规范协变Transformer架构用于self-learning Monte Carlo方法，提高了接受率，因为Transformer的注意力机制能提取到长程相互作用的特征。还有一篇文章提到，相比DNN，modified Transformer 在估算伪标量胶球的质量时，误差更小。

## CASK: A Gauge Covariant Transformer for Lattice Gauge Theory

- https://arxiv.org/abs/2501.16955

We propose a Transformer neural network architecture specifically designed for lattice QCD, focusing on preserving the fundamental symmetries required in lattice gauge theory. The proposed architecture is gauge covariant/equivariant, ensuring it respects gauge symmetry on the lattice, and is also equivariant under spacetime symmetries such as rotations and translations on the lattice. A key feature of our approach lies in the attention matrix, which forms the core of the Transformer architecture. To preserve symmetries, we define the attention matrix using a Frobenius inner product between link variables and extended staples. This construction ensures that the attention matrix remains invariant under gauge transformations, thereby making the entire Transformer architecture covariant. We evaluated the performance of the gauge covariant Transformer in the context of self-learning HMC. Numerical experiments show that the proposed architecture achieves higher performance compared to the gauge covariant neural networks, demonstrating its potential to improve lattice QCD calculations.

- 把规范协变Transformer 用在self-learning HMC里面，提高了接受率。性能优于他们之前开发的gauge covariant neural networks (“adaptive stout”)。
- the gauge covariant Transformer architecture CASK 不仅能保持规范不变性，还能通过注意力机制提取到长程相互作用的特征。

> In our numerical experiments, the surrogate links generated by CASK successfully absorbed the differences arising from the modified massive Dirac operator, resulting in an improved acceptance rate. The method consistently outperformed the gauge covariant neural networks (“adaptive stout”) developed in our previous study, illustrating how the attention-based design can enhance expressivity. These findings suggest that the gauge covariant Transformer approach is a promising route toward more efficient and flexible simulations in lattice QCD. Future work will explore larger lattice volumes, extended loop structures in the attention matrix, and further optimization of the training process to fully leverage the potential of CASK.


## Self-learning Monte Carlo with equivariant Transformer

- https://arxiv.org/abs/2306.11527
- Equivariant Transformer is all you need (proceedings) https://arxiv.org/abs/2310.13222

Machine learning and deep learning have revolutionized computational physics, particularly the simulation of complex systems. Considering equivariance and long-range correlations is essential for simulating physical systems. Equivariance imposes a strong inductive bias on the probability distribution described by a machine learning model, while long-range correlation is important for understanding classical/quantum phase transitions. Inspired by Transformers used in large language models, which can treat long-range dependencies in the networks, we introduce a symmetry equivariant Transformer for self-learning Monte Carlo. We evaluate our architecture on a spin-fermion model (i.e., the double exchange model) on a two-dimensional lattice. Our results show that the proposed method overcomes the poor acceptance rates of linear models and exhibits a similar scaling law to large language models, with model quality monotonically increasing with the number of layers. Our work paves the way for the development of more accurate and efficient Monte Carlo algorithms with machine learning for simulating complex physical systems.

- the effective model with the Transformer architecture can **capture the long-range correlation** in the original model.
- Leveraging the extensive capacity of the attention layer, we successfully construct an effective model of the system using the equivariant Transformer. We find that the model with the attention blocks **overcomes the poor acceptance rates** of linear models and exhibits a similar scaling law as in large language models.

## Estimation of the pseudoscalar glueball mass based on a modified Transformer

- https://arxiv.org/abs/2408.13280

A modified Transformer model is introduced for estimating the mass of pseudoscalar glueball in lattice QCD. The model takes as input a sequence of floating-point numbers with lengths ranging from 30 to 35 and produces a two-dimensional vector output. It integrates floating-point embeddings and positional encoding, and is trained using binary cross-entropy loss. The paper provides a detailed description of the model’s components and training methods, and compares the performance of the traditional least squares method, the previously used deep neural network, and the modified Transformer in mass estimation. The results show that the modified Transformer model achieves greater accuracy in mass estimation than the traditional least squares method. Additionally, compared to the deep neural network, this model utilizes positional encoding and can handle input sequences of varying lengths, offering enhanced adaptability.

- 相比DNN，modified Transformer 在估算伪标量胶球的质量时，误差更小。