# 基于简单动态迷宫环境的小模型推理能力训练

动态迷宫环境来自https://github.com/KotoHanon/LLM-Reasoning-Maze

***1. QWen2.5-0.5B-Instruct with GRPO***

训练环境：4090单卡

![](https://img.picui.cn/free/2025/03/13/67d2fd193e411.png)

可以看见训练效果还是比较理想，在只训了75个steps的情况下的正确率已经挺高了。但是我在观测LM的reasoning output时发现：

![](https://img.picui.cn/free/2025/03/13/67d2fd938e34e.png)

这意味着虽然LM给出了错误的思考过程，但竟然给出了正确的答案。一个可能的答案是：从理论上，GRPO应该对<answer>RR</answer>中的
RR分配reward = 1，response的其它部分都是0。然而，因为GRPO不存在critic，所以没有credit assignment的能力，LM误认为整个response
都是正确的。当然，这也与LM自身的能力有关系，拿0.5B来做reasoning还是太痴心妄想。我设计的rule-based reward没有对format进行奖励，
而是对不满足format的response进行惩罚（在一定程度上借鉴了[1]）：

$\text{Reward} = I(\text{Verifier}(y)) - 0.2I(\text{FormatCheck}(y)) - 0.005|\text{len}(y) - 512|$

这在一定程度上可以促进exploration。

***2. QWen2.5-0.5B-Instruct with PPO***

训练环境：A40单卡

![](https://img.picui.cn/free/2025/03/14/67d3d79016c4e.png)

训练效果不如GRPO，主要可能是因为PPO需要去学习critic，在做credit assignment的过程需要一定的step预热。还有一种可能是（来自知乎[2]）：
因为策略梯度本质上是贝尔曼方程，reward 的调整会通过 TD-error 作用到更早的时间步，传递的长度受 gae 参数 lambda, gamma 影响。而 TRPO, PPO 将对数比作用在损失及梯度上只会影响当前状态的策略输出，虽然最终也能通过状态价值间接影响到时间序列之前的状态，但作用并不直接。在 LLM 微调的场景，时间步就是文本的序列，每个位置的 token 是序列中之前所有 token 做为输入的预测。如果希望当前位置的 token 输出不要太偏离预训练模型，仅仅控制当前位置输出可能是不够的，必须干扰序列前方的 token，才能更好的控制当前位置的 token 生成。 RL 的 reward 能通过 gae 影响到前面位置的 gae 进而影响到前面位置的 token 生成策略，相比只控制当前位置的梯度，作用更直接，因而能更好的实现微调同时不偏离预训练的效果。

同时，我也发现：

![](https://img.picui.cn/free/2025/03/14/67d3d8d9c3004.png)

0.5B在给出正确答案的同时也给出了正确的推理过程！看来这与我上面对credit assignment的分析比较一致。

***3. QWen2.5-1.5B-Instruct with GRPO***

训练环境：A100(80G)单卡

![](https://img.picui.cn/free/2025/03/15/67d45a842f927.png)

和0.5B的训练曲线相比，最显著的一个差异就是1.5B的模型在做reasoning上明显不会那么偷懒。但仍然存在的问题是：

![](https://img.picui.cn/free/2025/03/15/67d45c73a45df.png)

reasoning过程有错误（相比于0.5B的连智能体位置都推理错了，1.5B“仅仅”是把两个障碍物的位置推理错了），但结果是对的。这个问题可能还是我前面分析的：GRPO的credit assignment问题。还有一种可能如DAPO[3]所分析的那样：过长的样本（对应于推理过程）中的token对损失的比例很低，这导致了胡言乱语。

***4. QWen2.5-1.5B-Instruct with PPO***

训练环境：A100(80G)双卡

![](https://img.picui.cn/free/2025/03/17/67d700d3a931e.png)

![](https://img.picui.cn/free/2025/03/17/67d701a47d2e1.png)

老实说，这个效果离我的预期有点远。更令我感到疑惑的是：QWen2.5-1.5B-Instruct似乎连我的prompt都理解错了，它有时会把每个动作代表的字母搞乱，这也许是它效果如此差的原因之一。我在想，是否是因为模型的参数少了，导致不能正确地理解我的prompt呢？然而，我现在手上没有更多的算力，我预感使用7B的version效果一定会更好，至少不会直接出现如此严重的hallucination。等以后有机会再研究7B的情况吧。

最后，我再尝试分析一下为什么GRPO的效果比PPO好了这么多。我觉得还是和PPO的critic有关，因为critic的TD特性，这使得error可以传播到当前token**前面**的每一个token上，而GRPO不会，它根本就不会去做credit assignment，属于**是就是，不是就不是**的问题。感觉这有点像Model-based和Model-free的performance gap。还有一种可能是：我在上面设计的rule-based reward中，把optimal reponse length设为了512，这也许会导致一些问题。因为一些研究[4]表明，推理token数量的过度增加反而会抑制推理表现。

---

**References**

[1] Aggarwal, Pranjal, and Sean Welleck. "L1: Controlling How Long A Reasoning Model Thinks With Reinforcement Learning." arXiv preprint arXiv:2503.04697 (2025).

[2] 在强化学习 PPO 算法中，为什么可以把 KL 散度直接放进负奖励？ 答主：Yunfei Liu https://www.zhihu.com/question/629107126/answer/3353465906

[3]Yu, Qiying, et al. "DAPO: An Open-Source LLM Reinforcement Learning System at Scale." arXiv preprint arXiv:2503.14476 (2025).

[4] Ballon, Marthe, Andres Algaba, and Vincent Ginis. "The Relationship Between Reasoning and Performance in Large Language Models--o3 (mini) Thinks Harder, Not Longer." arXiv preprint arXiv:2502.15631 (2025).


