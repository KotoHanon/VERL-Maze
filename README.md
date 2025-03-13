# 基于简单动态迷宫环境的小模型推理能力训练

动态迷宫环境来自https://github.com/KotoHanon/LLM-Reasoning-Maze

***1. QWen2.5-0.5B-Instruct with GRPO***

![](https://img.picui.cn/free/2025/03/13/67d2fd193e411.png)

可以看见训练效果还是比较理想，在只训了75个steps的情况下的正确率已经挺高了。但是我在观测LM的reasoning output时发现：

![](https://img.picui.cn/free/2025/03/13/67d2fd938e34e.png)

这意味着虽然LM给出了错误的思考过程，但竟然给出了正确的答案。一个可能的答案是：从理论上，GRPO应该对<answer>RR</answer>中的
RR分配reward = 1，response的其它部分都是0。然而，因为GRPO不存在critic，所以没有credit assignment的能力，LM误认为整个response
都是正确的。当然，这也与LM自身的能力有关系，拿0.5B来做reasoning还是太痴心妄想。我设计的rule-based reward没有对format进行奖励，
而是对不满足format的response进行惩罚，这在一定程度上可以促进exploration。
