# GFlowPlanner: Efficient Training of LLM Policy for Diverse Reasoning

We focus on generating diverse solution trajectories in multi-step reasoning problems. Specifically, we formulate LLM reasoning as a Markovian flow from an initial state to terminals and adapt the GFlowNets training approach to enable diverse reasoning and apply our method to embodied reasoning (BlocksWorld), puzzle solving (Game of 24), and logical reasoning (PrOntoQA) tasks. 

Our GFlowPlanner leads to:

1. Diverse-Reasoning: Multiple reasoning solutions to the reasoning tasks can be found via sampling.
2. Sample-Efficiency: Limited data (e.g. 15 examples) can train the LLM policy well.

Find more details in our paper:
Fangxu Yu*, Lai Jiang*, Haoqiang Kang*, Shibo Hao, Lianhui Qin, "[GFlowPlanner: Efficient Training of LLM Policy for Diverse Reasoning]()" (* Equal contribution)

## GFlowPlanner

![plot](./images/main_arch.png)

Our GFlowPlanner formulates multi-step reasoning tasks as flow:
1. Design reward $R(s_n)$ of terminal states for different tasks.
2. Collect trajectories with the local search technique.
3. Training LLM policy $P_{F}$ with trajectory balance loss.

## Code
**1) Download this GitHub**
```
git clone https://github.com/Yu-Fangxu/COLD-Attack.git
```

**2) Setup Environment**


