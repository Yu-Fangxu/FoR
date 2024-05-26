# GFlowPlanner for Embodied Reasoning (BlocksWorld)

## Task Introduction
Blocksworld problems capture commonsense block manipulations and consist of a set of blocks. Blocks are identified with unique colors and placed on a table or on top of other blocks. The goal is to arrange some of these blocks in a stack in a particular order.

A model needs to give a sequence of actions to rearrange blocks into stacks in a particular order. A state is defined as the current orientation of the blocks, and an action is a textual instruction to move these blocks. An action involves one of four verbs (*STACK, UNSTACK, PUT, PICKUP*) and targeted objects. We generate valid actions based on domain restrictions and the current block orientation. To transit between states, the LLM is prompted to predict the next state $s_t$ based on the last state $s_{t-1}$ and current action $a_t$. The planning process is terminated once a state meets all goal conditions or reaches a maximum step limit. 
