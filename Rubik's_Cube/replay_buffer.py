import torch
import heapq
import random
import pickle
import gzip
import numpy as np

import editdistance

class ReplayBuffer:
    """
    A relay buffer that uses a heap to keep the max_size items with the highest reward
    """

    def __init__(self, buffer_size, prb=True, sim_tolerance=0.25):
        self.buffer_size = buffer_size
        self.sim_tolerance = sim_tolerance
        self.prb = prb
        self.reset()

    def reset(self):
        self._buffer = {}

    def add(self, problem, plan, sample, log_reward):
        """
        add an item to the buffer, where item = [log reward, tensor of shape (seq_len, )]
        """
        # if the plans have already existed in the problem
        if problem not in self._buffer:
            self._buffer[problem] = {
                "sentences": [],
                "exists": set(),
            }
        if plan in self._buffer[problem]["exists"]:
            return
        # if the edit distance between item and any item in the buffer is small, skip it
        # tokenized_sentence = [
        #     x
        #     for x in item["tensor_sentence"].tolist()
        #     if x != self.termination_token_id
        # ]
        # for buffer_item in self._buffer[problem]["sentences"]:
            # tokenized_existing_sentence = [
            #     x for x in buffer_item[2].tolist() if x != self.termination_token_id
            # ]
            # if (
            #     editdistance.eval(tokenized_sentence, tokenized_existing_sentence)
            #     < (len(tokenized_sentence) + len(tokenized_existing_sentence))
            #     * self.sim_tolerance
            # ):
            
        heapq.heapify(self._buffer[problem]["sentences"])
        self._buffer[problem]["exists"].add(plan)
        heapq.heappush(
            self._buffer[problem]["sentences"],
            (
                log_reward,
                plan,
                sample
            ),
        )
            
        if len(self._buffer[problem]["sentences"]) > self.buffer_size:
            # popped = heapq.heappushpop(
            #     self._buffer[problem]["sentences"],
            #     (
            #                log_reward,
            #                 plan,
            #                 sample
            #     ),
            # )
            popped = heapq.heappop(self._buffer[problem]["sentences"])
            self._buffer[problem]["exists"].discard(popped[1])
        # else:
        #     heapq.heappush(
        #         self._buffer[problem]["sentences"],
        #         (
        #             log_reward,
        #             plan,
        #             sample
        #         ),
        #     )

    # def add_batch(self, prompt, sentences, logrewards, tokenizer):
    #     """
    #     add a batch of items to the buffer
    #     """
    #     str_prompt = " ".join([str(x) for x in prompt.tolist()])
    #     if str_prompt not in self._buffer:
    #         self._buffer[str_prompt] = {
    #             "tensor_prompt": prompt,
    #             "sentences": [],
    #             "exists": set(),
    #         }
    #     sentences[
    #         (sentences == self.termination_token_id).cumsum(dim=-1) >= 1
    #     ] = self.termination_token_id
    #     token_sentences = tokenizer.batch_decode(sentences)
    #     for i in range(sentences.size(0)):
    #         str_sentence = token_sentences[i].replace(".", "").strip()
    #         self.add(
    #             {
    #                 "logreward": logrewards[
    #                     i, (sentences[i] != self.termination_token_id).sum()
    #                 ].item(),
    #                 "str_prompt": str_prompt,
    #                 "str_sentence": str_sentence,
    #                 "tensor_sentence": sentences[i],
    #                 "full_logrewards": logrewards[i, :],
    #             }
    #         )

    def sample(self, batch_size, problem):
        """
        uniformly sample a batch of items from the buffer,
        and return a stacked tensor
        """
        # str_prompt = " ".join([str(x) for x in prompt.tolist()])
        if problem not in self._buffer:
            return None, None, None
        prompt_buffer = self._buffer[problem]["sentences"]
        sorted_buffer = sorted(prompt_buffer, key=lambda x: x[0])
        idx_list = np.arange(len(prompt_buffer))
        
        if self.prb:
            # sample_size = batch_size // 2
            # top_20_percentile = idx_list[int(len(sorted_buffer) * 0.8):]
            # idxs_1 = np.random.choice(
            #     top_20_percentile,
            #     sample_size,
            #     replace=True,
            # )
            # idx_1 = torch.from_numpy(idx_1)
            # other_80_percentile = idx_list[:int(len(sorted_buffer) * 0.8)+1]
            # idxs_2 = np.random.choice(
            #     other_80_percentile,
            #     sample_size,
            #     replace=True,
            # )
            # idx_2 = torch.from_numpy(idx_2)

            # idx = torch.cat([idx_1, idx_2], dim=-1)
            
            priorities  = [item[0] for item in prompt_buffer]
            priorities = torch.tensor(priorities, dtype=torch.float32)  # 确保priorities是float类型
            # priorities = torch.exp(priorities)
            priorities = priorities - torch.max(priorities)  # 从每个元素中减去最大值以增加数值稳定性

            # 计算概率分布
            probabilities = torch.exp(priorities) / torch.sum(torch.exp(priorities))

            # 从prompt_buffer中随机选择索引
            # 假设 prompt_buffer 的长度已知，存储在 prompt_buffer_length 变量中
            # print(probabilities)
            idx = torch.multinomial(probabilities, batch_size, replacement=True)
        else:
            idx = np.random.choice(
                len(prompt_buffer),
                batch_size,
                replace=True,
            )
        return [prompt_buffer[i][0] for i in idx], [prompt_buffer[i][1] for i in idx], [prompt_buffer[i][2] for i in idx],
        # return torch.nn.utils.rnn.pad_sequence(
        #     [prompt_buffer[i][2] for i in idx],
        #     batch_first=True,
        #     padding_value=self.termination_token_id,
        # ), torch.nn.utils.rnn.pad_sequence(
        #     [prompt_buffer[i][3] for i in idx],
        #     batch_first=True,
        #     padding_value=0,
        # )

    def print(self):
        for key in self._buffer:
            print(key)
            for item in self._buffer[key]["sentences"]:
                print(item[1])
            print("")

    def save(self, path):
        with gzip.open(path, "wb") as f:
            pickle.dump(self._buffer, f)