import random
import numpy as np
import torch
from pytorch_lightning import LightningModule
from transformers.trainer_pt_utils import LabelSmoother
from util import base_to_lora
import json
import os
import ast
import csv
import pickle
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.distributions import Categorical
from collections import defaultdict
from cube_utils import *
from solver import *
from util import *

from prompts_cube import cube_instruct

convert_to_int = lambda lst: np.array(list(map(lambda x: int(x), lst)))

class CubeGFNTask(LightningModule):
    def __init__(
        self,
        args,
        model,
        logZ,
        tokenizer,
        replay_buffer,
        train_data=None,
        val_data=None,
    ):
        super().__init__()
        self.args = args
        self.save_hyperparameters(ignore=["model", "tokenizer"])
        self.logZ = logZ
        self.model = model

        if args.use_lora:
            base_to_lora(self.model)

        self.tokenizer = tokenizer
        self.reward = None
        # self.prompts = json.load(open("data/blocksworld/my_mcts_prompts_update.json", 'r'))
        self.replay_buffer = replay_buffer
        self.train_data = train_data
        self.val_data = val_data
        self.n_samples = args.n_samples # 2 for step 4

        self.lr = args.lr
        self.logZ_lr = args.logZ_lr
        self.epsilon = self.args.epsilon_start
        self.get_lr_at_step = lambda step: min(step / 20 * self.lr, self.lr)

        # self.get_reward_temp_at_step = lambda step: self.args.reward_temp_start + (
        #    self.args.reward_temp_end - self.args.reward_temp_start
        # ) * min(1, step / self.args.reward_temp_horizon)

        self.ignore_token_id = LabelSmoother.ignore_index

        self.reward_temperature = self.args.reward_temp_start
        self.pf_temperature = self.args.pf_temp_start
        self.use_buffer_prob = self.args.use_buffer_prob

        self.traj = defaultdict(int)

    def forward(self, problem, pf_temp):
        INIT,  PLAN = problem
        GOAL = None
        INIT = INIT[0]

        (
            generated_text, 
            actions, 
            states,
            reward, 
            sample
        ) = self.generate_trajectories(
            initial_state = INIT,
            goal = None,
            max_steps = self.args.step,
            eos_token_id=self.tokenizer.encode('\n', add_special_tokens=False)[0],
            pf_temp = pf_temp
        )
        return generated_text, actions, states, sample, reward

    def training_step(self, problem, batch_idx):
        INIT, PLAN = problem
        INIT = INIT[0]
        actions = PLAN[0]
        ########################## Compute the reward for ground-truth trajectory ##########################

        LOG_R = []
        LOG_PF = []
        LOG_BF = []
        # Exploitation: Reuse the samples in the buffer

        if (
            random.random() < self.use_buffer_prob
            and self.replay_buffer.sample(self.n_samples, INIT)[0] is not None
        ):
            # Using a sample from the reward buffer
            (log_reward_list,
            state_list,
            sample_list
            ) = self.replay_buffer.sample(
                self.n_samples, INIT
            )

            for state, sample in zip(state_list, sample_list):
                print(state)
                state_ = eval(state, {'array': np.array})
                actions, states = state_[0], state_[1]
                # actions, states = state[0], state[1]
                log_pf, log_bf = self.forward_prob(actions, states)
                LOG_PF.append(log_pf)
                LOG_BF.append(log_bf)
            LOG_R.extend(log_reward_list)
            
        else:
            best_actions = None
            best_states = None
            best_reward = -9999
            for _ in range(self.n_samples):
                if np.random.rand() < self.args.pf_temp_prob:
                    pf_temp = self.pf_temperature
                else:
                    pf_temp = 1.0
                generated_text, actions, states, sample, reward = self.forward(
                    problem, pf_temp
                )

                if self.args.ll_weight == 0:
                    ll_reward = [1 for _ in range(self.args.step)]
                    ll_reward = torch.tensor(ll_reward).to(self.device)
                    ll_weight = 1
                else:
                    ll_reward = self.get_ll_reward(actions, states, None)
                    ll_weight = self.args.ll_weight
                # ll_reward = 3 * torch.pow(ll_reward, 1/3)
                # ll_reward = torch.tensor([1,0]).to(self.device)
                LOG_R.append(torch.log(reward + ll_weight * ll_reward.sum()))
                # print("generated reward: \n", reward)
                # print("generated ll: \n",  ll_reward)
                generated_text = (actions, states)
                self.replay_buffer.add(INIT, str(generated_text), sample, torch.log(reward + ll_weight * ll_reward.sum()))
                log_pf, log_bf = self.forward_prob(actions, states)
                LOG_PF.append(log_pf)
                LOG_BF.append(log_bf)

                actions_joined = '\n'.join(actions)
                self.traj[actions_joined] += 1

                if torch.log(reward + ll_weight * ll_reward.sum()) > best_reward:
                    best_actions  = actions
                    best_states = states
                    best_reward = torch.log(reward + ll_weight * ll_reward.sum())

                # conduct local search
            for _ in range(4):
                _, actions, states, reward, _ = self.local_search(initial_state = INIT,
                    goal = None,
                    max_steps = self.args.step,
                    plan=best_actions,  # use the highest to explore
                    states=best_states,
                    eos_token_id=self.tokenizer.encode('\n', add_special_tokens=False)[0],
                    pf_temp = pf_temp)

                if self.args.ll_weight == 0:
                    ll_reward = [1 for _ in range(self.args.step)]
                    ll_reward = torch.tensor(ll_reward).to(self.device)
                    ll_weight = 1
                else:
                    ll_reward = self.get_ll_reward(actions, states, None)
                    ll_weight = self.args.ll_weight

                log_reward = torch.log(reward + ll_weight * ll_reward.sum())

                # if log_reward is larger, then we accept it

                if log_reward > best_reward:
                    LOG_R.append(torch.log(reward + ll_weight * ll_reward.sum()))
                    generated_text = (actions, states)
                    self.replay_buffer.add(INIT, str(generated_text), sample, torch.log(reward + ll_weight * ll_reward.sum()))
                    log_pf, log_bf = self.forward_prob(actions, states)
                    LOG_PF.append(log_pf)
                    LOG_BF.append(log_bf)
            

        # Obtain the log_pf and log_reward

        LOG_PF = torch.stack(LOG_PF).to(self.model.device)
        LOG_R = torch.stack(LOG_R).to(self.model.device)
        LOG_BF = torch.stack(LOG_BF).to(self.model.device)
        LOG_R_temperd = LOG_R * self.reward_temperature
        if self.args.use_lora:
            base_to_lora(self.model)
    
        # Get the Trajectory balance loss
    
        loss = tb_loss(
            log_pf=LOG_PF,
            log_r=LOG_R_temperd,
            logz=self.logZ,
            log_bf=log_bf,
            logpartition=True
        )

        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            prog_bar=True,
            batch_size=self.args.batch_size
        )
        self.log(
            "train/logR",
            LOG_R.mean(),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            prog_bar=True,
            batch_size=self.args.batch_size
        )

        # print(list(self.traj.values()))

        return loss
        

    def test_step(self, problem, batch_idx):
        # pass
        if self.args.use_lora:
            base_to_lora(self.model)    # 确保转换成lora
        self.model.eval()           # 必须用eval

        INIT, PLAN = problem
        INIT = INIT[0]
        actions = PLAN[0]
        # print(GOAL)
        # print(INIT)
        total_success = 0
        total_solution = 0
        success_text = []

        for _ in range(40):
            (
            generated_text, 
            actions, 
            states,
            reward, 
            sample
            ) = self.generate_trajectories(
                initial_state = INIT,
                goal = None,
                max_steps = self.args.step,
                eos_token_id=self.tokenizer.encode('\n', add_special_tokens=False)[0],
                mode="test"
            )

            last_state = states[-1]
            
            if isSolved(last_state):
                total_success += 1
                
                actions_joined = '\n'.join(actions)
                if (INIT, actions_joined) not in success_text:
                    total_solution += 1
                    success_text.append((INIT, actions_joined))
        with open(f'/home/fangxu/GFlowPlan/success_plans_test/8_step/success_text_{batch_idx}.csv', mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            # 写入列名
            writer.writerow(['Goal', "Initial State", 'Generated plan'])
            # 写入数据
            writer.writerows(success_text)

        if total_success > 0:
            success = 1
        else:
            success = 0

        self.log(
            "test/success",
            success,
            sync_dist=True,
            prog_bar=True,
            batch_size=self.args.batch_size
        )
        self.log(
            "test/n_solutsion",
            total_solution,
            sync_dist=True,
            prog_bar=True,
            batch_size=self.args.batch_size
        )

    def validation_step(self, problem, batch_idx):
        # pass
        if self.args.use_lora:
            base_to_lora(self.model)    # 确保转换成lora
        self.model.eval()           # 必须用eval

        INIT, PLAN = problem
        INIT = INIT[0]
        actions = PLAN[0]
        # print(GOAL)
        # print(INIT)
        total_success = 0
        total_solution = 0
        success_text = []

        for _ in range(40):
            (
            generated_text, 
            actions, 
            states,
            reward, 
            sample
            ) = self.generate_trajectories(
                initial_state = INIT,
                goal = None,
                max_steps = self.args.step,
                eos_token_id=self.tokenizer.encode('\n', add_special_tokens=False)[0],
                mode="test"
            )

            last_state = states[-1]
            if isSolved(last_state):
                total_success += 1
                
                actions_joined = '\n'.join(actions)
                if (INIT, actions_joined) not in success_text:
                    total_solution += 1
                    success_text.append((INIT, actions_joined))
        with open(f'/home/fangxu/GFlowPlan/success_plans_valid/{self.args.step}_step/success_text_{batch_idx}.csv', mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            # 写入列名
            writer.writerow(['Goal', "Initial State", 'Generated plan'])
            # 写入数据
            writer.writerows(success_text)

        if total_success > 0:
            success = 1
        else:
            success = 0

        self.log(
            "val/success",
            success,
            sync_dist=True,
            prog_bar=True,
            batch_size=self.args.batch_size
        )
        self.log(
            "val/n_solutsion",
            total_solution,
            sync_dist=True,
            prog_bar=True,
            batch_size=self.args.batch_size
        )


    def on_train_batch_start(self, problem, batch_idx):
        pass

    def on_train_epoch_start(self):
        # Log scheduled quantities
        current_epoch = self.trainer.current_epoch
        if (current_epoch + 1) % 6 == 0:
            self.pf_temperature = self.args.pf_temp_start - (self.args.pf_temp_start - self.args.pf_temp_end) / (self.args.epochs // 6)

        if current_epoch < self.args.epochs // 2:
            self.epsilon = self.args.epsilon_start - (self.args.epsilon_start - self.args.epsilon_end) / (self.args.epochs // 2)
        
        if current_epoch < self.args.epochs // 2:
            self.reward_temperature = self.args.reward_temp_start + current_epoch * (self.args.reward_temp_end - self.args.reward_temp_start) / (self.args.epochs // 2)
        
        if current_epoch < self.args.epochs // 2:
            self.use_buffer_prob  = self.args.p_buffer_start + current_epoch * (self.args.p_buffer_end - self.args.p_buffer_start) / (self.args.epochs // 2)
        
        # self.reward_temperature = random.uniform(self.args.reward_temp_start, self.args.reward_temp_end)
        
        # self.epsilon = 0
        self.log("scheduled/R_temperature", self.reward_temperature, sync_dist=True)


    def configure_optimizers(self):
        if self.args.use_4bit:
            import bitsandbytes as bnb  # fmt: skip
            optimizer = bnb.optim.PagedAdamW8bit([{'params': self.model.parameters(), 'lr': self.lr},
                                    {'params': [self.logZ,], 'lr': self.logZ_lr}])
            return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": CosineAnnealingLR(optimizer, T_max=10, eta_min=5e-5),
                "monitor": "metric_to_track",
                "frequency": 10,
                # If "monitor" references validation metrics, then "frequency" should be set to a
                # multiple of "trainer.check_val_every_n_epoch".
            }
            }
        else:
            return torch.optim.AdamW([{'params': self.model.parameters(), 'lr': self.lr},
                                    {'params': [self.logZ,], 'lr': self.logZ_lr}])

    def generate_trajectories(self,
                            initial_state,
                            goal,
                            max_steps,
                            eos_token_id,
                            pf_temp=1.0,
                            mode="train",
                          ):
        """
        return: trajs, probability of each action in the trajs, log rewards of the trajs, log rewards of (state, action)
        """
        if self.args.use_lora:
            base_to_lora(self.model)
        self.model.eval()
        prompt = cube_instruct
        # turn initial state into prompt form
        initial_state_list = convert_to_int(initial_state.split()) # list of 24 number
        # print(initial_state_list)
         # string of 24 number: e.g. Upper: xxx, Right: xxx
        last_state = initial_state
        last_state_list = initial_state_list
        actions = []
        states  = []
        for step in range(max_steps):
            current_state = last_state
            current_state_list = last_state_list
            current_state = getCube(current_state_list)
            allowed_actions = ["U", "U'", "U2", "R", "R'", "R2", "F", "F'", "F2"]
            # print(prefix_past)
            allowed_actions_ = [act for act in allowed_actions if act not in actions]

            if len(allowed_actions_) != 0:

            # epsilon greedy
                if np.random.rand() < self.epsilon and mode == "train":
                    action = random.choice(allowed_actions_)
                    action = action
                else:
                    # inputs = prompt.replace("<init_state>", current_state.lstrip())\
                    #     .replace("<goals>", goal).replace("<action>", previous_action.lstrip()).replace("<step>", str(step).strip()).strip()
                    inputs = prompt.format(init_state=current_state).strip()
                    # print(inputs)
                    input_ids = self.tokenizer.encode(inputs.lstrip() + "\n", return_tensors='pt').to(self.device)
                    prefix_output = self.model(input_ids[:, :-1], use_cache=True)
                    prefix_past = prefix_output.past_key_values

                    action_logits = []
                    for ac in allowed_actions_:
                        a = ac
                        action_ids = self.tokenizer.encode(a, add_special_tokens=False,return_tensors='pt').to(self.device)
                        input_ids_with_action = torch.cat([input_ids[:, -1:], action_ids], dim=-1)
                        # 计算每个动作的输出和对应的 logits
                        outputs = self.model(input_ids_with_action, past_key_values=prefix_past, use_cache=True)
                        logits = outputs.logits  # 获取对应于 action_ids 的 logits
                        # 计算 log softmax 来得到对数概率
                        total_log_prob = torch.zeros(1).cuda()
                        for i in range(1, input_ids_with_action.shape[-1]):
                            probs = torch.softmax(logits[:, i - 1, :], dim=-1)
                            for j in range(1):
                                total_log_prob[j] += torch.log(probs[j, input_ids_with_action[j, i]])
                        action_logits.append(total_log_prob) 
                        # sample from tempered policy
                    action_logits = torch.stack(action_logits) / pf_temp
                    # 计算概率分布
                    action_logits = action_logits.to(torch.float32)

                    probabilities = torch.exp(action_logits) / torch.sum(torch.exp(action_logits))

                    dist = Categorical(probs=probabilities.t())
                    # idx = torch.multinomial(probabilities, 1)[0]
                    idx = dist.sample()
                    # print(idx)
                    action = allowed_actions_[idx]
                
            else:
                action = random.choice(allowed_actions)
                states.append(last_state_list)

            actions.append(action)
            new_state_list = doAlgStr(last_state_list, action)

            last_state_list = new_state_list
            states.append(last_state_list)
            if isSolved(last_state_list):
                r1 = 100
                r1 = torch.tensor(r1).to(self.device)
                return None, actions, states, r1, None
        if isSolved(states[-1]):
            r1 = 100
        else:
            r1 = 1
        r1 = torch.tensor(r1).to(self.device)
        
        return None, actions, states, r1, None

    def local_search(self,
                            initial_state,
                            goal,
                            max_steps,
                            plan,
                            states, 
                            eos_token_id,
                            pf_temp=1.0,
                            mode="train",
                          ):
        """
        return: trajs, probability of each action in the trajs, log rewards of the trajs, log rewards of (state, action)
        """
        K = len(plan) // 2
        states = []
        actions = []
        if self.args.use_lora:
            base_to_lora(self.model)
        self.model.eval()
        prompt = cube_instruct
        # turn initial state into prompt form
        initial_state_list = convert_to_int(initial_state.split()) # list of 24 number
        # print(initial_state_list)
         # string of 24 number: e.g. Upper: xxx, Right: xxx
        last_state = initial_state
        last_state_list = initial_state_list

        for step in range(max_steps):
            current_state_list = last_state_list
            if step < K:
                action = plan[step]
            else:
                allowed_actions = ["U", "U'", "U2", "R", "R'", "R2", "F", "F'", "F2"]
                # print("allowed: ",allowed_actions)
                # print("actions", actions)
                allowed_actions_ = [act for act in allowed_actions if act not in actions]
                if len(allowed_actions_) != 0:
                    action = random.choice(allowed_actions_)
                else:
                    action = random.choice(allowed_actions)
                action = action
            actions.append(action)
        
            new_state = doAlgStr(current_state_list, action)

            last_state = new_state
            states.append(current_state_list)

            if isSolved(last_state):
                r1 = 100
           
                r1 = torch.tensor(r1).to(self.device)

                return None, actions, states, r1, None

        if isSolved(states[-1]):
            r1 = 100
        else:
            r1 = 1
        r1 = torch.tensor(r1).to(self.device)

        return None, actions, states, r1, None

    def forward_prob(self, actions, states):
        if self.args.use_lora:
            base_to_lora(self.model)
        # self.model.train()
        prompt = cube_instruct

        initial_state = states[0]

        last_state = initial_state
        log_pf = []
        log_bf = []
        for step in range(len(actions)):
            current_state = last_state
            allowed_actions = ["U", "U'", "U2", "R", "R'", "R2", "F", "F'", "F2"]
            inputs = prompt.format(init_state=current_state).strip()
            input_ids = self.tokenizer.encode(inputs.lstrip() + "\n", return_tensors='pt').to(self.device)
            action = actions[step]
            bsz = len(allowed_actions)  
            action_texts = allowed_actions
            action_ids = [self.tokenizer.encode(a, add_special_tokens=False, return_tensors='pt').to(self.device) for a in action_texts]
            max_length = max(len(aid[0]) for aid in action_ids)
            padded_action_ids = [torch.cat([aid, torch.full((1, max_length - len(aid[0])), self.tokenizer.pad_token_id, device=self.device)], dim=-1) for aid in action_ids]
            batch_input_ids_with_actions = torch.cat([torch.cat([input_ids, pid], dim=-1) for pid in padded_action_ids], dim=0)
            batch_outputs = self.model(batch_input_ids_with_actions, use_cache=True)
            batch_logits = batch_outputs.logits
            total_log_prob = torch.zeros(bsz).cuda()
            for i in range(input_ids.shape[-1], batch_input_ids_with_actions.shape[-1]):
                probs = torch.softmax(batch_logits[:, i - 1, :], dim=-1)
                for j in range(bsz):
                    if batch_input_ids_with_actions[j, i] != self.tokenizer.pad_token_id:
                        total_log_prob[j] += torch.log(probs[j, batch_input_ids_with_actions[j, i]])
            action_logits = total_log_prob
            probabilities = torch.exp(action_logits) / torch.sum(torch.exp(action_logits))

            idx = allowed_actions.index(action.capitalize())

            log_pf.append(torch.log(probabilities[idx]))
            
            if step < len(actions)-1:
                last_state = states[step+1]
            
            allowed_actions = ["U", "U'", "U2", "R", "R'", "R2", "F", "F'", "F2"]
            pb = torch.tensor(1 / len(allowed_actions))
            log_bf.append(torch.log(pb))
        return torch.stack(log_pf).sum(), torch.stack(log_bf).sum()

    def get_ll_reward(self, actions, states, goal):

        reward = []

        for idx in range(len(states)-1):
            state_pre = states[idx]
            state_post = states[idx+1]
            _, depth_pre = solveCube(state_pre)
            _, depth_post = solveCube(state_post)

            intuition = torch.exp(torch.tensor([depth_pre - depth_post]))
            reward.append(intuition)

        return torch.tensor(reward).to(self.device)


