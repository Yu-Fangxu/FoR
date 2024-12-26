import random
import sys
import numpy as np
import torch
from pytorch_lightning import LightningModule
from transformers.trainer_pt_utils import LabelSmoother
from util import lora_to_base, base_to_lora
from bw_utils import *
import yaml
import json
import bitsandbytes as bnb
from evaluate_blocksworld import Evaluate_BlocksWorld
import torch.nn.functional as F
import csv
import re
import pickle
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.distributions import Categorical
from collections import defaultdict
import openai
from openai import OpenAI
import logging
logging.getLogger("openai").setLevel(logging.CRITICAL)

API_KEY = "token-abc123"
API_BASE = "http://localhost:8000/v1"

def is_equal(answer, output):
    try:
        output = int(output)
        answer = int(answer)
        
        return output == answer
    except ValueError:
        pass
    try:
        output = float(output)
        answer = float(answer)
        return output == answer
    except ValueError:
        pass
    return output == answer

def extract_prompt(text, prompt_name):
    pattern = rf"{re.escape(prompt_name)}:\s*(.*?)\n"
    match = re.search(pattern, text)

    if match:
        return match.group(1)
    else:
        return ""

class GSM8KGFNTask(LightningModule):
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
        self.replay_buffer = replay_buffer
        self.train_data = train_data
        self.val_data = val_data
        self.n_samples = args.n_samples # 2 for step 4

        self.lr = args.lr
        self.logZ_lr = args.logZ_lr
        self.epsilon = self.args.epsilon_start
        self.get_lr_at_step = lambda step: min(step / 20 * self.lr, self.lr)
        self.ignore_token_id = LabelSmoother.ignore_index

        self.reward_temperature = self.args.reward_temp_start
        self.pf_temperature = self.args.pf_temp_start
        self.use_buffer_prob = self.args.use_buffer_prob
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_quant_type="nf4",
            llm_int8_threshold=6.0,
            bnb_4bit_use_double_quant=True,
        )
        self.client = OpenAI(
                    api_key=API_KEY,
                    base_url=API_BASE,
                    # timeout=300
                    )
        prompts='./prompts/interactive_examples.json'
        question_prompts='./prompts/useful_examples.json'
        with open(prompts) as f:
            self.prompts = json.load(f)
        with open(question_prompts) as f:
            self.question_prompts = json.load(f)

        self.test_correct = 0
        self.test_total_success = 0
        self.num_test = 0
        self.world_tokenizer = AutoTokenizer.from_pretrained(args.world_model, add_bos_token=False, padding_side='left')
        self.world_tokenizer.pad_token = self.world_tokenizer.eos_token
        transition_path = f"./transitions/{args.step}/transition.pkl"
        self.transitions = {}
        self.intermediate_reward_dict = {}

        self.traj = defaultdict(int)
        self.s_a = {}
        self.yes_no = self.tokenizer.encode('Yes No', add_special_tokens=False)
        
        
    def forward(self, problem, pf_temp):
        INIT, GOAL = problem
        GOAL = GOAL[0]
        INIT = INIT[0]

        (
            intermediate_reward, 
            actions, 
            states,
            reward, 
            answer
        ) = self.generate_trajectories(
            initial_state = INIT,
            goal = f'have that {GOAL}.',
            max_steps = self.args.step,
            eos_token_id=self.tokenizer.encode('\n', add_special_tokens=False)[0],
            pf_temp = pf_temp
        )
        return intermediate_reward, actions, states, answer, reward

    def training_step(self, problem, batch_idx):
        INIT, GOAL = problem
        # print(INIT)
        GOAL = GOAL[0]
        INIT = INIT[0]
        ########################## Compute the reward for ground-truth trajectory ##########################

        LOG_R = []
        LOG_PF = []
        LOG_BF = []

        if (
            random.random() < self.use_buffer_prob
            and self.replay_buffer.sample(self.n_samples, GOAL + INIT)[0] is not None
        ):
            # Using a sample from the reward buffer
            (log_reward_list,
            state_list,
            sample_list
            ) = self.replay_buffer.sample(
                self.n_samples, GOAL + INIT
            )

            for i, (state, sample) in enumerate(zip(state_list, sample_list)):
                (actions, states) = eval(state)
                log_pf, log_bf = self.forward_prob(GOAL, actions, states)
                if log_pf == -float('inf'):
                    continue
                LOG_PF.append(log_pf)
                LOG_R.append(log_reward_list[i])
                # LOG_BF.append(log_bf)
            # LOG_R.extend(log_reward_list)
            
        else:
            best_actions = None
            best_states = None
            best_reward = -9999
            for _ in range(self.n_samples):
                if np.random.rand() < self.args.pf_temp_prob:
                    pf_temp = self.pf_temperature
                else:
                    pf_temp = 1.0
                intermediate_reward, actions, states, answer, reward = self.forward(
                    problem, pf_temp
                )
                ll_weight = self.args.ll_weight

                generated_text = (actions, states)
                self.replay_buffer.add(GOAL + INIT, str(generated_text), answer, torch.log(reward + ll_weight * intermediate_reward.sum()))
                log_pf, _ = self.forward_prob(GOAL, actions, states)
                if log_pf == -float('inf'):
                    continue
                LOG_PF.append(log_pf)
                LOG_R.append(torch.log(reward + ll_weight * sum(intermediate_reward)))
                # LOG_BF.append(log_bf)

                actions_joined = '\n'.join(actions)
                self.traj[actions_joined] += 1

                if torch.log(reward + ll_weight * intermediate_reward.sum()) > best_reward:
                    best_actions = actions
                    best_states = states
                    best_reward = torch.log(reward + ll_weight * intermediate_reward.sum())

                # conduct local search
            for _ in range(0):
                intermediate_reward, actions, states, reward, answer = self.local_search(initial_state = INIT,
                    goal = GOAL,
                    max_steps = self.args.step,
                    plan=best_actions,  # use the highest to explore
                    states=best_states,
                    eos_token_id=self.tokenizer.encode('\n', add_special_tokens=False)[0],
                    pf_temp = pf_temp)
                log_reward = torch.log(reward + ll_weight * intermediate_reward.sum())
                if log_reward > best_reward:
                    generated_text = (actions, states)
                    self.replay_buffer.add(GOAL + INIT, str(generated_text), answer, torch.log(reward + ll_weight * intermediate_reward.sum()))
                    log_pf, _ = self.forward_prob(GOAL, actions, states)
                    if log_pf == -float('inf'):
                        continue
                    LOG_PF.append(log_pf)
                    LOG_R.append(torch.log(reward + ll_weight * intermediate_reward.sum()))
                    # LOG_BF.append(log_bf)
            
        # Obtain the log_pf and log_reward

        LOG_PF = torch.stack(LOG_PF).to(self.model.device)
        LOG_R = torch.stack(LOG_R).to(self.model.device)
        # LOG_BF = torch.stack(LOG_BF).to(self.model.device)
        LOG_R_temperd = LOG_R * self.reward_temperature
        if self.args.use_lora:
            base_to_lora(self.model)
    
        # Get the Trajectory balance loss
    
        loss = tb_loss(
            log_pf=LOG_PF,
            log_r=LOG_R_temperd,
            logz=self.logZ,
            log_bf=None,
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

        return loss
        

    def test_step(self, problem, batch_idx):
        # pass
        if self.args.use_lora:
            base_to_lora(self.model)    
        self.model.eval()      

        INIT, GOAL = problem
        GOAL = GOAL[0]
        INIT = INIT[0]
        total_success = 0
        total_solution = 0
        success_text = []

        for _ in range(20):
            (
            intermediate_reward, 
            actions, 
            states,
            reward, 
            answer
            ) = self.generate_trajectories(
                initial_state = INIT,
                goal = GOAL,
                max_steps = self.args.step,
                eos_token_id=self.tokenizer.encode('\n', add_special_tokens=False)[0],
                mode="test",
                pf_temp=0.5
            )

            predicted = states[-1].strip().split('\n')[-1]
            match = re.match(r'.*The answer is .*?([ $.0-9,\-]+).*\.$', predicted)
            if match is None:
                continue
            sub_answer = match[1].replace(',', '').replace('$', '').replace(' ', '')

            print(f"GOAL:{GOAL}; sub_answer={sub_answer}")
            if is_equal(answer=GOAL, output=sub_answer):
                total_success += 1
                solution = states[-1].split("The answer is")[-1].strip()
                if (INIT, solution, GOAL) not in success_text:
                    total_solution += 1
                    success_text.append((INIT, solution, GOAL))
                print(solution)
        with open(f'./success_plans_test/gfn/success_text_{batch_idx}.csv', mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(["Question", 'Solutions', "Answer"])
            writer.writerows(success_text)

        if total_success > 0:
            success = 1
        else:
            success = 0

        self.num_test += 1
        self.test_correct += success
        self.test_total_success += total_success
        print(f"avg acc: {self.test_total_success/(self.num_test*20)}")
        print(f"best20 acc: {self.test_correct/self.num_test}")

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
            base_to_lora(self.model)    
        self.model.eval()           

        INIT, GOAL = problem
        GOAL = GOAL[0]
        INIT = INIT[0]

        total_success = 0
        total_solution = 0
        success_text = []

        for _ in range(4):

            (
            intermediate_reward, 
            actions, 
            states,
            reward, 
            answer
            ) = self.generate_trajectories(
                initial_state = INIT,
                goal = f'have that {GOAL}.',
                max_steps = self.args.step,
                eos_token_id=self.tokenizer.encode('\n', add_special_tokens=False)[0],
                mode="test"
            )
            predicted = states[-1].strip().split('\n')[-1]
            match = re.match(r'.*The answer is .*?([ $.0-9,\-]+).*\.$', predicted)
            if match is None:
                continue
            sub_answer = match[1].replace(',', '').replace('$', '').replace(' ', '')

            if is_equal(GOAL, sub_answer):
                total_success += 1
                solution = states[-1].split("The answer is 9.\n\n")[-1].strip()
                if (INIT, solution, GOAL) not in success_text:
                    total_solution += 1
                    success_text.append((INIT, solution, GOAL))


        with open(f'./success_plans_valid/{self.args.step}_step/success_text_{batch_idx}.csv', mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(["Question", "Solutions", "Answer"])
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
        
        sorted_dict = sorted(self.traj.items(), key=lambda x: x[1], reverse=True)

    def on_train_batch_start(self, problem, batch_idx):
        pass

    def on_train_epoch_start(self):

        current_epoch = self.trainer.current_epoch
        if (current_epoch + 1) % 6 == 0:
            self.pf_temperature = self.args.pf_temp_start - (self.args.pf_temp_start - self.args.pf_temp_end) / (self.args.epochs // 6)

        if current_epoch < self.args.epochs // 2:
            self.epsilon = self.args.epsilon_start - (self.args.epsilon_start - self.args.epsilon_end) / (self.args.epochs // 2)
        
        if current_epoch < self.args.epochs // 2:
            self.reward_temperature = self.args.reward_temp_start + current_epoch * (self.args.reward_temp_end - self.args.reward_temp_start) / (self.args.epochs // 2)
        
        if current_epoch < self.args.epochs // 2:
            self.use_buffer_prob  = self.args.p_buffer_start + current_epoch * (self.args.p_buffer_end - self.args.p_buffer_start) / (self.args.epochs // 2)

        self.log("scheduled/R_temperature", self.reward_temperature, sync_dist=True)

        transition_path = f"./transitions/{self.args.step}/transition.pkl"
        with open(transition_path, 'wb') as f:
            pickle.dump(self.transitions, f)



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

    @torch.no_grad()
    def query_next_token(self, prompts):
        if self.args.use_lora:
            lora_to_base(self.model)
        if isinstance(prompts, str):
            prompts = [prompts]
        ret = []
        for prompt in prompts:
            tokens = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
            output = self.model.forward(tokens)
            ret.append(output.logits)
        outputs = torch.cat(ret, dim=0)
        filtered = outputs[:, -1, self.yes_no]
        dist = torch.softmax(filtered, dim=-1)
        return dist

    def generate_all_actions(self, question, state, depth):

        overall_question = re.match('.*((Calculate|calculate|how|How|what|What|Find|find|True or false).*)$', question)[1]
        overall_question = overall_question[0].upper() + overall_question[1:]
        subquestion_prefix = self.prompts["subquestion_prefix"].format(depth)
        # agent_input = state + subquestion_prefix
        agent_input = state
        overall_question_output = state + self.prompts["overall_question_prefix"].format(depth, overall_question)

        if depth == self.args.step:
            agent_output = [overall_question_output]
        else:
            agent_output = self.query_LM(self.model, self.world_tokenizer, prompt=(agent_input, f"{subquestion_prefix}"), do_sample=True, num_return_sequences=16, eos_token_id=self.tokenizer.encode('\n', add_special_tokens=False)[0],
                                         temperature=0.5)
        questions = [o.split(subquestion_prefix)[-1] for o in agent_output]
        
        r0 = torch.ones(len(questions)) * 0.1
        return agent_output, agent_output, r0

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
        question = initial_state
        prompt = self.prompts["input"] + self.prompts["question_prefix"] + question.strip() + "\n"
        input_question_prompts = self.question_prompts["input"] + self.question_prompts["subquestion_prefix"] + question.strip() + "\n"
        last_state = prompt
        actions = []
        states  = []
        intermediate_reward = []
        for step in range(max_steps):
            previous_action = ""
            current_state = last_state
            
            if last_state in self.s_a:
                # if s, a, s' have been observed
                agent_output, allowed_actions, r0 = self.s_a[last_state]
            else:
                agent_output, allowed_actions, r0 = self.generate_all_actions(question, last_state, step+1)
                self.s_a[last_state] = (agent_output, allowed_actions, r0)
            allowed_actions_ = [act for act in allowed_actions if act not in actions]

            if len(allowed_actions_) != 0:
                if self.args.use_lora:
                    base_to_lora(self.model)
            # epsilon greedy
                if np.random.rand() < self.epsilon and mode == "train":
                    action = random.choice(allowed_actions_)
                else:
                    inputs = last_state
                    input_ids = self.tokenizer.encode(inputs.lstrip() + "\n", return_tensors='pt').to(self.device)
                    
                    prefix_output = self.model(input_ids[:, :-1], use_cache=True)
                    prefix_past = prefix_output.past_key_values

                    action_logits = []
                    for ac in allowed_actions_:
                        a = ac
                        action_ids = self.tokenizer.encode(a, add_special_tokens=False,return_tensors='pt').to(self.device)
                        input_ids_with_action = torch.cat([input_ids[:, -1:], action_ids], dim=-1)
                        
                        outputs = self.model(input_ids_with_action, past_key_values=prefix_past, use_cache=True)
                        logits = outputs.logits 

                        total_log_prob = torch.zeros(1).cuda()
                        for i in range(1, input_ids_with_action.shape[-1]):
                            probs = torch.softmax(logits[:, i - 1, :], dim=-1)
                            for j in range(1):
                                total_log_prob[j] += torch.log(probs[j, input_ids_with_action[j, i]])
                        action_logits.append(total_log_prob) 
                        # sample from tempered policy
                    action_logits = torch.stack(action_logits) / pf_temp
                    action_logits = action_logits - torch.max(action_logits)  
                    action_logits = action_logits.to(torch.float32)
                    
                    probabilities = torch.exp(action_logits) / torch.sum(torch.exp(action_logits))
                    if torch.isnan(probabilities).any():
                        probabilities = torch.empty_like(probabilities).uniform_(-1.0, 1.0)
                        probabilities = torch.exp(probabilities) / torch.sum(torch.exp(probabilities))
                    dist = Categorical(probs=probabilities.t())
                    # idx = torch.multinomial(probabilities, 1)[0]
                    idx = dist.sample()
                    # print(idx)
                    action = allowed_actions_[idx]
                
            else:
                action = random.choice(allowed_actions)
            idx = allowed_actions.index(action) 
            intermediate_reward.append(r0[idx])
            states.append(last_state)
            actions.append(action)
            last_action = action

            last_state += action.strip()
            last_state += "\n"
            world_input = last_state
            
            answer_dict = defaultdict(lambda: [])
            
            if (last_state, last_action) in self.transitions:
                answer_list = []
                # if s, a, s' have been observed
                world_output = self.transitions[(last_state, last_action)]
                for output in world_output:
                    result = output.strip().split('\n')[-1]
                    match = re.match(r'.*The answer is .*?([ $.0-9,\-]+).*\.$', result)
                    if match is None:
                        continue
                    sub_answer = match[1].replace(',', '').replace('$', '').replace(' ', '')
                    answer_dict[sub_answer].append(output)
                    answer_list.append(sub_answer)
                sorted_answer_dict = sorted(answer_dict.items(), key=lambda p: len(p[1]), reverse=True)
            else:
                answer_list = []
                sampled = 0
                n_sample_confidence = 8
                speedup_confidence_batch_size = n_sample_confidence
                while sampled < n_sample_confidence:
                    world_output = self.query_LM(self.model, self.world_tokenizer, (world_input, f"{self.prompts['answer_prefix'].format(step+1)}"), do_sample=False, num_return_sequences=1,
                                    eos_token_id=eos_token_id, temperature=0)
                    sampled += speedup_confidence_batch_size
                    for output in world_output:
                        result = output.strip().split('\n')[-1]
                        match = re.match(r'.*The answer is .*?([ $.0-9,\-]+).*\.$', result)
                        if match is None:
                            continue
                        sub_answer = match[1].replace(',', '').replace('$', '').replace(' ', '')
                        answer_dict[sub_answer].append(output)
                        answer_list.append(sub_answer)
                    if len(answer_dict) == 0:
                        continue
                    
                    sorted_answer_dict = sorted(answer_dict.items(), key=lambda p: len(p[1]), reverse=True)
                    
                    max_len = len(sorted_answer_dict[0][1])
                    if max_len < 2:
                        continue
                    if len(sorted_answer_dict) < 2:
                        break
                    second_max_len = len(sorted_answer_dict[1][1])
                    if max_len >= len(answer_dict) / 2 and max_len > second_max_len:
                        break
                
            new_state = world_input + world_output[0] + "\n"
            self.transitions[(last_state, last_action)] = world_output
            last_state = new_state

            # answer = sorted_answer_dict[0][1][0]  # [0]: maximum; [1]: list of outputs; [0]: first output in the list
            try:
                if sorted_answer_dict[0][0]:
                    pass  
            except Exception:  
                continue  
            answer = sorted_answer_dict[0][0]
            if is_equal(goal, answer):
                states.append(last_state)
                r1 = 100
                r1 = torch.tensor(r1).to(self.device)
                intermediate_reward = torch.tensor(intermediate_reward)
                return intermediate_reward, actions, states, r1, answer
        states.append(last_state)
        if len(answer_dict) == 0:
            r1 = 1e-4
            answer = None
        else:
            answer = sorted_answer_dict[0][0]
            if answer in goal:
                r1 = 100
            else:
                r1 = 1e-4


        r1 = torch.tensor(r1).to(self.device)
        intermediate_reward = torch.tensor(intermediate_reward)
        return intermediate_reward, actions, states, r1, answer

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
        K = self.args.step // 2
        # last_state = states[K]
        # actions = actions[:K]
        # states = states[:(K-1)]
        states = []
        actions = []
        if self.args.use_lora:
            base_to_lora(self.model)
        self.model.eval()
        question = initial_state
        # prompt = sample_prompt(self.init_prompt, shuffle_prompt=False, num_shot=1)
        prompt = self.prompts["input"] + self.prompts["question_prefix"] + question.strip() + "\n"
        # input_question_prompts = self.question_prompts["input"] + self.question_prompts["subquestion_prefix"] + question.strip() + "\n"
        last_state = prompt
        intermediate_reward = []
        
        for step in range(max_steps):
            # epsilon greedy
            # print(last_state)
            print(self.s_a)
            _, allowed_actions, r0 = self.s_a[last_state]
            if step < K:
                action = plan[step]
            else:
                # print("allowed: ",allowed_actions)
                # print("actions", actions)
                allowed_actions_ = [act for act in allowed_actions if act not in actions]
                if len(allowed_actions_) != 0:
                    action = random.choice(allowed_actions_)
                else:
                    action = random.choice(allowed_actions)
                action = action
            idx = allowed_actions.index(action) 
            intermediate_reward.append(r0[idx])
            states.append(last_state)
            last_action = action

            last_state += action.strip()
            last_state += "\n"
            
            # world_input = last_state  + self.prompts['answer_prefix'].format(step+1)
            

            actions.append(action)
            
            last_action = action
            world_input = last_state
            
            answer_dict = defaultdict(lambda: [])
            
            if (last_state, last_action) in self.transitions:
                answer_list = []
                # if s, a, s' have been observed
                world_output = self.transitions[(last_state, last_action)]
                for output in world_output:
                    result = output.strip().split('\n')[-1]
                    match = re.match(r'.*The answer is .*?([ $.0-9,\-]+).*\.$', result)
                    if match is None:
                        continue
                    sub_answer = match[1].replace(',', '').replace('$', '').replace(' ', '')
                    answer_dict[sub_answer].append(output)
                    answer_list.append(sub_answer)
                sorted_answer_dict = sorted(answer_dict.items(), key=lambda p: len(p[1]), reverse=True)

            else:
                # if s, a, s' have not been observed, use World Model to predict the state and store it.
                # lora_to_base(self.model)
                answer_list = []
                sampled = 0
                n_sample_confidence = 8
                speedup_confidence_batch_size = n_sample_confidence
                while sampled < n_sample_confidence:
                    world_output = self.query_LM(self.model, self.world_tokenizer, (world_input, f"{self.prompts['answer_prefix'].format(step+1)}"), do_sample=False, num_return_sequences=1,
                                    eos_token_id=eos_token_id, temperature=0)
                    sampled += speedup_confidence_batch_size
                    for output in world_output:
                        result = output.strip().split('\n')[-1]
                        match = re.match(r'.*The answer is .*?([ $.0-9,\-]+).*\.$', result)
                        # print("asnwer:", result)
                        # print("ground_truth\n", goal)
                        if match is None:
                            continue
                        sub_answer = match[1].replace(',', '').replace('$', '').replace(' ', '')
                        answer_dict[sub_answer].append(output)
                        answer_list.append(sub_answer)
                    # print(result)
                    if len(answer_dict) == 0:
                        continue
                    
                    sorted_answer_dict = sorted(answer_dict.items(), key=lambda p: len(p[1]), reverse=True)
                    
                    max_len = len(sorted_answer_dict[0][1])
                    if max_len < 2:
                        continue
                    if len(sorted_answer_dict) < 2:
                        break
                    second_max_len = len(sorted_answer_dict[1][1])
                    if max_len >= len(answer_dict) / 2 and max_len > second_max_len:
                        break
            new_state = world_input + world_output[0] + "\n"
            last_state = new_state
        
            try:
                if sorted_answer_dict[0][0]:  
                    pass  
            except Exception:  
                continue  
            answer = sorted_answer_dict[0][0]
            # print("asnwer:", result)
            # print("ground_truth\n", goal)
            if answer in goal:
                states.append(last_state)
                r1 = 100
                r1 = torch.tensor(r1).to(self.device)
                intermediate_reward = torch.tensor(intermediate_reward)
                return intermediate_reward, actions, states, r1, answer
        states.append(last_state)
        # print(last_state)
        if len(answer_dict) == 0:
            r1 = 1e-4
            answer = None
        else:
            # answer = sorted_answer_dict[0][1][0]  # [0]: maximum; [1]: list of outputs; [0]: first output in the list
            answer = sorted_answer_dict[0][0]
            # print("asnwer:", result)
            # print("ground_truth\n", goal)
            if answer in goal:
                r1 = 100
            else:
                r1 = 1e-4

        r1 = torch.tensor(r1).to(self.device)
        intermediate_reward = torch.tensor(intermediate_reward)
        return intermediate_reward, actions, states, r1, answer

    def forward_prob(self, goal, actions, states):
        if self.args.use_lora:
            base_to_lora(self.model)
        initial_state = states[0]

        last_state = initial_state
        log_pf = []
        log_bf = []
        for step in range(len(actions)):
            previous_action = ""
            current_state = last_state
            _, allowed_actions, _ = self.s_a[last_state]

            inputs = last_state

            input_ids = self.tokenizer.encode(inputs.lstrip() + "\n", return_tensors='pt').to(self.device)
            
            action = actions[step]
            bsz = len(allowed_actions)  
            action_texts = [ac for ac in allowed_actions]
            action_ids = [self.tokenizer.encode(a, add_special_tokens=False, return_tensors='pt').to(self.device) for a in action_texts]
            # action_ids = self.tokenizer(allowed_actions)
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
            action_logits = total_log_prob.to(torch.float32)
            action_logits = action_logits - torch.max(action_logits)
            probabilities = torch.exp(action_logits) / torch.sum(torch.exp(action_logits))

            idx = allowed_actions.index(action)

            log_pf.append(torch.log(probabilities[idx]))
            
            if step < len(actions)-1:
                last_state = states[step+1]
        return torch.stack(log_pf).mean(), None

    def get_intermediate_reward(self, actions, states, goal):

        reward = []

        prompt = sample_prompt(self.init_prompt, shuffle_prompt=False, num_shot=4)
        for step_idx, (state, action) in enumerate(zip(states, actions)):
            icl_template = prompt["icl_list"][step_idx // 2]
            if step_idx == 0:
                previous_action = ""
                current_state = state
            else:
                previous_action = actions[step_idx-1] + "\n"
                current_state = states[step_idx-1]
            inputs = icl_template.replace("<init_state>", current_state.lstrip())\
                .replace("<goals>", goal).replace("<action>", previous_action.lstrip())
            # print(inputs + action)
            intuition = self.get_likelihood(inputs, [inputs + action.lstrip()])[0]
            self.intermediate_reward_dict[(step_idx, state, action, goal)] = intuition
            reward.append(intuition)

        return torch.tensor(reward).to(self.device)

    def get_likelihood(
            self,
            prefix: str,
            contents: list[str],
    ):
        lora_to_base(self.model)
        bsz = len(contents)
        prefix_tokens = self.world_tokenizer.encode(prefix, add_special_tokens=True)
        prompts_tokens = [self.world_tokenizer.encode(x, add_special_tokens=True) for x in contents]

        for prompt_tokens in prompts_tokens:
            assert prompt_tokens[: len(prefix_tokens)] == prefix_tokens

        max_prompt_size = max([len(t) for t in prompts_tokens])
        total_len = max_prompt_size
        tokens = torch.full((bsz, total_len), self.world_tokenizer.pad_token_id).cuda().long()

        for k, t in enumerate(prompts_tokens):
            tokens[k, : len(t)] = torch.tensor(t)[:2048].long()

        with torch.no_grad():
            # outputs = self.world_model(tokens)
            outputs = self.model(tokens)
            logits = outputs.logits
        acc_probs = torch.zeros(bsz).cuda()
        for i in range(len(prefix_tokens), max_prompt_size):
            probs = torch.softmax(logits[:, i - 1, :], dim=-1)
            for j in range(bsz):
                if tokens[j, i] != self.world_tokenizer.pad_token_id:
                    acc_probs[j] += torch.log(probs[j, tokens[j, i]])
        return acc_probs
      
    def query_LM(self, worldmodel, tokenizer, prompt, eos_token_id, num_return_sequences=1, do_sample=True, temperature=1.0):
        all_results = []
        response = self.client.completions.create(
                   model='meta-llama/Llama-3-8B',
            prompt=prompt[0] + "\n" + prompt[1],
            n=num_return_sequences,
            max_tokens=150,
            temperature=temperature,
        )
        completions = response.choices

        for completion in completions:
            raw = completion.text
            raw = raw.split("\n")[0]
            result = prompt[1] + raw
            all_results.append(result)
        return all_results


    def is_terminal_question(self, prompt, prompt_index):
        prompt = prompt.split('\n\n')[-1]
        if 'Now we can answer' in prompt:
            return True
        question = prompt.split('\n')[0]
        if f'Question {prompt_index}.' not in prompt:
            return False
        last_sub = prompt.split(f'Question {prompt_index}.')[-1].split('\n')[0]
        if last_sub in question:
            return True
        return False
