import random
import numpy as np
import torch
from pytorch_lightning import LightningModule
from transformers.trainer_pt_utils import LabelSmoother
from util import base_to_lora, tb_loss

from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.distributions import Categorical

import Task
from util_trans import *
import Utils
from utils_arc import *
from arc_prompt import arc_instruct
# Generate a map of all callable functions in the Utils module

func_map = {func_name: getattr(Utils, func_name) for func_name in dir(Utils) if callable(getattr(Utils, func_name))}
ast = lambda g: tuple(tuple(r) for r in g)

def convert_tensors_to_numbers(data):
    for split in ['train', 'test']:
        for entry in data[split]:
            entry['input'] = [[t.item() for t in tensor_list] for tensor_list in entry['input']]
            entry['output'] = [[t.item() for t in tensor_list] for tensor_list in entry['output']]
    return data


def is_terminal(state: ARCState) -> bool:
    if state.candidate.score == 0:
        return True
    else:
        return False

def update_state(state: ARCState, action: ARCAction):
    startOps = ("switchColors", "cropShape", "cropAllBackground", "minimize", \
                "maxColorFromCell", "deleteShapes", "replicateShapes","colorByPixels", \
                "paintGridLikeBackground") # applyEvolve?
    repeatIfPerfect = ("extendColor", "moveAllShapes")
    firstIt = True
    state = copy.deepcopy(state)
    t, c, cTask = state.task, state.candidate, state.ctask

    for s in range(t.nTrain):
        cTask["train"][s]["input"] = action(c.t.trainSamples[s].inMatrix).tolist()
        if c.t.sameIOShapes and len(c.t.fixedColors) != 0:
            cTask["train"][s]["input"] = Utils.correctFixedColors(\
                    c.t.trainSamples[s].inMatrix.m,\
                    np.array(cTask["train"][s]["input"]),\
                    c.t.fixedColors, c.t.commonOnlyChangedInColors).tolist()
    
    newPredictions = []
    for s in range(t.nTest):
        newOutput = action(c.t.testSamples[s].inMatrix)
        newPredictions.append(newOutput)
        cTask["test"][s]["input"] = newOutput.tolist()
        if c.t.sameIOShapes and len(c.t.fixedColors) != 0:
            cTask["test"][s]["input"] = Utils.correctFixedColors(\
                    c.t.testSamples[s].inMatrix.m,\
                    np.array(cTask["test"][s]["input"]),\
                    c.t.fixedColors, c.t.commonOnlyChangedInColors).tolist()

    cScore = sum([Utils.incorrectPixels(np.array(cTask["train"][s]["input"]), \
                                        t.trainSamples[s].outMatrix.m) for s in range(t.nTrain)])
    changedPixels = sum([Utils.incorrectPixels(c.t.trainSamples[s].inMatrix.m, \
                                                np.array(cTask["train"][s]["input"])) for s in range(t.nTrain)])
    step_idx = state.step_idx
    newCandidate = Candidate(c.ops+[action], c.tasks + [copy.deepcopy(cTask)], cScore, predictions=newPredictions)

    # if firstIt and str(action)[28:60].startswith(startOps):
    #     if all([np.array_equal(np.array(cTask["train"][s]["input"]), t.trainSamples[s].inMatrix.m) for s in range(t.nTrain)]):
    #     continue
    #     newCandidate.generateTask()
    #     tryOperations(t, newCandidate, cTask, b2c)
    # elif str(action)[28:60].startswith(repeatIfPerfect) and c.score - changedPixels == cScore and changedPixels != 0:
    #     newCandidate.generateTask()
    #     tryOperations(t, newCandidate, cTask, b2c)


    newCandidate.generateTask()
    state = ARCState(step_idx=step_idx+1,  task=t, originalT=state.originalT, candidate=newCandidate, ctask=cTask)
    
    success = is_terminal(state)


    return state, success



def init_state(example) -> ARCState:
    """Initialize the world model.

    :return: the initial state
    """
    originalT, task = example
    taskId = originalT.index
    taskNeedsRecoloring = needsRecoloring(originalT)

    if taskNeedsRecoloring:
        task, trainRels, trainInvRels, testRels, testInvRels = orderTaskColors(originalT)
        t = Task.Task(task, taskId, submission=False)
    else:
        t = originalT
    cTask = copy.deepcopy(task)

    if t.sameIOShapes:
        taskNeedsCropping = needsCropping(t)
    else:
        taskNeedsCropping = False
    if taskNeedsCropping:
        cropPositions, backgrounds = cropTask(t, cTask)
        t2 = Task.Task(cTask, taskId, submission=False, backgrounds=backgrounds)
    elif t.hasUnchangedGrid:
        if t.gridCellsHaveOneColor:
            ignoreGrid(t, cTask) # This modifies cTask, ignoring the grid
            t2 = Task.Task(cTask, taskId, submission=False)
        elif t.outGridCellsHaveOneColor:
            ignoreGrid(t, cTask, inMatrix=False)
            t2 = Task.Task(cTask, taskId, submission=False)
        else:
            t2 = t
    elif t.hasUnchangedAsymmetricGrid and t.assymmetricGridCellsHaveOneColor:
        ignoreAsymmetricGrid(t, cTask)
        t2 = Task.Task(cTask, taskId, submission=False)
    #if t.hasUnchangedGrid:
    #    ignoreGeneralGrid(t, cTask)
    #    t2 = Task(cTask, taskId, submission=False)
    #if t.hasUnchangedAsymmetricGrid:
    #    ignoreGeneralAsymmetricGrid(t, cTask)
    #    t2 = Task(cTask, taskId, submission=False)
    else:
        t2 = t

    if t2.sameIOShapes:
        hasRotated = False
        if t2.hasOneFullBorder:
            hasRotated, rotateParams = rotateTaskWithOneBorder(t2, cTask)
        elif t2.requiresHVRotation:
            hasRotated, rotateParams = rotateHVTask(t2, cTask)
        if hasRotated!=False:
            cTask = hasRotated.copy()
            t2 = Task.Task(cTask, taskId, submission=False)

    cScore = sum([Utils.incorrectPixels(np.array(cTask["train"][s]["input"]), \
                                        t2.trainSamples[s].outMatrix.m) for s in range(t.nTrain)])
    
    dummyPredictions = [sample.inMatrix.m for sample in t2.testSamples]
    c = Candidate([], [task], score=cScore, predictions=dummyPredictions)
    c.t = t2
    b1c = Best1Candidates(c)

    # Generate the three candidates with best possible score
    prevScore = sum([c.score for c in b1c.candidates])
    firstIt = True
    
    return ARCState(step_idx=0, task=t2, originalT=originalT, candidate=c, ctask=cTask)

def different_views(task):
    object_view = False # total 16 possibilities
    if object_view:
        # Toggle on or off - 2 possibilities each = total 4 possibilities
        diag = False
        multicolor = False
        
        # choose one of the three (or none) - 4 possibilities
        by_row = False
        by_col = False
        by_color = False
        
        # if context length allows (We want this to be true as much as possible)
        more_info = True

    pixel_view = False # total 1 possibility

    ## Other parameters
    hide_grid = False # hides original input grid, only set to True if there are not enough tokens
    missing_color = False # if we want to view no color as a color 'j' - Not used this time

    # No need to change this - this just gives all the various expert types
    expert_types = ''
    if object_view: 
        expert_types += 'obj'
        if diag: expert_types += 'diag'
        if multicolor: expert_types += 'multi'
        if by_color: expert_types += 'by_color'
        if by_row: expert_types += 'by_row'
        if by_col: expert_types += 'by_col'
        if more_info: expert_types += 'moreinfo'
    if pixel_view: expert_types += 'pixel'
        # convert to string representation
    for each in task:
        for field in ['input', 'output']:
            if field in each: 
                # each[field] = rotate_90_clockwise(each[field])
                each[field] = array_to_string(each[field])
                # if we want to treat no cells as a color
                if missing_color:
                    each[field] = replace(each[field],[['.']],[['j']])
        each['input_grid_size'] = get_size(each['input'])

        # objects_diag = get_objects(each['input'], diag=True, by_color = by_color, by_row = by_row, by_col = by_col, multicolor = multicolor, more_info = more_info)
        # objects = get_objects(each['input'], diag=False, by_color = by_color, by_row = by_row, by_col = by_col, multicolor = multicolor, more_info = more_info)

        # if len(objects) > 8:
        #     if len(objects_diag) < len(objects) / 2:
        #         object_view = True
        #         diag = True
        #     else:
        #         object_view = False
        
        # else:
        #     object_view = True

        if object_view: each['input_objects'] = get_objects(each['input'], diag=diag, by_color = by_color, by_row = by_row, by_col = by_col, multicolor = multicolor, more_info = more_info)
        if pixel_view: each['input_pixel_coords'] = get_pixel_coords(each['input'])
        each['output_grid_size'] = get_size(each['output'])
        if object_view: each['output_objects'] = get_objects(each['output'], diag=diag, by_color = by_color, by_row = by_row, by_col = by_col, multicolor = multicolor, more_info = more_info)
        if pixel_view: each['output_pixel_coords'] = get_pixel_coords(each['output'])
        if 'code_output' in each:
            each['code_output_grid_size'] = get_size(each['code_output'])
            if object_view: each['code_output_objects'] = get_objects(each['code_output'], diag=diag, by_color = by_color, by_row = by_row, by_col = by_col, multicolor = multicolor, more_info = more_info)
            if pixel_view: each['code_output_pixel_coords'] = get_pixel_coords(each['code_output'])
        
        if get_size(each['input']) == get_size(each['output']):
            difference_view = True
        else:
            difference_view = False
        difference_view = False
        if difference_view: each['input_output_difference'] = get_difference_coords(each['input'], each['output'])
    
    return task

def eval_output(program, example):
        # test_case = example['test']
    test_case = [s.inMatrix for s in example.testSamples]
    gt_case = [s.outMatrix.m for s in example.testSamples]
    for input, output in zip(test_case, gt_case):
        # print("case")
        for func in program:
            # print(func)
            try:
                input = func(input)
            except Exception as e:
                return 0
        input = np.array(input)
        if np.array_equal(input, output) == False:
            return 0
    return 1


def construct_prompt(example):
        # test_case = example['test']
    t, c, cTask = example.task, example.candidate, example.ctask
    train_case = [cTask["train"][s]["input"] for s in range(t.nTrain)]
    gt_case = [t.trainSamples[s].outMatrix.m for s in range(t.nTrain)]

    training_data = []

    for input, output in zip(train_case, gt_case):
        # {"input": [[0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0]], "output": [[0, 0, 0, 7, 7, 7, 7, 7, 7, 7, 7, 7, 0, 0, 0, 0]]}
        training_data.append({"input": input, "output": output})

    prompt = different_views(training_data)
    return str(prompt).replace(', ',',').replace(' ','')

class ARCGFNTask(LightningModule):
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
        # self.prompts = json.load(open("data/1D-ARC/my_mcts_prompts_update.json", "r"))
        self.replay_buffer = replay_buffer
        self.train_data = train_data
        self.val_data = val_data
        self.n_samples = args.n_samples  # 2 for step 4
        # with open("data/1D-ARC/bw_config.yaml", "r") as file:
        #     self.config = yaml.safe_load(file)
        # self.domain_pddl = f'gpt-plan-benchmark/gpt_plan_test/instances/{self.config["domain_file"]}'  # TO CHANGE

        self.lr = args.lr
        self.logZ_lr = args.logZ_lr
        self.epsilon = self.args.epsilon_start
        self.get_lr_at_step = lambda step: min(step / 20 * self.lr, self.lr)

        self.ignore_token_id = LabelSmoother.ignore_index

        self.reward_temperature = self.args.reward_temp_start
        self.pf_temperature = self.args.pf_temp_start
        self.use_buffer_prob = self.args.use_buffer_prob

        
    def forward(self, problem, pf_temp):
        # originalT, task = problem
        # goal: success on test
        # state: 
        grids, taskId = problem
        
        # grids = convert_tensors_to_numbers(grids)
        
        originalT = Task.Task(grids, taskId, submission=False)
        (generated_text, actions, states, reward, sample) = self.generate_trajectories(
            initial_state=[originalT, grids],
            goal=None,
            max_steps=self.args.step,
            eos_token_id=self.tokenizer.encode("\n", add_special_tokens=False)[0],
            pf_temp=pf_temp,
        )
        return generated_text, actions, states, sample, reward

    def training_step(self, problem, batch_idx):
        grids, taskId = problem
        grids = convert_tensors_to_numbers(grids)
        
        originalT = Task.Task(grids, taskId, submission=False)
        # goal: success on test
        # state: 
        # originalT = eval(originalT[0])
        # task = eval(task[0])
        ########################## Compute the reward for ground-truth trajectory ##########################

        LOG_R = []
        LOG_PF = []
        LOG_BF = []
        # Exploitation: Reuse the samples in the buffer

        if (
            random.random() < self.use_buffer_prob
            and self.replay_buffer.sample(self.n_samples, str(grids))[0] is not None
        ):
            # Using a sample from the reward buffer
            (log_reward_list, state_list, sample_list) = self.replay_buffer.sample(
                self.n_samples, str(grids)
            )
            
            for state, sample in zip(state_list, sample_list):
                (actions, states) = eval(state)
                log_pf, log_bf = self.forward_prob(actions, states)
                LOG_PF.append(log_pf)
                LOG_BF.append(log_bf)
            LOG_R.extend(log_reward_list)

        else:
            best_actions = None
            best_states = None
            # larger reward than original ones
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

                LOG_R.append(torch.log(reward + ll_weight * ll_reward.sum()))
                # print(f"three losses, {reward}, {ll_weight}, {ll_reward.sum()}")
                generated_text = (actions, states)
                # pickled_data = p(ckle.dumps(generated_text)
                # encoded_data = base64.b64encode(pickled_data).decode('utf-8')  # Convert to Base64 string

                # generated_text_str = json.dumps(generated_text, cls=CustomEncoder)
                self.replay_buffer.add(
                    str(grids),
                    str(generated_text),
                    sample,
                    torch.log(reward + ll_weight * ll_reward.sum()),
                )
                log_pf, log_bf = self.forward_prob(actions, states)
                LOG_PF.append(log_pf)
                LOG_BF.append(log_bf)

                if torch.log(reward + ll_weight * ll_reward.sum()) > best_reward:
                    best_actions = actions
                    best_states = states
                    best_reward = torch.log(reward + ll_weight * ll_reward.sum())
                # print("beststates", best_states)
                # conduct local search
                # reduce to 8 or 4
            for _ in range(4):
                _, actions, states, reward, _ = self.local_search(
                    initial_state=[originalT, grids],
                    goal=None,
                    max_steps=self.args.step,
                    plan=best_actions,
                    states=best_states,
                    eos_token_id=self.tokenizer.encode("\n", add_special_tokens=False)[
                        0
                    ],
                    pf_temp=pf_temp,
                )

                if self.args.ll_weight == 0:
                    ll_reward = [1 for _ in range(self.args.step)]
                    ll_reward = torch.tensor(ll_reward).to(self.device)
                    ll_weight = 1
                else:
                    ll_reward = self.get_ll_reward(actions, states, None)
                    ll_reward = -1 / ll_reward
                    ll_weight = self.args.ll_weight

                log_reward = torch.log(reward + ll_weight * ll_reward.sum())

                # if log_reward is larger, then we accept it

                if log_reward > best_reward:
                    LOG_R.append(torch.log(reward + ll_weight * ll_reward.sum()))
                    generated_text = (actions, states)
                    # pickled_data = pickle.dumps(generated_text)
                    # encoded_data = base64.b64encode(pickled_data).decode('utf-8') 
                    # prob of using replay buffer to 0, main training framework to run correctly.
                    self.replay_buffer.add(
                        str(grids),
                        str(generated_text),
                        sample,
                        torch.log(reward + ll_weight * ll_reward.sum()),
                    )
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
            log_bf=None,
            logpartition=True,
        )
        print(f"Training loss {loss}")

        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            prog_bar=True,
            batch_size=self.args.batch_size,
        )
        self.log(
            "train/logR",
            LOG_R.mean(),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            prog_bar=True,
            batch_size=self.args.batch_size,
        )

        return loss

    def test_step(self, problem, batch_idx):
        # pass
        if self.args.use_lora:
            base_to_lora(self.model)  # 确保转换成lora
        self.model.eval()  # 必须用eval

        grids, taskId = problem
        grids = convert_tensors_to_numbers(grids)
        
        originalT = Task.Task(grids, taskId, submission=False)
        total_success = 0
        total_solution = 0
        success_text = []

        for _ in range(40):
            (generated_text, actions, states, reward, sample) = (
                self.generate_trajectories(
                    initial_state=[originalT, grids],
                    goal=None,
                    max_steps=self.args.step,
                    eos_token_id=self.tokenizer.encode("\n", add_special_tokens=False)[
                        0
                    ],
                    mode="test",
                )
            )
            test_success = eval_output(actions, originalT)
            if test_success:
                total_success += 1
                if grids not in success_text:
                    total_solution += 1
                    success_text.append(grids)
            # input and output grid states[-1] matches the target state
            # goal_statement = f"{GOAL}"
            # print("test_jalend", GOAL, states[-1])
            # if(GOAL == states[-1]):
            #     total_success += 1
            #     total_solution += 1

        if total_success > 0:
            success = 1
        else:
            success = 0
        print(total_success)

        self.log(
            "test/success",
            success,
            sync_dist=True,
            prog_bar=True,
            batch_size=self.args.batch_size,
        )
        self.log(
            "test/n_solutsion",
            total_solution,
            sync_dist=True,
            prog_bar=True,
            batch_size=self.args.batch_size,
        )

    def validation_step(self, problem, batch_idx):
        print("Jalend:started validation")
        if self.args.use_lora:
            base_to_lora(self.model)  # 确保转换成lora
        self.model.eval()  # 必须用eval

        grids, taskId = problem
        grids = convert_tensors_to_numbers(grids)
        
        originalT = Task.Task(grids, taskId, submission=False)

        total_success = 0
        total_solution = 0
        success_text = []

        for _ in range(20):

            (generated_text, actions, states, reward, sample) = (
                self.generate_trajectories(
                    initial_state=[originalT, grids],
                    goal=None,
                    max_steps=self.args.step,
                    eos_token_id=self.tokenizer.encode("\n", add_special_tokens=False)[
                        0
                    ],
                    mode="test",
                )
            )
            test_success = eval_output(actions, originalT)
            if test_success:
                total_success += 1
                if grids not in success_text:
                    total_solution += 1
                    success_text.append(grids)
          # input and output grid states[-1] matches the target state
            # goal_statement = f"{GOAL}"
            # print("validation_jalend", GOAL, states[-1])
            # if(GOAL == states[-1]):
            #     total_success += 1
            #     total_solution += 1
            
        if total_success > 0:
            success = 1
        else:
            success = 0

        self.log(
            "val/success",
            success,
            sync_dist=True,
            prog_bar=True,
            batch_size=self.args.batch_size,
        )
        self.log(
            "val/n_solutsion",
            total_solution,
            sync_dist=True,
            prog_bar=True,
            batch_size=self.args.batch_size,
        )

    def on_train_batch_start(self, problem, batch_idx):
        pass

    def on_train_epoch_start(self):
        # Log scheduled quantities
        current_epoch = self.trainer.current_epoch
        if (current_epoch + 1) % 6 == 0:
            self.pf_temperature = self.args.pf_temp_start - (
                self.args.pf_temp_start - self.args.pf_temp_end
            ) / (self.args.epochs // 6)

        if current_epoch < self.args.epochs // 2:
            self.epsilon = self.args.epsilon_start - (
                self.args.epsilon_start - self.args.epsilon_end
            ) / (self.args.epochs // 2)

        if current_epoch < self.args.epochs // 2:
            self.reward_temperature = self.args.reward_temp_start + current_epoch * (
                self.args.reward_temp_end - self.args.reward_temp_start
            ) / (self.args.epochs // 2)

        if current_epoch < self.args.epochs // 2:
            self.use_buffer_prob = self.args.p_buffer_start + current_epoch * (
                self.args.p_buffer_end - self.args.p_buffer_start
            ) / (self.args.epochs // 2)

        self.log("scheduled/R_temperature", self.reward_temperature, sync_dist=True)

    def configure_optimizers(self):
        if self.args.use_4bit:
            import bitsandbytes as bnb  # fmt: skip
            optimizer = bnb.optim.PagedAdamW8bit(
                [
                    {"params": self.model.parameters(), "lr": self.lr},
                    {
                        "params": [
                            self.logZ,
                        ],
                        "lr": self.logZ_lr,
                    },
                ]
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": CosineAnnealingLR(optimizer, T_max=10, eta_min=5e-5),
                    "monitor": "metric_to_track",
                    "frequency": 10,
                    # If "monitor" references validation metrics, then "frequency" should be set to a
                    # multiple of "trainer.check_val_every_n_epoch".
                },
            }
        else:
            return torch.optim.AdamW(
                [
                    {"params": self.model.parameters(), "lr": self.lr},
                    {
                        "params": [
                            self.logZ,
                        ],
                        "lr": self.logZ_lr,
                    },
                ]
            )

    def generate_trajectories(
        self,
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
        startOps = ("switchColors", "cropShape", "cropAllBackground", "minimize", \
                "maxColorFromCell", "deleteShapes", "replicateShapes","colorByPixels", \
                "paintGridLikeBackground") # applyEvolve?
        repeatIfPerfect = ("extendColor", "moveAllShapes")
        firstIt = True
        if self.args.use_lora:
            base_to_lora(self.model)
        self.model.eval()
        prompt = arc_instruct
        last_state = initial_state
        last_state = init_state(last_state)
        actions = []
        states = []
        
        for step in range(max_steps):
            current_state = last_state

            possibleOps = Utils.getPossibleOperations(current_state.task, current_state.candidate)
            # print(possibleOps)
            
            filtered_func_list = []
            other_func_list = []
            for func in possibleOps:
                # try:
                _, success_proxy = update_state(current_state, func)
                if success_proxy:
                    filtered_func_list.append(func)

                else:
                    other_func_list.append(func)
                # except Exception as e:
                    # print(f"Error applying function {func}: {e}")
            allowed_actions_ = filtered_func_list.copy()
            num_random_from_other = 5  
            if len(other_func_list) > 0:
                num_random_from_other = min(num_random_from_other, len(other_func_list))
                allowed_actions_.extend(
                    random.sample(other_func_list, num_random_from_other)
                )
            allowed_actions = allowed_actions_
            # print("len(allowed_actions)", len(allowed_actions))
            if len(allowed_actions_) != 0:

                # epsilon greedy
                if np.random.rand() < self.epsilon and mode == "train":
                    action = random.choice(allowed_actions_)
                    # action = action.lower()
                else:
                    
                    current_state_text = construct_prompt(current_state)
                    inputs = prompt.replace("<init_state>", str(current_state_text)).strip()
                    input_ids = self.tokenizer.encode(inputs.strip() + "\n", add_special_tokens=False, return_tensors="pt").to(self.device)
                    prefix_output = self.model(input_ids[:, :-1], use_cache=True)
                    prefix_past = prefix_output.past_key_values

                    action_logits = []
                    for ac in allowed_actions_:
                        a = str(ac)
                        action_ids = self.tokenizer.encode(
                            a, add_special_tokens=False, return_tensors="pt"
                        ).to(self.device)
                        input_ids_with_action = torch.cat(
                            [input_ids[:, -1:], action_ids], dim=-1
                        )
                        outputs = self.model(
                            input_ids_with_action,
                            past_key_values=prefix_past,
                            use_cache=True,
                        )
                        logits = outputs.logits  # 获取对应于 action_ids 的 logits
                        total_log_prob = torch.zeros(1).cuda()
                        for i in range(1, input_ids_with_action.shape[-1]):
                            probs = torch.softmax(logits[:, i - 1, :], dim=-1)
                            for j in range(1):
                                total_log_prob[j] += torch.log(
                                    probs[j, input_ids_with_action[j, i]]
                                )
                        action_logits.append(total_log_prob)
                        # sample from tempered policy
                    # print("Action logits", action_logits)
                    # print("pf_temp", pf_temp)
                    action_logits = torch.stack(action_logits) / pf_temp

                    action_logits = action_logits.to(torch.float32)
                    max_logits = torch.max(action_logits)  # Find the maximum value
                    normalized_logits = (
                        action_logits - max_logits
                    )  # Subtract the max from all logits
                    action_logits = normalized_logits
                    # print(action_logits)
                    # print(torch.exp(action_logits))
                    # print(torch.sum(torch.exp(action_logits)))

                    probabilities = torch.exp(action_logits) / torch.sum(
                        torch.exp(action_logits)
                    )

                    dist = Categorical(probs=probabilities.t())

                    idx = dist.sample()

                    action = allowed_actions_[idx]

            else:
                action = random.choice(allowed_actions)
            # if "I have that, " not in last_state:
            #     last_state_ = "I have that, " + last_state
            #     states.append(last_state_.split("I have that, ")[1].strip())
            # else:
            #     states.append(last_state.split("I have that, ")[1].strip())
            # print(last_state)
            states.append(last_state)
            actions.append(action)

            new_state, success = update_state(current_state, action)
            last_state = new_state
            # if firstIt and str(action)[28:60].startswith(startOps) :
            #     if all([np.array_equal(np.array(last_state.cTask["train"][s]["input"]), last_state.task.trainSamples[s].inMatrix.m) for s in range(last_state.task.nTrain)]):
            #         break
            #     # newCandidate.generateTask()
            #     # tryOperations(t, newCandidate, cTask, b2c)
            # elif str(action)[28:60].startswith(repeatIfPerfect):# and last_state.candidate.score - changedPixels == cScore and changedPixels != 0:
            #     # newCandidate.generateTask()
            #     # tryOperations(t, newCandidate, cTask, b2c)
            #     break
            
            # Convert new_state from a numpy array to a list if it's a numpy array

            

            # whether reach the goal:

            if success:
                test_success = eval_output(actions, initial_state[0])
                if test_success:
                    r1 = 100
                    r1 = torch.tensor(r1).to(self.device)

                    return None, actions, states, r1, None
       
       
        if success:
            test_success = eval_output(actions, initial_state[0])
            if test_success:
                r1 = 100
            else:
                r1 = 1
        else:
            r1 = 1
        r1 = torch.tensor(r1).to(self.device)
        return None, actions, states, r1, None

    def local_search(
        self,
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
        states = []
        actions = []
        if self.args.use_lora:
            base_to_lora(self.model)
        self.model.eval()
        last_state = initial_state
        last_state = init_state(last_state)

        startOps = ("switchColors", "cropShape", "cropAllBackground", "minimize", \
                "maxColorFromCell", "deleteShapes", "replicateShapes","colorByPixels", \
                "paintGridLikeBackground") # applyEvolve?
        repeatIfPerfect = ("extendColor", "moveAllShapes")
        firstIt = True
        current_state = last_state
        for step in range(max_steps):

            # epsilon greedy
            if step < K:
                action = plan[step]
                
            else:
                # change function generate_all_actions
                current_state = last_state
                # change function generate_all_actions
                possibleOps = Utils.getPossibleOperations(current_state.task, current_state.candidate)
                # allowed_actions = generate_all_actions(last_state)
                # func_list = getPossibleOperations(task, cand)
                # allowed_actions_ = [
                #     act for act in allowed_actions if act.lower() not in actions
                # ]
                # print(func_list)
                filtered_func_list = []
                other_func_list = []
                for func in possibleOps:
                    try:
                        # Apply the function to the intermediate state
                        # result = func(Matrix(eval(initial_state)))
                        _, success_proxy = update_state(current_state, func)
                        # print(result)
                        # print(target_state)
                        # If the result matches the target state, keep the function
                        if success_proxy:
                            filtered_func_list.append(func)
                            # print(f"Function {func} can transform the state successfully")
                        else:
                            other_func_list.append(func)
                    except Exception as e:
                        print(f"Error applying function {func}: {e}")
                # print("Length of filtered function list,", len(filtered_func_list))

                # Choose all functions from filtered_func_list and a few from other_func_list
                allowed_actions_ = filtered_func_list.copy()

                # Define how many functions you want to randomly pick from other_func_list
                num_random_from_other = 5  # For example, pick 3 random functions

                # Ensure that we don't pick more than what's available in other_func_list
                if len(other_func_list) > 0:
                    num_random_from_other = min(num_random_from_other, len(other_func_list))
                    allowed_actions_.extend(
                        random.sample(other_func_list, num_random_from_other)
                    )
                allowed_actions = allowed_actions_

                if len(allowed_actions_) != 0:
                    action = random.choice(allowed_actions_)
                else:
                    action = random.choice(allowed_actions)
                action = action
            states.append(last_state)
            # if "I have that, " not in last_state:
            #     last_state_ = "I have that, " + last_state
            #     states.append(last_state_.split("I have that, ")[1].strip())
            # else:
            #     states.append(last_state.split("I have that, ")[1].strip())

            actions.append(action)

            last_action = action

           
            new_state, success = update_state(current_state, action)
            # Convert new_state from a numpy array to a list if it's a numpy array

            last_state = new_state
            # states.append(last_state)
            
            # whether reach the goal:
            # if firstIt and str(action)[28:60].startswith(startOps) :
            #     if all([np.array_equal(np.array(last_state.cTask["train"][s]["input"]), last_state.task.trainSamples[s].inMatrix.m) for s in range(last_state.task.nTrain)]):
            #         break
            #     # newCandidate.generateTask()
            #     # tryOperations(t, newCandidate, cTask, b2c)
            # elif str(action)[28:60].startswith(repeatIfPerfect):# and last_state.candidate.score - changedPixels == cScore and changedPixels != 0:
            #     # newCandidate.generateTask()
            #     # tryOperations(t, newCandidate, cTask, b2c)
            #     break
            if success:
                test_success = eval_output(actions, last_state.originalT)
                if test_success:
                    r1 = 100
                    r1 = torch.tensor(r1).to(self.device)

                    return None, actions, states, r1, None
       
       
        if success:
            test_success = eval_output(actions, last_state.originalT)
            if test_success:
                r1 = 100
            else:
                r1 = 10
        else:
            r1 = 1
        r1 = torch.tensor(r1).to(self.device)
        return None, actions, states, r1, None

    def forward_prob(self, actions, states):
        if self.args.use_lora:
            base_to_lora(self.model)
        # prompt = sample_prompt(self.init_prompt, shuffle_prompt=False, num_shot=1)
        prompt = arc_instruct

        initial_state = states[0]

        last_state = initial_state
        # last_state = init_state(last_state)
        log_pf = []
        log_bf = []
        # print("len of actions",len(actions))
        for step in range(len(actions)):
            current_state = last_state
            possibleOps = Utils.getPossibleOperations(current_state.task, current_state.candidate)
            filtered_func_list = []
            other_func_list = []
            for func in possibleOps:
                _, success_proxy = update_state(current_state, func)
                # print(result)
                # print(target_state)
                # If the result matches the target state, keep the function
                if success_proxy:
                    filtered_func_list.append(func)
                    # print(f"Function {func} can transform the state successfully")
                else:
                    other_func_list.append(func)
                # except Exception as e:
                    # print(f"Error applying function {func}: {e}")
            # print("Length of filtered function list,", len(filtered_func_list))

            # Choose all functions from filtered_func_list and a few from other_func_list
            allowed_actions_ = filtered_func_list.copy()

            # Define how many functions you want to randomly pick from other_func_list
            num_random_from_other = 5  # For example, pick 3 random functions

            # Ensure that we don't pick more than what's available in other_func_list
            if len(other_func_list) > 0:
                num_random_from_other = min(num_random_from_other, len(other_func_list))
                allowed_actions_.extend(
                    random.sample(other_func_list, num_random_from_other)
                )
            allowed_actions = allowed_actions_

            # print(allowed_actions_)

            current_state_text = construct_prompt(current_state)
            inputs = prompt.replace("<init_state>", str(current_state_text)).strip()

            input_ids = self.tokenizer.encode(
                inputs.lstrip() + "\n", return_tensors="pt"
            ).to(self.device)

            action = actions[step]

            bsz = len(allowed_actions)
            action_texts = [str(ac) for ac in allowed_actions]
            action_ids = [
                self.tokenizer.encode(
                    a, add_special_tokens=False, return_tensors="pt"
                ).to(self.device)
                for a in action_texts
            ]

            max_length = max(len(aid[0]) for aid in action_ids)
            padded_action_ids = [
                torch.cat(
                    [
                        aid,
                        torch.full(
                            (1, max_length - len(aid[0])),
                            self.tokenizer.pad_token_id,
                            device=self.device,
                        ),
                    ],
                    dim=-1,
                )
                for aid in action_ids
            ]
            batch_input_ids_with_actions = torch.cat(
                [torch.cat([input_ids, pid], dim=-1) for pid in padded_action_ids],
                dim=0,
            )
            batch_outputs = self.model(batch_input_ids_with_actions, use_cache=True)
            batch_logits = batch_outputs.logits
            # calculate the probability
            total_log_prob = torch.zeros(bsz).cuda()
            for i in range(input_ids.shape[-1], batch_input_ids_with_actions.shape[-1]):
                probs = torch.softmax(batch_logits[:, i - 1, :], dim=-1)
                for j in range(bsz):
                    if (
                        batch_input_ids_with_actions[j, i]
                        != self.tokenizer.pad_token_id
                    ):
                        total_log_prob[j] += torch.log(
                            probs[j, batch_input_ids_with_actions[j, i]]
                        )
            action_logits = total_log_prob
            # max_logits = torch.max(action_logits)  # Find the maximum value
            # normalized_logits = (
            #     action_logits - max_logits
            # )  # Subtract the max from all logits
            # action_logits = normalized_logits
            # print(action_logits)
            # print(torch.exp(action_logits))
            # print(torch.sum(torch.exp(action_logits)))

            probabilities = torch.exp(action_logits) / torch.sum(
                torch.exp(action_logits)
            )

            # import torch.nn.functional as F

            # print("action logits", torch.sum(torch.exp(action_logits)))
            # print("expo", torch.exp(action_logits))
            # print("logits", action_logits)

            # Use softmax to calculate the probabilities directly
            # probabilities = F.softmax(action_logits, dim=0)
            # Define a small epsilon value for smoothing
            action_logits = action_logits.to(torch.float64)

            probabilities = torch.exp(action_logits) / torch.sum(
                torch.exp(action_logits)
            )
            epsilon = 1e-9
            # print("Befora dding ", probabilities)
            # Apply smoothing to avoid log(0)
            probabilities = probabilities + epsilon

            # print(f"Training loss probs after epsilon: {probabilities}")

            # # Make sure no log(0) happens by checking if idx is valid
            # log_pf.append(torch.log(probabilities[idx]))
            # print(f"traing loss probs: {probabilities}")
            def compare_partial_funcs(func1, func2):
                """Compare two partial functions for equality by checking their function, args, and keywords."""
                
                # Compare the function objects
                if func1.func != func2.func:
                    return False
                
                # Compare the arguments
                if len(func1.args) != len(func2.args):
                    return False
                
                for arg1, arg2 in zip(func1.args, func2.args):
                    if isinstance(arg1, np.ndarray) and isinstance(arg2, np.ndarray):
                        # Compare numpy arrays using array_equal
                        if not np.array_equal(arg1, arg2):
                            return False
                    else:
                        # Compare other arguments directly
                        if arg1 != arg2:
                            return False
                
                # Compare the keywords
                if func1.keywords.keys() != func2.keywords.keys():
                    return False
                
                for key in func1.keywords:
                    kwarg1 = func1.keywords[key]
                    kwarg2 = func2.keywords[key]
                    
                    if isinstance(kwarg1, np.ndarray) and isinstance(kwarg2, np.ndarray):
                        # Compare numpy arrays in the keyword arguments
                        if not np.array_equal(kwarg1, kwarg2):
                            return False
                    else:
                        # Compare other keyword arguments directly
                        if kwarg1 != kwarg2:
                            return False
                
                return True

            from functools import partial

            # Example of comparing the action with allowed actions
            for idx, allowed_action in enumerate(allowed_actions):
                if isinstance(allowed_action, partial) and isinstance(action, partial):
                    if compare_partial_funcs(allowed_action, action):
                        break
            else:
                idx = 0
            # idx = allowed_actions.index(str(action))

            log_pf.append(torch.log(probabilities[idx]))

            if step < len(states) - 1:
                last_state = states[step + 1]

            # allowed_actions = generate_all_actions(last_state)
            pb = torch.tensor(1 / len(allowed_actions))
            # print(f"length of allowed actions {pb}")
            log_bf.append(torch.log(pb))
        return torch.stack(log_pf).sum(), torch.stack(log_bf).sum()

    def get_ll_reward(self, actions, states, goal):

        reward = []

        for idx in range(len(states)-1):
            state_pre = states[idx]
            state_post = states[idx+1]
            intuition = torch.exp(torch.tensor([state_pre.candidate.score - state_post.candidate.score]))
            reward.append(intuition)

        return torch.tensor(reward).to(self.device)

    # def get_ll_reward(self, actions, states, goal):

    #     reward = []

    #     prompt = sample_prompt(self.init_prompt, shuffle_prompt=False, num_shot=4)
    #     for step_idx, (state, action) in enumerate(zip(states, actions)):
    #         action = str(action)
    #         if isinstance(state, np.ndarray):
    #             # Convert the numpy array to a list
    #             state = state.tolist()
    #         # Convert the state to a string
    #         state = str(state)
    #         icl_template = prompt["icl_list"][step_idx // 2]
    #         if step_idx == 0:
    #             previous_action = ""
    #             current_state = state
    #         else:
    #             previous_action = str(actions[step_idx - 1]) + "\n"
    #             if isinstance(states[step_idx - 1], np.ndarray):
    #                 states[step_idx - 1] = states[step_idx - 1].tolist()
    #             current_state = str(states[step_idx - 1])
    #         inputs = (
    #             icl_template.replace("<init_state>", current_state.lstrip())
    #             .replace("<goals>", goal)
    #             .replace("<action>", previous_action.lstrip())
    #         )

    #         intuition = self.get_likelihood(inputs, [inputs + action.lstrip()])[0]
    #         self.ll_reward_dict[(step_idx, state, action, goal)] = intuition
    #         reward.append(intuition)

    #     return torch.tensor(reward).to(self.device)

    # def get_likelihood(
    #     self,
    #     prefix: str,
    #     contents: list[str],
    # ):
    #     bsz = len(contents)
    #     prefix_tokens = self.world_tokenizer.encode(prefix, add_special_tokens=True)
    #     prompts_tokens = [
    #         self.world_tokenizer.encode(x, add_special_tokens=True) for x in contents
    #     ]

    #     for prompt_tokens in prompts_tokens:
    #         assert prompt_tokens[: len(prefix_tokens)] == prefix_tokens

    #     max_prompt_size = max([len(t) for t in prompts_tokens])
    #     total_len = max_prompt_size
    #     tokens = (
    #         torch.full((bsz, total_len), self.world_tokenizer.pad_token_id)
    #         .cuda()
    #         .long()
    #     )

    #     for k, t in enumerate(prompts_tokens):
    #         tokens[k, : len(t)] = torch.tensor(t)[:2048].long()

    #     with torch.no_grad():
    #         outputs = self.model(tokens)
    #         logits = outputs.logits
    #     acc_probs = torch.zeros(bsz).cuda()
    #     for i in range(len(prefix_tokens), max_prompt_size):
    #         probs = torch.softmax(logits[:, i - 1, :], dim=-1)
    #         for j in range(bsz):
    #             if tokens[j, i] != self.world_tokenizer.pad_token_id:
    #                 acc_probs[j] += torch.log(probs[j, tokens[j, i]])

    #     return acc_probs
