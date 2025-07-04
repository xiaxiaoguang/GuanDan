import torch
import torch.nn as nn
from harl.utils.envs_tools import check
from harl.models.base.cnn import CNNBase
from harl.models.base.mlp import MLPBase
from harl.models.base.rnn import RNNLayer
from harl.models.base.act import ACTLayer
from harl.models.policy_models.baseline_kid import make_decision
from harl.utils.envs_tools import get_shape_from_obs_space
from harl.envs.GuanDanEnv.env import GuanDanEnv
import json

###0629 by WMQ
def json_to_action(json_response):
    env = GuanDanEnv()
    env.reset()
    return env.get_action_id(json_response)

###0629 by WMQ
def binary_to_index(binary_array):
    '''get the index of the all 1 element in a 0-1 vector, using torch'''
    binary_array = torch.tensor(binary_array, dtype=torch.bool)
    indices = torch.nonzero(binary_array, as_tuple=False)
    return indices.squeeze().tolist() if indices.numel() > 0 else []

###0629 by WMQ
def history_to_json(info, obs):
    """Convert history to JSON format for decision making."""
    json_format = {"requests": [], "responses": []}
    
    # player = [[] for _ in range(4)]
    # for i in range(4):
    #     player[i] = info[0][i]
    # for i in range(6, len(info[0]), 2):
    #     player[i] -= info[0][i]
    
    # current_player = -1
    # for i in range(4):
    #     if obs[0][:108] == player[i]:
    #         current_player = i
    
    # assert current_player != -1, "Current player not found in history."
    
    current_player = ((len(info[0]) - 4) % 8) // 2
    
    json_format["requests"].append({
        "stage": "deal",
        "deliver": binary_to_index(info[0][current_player]),
        "your_id": current_player,
        "global":{
            "level": "2",
            "tribute": 0,
            "first": None,
            "last": None
        }
    })
    
    rounds = (len(info[0]) + 4) // 8
    
    for rr in range(rounds):    
        
        hhh = [[] for _ in range(4)]
        
        for j in range(4):
            jjdex = (j + current_player) % 4
            if 2*j + 8*rr + 2*current_player -4 < 4:
                hhh[jjdex] = [[],[]] 
            else:    
                hhh[jjdex] = [
                    binary_to_index(info[0][2*j + 8*rr + 2*current_player -4]),
                    binary_to_index(info[0][2*j + 8*rr + 2*current_player -3])
                    ]
                
        json_format["requests"].append({
            "stage": "play",
            "history": hhh,
            "global":{"level":"2","tribute":0,"first":None,"last":None,"tribute_cards":{},"return_cards":{},"resist":False},
            "done":[],"pass_on":-1
        })
        
        if 8*rr + 2*current_player -4 < 4:
            json_format["responses"] = [[],[]] 
        else:    
            json_format["responses"] = [
                binary_to_index(info[0][8*rr + 2*current_player -4]),
                binary_to_index(info[0][8*rr + 2*current_player -3])
                ]
        
    return json_format
    

class ExpertPolicy(nn.Module):
    """Expert policy model. Outputs actions given observations."""

    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        """Initialize StochasticPolicy model.
        Args:
            args: (dict) arguments containing relevant model information.
            obs_space: (gym.Space) observation space.
            action_space: (gym.Space) action space.
            device: (torch.device) specifies the device to run on (cpu/gpu).
        """
        super(ExpertPolicy, self).__init__()
        self.hidden_sizes = args["hidden_sizes"]
        self.args = args
        self.gain = args["gain"]
        self.initialization_method = args["initialization_method"]
        self.use_policy_active_masks = args["use_policy_active_masks"]
        self.use_naive_recurrent_policy = args["use_naive_recurrent_policy"]
        self.use_recurrent_policy = args["use_recurrent_policy"]
        self.recurrent_n = args["recurrent_n"]
        self.tpdv = dict(dtype=torch.float32, device=device)

        obs_shape = get_shape_from_obs_space(obs_space)
        base = CNNBase if len(obs_shape) == 3 else MLPBase
        self.base = base(args, obs_shape)

        if self.use_naive_recurrent_policy or self.use_recurrent_policy:
            self.rnn = RNNLayer(
                self.hidden_sizes[-1],
                self.hidden_sizes[-1],
                self.recurrent_n,
                self.initialization_method,
            )
        
        self.act = ACTLayer(
            action_space,
            self.hidden_sizes[-1],
            self.initialization_method,
            self.gain,
            args,
        )

        self.to(device)
        
    ###0629 by WMQ
    def forward(
        self, obs, rnn_states, masks, history, available_actions=None, deterministic=False
    ):
        """Compute actions from the given inputs.
        Args:
            obs: (np.ndarray / torch.Tensor) observation inputs into network.
            rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
            masks: (np.ndarray / torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
            available_actions: (np.ndarray / torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
            deterministic: (bool) whether to sample from action distribution or return the mode.
        Returns:
            actions: (torch.Tensor) actions to take.
            action_log_probs: (torch.Tensor) log probabilities of taken actions.
            rnn_states: (torch.Tensor) updated RNN hidden states.
        """

        json_input = history_to_json(history, obs)
        # print(type(json_input))
        response = make_decision(json_input)        
        action_id = json_to_action(response[0])
        
        actions = torch.zeros((1, 367), dtype=torch.float32)
        actions[0, action_id] = 1.0  # Set the action at
        action_log_probs = torch.zeros((1 , 367))
        rnn_states = torch.zeros(rnn_states.shape, dtype=torch.float32)

        return actions, action_log_probs, rnn_states

    def evaluate_actions(
        self, obs, rnn_states, action, masks, available_actions=None, active_masks=None
    ):
        """Compute action log probability, distribution entropy, and action distribution.
        Args:
            obs: (np.ndarray / torch.Tensor) observation inputs into network.
            rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
            action: (np.ndarray / torch.Tensor) actions whose entropy and log probability to evaluate.
            masks: (np.ndarray / torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
            available_actions: (np.ndarray / torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
            active_masks: (np.ndarray / torch.Tensor) denotes whether an agent is active or dead.
        Returns:
            action_log_probs: (torch.Tensor) log probabilities of the input actions.
            dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
            action_distribution: (torch.distributions) action distribution.
        """
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)

        actor_features = self.base(obs)

        if self.use_naive_recurrent_policy or self.use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        action_log_probs, dist_entropy, action_distribution = self.act.evaluate_actions(
            actor_features,
            action,
            available_actions,
            active_masks=active_masks if self.use_policy_active_masks else None,
        )

        return action_log_probs, dist_entropy, action_distribution
