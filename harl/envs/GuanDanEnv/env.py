import numpy as np
import random
from harl.envs.GuanDanEnv.guandan_utils import Utils, Error
import warnings
from collections import Counter
from gym.spaces import Discrete, Box, Dict
import warnings
    
class GuanDanEnv():
    '''
    Usage:
    Step1 Call GuanDanEnv() to create an instance
    Step2 Call GuanDanEnv.reset(config) to reset a match  
    '''
    def __init__(self, args):
        self.args = args
        self.cardscale = ['A','2','3','4','5','6','7','8','9','0','J','Q','K']
        self.suitset = ['h','d','s','c']
        self.point_order = ['2', '3', '4', '5', '6', '7', '8', '9', '0', 'J', 'Q', 'K', 'A']
        self.Normaltypes = ("single", "pair", "three", "straight", "set", "three_straight", "triple_pairs")
        self.scaletypes = ("straight", "three_straight", "triple_pairs")
        self.Utils = Utils()
        
        # Initialize environment based on args
        self.n_agents = 4  # Fixed for GuanDan
        self.episode_length = args.get("episode_length", 200)
        
        # Define observation and action spaces
        self.action_space = [Discrete(108) for _ in range(self.n_agents)]
        self.observation_space = [Box(low=0, high=1, shape=(150,), dtype=np.float32) 
                                for _ in range(self.n_agents)]
        self.share_observation_space = [Box(low=0, high=1, shape=(600,), dtype=np.float32)
                                    for _ in range(self.n_agents)]
    
        # 初始化游戏状态变量
        self.player_decks = [[] for _ in range(4)]
        self.lastMove = {'player': -1, 'action': [], 'claim': []}
        self.history = []    
        self.reset()

    def reset(self):
        """Reset the environment and return initial observations."""
        config = {
            'seed': self.args.get("seed", None),
            'level': self.args.get("level", '2')
        }
        
        self.level = config.get('level', '2')
        self._set_level()
        
        self.total_deck = [i for i in range(108)]
        self.card_todeal = [i for i in range(108)]
        random.shuffle(self.card_todeal)
        self.player_decks = [self.card_todeal[dpos*27 : (dpos+1)*27] for dpos in range(4)]
        
        self.done = False
        self.history = []  
        self.round = 0
        self.played_cards = [[] for _ in range(4)]
        self.reward = {i: 0 for i in range(4)}
        self.pass_on = -1
        self.lastMove = {'player': -1, 'action': [], 'claim': []}
        self.cleared = []
        self.game_state_info = "Running"
        
        # Get initial observations for all agents
        obs = self._get_obs_all()
        state = self._get_state()
        avail_actions = self._get_avail_actions()
    
        # 确保返回格式正确
        obs_array = [self._process_obs(obs[i]) for i in range(self.n_agents)]
        state_array = [state for _ in range(self.n_agents)]
    
        return obs_array, state_array, avail_actions
        
    def step(self, actions):
        """
        Execute one timestep of the environment.
        Args:
            actions: List of actions for each agent
        Returns:
            obs: List of agent observations
            state: Shared global state
            rewards: List of rewards for each agent
            dones: Whether the episode is done
            infos: Additional information
            avail_actions: Available actions for next step
        """
        # Convert actions to expected format
        curr_player = (self.round % 4)  # Simple turn management - should be improved
        action_nums = actions[curr_player]  # Get action for current player
        
        # Create response dictionary expected by original step method
        response = {
            'player': curr_player,
            'action': [action_nums],  # Single action for now
            'claim': [action_nums]    # Using same as action for simplicity
        }
        
        # Call original step logic
        obs_dict = self._step_internal(response)
        
        # Process results for HARL framework
        obs = [self._process_obs(obs_dict[i]) for i in range(self.n_agents)]
        state = self._get_state()
        state_array = [state for _ in range(self.n_agents)]
        rewards = [[self.reward[i]] for i in range(self.n_agents)]
        dones = [self.done for _ in range(self.n_agents)]
        infos = [{'game_state': self.game_state_info} for _ in range(self.n_agents)]
        avail_actions = self._get_avail_actions()
        
        return obs, state_array, rewards, dones, infos, avail_actions
    
    def _step_internal(self, response):
        """Original step logic adapted for internal use."""
        self.round += 1
        curr_player = response['player']
        action = response['action']
        claim = response['claim']
        
        # ... (rest of original step method logic)
        
        return self._get_obs_all()

    def _get_obs_space(self):
        """Define observation space for each agent."""
        # Observation includes: current cards, last move, game state info
        # Adjust dimensions as needed
        return Box(low=0, high=1, shape=(150,), dtype=np.float32)

    def _get_state_space(self):
        """Define shared state space."""
        # State includes: all players' played cards, current game state
        return Box(low=0, high=1, shape=(600,), dtype=np.float32)

    def _get_avail_actions(self):
        """Get available actions for each agent."""
        avail_actions = []
        for i in range(self.n_agents):
            if i in self.cleared:
                avail_actions.append([0]*108)  # No actions available if cleared
            else:
                avail_actions.append([1 if card in self.player_decks[i] else 0 for card in range(108)])
        return avail_actions

    def _get_obs_all(self):
        """Get observations for all players."""
        return {i: self._get_obs(i) for i in range(4)}

    def _get_obs(self, player):
        '''
        getting observation for player
        player: player_id (-1: all players)
        '''
        # 修改返回值结构，确保包含所有必要字段
        return {
            "deck": self.player_decks[player],  # 玩家当前手牌
            "last_move": self.lastMove,         # 最后出的牌
            "history": self.history,            # 历史记录
            "status": self.game_state_info,     # 游戏状态
            "player_id": player,                # 玩家ID
            "level": self.level,                # 当前级别
            "reward": self.reward[player]       # 当前奖励
        }
        
    def _get_state(self):
        """Get the shared global state."""
        # Combine information from all players
        state = np.zeros(600, dtype=np.float32)  # Adjust size as needed
        
        # Encode each player's current cards
        for i in range(4):
            for card in self.player_decks[i]:
                state[i*108 + card] = 1
                
        # Encode game state information
        # ... (add more state information as needed)
        
        return state

    def _process_obs(self, obs_dict):
        """Process observation dictionary into numpy array."""
        obs = np.zeros(150, dtype=np.float32)  # 确保与observation_space形状一致
    
        # 安全访问字典字段
        if 'deck' in obs_dict:
            for card in obs_dict['deck']:
                obs[card % 108] = 1  # 标记拥有的牌
    
        # 编码最后出的牌信息（示例）
        if 'last_move' in obs_dict and obs_dict['last_move']['action']:
            last_card = obs_dict['last_move']['action'][0]
            obs[108 + last_card % 108] = 1
    
        # 可以添加更多特征的编码...
    
        return obs
        
    
    def _set_reward(self):
        '''
        setting rewards
        if terminating: winner team gets reward 1~3
        else: rewards 0
        '''
        self.reward = {
            0: 0,
            1: 0,
            2: 0,
            3: 0
        }
        if self.done:
            if len(self.cleared) == 2: # Must be a double-dweller
                self.reward[self.cleared[0]] = 3
                self.reward[self.cleared[1]] = 3
            elif (self.cleared[2] - self.cleared[0]) % 2 == 0:
                self.reward[self.cleared[0]] = 2
                self.reward[self.cleared[2]] = 2
            else:
                self.reward[self.cleared[0]] = 1
                self.reward[(self.cleared[0] + 2) % 4] = 1
    
    def _raise_error(self, errno, detail):
        raise Error(self.errset[errno]+": "+detail)
    
    def _end_game(self, fault_player):
        '''
        ending game on player's action exceptions
        '''
        self.reward = {
            0: 0,
            1: 0,
            2: 0,
            3: 0
        }
        self.done = True
        self.reward[fault_player] = -3
        return self._get_obs(-1)
    
    def _get_obs(self, player):
        '''
        getting observation for player
        player: player_id (-1: all players)
        '''
        obs_set = {}
        for i in range(4):
            obs_set[i] = {
                "id": i,
                "level": self.level,
                "status": self.game_state_info,
                "deck": self.player_decks[i],
                "last_move": self.lastMove,
                "history": self.history,
                "reward": self.reward[i]
            }
        if player == -1:
            return obs_set
        else:
            return { player: obs_set[player] }
                
    def _set_level(self):     
        self.point_order.remove(self.level)
        self.point_order.append(self.level)
        self.point_order.extend(["o", "O"])
        
    def _is_legal_claim(self, action: list, claim: list):
        covering = "h" + self.level
        if len(action) != len(claim):
            return False
        action_pok = [self.Utils.Num2Poker(p) for p in action]
        claim_pok = [self.Utils.Num2Poker(p) for p in claim]
        for pok in action_pok:
            if pok != covering:
                if pok in claim_pok:
                    claim_pok.remove(pok)
                else:
                    return False
        for pok in claim_pok:
            if pok[1] == 'o' or pok[1] == 'O':
                return False
        return True
    
    def _check_poker_type(self, poker: list):
        if poker == []:
            return "pass", ()
        # covering = "h" + level
        poker = [self.Utils.Num2Poker(p) for p in poker]
        if len(poker) == 1:
            return "single", (poker[0][1])
        if len(poker) == 2:
            if poker[0][1] == poker[1][1]:
                return "pair", (poker[0][1])
            return "invalid", ()
        # 大于等于三张
        points = [p[1] for p in poker]
        cnt = Counter(points)
        vals = list(cnt.values())
        if len(poker) == 3:
            if "o" in points: 
                return "invalid", ()
            if vals.count(3) == 1:
                return "three", (points[0])
            return "invalid", ()
        if len(poker) == 4: # should be a bomb
            if "o" in points or "O" in points: # should be a rocket
                if cnt["o"] == 2 and cnt["O"] == 2:
                    return "rocket", ("jo")
                return "invalid", ()
            if vals.count(4) == 1:
                return "bomb", (4, points[0])
            return "invalid", ()
        if len(poker) == 5: # could be straight, straight flush, three&two or bomb
            if vals.count(5) == 1:
                return "bomb", (5, points[0])
            if vals.count(3) == 1 and vals.count(2) == 1: # set: 三带二 
                three = ''
                two = ''
                for k in list(cnt.keys()):
                    if cnt[k] == 3:
                        three = k
                    elif cnt[k] == 2:
                        two = k
                return "set", (three, two)
            if vals.count(1) == 5: # should be straight
                points.sort(key=lambda x: self.cardscale.index(x))
                suits = [p[0] for p in poker]
                suit_cnt = Counter(suits)
                suit_vals = list(suit_cnt.values())
                flush = False
                if suit_vals.count(5) == 1:
                    flush = True
                first = points[0]
                if first == 'A':
                    if points == ['A', '0', 'J', 'Q', 'K']:
                        if flush:
                            return "straight_flush", ('0')
                        return "straight", ('0')
                sup_straight = [self.cardscale[self.cardscale.index(first)+i] for i in range(5)]
                if points == sup_straight:
                    if flush:
                        return "straight_flush", (first)
                    return "straight", (first)
            return "invalid", ()
        if len(poker) == 6: # could be triple_pairs, three_straight, bomb
            if vals.count(6) == 1:
                return "bomb", (6, points[0])
            if vals.count(3) == 2:
                ks = []
                for k in list(cnt.keys()):
                    ks.append(k)
                ks.sort(key=lambda x: self.cardscale.index(x))
                if 'A' in ks:
                    if ks == ['A', '2']:
                        return "three_straight", ('A')
                    if ks == ['A', 'K']:
                        return "three_straight", ('K')
                    return "invalid", ()
                if self.cardscale.index(ks[1]) - self.cardscale.index(ks[0]) == 1:
                    return "three_straight", (ks[0])
            if vals.count(2) == 3:
                ks = []
                for k in list(cnt.keys()):
                    ks.append(k)
                ks.sort(key=lambda x: self.cardscale.index(x))
                if 'A' in ks:
                    if ks == ['A', 'Q', 'K']:
                        return "triple_pairs", ('Q')
                    if ks == ['A', '2', '3']:
                        return "triple_pairs", ('A')
                    return "invalid", ()
                pairs = [self.cardscale[self.cardscale.index(ks[0])+i] for i in range(3)]
                if ks == pairs:
                    return "triple_pairs", (ks[0])
            return "invalid", ()
        if len(poker) > 6 and len(poker) <= 10:
            if vals.count(len(poker)) == 1:
                bomb = points[0]
                return "bomb", (len(poker), bomb)
        return "invalid", ()
    
    def _check_bigger(self, type1, point1, type2, point2):
        '''
        Check if poker2(type2, point2) is bigger than poker1(type1, point1)
        Assumption: type1 and type2 are VALID cardtypes. Must check types before calling this function 
        '''
        if type2 == "rocket":
            return True
        if type1 == "rocket":
            return False
        if type1 in self.Normaltypes:
            if type2 not in self.Normaltypes:
                return True
            if type1 == type2:
                if type1 in self.scaletypes and self.cardscale.index(point2[0]) > self.cardscale.index(point1[0]):
                    return True
                if type1 not in self.scaletypes and self.point_order.index(point2[0]) > self.point_order.index(point1[0]):
                    return True
                return False
            return "error"
        if type2 in self.Normaltypes:
            return "error"
        if type1 == "bomb":
            if type2 == "bomb":
                if point2[0] == point1[0] and self.point_order.index(point2[1]) > self.point_order.index(point1[1]):
                    return True
                if point2[0] > point1[0]:
                    return True
            if type2 == "straight_flush":
                if point1[0] < 6:
                    return True
        if type1 == "straight_flush":
            if type2 == "bomb":
                if point2[0] >= 6:
                    return True
                return False
            if type2 == "straight_flush":
                if self.cardscale.index(point2[0]) > self.cardscale.index(point1[0]):
                    return True
        return False
    