import numpy as np
import random
from harl.envs.GuanDanEnv.utils import Utils, Error
import warnings
from collections import Counter
from gym import spaces
from copy import deepcopy

def to_one_hot(x):
    one_hot = np.zeros(108, dtype=np.int8)
    one_hot[x] = 1
    return one_hot

class GuanDanEnv():
    '''
    Usage:
    Step1 Call GuanDanEnv() to create an instance
    Step2 Call GuanDanEnv.reset(config) to reset a match  
    '''
    def __init__(self):
        self.cardscale = ['A','2','3','4','5','6','7','8','9','0','J','Q','K']
        self.suitset = ['h','d','s','c']
        self.point_order = ['2', '3', '4', '5', '6', '7', '8', '9', '0', 'J', 'Q', 'K', 'A']
        self.Normaltypes = ("single", "pair", "three", "straight", "set", "three_straight", "triple_pairs")
        self.scaletypes = ("straight", "three_straight", "triple_pairs")
        self.Utils = Utils()
        self.level = None
        self.done = False
        self.game_state_info = "Init"
        self.cleared = [] # list of cleared players (have played all their decks)
        self.agent_names = ['player_%d' % i for i in range(4)]
        self.errset = {
            0: "Initialization Fault",
            1: "PlayerAction Fault",
            2: "Game Fault"
        }
        
        self.n_agents = 4
        self.curr_player = 0 # initialize

        # self.observation_space = self.repeat(spaces.Dict({
        #     "hand": spaces.MultiBinary(108),
        #     "played_cards": spaces.MultiBinary(108),
        #     "last_move": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
        #     "turn": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
        #     "cleared": spaces.MultiBinary(4),
        #     "level": spaces.MultiBinary(13),
        #     "pass_on": spaces.MultiBinary(4)
        # }))
        self.observation_space = self.repeat(
            spaces.Box(low=0, high=1, shape=(239,), dtype=np.float32)
        )

        # 可行的优化，每个玩家的手牌给critic是必要的，其余的附加信息只需要一遍
        self.share_observation_space = self.repeat(
            spaces.Box(low=0, high=1, shape=(563,), dtype=np.float32)
        )

        # 1 (Pass)
        # 15 (Single)
        # + 15 (Pair)
        # + 13 (Three)
        # + 156 (Three with Two)
        # + 12 (Triple Pairs)
        # + 13 (Three of a Kind Chains)
        # + 10 (Straights)
        # + 91 (Bombs)
        # + 40 (Straight Flush)
        # + 1 (Rocket)
        # = ⭐ 367 actions ⭐
        self.action_space = self.repeat(spaces.Box(low=0, high=1, shape=(367,), dtype=np.float32))

    def get_curr_player(self):
        return self.curr_player
    
    def repeat(self, a):
        return [a for _ in range(self.n_agents)]

    def seed(self,qwq):
        pass

    def reset(self, config={}):
        '''
        Call this function to start different matches
        @ config: contains match initalization info
        '''
        if 'seed' in config:
            self.seed = config['seed']
            random.seed(self.seed)
        if 'level' in config and config['level'] in self.cardscale:
            self.level = config['level']
        else:
            self.level = '2'
            warnings.warn("ResetConfigWarning: Level configuration fault or no level designated.")
        self.curr_player = 0
        self.point_order = ['2', '3', '4', '5', '6', '7', '8', '9', '0', 'J', 'Q', 'K', 'A']
        self._set_level()
        self.total_deck = [i for i in range(108)]
        self.card_todeal = [i for i in range(108)]
        random.shuffle(self.card_todeal)
        self.player_decks = [self.card_todeal[dpos*27 : (dpos+1) * 27] for dpos in range(4)]
        self.done = False
        self.history = []
        for dpos in range(4):
            self.history.append(to_one_hot(self.player_decks[dpos]))
        self.round = 0
        self.played_cards = [[] for _ in range(4)]
        self.reward = {
            0: 0,
            1: 0,
            2: 0,
            3: 0
        }
        self.pass_on = -1
        self.lastMove = {
            'player': -1, # the first round
            'action': [], 
            'claim': []
        }
        self.cleared = []
        self.game_state_info = "Running"
        # You just need to implement this function , by parameter 0 it return a correct obs of player 0
        return self.get_obs(), self.get_state(), None, self.history


    def get_obs(self):
        """Returns all agent observations in a list.

        NOTE: Agents should have access only to their local observations
        during decentralised execution.
        """
        return np.array([self.__get_obs(agent_id) for agent_id in range(self.n_agents)])
    
    def _get_point_map(self, cards):

        point_map = {pt: [] for pt in self.cardscale + ["o", "O"]}
        # print(self.cardscale)
        # ['A', '2', '3', '4', '5', '6', '7', '8', '9', '0', 'J', 'Q', 'K']
        for c in cards:
            base_id = c % 54
            if base_id < 52:
                pt = self.cardscale[base_id // 4]
            elif base_id == 52:
                pt = "o"
            else:
                pt = "O"
            point_map[pt].append(c)

        return point_map
    
    def _generate_consecutive_pairs(self, length=3):
        """
        Generate all valid triple-pairs (3 consecutive points, each needing 2 cards).
        """
        result = []
        # use cardscale circularly: A can follow K
        extended = self.cardscale + self.cardscale[:length - 1]  # for wraparound
        for i in range(len(self.cardscale)):
            group = extended[i:i+length]
            if '2' == group[2]:
                continue  
            result.append(group)
        return result


    def _generate_consecutive_triples(self, length=2):
        """
        Generate all valid triple chains (2 consecutive points, each needing 3 cards).
        """
        result = []
        extended = self.cardscale + self.cardscale[:length - 1]
        for i in range(len(self.cardscale)):
            group = extended[i:i+length]
            # if '2' in group and group != ['A', '2']:
            #     continue  
            result.append(group)
        return result


    def _generate_straights(self, length=5):
        """
        Generate all valid straights: must be exactly 5 consecutive cards.
        A, 2 can be either low (A-2-3-4-5) or high (10-J-Q-K-A).
        """
        valid = []
        # Reuse cardscale with adjusted rules
        pure_order = ['3','4','5','6','7','8','9','0','J','Q','K','A','2']  # ascending
        extended = pure_order + pure_order[:length - 1]
        for i in range(len(pure_order)):
            group = extended[i:i+length]
            # if '2' in group and group != ['A', '2', '3', '4', '5']:
                # continue  # 2 can only appear in A-2-3-4-5
            valid.append(group)
        return valid


    def enumerate_legal_actions(self, hand_cards):
        """
        Enumerate all valid actions given current 108-dim hand vector (0/1)
        Yield: (action, claim, action_id)
        """

        point_map = self._get_point_map(hand_cards)
        

        def find_cards(point, count):
            # Find up to 'count' cards of a given point
            return point_map.get(point, [])[:count]

        action_id = 0

        # 0. Pass
        if self.legal_check([],[]):
            yield [], [], 0
        action_id += 1
        # 1–15: Single (A, 2, ..., Joker)
        single_order = self.cardscale + ["o", "O"]  # x: joker, X: JOKER
        for i, pt in enumerate(single_order):
            cards = find_cards(pt, 1)
            if len(cards) == 1:
                if self.legal_check(cards, cards):
                    yield cards, cards, action_id
            action_id += 1

        # 16–30: Pair
        for i, pt in enumerate(single_order):
            cards = find_cards(pt, 2)
            if len(cards) == 2:
                if self.legal_check(cards, cards):
                    yield cards, cards, action_id
            action_id += 1

        # 31–43: Three of a kind
        for i, pt in enumerate(self.cardscale):
            cards = find_cards(pt, 3)
            if len(cards) == 3:
                if self.legal_check(cards, cards):
                    yield cards, cards, action_id
            action_id += 1

        # 44–199: Three with Two (13 points × 12 combos)
        for i, pt3 in enumerate(self.cardscale):
            triple = find_cards(pt3, 3)

            if len(triple) < 3:
                action_id += 12
                continue

            for j, pt2 in enumerate(self.cardscale):
                if pt2 == pt3:
                    continue
                pair = find_cards(pt2, 2)
                if len(pair) == 2:
                    cards = triple + pair
                    if self.legal_check(cards, cards):
                        yield cards, cards, action_id + j
            
            action_id += 12

        # 200–211: Triple pairs (aa2233 → qqkkaa)
        triple_pairs_start = self._generate_consecutive_pairs(length=3)
        for i, start_pts in enumerate(triple_pairs_start):
            all_cards = []
            valid = True
            for pt in start_pts:
                cards = find_cards(pt, 2)
                if len(cards) < 2:
                    valid = False
                    break
                all_cards += cards
            if valid and self.legal_check(all_cards, all_cards):
                yield all_cards, all_cards, action_id
            action_id += 1

        # 212–224: Three-of-a-kind chains (aaa222 to kkkaaa)
        triple_chains_start = self._generate_consecutive_triples(length=2)
        for i, start_pts in enumerate(triple_chains_start):
            all_cards = []
            valid = True
            for pt in start_pts:
                cards = find_cards(pt, 3)
                if len(cards) < 3:
                    valid = False
                    break
                all_cards += cards
            if valid and self.legal_check(all_cards, all_cards):
                yield all_cards, all_cards, action_id
            action_id += 1

        # 225–234: Straights (a2345 to 10JQKA) → 10 options
        straight_sets = self._generate_straights(length=5)
        for pts in straight_sets:
            cards = []
            valid = True
            for pt in pts:
                c = find_cards(pt, 1)
                if not c:
                    valid = False
                    break
                cards += c
            if valid and self.legal_check(cards, cards):
                yield cards, cards, action_id
            action_id += 1

        # 235–325: Bombs (13 types × up to 7 sizes)
        for i, pt in enumerate(self.cardscale):
            for size in range(4, 11):
                cards = find_cards(pt, size)
                if len(cards) == size:
                    if self.legal_check(cards, cards):
                        yield cards ,cards ,action_id
                action_id += 1

        # 326–365: Straight Flush (10 types × 4 suits)
        suits = [0,1,2,3]
        for pts in straight_sets:
            for suit in suits:
                cards = []
                for pt in pts:
                    for card in point_map.get(pt, []):
                        if card % 4 == suit:
                            cards.append(card)
                            break
                if len(cards) == 5 and self.legal_check(cards, cards):
                    yield cards, cards, action_id
                action_id += 1

        # 366: Rocket (4 jokers)
        jokers = find_cards("x", 2) + find_cards("X", 2)
        if len(jokers) >= 4:
            cards = jokers[:4]
            if self.legal_check(cards, cards):
                yield cards, cards, 366

    def get_action_id(self, hand_cards):
        """
        write a mapping from hand_cards(a list of card_id, index from 0 to 107) to action_id(order listed as below)
        """

        # 1 (Pass)
        # 15 (Single)
        # + 15 (Pair)
        # + 13 (Three)
        # + 156 (Three with Two)
        # + 12 (Triple Pairs)
        # + 13 (Three of a Kind Chains)
        # + 10 (Straights)
        # + 91 (Bombs)
        # + 40 (Straight Flush)
        # + 1 (Rocket)
        # = ⭐ 367 actions ⭐
        
        point_map = self._get_point_map(hand_cards)
        
        if len(hand_cards) == 0:
            return 0
        elif len(hand_cards) == 1:
            # Single
            pt = self.Utils.Num2Poker(hand_cards[0])[1]
            if pt in self.cardscale + ["o", "O"]:
                return 1 + (self.cardscale + ["o", "O"]).index(pt)
            else:
                return -1
        elif len(hand_cards) == 2:
            # Pair
            pt0 = self.Utils.Num2Poker(hand_cards[0])[1]
            pt1 = self.Utils.Num2Poker(hand_cards[1])[1]
            if pt0 == pt1 and pt0 in self.cardscale + ["o", "O"]:
                return 16 + (self.cardscale + ["o", "O"]).index(pt0)
            else:
                return -1
        elif len(hand_cards) == 3:
            # Three of a kind
            pt0 = self.Utils.Num2Poker(hand_cards[0])[1]
            pt1 = self.Utils.Num2Poker(hand_cards[1])[1]
            pt2 = self.Utils.Num2Poker(hand_cards[2])[1]
            if pt0 == pt1 and pt1 == pt2 and pt0 in self.cardscale:
                return 31 + (self.cardscale + ["o", "O"]).index(pt0)
            else:
                return -1
            
        elif len(hand_cards) >= 4:
            point_count = Counter([self.Utils.Num2Poker(c)[1] for c in hand_cards])
            
            # Three with Two
            if len(point_count) == 2 and len(hand_cards) == 5:
                pt3, pt2 = point_count.most_common(2)
                if pt3[1] == 3 and pt2[1] == 2:
                    return 44 + (self.cardscale + ["o", "O"]).index(pt3[0]) * 12 + (self.cardscale + ["o", "O"]).index(pt2[0])
                elif pt3[1] == 2 and pt2[1] == 3:
                    return 44 + (self.cardscale + ["o", "O"]).index(pt2[0]) * 12 + (self.cardscale + ["o", "O"]).index(pt3[0])
                
            # Pairs Chains
            elif len(point_count) == 3 and all(v == 2 for v in point_count.values()):
                # sort according to cardscale
                pts = sorted(point_count.keys(), key=lambda x: (self.cardscale + ["o", "O"]).index(x))
                # print(self._generate_consecutive_pairs())
                if pts in self._generate_consecutive_pairs():
                    return 200 + self._generate_consecutive_pairs().index(pts[:3])
                elif pts == ['A', 'Q', 'K']:
                    return 200 + self._generate_consecutive_pairs().index(['Q', 'K', 'A'])
                
            # Triple Pairs
            elif len(point_count) == 2 and all(v == 3 for v in point_count.values()):
                pts = sorted(point_count.keys(), key=lambda x: (self.cardscale + ["o", "O"]).index(x))
                if pts in self._generate_consecutive_triples():
                    return 212 + self._generate_consecutive_triples().index(pts[:2])
                elif pts == ['A', 'K']:
                    return 212 + self._generate_consecutive_triples().index(['K', 'A'])
                
            # Straights
            elif len(hand_cards) == 5 and len(point_count) == 5:
                pts = [self.Utils.Num2Poker(c)[1] for c in hand_cards]
                # print(self._generate_straights())
                if all(pt in self.cardscale for pt in pts):
                    pts.sort(key=lambda x: (self.cardscale + ["o", "O"]).index(x))
                    # print('pts',pts)
                    if pts in self._generate_straights():
                        return 225 + self._generate_straights().index(pts)
                    elif pts == ['A', '0', 'J', 'Q', 'K']:
                        return 225 + self._generate_straights().index(['0', 'J', 'Q', 'K', 'A'])
                        
            # Bombs
            elif len(hand_cards) <= 10 and len(point_count) == 1:
                pt = self.Utils.Num2Poker(hand_cards[0])[1]
                return 238 + (len(hand_cards) - 4) + (self.cardscale).index(pt) * 7
            # Straight Flush
            elif len(hand_cards) == 5:
                suits = [c % 4 for c in hand_cards]
                if len(set(suits)) == 1:
                    pts = [self.Utils.Num2Poker(c)[1] for c in hand_cards]
                    if all(pt in self.cardscale for pt in pts):
                        pts.sort(key=lambda x: (self.cardscale + ["o", "O"]).index(x))
                        if pts in self._generate_straights():
                            return 326 + self._generate_straights().index(pts) * 4 + suits[0]
            # Rocket
            elif len(hand_cards) == 4 and all(self.Utils.Num2Poker(c)[1] in ["o", "O"] for c in hand_cards):
                return 366
        return -1  # Invalid action


    def legal_check(self,claim,action):
        if not self._is_legal_claim(action, claim):
            return False
        
        cur_pokertype, cur_points = self._check_poker_type(claim)
        if cur_pokertype == 'invalid':
            return False
        
        if len(self.lastMove["action"]) == 0:  # First play of round
            if cur_pokertype == "pass":
                return False
        else:
            if cur_pokertype != "pass":
                last_type, last_pts = self._check_poker_type(self.lastMove["claim"])
                bigger = self._check_bigger(last_type, last_pts, cur_pokertype, cur_points)
                if bigger == "error":
                    return False
                if not bigger:
                    return False
        return True


    def id2response(self, prob_vector: np.ndarray):
        """
        Convert a (367,) probability vector to a response dict:
        {
            "player": self.curr_player,
            "action": [int card indices],
            "claim": [used for _check_poker_type]
        }
        """

        assert prob_vector.shape[-1] == 367

        assert not np.isnan(prob_vector).any()

        # def softmax(x):
        #     """Compute softmax values for each sets of scores in x."""
        #     e_x = np.exp(x - np.max(x))
        #     return e_x / e_x.sum()
        # prob_vector = softmax(prob_vector[0])
        prob_vector = prob_vector[0]
        curr_player = self.curr_player
        hand = self.player_decks[curr_player]
        # print(hand)
        legal_actions = self.enumerate_legal_actions(hand)  # List of (action, claim, id)
        best_id = -1
        best_prob = -np.inf
        best_action = None
        best_claim = None

        for action, claim, act_id in legal_actions:
            if act_id < 0 or act_id >= 367:
                continue
 
            prob = prob_vector[act_id]

            if prob > best_prob or best_id < 0:
                best_prob = prob
                best_id = act_id
                best_action = action
                best_claim = claim

        if best_action is None or best_action == 0: 
            # Fallback: pass
            return {
                "player": curr_player,
                "action": [],
                "claim": [],
                "action_id" : 0,
            }

        return {
            "player": curr_player,
            "action": best_action,
            "claim": best_claim,
            "action_id" : best_id,
        }

    def step(self, action):

        ###0628###
        # print("Action:", action.shape)
        # (4, 367)

        self.round += 1
        response = self.id2response(action)  # Convert ID to actual action + claim
        curr_player = response["player"]
        action = response["action"]
        claim = response["claim"]
        
        ###0628###
        # print("response:", response)
        # response: {'player': 3, 'action': [62, 63, 8, 29, 83], 'claim': [62, 63, 8, 29, 83], 'action_id': 75}

        # Error checking
        # if not self._is_legal_claim(action, claim):
        #     print('caonima?')
        #     self.game_state_info = f"Player {curr_player}: ILLEGAL CLAIM"
        #     self.done = True
        #     return self._end_game(curr_player)

        for card in action:
            if card in self.player_decks[curr_player]:
                self.player_decks[curr_player].remove(card)
                self.played_cards[curr_player].append(card)
            else:
                self.game_state_info = f"Player {curr_player}: NOT YOUR POKER"
                self.done = True
                return self._end_game(curr_player)

        cur_pokertype, cur_points = self._check_poker_type(claim)
        if cur_pokertype == 'invalid':
            self.game_state_info = f"Player {curr_player}: INVALID TYPE"
            self.done = True
            return self._end_game(curr_player)

        if len(self.lastMove["action"]) == 0:  # First play of round
            if cur_pokertype == "pass":
                self.game_state_info = f"Player {curr_player}: ILLEGAL PASS AS FIRST-HAND"
                self.done = True
                return self._end_game(curr_player)
            self.lastMove = response
            self.pass_on = -1
        else:
            if cur_pokertype != "pass":
                last_type, last_pts = self._check_poker_type(self.lastMove["claim"])
                bigger = self._check_bigger(last_type, last_pts, cur_pokertype, cur_points)
                if bigger == "error":
                    self.game_state_info = f"Player {curr_player}: POKERTYPE MISMATCH"
                    self.done = True
                    return self._end_game(curr_player)
                if not bigger:
                    self.game_state_info = f"Player {curr_player}: CANNOT BEAT LASTMOVE"
                    self.done = True
                    return self._end_game(curr_player)
                self.lastMove = response
                self.pass_on = -1

        # print(response) ###0628###
        self.history.append(to_one_hot(action))  # Record the action in history
        self.history.append(to_one_hot(claim))  # Record the claim in history

        # Check if player cleared all cards
        if len(self.player_decks[curr_player]) == 0:
            self.cleared.append(curr_player)
            if len(self.cleared) == 3 or (len(self.cleared) == 2 and (self.cleared[1] - self.cleared[0]) % 2 == 0):
                self.done = True
                self.game_state_info = "Finished"
            self.pass_on = curr_player

        rewards = self._set_reward(curr_player)
        # === Prepare next player ===
        if not self.done:
            next_player = (curr_player + 1) % 4
            if next_player == self.pass_on:
                next_player = (self.pass_on + 2) % 4
                self.lastMove = {"player": -1, "action": [], "claim": [], 'action_id' : 0}

            while next_player in self.cleared:
                next_player = (next_player + 1) % 4
                if next_player == self.pass_on:
                    next_player = (self.pass_on + 2) % 4
                    self.lastMove = {"player": -1, "action": [], "claim": [] , 'action_id' : 0}

            if next_player == self.lastMove['player']:
                self.lastMove = {"player": -1, "action": [], "claim": [] , 'action_id' : 0}

            self.curr_player = next_player

        ###0628###
        # print("History:", self.history)

        # === Prepare output ===
        obs = self.get_obs()
        share_obs = np.array([self.get_state() for _ in range(4)])
        dones = np.array([len(self.player_decks[i]) == 0 or self.done for i in range(4)])
        infos = [{"state": self.game_state_info, "actions" : action}]
        return obs, share_obs, rewards, dones, infos, None, self.history
        
    def _set_reward(self,curr_player):
        '''
        setting rewards
        if terminating: winner team gets reward 1~3
        else: rewards 0
        '''
        rewards = [0] * self.n_agents

        if self.done:
            if len(self.cleared) == 2: # Must be a double-dweller
                rewards[self.cleared[0]] += 3
                rewards[self.cleared[1]] += 3
            elif (self.cleared[2] - self.cleared[0]) % 2 == 0:
                rewards[self.cleared[0]] += 2
                rewards[self.cleared[2]] += 2
            else:
                rewards[self.cleared[0]] += 1
                rewards[(self.cleared[0] + 2) % 4] += 1


        return rewards
    
    def _raise_error(self, errno, detail):
        raise Error(self.errset[errno]+": "+detail)
    
    def _end_game(self, fault_player):
        """
        Ending game due to player's illegal action.
        """
        # Assign heavy penalty to fault_player, zero to others
        print(fault_player,self.game_state_info)
        rewards = np.zeros(4, dtype=np.float32)
        rewards[fault_player] = 0
        self.done = True
        self.reward = {i: float(rewards[i]) for i in range(4)}  # optional bookkeeping
        # All players get same shared state and marked as done
        obs = [self.__get_obs(i) for i in range(4)]
        share_obs = [self.get_state() for _ in range(4)]
        dones = [True] * 4
        infos = [{"state": self.game_state_info}]
        available_actions = None

        return obs, share_obs, rewards, dones, infos, available_actions
    
    # def _get_obs(self, player):
    #     '''
    #     getting observation for player
    #     player: player_id (-1: all players)
    #     '''
    #     obs_set = {}
    #     for i in range(4):
    #         obs_set[i] = {
    #             "id": i,
    #             "level": self.level,
    #             "status": self.game_state_info,
    #             "deck": self.player_decks[i],
    #             "last_move": self.lastMove,
    #             "history": self.history,
    #             "reward": self.reward[i]
    #         }
    #     if player == -1:
    #         return obs_set
    #     else:
    #         return { player: obs_set[player] }
                
    def get_state(self):
        """Returns the global shared observation (state) for training the critic."""
        state = []
        for pid in range(4):
            hand_vec = np.zeros(108, dtype=np.float32)
            for card in self.player_decks[pid]:
                hand_vec[card] = 1.0
            state.append(hand_vec)

        played_vec = np.zeros(108, dtype=np.float32)
        for p in range(4):
            for card in self.played_cards[p]:
                played_vec[card] = 1.0
        state.append(played_vec)

        if self.lastMove['player'] == -1:
            last_move_id = 0.0
        else:
            last_move_id = self.lastMove['action_id'] / 365.0

        state.append(np.array([last_move_id], dtype=np.float32))

        state.append(np.array([self.round / 100.0], dtype=np.float32))

        cleared_vec = np.zeros(4, dtype=np.float32)
        for pid in self.cleared:
            cleared_vec[pid] = 1.0
        state.append(cleared_vec)

        level_vec = np.zeros(13, dtype=np.float32)
        if self.level in self.cardscale:
            level_vec[self.cardscale.index(self.level)] = 1.0
        state.append(level_vec)

        pass_vec = np.zeros(4, dtype=np.float32)
        if self.pass_on != -1:
            pass_vec[self.pass_on] = 1.0
        state.append(pass_vec)

        return np.concatenate(state).astype(np.float32)


    def __get_obs(self, agent_id):
        obs = []
        # 1. hand: binary vector (length 108)
        hand_vec = np.zeros(108, dtype=np.float32)
        for card in self.player_decks[agent_id]:
            hand_vec[card] = 1.0
        obs.append(hand_vec)

        # 2. played_cards: binary vector of all played cards
        played_vec = np.zeros(108, dtype=np.float32)
        for p in range(4):
            for card in self.played_cards[p]:
                played_vec[card] = 1.0
        obs.append(played_vec)

        # 3. last_move_id: float in [0, 1], -1 means no move yet → encode as 0.0
        if self.lastMove['player'] == -1:
            last_move_id = 0.0
        else:
            last_move_id = self.lastMove['action_id'] / 365.0  # Normalize

        obs.append(np.array([last_move_id], dtype=np.float32))
        
        # 5. turn: normalized round index
        turn = np.array([self.round / 100.0], dtype=np.float32)  # Assume max 100 rounds
        obs.append(turn)

        # 6. cleared: 1 if a player has cleared their cards
        cleared_vec = np.zeros(4, dtype=np.float32)
        for pid in self.cleared:
            cleared_vec[pid] = 1.0
        obs.append(cleared_vec)

        # 7. level: one-hot vector (13), position of self.level in self.cardscale
        level_vec = np.zeros(13, dtype=np.float32)
        if self.level in self.cardscale:
            level_index = self.cardscale.index(self.level)
            level_vec[level_index] = 1.0
        obs.append(level_vec)

        # 8. pass_on: one-hot of player who passed last
        pass_vec = np.zeros(4, dtype=np.float32)
        if self.pass_on != -1:
            pass_vec[self.pass_on] = 1.0
        obs.append(pass_vec)

        return np.concatenate(obs).astype(np.float32)



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
        '''
        Check the type of poker, return a tuple (type, point)
        '''
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
    
    
if __name__ == "__main__":
    env = GuanDanEnv()
    env.reset()
    test_data = [
        # ['h7'],
        # ['jo','jo'],
        
        # ['c8','d8','s8'], # 38
        # ['c0','d0','s0'], # 40
        # ['cA','dA','sA'], # 31
        # ['c2','d2','s2'], # 32
        
        # ['hA','h2','s3','h4','d5'], # 236
        # ['h2','s3','h4','d5','c6'], # 237
        # ['h9','s0','hJ','sQ','cK'], # 231
        # ['s0','sJ','hQ','dK','dA'], # 232
        # ['s0','sJ','hA','dK','dQ'], # 232
        # ['h5','s3','h2','d4','c6'], # 237
        
        # ['h0','s0','sJ','hQ','dJ','cQ'], # 209
        # ['dA','sA','s2','h3','d2','c3'], # 200
        # ['hA','sQ','sK','hQ','dK','cA'], # 211
        # ['h2','s2','s4','h3','d3','c4'], # 201
        # ['d0','s9','s8','h8','d9','c0'], # 207
        # ['h4','s5','s6','h5','d4','c6'], # 203
        
        # ['cK','dK','sK','cA','dA'], # 188
        # ['cA','dA','sA','c2','d2'], # 45
        # ['c0','d0','s0','cA','dA'], # 152
        
        # ['h9','d9','s9','d0','c0','h0'], # 220
        # ['hJ','dJ','sJ','d0','c0','h0'], # 221
        # ['hA','dA','sA','d2','c2','h2'], # 212
        # ['hA','dA','sA','dK','cK','hK'], # 224
        
        # ['jo','jo','jO','jO'],
        
        # ['hA','hA','dA','dA','sA'], #239
        # ['hA','hA','dA','dA','sA','cA'], #240
        # ['h2','h2','d2','d2','s2','c2'], #247
        # ['h3','h3','d3','d3','s3','c3'], #254
        # ['h2','h2','d2','d2','s2','c2','c2'], #248
        # ['h3','h3','d3','c3','c3'], #253
        # ['hJ','dJ','sJ','cJ','cJ'], #309
        
    ]
    for tt in test_data:
        hand_cards = [env.Utils.Poker2Num(p,[]) for p in tt]
        print("#"*10)
        print(env.get_action_id(hand_cards))
        print(*env.enumerate_legal_actions(hand_cards))
        for i in hand_cards:
            print(env.Utils.Num2Poker(i))
    