#!/usr/bin/env python3
from collections import Counter
import random
from copy import deepcopy
import time
from collections import defaultdict
import json
import numpy as np
import os
import warnings
    
class MyGuanDanEnv():
    '''
    Usage:
    Step1 Call GuanDanEnv() to create an instance
    Step2 Call GuanDanEnv.reset(config) to reset a match  
    '''
    def __init__(self):
        self.cardscale = ['A','2','3','4','5','6','7','8','9','0','J','Q','K']
        self.suitset = ['h','d','s','c']
        self.point_order = ['2', '3', '4', '5', '6', '7', '8', '9', '0', 'J', 'Q', 'K', 'A']
        # 普通牌型(不能比大小): 单牌、      对子、  三张、  顺子、      三带一对、  三顺、          连对
        self.Normaltypes = ("single", "pair", "three", "straight", "set", "three_straight", "triple_pairs")
        # 需要按顺序的牌型: 顺子、三顺、连对
        self.scaletypes = ("straight", "three_straight", "triple_pairs")
        self.Utils = Utils()
        self.level = None
        self.seed = None
        self.done = False
        self.game_state_info = "Init"
        self.cleared = [] # list of cleared players (have played all their decks)
        self.agent_names = ['player_%d' % i for i in range(4)]
        self.errset = {
            0: "Initialization Fault",
            1: "PlayerAction Fault",
            2: "Game Fault"
        }
        
    def reset(self, config={}):
        '''
        Call this function to start different matches
        @ config: contains match initalization info

        '''
        random.seed(time.time())
        self.level = self.cardscale[random.randint(0,len(self.cardscale)-1)]
        
        self.point_order = ['2', '3', '4', '5', '6', '7', '8', '9', '0', 'J', 'Q', 'K', 'A']
        self._set_level()
        self.total_deck = [i for i in range(108)]
        self.card_todeal = [i for i in range(108)]
        random.shuffle(self.card_todeal)
        self.player_decks = [self.card_todeal[dpos*27 : (dpos+1) * 27] for dpos in range(4)]
        self.done = False
        self.history = []  
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
        return self._get_obs(0) # Each match starts from the player on 0 position
        
    def step(self, response):
        self.round += 1
        self.reward = None
        curr_player = response['player']
        action = response['action']
        claim = response['claim']
        if not self._is_legal_claim(action, claim): # not a legal claim
            self.game_state_info = f"Player {curr_player}: ILLEGAL CLAIM"
            return self._end_game(curr_player)
        for poker_no in action: 
            if poker_no in self.player_decks[curr_player]:
                self.player_decks[curr_player].remove(poker_no)
                self.played_cards[curr_player].append(poker_no)
            else:
                self.game_state_info = f"Player {curr_player}: NOT YOUR POKER"
                return self._end_game(curr_player)
        cur_pokertype, cur_points = self._check_poker_type(claim)
        if cur_pokertype == 'invalid':
            self.game_state_info = f"Player {curr_player}: INVALID TYPE"
            return self._end_game(curr_player)
        if len(self.lastMove['action']) == 0: # first-hand
            if cur_pokertype == 'pass':
                self.game_state_info = f"Player {curr_player}: ILLEGAL PASS AS FIRST-HAND"
                return self._end_game(curr_player)
            self.lastMove = response
            self.pass_on = -1
        else:
            if cur_pokertype != 'pass': # if currplayer passes, do nothing
                last_pokertype, last_points = self._check_poker_type(self.lastMove['claim'])
                bigger = self._check_bigger(last_pokertype, last_points, cur_pokertype, cur_points)
                if bigger == "error":
                    self.game_state_info = f"Player {curr_player}: POKERTYPE MISMATCH"
                    return self._end_game(curr_player)
                if not bigger:
                    self.game_state_info = f"Player {curr_player}: CANNOT BEAT LASTMOVE"
                    return self._end_game(curr_player)
                self.lastMove = response
                self.pass_on = -1
        
        self.history.append(response)
        if len(self.player_decks[curr_player]) == 0: # Finishing this round
            self.cleared.append(curr_player)
            if len(self.cleared) == 3: # match sealed
                self.done = True
                self.game_state_info = "Finished"
            elif len(self.cleared) == 2 and (self.cleared[1] - self.cleared[0]) % 2 == 0:
                self.done = True
                self.game_state_info = "Finished"
            self.pass_on = curr_player
            
        self._set_reward()
        
        if not self.done:
            next_player = (curr_player + 1) % 4
            if next_player == self.pass_on: # Successfully pass to teammate
                next_player = (self.pass_on + 2) % 4
                self.lastMove = {
                    "player": -1,
                    "action": [],
                    "claim": []
                }
            while next_player in self.cleared:
                next_player = (next_player + 1) % 4
                if next_player == self.pass_on: # Successfully pass to teammate
                    next_player = (self.pass_on + 2) % 4
                    self.lastMove = {
                        "player": -1,
                        "action": [],
                        "claim": []
                    }
            if next_player == self.lastMove['player']:
                self.lastMove = {
                    "player": -1,
                    "action": [],
                    "claim": []
                }
            return self._get_obs(next_player)
        return self._get_obs(-1) # Done. Send a signal to all players     
        
    
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
            return "single", (poker[0][1], )
        if len(poker) == 2:
            if poker[0][1] == poker[1][1]:
                return "pair", (poker[0][1], )
            return "invalid", ()
        # 大于等于三张
        points = [p[1] for p in poker]
        cnt = Counter(points)
        vals = list(cnt.values())
        if len(poker) == 3:
            if "o" in points: 
                return "invalid", ()
            if vals.count(3) == 1:
                return "three", (points[0], )
            return "invalid", ()
        if len(poker) == 4: # should be a bomb
            if "o" in points or "O" in points: # should be a rocket
                if cnt["o"] == 2 and cnt["O"] == 2:
                    return "rocket", ("jo", )
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
                            return "straight_flush", ('0', )
                        return "straight", ('0', )
                sup_straight = [self.cardscale[self.cardscale.index(first)+i] for i in range(5)]
                if points == sup_straight:
                    if flush:
                        return "straight_flush", (first, )
                    return "straight", (first, )
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
                        return "three_straight", ('A', )
                    if ks == ['A', 'K']:
                        return "three_straight", ('K', )
                    return "invalid", ()
                if self.cardscale.index(ks[1]) - self.cardscale.index(ks[0]) == 1:
                    return "three_straight", (ks[0], )
            if vals.count(2) == 3:
                ks = []
                for k in list(cnt.keys()):
                    ks.append(k)
                ks.sort(key=lambda x: self.cardscale.index(x))
                if 'A' in ks:
                    if ks == ['A', 'Q', 'K']:
                        return "triple_pairs", ('Q', )
                    if ks == ['A', '2', '3']:
                        return "triple_pairs", ('A', )
                    return "invalid", ()
                pairs = [self.cardscale[self.cardscale.index(ks[0])+i] for i in range(3)]
                if ks == pairs:
                    return "triple_pairs", (ks[0], )
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
    

class Utils():
    
    def __init__(self):
        self.cardscale = ['A','2','3','4','5','6','7','8','9','0','J','Q','K']
        self.suitset = ['h','d','s','c']
        self.jokers = ['jo', 'jO']
    
    def Num2Poker(self, num: int):
        num_in_deck = num % 54
        if num_in_deck == 52:
            return "jo"
        if num_in_deck == 53:
            return "jO"
        # Normal cards:
        pokernumber = self.cardscale[num_in_deck // 4]
        pokersuit = self.suitset[num_in_deck % 4]
        return pokersuit + pokernumber
    
    def Poker2Num(self, poker: str, deck):
        num_in_deck = -1
        if poker[1] == "o":
            num_in_deck = 52
        elif poker[1] == "O":
            num_in_deck = 53
        else:
            num_in_deck = self.cardscale.index(poker[1])*4 + self.suitset.index(poker[0])
        if num_in_deck == -1:
            return -1
        if num_in_deck in deck:
            return num_in_deck
        return num_in_deck + 54
    
class Error(Exception):
    def __init__(self, ErrorInfo):
        self.ErrorInfo = ErrorInfo
    
    def __str__(self):
        return self.ErrorInfo  
    

class Env(MyGuanDanEnv):
    def __init__(self):
        super().__init__()

    def set_level(self, level:str):
        if level in self.cardscale:
            self.level = level
        else:
            self.level = '2'
            warnings.warn("ResetConfigWarning: Level configuration fault or no level designated.")
            
        self.point_order = ['2', '3', '4', '5', '6', '7', '8', '9', '0', 'J', 'Q', 'K', 'A']
        self._set_level()
    
    def reset_with_decks(self, decks, level:str='2'):
        self.reset()

        self.set_level(level)
        self.player_decks = decks

        return self._get_obs(0)


class CardClaim:
    def __init__(self, claim:list, env:Env):
        self.claim = claim
        self.env = env
        self.pokertype, self.points = env._check_poker_type(claim)
    def __lt__(self, other):
        return self.env._check_bigger(self.pokertype, self.points, other.pokertype, other.points)

class EnvUtils(Utils):
    def __init__(self):
        super().__init__()

    def Poker2Num_in_deck(self, poker: str):
        num_in_deck = -1
        if poker[1] == "o":
            num_in_deck = 52
        elif poker[1] == "O":
            num_in_deck = 53
        else:
            num_in_deck = self.cardscale.index(poker[1])*4 + self.suitset.index(poker[0])
        return num_in_deck
    
    def check_covering(self, claim:list, cards:list):
        pass

class Config:
    EPSILON_CLIP = 0.5
    EPSILON = 0.5
    GAMMA = 0.99
    LMBDA = 0.99
    ACTOR_LR = 5e-5
    CRITIC_LR = 5e-5

    STATE_DIM = 255 + 3 + 108 * 4 + 108
    GAME_STATE_DIM = 4 * (108 + 17 * 15)
    # 15个点数，加上pass和各种炸弹有17种牌型 (11 + 6) * 15
    ACTION_DIM = 17 * 15 # 255

    HIDDEN_DIM = 32

    EPOCHS = 10
    BATCH_SIZE = 512


    DEVICE = "cpu"
    EPOCH_NUMBER = 40000
    ONLINE = False



"""
原始观察数据格式:
只有自己的信息
i : 自己的player_id

obs_set[i] = {
            "id": i,
            "level": self.level,
            "status": self.game_state_info,
            "deck": self.player_decks[i],
            "last_move": self.lastMove,
            "history": self.history,
            "reward": self.reward[i]
        }

"""

class ObsData:
    def __init__(self, feature:list, legal_act:list, legal_act_mask, game_feature):
        self.feature = feature
        self.legal_act = legal_act
        self.legal_act_mask = legal_act_mask
        self.game_feature = game_feature

def reward_shaping(
    player_id, prev_obs:dict, curr_obs:dict, prev_obs_data:ObsData, curr_obs_data:ObsData
):
    # 手牌数量
    prev_deck_len = len(prev_obs[player_id]["deck"])
    curr_deck_len = len(curr_obs[player_id]["deck"])

    # 奖励
    # 手牌数量变化
    deck_len_change = prev_deck_len - curr_deck_len

    # 可以出的牌型
    legal_act_change = 0

    
    # 完成对局奖励
    reward_win = 0
    # 如果游戏结束，根据队伍胜负给予奖励
    if curr_obs[player_id]["status"] == "Finished":
        teammates = [player_id, (player_id + 2) % 4]
        opps = [opp for opp in range(4) if opp not in teammates]
        env_reward = [curr_obs[i]["reward"] for i in range(4)]
        for i, e_r in enumerate(env_reward):
            sign = 1 if i in teammates else -1
            reward_win += sign * e_r


    # 考虑出牌的牌型奖励
    # 假设在 curr_obs 中有记录出的牌型
    # 这里需要根据具体实现来添加
    
    # 奖励计算
    reward_power = {
        "deck_len_change" : 0.1,
        "reward_win" : 1,
    }
    reward = deck_len_change * reward_power["deck_len_change"] \
        + reward_win * reward_power["reward_win"]
    return reward

# 只返回类型和点数，不用返回具体的牌
def _get_legal_act(deck, covering:str, env:Env, utils:EnvUtils):
    hand_cards = [utils.Num2Poker(p) for p in deck]
    normal_cards = [card for card in hand_cards if card != covering and card[1] not in ['o', 'O']]
    points = [card[1] for card in hand_cards]
    normal_points = [card[1] for card in normal_cards]
    point_counts = Counter(points)
    normal_point_counts = Counter(normal_points)
    covering_cnt = hand_cards.count(covering)
    # 2 - A o O
    point_order = env.point_order
    cardscale_extra = utils.cardscale + ["A"]

    result = []
    
    # 单张
    for point in point_order:
        if point_counts.get(point, 0) > 0:
            result.append(("single", (point, )))

    # 对子
    for point in point_order:
        if (covering_cnt + point_counts.get(point, 0) >= 2 and point not in [covering[1], "o", "O"]) \
            or point_counts.get(point, 0) >= 2:
            result.append(("pair", (point, )))

    # 三条
    for point in point_order:
        if (covering_cnt + point_counts.get(point, 0) >= 3 and point not in [covering[1], "o", "O"]) \
            or point_counts.get(point, 0) >= 3:
            result.append(("three", (point, )))

    # 同花顺 杂花顺
    suits = [set([card[0] for card in normal_cards if card[1] == point]) for point in cardscale_extra]
    for start in range(10):
        end = start + 5
        sub_points = cardscale_extra[start:end]
        sub_suits = suits[start:end]
        include_num = sum([1 if normal_point_counts.get(point, 0) > 0 else 0 for point in sub_points])
        suits_num = [sum([1 if suit in s else 0 for s in sub_suits]) for suit in utils.suitset]
        max_suits_num = max(suits_num)
        if include_num == 5 and max_suits_num == 5:
            result.append(("straight_flush", (sub_points[0], )))
        elif include_num + covering_cnt >= 5 and max_suits_num + covering_cnt >= 5:
            result.append(("straight_flush", (sub_points[0], )))
            # Log(f"check out straight flush: {sub_points[0]} in hand cards: {hand_cards}", "episode")
        elif include_num + covering_cnt >= 5:
            result.append(("straight", (sub_points[0], )))


    # 三带二
    for point in env.point_order[:-2]:
        if (covering_cnt + normal_point_counts.get(point, 0) >= 3 and point not in ["o", "O"]):
            covering_consumed = 3 - normal_point_counts.get(point, 0)
            covering_consumed = max(covering_consumed, 0)
            for sub_point in point_order:
                if sub_point == point:
                    continue
                if normal_point_counts.get(sub_point, 0) >= 2:
                    result.append(("set", (point, sub_point)))
                    break
                elif normal_point_counts.get(sub_point, 0) + covering_cnt - covering_consumed >= 2 and sub_point not in [covering[1], "o", "O"]:
                    result.append(("set", (point, sub_point)))
                    break

    # 三顺
    for start in range(len(utils.cardscale)):
        end = start + 2
        sub_points = cardscale_extra[start:end]
        include_num = sum([min(normal_point_counts.get(point, 0), 3) for point in sub_points])
        if include_num + covering_cnt >= 6:
            result.append(("three_straight", (sub_points[0], )))

    # 连对
    for start in range(len(utils.cardscale) - len(['K'])):
        end = start + 3
        sub_points = cardscale_extra[start:end]
        include_num = sum([min(normal_point_counts.get(point, 0), 2) for point in sub_points])
        if include_num + covering_cnt >= 6:
            result.append(("triple_pairs", (sub_points[0], )))
    
    # 炸弹
    for point in env.point_order[:-2]:
        if covering_cnt + normal_point_counts.get(point, 0) >= 4:
            for num in range(4, covering_cnt + normal_point_counts.get(point, 0) + 1):
                result.append(("bomb", (num, point)))

    # 王炸
    if point_counts.get("o", 0) + point_counts.get("O", 0) == 4:
        result.append(("rocket", ("jo", )))

    return result

def observation_process(raw_obs, game_obs=None) -> ObsData:
    player_id:int = raw_obs["id"]
    player_level:str = raw_obs["level"]
    player_status:str = raw_obs["status"]
    player_deck:list = raw_obs["deck"]
    last_move:dict = raw_obs["last_move"]
    history:list = raw_obs["history"]
    reward:float = raw_obs["reward"]

    env = Env()
    env.set_level(player_level)
    utils = EnvUtils()
    player_poker = [utils.Num2Poker(p) for p in player_deck]
    covering = "h" + player_level
    covering_cnt = player_poker.count(covering)

    # 点数顺序，包括普通牌和特殊的王
    point_order = env.point_order
    # print(point_order)
    # 玩家的手牌
    hand_cards = player_poker
    # 每个点数的数量
    points = [card[1] for card in hand_cards]
    point_counts = Counter(points)

    one_hot_deck = list(get_one_hot(player_deck, 108))
    # 手牌中每个点数的数量，不区分花色 15
    # point_count_in_hand = [point_counts.get(point, 0) for point in point_order]
    # 手牌one_hot 108
    # hand_cards_one_hot = get_one_hot(player_deck, 108)

    # 是否有炸弹（4个相同的点数，不包括王） 1
    # has_bomb = 0
    # for point, count in point_counts.items():
    #     if point not in ["o", "O"] and (count >= 4 or (count + covering_cnt >= 4 and point != covering[1])):
    #         has_bomb = 1
    #         break
    
    # 是否有王炸（两张小王和两张大王） 1
    # has_rocket = 0
    # if point_counts.get("o", 0) == 2 and point_counts.get("O", 0) == 2:
    #     has_rocket = 1

    # 获取上家出的牌型和点数 point, bomb_level 2
    last_move_feature = [-1, -1, -1]
    if last_move["player"] != -1:
        last_move_player = (4 + player_id - last_move["player"]) % 4
        last_pokertype, last_points = env._check_poker_type(last_move["claim"])
        if last_pokertype in ["single", "pair", "three", "set"]:
            last_move_feature = [last_move_player, point_order.index(last_points[0]), -1]
        elif last_pokertype in ["straight", "three_straight", "triple_pairs"]:
            last_move_feature = [last_move_player, utils.cardscale.index(last_points[0]), -1]
        elif last_pokertype in ["bomb"]:
            last_move_feature = [last_move_player, point_order.index(last_points[1]), last_points[0] - 4]
        elif last_pokertype in ["rocket"]:
            last_move_feature = [last_move_player, -1, 7]
        elif last_pokertype in ["straight_flush"]:
            last_move_feature = [last_move_player, -1, 8]
        else:
            last_move_feature = [last_move_player, -1, -1]
    else:
        last_move_feature = [-1, -1, -1]

    # 游戏状态信息 1
    # game_state = [len(history)]
    
    # 四个人分别已经打过的牌
    player_history = [[], [], [], []]
    for h in history:
        if h["player"] != -1:
            # 自己在第一个，剩下的依次排列
            player_history[(h["player"] + 4 - player_id) % 4] + h["action"]
    
    player_used_decks = []
    for ph in player_history:
        player_used_decks += list(get_one_hot(ph, 108))



    # 四个人剩余手牌的数量 4
    # player_used_num = [27 - len(ph) for ph in player_history]

    # 归一化处理
    # feature = np.array(feature)
    # feature = (feature - np.min(feature)) / np.max(feature)

    legal_act = _get_legal_act(tuple(player_deck), covering, env, utils)
    # print(legal_act)

    Log(f"last move: {last_move}", "episode")
    
    legal_act_mask = get_leagl_act_mask(legal_act, point_order, env, last_move)
    
    #  255 + 3 + 108 * 4 + 108
    feature = (
        one_hot_deck
        # point_count_in_hand
        # + [has_bomb, has_rocket]
        + last_move_feature
        # + game_state
        # + player_used_num
        + player_used_decks
        + list(legal_act_mask)
    )

    if len(feature) != Config.STATE_DIM:
        print(feature)
        raise Exception("feature length is not correct")

    game_feature = None
    if game_obs is not None:
        game_feature = [] # 4 * (108 + 17 * 15)
        for p in range(4):
            p_id = (player_id + p) % 4
            if p_id == player_id:
                game_feature += one_hot_deck
                game_feature += list(legal_act_mask)
            else:
                game_feature += list(get_one_hot(game_obs[p_id]["deck"], 108))
                game_feature += list(get_leagl_act_mask(
                    _get_legal_act(game_obs[p_id]["deck"], covering, env, utils)
                    , point_order, env))


    return ObsData(feature, legal_act, legal_act_mask, game_feature)

def get_action(deck, act, covering, env, utils):
    return LegalAction(deck, act, covering, env, utils).get_action()


def get_leagl_act_mask(legal_act, point_order, env, last_move):

    if last_move["player"] != -1:
        # 只能出上轮出过的牌型，且要比上一轮的大
        last_move_type, last_move_point = env._check_poker_type(last_move["claim"])
        legal_act = [("pass", tuple()), ] + [act for act in legal_act if env._check_bigger(last_move_type, last_move_point, act[0], act[1]) == True]


    legal_act_mask = np.zeros((17, 15))
    card_types = ["pass", "single", "pair", "three", "straight", "set", "three_straight", "triple_pairs", "rocket", "straight_flush", "bomb"]
    point_range = [15,      15,      15,     13,         10,       13,          13,            12,             15,    10,             13 * 7]
    for act in legal_act:
        type_index = card_types.index(act[0])
        if act[0] in ["pass", "rocket"]: # 全1或全0
            legal_act_mask[type_index] = 1
        elif act[0] in ["bomb"]: # 特殊处理
            legal_act_mask[type_index + act[1][0] - 4][point_order.index(act[1][1])] = 1
        elif act[0] in env.scaletypes or act[0] == "straight_flush":
            legal_act_mask[type_index][env.cardscale.index(act[1][0])] = 1
        else:
            legal_act_mask[type_index][point_order.index(act[1][0])] = 1

    return legal_act_mask.reshape(-1)

class LegalAction:
    def __init__(self, deck, act, covering, env, utils):
        self.deck = deck
        self.act = act
        self.covering = covering
        self.env = env
        self.utils = utils
        self.hand_cards = [utils.Num2Poker(p) for p in self.deck]
        self.covering_deck = [deck[i] for i, c in enumerate(self.hand_cards) if c == self.covering]
        self.normal_cards = [card for card in self.hand_cards if card != self.covering]
        self.points = [card[1] for card in self.hand_cards]
        self.normal_points = [card[1] for card in self.normal_cards]
        self.points_counts = Counter(self.points)
        self.normal_points_counts = Counter(self.normal_points)
        self.covering_cnt = self.hand_cards.count(self.covering)
        self.point_order = self.env.point_order
        self.cardscale_extra = self.utils.cardscale + ["A"]

        self.card_types = ["pass", "single", "pair", "three", "straight", "set", "three_straight", "triple_pairs", "rocket", "straight_flush", "bomb"]
        self.action_type_index = self.act // 15
        self.action_point_index = self.act % 15
        
        self.action_type = self.card_types[min(self.action_type_index, len(self.card_types) - 1)]
        Log(f"action to response: {self.action_type}, hand cards: {self.hand_cards}, covering count: {self.covering_cnt}", "episode")
        # print(self.act, self.action_type_index, self.action_type)
        # print(self.point_order)

    def get_action(self):
        if self.action_type == "pass":
            return [[], []]
        elif self.action_type == "rocket":
            rocket_deck = [52, 53, 106, 107]
            return [rocket_deck, rocket_deck]
        elif self.action_type == "bomb":
            bomb_level = 5 + self.action_type_index - len(self.card_types)
            bomb_point = self.point_order[self.action_point_index]
            return self.get_same_action(bomb_point, bomb_level)
        elif self.action_type in ["single", "pair", "three"]:
            point = self.point_order[self.action_point_index]
            point_num = ["single", "pair", "three"].index(self.action_type) + 1
            return self.get_same_action(point, point_num)
        elif self.action_type == "set":
            point = self.point_order[self.action_point_index]
            point_num = 3
            three_act = self.get_same_action(point, point_num)
            covering_consumed = sum([1 if n in self.covering_deck else 0 for n in three_act[0]])
            two_act = self.get_min_pair_without_some_cards(point, covering_consumed)
            return [three_act[0] + two_act[0], three_act[1] + two_act[1]]
        elif self.action_type in ["straight", "three_straight", "triple_pairs"]:
            straight_start = self.action_point_index
            if self.action_type == "straight":
                straight_end = straight_start + 5
                point_num = 1
            elif self.action_type == "three_straight":
                straight_end = straight_start + 2
                point_num = 3
            elif self.action_type == "triple_pairs":
                straight_end = straight_start + 3
                point_num = 2
            straight_points = self.cardscale_extra[straight_start:straight_end]
            result = [[], []]
            for point in straight_points:
                act = self.get_same_action(point, point_num)
                for n in act[0]:
                    if n in self.covering_deck:
                        self.covering_cnt -= 1
                        self.covering_deck.remove(n)
                result[0].extend(act[0])
                result[1].extend(act[1])
            return result
        elif self.action_type == "straight_flush":
            flush_start = self.action_point_index
            flush_end = flush_start + 5
            flush_points = self.cardscale_extra[flush_start:flush_end]
            suits = [set([card[0] for card in self.normal_cards if card[1] == point and card != self.covering]) \
                    for point in flush_points]
            suits_num = [sum([1 if suit in s else 0 for s in suits]) for suit in self.utils.suitset]
            max_suits_num = max(suits_num)
            covering_consumed = 5 - max_suits_num
            suit = self.utils.suitset[suits_num.index(max_suits_num)]
            result = [self.covering_deck[:covering_consumed], []]
            for point, suit_set in zip(flush_points, suits):
                card = suit + point
                if suit in suit_set:
                    num = self.deck[self.hand_cards.index(card)]
                    result[0].append(num)
                    result[1].append(num)
                else:
                    result[1].append(self.utils.Poker2Num_in_deck(card))
            return result

    def get_same_action(self, point, point_num):
        single_deck = [self.deck[i] for i, c in enumerate(self.hand_cards) if c[1] == point and c != self.covering]
        # print("same action", point_num, self.normal_points_counts.get(point, 0), point)
        if self.normal_points_counts.get(point, 0) >= point_num:
            return [single_deck[:point_num], single_deck[:point_num]]
        else:
            target_num = single_deck[0] if len(single_deck) > 0 else self.utils.Poker2Num_in_deck("h" + point)
            covering_used = point_num - self.normal_points_counts.get(point, 0)
            return [single_deck + self.covering_deck[:point_num - self.normal_points_counts.get(point, 0)]
            , single_deck + [target_num] * (point_num - self.normal_points_counts.get(point, 0))]

    def get_min_pair_without_some_cards(self, point, covering_consumed):
        for covering_target_use in range(self.covering_cnt - covering_consumed + 1):
            for pair_point in self.point_order:
                normal_point_num = self.normal_points_counts.get(pair_point, 0)
                if pair_point == point:
                    normal_point_num -= 3
                if normal_point_num + covering_target_use < 2:
                    continue
                two_deck = [self.deck[i] for i, c in enumerate(self.hand_cards) \
                    if c[1] == pair_point and c != self.covering]
                covering_target_deck = [two_deck[0] if len(two_deck) > 0 \
                    else self.utils.Poker2Num_in_deck("h" + pair_point)] \
                        * covering_target_use
                if covering_target_use > 0:
                    return [
                        two_deck[-(2 - covering_target_use):] + self.covering_deck[-covering_target_use:],
                        two_deck[-(2 - covering_target_use):] + covering_target_deck
                    ]
                else:
                    return [two_deck[-2:], two_deck[-2:]]


class RuleBasedPolicy:
    def take_action(self, raw_obs):
        # 返回 15 * 17 = 255 维向量
        # 15个点数，加上pass和各种炸弹有17种牌型 (11 + 6) * 15

        player_id:int = raw_obs["id"]
        player_level:str = raw_obs["level"]
        player_status:str = raw_obs["status"]
        player_deck:list = raw_obs["deck"]
        last_move:dict = raw_obs["last_move"]
        history:list = raw_obs["history"]
        reward:float = raw_obs["reward"]

        deck = player_deck

        env = Env()
        env.set_level(player_level)
        utils = EnvUtils()
        player_poker = [utils.Num2Poker(p) for p in player_deck]
        covering = "h" + player_level
        covering_cnt = player_poker.count(covering)

        # 点数顺序，包括普通牌和特殊的王
        point_order = env.point_order
        # print(point_order)
        # 玩家的手牌
        hand_cards = player_poker
        # 每个点数的数量
        points = [card[1] for card in hand_cards]
        normal_points = [card[1] for card in hand_cards if card != covering]
        point_counts = Counter(points)
        normal_counts = Counter(normal_points)

        last_move_type, last_move_point = env._check_poker_type(last_move["claim"])
        if last_move["player"] == (player_id + 2) % 2:
            if last_move_type in ["straight_flush", "rocket", "bomb"]:
                return 0
            elif last_move_type in env.scaletypes and env.cardscale.index(last_move_point[0]) >= 9:
                return 0
            elif point_order.index(last_move_point[0]) >= 9:
                return 0
        
        _legal_act = _get_legal_act(deck, covering, env, utils)
        legal_act_mask = get_leagl_act_mask(_legal_act, point_order, env, last_move)


        # 概率分五挡
        # 第一档: 推荐打出的普通牌型，即可打出最多牌的普通牌型，例如不能组成其他牌型的最小单牌 4-5 分为7*15=105个数
        # 第二档: 可以打出的特殊牌型，炸弹等 3-4 分为8*15=120约为128个数
        # 第三挡: 不推荐打出的特殊牌型 2-3 分为8*15=120个数
        # 第三档: 不推荐打出的普通牌型，即要拆其他牌型的牌，例如可以组成对子的单牌，三带二也放这类中 1-2 分为7*15=105个数
        # 最后一档: 过牌 1

        result = np.zeros((17, 15))
        card_types = ["pass", "single", "pair", "three", "straight", "set", "three_straight", "triple_pairs", "rocket", "straight_flush", "bomb"]
        legal_act = []
        for act in _legal_act:
            type_index = card_types.index(act[0])
            if act[0] in ["pass", "rocket"]: # 全1或全0
                legal_act.append(list(act) + [type_index * 15])
            elif act[0] in ["bomb"]: # 特殊处理
                legal_act.append(list(act) + [(type_index + act[1][0] - 4) * 15 + point_order.index(act[1][1])])
            elif act[0] in env.scaletypes or act[0] == "straight_flush":
                legal_act.append(list(act) + [type_index * 15 + env.cardscale.index(act[1][0])])
            else:
                legal_act.append(list(act) + [type_index * 15 + point_order.index(act[1][0])])
        
        card_order = ["straight_flush", "rocket", "bomb", "three_straight", "triple_pairs", "straight", "three", "pair", "single", "set", "pass"]
        low_card = set()
        for card_type in card_order:
            type_index = card_types.index(card_type)
            if card_type == "pass": # 过牌是1
                result[type_index] = 1
            elif card_type == "straight_flush": # 同花顺是最大的，所以推荐最后出
                result[type_index] = np.linspace(3 + 1 / 256, 3 + 16 / 256, 15, endpoint=False)[::-1]
                # 将所有同花顺所需牌加入low_card列表中
                for act in legal_act:
                    if act[0] == "straight_flush":
                        d, cd = get_action(deck, act[2], covering, env, utils)
                        if any([p in low_card for p in d]):
                            continue
                        low_card |= set(d)
            elif card_type == "rocket": # 火箭只排在同花顺之下，而且不会拆牌
                result[type_index] = 3 + 16 / 256
                for act in legal_act:
                    if act[0] == "rocket":
                        d, cd = get_action(deck, act[2], covering, env, utils)
                        if any([p in low_card for p in d]):
                            continue
                        low_card |= set(d)
            elif card_type == "bomb": # 炸弹是特殊牌型级别最低的
                for act in legal_act:
                    if act[0] == "bomb":
                        num = act[1][0]
                        point = act[1][1]
                        d, cd = get_action(deck, act[2], covering, env, utils)
                        # 检查是否需要拆牌
                        if normal_counts.get(point, 0) == num and not any([p in low_card for p in d]):
                            card_level = 4
                            low_card |= set(d)
                        else:
                            card_level = 3 - 1 / 256
                        result[type_index + num - 4][point_order.index(point)] = card_level - (num - 4) * 15 / 256 - point_order.index(point) / 256
            elif card_type in env.scaletypes:
                for act in legal_act:
                    if act[0] == card_type:
                        d, cd = get_action(deck, act[2], covering, env, utils)
                        if not any([p in low_card for p in d]):
                            card_level = 5
                            low_card |= set(d)
                        else:
                            card_level = 2
                        result[type_index][env.cardscale.index(act[1][0])] = card_level - env.cardscale.index(act[1][0]) * 15 / 256 - card_order.index(card_type) / 256
            else:
                for act in legal_act:
                    if act[0] == card_type:
                        d, cd = get_action(deck, act[2], covering, env, utils)
                        if not any([p in low_card for p in d]):
                            card_level = 5
                            low_card |= set(d)
                        else:
                            card_level = 2
                        result[type_index][point_order.index(act[1][0])] = card_level - point_order.index(act[1][0]) * 15 / 256 - card_order.index(card_type) / 256

        result = result.reshape(-1)
        result *= legal_act_mask
        result = result / np.sum(result)
        return result


def Log(message:str):
    Log(message, "log")

def Log(message:str, log_type:str="log"):
    if True:
        return
    file_name = f"./log/{log_type}.txt"
    if not os.path.exists("./log/"):
        os.mkdir("./log/")
    message = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    with open(file_name, "a") as f:
        f.write(f"{message}\n")
def get_one_hot(positions, max_len:int) -> np.array:
    one_hot = np.zeros((max_len, ))
    for pos in positions:
        one_hot[pos] = 1
    return one_hot



################
################


# 导入agent
agent = RuleBasedPolicy()

# 全局变量
player_id = 0
deck = []
level = 'A'
first = 0
last = 3
tribute = 0
resist = False
tribute_cards = {}
return_cards = {}

def get_full_input():
    with open('/data/wmqTemp/GuanDan/Rule_Based_Bot_yqb/testAI2.json', 'r') as f:
        return json.load(f)
    # return json.loads(input())

full_json = get_full_input()

print(type(full_json))

self_round_num = 0

def get_input():
    global self_round_num, full_json
    self_round_num += 1
    request = full_json['requests'][self_round_num - 1]
    if len(full_json['responses']) >= self_round_num:
        response = full_json['responses'][self_round_num - 1]
    else:
        response = []
    return request, response

def output(response: list):
    response = {'response' : response}
    print(json.dumps(response))

response = []

# 循环输入
raw_obs = {
            "id": player_id, # 不变
            "level": level, # 不变
            "status": "Running", # 不变
            "reward": 0,  # 不变

            "deck": deck,
            "last_move": {
                'player': -1, # the first round
                'action': [], 
                'claim': []
                },
            "history": [], # list[response] response: {id:int, action:[], claim:[]}
        }
last_move = {
            'player': -1, # the first round
            'action': [], 
            'claim': []
        }
JsonInput = {}
ResponseOutput = []

isDealTribute = False

for i in range(len(full_json['requests'])):
    # JsonInput, ResponseOutput = get_input()
    
    self_round_num += 1
    request = full_json['requests'][self_round_num - 1]
    if len(full_json['responses']) >= self_round_num:
        response = full_json['responses'][self_round_num - 1]
    else:
        response = []
        
    JsonInput = request
    ResponseOutput = response
    
    if JsonInput['stage'] == 'deal':
        # 第一次输入
        StartInput = JsonInput
        # 配置一部分全局变量
        player_id = StartInput['your_id']
        raw_obs["id"] = player_id
        next_player_id = (player_id + 1) % 4
        deck = StartInput['deliver']
        raw_obs["deck"] = deck
        level = StartInput['global']['level']
        raw_obs["level"] = level
        first = StartInput['global']['first'] if StartInput['global']['first'] is not None else 0
        last = StartInput['global']['last'] if StartInput['global']['last'] is not None else 3
        tribute = StartInput['global']['tribute']
    elif JsonInput['stage'] == 'tribute':
        # 第二次输入
        SecondInput = JsonInput
        # 配置一部分全局变量
        resist = SecondInput['global']['resist']
    elif JsonInput['stage'] == 'return':
        # 第二次输入
        SecondInput = JsonInput
        # 配置一部分全局变量
        resist = SecondInput['global']['resist']
        tribute_cards = SecondInput['global']['tribute_cards']
        return_cards = SecondInput['global']['return_cards']
    else:
        # 第三次及以后输入
        CircleInput = JsonInput
        if not isDealTribute:
            isDealTribute = True
            if CircleInput['global']['tribute_cards'] is not None and len(CircleInput['global']['tribute_cards']) > 0:
                env = Env()
                utils = EnvUtils()
                env.set_level(level)
                tribute_cards = CircleInput['global']['tribute_cards']
                return_cards = CircleInput['global']['return_cards']
                # 判断自己是进贡还是还贡
                if str(player_id) in tribute_cards.keys():
                    # 进贡
                    deck.remove(tribute_cards[str(player_id)])
                    # 判断给自己还贡的牌
                    if len(tribute_cards) == 1:
                        deck.append(list(return_cards.values())[0][0])
                    else:
                        # 判断自己进贡的牌是大牌还是小牌
                        self_tribute_card = utils.Num2Poker(tribute_cards[str(player_id)])
                        teammate_tribute_card = utils.Num2Poker(tribute_cards[str((player_id + 2) % 4)])
                        if env.point_order.index(self_tribute_card[1]) == env.point_order.index(teammate_tribute_card[1]):
                            self_big = last == player_id
                        else:
                            self_big = env.point_order.index(self_tribute_card[1]) > env.point_order.index(teammate_tribute_card[1])
                        return_num = return_cards[str(first)] if self_big else return_cards[str((first + 2) % 4)]
                        deck.append(return_num)
                else:
                    # 还贡
                    deck.remove(return_cards[str(player_id)])
                    # 判断给自己进贡的牌
                    if len(tribute_cards) == 1:
                        deck.append(list(tribute_cards.values())[0][0])
                    else:
                        # 判断给自己进贡的牌是大牌还是小牌
                        self_big = first == player_id
                        last_tribute_card = utils.Num2Poker(tribute_cards[str(last)])
                        other_tribute_card = utils.Num2Poker(tribute_cards[str((last + 2) % 4)])
                        if env.point_order.index(last_tribute_card[1]) == env.point_order.index(other_tribute_card[1]):
                            tribute_num = tribute_cards[str(last)] if self_big else tribute_cards[str((last + 2) % 4)]
                        else:
                            big_id = last if env.point_order.index(last_tribute_card[1]) > tribute_cards[str((last + 2) % 4)] else (last + 2)%4
                            tribute_num = tribute_cards[str(big_id)] if self_big else tribute_cards[str((big_id + 2) % 4)]
                        deck.append(tribute_num)
        # 找到last_move, history
        last_move = {
                    'player': -1, # the first round
                    'action': [], 
                    'claim': []
                    }
        done = CircleInput['done'] # 已经出完的玩家id list[int]
        pass_on = CircleInput['pass_on'] # 出完的玩家id int -1
        raw_history = CircleInput['history'] # list[response] response: [action, claim] 默认为[]
        h_player_id = player_id
        for resp in raw_history:
            # print(resp)
            if resp == [] or resp == [[], []]:
                h_player_id = (h_player_id + 1) % 4
                continue
            raw_obs["history"].append({"player": h_player_id, "action": resp['response'][0], "claim": resp['response'][1]}) ######
            if h_player_id != player_id:
                last_move = {"player": h_player_id, "action": resp['response'][0], "claim": resp['response'][1]}    #######
            h_player_id = (h_player_id + 1) % 4
        
        raw_obs["last_move"] = last_move

        # 获取该回合的输出
        action = ResponseOutput

        # 更新deck
        if action != []:
            for num in action[0]:
                deck.remove(num)
                raw_obs["deck"] = deck

if JsonInput['stage'] == 'play':
    # 计算得到下一步要出的牌
    curr_obs = raw_obs

    action = agent.take_action(curr_obs)
    action = np.argmax(action)
    env = Env()
    env.set_level(level)
    action_r = get_action(curr_obs["deck"], action, "h" + level, env, EnvUtils())
    response = [] if action_r == [[],[]] else action_r
    output(response)
elif JsonInput['stage'] == 'tribute':
    # 进贡
    # 输出最大的一张手牌（配子除外）
    if resist:
        output([])
    else:
        env = Env()
        utils = EnvUtils()
        env.set_level(level)
        hand_cards = [(i, utils.Num2Poker(p)) for i, p in enumerate(deck)]
        hand_cards.sort(key=lambda x: env.point_order.index(x[1][1]), reverse=True)
        for i, card in hand_cards:
            if card != "h" + level:
                response = [deck[i]]
                break
        output(response)
elif JsonInput['stage'] == 'return':
    # 还贡
    # 输出最小的一张手牌
    if resist:
        output([])
    else:
        env = Env()
        utils = EnvUtils()
        env.set_level(level)
        hand_cards = [(i, utils.Num2Poker(p)) for i, p in enumerate(deck)]
        hand_cards.sort(key=lambda x: env.point_order.index(x[1][1]), reverse=False)
        for i, card in hand_cards:
            if card != "h" + level:
                response = [deck[i]]
                break
        output(response)
else:
    output([])
    
