import os
import sys
import random
import numpy as np
import json
from collections import Counter

cardscale = ['A','2','3','4','5','6','7','8','9','0','J','Q','K']
suitset = ['h','d','s','c']
jokers = ['jo', 'jO']
pointorder = ['2','3','4','5','6','7','8','9','0','J','Q','K','A']
Normaltypes = ("single", "pair", "three", "straight", "set", "three_straight", "triple_pairs")
scaletypes = ("straight", "three_straight", "triple_pairs")
errored = [None for i in range(4)]

###################################################牌型编码###################################################
# 红桃 方块 黑桃 草花
# 3 4 5 6 7 8 9 10 J Q K A 2 joker & Joker
# (0-hA 1-dA 2-sA 3-cA) (4-h2 5-d2 6-s2 7-c2) …… 52-joker 53-Joker
# 请注意：10记为0
# 共2副108张
###################################################牌型编码###################################################
def num2Poker(num): # num: int-[0,107]
    # Already a poker
    if type(num) is str and (num in jokers or (num[0] in suitset and num[1] in cardscale)):
        return num
    # Locate in 1 single deck
    NumInDeck = num % 54
    # joker and Joker:
    if NumInDeck == 52:
        return "jo"
    if NumInDeck == 53:
        return "jO"
    # Normal cards:
    pokernumber = cardscale[NumInDeck // 4]
    pokersuit = suitset[NumInDeck % 4]
    return pokersuit + pokernumber

def Poker2Num(poker, deck): # poker: str
    NumInDeck = -1
    if poker[1] == "o":
        NumInDeck = 52
    elif poker[1] == "O":
        NumInDeck = 53
    else:
        NumInDeck = cardscale.index(poker[1])*4 + suitset.index(poker[0])
    if NumInDeck in deck:
        return NumInDeck
    else:
        return NumInDeck + 54

def isValidTribute(allo: list, poker: int, level: str):
    # allo: deck of the player
    if poker not in allo:
        return False
    deck = [num2Poker(p) for p in allo]
    poker = num2Poker(poker)
    if "jO" in deck:
        if poker != "jO":
            return False
        return True
    elif "jo" in deck:
        if poker != "jo":
            return False
        return True
    else:
        covering = "h" + level
        points = [p[1] for p in deck]
        point_alldiff = list(set(points))
        if level in points:
            if covering not in deck:
                if poker[1] != level:
                    return False
                return True
            with_level = [p for p in deck if p[1] == level] # make sure there's no other levels
            level_suit = [p[0] for p in with_level]
            if 'd' in level_suit or 's' in level_suit or 'c' in level_suit:
                if poker[1] != level:
                    return False
                return True
            point_alldiff.remove(level)
        point_alldiff.sort(key=lambda x: pointorder.index(x), reverse=True)
        if poker[1] != point_alldiff[0]:
            return False
        return True   

def isValidReturn(allo: list, poker: int, level: str):
    if poker not in allo:
        return False
    poker = num2Poker(poker)
    if level == '9':
        if pointorder.index(poker[1]) > pointorder.index('8'):
            return False
        return True
    if pointorder.index(poker[1]) > pointorder.index('9'):
        return False
    return True


def checkPokerType(poker: list):
    if poker == []:
        return "pass", ()
    # covering = "h" + level
    poker = [num2Poker(p) for p in poker]
    if len(poker) == 1:
        return "single", (poker[0][1])
    if len(poker) == 2:
        if poker[0][1] == poker[1][1]:
            if poker[0][0] == 'j': # Jokers
                if poker[0] == poker[1]:
                    return "pair", (poker[0][1]) # 大小王用"jo""jO"区分
                else: return "invalid", ()
            else: return "pair", (poker[0][1])
    points = [p[1] for p in poker]
    cnt = Counter(points)
    vals = list(cnt.values())
    if len(poker) == 3:
        if "o" in points or "O" in points: 
            return "invalid", ()
        if vals.count(3) == 1:
            return "three", (points[0])
    if len(poker) == 4: # should be a bomb
        if "o" in points or "O" in points: # should be a rocket
            if cnt["o"] == 2 and cnt["O"] == 2:
                return "rocket", ("jo")
            else:
                return "invalid", ()
        if vals.count(4) == 1:
            return "bomb", (4, points[0])
    if len(poker) == 5: # could be straight, straight flush, three&two or bomb
        if vals.count(5) == 1:
            return "bomb", (5, points[0])
        if vals.count(3) == 1:
            if vals.count(2) == 1: # set: 三带二 or bomb with 2 coverings
                three = ''
                two = ''
                for k in list(cnt.keys()):
                    if cnt[k] == 3:
                        three = k
                    elif cnt[k] == 2:
                        two = k
                return "set", (three, two)
            return "invalid", ()
        if vals.count(1) >= 4: # should be straight
            points.sort(key=lambda x: cardscale.index(x))
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
            sup_straight = [cardscale[cardscale.index(first)+i] for i in range(5)]
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
            ks.sort(key=lambda x: cardscale.index(x))
            if 'A' in ks:
                if ks == ['A', '2']:
                    return "three_straight", ('A')
                if ks == ['A', 'K']:
                    return "three_straight", ('K')
                return "invalid", ()
            if cardscale.index(ks[1]) - cardscale.index(ks[0]) == 1:
                return "three_straight", (ks[0])
            return "invalid", ()
        if vals.count(2) == 3:
            ks = []
            for k in list(cnt.keys()):
                ks.append(k)
            ks.sort(key=lambda x: cardscale.index(x))
            if 'A' in ks:
                if ks == ['A', 'Q', 'K']:
                    return "triple_pairs", ('Q')
                if ks == ['A', '2', '3']:
                    return "triple_pairs", ('A')
                return "invalid", ()
            pairs = [cardscale[cardscale.index(ks[0])+i] for i in range(3)]
            if ks == pairs:
                return "triple_pairs", (ks[0])
            return "invalid", ()
        
        return "invalid", ()
    if len(poker) > 6 and len(poker) <= 10:
        if vals.count(len(poker)) == 1:
            bomb = points[0]
            return "bomb", (len(poker), bomb)
    return "invalid", ()


def checkBigger(pokertype1, points1, pokertype2, points2): # return True if pokertype2 is bigger than pokertype1
                                         # report an error if pokertype2 mismatches pokertype1 
    if pokertype2 == "rocket":
        return True
    if pokertype1 == "rocket":
        return False
    
    if pokertype1 in Normaltypes:
        if pokertype2 in Normaltypes:
            if pokertype1 == pokertype2:
                if pokertype1 in scaletypes and cardscale.index(points2[0]) > cardscale.index(points1[0]):
                    return True
                if pokertype1 not in scaletypes and pointorder.index(points2[0]) > pointorder.index(points1[0]):
                    return True
                return False
            return "error"
        return True
    if pokertype2 in Normaltypes:
        return "error"
    if pokertype1 == "bomb":
        if pokertype2 == "bomb":
            if points2[0] == points1[0] and pointorder.index(points2[1]) > pointorder.index(points1[1]):
                return True
            if points2[0] > points1[0]:
                return True
            return False
        if pokertype2 == "straight_flush":
            if points1[0] < 6:
                return True
            return False
        return False
    if pokertype1 == "straight_flush":
        if pokertype2 == "bomb":
            if points2[0] >= 6:
                return True
            return False
        if pokertype2 == "straight_flush":
            if cardscale.index(points2[0]) > cardscale.index(points1[0]):
                return True
            return False
        return False
    return False

class ExpertAI:
    def __init__(self):
        self.hold = []
        self.id = -1
        self.level = '2'
        self.first = None
        self.last = None
        self.tribute = 0
        self.tributed = -1
        self.got_return = False

    def process_request(self, full_input):
        requests = full_input["requests"]
        responses = full_input.get("responses", [])
        
        for i in range(len(requests)-1):
            req = requests[i]
            res = responses[i] if i < len(responses) else []
            
            if req["stage"] == "deal":
                self.hold.extend(req["deliver"])
                self.id = req["your_id"]
                self.first = req["global"]["first"]
                self.last = req["global"]["last"]
                self.tribute = req["global"]["tribute"]
                self.level = req["global"]["level"]
                pointorder.remove(self.level)
                pointorder.append(self.level)
                pointorder.extend(['o', 'O'])
                
            elif req["stage"] == "tribute" and res:
                self.hold.remove(res[0])
                self.tributed = res[0]
                
            elif req["stage"] == "return" and res:
                if self.tribute == 1:
                    pok = req["global"]["tribute_cards"][str(self.last)]
                    self.hold.append(pok)
                    self.hold.remove(res[0])
                elif res:
                    tribute_list = list(req["global"]["tribute_cards"].values())
                    if pointorder.index(num2Poker(tribute_list[0])[1]) == pointorder.index(num2Poker(tribute_list[1])[1]):
                        pok = req["global"]["tribute_cards"][str((self.id+1)%4)]
                    elif pointorder.index(num2Poker(tribute_list[0])[1]) > pointorder.index(num2Poker(tribute_list[1])[1]):
                        pok = tribute_list[0] if self.id == self.first else tribute_list[1]
                    else:
                        pok = tribute_list[1] if self.id == self.first else tribute_list[0]
                    self.hold.append(pok)
                    self.hold.remove(res[0])
                    
            elif req["stage"] == "play":
                if self.tributed != -1 and not self.got_return:
                    if self.tribute == 1:
                        self.hold.append(req["global"]["return_cards"][str(self.first)])
                    elif pointorder.index(num2Poker(req["global"]["tribute_cards"][str(self.id)])[1]) == pointorder.index(num2Poker(req["global"]["tribute_cards"][str((self.id+2)%4)])[1]):
                        self.hold.append(req["global"]["return_cards"][str((self.id-1)%4)])
                    elif pointorder.index(num2Poker(req["global"]["tribute_cards"][str(self.id)])[1]) > pointorder.index(num2Poker(req["global"]["tribute_cards"][str((self.id+2)%4)])[1]):
                        self.hold.append(req["global"]["return_cards"][str(self.first)])
                    else:
                        self.hold.append(req["global"]["return_cards"][str((self.first+2)%4)])
                    self.got_return = True
                if res:
                    for cdid in res[0]:
                        self.hold.remove(cdid)

    def get_tribute_response(self, cur_req):
        resist = cur_req["global"]["resist"]
        if resist:
            return []
        for poker in self.hold:
            if isValidTribute(self.hold, poker, self.level):
                return [poker]
        return []

    def get_return_response(self, cur_req):
        resist = cur_req["global"]["resist"]
        if resist:
            return []
        for poker in self.hold:
            if isValidReturn(self.hold, poker, self.level):
                return [poker]
        return []

    def get_play_response(self, cur_req):
        history = cur_req["history"]
        power = True
        last_action = []
        last_claim = []
        
        for move in reversed(history):
            if "player" in move and move["player"] == self.id:
                break
            if "response" in move and move["response"][0]:
                power = False
                last_action = move["response"][0]
                last_claim = move["response"][1]
                break

        if power:
            # When having power, play the largest non-set combination
            return self.play_largest_non_set()
        else:
            # When not having power, try to beat with smallest possible combination
            return self.beat_with_smallest(last_action, last_claim)

    def play_largest_non_set(self):
        # Find all possible combinations, sorted by size (descending)
        combinations = self.find_all_combinations()
        # Filter out sets
        valid_combos = [c for c in combinations if checkPokerType(c)[0] != "set"]
        
        if valid_combos:
            # Sort by size (descending) then by strength (ascending)
            valid_combos.sort(key=lambda x: (-len(x), self.get_combination_strength(x)))
            return [valid_combos[0], valid_combos[0]]
        else:
            # If only sets available, play the smallest one
            if combinations:
                combinations.sort(key=lambda x: (len(x), self.get_combination_strength(x)))
                return [combinations[0], combinations[0]]
            # If nothing left (shouldn't happen), pass
            return []

    def beat_with_smallest(self, last_action, last_claim):
        if not last_action:
            return []
            
        last_type, last_points = checkPokerType(last_claim)
        if last_type == "invalid":
            return []
            
        # Find all combinations that can beat the last move
        possible_moves = []
        for combo in self.find_all_combinations():
            combo_type, combo_points = checkPokerType(combo)
            if combo_type == "invalid":
                continue
            result = checkBigger(last_type, last_points, combo_type, combo_points)
            if result is True:
                possible_moves.append(combo)
                
        if not possible_moves:
            return []
            
        # Sort by size (ascending) then by strength (ascending) to find smallest possible move
        possible_moves.sort(key=lambda x: (len(x), self.get_combination_strength(x)))
        return [possible_moves[0], possible_moves[0]]

    def find_all_combinations(self):
        # This is a simplified version - in practice you'd want to generate all possible valid combinations
        # Here we just return single cards, pairs, and three-of-a-kinds for simplicity
        combinations = []
        point_counts = Counter(num2Poker(c)[1] for c in self.hold)
        
        # Add single cards
        combinations.extend([[c] for c in self.hold])
        
        # Add pairs
        for point, count in point_counts.items():
            if count >= 2:
                cards = [c for c in self.hold if num2Poker(c)[1] == point]
                combinations.append(cards[:2])
                if count >= 3:
                    combinations.append(cards[:3])
                    
        return combinations

    def get_combination_strength(self, combo):
        # Lower is better (we want to play weaker cards first)
        combo_type, combo_points = checkPokerType(combo)
        if not combo_points:
            return float('inf')
            
        if combo_type in scaletypes:
            return cardscale.index(combo_points[0])
        return pointorder.index(combo_points[0])

    def get_response(self, full_input):
        self.process_request(full_input)
        cur_req = full_input["requests"][-1]
        
        if cur_req["stage"] == "deal":
            return {"response": [], "debug": "deal stage"}
        elif cur_req["stage"] == "tribute":
            response = self.get_tribute_response(cur_req)
            return {"response": response, "debug": "tribute stage"}
        elif cur_req["stage"] == "return":
            response = self.get_return_response(cur_req)
            return {"response": response, "debug": "return stage"}
        elif cur_req["stage"] == "play":
            response = self.get_play_response(cur_req)
            return {"response": response, "debug": "play stage"}

# Main execution
_online = os.environ.get("USER", "") == "root"
if _online:
    full_input = json.loads(input())
else:
    with open("testAI.json") as fo:
        full_input = json.load(fo)

ai = ExpertAI()
response = ai.get_response(full_input)
print(json.dumps(response))