import numpy as np
import random


class Utils():
    
    def __init__(self):
        self.cardscale = ['A','2','3','4','5','6','7','8','9','0','J','Q','K']
        #h=heart 红桃 d=Diamond 方块 s=Spade 黑桃 c=Club 梅花 按这个顺序
        self.suitset = ['h','d','s','c']
        self.jokers = ['jo', 'jO']
        self.level_to_value={'A':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'0':10,'J':11,'Q':12,'K':13,'joker':14,'JOker':15,'o':14,'O':15}
        self.value_to_level={1:'A',2:'2',3:'3',4:'4',5:'5',6:'6',7:'7',8:'8',9:'9',10:'0',11:'J',12:'Q',13:'K',14:'joker',15:'JOker'}
        # 不论打几，顺子都是1-13
        self.straight_rank=[1,2,3,4,5,6,7,8,9,10,11,12,13]
        # 根据打的级来确定牌的大小顺序，从小到大
        self.card_value_ranks={
            1:[2,3,4,5,6,7,8,9,10,11,12,13,1,14,15],
            2:[3,4,5,6,7,8,9,10,11,12,13,1,2,14,15],
            3:[2,4,5,6,7,8,9,10,11,12,13,1,3,14,15],
            4:[2,3,5,6,7,8,9,10,11,12,13,1,4,14,15],
            5:[2,3,4,6,7,8,9,10,11,12,13,1,5,14,15],
            6:[2,3,4,5,7,8,9,10,11,12,13,1,6,14,15],
            7:[2,3,4,5,6,8,9,10,11,12,13,1,7,14,15],
            8:[2,3,4,5,6,7,9,10,11,12,13,1,8,14,15],
            9:[2,3,4,5,6,7,8,10,11,12,13,1,9,14,15],
            10:[2,3,4,5,6,7,8,9,11,12,13,1,10,14,15],
            11:[2,3,4,5,6,7,8,9,10,12,13,1,11,14,15],
            12:[2,3,4,5,6,7,8,9,10,11,13,1,12,14,15],
            13:[2,3,4,5,6,7,8,9,10,11,12,1,13,14,15],
        }

    '''
    get_card_num函数:用于获取牌的点数
    输入：
        card: 牌的编号(0-107)
    输出：
        card_num: 牌的点数(1-15)
    '''
    def get_card_num(self, card):
        t=0
        if(card>=54):
            t=card-54
        else:
            t=card
        if(t<=51):
            card_num=t//4+1
        elif(t==52): 
            #joker
            card_num=14
        else:
            #JOker
            card_num=15
        return card_num

    '''
    get_cards_nums函数:用于获取牌的点数列表
    输入：
        cards: 牌的编号(0-107)的列表
    输出：
        card_nums: 牌的点数(1-15)的列表
    '''
    def get_cards_nums(self,cards):
        return [self.get_card_num(card) for card in cards]
    
    '''
    get_card_ids函数:用于获取某个点数牌的所有可能编号的列表,id从小到大
    输入：
        card_num: 牌的点数(1-15)
    输出：
        card_id_list: 牌的编号(0-107)的列表,id从小到大
    '''
    def get_card_ids(self, card_num):
        card_id_list=[]
        if(card_num<=13):
            start_id=(card_num-1)*4
            for i in range(4):
                card_id_list.append(start_id+i)
            for i in range(4):
                card_id_list.append(start_id+i+54)
        elif(card_num==14):
            return [52,106]
        elif(card_num==15):
            return [53,107]
        return card_id_list
    
    '''
    get_possible_ids函数:用于获取某个点数牌的所有可能编号的列表,id从小到大
    输入：
        card_num: 牌的点数(1-15)
        color: 牌的花色(h,d,s,c)
    输出：
        card_id_list: 牌的编号(0-107)的列表,id从小到大
    '''
    def get_possible_ids(self, card_num, color):
        base=(card_num-1)*4+self.suitset.index(color)
        return [base,base+54]
    
    '''
    Num2Poker函数:用于将牌的编号转换为牌的名称
    输入：
        num: 牌的编号(0-107)
    输出：
        poker: 牌的名称(除了joker和JOker之外, 都是h,d,s,c+'A'~'K'的两位字符串)
    '''
    def Num2Poker(self, num: int):
        num_in_deck = num % 54
        if num_in_deck == 52:
            return "joker"
        if num_in_deck == 53:
            return "JOker"
        # Normal cards:
        pokernumber = self.cardscale[num_in_deck // 4]
        pokersuit = self.suitset[num_in_deck % 4]
        return pokersuit + pokernumber
    
    '''
    Num2Pokers函数:用于将牌的编号列表转换为牌的名称列表
    输入：
        nums: 牌的编号(0-107)的列表
    输出：
        pokers: 牌的名称(除了joker和JOker之外, 都是h,d,s,c+'A'~'K'的两位字符串)的列表
    '''
    def Num2Pokers(self,nums):
        return [self.Num2Poker(num) for num in nums]
    
    '''
    Poker2Num函数:用于将牌的名称转换为牌的编号
    输入：
        poker: 牌的名称(除了joker和JOker之外, 都是h,d,s,c+'A'~'K'的两位字符串)
        deck: 牌的编号(0-107)的列表,相当于是候选牌的集合,不然同一张牌对应2个编号
    输出：
        num: 牌的编号(0-107)
    '''
    def Poker2Num(self, poker: str, deck):
        num_in_deck = -1
        if poker[0] == "j":
            num_in_deck = 52
        elif poker[0] == "J":
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