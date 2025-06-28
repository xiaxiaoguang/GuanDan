import numpy as np
import random
from expert_utils import *
import warnings
from collections import Counter
    
class ExpertGuanDanEnv():
    '''
    Usage:
    Step1 Call GuanDanEnv() to create an instance
    Step2 Call GuanDanEnv.reset(config) to reset a match  
    '''
    def __init__(self):
        self.cardscale = ['A','2','3','4','5','6','7','8','9','0','J','Q','K']
        self.nocard=[[],[[],[]]]
        self.straight_patterns = [
            np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]),
            np.array([0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]),
            np.array([0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]),
            np.array([0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]),
            np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0]),
            np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0]),
            np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0]),
            np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0]),
            np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]),
            np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]),
        ]
        # 以下是dfs匹配同花顺时要用的
        self.straight_flush_nums={
            "h":0,
            "d":0,
            "s":0,
            "c":0
        }
        self.straight_flush_start_nums={
            "h":[],
            "d":[],
            "s":[],
            "c":[]
        }
        # 以下是dfs匹配顺子时要用的
        self.turn_minus_bomb=100 #给定手牌，计算轮次-炸弹的数量，越小越好
        self.straight_nums=0
        self.straight_start_nums=[]
        
        self.card_scale_num=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
        self.colors = ['h','d','s','c']
        self.point_order = ['2', '3', '4', '5', '6', '7', '8', '9', '0', 'J', 'Q', 'K', 'A']
        self.types = ["single", "pair", "three", "straight", "set", "three_straight", "triple_pairs","bomb","straight_flush","rocket"]
        self.Normaltypes = ["single", "pair", "three", "straight", "set", "three_straight", "triple_pairs"]
        self.bombtypes = ["bomb","straight_flush"]
        self.scaletypes = ["straight", "three_straight", "triple_pairs"]
        self.bomb_start_points=[0,13,36,49,62,75,88] #这个是炸弹的相对大小对应的起始分数,不包括同花顺
        self.utils = Utils()
        self.level = None
        self.red_num=0
        self.shangyou="me"
        self.seed = None
        self.done = False
        self.game_state_info = "Init"
        self.cleared = [] # list of cleared players (have played all their decks)
        self.last_play_person="me"
        self.my_current_cards=[] #这个是0-107的列表
        self.my_current_cards_with_color={} #这个是带花色的字典，key是牌的编号，value是花色字典，花色字典的key是花色，value是张数
        self.my_current_cards_without_color={} #这个是不管花色的字典，key是牌的编号，value是张数
        self.my_deliver=[] #这个是0-107的列表
        self.my_deliver_without_color={} #这个是0-107的列表
        self.my_deliver_with_color={} #这个是带花色的字典，key是牌的编号，value是花色字典，花色字典的key是花色，value是张数
        #以下几个是组牌的时候用到的
        self.cards={}
        self.cards_final={}
        self.cards_start_num={}
        self.bombs_ids=[] #这个是炸弹的id列表
        self.remaining_card_ids=[] #这个是0-107的列表
        self.remaining_card_with_color={} #这个是带花色的字典，key是牌的编号，value是花色字典，花色字典的key是花色，value是张数
        self.remaining_card_without_color={} #这个是不管花色的字典，key是牌的编号，value是张数
        self.my_current_response=[]
        self.card_value_rank=[] #这个是基于当前级的牌大小顺序，从小到大的数字列表
        self.current_card_num={"me":27, "duijia":27, "shangjia":27, "xiajia":27}
        self.agent_names = ['player_%d' % i for i in range(4)]
        self.errset = {
            0: "Initialization Fault",
            1: "PlayerAction Fault",
            2: "Game Fault"
        }
    
    '''
    remove_when_grouping函数:用于在组牌的时候移除已经组过的牌
    输入：
        card_id: 牌的编号(0-107)
    '''
    def remove_when_grouping(self,card_id):
        self.remaining_card_ids.remove(card_id)
        card=self.utils.Num2Poker(card_id)
        if(card[0]=='j'):
            self.remaining_card_without_color[14]-=1
            self.remaining_card_with_color[14]-=1
        elif(card[0]=='J'):
            self.remaining_card_without_color[15]-=1
            self.remaining_card_with_color[15]-=1
        else:
            color=card[0]
            card_num=self.utils.level_to_value[card[1]]
            self.remaining_card_without_color[card_num]-=1
            self.remaining_card_with_color[card_num][color]-=1
    
    def update_card_nums(self,history,done):
        '''
        update_card_nums函数:用于更新当前手牌的数量
        输入:
            history: 历史记录，包含每个人的出牌记录
            done: 是否结束
        '''
        pass
        
    
    
    def group_into_1234(self):
        for card_num, num_of_cards in self.remaining_card_without_color.items():
            if num_of_cards >= 4:
                self.cards["bomb"].append([card_num,num_of_cards])
            elif num_of_cards == 3:
                self.cards["three"].append(card_num)
            elif num_of_cards == 2:
                self.cards["pair"].append(card_num)
            elif num_of_cards == 1:
                self.cards["single"].append(card_num) 
     
    '''
    group_into_123422233函数:用于计算给定牌的轮次数-炸弹数
    输入:给定牌的array, 长度为13(A到K)
    输出:轮次数-炸弹数
    '''         
    def group_into_123422233(self,num_array):
        array=num_array.copy()
        l=len(array)
        bomb_cnt=0
        turn_cnt=0
        #提取炸弹
        for i in range(l):
            if(array[i]>=4 and array[i]<8):
                bomb_cnt+=1
                array[i]=0
            elif(array[i]==8):
                bomb_cnt+=2
                array[i]=0
        # 提取钢板,不组AAA222，如果打2，也不组222333
        s=1
        if(self.level==2):
            s=2
        for start_num in range(s,l-1):
            if(array[start_num]==3 and array[start_num+1]==3):
                turn_cnt+=1
                array[start_num]=0
                array[start_num+1]=0
        #特判KA
        if(array[l-1]==3 and array[0]==3):
            turn_cnt+=1
            array[l-1]=0
            array[0]=0
        # 提取三连对,不组AA2233
        for start_num in range(1,l-2):
            if(array[start_num]==2 and array[start_num+1]==2 and array[start_num+2]==2):
                turn_cnt+=1
                array[start_num]=0
                array[start_num+1]=0
                array[start_num+2]=0
        #特判QKA
        if(array[l-2]==2 and array[l-1]==2 and array[0]==2):
            turn_cnt+=1
            array[l-2]=0
            array[l-1]=0
            array[0]=0
        #提取123普通轮次,注意得考虑三带二可以压缩轮次
        cnt_2=0
        cnt_3=0
        for i in range(l):
            if(array[i]>0):
                if(array[i]==2 and self.card_value_rank.index(i+1)<=10):
                    cnt_2+=1
                elif(array[i]==3):
                    cnt_3+=1
                turn_cnt+=1
        if(cnt_2>cnt_3):
            turn_cnt-=cnt_3
        else:
            turn_cnt-=cnt_2
        return turn_cnt-bomb_cnt
                
    '''
    get_possible_straights函数:用于获取可能的顺子起始点
    输出：可能的顺子起始点列表
    '''
    def get_possible_straights(self,flush=False):

        avail_straight_nums=[n for n in range(1,14) if self.remaining_card_without_color[n]>0]
        possible_starts=[]
        if(flush==False):
            if(self.remaining_card_without_color[2]==1 and self.remaining_card_without_color[3]==1 
            and self.remaining_card_without_color[4]==1 and self.remaining_card_without_color[5]==1 
            and 1 in avail_straight_nums):
                possible_starts.append(1)
            for start_num in range(2,10):
                for i in range(5):
                    if(start_num+i not in avail_straight_nums):
                        break
                    elif(i==4):
                        possible_starts.append(start_num)
        else:
            for start_num in range(1,10):
                for i in range(5):
                    if(start_num+i not in avail_straight_nums):
                        break
                    elif(i==4):
                        possible_starts.append(start_num)
        #特判10JQKA
        if(10 in avail_straight_nums and 11 in avail_straight_nums and 12 in avail_straight_nums and 13 in avail_straight_nums and 1 in avail_straight_nums):
            possible_starts.append(10)
        return possible_starts
    
    '''
    get_possible_flush_starts函数:用于获取可能的同花顺起始点
    输出:可能的同花顺起始点字典,key是花色,value是起始点列表
    '''
    def get_possible_flush_starts(self):
        possible_flush_starts={'h':[],'d':[],'s':[],'c':[]}
        for color in ['h','d','s','c']:
            possible_starts=self.get_possible_straights(flush=True)
            for start_num in possible_starts:
                if(start_num==10):
                    if(self.remaining_card_with_color[10][color]>0 and self.remaining_card_with_color[11][color]>0 and self.remaining_card_with_color[12][color]>0 and self.remaining_card_with_color[13][color]>0 and self.remaining_card_with_color[1][color]>0):
                        possible_flush_starts[color].append(10)
                else:
                    if(self.remaining_card_with_color[start_num][color]>0 and self.remaining_card_with_color[start_num+1][color]>0 and self.remaining_card_with_color[start_num+2][color]>0 and self.remaining_card_with_color[start_num+3][color]>0 and self.remaining_card_with_color[start_num+4][color]>0):
                        possible_flush_starts[color].append(start_num)
        return possible_flush_starts
    
    '''
    get_possible_triple_pair_starts函数:用于获取可能的三连对起始点
    输出:可能的三连对起始点列表
    '''
    def get_possible_triple_pair_starts(self):
        possible_triple_pair_starts=[]
        s=1
        if(self.level==2 or self.level==3):
            s=2
        for start_num in range(s,13):
            if start_num==12:
                if(self.remaining_card_without_color[12]==2 and self.remaining_card_without_color[13]==2 and self.remaining_card_without_color[1]==2):
                    possible_triple_pair_starts.append(12)
            else:
                if(self.remaining_card_without_color[start_num]==2 and self.remaining_card_without_color[start_num+1]==2 and self.remaining_card_without_color[start_num+2]==2):
                    possible_triple_pair_starts.append(start_num)
        return possible_triple_pair_starts
    
    '''
    get_possible_three_straight_starts函数:用于获取可能的钢板起始点
    输出:可能的钢板起始点列表
    '''
    def get_possible_three_straight_starts(self):
        possible_three_straight_starts=[]
        s=2
        # 如果打2，出222333也很亏
        if(self.level==2):
            s=3
        for start_num in range(s,14):
            if(start_num==13):
                if(self.remaining_card_without_color[13]==3 and self.remaining_card_without_color[1]==3):
                    possible_three_straight_starts.append(13)
            elif(self.remaining_card_without_color[start_num]==3 and self.remaining_card_without_color[start_num+1]==3):
                possible_three_straight_starts.append(start_num)
        return possible_three_straight_starts
    
    '''
    only_bombs_remaining函数:用于判断是否只剩下炸弹
    '''
    def only_bombs_remaining(self):
        if(self.cards_final["single"]==[] and self.cards_final["pair"]==[] and self.cards_final["three"]==[] and self.cards_final["set"]==[] and self.cards_final["straight"]==[] and self.cards_final["triple_pairs"]==[] and self.cards_final["three_straight"]==[]):
            return True
        return False
    
    '''
    deal_with_jokers函数:用于处理大小王,无返回值
    '''
    def deal_with_jokers(self):
        # 处理大小王
        joker_ids=[i for i in [52,106] if i in self.remaining_card_ids]
        JOker_ids=[i for i in [53,107] if i in self.remaining_card_ids]
        if(self.remaining_card_without_color[14]==2):
            self.cards_final["pair"].append([52,106])
        elif(self.remaining_card_without_color[14]==1):
            self.cards_final["single"].append(joker_ids)
        if(self.remaining_card_without_color[15]==2):
            self.cards_final["pair"].append([53,107])
        elif(self.remaining_card_without_color[15]==1):
            self.cards_final["single"].append(JOker_ids)
        for joker_id in joker_ids:
            self.remove_when_grouping(joker_id)
        for JOker_id in JOker_ids:
            self.remove_when_grouping(JOker_id)
            
    
    '''
    dfs_search_straight_flush函数:用于dfs搜索同花顺
    输入:
        num_array: 剩余的牌数列表
        color: 花色
    输出：
        already_num: 已经组好的同花顺数量
        already_start_num: 已经组好的同花顺起始点列表
        color: 花色
    '''
    def dfs_search_straight_flush(self,num_array,color):
        ret_already_num=0
        ret_already_start_num=[]
        for id,pattern in enumerate(self.straight_patterns):
            if(np.all(num_array>=pattern)):
                a,b,_=self.dfs_search_straight_flush(num_array-pattern,color)
                if(a+1>ret_already_num):
                    ret_already_num=a+1
                    b.extend([id+1])
                    ret_already_start_num=b
        #如果flag为False，说明当前已经没有找到同花顺，dfs搜到头了
        if(self.straight_flush_nums[color]<ret_already_num):
            self.straight_flush_nums[color]=ret_already_num
            self.straight_flush_start_nums[color]=ret_already_start_num
        return ret_already_num,ret_already_start_num,color
    
    '''
    dfs_search_straight函数:用于dfs搜索普通顺子,最终选择一个让总轮次数-炸弹数最小的方案。
    输入:
        num_array: 剩余的牌数列表
        current_num:现在已匹配的顺子数
        current_start_nums:现在已匹配顺子的开头的牌(点数)的集合
    输出：
        无
    '''
    def dfs_search_straight(self,num_array,current_num,current_start_nums):
        tmb=self.group_into_123422233(num_array)
        if(current_num+tmb<self.turn_minus_bomb):
            self.turn_minus_bomb=current_num+tmb
            self.straight_nums=current_num
            self.straight_start_nums=current_start_nums
        for id,pattern in enumerate(self.straight_patterns):
            if(np.all(num_array>=pattern)):
                start_nums=current_start_nums.copy()
                start_nums.append(id+1)
                self.dfs_search_straight(num_array-pattern,current_num+1,start_nums)
        
    '''
    group_cards_with_color函数:用于将当前手牌分组为不同牌型（支持花色）
    输出：分类后的牌型字典，带花色信息
    '''
    def group_cards_with_color(self):
        
        #这个cards是最开始分为单张、对子、三张、炸弹，为后面提供参考,存放数字即可
        self.cards = {
            "single": [],         # 单张
            "pair": [],           # 对子
            "three": [],          # 三张
            "bomb": [],           # 炸弹
        }
        
        #这个才是最终的牌型字典，存放id
        self.cards_final = {
            "single": [],         # 单张
            "pair": [],           # 对子
            "three": [],          # 三张
            "bomb": [],           # 炸弹
            "set": [],            # 三带二
            "straight": [],       # 顺子
            "straight_flush": [],  # 同花顺
            "triple_pairs": [],    # 三连对
            "three_straight": [],  # 钢板（222333这种连续三张）
            "red_level": [],      # 红配
        }
        
        # 用于记录每种牌型对应的起始点数，不需要记录id
        self.cards_start_num={
            "single": [],         # 单张
            "pair": [],           # 对子
            "three": [],          # 三张
            "bomb": [],           # 炸弹
            "set": [],            # 三带二
            "straight": [],       # 顺子
            "straight_flush": [],  # 同花顺
            "triple_pairs": [],    # 三连对
            "three_straight": [],  # 钢板（222333这种连续三张）
        }

        # 还未被组的手牌
        self.remaining_card_ids=self.my_current_cards.copy()
        self.remaining_card_with_color=self.my_current_cards_with_color.copy()
        self.remaining_card_without_color=self.my_current_cards_without_color.copy()
        
        # 0. 拎出级牌，分类单张、对子、三张和炸弹
        # 注意，red_level这里不是嵌套列表，因为没有必要
        if(self.red_num in self.remaining_card_ids):
            self.cards_final["red_level"].append(self.red_num)
            self.remove_when_grouping(self.red_num)
        if(self.red_num+54 in self.remaining_card_ids):
            self.cards_final["red_level"].append(self.red_num+54)
            self.remove_when_grouping(self.red_num+54)
            
        self.group_into_1234()
        
        # 1.处理大小王
        self.deal_with_jokers()


        # 2. 识别同花顺,想贪心一下（顺子中的同花色牌）
        '''
        avail_flush_start_nums=self.get_possible_flush_starts()
        #看看能组多少个同花顺
        for color in ['h','d','s','c']:
            while(len(avail_flush_start_nums[color])>0):
                first_start_num=avail_flush_start_nums[color][0]
                all_card_ids=[]
                if(first_start_num==10):
                    for i in range(10,14):
                        possible_ids=self.utils.get_possible_ids(i,color)
                        if(possible_ids[0] in self.remaining_card_ids):
                            all_card_ids.append(possible_ids[0])
                            self.remove_when_grouping(possible_ids[0])
                        else:
                            all_card_ids.append(possible_ids[1])
                            self.remove_when_grouping(possible_ids[1])
                    possible_ids=self.utils.get_possible_ids(1,color)
                    if(possible_ids[0] in self.remaining_card_ids):
                        all_card_ids.append(possible_ids[0])
                        self.remove_when_grouping(possible_ids[0])
                    else:
                        all_card_ids.append(possible_ids[1])
                        self.remove_when_grouping(possible_ids[1])
                else:
                    for i in range(first_start_num,first_start_num+5):
                        possible_ids=self.utils.get_possible_ids(i,color)
                        if(possible_ids[0] in self.remaining_card_ids):
                            all_card_ids.append(possible_ids[0])
                            self.remove_when_grouping(possible_ids[0])
                        else:
                            all_card_ids.append(possible_ids[1])
                            self.remove_when_grouping(possible_ids[1])
                self.cards_final["straight_flush"].append(all_card_ids)
                self.bombs_ids.extend(all_card_ids)
                #更新avail_flush_start_nums,因为已经组了一条同花顺
                avail_flush_start_nums=self.get_possible_flush_starts()
        '''
        # 尝试用dfs搜索同花顺
        for color in ['h','d','s','c']:
            num_list=[]
            for num in range(1,14):
                num_list.append(self.remaining_card_with_color[num][color])
            array=np.array(num_list)
            _,_,_=self.dfs_search_straight_flush(array,color)
            for i in self.straight_flush_start_nums[color]:
                all_card_ids=[]
                if(i!=10):
                    for j in range(5):
                        possible_ids=self.utils.get_possible_ids(i+j,color)
                        if(possible_ids[0] in self.remaining_card_ids):
                            all_card_ids.append(possible_ids[0])
                            self.remove_when_grouping(possible_ids[0])
                        else:
                            all_card_ids.append(possible_ids[1])
                            self.remove_when_grouping(possible_ids[1])
                else:
                    # 10JQKA
                    for j in range(4):
                        possible_ids=self.utils.get_possible_ids(i+j,color)
                        if(possible_ids[0] in self.remaining_card_ids):
                            all_card_ids.append(possible_ids[0])
                            self.remove_when_grouping(possible_ids[0])
                        else:
                            all_card_ids.append(possible_ids[1])
                            self.remove_when_grouping(possible_ids[1])
                    possible_ids=self.utils.get_possible_ids(1,color)
                    if(possible_ids[0] in self.remaining_card_ids):
                        all_card_ids.append(possible_ids[0])
                        self.remove_when_grouping(possible_ids[0])
                    else:
                        all_card_ids.append(possible_ids[1])
                        self.remove_when_grouping(possible_ids[1])
                self.cards_final["straight_flush"].append(all_card_ids)
                self.bombs_ids.extend(all_card_ids)
                
        # 2.5 如果有红配，继续尝试组同花顺
        if(self.cards_final["red_level"]!=[]):
            red_level_len=len(self.cards_final["red_level"])
            for color in ['h','d','s','c']:
                temp_color_list=[]
                for num in range(1,14):
                    temp_color_list.append(int(self.remaining_card_with_color[num][color]>0))
                temp_color_array=np.array(temp_color_list)
                for start_num in range(1,11):
                    while(sum(temp_color_array[start_num-1:start_num+4])==4):
                        # 可以组同花顺
                        all_card_ids=[]
                        if(start_num!=10):
                            for i in range(5):
                                if(self.remaining_card_with_color[start_num+i][color]==0):
                                    all_card_ids.append(self.cards_final["red_level"][0])
                                    self.cards_final["red_level"].remove(self.cards_final["red_level"][0])
                                    red_level_len-=1
                                else:
                                    possible_ids=self.utils.get_possible_ids(start_num+i,color)
                                    if(possible_ids[0] in self.remaining_card_ids):
                                        all_card_ids.append(possible_ids[0])
                                        self.remove_when_grouping(possible_ids[0])
                                    else:
                                        all_card_ids.append(possible_ids[1])
                                        self.remove_when_grouping(possible_ids[1])
                        else:
                            # 10JQKA
                            for i in range(4):
                                if(self.remaining_card_with_color[start_num+i][color]==0):
                                    all_card_ids.append(self.cards_final["red_level"][0])
                                    self.cards_final["red_level"].remove(self.cards_final["red_level"][0])
                                    red_level_len-=1
                                else:
                                    possible_ids=self.utils.get_possible_ids(start_num+i,color)
                                    if(possible_ids[0] in self.remaining_card_ids):
                                        all_card_ids.append(possible_ids[0])
                                        self.remove_when_grouping(possible_ids[0])
                                    else:
                                        all_card_ids.append(possible_ids[1])
                                        self.remove_when_grouping(possible_ids[1])
                            if(self.remaining_card_with_color[1][color]==0):
                                all_card_ids.append(self.cards_final["red_level"][0])
                                self.cards_final["red_level"].remove(self.cards_final["red_level"][0])
                                red_level_len-=1
                            else:
                                possible_ids=self.utils.get_possible_ids(1,color)
                                if(possible_ids[0] in self.remaining_card_ids):
                                    all_card_ids.append(possible_ids[0])
                                    self.remove_when_grouping(possible_ids[0])
                                else:
                                    all_card_ids.append(possible_ids[1])
                                    self.remove_when_grouping(possible_ids[1])
                        self.cards_final["straight_flush"].append(all_card_ids)
                        self.bombs_ids.extend(all_card_ids)
                        if(red_level_len==0):
                            break
                        #更新temp_color_array
                        temp_color_list=[]
                        for num in range(1,14):
                            temp_color_list.append(int(self.remaining_card_with_color[num][color]>0))
                        temp_color_array=np.array(temp_color_list)
                    if(red_level_len==0):
                        break
                if(red_level_len==0):
                    break
                
        # 尝试dfs匹配顺子
        num_list=[]
        for num in range(1,14):
            num_list.append(self.remaining_card_without_color[num])
        array=np.array(num_list)
        initial_turn_minus_bomb=self.group_into_123422233(array)
        self.turn_minus_bomb=initial_turn_minus_bomb
        self.dfs_search_straight(array,0,[])
        for i in self.straight_start_nums:
            all_card_ids=[]
            if(i!=10):
                for j in range(5):
                    possible_ids=self.utils.get_card_ids(i+j)
                    candidate_ids=[id for id in possible_ids if id in self.remaining_card_ids]
                    all_card_ids.append(candidate_ids[0])
                    self.remove_when_grouping(candidate_ids[0])
            else:
                # 10JQKA
                for j in range(4):
                    possible_ids=self.utils.get_card_ids(i+j)
                    candidate_ids=[id for id in possible_ids if id in self.remaining_card_ids]
                    all_card_ids.append(candidate_ids[0])
                    self.remove_when_grouping(candidate_ids[0])
                possible_ids=self.utils.get_card_ids(1)
                candidate_ids=[id for id in possible_ids if id in self.remaining_card_ids]
                all_card_ids.append(candidate_ids[0])
                self.remove_when_grouping(candidate_ids[0])
            self.cards_final["straight"].append(all_card_ids)
        
        # 3. 识别炸弹
        for card_num, num_of_cards in self.remaining_card_without_color.items():
            if num_of_cards >= 4:
                possible_ids=self.utils.get_card_ids(card_num)
                candidate_ids=[i for i in possible_ids if i in self.remaining_card_ids]
                # 8个头拆成2个4炸
                if(len(candidate_ids)>=8):
                    self.cards_final["bomb"].append(candidate_ids[:4])
                    self.cards_final["bomb"].append(candidate_ids[4:])
                else:
                    self.cards_final["bomb"].append(candidate_ids)
                for id in candidate_ids:
                    self.remove_when_grouping(id)
                self.bombs_ids.extend(candidate_ids)  
                    
        # 4. 识别顺子（长度 = 5，不要求同花）
        # 因为已经组了同花顺，需要重新整理
        '''
        self.group_into_1234()
        avail_straight_nums=self.get_possible_straights()
        while(len(avail_straight_nums)>0):
            first_start_num=avail_straight_nums[0]
            all_card_ids=[]
            if(first_start_num!=10):
                for i in range(first_start_num,first_start_num+5):
                    possible_ids=self.utils.get_card_ids(i)
                    candidate_ids=[i for i in possible_ids if i in self.remaining_card_ids]
                    all_card_ids.append(candidate_ids[0])
                    self.remove_when_grouping(candidate_ids[0])  
            else:
                for i in range(10,14):
                    possible_ids=self.utils.get_card_ids(i)
                    candidate_ids=[i for i in possible_ids if i in self.remaining_card_ids]
                    all_card_ids.append(candidate_ids[0])
                    self.remove_when_grouping(candidate_ids[0])
                possible_ids=self.utils.get_card_ids(1)
                candidate_ids=[i for i in possible_ids if i in self.remaining_card_ids]
                all_card_ids.append(candidate_ids[0])
                self.remove_when_grouping(candidate_ids[0])
            self.cards_final["straight"].append(all_card_ids)
            self.cards_final["straight"].append(all_card_ids)
            avail_straight_nums=self.get_possible_straights()
        '''      
        # 5. 识别三连对（triple_pairs, 连续三对）
        self.group_into_1234()
        avail_triple_pair_starts=self.get_possible_triple_pair_starts()
        while(len(avail_triple_pair_starts)>0):
            first_start_num=avail_triple_pair_starts[0]
            all_card_ids=[]
            if(first_start_num==12):
                for i in range(12,14):
                    possible_ids=self.utils.get_card_ids(i)
                    candidate_ids=[i for i in possible_ids if i in self.remaining_card_ids]
                    all_card_ids.extend(candidate_ids)
                    self.remove_when_grouping(candidate_ids[0])
                    self.remove_when_grouping(candidate_ids[1])
                possible_ids=self.utils.get_card_ids(1)
                candidate_ids=[i for i in possible_ids if i in self.remaining_card_ids]
                all_card_ids.extend(candidate_ids)
                self.remove_when_grouping(candidate_ids[0])
                self.remove_when_grouping(candidate_ids[1])
                self.cards_final["triple_pairs"].append(all_card_ids)
            else:
                for i in range(first_start_num,first_start_num+3):
                    possible_ids=self.utils.get_card_ids(i)
                    candidate_ids=[i for i in possible_ids if i in self.remaining_card_ids]
                    all_card_ids.extend(candidate_ids)
                    self.remove_when_grouping(candidate_ids[0])
                    self.remove_when_grouping(candidate_ids[1])
                self.cards_final["triple_pairs"].append(all_card_ids)
            avail_triple_pair_starts=self.get_possible_triple_pair_starts()

        # 6. 识别钢板（three_straight, 如 222333）
        self.group_into_1234()
        avail_three_straight_starts=self.get_possible_three_straight_starts()
        while(len(avail_three_straight_starts)>0):
            first_start_num=avail_three_straight_starts[0]
            all_card_ids=[] 
            if(first_start_num==13):
                possible_ids=self.utils.get_card_ids(13)
                candidate_ids=[i for i in possible_ids if i in self.remaining_card_ids]
                all_card_ids.extend(candidate_ids)
                self.remove_when_grouping(candidate_ids[0])
                possible_ids=self.utils.get_card_ids(1)
                candidate_ids=[i for i in possible_ids if i in self.remaining_card_ids]
                all_card_ids.extend(candidate_ids)
                self.remove_when_grouping(candidate_ids[0])
            else:
                for i in range(first_start_num,first_start_num+2):
                    possible_ids=self.utils.get_card_ids(i)
                    candidate_ids=[i for i in possible_ids if i in self.remaining_card_ids]
                    all_card_ids.extend(candidate_ids)
                    self.remove_when_grouping(candidate_ids[0])
                    self.remove_when_grouping(candidate_ids[1])
                    self.remove_when_grouping(candidate_ids[2])
            self.cards_final["three_straight"].append(all_card_ids)
            avail_three_straight_starts=self.get_possible_three_straight_starts()
        
        # 7.最后处理剩下的牌，组成123等常规牌型(先不考虑三带二)
        for card_num, num_of_cards in self.remaining_card_without_color.items():
            if num_of_cards == 3:
                possible_ids=self.utils.get_card_ids(card_num)
                candidate_ids=[i for i in possible_ids if i in self.remaining_card_ids]
                self.cards_final["three"].append(candidate_ids)
            elif num_of_cards == 2:
                possible_ids=self.utils.get_card_ids(card_num)
                candidate_ids=[i for i in possible_ids if i in self.remaining_card_ids]
                self.cards_final["pair"].append(candidate_ids)
            elif num_of_cards == 1:
                possible_ids=self.utils.get_card_ids(card_num)
                candidate_ids=[i for i in possible_ids if i in self.remaining_card_ids]
                self.cards_final["single"].append(candidate_ids)
                
        # 8. 最后处理红配
        red_num=len(self.cards_final["red_level"])
        # 如果有1个红配
        if(red_num==1):
            if(self.cards_final["three"]!=[]):
                min_idx=100
                choice=0
                for i,cards in enumerate(self.cards_final["three"]):
                    if(self.card_value_rank.index(self.utils.get_card_num(cards[0]))<min_idx):
                        min_idx=self.card_value_rank.index(self.utils.get_card_num(cards[0]))
                        choice=i
                self.cards_final["bomb"].append([self.cards_final["three"][choice][0],self.cards_final["three"][choice][1],self.cards_final["three"][choice][2],self.cards_final["red_level"][0]])
                self.cards_final["three"].remove(self.cards_final["three"][choice])
                self.bombs_ids.extend(self.cards_final["bomb"][-1])
            # 如果有钢板,用一个红配凑成炸弹,剩下的变光三
            elif(self.cards_final["three_straight"]!=[]):
                self.cards_final["bomb"].append([self.cards_final["three_straight"][0][0],self.cards_final["three_straight"][0][1],self.cards_final["three_straight"][0][2],self.cards_final["red_level"][0]])
                self.cards_final["three"].append([self.cards_final["three_straight"][0][3],self.cards_final["three_straight"][0][4],self.cards_final["three_straight"][0][5]])
                self.cards_final["three_straight"].remove(self.cards_final["three_straight"][0])
                self.bombs_ids.extend(self.cards_final["bomb"][-1])
            elif(self.cards_final["bomb"]!=[]):
                self.cards_final["bomb"][0].extend(self.cards_final["red_level"])
                self.bombs_ids.append(self.cards_final["red_level"][0])
            else:
                self.cards_final["single"].append(self.cards_final["red_level"])
        # 如果有2个红配
        elif(red_num==2):
            if(self.cards_final["three"]!=[]):
                # 看看有几张光三
                three_len=len(self.cards_final["three"])
                # 如果>=2个光三，则凑2把炸弹
                if(three_len>=2):
                    self.cards_final["bomb"].append([self.cards_final["three"][0][0],self.cards_final["three"][0][1],self.cards_final["three"][0][2],self.cards_final["red_level"][0]])
                    self.cards_final["bomb"].append([self.cards_final["three"][1][0],self.cards_final["three"][1][1],self.cards_final["three"][1][2],self.cards_final["red_level"][1]])
                    self.cards_final["three"].remove(self.cards_final["three"][0])
                    self.cards_final["three"].remove(self.cards_final["three"][0])
                    self.bombs_ids.extend(self.cards_final["bomb"][-1])
                    self.bombs_ids.extend(self.cards_final["bomb"][-2])
                # 否则凑一把炸弹，如果有原炸弹则剩下一个红配用于增大原炸弹
                elif(three_len==1):
                    if(self.cards_final["bomb"]!=[]):
                        self.cards_final["bomb"][0].extend([self.cards_final["red_level"][1]])
                    else:
                        self.cards_final["single"].append([self.cards_final["red_level"][1]])
                    self.cards_final["bomb"].append([self.cards_final["three"][0][0],self.cards_final["three"][0][1],self.cards_final["three"][0][2],self.cards_final["red_level"][0]])
                    self.cards_final["three"].remove(self.cards_final["three"][0])
                    self.bombs_ids.extend(self.cards_final["bomb"][-1])
            # 如果有三顺,两个红配刚好配两把炸弹
            elif(self.cards_final["three_straight"]!=[]):
                min_idx=100
                choice=0
                for i,cards in enumerate(self.cards_final["three_straight"]):
                    if(self.card_value_rank.index(self.utils.get_card_num(cards[0]))<min_idx):
                        min_idx=self.card_value_rank.index(self.utils.get_card_num(cards[0]))
                        choice=i
                self.cards_final["bomb"].append([self.cards_final["three_straight"][choice][0],self.cards_final["three_straight"][choice][1],self.cards_final["three_straight"][choice][2],self.cards_final["red_level"][0]])
                self.cards_final["bomb"].append([self.cards_final["three_straight"][choice][3],self.cards_final["three_straight"][choice][4],self.cards_final["three_straight"][choice][5],self.cards_final["red_level"][1]])
                self.cards_final["three_straight"].remove(self.cards_final["three_straight"][choice])
                self.bombs_ids.extend(self.cards_final["bomb"][-1])
                self.bombs_ids.extend(self.cards_final["bomb"][-2])
            # 否则如果有对子，用两个红配凑成炸弹
            else:
                if(self.cards_final["pair"]!=[]):
                    min_idx=100
                    choice=0
                    for i,cards in enumerate(self.cards_final["pair"]):
                        if(self.card_value_rank.index(self.utils.get_card_num(cards[0]))<min_idx):
                            min_idx=self.card_value_rank.index(self.utils.get_card_num(cards[0]))
                            choice=i
                    self.cards_final["bomb"].append([self.cards_final["pair"][choice][0],self.cards_final["pair"][choice][1],self.cards_final["red_level"][0],self.cards_final["red_level"][1]])
                    self.cards_final["pair"].remove(self.cards_final["pair"][choice])
                    self.bombs_ids.extend(self.cards_final["bomb"][-1])
                else:
                    if(self.cards_final["bomb"]!=[]):
                        self.cards_final["bomb"][0].extend(self.cards_final["red_level"])
                        self.bombs_ids.extend(self.cards_final["red_level"])
                    else:
                        self.cards_final["pair"].append(self.cards_final["red_level"])
    
    '''
    from_id_get_type函数:用于从id获取牌型和是第几个这样的牌型
    输出:[牌型,第几个]
    '''
    def from_id_get_type(self,id):
        for type in self.types:
            length=len(self.cards_final[type])
            for i in range(length): 
                if id in self.cards_final[type][i]:
                    return [type,i]
        return None
    
    '''
    pick_a_smallest_card函数:用于从剩余牌中选出最小的牌(避开炸弹)
    输出:最小的牌的id
    '''
    def pick_a_smallest_card(self):
        picked_id=-1
        for rank in self.card_value_rank:
            if(self.my_current_cards_without_color[rank]>0):
                cards_ids=self.utils.get_card_ids(rank)
                intersaction_ids=[item for item in cards_ids if item in self.my_current_cards]
                #避开炸弹、火箭、同花顺
                intersaction_without_bomb_ids=[item for item in intersaction_ids 
                if item not in self.bombs_ids]
                
                if(intersaction_without_bomb_ids!=[]):
                    picked_id=intersaction_without_bomb_ids[0]
                    break
        #上面先避开炸弹，但如果挑不出来，说明剩了净手炸弹
        if(picked_id==-1):
            for rank in self.card_value_rank:
                if(self.my_current_cards_without_color[rank]>0):
                    cards_ids=self.utils.get_card_ids(rank)
                    intersaction_ids=[item for item in cards_ids if item in self.my_current_cards]
                    if(intersaction_ids!=[]):
                        picked_id=intersaction_ids[0]
                        break
        return picked_id
    
    '''
    relative_score_of_bomb函数:用于计算炸弹的相对大小对应的分数
    输入：
        bomb_rank: 炸弹的rank,从A到K
        bomb_cards_num: 炸弹的cards_num,从4到10
    输出：
        炸弹的相对分数
    具体实现：
        例如所有13个四线炸弹是0-12
        所有13个五线炸弹是13-25
        所有同花顺是26-35
        所有13个六线炸弹是36-48
        所有13个七线炸弹是49-61
        所有13个八线炸弹是62-74
        所有13个九线炸弹是75-87
        所有13个十线炸弹是88-100
        王炸是101
    '''
    def relative_score_of_bomb(self,bomb_rank,bomb_cards_num):
        if(bomb_rank>=14):
            return 101
        index=self.card_value_rank.index(bomb_rank)
        start_point=self.bomb_start_points[bomb_cards_num-4]
        return start_point+index
    
    '''
    relative_score_of_straight_flush函数:用于计算同花顺的相对大小对应的分数
    输入：
        straight_flush_start_rank: 同花顺的start_rank,从'A'到'0'
    输出：
        同花顺的相对分数
    '''
    def relative_score_of_straight_flush(self,straight_flush_start_rank):
        #rank=self.utils.level_to_value[straight_flush_start_rank]
        return 26+straight_flush_start_rank-1
    
    '''
    get_bomb_nums函数:无输入,返回现有手牌总炸弹数
    '''
    def get_bomb_nums(self):
        return len(self.cards_final["bomb"])+len(self.cards_final["straight_flush"])
    '''
    decide_cards_when_first函数:用于决定该轮我首发出牌出什么
    输出:[[actual_cards],[claim_cards]]
    '''
    def decide_cards_when_first(self):
        picked_id=self.pick_a_smallest_card()
        specific_type=self.from_id_get_type(picked_id)
        if(specific_type!=None):
            type,i=specific_type
            if(self.red_num in self.cards_final[type][i] and self.red_num+54 in self.cards_final[type][i]):
                if(type=="bomb"):
                    length=len(self.cards_final[type][i])
                    temp=self.cards_final[type][i][:length-2]
                    temp.extend([self.cards_final[type][i][0],self.cards_final[type][i][0]])
                    return [self.cards_final[type][i],temp]
            elif(self.red_num in self.cards_final[type][i] or self.red_num+54 in self.cards_final[type][i]):
                if(type=="bomb"):
                    length=len(self.cards_final[type][i])
                    temp=self.cards_final[type][i][:length-1]
                    temp.extend([self.cards_final[type][i][0]])
                    return [self.cards_final[type][i],temp]
                elif(type=="straight_flush"):
                    red_flag=False
                    if(self.red_num in self.cards_final[type][i]):
                        red_idx=self.cards_final[type][i].index(self.red_num)
                        red_flag=True
                    elif(self.red_num+54 in self.cards_final[type][i]):
                        red_idx=self.cards_final[type][i].index(self.red_num+54)
                        red_flag=True
                    if(red_flag):
                        if(red_idx==0):
                            temp=[self.cards_final[type][i][1]-4]
                            temp.extend(self.cards_final[type][i][1:])
                        else:
                            temp=self.cards_final[type][i].copy()
                            temp[red_idx]=temp[red_idx-1]+4
                    if(red_flag):
                        return [self.cards_final[type][i],temp]
                    else:
                        return [self.cards_final[type][i],self.cards_final[type][i]]
            if(type=="three"):
                # 看看有没有合适的三带二, 要求带的对子比较小
                idx=100
                choice=0
                for j in range(len(self.cards_final["pair"])):
                    card_point=self.utils.get_card_num(self.cards_final["pair"][j][0])
                    if(self.card_value_rank.index(card_point)<idx):
                        idx=self.card_value_rank.index(card_point)
                        choice=j
                # 如果idx<=6，或者idx<=14且我牌数<=10，则可以带这个对子
                if(idx<=6 or idx<=14 and self.current_card_num["me"]<=10):
                    self.cards_final[type][i].extend(self.cards_final["pair"][choice])
                    return [self.cards_final[type][i], self.cards_final[type][i]]
                else:
                    return [self.cards_final[type][i],self.cards_final[type][i]]
            # 如果只剩一手三带二，直接走；如果三个头比较小，也走
            elif(type=="pair"):
                if(self.current_card_num["me"]==5 and len(self.cards_final["three"])==1):
                    self.cards_final[type][i].extend(self.cards_final["three"][0])
                    return [self.cards_final[type][i],self.cards_final[type][i]]
                else:
                    # 看看有没有合适的三带二, 要求三带比较小
                    idx=100
                    choice=0
                    for j in range(len(self.cards_final["three"])):
                        card_point=self.utils.get_card_num(self.cards_final["three"][j][0])
                        if(self.card_value_rank.index(card_point)<idx):
                            idx=self.card_value_rank.index(card_point)
                            choice=j
                    if(idx<=6 or idx+self.current_card_num["me"]<=21):
                        self.cards_final[type][i].extend(self.cards_final["three"][choice])
                        return [self.cards_final[type][i],self.cards_final[type][i]]
                    else:
                        return [self.cards_final[type][i],self.cards_final[type][i]]
            else:
                return [self.cards_final[type][i],self.cards_final[type][i]]
        else:
            return None
    
    '''
    decide_cards_when_not_first函数:用于决定该轮我非首发出牌出什么(就是得压别人)
    输入：
        last_poker_type: 上轮出的牌型,[type,params]
    输出:[[actual_cards],[claim_cards]]
    '''
    def decide_cards_when_not_first(self,last_poker_type):   
        type=last_poker_type[0]
        params=last_poker_type[1]
        if(type=="bomb"):
            last_bomb_len=params[0]
            last_bomb_point=params[1]
            last_poker_point=self.utils.level_to_value[last_bomb_point]
        elif(type=="set"):
            #取决于这个三的点数，和二没关系
            last_bomb_len=0
            last_poker_point=self.utils.level_to_value[params[0]]
        else:
            last_bomb_len=0
            last_poker_point=self.utils.level_to_value[params[0]]
    
        min_score_gap=1000
        picked_index=-1
        if(self.cards_final[type]!=[]):
            for idx,cards in enumerate(self.cards_final[type]):
                # 获取这张牌的点数
                if(type=="straight_flush"):
                    # 肯定不炸对家
                    if(self.last_play_person=="duijia"):
                        return []
                    # 根据炸弹大小和上家或下家的牌数，决定要不要追炸弹
                    if(self.only_bombs_remaining()==False):
                        if(self.last_play_person=="shangjia"):
                            if(self.current_card_num["shangjia"]>3+4*self.get_bomb_nums() or self.current_card_num["shangjia"]==4):
                                return []
                        elif(self.last_play_person=="xiajia"):
                            if(self.current_card_num["xiajia"]>2+3*self.get_bomb_nums() or self.current_card_num["xiajia"]==4):
                                return []
                    if(self.red_num == cards[0] or self.red_num+54 == cards[0]):
                        this_poker_point=self.utils.get_card_num(cards[1])-1
                    else:
                        this_poker_point=self.utils.get_card_num(cards[0])
                else:
                    this_poker_point=self.utils.get_card_num(cards[0])
                if(type=="bomb"):
                    this_bomb_len=len(cards)
                    if(self.last_play_person=="duijia"):
                        return []
                    # 根据炸弹数量和上家或下家的牌数，决定要不要追炸弹
                    if(self.only_bombs_remaining()==False):
                        if(self.last_play_person=="shangjia"):
                            if(self.current_card_num["shangjia"]> 5*self.get_bomb_nums()+2 or self.current_card_num["shangjia"]==4):
                                continue
                        elif(self.last_play_person=="xiajia"):
                            if(self.current_card_num["xiajia"]> 5*self.get_bomb_nums() or self.current_card_num["xiajia"]==4):
                                continue
                        
                    check_result=self.check_bigger(type,this_poker_point,type,last_poker_point,this_bomb_len,last_bomb_len)
                else:
                    if(self.last_play_person=="duijia"):
                        # 如果对家，顺过小牌，不压大牌(但是自己只剩单张/对子/光三除外)
                        if(type=="single" or type=="pair"):
                            if(self.current_card_num["me"]==1 and type=="single"):
                                check_result=self.check_bigger(type,this_poker_point,type,last_poker_point)
                                if(check_result[0]):
                                    return [[cards[0]],[cards[0]]]
                            elif(self.current_card_num["me"]==2 and type=="pair"):
                                check_result=self.check_bigger(type,this_poker_point,type,last_poker_point)
                                if(check_result[0]):
                                    return [cards,cards]
                            if(this_poker_point>=10 or this_poker_point==self.level or this_poker_point==1):
                                continue
                        elif(type=="three" or type=="three_straight" or type=="triple_pairs"):
                            if(self.current_card_num["me"]==3 and type=="three"):
                                check_result=self.check_bigger(type,this_poker_point,type,last_poker_point)
                                if(check_result[0]):
                                    return [cards,cards]
                            elif(self.current_card_num["me"]==6 and type=="three_straight"):
                                check_result=self.check_bigger(type,this_poker_point,type,last_poker_point)
                                if(check_result[0]):
                                    return [cards,cards]
                            elif(self.current_card_num["me"]==6 and type=="triple_pairs"):
                                check_result=self.check_bigger(type,this_poker_point,type,last_poker_point)
                                if(check_result[0]):
                                    return [cards,cards]
                            if(this_poker_point>=8 or this_poker_point==self.level or this_poker_point==1):
                                continue
                        elif(type=="straight"):
                            if(self.current_card_num["me"]==5):
                                check_result=self.check_bigger(type,this_poker_point,type,last_poker_point)
                                if(check_result[0]):
                                    return [cards,cards]
                            if(this_poker_point>=6 or this_poker_point==self.level or this_poker_point==1):
                                continue

                    check_result=self.check_bigger(type,this_poker_point,type,last_poker_point)
                if(check_result[0]):
                    if(check_result[3]<min_score_gap):
                        min_score_gap=check_result[3]
                        picked_index=idx
            if(picked_index!=-1):
                if(self.red_num in self.cards_final[type][picked_index] and self.red_num+54 in self.cards_final[type][picked_index]):
                    if(type=="bomb"):
                        length=len(self.cards_final[type][picked_index])
                        temp=self.cards_final[type][picked_index][:length-2]
                        temp.extend([self.cards_final[type][picked_index][0],self.cards_final[type][picked_index][0]])
                        return [self.cards_final[type][picked_index],temp]
                elif(self.red_num in self.cards_final[type][picked_index] or self.red_num+54 in self.cards_final[type][picked_index]):
                    if(type=="bomb"):
                        length=len(self.cards_final[type][picked_index])
                        temp=self.cards_final[type][picked_index][:length-1]
                        temp.extend([self.cards_final[type][picked_index][0]])
                        return [self.cards_final[type][picked_index],temp]
                    elif(type=="straight_flush"):
                        red_flag=False
                        if(self.red_num in self.cards_final[type][picked_index]):
                            red_idx=self.cards_final[type][picked_index].index(self.red_num)
                            red_flag=True
                        elif(self.red_num+54 in self.cards_final[type][picked_index]):
                            red_idx=self.cards_final[type][picked_index].index(self.red_num+54)
                            red_flag=True
                        if(red_flag):
                            if(red_idx==0):
                                temp=[self.cards_final[type][picked_index][1]-4]
                                temp.extend(self.cards_final[type][picked_index][1:])
                            else:
                                temp=self.cards_final[type][picked_index].copy()
                                temp[red_idx]=temp[red_idx-1]+4
                        if(red_flag):
                            return [self.cards_final[type][picked_index],temp]
                        else:
                            return [self.cards_final[type][picked_index],self.cards_final[type][picked_index]]
                return [self.cards_final[type][picked_index],self.cards_final[type][picked_index]]
                
        if(self.last_play_person=="duijia"):
            return []
        # 接对家的牌就到此为止了
        # 但是接其他家的牌时，模型得学会变通，比如拆牌，接下来是考虑拆牌
        if(type=="single"):
            for idx,cards in enumerate(self.cards_final["pair"]):
                this_poker_point=self.utils.get_card_num(cards[0])
                # 别拆太小的对子
                if(self.card_value_rank.index(this_poker_point)<=7):
                    continue
                check_result=self.check_bigger("single",this_poker_point,type,last_poker_point)
                if(check_result[0]):
                    if(check_result[3]<min_score_gap):
                        min_score_gap=check_result[3]
                        picked_index=idx
            if(picked_index!=-1):
                return [[self.cards_final["pair"][picked_index][0]],[self.cards_final["pair"][picked_index][0]]]
            for idx,cards in enumerate(self.cards_final["three"]):
                this_poker_point=self.utils.get_card_num(cards[0])
                if(self.card_value_rank.index(this_poker_point)<=7):
                    continue
                check_result=self.check_bigger("single",this_poker_point,type,last_poker_point)
                if(check_result[0]):
                    if(check_result[3]<min_score_gap):
                        min_score_gap=check_result[3]
                        picked_index=idx
            if(picked_index!=-1):
                return [[self.cards_final["three"][picked_index][0]],[self.cards_final["three"][picked_index][0]]]
            for idx,cards in enumerate(self.cards_final["three_straight"]):
                this_poker_point=self.utils.get_card_num(cards[0])
                if(self.card_value_rank.index(this_poker_point)<=7):
                    continue
                check_result=self.check_bigger("single",this_poker_point,type,last_poker_point)
                if(check_result[0]):
                    if(check_result[3]<min_score_gap):
                        min_score_gap=check_result[3]
                        picked_index=idx
            if(picked_index!=-1):
                return [[self.cards_final["three_straight"][picked_index][0]],[self.cards_final["three_straight"][picked_index][0]]]
            # 学会拆级牌炸弹
            for idx,cards in enumerate(self.cards_final["bomb"]):
                this_poker_point=self.utils.get_card_num(cards[0])
                if(this_poker_point==self.level):
                    check_result=self.check_bigger("single",this_poker_point,type,last_poker_point)
                    if(check_result[0]):
                        return [[cards[0]],[cards[0]]]
                    
        if(type=="pair"):
            #学会拆级牌炸弹
            for idx,cards in enumerate(self.cards_final["bomb"]):
                this_poker_point=self.utils.get_card_num(cards[0])
                if(this_poker_point==self.level):
                    check_result=self.check_bigger("pair",this_poker_point,"pair",last_poker_point)
                    if(check_result[0]):
                        return [[cards[0],cards[1]],[cards[0],cards[1]]]
                    
            for idx,cards in enumerate(self.cards_final["triple_pairs"]):
                this_poker_point=self.utils.get_card_num(cards[0])
                # 别拆太小的三连对
                if(this_poker_point<=8 and this_poker_point!=self.level):
                    continue
                check_result=self.check_bigger("pair",this_poker_point,type,last_poker_point)
                if(check_result[0]):
                    if(check_result[3]<min_score_gap):
                        min_score_gap=check_result[3]
                        picked_index=idx
            if(picked_index!=-1):
                return [[self.cards_final["triple_pairs"][picked_index][0],self.cards_final["triple_pairs"][picked_index][1]],[self.cards_final["triple_pairs"][picked_index][0],self.cards_final["triple_pairs"][picked_index][1]]]
            
            for idx,cards in enumerate(self.cards_final["three"]):
                this_poker_point=self.utils.get_card_num(cards[0])
                # 别拆太小的光三
                if(this_poker_point<=9 and this_poker_point!=self.level and this_poker_point!=1):
                    continue
                check_result=self.check_bigger("pair",this_poker_point,type,last_poker_point)
                if(check_result[0]):
                    if(check_result[3]<min_score_gap):
                        min_score_gap=check_result[3]
                        picked_index=idx
            if(picked_index!=-1):
                return [[self.cards_final["three"][picked_index][0],self.cards_final["three"][picked_index][1]],[self.cards_final["three"][picked_index][0],self.cards_final["three"][picked_index][1]]]
            
            for idx,cards in enumerate(self.cards_final["three_straight"]):
                this_poker_point=self.utils.get_card_num(cards[0])
                # 别拆太小的三顺
                if(this_poker_point<=9 and this_poker_point!=self.level and this_poker_point!=1):
                    continue
                check_result=self.check_bigger("pair",this_poker_point,type,last_poker_point)
                if(check_result[0]):
                    if(check_result[3]<min_score_gap):
                        min_score_gap=check_result[3]
                        picked_index=idx
            if(picked_index!=-1):
                return [[self.cards_final["three_straight"][picked_index][0],self.cards_final["three_straight"][picked_index][1]],[self.cards_final["three_straight"][picked_index][0],self.cards_final["three_straight"][picked_index][1]]]
        
        if(type=="three"):
            for idx,cards in enumerate(self.cards_final["three_straight"]):
                this_poker_point=self.utils.get_card_num(cards[0])
                check_result=self.check_bigger("three",this_poker_point,type,last_poker_point)
                if(check_result[0]):
                    if(check_result[3]<min_score_gap):
                        min_score_gap=check_result[3]
                        picked_index=idx
            if(picked_index!=-1):
                return [[self.cards_final["three_straight"][picked_index][0],self.cards_final["three_straight"][picked_index][1],self.cards_final["three_straight"][picked_index][2]],[self.cards_final["three_straight"][picked_index][0],self.cards_final["three_straight"][picked_index][1],self.cards_final["three_straight"][picked_index][2]]]
            
            #学会拆级牌炸弹
            for idx,cards in enumerate(self.cards_final["bomb"]):
                this_poker_point=self.utils.get_card_num(cards[0])
                if(this_poker_point==self.level):
                    check_result=self.check_bigger("three",this_poker_point,"three",last_poker_point)
                    if(check_result[0]):
                        return [[cards[0],cards[1],cards[2]],[cards[0],cards[1],cards[2]]]
                    
        if(type=="set"):
            for idx,cards in enumerate(self.cards_final["three"]):
                this_poker_point=self.utils.get_card_num(cards[0])
                check_result=self.check_bigger("three",this_poker_point,"three",last_poker_point)
                if(check_result[0]):
                    if(check_result[3]<min_score_gap):
                        min_score_gap=check_result[3]
                        picked_index=idx
            if(picked_index!=-1):
                two_id=-1
                rank=100
                for idx,cards in enumerate(self.cards_final["pair"]):
                    two_point=self.utils.get_card_num(cards[0])
                    two_point_rank=self.card_value_rank.index(two_point)
                    # 如果只剩三带二这一手牌了另说
                    if(self.current_card_num["me"]!=5):
                        if(two_point_rank<rank):
                            rank=two_point_rank
                            two_id=idx
                if(rank<=9 or self.current_card_num["me"]==5 and rank<=20):
                    return [[self.cards_final["three"][picked_index][0],self.cards_final["three"][picked_index][1],self.cards_final["three"][picked_index][2],
                            self.cards_final["pair"][two_id][0],self.cards_final["pair"][two_id][1]],
                            [self.cards_final["three"][picked_index][0],self.cards_final["three"][picked_index][1],self.cards_final["three"][picked_index][2],
                            self.cards_final["pair"][two_id][0],self.cards_final["pair"][two_id][1]]]
            else:
                #学会拆级牌炸弹
                for idx,cardss in enumerate(self.cards_final["bomb"]):
                    this_poker_point=self.utils.get_card_num(cardss[0])
                    if(this_poker_point==self.level):
                        check_result=self.check_bigger("three",this_poker_point,"three",last_poker_point)
                        if(check_result[0]):
                            two_id=-1
                            rank=100
                            for idx,cards in enumerate(self.cards_final["pair"]):
                                two_point=self.utils.get_card_num(cards[0])
                                two_point_rank=self.card_value_rank.index(two_point)
                                if(two_point_rank<rank):
                                    rank=two_point_rank
                                    two_id=idx
                            if(rank<=7):
                                return [[cardss[0],cardss[1],cardss[2],
                                        self.cards_final["pair"][two_id][0],self.cards_final["pair"][two_id][1]],
                                        [cardss[0],cardss[1],cardss[2],
                                        self.cards_final["pair"][two_id][0],self.cards_final["pair"][two_id][1]]]
        # 我要不要扔炸弹取决于上家或下家的牌数
        # 根据炸弹数量和上家或下家的牌数，决定要不要开炸
        if(self.only_bombs_remaining()==False):
            if(self.last_play_person=="shangjia"):
                if(self.current_card_num["shangjia"]> 5*self.get_bomb_nums()+2 or self.current_card_num["shangjia"]==4):
                    # TODO:如果自己空枪是要打的
                    return []
            elif(self.last_play_person=="xiajia"):
                if(self.current_card_num["xiajia"]> 5*self.get_bomb_nums() or self.current_card_num["xiajia"]==4):
                    return []  
            
        if(self.cards_final["bomb"]!=[]):
            for idx,cards in enumerate(self.cards_final["bomb"]):
                # 获取这张牌的点数
                this_poker_point=self.utils.get_card_num(cards[0])
                this_bomb_len=len(cards)
                
                check_result=self.check_bigger("bomb",this_poker_point,type,last_poker_point,this_bomb_len,last_bomb_len)
                if(check_result[0]):
                    if(check_result[3]<min_score_gap):
                        min_score_gap=check_result[3]
                        picked_index=idx
            if(picked_index!=-1):
                if(self.red_num in self.cards_final["bomb"][picked_index] and self.red_num+54 in self.cards_final["bomb"][picked_index]):
                    length=len(self.cards_final["bomb"][picked_index])
                    temp=self.cards_final["bomb"][picked_index][:length-2]
                    temp.extend([self.cards_final["bomb"][picked_index][0],self.cards_final["bomb"][picked_index][0]])
                    return [self.cards_final["bomb"][picked_index],temp]
                elif(self.red_num in self.cards_final["bomb"][picked_index] or self.red_num+54 in self.cards_final["bomb"][picked_index]):
                    length=len(self.cards_final["bomb"][picked_index])
                    temp=self.cards_final["bomb"][picked_index][:length-1]
                    temp.extend([self.cards_final["bomb"][picked_index][0]])
                    return [self.cards_final["bomb"][picked_index],temp]
                else:
                    return [self.cards_final["bomb"][picked_index],self.cards_final["bomb"][picked_index]]
        
        if(self.cards_final["straight_flush"]!=[]):
            for idx,cards in enumerate(self.cards_final["straight_flush"]):
                # 获取这张牌的点数
                if(self.red_num == cards[0] or self.red_num+54 == cards[0]):
                    this_poker_point=self.utils.get_card_num(cards[1])-1
                else:
                    this_poker_point=self.utils.get_card_num(cards[0])
                check_result=self.check_bigger("straight_flush",this_poker_point,type,last_poker_point,0,last_bomb_len)
                if(check_result[0]):
                    if(check_result[3]<min_score_gap):
                        min_score_gap=check_result[3]
                        picked_index=idx
            if(picked_index!=-1):
                red_flag=False
                type="straight_flush"
                if(self.red_num in self.cards_final[type][picked_index]):
                    red_idx=self.cards_final[type][picked_index].index(self.red_num)
                    red_flag=True
                elif(self.red_num+54 in self.cards_final[type][picked_index]):
                    red_idx=self.cards_final[type][picked_index].index(self.red_num+54)
                    red_flag=True
                if(red_flag):
                    if(red_idx==0):
                        temp=[self.cards_final[type][picked_index][1]-4]
                        temp.extend(self.cards_final[type][picked_index][1:])
                    else:
                        temp=self.cards_final[type][picked_index].copy()
                        temp[red_idx]=temp[red_idx-1]+4
                if(red_flag):
                    return [self.cards_final[type][picked_index],temp]
                else:
                    return [self.cards_final[type][picked_index],self.cards_final[type][picked_index]]
        return []
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
        cur_pokertype, cur_points = self.check_poker_type(claim)
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
                last_pokertype, last_points = self.check_poker_type(self.lastMove['claim'])
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
        action_pok = [self.utils.Num2Poker(p) for p in action]
        claim_pok = [self.utils.Num2Poker(p) for p in claim]
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
    
    '''
    check_poker_type函数:用于检查牌型
    输入：
        poker: 牌(除了joker和JOker之外, 都是h,d,s,c+'A'~'K'的两位字符串)的列表
    输出：
        牌型,牌型参数
    '''
    def check_poker_type(self, poker: list):
        if poker == []:
            return ["pass", []]
        # covering = "h" + level
        if len(poker) == 1:
            return ["single", [poker[0][1]]]
        if len(poker) == 2:
            if poker[0][1] == poker[1][1]:
                return ["pair", [poker[0][1]]]
            return ["invalid", []]
        # 大于等于三张
        points = [p[1] for p in poker]
        cnt = Counter(points)
        vals = list(cnt.values())
        if len(poker) == 3:
            if "o" in points or "O" in points: 
                return ["invalid", []]
            if vals.count(3) == 1:
                return ["three", [points[0]]]
            return ["invalid", []]
        if len(poker) == 4: # should be a bomb
            if "o" in points or "O" in points: # should be a rocket
                if cnt["o"] == 2 and cnt["O"] == 2:
                    return ["rocket", ["o"]]
                return ["invalid", []]
            if vals.count(4) == 1:
                return ["bomb", [4, points[0]]]
            return ["invalid", []]
        if len(poker) == 5: # could be straight, straight flush, three&two or bomb
            if vals.count(5) == 1:
                return ["bomb", [5, points[0]]]
            if vals.count(3) == 1 and vals.count(2) == 1: # set: 三带二 
                three = ''
                two = ''
                for k in list(cnt.keys()):
                    if cnt[k] == 3:
                        three = k
                    elif cnt[k] == 2:
                        two = k
                return ["set", [three, two]]
            if vals.count(1) == 5: # should be straight
                # 顺子里不能有大小王
                if("o" in points or "O" in points):
                    return ["invalid", []]
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
                            return ["straight_flush", ['0']]
                        return ["straight", ['0']]
                sup_straight = [self.cardscale[self.cardscale.index(first)+i] for i in range(5)]
                if points == sup_straight:
                    if flush:
                        return ["straight_flush", [first]]
                    return ["straight", [first]]
            return ["invalid", []]
        if len(poker) == 6: # could be triple_pairs, three_straight, bomb
            if vals.count(6) == 1:
                return ["bomb", [6, points[0]]]
            if vals.count(3) == 2:
                ks = []
                for k in list(cnt.keys()):
                    ks.append(k)
                ks.sort(key=lambda x: self.cardscale.index(x))
                if 'A' in ks:
                    if ks == ['A', '2']:
                        return ["three_straight", ['A']]
                    if ks == ['A', 'K']:
                        return ["three_straight", ['K']]
                    return ["invalid", []]
                if self.cardscale.index(ks[1]) - self.cardscale.index(ks[0]) == 1:
                    return ["three_straight", [ks[0]]]
            if vals.count(2) == 3:
                ks = []
                for k in list(cnt.keys()):
                    ks.append(k)
                ks.sort(key=lambda x: self.cardscale.index(x))
                if 'A' in ks:
                    if ks == ['A', 'Q', 'K']:
                        return ["triple_pairs", ['Q']]
                    if ks == ['A', '2', '3']:
                        return ["triple_pairs", ['A']]
                    return ["invalid", []]
                pairs = [self.cardscale[self.cardscale.index(ks[0])+i] for i in range(3)]
                if ks == pairs:
                    return ["triple_pairs", [ks[0]]]
            return ["invalid", []]
        if len(poker) > 6 and len(poker) <= 10:
            if vals.count(len(poker)) == 1:
                bomb = points[0]
                return ["bomb", [len(poker), bomb]]
        return ["invalid", []]
    
    '''
    check_bigger函数:用于检查牌的大小
    输入：
        type1, point1, type2, point2: 牌型,牌型参数
    输出：
        是否1比2更大,type1是否是炸,type2是否是炸,相对分数差(1-2)
    '''
    def check_bigger(self, type1, point1, type2, point2,bomblen1=0,bomblen2=0):

        if type1 in self.bombtypes and type2 in self.bombtypes:
            relative_score1=0
            relative_score2=0
            if(type1=="straight_flush"):
                relative_score1=self.relative_score_of_straight_flush(point1)
            else:
                relative_score1=self.relative_score_of_bomb(point1,bomblen1)
            if(type2=="straight_flush"):
                relative_score2=self.relative_score_of_straight_flush(point2)
            else:
                relative_score2=self.relative_score_of_bomb(point2,bomblen2)
            return relative_score1>relative_score2,1,1,relative_score1-relative_score2
        elif type1 in self.bombtypes and type2 not in self.bombtypes:
            relative_score1=0
            if(type1=="straight_flush"):
                relative_score1=self.relative_score_of_straight_flush(point1)
            else:
                relative_score1=self.relative_score_of_bomb(point1,bomblen1)
            return True,1,0,relative_score1
        elif type1 not in self.bombtypes and type2 in self.bombtypes:
            relative_score2=0
            if(type2=="straight_flush"):
                relative_score2=self.relative_score_of_straight_flush(point2)
            else:
                relative_score2=self.relative_score_of_bomb(point2,bomblen2)
            return False,0,1,-relative_score2
        else:
            if(type1!=type2):
                return False,-1,-1,-1
            else:
                if(type1=="single" or type1=="pair" or type1=="three" or type1=="set"):
                    rank1=self.card_value_rank.index(point1)
                    rank2=self.card_value_rank.index(point2)
                    return rank1>rank2,0,0,rank1-rank2
                elif(type1=="straight" or type1=="three_straight" or type1=="triple_pairs"):
                    rank1=self.card_scale_num.index(point1)
                    rank2=self.card_scale_num.index(point2)
                    return rank1>rank2,0,0,rank1-rank2
