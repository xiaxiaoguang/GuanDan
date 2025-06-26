'''
@author: 杨奇滨
@date: 2024-12-14
@description: 一个掼蛋AI bot
'''
import json
from expert_utils import *
from expert_env import *
    
def expert_decision(state):
    # 初始化环境
    env=ExpertGuanDanEnv()

    env.level=state.level
    env.red_num=(env.level-1)*4

    #当局大小顺序的列表
    env.card_value_rank=env.utils.card_value_ranks[env.level]

    #取出我初始的发牌，虽然state里有所有人的发牌，但是我只能看见我的
    env.my_deliver=state.deliver[env.my_id]

    #需要维护一个当前手牌的列表，初始化为my_deliver
    env.my_current_cards=env.my_deliver

    # 这是一个15维字典，分别表示初始发牌不管花色的A(1),2,3,4,5,6,7,8,9,10,J(11),Q(12),K(13),joker(14),JOker(15)的张数
    env.my_deliver_without_color={1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0,10:0,11:0,12:0,13:0,14:0,15:0}
    # 这是带花色的字典，key是牌的编号，value是花色字典，花色字典的key是花色，value是张数
    env.my_deliver_with_color={
        1:{'h':0,'d':0,'s':0,'c':0},2:{'h':0,'d':0,'s':0,'c':0},3:{'h':0,'d':0,'s':0,'c':0},
        4:{'h':0,'d':0,'s':0,'c':0},5:{'h':0,'d':0,'s':0,'c':0},6:{'h':0,'d':0,'s':0,'c':0},
        7:{'h':0,'d':0,'s':0,'c':0},8:{'h':0,'d':0,'s':0,'c':0},9:{'h':0,'d':0,'s':0,'c':0},
        10:{'h':0,'d':0,'s':0,'c':0},11:{'h':0,'d':0,'s':0,'c':0},12:{'h':0,'d':0,'s':0,'c':0},
        13:{'h':0,'d':0,'s':0,'c':0},14:0,15:0
    }

    # 初始化发牌
    for i in range(len(env.my_deliver)):
        card_num=env.utils.get_card_num(env.my_deliver[i])
        card=env.utils.Num2Poker(env.my_deliver[i])
        env.my_deliver_without_color[card_num]+=1
        if(card_num>=14):
            env.my_deliver_with_color[card_num]+=1
        else:
            env.my_deliver_with_color[card_num][card[0]]+=1

    # 初始化current_cards
    env.my_current_cards_with_color=state.all_cards[env.my_id]
    for rank,suits in env.my_current_cards_with_color.items():
        if(rank<=13):
            env.my_current_cards_without_color[rank]=suits['h']+suits['d']+suits['s']+suits['c']
        else:
            env.my_current_cards_without_color[rank]=suits
            
    env.current_card_num["me"]=state.current_card_num[env.my_id]
    env.current_card_num["duijia"]=state.current_card_num[(env.my_id+2)%4]
    env.current_card_num["shangjia"]=state.current_card_num[(env.my_id+3)%4]
    env.current_card_num["xiajia"]=state.current_card_num[(env.my_id+1)%4]
            
    #应该没有进还贡，不是deal就是play。这里是play,取出history
    env.group_cards_with_color()
    if(len(state.done)==0):
        # 四人都还没出完牌
        if(state.history==[] or len(state.history)>=4 and state.history[-1] in env.nocard and state.history[2] in env.nocard and state.history[3] in env.nocard):
            # 上一圈我大了，或者我是首发牌的一方
            env.my_current_response=env.decide_cards_when_first()
        else:
            if(state.history[-1] not in env.nocard):
                env.last_play_person="shangjia"
                #以claim为准
                last_play_cards=state.history[-1][1]    
            elif(state.history[-2] not in env.nocard):
                env.last_play_person="duijia"
                last_play_cards=state.history[-2][1]
            else:
                env.last_play_person="xiajia"
                last_play_cards=state.history[-3][1]

            cards_pokers=env.utils.Num2Pokers(last_play_cards)
            last_poker_type=env.check_poker_type(cards_pokers) 
            env.my_current_response=env.decide_cards_when_not_first(last_poker_type)

    elif(len(state.done)==1):
        # 至少在这一圈之前有一家已经出完牌了
        if(state.history[-1] in env.nocard and state.history[-2] in env.nocard):
            if(env.just_done==(env.my_id+1)%4):
                # 我的下家刚刚做头游，我的上家和对家都没接牌
                env.last_play_person="xiajia"
                last_play_cards=state.history[-3][1]
                cards_pokers=env.utils.Num2Pokers(last_play_cards)
                last_poker_type=env.check_poker_type(cards_pokers) 
                env.my_current_response=env.decide_cards_when_not_first(last_poker_type)
            else:
                env.my_current_response=env.decide_cards_when_first()
        else:
            if(env.just_done==-1):
                if((env.my_id+1)%4 in env.done):
                    # 下家是上游
                    if(state.history[-1] not in env.nocard):
                        env.last_play_person="shangjia"
                        last_play_cards=state.history[-1][1]
                    elif(state.history[-2] not in env.nocard):
                        env.last_play_person="duijia"
                        last_play_cards=state.history[-2][1]
                elif((env.my_id+2)%4 in env.done):
                    # 对家是上游
                    if(state.history[-1] not in env.nocard):
                        env.last_play_person="shangjia"
                        last_play_cards=state.history[-1][1]
                    elif(state.history[-2] not in env.nocard):
                        env.last_play_person="xiajia"
                        last_play_cards=state.history[-2][1]
                else:
                    # 上家是上游
                    if(state.history[-1] not in env.nocard):
                        env.last_play_person="duijia"
                        last_play_cards=state.history[-1][1]
                    elif(state.history[-2] not in env.nocard):
                        env.last_play_person="xiajia"
                        last_play_cards=state.history[-2][1]
            else:
                if(env.just_done==(env.my_id+1)%4 or env.just_done==(env.my_id+2)%4):
                    # 我的下家或对家刚刚做头游
                    if(state.history[-1] not in env.nocard):
                        env.last_play_person="shangjia"
                        last_play_cards=state.history[-1][1]
                    elif(state.history[-2] not in env.nocard):
                        env.last_play_person="duijia"
                        last_play_cards=state.history[-2][1]
                elif(env.just_done==(env.my_id+3)%4):
                    # 我的上家刚刚做头游
                    env.last_play_person="shangjia"
                    last_play_cards=state.history[-1][1]
            cards_pokers=env.utils.Num2Pokers(last_play_cards)
            last_poker_type=env.check_poker_type(cards_pokers) 
            env.my_current_response=env.decide_cards_when_not_first(last_poker_type)
            
    elif(len(state.done)==2):
        # 有2人已经出完牌了，分类讨论
        # 上一个人没出牌
        if(state.history[-1] in env.nocard):
            if(env.just_done==-1):
                # 上一圈我大了
                env.my_current_response=env.decide_cards_when_first()
            else:
                # 走的两个人只能是对家和下家，需要判断一下谁刚走
                if(env.just_done==(env.my_id+1)%4):
                    # 下家刚走
                    env.last_play_person="xiajia"
                    last_play_cards=state.history[-2][1]
                    cards_pokers=env.utils.Num2Pokers(last_play_cards)
                    last_poker_type=env.check_poker_type(cards_pokers) 
                    env.my_current_response=env.decide_cards_when_not_first(last_poker_type)
                else:
                    # 对家刚走
                    env.last_play_person="duijia"
                    last_play_cards=state.history[-2][1]
                    cards_pokers=env.utils.Num2Pokers(last_play_cards)
                    last_poker_type=env.check_poker_type(cards_pokers) 
                    env.my_current_response=env.decide_cards_when_not_first(last_poker_type)
        else:
            #上一个人出牌了
            if(env.just_done==-1):
                # 一打一就当上家打好了
                env.last_play_person="shangjia"
                last_play_cards=state.history[-1][1]
                cards_pokers=env.utils.Num2Pokers(last_play_cards)
                last_poker_type=env.check_poker_type(cards_pokers) 
                env.my_current_response=env.decide_cards_when_not_first(last_poker_type)
            else:
                if((env.my_id+1)%4 in env.done):
                    #走的两个人是下家和对家，我正常压上家即可
                    env.last_play_person="shangjia"
                    last_play_cards=state.history[-1][1]
                    cards_pokers=env.utils.Num2Pokers(last_play_cards)
                    last_poker_type=env.check_poker_type(cards_pokers) 
                    env.my_current_response=env.decide_cards_when_not_first(last_poker_type)
                else:
                    #走的两个人是上家和对家，谁是刚走的
                    if(env.just_done==(env.my_id+2)%4):
                        # 对家刚走
                        env.last_play_person="duijia"
                        last_play_cards=state.history[-1][1]
                        cards_pokers=env.utils.Num2Pokers(last_play_cards)
                        last_poker_type=env.check_poker_type(cards_pokers) 
                        env.my_current_response=env.decide_cards_when_not_first(last_poker_type)
                    else:
                        # 上家刚走
                        env.last_play_person="shangjia"
                        last_play_cards=state.history[-1][1]
                        cards_pokers=env.utils.Num2Pokers(last_play_cards)
                        last_poker_type=env.check_poker_type(cards_pokers) 
                        env.my_current_response=env.decide_cards_when_not_first(last_poker_type)

    return env.my_current_response

