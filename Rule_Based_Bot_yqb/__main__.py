'''
@author: 杨奇滨
@date: 2024-12-14-2025.1.9
@description: 一个掼蛋AI bot
'''
import json
from expert_utils import *
from expert_env import *
   
if __name__ == "__main__":
    
    # 初始化环境
    env=ExpertGuanDanEnv()
    # 解析读入的JSON
    full_input = json.loads(input())

    # 分析自己收到的输入和自己过往的输出，并恢复状态
    all_requests = full_input["requests"]
    all_responses = full_input["responses"]
    len_responses=len(all_responses)

    #把发牌专拎出来
    deal_request=all_requests[0]

    #取出全局信息
    globalInfo=deal_request["global"]

    #env.level为数字，需要转换为数字level_num
    level_num=env.utils.level_to_value[globalInfo["level"]]
    env.red_num=(level_num-1)*4
    env.level=level_num

    #当局大小顺序的列表
    env.card_value_rank=env.utils.card_value_ranks[level_num]

    #取出我初始的发牌
    env.my_deliver=deal_request["deliver"]
    env.my_id=deal_request["your_id"]

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
    env.my_current_cards_without_color=env.my_deliver_without_color
    env.my_current_cards_with_color=env.my_deliver_with_color

    for i in range(len_responses):
        request = all_requests[i] # i回合我的输入
        response = all_responses[i] # i回合我的输出
        # TODO: 根据规则，处理这些输入输出，从而逐渐恢复状态到当前回合
        if(request["stage"]=="play"):
            history=request["history"]
            done=request["done"]
            env.update_card_nums(history,done)
                    
        if(response not in env.nocard):
            #如果我出了牌，那么需要更新我的手牌;此时response形如[[...][...]],第一个列表为actual而第二个为claim
            actual_played_cards=response[0]
            # 更新我的手牌
            env.my_current_cards=[item for item in env.my_current_cards if item not in actual_played_cards]
            # 更新我的手牌不管花色的字典
            for played_card in actual_played_cards:
                card_num=env.utils.get_card_num(played_card)
                card=env.utils.Num2Poker(played_card)
                env.my_current_cards_without_color[card_num]-=1
                if(card_num>=14):
                    env.my_current_cards_with_color[card_num]-=1
                else:
                    env.my_current_cards_with_color[card_num][card[0]]-=1
            #更新我的手牌数
            env.current_card_num["me"]-=len(actual_played_cards)
            
    # 保存倒数第二个history的Len，为了确定是否是history len=4转向3，这样意味着最新的一回合上游刚刚产生。 
    last_history_len=0 
    if(len(all_requests)>=2):   
        last_request=all_requests[-2]
        if(last_request["stage"]=="play"):
            last_history=last_request["history"]
            last_history_len=len(last_history)
            last_history_done=last_request["done"]

    # 看看自己最新一回合输入
    curr_request = all_requests[-1]
    if(curr_request["stage"]=="deal"):
        env.my_current_response=[]
    else:
        #应该没有进还贡，不是deal就是play。这里是play,取出history
        env.group_cards_with_color()
        curr_history=curr_request["history"]
        done=curr_request["done"]
        env.update_card_nums(curr_history,done)
        if(done!=[]):
            env.first_num=done[0]
            if(env.first_num==(env.my_id+1)%4):
                env.shangyou="xiajia"
            elif(env.first_num==(env.my_id+2)%4):
                env.shangyou="duijia"
            elif(env.first_num==(env.my_id+3)%4):
                env.shangyou="shangjia"
            
        curr_history_len=len(curr_history)
        if(curr_history_len==4):
            # 至少在这一圈之前四人都还没出完牌
            if(curr_history[1] in env.nocard and curr_history[2] in env.nocard and curr_history[3] in env.nocard):
                # 上一圈我大了，或者我是首发牌的一方
                env.my_current_response=env.decide_cards_when_first()
            else:
                last_play_k = max(k for k in range(curr_history_len) if curr_history[k] not in env.nocard)
                if(last_play_k!=2):
                    # 如果最近一次不是对家出的牌，我需要压牌
                    last_play_cards=curr_history[last_play_k][1]
                    if(last_play_k==1):
                        env.last_play_person="xiajia"
                    else:
                        env.last_play_person="shangjia"
                    # 以claim为准
                    cards_pokers=env.utils.Num2Pokers(last_play_cards)
                    last_poker_type=env.check_poker_type(cards_pokers) 
                    env.my_current_response=env.decide_cards_when_not_first(last_poker_type)
                else:
                    # 如果最近一次是上家出牌，动态考虑压牌，小牌可以顺过
                    env.last_play_person="duijia"
                    last_play_cards=curr_history[last_play_k][1]
                    cards_pokers=env.utils.Num2Pokers(last_play_cards)
                    last_poker_type=env.check_poker_type(cards_pokers) 
                    env.my_current_response=env.decide_cards_when_not_first(last_poker_type)

        elif(curr_history_len==3):
            # 至少在这一圈之前有一家已经出完牌了
            if(last_history_len==4 and last_history_done==[]):
                # 说明这一圈产生了上游,这一圈的history是不包括我自己的，而是下家->对家->上家
                last_play_k = max(k for k in range(curr_history_len) if curr_history[k] not in env.nocard)
                if(last_play_k==0):
                    env.last_play_person="xiajia"
                elif(last_play_k==1):
                    env.last_play_person="duijia"
                else:
                    env.last_play_person="shangjia"
                last_play_cards=curr_history[last_play_k][1]
                cards_pokers=env.utils.Num2Pokers(last_play_cards)
                last_poker_type=env.check_poker_type(cards_pokers) 
                env.my_current_response=env.decide_cards_when_not_first(last_poker_type)
            elif(len(done)==2):
                # 说明这一圈产生了二游且没有打断接风
                second_num=done[1]
                if(second_num==(env.my_id+1)%4):
                    env.last_play_person="xiajia"
                elif(second_num==(env.my_id+2)%4):
                    env.last_play_person="duijia"
                elif(second_num==(env.my_id+3)%4):
                    env.last_play_person="shangjia"
                last_play_k = max(k for k in range(curr_history_len) if curr_history[k] not in env.nocard)
                last_play_cards=curr_history[last_play_k][1]
                cards_pokers=env.utils.Num2Pokers(last_play_cards)
                last_poker_type=env.check_poker_type(cards_pokers) 
                env.my_current_response=env.decide_cards_when_not_first(last_poker_type)
            else:
                if(curr_history[1] in env.nocard and curr_history[2] in env.nocard):
                    # 上一圈我大了，或者对家给我接风了
                    env.my_current_response=env.decide_cards_when_first()
                else:
                    last_play_k = max(k for k in range(curr_history_len) if curr_history[k] not in env.nocard)
                    '''
                    if(env.current_card_num["duijia"]==0):
                        if(env.current_card_num["xiajia"]==0 or env.current_card_num["shangjia"]==0):
                            if(env.shangyou!="duijia"):
                                env.last_play_person="duijia"
                        else:
                            if(last_play_k==1):
                                env.last_play_person="xiajia"
                            else:
                                env.last_play_person="shangjia"
                    elif(env.current_card_num["shangjia"]==0):
                        # 如果最近一次不是对家出牌，我需要压牌
                        if(env.shangyou=="shangjia"):
                            if(last_play_k==1):
                                env.last_play_person="xiajia"
                            else:
                                env.last_play_person="duijia"
                        elif(env.shangyou=="duijia"):
                            env.last_play_person="shangjia"
                    elif(env.current_card_num["xiajia"]==0):
                        # 如果最近一次不是对家出牌，我需要压牌
                        if(env.shangyou=="xiajia"):
                            if(last_play_k==1):
                                env.last_play_person="duijia"
                            else:
                                env.last_play_person="shangjia"
                        elif(env.shangyou=="duijia"):
                            env.last_play_person="xiajia"
                    else:
                        #说明最近一次是对家出牌
                        env.last_play_person="duijia"
                    # 以claim为准
                    last_play_cards=curr_history[last_play_k][1]
                    cards_pokers=env.utils.Num2Pokers(last_play_cards)
                    last_poker_type=env.check_poker_type(cards_pokers) 
                    env.my_current_response=env.decide_cards_when_not_first(last_poker_type)
                    '''
                    if(env.shangyou=="duijia"):
                        # 我要一打二了，都当上家打算了
                        if(last_play_k==1):
                            env.last_play_person="xiajia"
                        else:
                            env.last_play_person="shangjia"
                        last_play_cards=curr_history[last_play_k][1]
                        cards_pokers=env.utils.Num2Pokers(last_play_cards)
                        last_poker_type=env.check_poker_type(cards_pokers) 
                        env.my_current_response=env.decide_cards_when_not_first(last_poker_type)
                    elif(env.shangyou=="xiajia"):
                        if(last_play_k==1):
                            env.last_play_person="duijia"
                        else:
                            env.last_play_person="shangjia"
                        last_play_cards=curr_history[last_play_k][1]
                        cards_pokers=env.utils.Num2Pokers(last_play_cards)
                        last_poker_type=env.check_poker_type(cards_pokers) 
                        env.my_current_response=env.decide_cards_when_not_first(last_poker_type)
                    else: #上家是上游
                        if(last_play_k==1):
                            env.last_play_person="xiajia"
                        else:
                            env.last_play_person="duijia"
                        last_play_cards=curr_history[last_play_k][1]
                        cards_pokers=env.utils.Num2Pokers(last_play_cards)
                        last_poker_type=env.check_poker_type(cards_pokers) 
                        env.my_current_response=env.decide_cards_when_not_first(last_poker_type)
                
                
        elif(curr_history_len==2):
            # 有2人已经出完牌了，分类讨论
            if(curr_history[1] in env.nocard):
                # 上一圈我大了，或者对家给我接风了
                env.my_current_response=env.decide_cards_when_first()
            else:
                last_play_cards=curr_history[1][1]
                # 以claim为准，且此时last_play_k一定为1
                cards_pokers=env.utils.Num2Pokers(last_play_cards)
                last_poker_type=env.check_poker_type(cards_pokers) 
                env.my_current_response=env.decide_cards_when_not_first(last_poker_type)
            
    print(json.dumps({
        "response": env.my_current_response,
    }))

