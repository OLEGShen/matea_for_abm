import csv
import json
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '../../../'))
from simulation.models.agents.LLMAgent import LLMAgent
class RiderLLMAgent(LLMAgent):
    def __init__(self, role_param_dict):

        super().__init__('R',has_chat_history=False,online_track=False,json_format=True,system_prompt = '',llm_model='deepseek-r1:32b')
        self.system_prompt = '''You are playing a human,your job is a Meituan rider.You need to earn money by working.
                                You live in a town, the town map is presented as coordinates.
                                {role_description}
                                '''
        self.sensible_degree = 0.7
        self.SR_CoT = False
        self.role_description = ''
        self.personality = 'lazy'
        if role_param_dict is not None:
            # self.sensible_degree = role_param_dict['sensible_degree']
            # self.SR_CoT = role_param_dict['SR_CoT']
            self.role_description = role_param_dict['role_description']
            self.personality = role_param_dict['personality']
    def decide_time(self,runner_step,info):
        self.llm_model='deepseek-r1:32b'
        time_prompt = '''
        Among {rider_num} riders, your riding distance yesterday ranked {dis_rank}, the amount of money earned ranked 
        {money_rank}, and the number of orders received ranked {order_rank}.
        The time you started working yesterday was {before_go_work_time}, and the time you finished working yesterday 
        was {before_get_off_work_time},Please use this information to decide whether you want to change your working hours today.
        {rational_or_sensible_prompt}
         Based on the above information, please give your time to go to work today, time to get off work today.
         You should think step by step.
          - When you speak, you must use the following format in json:
              {{
              "go_to_work_time": the time，The time should be in the format *:00,
              "get_off_work_time": the time，The time should be in the format *:00,
              }}
          '''
        rational_prompt = 'You should think in a completely rational way, without considering personal characteristics, ' \
                          'and you need to do more mathematical calculations and analysis.'
        sensible_prompt = 'You should think in a completely emotional way, mainly considering your personal character ' \
                          'without rational calculation and analysis, and your character is {personality}. '
        param_dict={
            'role_description': self.role_description,
            'rational_or_sensible_prompt':'',
            'rider_num': info['rider_num'],
            'dis_rank': str(info['dis_rank'])+'/'+str(info['rider_num']),
            'money_rank': str(info['money_rank'])+'/'+str(info['rider_num']),
            'order_rank': str(info['order_rank'])+'/'+str(info['rider_num']),
            'before_go_work_time':str(info['before_go_work_time'])+':00',
            'before_get_off_work_time':str(info['before_get_off_work_time'])+':00',
        }
        try:
            if self.SR_CoT:
                param_dict['rational_or_sensible_prompt'] = rational_prompt
                print(param_dict)
                response_rational,rational_think = self.get_response(time_prompt, input_param_dict=param_dict)
                param_dict['rational_or_sensible_prompt'] = sensible_prompt
                param_dict.update({'personality': self.personality})
                print(param_dict)
                response_sensible,sensible_think = self.get_response(time_prompt, input_param_dict=param_dict)

                if self.sensible_degree > 0.5:
                    time_int1 = response_sensible.get('go_to_work_time').split(':')
                    time_int2 = response_sensible.get('get_off_work_time').split(':')
                else:
                    time_int1 = response_rational.get('go_to_work_time').split(':')
                    time_int2 = response_rational.get('get_off_work_time').split(':')
                result={
                    "go_work_time":str(time_int1[0])+':00',
                    'get_off_work_time':str(time_int2[0])+':00',
                }
                self.llm_thought_log(runner_step, param_dict, ration_thought=response_rational.get('chain_of_thought'),
                                     sensibility_thought=response_sensible.get('chain_of_thought'), result=result,think=rational_think)
                return int(time_int1[0]),int(time_int2[0])
            else:
                print(param_dict)
                response,think = self.get_response(time_prompt,input_param_dict=param_dict)
                time_int1 = response.get('go_to_work_time').split(':')
                time_int2 = response.get('get_off_work_time').split(':')
                # work_hour = int(time_int2[0]) - int(time_int1[0])
                result = {
                    "go_work_time": str(time_int1[0]) + ':00',
                    'get_off_work_time': str(time_int2[0]) + ':00',
                }
                self.llm_thought_log(runner_step, param_dict, result=result,think=think)
                return int(time_int1[0]),int(time_int2[0])
        except Exception as e:
            print(e)
            print('LLM 选择工作时间失败！')
            result = {
                "go_work_time": str(info['before_go_work_time']) + ':00',
                'get_off_work_time': str(info['before_get_off_work_time']) + ':00',
            }
            self.llm_thought_log(runner_step, param_dict, mixed_thought='LLM 选择工作时间失败！',result=result,think=think)
            return info['before_go_work_time'],info['before_get_off_work_time']
    def take_order(self,runner_step, info):
        self.llm_model='ChatGPT'
        order_prompt = '''
                        Now there is the following order information, each order information is represented by a list
                        item, including the Includes pickup and delivery locations for orders and the money that can be
                        obtained.The order information is as follows:
                        {order_list}
                        Your current location is {now_location}.
                        You can currently accept up to {accept_count} order
                        {rational_or_sensible_prompt}
                        Based on the above information, please give The list of order numbers you choose to take.
                        The order number you choose must be in the given order list.
                        You should think step by step.
                          - When you speak, you must use the following format in json:
                              {{
                              "order_list":A list of only order_id that you choose to take.
                              }}
                        '''
        rational_prompt = 'You should think in a completely rational way, without considering personal characteristics, ' \
                          'and you need to do more mathematical calculations and analysis. For example, the Euclidean ' \
                          'distance formula is $d = \sqrt{(x1-x2)^2 + (y1-y2)^2}$ '
        sensible_prompt = 'You should think in a completely emotional way, mainly considering your personal character ' \
                          'without rational calculation and analysis, and your character is {personality}. ' \
                          'You should also consider from a psychological perspective, such as marginal effect and Matthew effect.'
        param_dict = {
            'role_description': self.role_description,
            'rational_or_sensible_prompt': '',
            'order_list' : info['order_list'],
            'now_location':info['now_location'],
            'accept_count': info['accept_count']
        }
        try:
            if self.SR_CoT:
                param_dict['rational_or_sensible_prompt'] = rational_prompt
                print(param_dict)
                response_rational=  self.get_response(order_prompt,input_param_dict = param_dict)
                param_dict['rational_or_sensible_prompt'] = sensible_prompt
                param_dict.update({'personality':self.personality})
                print(param_dict)
                response_sensible = self.get_response(order_prompt, input_param_dict=param_dict)

                if self.sensible_degree>0.5:
                    final_result = response_sensible['order_list']
                else:
                    final_result = response_rational['order_list']
                self.llm_thought_log(runner_step, param_dict, ration_thought=response_rational.get('chain_of_thought'),
                                     sensibility_thought=response_sensible.get('chain_of_thought'),result=final_result)
                return final_result
            else:
                print(param_dict)
                response= self.get_response(order_prompt, input_param_dict=param_dict)

                final_result = response['order_list']
                self.llm_thought_log(runner_step, param_dict,
                                     result=final_result)
                return final_result
        except Exception as e:
            print(e)
            print('LLM 选择订单失败！')
            self.llm_thought_log(runner_step, param_dict, mixed_thought='LLM 选择订单失败！',result=[])
            return []
    def llm_thought_log(self, runner_step, param_dict, ration_thought='',sensibility_thought='',mixed_thought='',result=None,think=''):
        file_path = f"agent_log/{self.start_time}/{self.id}_thought.json"
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                try:
                    data = json.load(file)  # 读取已有的 JSON 列表
                except json.JSONDecodeError:
                    data = []  # 文件为空或不是 JSON，初始化为空列表
        else:
            data = []  # 文件不存在，初始化为空列表

            # 添加新的 JSON 数据
        new_data = {
            "runner_step":runner_step,
            "param_dict":param_dict,
            "ration_thought":ration_thought,
            "sensibility_thought":sensibility_thought,
            "mixed_thought":mixed_thought,
            "result":result,
            'think':think
        }
        data.append(new_data)
        # 将更新后的内容写回文件
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    role_param_dict = {
            "role_description": "You are Chloe Lewis, a 26-year-old Male delivery rider who is an experienced driver seeking a steady income.",
            "personality": "hard_working,You are a hard-working rider who will work longer hours and take more orders to earn more money."
    }

    agent = RiderLLMAgent(role_param_dict)
    now_orders_info = [{
        "order_id": 1,
        "pickup_location": [32,33],
        "delivery_location": [92,108],
        "money": 5
    }]
    info = {
        'order_list' :now_orders_info,
        'now_location': [2,19],
        'accept_count': 3
    }
    time_info ={
        'dis_rank':34,
        'money_rank':34,
        'order_rank': 34,
        'before_go_work_time': 8,
        'before_get_off_work_time': 18,
        'rider_num': 100,
    }
    up,sleep = agent.take_order(10,info)
    print(up)
    print(sleep)