from refer import REFER

import random
import os
from http import HTTPStatus
import dashscope
import json
import re
import time
import eventlet

dashscope.api_key = "sk-f1c7bec047694f26b9fa1569a59af88a"

def call_with_messages(user_content):
    messages = [{'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': user_content}]
    response = dashscope.Generation.call(
        #dashscope.Generation.Models.qwen_max_1201,
        model='qwen-max',
        messages=messages,
        # set the random seed, optional, default to 1234 if not set
        seed=random.randint(1, 10000),
        result_format='message',  # set the result to be "message" format.
    )
    if response.status_code == HTTPStatus.OK:
        # print(response)
        pass
    else:
        print('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
            response.request_id, response.status_code,
            response.code, response.message
        ))
    return response


# user_content = 'Rerferring text descibes one object with some states or attributes information. I hope you can tell me what the referring object is. \
#                 For example, given the referring text: woman in white shirt looking down at laptop computer. \
#                 You should output:  a woman. Then given the text : the visible table to the right of the bowl. Give your output.'
# user_content = "Rerferring text descibes one object with some states or attributes information. I hope you can tell me what the referring object is. \
#                 For example, given the referring text: woman in white shirt looking down at laptop computer. \
#                 You should output:  the referring object is 'a woman'. Given the referring text: a caka on table. You should output : the referring object is 'a cake'. \
#                 Then given the text : {}. Give your output."

# user_content = 'Rerferring text descibes one object with some states or attributes information. I hope you can break down the referring text. \
#                 Each output sentence should only include one state or attribute. \
#                 For example, given the referring text: woman in white shirt looking down at laptop computer. \
#                 You should output: 1. a woman. 2. The woman in white shirt. 3. The woman looking down at the computer.\
#                 Then given the text : the teddy bear beneath and between the two bears with red sweaters and two dollar price tags. Give your output.'
user_content = 'Rerferring text descibes one object with some states or attributes information. I hope you can break down the referring text. \
                Each output sentence should only include one state or attribute. \
                For example, given the referring text: woman in white shirt looking down at laptop computer. \
                You should output: 1. a woman. 2. The woman in white shirt. 3. The woman looking down at the computer.\
                Then given the text : {}. Give your output.'

dataset = 'refcocog'
splitBy = 'google'

refer = REFER(dataset=dataset, splitBy=splitBy)
ref_ids = refer.getRefIds()

# print(len(ref_ids))
# print (len(refer.Imgs))
# print (len(refer.imgToRefs))

ref_ids = refer.getRefIds(split='val')
print ('There are %s training referred objects.' % len(ref_ids))

one_text_ref_num = 0
multi_text_ref_num = 0

# json_dir = '/home/yzq/mnt/data/code/RIS/coco/refer/refcocog/QianWen_Ref_Obj'
json_dir = '/home/yzq/mnt/data/code/RIS/coco/refer/refcocog/QianWen_Ref_Sents'
if not os.path.exists(json_dir):
    os.makedirs(json_dir)

exist_ref_id_jsons = os.listdir(json_dir)

# print(exist_ref_id_jsons)
# eventlet.monkey_patch()
for k, ref_id in enumerate(ref_ids[:]):
# for k, ref_id in enumerate(ref_ids[4000:8000]):
    ref = refer.loadRefs(ref_id)[0]
    ref_sents = ref['sentences']
    
    if (str(ref_id)+'.json') in exist_ref_id_jsons:
        print('{} json file exists...'.format(str(ref_id)+'.json'))
        continue
    else:
        current_json_dict = {}
        time_s = time.time()
        
        # with eventlet.Timeout(9,False):
        for sent in ref_sents:
            # print('*'*20)
            raw_sent = sent['sent']
            sent_id = sent['sent_id']
            
            # obtain refer object
            # try:
            #     qianwen_res = call_with_messages(user_content=user_content.format(raw_sent))
            #     if len(qianwen_res['output']['choices'][0]['message']['content'].split(' '))>10:
            #         real_content = raw_sent
            #     else:
            #         real_content = qianwen_res['output']['choices'][0]['message']['content'].split("'")[1]
            # except:
            #     print(qianwen_res)
            #     real_content = raw_sent

            # obtain refer sents
            match_list = []
            # try:
            qianwen_res = call_with_messages(user_content=user_content.format(raw_sent))
            Qwen_content = qianwen_res['output']['choices'][0]['message']['content']
            match_1 = re.search("1..*.", Qwen_content)
            match_2 = re.search("2..*.", Qwen_content)
            match_3 = re.search("3..*.", Qwen_content)
            match_4 = re.search("4..*.", Qwen_content)
            match_5 = re.search("5..*.", Qwen_content)
            if match_1 is not None:
                match_list.append(match_1.group().split('.')[1])
            else:
                match_list.append(raw_sent)
            if match_2 is not None:
                match_list.append(match_2.group().split('.')[1])
            else:
                match_list.append(raw_sent)
            if match_3 is not None:
                match_list.append(match_3.group().split('.')[1])
            else:
                match_list.append(raw_sent)
            if match_4 is not None:
                match_list.append(match_4.group().split('.')[1])
            else:
                match_list.append(raw_sent)
            if match_5 is not None:
                match_list.append(match_5.group().split('.')[1])
            else:
                match_list.append(raw_sent)
            # except:
            #     continue
            #     qianwen_res = call_with_messages(user_content=user_content.format(raw_sent))
            #     print(qianwen_res)
            #     breakpoint()
        
            print(k, '-->', raw_sent, ' --- : --- ')
            print(match_list[:2])
            # for sent in match_list:
            #     print(sent)
            current_json_dict[sent_id] = match_list
        time_e = time.time()
        print('spend {}'.format(time_e-time_s))
        if len(current_json_dict.keys()) != len(ref_sents):
            print('not each sent has llm out !')
        else:
            pass
            # content = json.dumps(current_json_dict)
            # f2 = open(os.path.join(json_dir, '{}.json'.format(ref_id)), 'w')
            # f2.write(content)
            # f2.close()
            # print(' save {} done'.format(os.path.join(json_dir, '{}.json'.format(ref_id))))