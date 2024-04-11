""" LLMs for DeepShovel: 结构化数据抽取 - 三元组抽取（输入是预处理好的段落文件） """
# 分步骤提示大模型抽取三元组：1. 抽取实体；2. 实体关联到本体；3. 抽取关系
# 需要提供 <本体> 类型定义和 <关系> 结构规范
# 以mgkg的段落文本为例
import os
import openai
import requests
import json
import logging
import time
import random
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


load_dotenv()
proxies = {
    'http': 'http://10.10.1.3:10000',
    'https': 'http://10.10.1.3:10000'
}
sjtu_temp = 'sk-q204wot6pxjP0EwwE9B63f3b3d1a462091Ac9bC7B9Ee2703'


# 调用API并使用重试机制处理rate limit error和其他异常
def get_completion(prompt):
    messages = [
        {"role": "system", "content": "你是城市规划领域的专家，帮我从相关文献中抽取实体和关系，以辅助构建城市规划的领域知识图谱"},
        {"role": "user", "content": prompt}
    ]
    response = requests.post(
        url='https://openai.acemap.cn/v1/chat/completions',
        headers={'Authorization': f'{sjtu_temp}'},
        json={
            'model': 'gpt-3.5-turbo',
            'messages': messages
        },
        # proxies=proxies,  # 设置代理
        verify=False
    )

    return response.json()['choices'][0]['message']['content']


# 从schema文件中获取本体和关系
def get_schema():
    schema_path = "../data/beiguiyuan/标准数据标注-9个标准/关系类型总结-9个标准/9个标准的关系总结-去重.json"
    with open(schema_path, "r") as file:
        schema = json.load(file)

    entity_ls, relation_ls = [], []
    for item in schema:
        if item['subject_type'] != "Number" and item['subject_type'] != "number":
            entity_ls.append(item['subject_type'])
        if item['object_type'] != "Number" and item['object_type'] != "number":
            entity_ls.append(item['object_type'])
        relation_ls.append(item['predicate'])

    # entity_ls和relation_ls去重
    entity_ls = list(set(entity_ls))
    relation_ls = list(set(relation_ls))
    entity_ls.append('数字')

    print(f"# entity count: {len(entity_ls)}, relation count: {len(relation_ls)}")
    print("# entity_label_ls:", entity_ls)
    print("# relation_ls:", relation_ls)
    return entity_ls, relation_ls


def triple_extraction(paragraph: str, entity_labels: list, schema_relations: list):
    # 1. 抽取实体
    prompt1 = f"""
我将给你一段文字。请尽可能多的从中提取命名实体。你的回答应该只包含一个列表，不包含其他内容。

---
以下是一个例子：

段落：
商务办公建筑：供非行政办公单位的办公使用的建筑，也被称为写字楼（包括SOHU办公楼）。

你的回答：
[
    "商务办公建筑",
    "写字楼",
    "SOHU办公楼"
]

---
以下是你要处理的段落：
{paragraph}
"""

    entity_list = get_completion(prompt1)
    
    # 2. 实体关联到本体
    prompt2 = f"""
这是你刚刚生成的实体列表：
{entity_list}

将每个实体分类到以下列表中的一个类别中。你不应该将任何实体分类到以下列表中没有的类别中。
{entity_labels}

你的回答应该是一个 JSON 字典，其中实体是键，类别是值。除了 JSON 字典之外，你的回答中不应该包含任何其他内容。

---
以下是一个例子：

段落：
商务办公建筑：供非行政办公单位的办公使用的建筑，也被称为写字楼（包括SOHU办公楼）。

实体列表：
[
    "商务办公建筑",
    "写字楼",
    "SOHU办公楼"
]

你的回答:
{{
    "商务办公建筑": "建筑类型",
    "写字楼": "名称",
    "SOHU办公楼": "建筑类型"
}}
---
"""

    entity_category_dict = get_completion(prompt2)
    
    # 3. 抽取关系
    prompt3 = f"""
以下是一个段落：
{paragraph}

以下是你刚刚生成的“实体列表”：
{entity_list}

从段落中提取尽可能多的关系。你的结果应该是一个三元组列表，除此之外不应包含任何其他内容。
每个三元组中的第一个和第三个元素应在你生成的“实体列表”中，而第二个元素应在以下“关系类别列表”中。
你不应该提取任何其第二个元素不在以下“关系类别列表”中的关系。

以下是“关系类别列表”：
{schema_relations}

---
以下是一个例子：

段落：
商务办公建筑：供非行政办公单位的办公使用的建筑，也被称为写字楼（包括SOHU办公楼）。

实体列表：
[
    "商务办公建筑",
    "写字楼",
    "SOHU办公楼"
]

你的回答：
[
    ["商务办公建筑", "别称", "写字楼"],
    ["商务办公建筑", "建筑子类", "SOHU办公楼"],
]
---
"""

    relation_list = get_completion(prompt3)
    
    try:
        p_entity_list = json.loads(entity_list)
        p_entity_category_dict = json.loads(entity_category_dict)
        p_relation_list = json.loads(relation_list)
        # print("# JSON load successful!")
        load_flag = True
        return {
            "paragraph": paragraph,
            "entity_list": p_entity_list,
            "entity_category_dict": p_entity_category_dict,
            "relation_list": p_relation_list
        }, load_flag
    except:
        # print("# JSON load failed!")
        load_flag = False
        return {
            "paragraph": paragraph,
            "entity_list": entity_list,
            "entity_category_dict": entity_category_dict,
            "relation_list": relation_list
        }, load_flag


entity_labels, schema_relations = get_schema()


text_ls = []
with open('../data/beiguiyuan/标准数据标注-9个标准/标注数据-9个标准/城市电力规划规范.jsonl', 'r', encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        text_ls.append(data['text'])
wrt_file = "../results/city_result/城市电力规划规范_result.json"
with open(wrt_file, "w", newline='\n') as wrt_single_file:
    for text in tqdm(text_ls, total=len(text_ls), desc='Processing paragraphs'):
        result, load_flag = triple_extraction(text, entity_labels, schema_relations)
        json.dump(result, wrt_single_file, ensure_ascii=False)
        wrt_single_file.write('\n')  # 添加换行符


text_ls = []
with open('../data/beiguiyuan/标准数据标注-9个标准/标注数据-9个标准/城市道路交通设计规范.jsonl', 'r', encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        text_ls.append(data['text'])
wrt_file = "../results/city_result/城市道路交通设计规范_result.json"
with open(wrt_file, "w", newline='\n') as wrt_single_file:
    for text in tqdm(text_ls, total=len(text_ls), desc='Processing paragraphs'):
        result, load_flag = triple_extraction(text, entity_labels, schema_relations)
        json.dump(result, wrt_single_file, ensure_ascii=False)
        wrt_single_file.write('\n')  # 添加换行符


text_ls = []
with open('../data/beiguiyuan/标准数据标注-9个标准/标注数据-9个标准/城市道路工程技术规范.jsonl', 'r', encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        text_ls.append(data['text'])

wrt_file = "../results/city_result/城市道路工程技术规范_result.json"
with open(wrt_file, "w", newline='\n') as wrt_single_file:
    for text in tqdm(text_ls, total=len(text_ls), desc='Processing paragraphs'):
        result, load_flag = triple_extraction(text, entity_labels, schema_relations)
        json.dump(result, wrt_single_file, ensure_ascii=False)
        wrt_single_file.write('\n')  # 添加换行符


# """
# paragraph_dict = {
#     "paragraph": paragraph,
#     "relations": [
#         {
#             "head": head,
#             "head_label": head_label,
#             "relation": relation,
#             "tail": tail,
#             "tail_label": tail_label
#         },
#         {
#             "head": head,
#             "head_label": head_label,
#             "relation": relation,
#             "tail": tail,
#             "tail_label": tail_label
#         },
#         ...
#     ]
# }
# """
# paragraph_file = "../data/mgkg_data/paragraph_pubmed_0720_test.txt"
# result_write_file = "../results/mgkg_result/paragraph_pubmed_0720_test_result.json"
# result_single_write_file = "../results/mgkg_result/paragraph_pubmed_0720_test_result_single.json"
# retry_write_file = "../data/mgkg_data/paragraph_pubmed_0720_test_retry.txt"

# # paragraph_file = "../data/mgkg_data/paragraph_pubmed_0720.txt"
# # result_write_file = "../results/mgkg_result/paragraph_pubmed_0720_result.json"
# # result_single_write_file = "../results/mgkg_result/paragraph_pubmed_0720_result_single.json"
# # retry_write_file = "../data/mgkg_data/paragraph_pubmed_0720_retry.txt"
# with open(paragraph_file, "r") as file:
#     paragraphs = file.readlines()

# retry_paragraphs = []   # JSON load失败的paragraph
# paragraph_dict_list = []    # 所有paragraph的三元组抽取结果
# success_cnt = 0
# fail_cnt = 0

# # 将抽取结果paragraph_dict持续性写入result_single_write_file文件
# with open(result_single_write_file, "w", newline='\n') as wrt_single_file:
#     for paragraph in tqdm(paragraphs, total=len(paragraphs), desc='Processing paragraphs'):
#         result, load_flag = triple_extraction(paragraph, entity_labels, schema_relations)

#         # 折中方案：JSON load失败，重试
#         for i in range(3):
#             if not load_flag:
#                 print("# JSON load failed! Retry...")
#                 time.sleep(5)  # Wait before retrying
#                 result, load_flag = triple_extraction(paragraph, entity_labels, schema_relations)
#             else:
#                 break

#         if load_flag:   # JSON load成功
#             success_cnt += 1
#             entities = result['entity_list']
#             entity_labels = result['entity_category_dict']
#             relations = result['relation_list']
#             relation_dict_list = []  # 每个paragraph抽取的的所有relation
#             for item in relations:
#                 if len(item) != 3:
#                     continue
#                 head = item[0]
#                 relation = item[1]
#                 tail = item[2]
#                 head_label = ""
#                 tail_label = ""
#                 # print(f'{head}, {relation}, {tail}')
#                 entity_keys = entity_labels.keys()
#                 for key in entity_keys:
#                     if key in head:
#                         head_label = entity_labels[key]
#                     if key in tail:
#                         tail_label = entity_labels[key]
#                 relation_dict = {   # paragraph中抽取的一个relation
#                     "head": head,
#                     "head_label": head_label,
#                     "relation": relation,
#                     "tail": tail,
#                     "tail_label": tail_label
#                 }
#                 relation_dict_list.append(relation_dict)
#             paragraph_dict = {
#                 "paragraph": paragraph,
#                 "relations": relation_dict_list
#             }
#             paragraph_dict_list.append(paragraph_dict)
#         else:   # JSON load失败
#             fail_cnt += 1
#             retry_paragraphs.append(paragraph)
#             paragraph_dict = {
#                 "paragraph": paragraph,
#                 "relations": "failure"
#             }
#             paragraph_dict_list.append(paragraph_dict)
#         json.dump(paragraph_dict, wrt_single_file, indent=4)
#         wrt_single_file.write('\n')  # 添加换行符
#         print("# paragraph_cnt {}, success_cnt {}, fail_cnt {}".format(len(paragraphs), success_cnt, fail_cnt))


# print("# paragraph_cnt:", len(paragraphs))
# print("# success_cnt:", success_cnt)
# print("# fail_cnt:", fail_cnt)

# # 将抽取结果paragraph_dict_list一次性写入文件
# with open(result_write_file, "w") as json_file:
#     json.dump(paragraph_dict_list, json_file, indent=4)

# # 将JSON load失败的paragraph一次性写入新文件
# with open(retry_write_file, 'w') as retry_file:
#     for item in retry_paragraphs:
#         retry_file.write("%s" % item)
