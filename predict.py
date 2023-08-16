from sklearn import datasets
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
import os
import numpy as np
import pandas as pd
from pandas import DataFrame
import json
import requests
import joblib
import difflib
from sklearn.tree import DecisionTreeClassifier 
from matplotlib import pyplot as plt
from sklearn import tree
from pandas import DataFrame
import openai


# 定义用于与模型交互的API函数
def information_extraction_with_gpt(prompt):
    response = openai.Completion.create(
        engine='gpt-3.5-turbo', # 或者使用'gpt-3.5-turbo'
        prompt=prompt,
        max_tokens=200,  # 设定生成的最大长度
        temperature=0.7,  # 控制生成文本的随机程度
        n=1,  # 生成的候选回复数量
        stop=None,  # 可以使用特定的字符串停止回复生成
        echo=True  # 回显输入的prompt
    )

    # 提取生成的回复
    reply = response.choices[0].text.strip()
    return reply


def string_similar(s1, s2):
    # 获取词向量
    vectors = openai.Embed(list([word1, word2]))
    # 提取词向量
    vector1 = vectors['embeddings'][0]
    vector2 = vectors['embeddings'][1]
    # 计算余弦相似度
    similarity = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))

    return similarity


def judgetype(entitytype, entitylist):
    flag = 0
    for entity in entitylist:
        if entitytype == entity["type"]:
            flag = 1
        else:
            continue
    if flag == 1:
        return True
    else:
        return False

object_path = "D:/Requirementdata.xlsx"
try:
    pd_sheets = pd.ExcelFile(object_path)
except Exception as e:
    print("读取 {} 文件失败".format(object_path), e)
df = pd.read_excel(pd_sheets, "Sheet1", header=[0])
requirement_des = []
num = 0
Redic = {}
Repair_list = []
# 0-similar,1-Refines，2-Requires, 3-Change, 4-Conflict
label_list = []
for row in df.itertuples(index=True):
    row_list = list(row)
    #需求1
    R1des = row_list[1:2]
    #需求2
    R2des = row_list[2:3]
    #similar
    similarlist = row_list[3:4]
    #refines
    constraintlist = row_list[4:5]
    #requires
    preconditionlist = row_list[5:6]
    # change
    change_list = row_list[6:7]
    # conflict 
    conflict_list = row_list[7:8]

    templist = []
    templist.append(R1des[0])
    templist.append(R2des[0])
    Repair_list.append(templist)
    if similarlist[0] == 1:
        label_list.append(0)
    elif constraintlist[0] == 1:
        label_list.append(1)
    elif preconditionlist[0] == 1:
        label_list.append(2)
    elif change_list[0] == 1:
        label_list.append(3)
    elif conflict_list[0] == 1:
        label_list.append(4)
    else:
        label_list.append(6)

#抽取实体
url = "http://0.0.0.0/api/EntityRelationEx/"
headers = {'Content-Type': 'application/json'}
typeveclist = []
finaltypelist = []
for newpair in Repair_list:
    typevec = []
    for num in range(0,27):
        typevec.append(0)
    #第一个text抽取
    response = information_extraction_with_gpt(newpair[0])
    result = response.json()
    #第二个text抽取
    response = information_extraction_with_gpt(newpair[1])
    resulttwo = responsetwo.json()
    if (judgetype("meetcon",result[0]["entities"]) == True and judgetype("meetcon",resulttwo[0]["entities"]) == False) or (judgetype("meetcon",result[0]["entities"]) == False and judgetype("meetcon",resulttwo[0]["entities"]) == True):
        typevec[26] = 2
    typeveclist.append(typevec)
    arraytypeveclist = np.array(typevec)
    finaltypelist.append(arraytypeveclist)

X = finaltypelist
y = label_list

# Fit the classifier with default hyper-parameters
clf = joblib.load("D:/DTmodel.pkl")
result = clf.predict(X)

df = DataFrame(resultlist, columns=["result","groundtruth"])
df.to_excel('D:/result.xlsx', sheet_name='Sheet1', index=False)
