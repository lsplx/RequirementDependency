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
def string_similar(s1, s2):
    return difflib.SequenceMatcher(None, s1, s2).quick_ratio()

def judgetype(entitytype,entitylist):
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
# 0表示sim,1表示constraint，2表示precondition
label_list = []
for row in df.itertuples(index=True):
    row_list = list(row)
    #需求1
    R1des = row_list[1:2]
    #需求2
    R2des = row_list[2:3]
    #similar
    similarlist = row_list[3:4]
    #constraint
    constraintlist = row_list[4:5]
    #precondition
    preconditionlist = row_list[5:6]
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
    else:
        label_list.append(4)


#抽取实体
url = "http://0.0.0.0/api/EntityRelationEx/"
headers = {'Content-Type': 'application/json'}
type_dic = {"agent":1, "constraint":2, "event":3, "input":4, "output":5,  "operation":6}
typeveclist = []
finaltypelist = []
for newpair in Repair_list:
    #1 agent, 2 operation, 3 constraint, 4 event, 5 input, 6 output
    typevec = []
    for num in range(0,27):
        typevec.append(0)
    #第一个text抽取
    datas = {"text":newpair[0]}
    data = json.dumps(datas)
    response = requests.post(url, data=data, headers=headers)
    result = response.json()
    #第二个text抽取
    datastwo = {"text":newpair[1]}
    datatwo = json.dumps(datastwo)
    responsetwo = requests.post(url, data=datatwo, headers=headers)
    resulttwo = responsetwo.json()
    for entityone in result[0]["entities"]:
        newentityone = entityone["value"].replace(" ","")
        newentityone = newentityone.replace("准确率","")
        for entitytwo in resulttwo[0]["entities"]:
            newentitytwo = entitytwo["value"].replace(" ","")
            newentitytwo = newentitytwo.replace("准确率","")
            simscore = string_similar(newentityone,newentitytwo)
            if (newentityone in newentitytwo) or (newentitytwo in newentityone) or simscore>0.8:
                if (entityone["type"] == "operation" and entitytwo["type"] != "operation") or (entitytwo["type"] == "operation" and entityone["type"] != "operation"):
                    continue
                elif entityone["type"] == "operation" and entitytwo["type"] == "operation":
                    typevec[25] = 1
                elif entityone["type"] != "meetcon" and entitytwo["type"] != "meetcon":
                    changenum = (type_dic[entityone["type"]] - 1) * 5 +  type_dic[entitytwo["type"]]
                    typevec[changenum -1] = 1
                else:
                    typevec[26] = 1
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
TP = 0
TPsim = 0
allsim = 0
TPcon = 0
allcon = 0
TPpre = 0
allpre = 0
TPother = 0
allother = 0
#总的指标
for num, each in enumerate(label_list) :
    if each == 0:
        allsim += 1
    if each == 1:
        allcon += 1
    if each == 2:
        allpre += 1
    if each == 4:
        allother += 1
    if each == result[num]:
        TP += 1
    if each == 0 and each == result[num]:
        TPsim += 1
    if each == 1 and each == result[num]:
        TPcon += 1
    if each == 2 and each == result[num]:
        TPpre += 1
    if each == 4 and each == result[num]:
        TPother += 1
#算FP
simFP = 0
conFP = 0
preFP = 0
for num, each in enumerate(result) :
    if each == 0 and label_list[num] != 0:
        simFP += 1
    if each == 1 and label_list[num] != 1:
        conFP += 1
    if each == 2 and label_list[num] != 2:
        preFP += 1
sumacc = TP/len(label_list)
simPre = TPsim/(TPsim + simFP)
conPre = TPcon/(TPcon + conFP)
prePre = TPpre/(TPpre + preFP)

simRecall = TPsim/allsim
conRecall = TPcon/allcon
preRecall = TPpre/allpre

simF1 = 2*simPre*simRecall/(simPre+simRecall)
conF1 = 2*conPre*conRecall/(conPre+conRecall)
preF1 = 2*prePre*preRecall/(prePre+preRecall)
otheracc = TPother/allother
resultlist = []
print("sumacc: " + str(sumacc))
print("simPre: " + str(simPre) + ";" + " simRecall: " + str(simRecall) + ";" + " simF1: " + str(simF1))
print("conPre: " + str(conPre) + ";" + " conRecall: " + str(conRecall) + ";" + " conF1: " + str(conF1))
print("prePre: " + str(prePre) + ";" + " preRecall: " + str(preRecall) + ";" + " preF1: " + str(preF1))
# print("conacc: " + str(conacc))
# print("preacc: " + str(preacc))
print("otheracc: " + str(otheracc))
for num, each in enumerate(result) :
    templist = []
    templist.append(each)
    templist.append(label_list[num])
    resultlist.append(templist)

df = DataFrame(resultlist,columns=["result","groundtruth"])
df.to_excel('D:/result.xlsx', sheet_name='Sheet1', index=False)
