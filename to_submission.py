import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
city_result=[]
citys=["A","B","C","D"]
count=[118,30,135,75]
date_list=[ 21200615,
            21200616,
            21200617,
            21200618,
            21200619,
            21200620,
            21200621,
            21200622,
            21200623,
            21200624,
            21200625,
            21200626,
            21200627,
            21200628,
            21200629,
            21200630,
            21200701,
            21200702,
            21200703,
            21200704,
            21200705,
            21200706,
            21200707,
            21200708,
            21200709,
            21200710,
            21200711,
            21200712,
            21200713,
            21200714]
date={i:j for i,j in zip(range(46,76),date_list)}
def to_int(x):
    return int(x)
def to_date(x):
    return date[x]
for i,j in zip(citys,count):
    city_result.append( pd.read_csv("predict_bei_{}.csv".format(i),skiprows=range(j),names=["区域","天数","新增感染人数"]))
for city,result in zip(citys,city_result):
    result["城市"]=city
    city_result[citys.index(city)]=result.iloc[:,[3,0,1,2]]
for i in range(len(city_result)):
    city_result[i].iloc[:,1:]=city_result[i].iloc[:,1:].applymap(to_int)
    city_result[i].iloc[:,2:3]=city_result[i].iloc[:,2:3].applymap(to_date)
for i in city_result:
    print(i)
final=pd.concat(city_result,ignore_index=True)
print(final)


