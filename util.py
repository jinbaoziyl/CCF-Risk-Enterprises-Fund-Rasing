#encoding:utf-8
import math
import datetime
import pandas as pd

def days(str1,str2):
    date1=datetime.datetime.strptime(str1[0:10],"%Y-%m-%d")
    date2=datetime.datetime.strptime(str2[0:10],"%Y-%m-%d")
    num=(date1-date2).days
    return num

def days_v1(str1,str2):
    # print(str1, str2)
    date1=datetime.datetime.strptime(str1[0:10],"%Y/%m/%d")
    date2=datetime.datetime.strptime(str2[0:10],"%Y/%m/%d")
    num=(date1-date2).days
    return num

def months(str1,str2):
    year1=datetime.datetime.strptime(str1[0:10],"%Y-%m-%d").year
    year2=datetime.datetime.strptime(str2[0:10],"%Y-%m-%d").year
    month1=datetime.datetime.strptime(str1[0:10],"%Y-%m-%d").month
    month2=datetime.datetime.strptime(str2[0:10],"%Y-%m-%d").month
    num=(year1-year2)*12+(month1-month2)
    return num

def months_v1(str1,str2):
    year1=datetime.datetime.strptime(str1[0:10],"%Y/%m/%d").year
    year2=datetime.datetime.strptime(str2[0:10],"%Y/%m/%d").year
    month1=datetime.datetime.strptime(str1[0:10],"%Y/%m/%d").month
    month2=datetime.datetime.strptime(str2[0:10],"%Y/%m/%d").month
    num=(year1-year2)*12+(month1-month2)
    return num

def check_date(str):
    try:
        datetime.datetime.strptime(str[0:10], "%Y-%m-%d")
        return True
    except Exception as e:
        return False


