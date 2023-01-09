import json
import numpy as np
import random
import torch

fitdict = {'Small':1,'True to Size':2,'Large':3}
sizedic={'':5, 'P-SR':3.5, '8':4.5, 'M-L':5.5, 'SP':3.5, '0L':1.5, '52L':9, '36':4.5, 'ML':5.5, '40R':5.5, '20WR':9, '38L':5.5, '26R':5, '8L':5, '2P':2.5, 'XLR':7, '10L':5.5, '18R':7, '48R':9, 'XSP':2.5, '2XS':2, 
'52R':3, '3XR':9, '2XL':8, '4L':10, '8P':4.5, '54R':4, '30R':1, 'XL':7, '14-16':6.5, '3XL':9, 'SR':4, '10':5, '18WR':5, '4XS':0.5, '2X':8, '31':1, 'XLL':7.5, '31R':1, '2XSR':3, '52':9, '32':2, '6L':4, '50':9, '24':5, 
'16WR':5, '20W':5, '22-24':5, '33R':2, '44L':7.5, 'S':5, '22W':5, '18L':7.5, '20':8, '42L':6.5, '16R':7, '1X':7, '14L':6.5, '0':1, '14P':6, '2L':8, '48':9, 'XSL':3.5, '16P':6.5, '4':4, 'None':5, '27':5, '42':6, '44':7, 
'28R':6, '5':4, '3XS':1, '36R':6, '18/20':8, '18-20':8, '6':4, '46L':8.5, '3X':9, 'S-M':4.5, '40L':5.5, '44R':7, '14':6, 'XXSR':2, '00':1, 'ONESIZE':2.5, '18WP':4.5, '38R':5, '6P':4, '34R':4, 'XS-S':3.5, '-1':5, '54':6, '0P':1, '0X':6, 
'XXS':2, '5R':4, '25R':7, '32R':1, '1':2.5, '22WR':7, '6R':4, '22R':8, 'L-XL':6.5, '1XR':7, '10P':5, '46':8, '1R':2.5, 'M':5, '20R':8, '18':7, '00R':1, '10R':5, '16W':4, 'MR':5, '16WL':4.5, '26':9, 'XS':3, '3R':3.5, 
'0R':1.5, 'M-LR':5.5, '4R':4, '4P':3.5, '4XL':10, 'XLP':6.5, 'XSR':3, 'NONE':5, '2':3, 'SL':4, '29R':1, '12P':5.5, '12L':6, '40':5, '30':1, '22':8, 'MP':4.5, '29':0.5, '12':6, 'XXLR':8, '25':1, '24R':1, '3':3.5, '34':4,
 '28':1, '2XR':8, '3XSR':1, '18W':7, 'LL':6, '8R':5, '14/16':6.5, '42R':6, '50R':10, '16':7, 'LR':6, 'L':6, '14R':6, 'P-S':3.5, '46R':8, '38':5, '16L':7, 'XXL':8, '27R':6, '12R':6, '2R':3}
rentfordic = {'Work':1, 'Other':2, 'Date':3, 'Everyday':4, 'Formal Affair':4, 'Party':6, 'Wedding':7, 'Vacation':8, '':9}
bustdic = {'AA':7.5,'A':10, 'B':12.5, 'C':15,'D':17.5,'D+':20,'DDD/E':20, 'DD':22.5,'F':22.5,'G':25,'H':27.5,'I':30,'J':32.5}
bodydic = { 'PETITE':1, 'STRAIGHT & NARROW':2, 'PEAR':3, 'APPLE':4,'ATHLETIC':5, 'HOURGLASS':6, 'FULL BUST':7}
def Height(s:str): 
    Foot = int(s[0])
    inch = int(s[2:s.find('"')])
    return Foot * 12 + inch - 50


def LoadData():
    proportion_of_trainings = 0.8
    num1=0
    num2=0
    num3=0
    with open(r'MLproject\data\final_nolost.json',encoding='utf-8') as f:
        data = json.load(f)
    flag = 0
    for sample in data:
        fit = fitdict[sample['fit']]
        if fit==1:
            num1+=1
        if fit==2:
            num2+=1
        if fit==3:
            num3+=1
        age = int(sample['age'])
        size = sizedic[sample['size']]
        if size>=4 and size<=6 :
            a = random.randint(0,0)
            if a==1:
                continue
        op = np.array([size-fit+2,size,fit])
        rentfor = rentfordic[sample['rented_for']]
        usuallywear = int(sample['usually_wear'])
        height = Height(sample['height'])
        if(len(sample['weight'])>6):
            continue
        weight = int(sample['weight'][0:sample['weight'].find('LBS')])
        bust1 = int(sample['bust_size'][0:2])
        bust2 = bustdic[sample['bust_size'][2:]]
        bodytype = bodydic[sample['body_type']]
        
        
        if flag==0:
            Input = np.array([height,weight,bust1,bust2,bodytype])
            # Input = np.array([height,weight,bust1,bust2])
            Output = op
            flag = 1
        else:
            Input = np.vstack((Input,np.array([height,weight,bust1,bust2,bodytype])))
            # Input = np.vstack((Input,np.array([height,weight,bust1,bust2])))
            Output = np.vstack((Output,op))
    Input = (Input-Input.min(axis=0))/Input.max(axis=0)

    Data_of_Input_and_Output = list(zip(Input, Output))
    # Shuffle the data randomly
    random.shuffle(Data_of_Input_and_Output) 
    x = int(len(Data_of_Input_and_Output)*proportion_of_trainings)
    # Split the data
    training_data = Data_of_Input_and_Output[:x]
    testing_data = Data_of_Input_and_Output[x:]
    # Unzip the data into separate inputs and outputs
    Train_Input, Train_GT = zip(*training_data)
    Test_Input, Test_GT = zip(*testing_data)
    return Train_Input, Train_GT, Test_Input, Test_GT
    