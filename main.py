from Network import DNN
import sys
import torch
sys.path.append('MLproject\\data\\dataset.py')
import data.dataset as Load

Train_Input, Train_GT, Test_Input, Test_GT=Load.LoadData()
#转成tensor格式
Train_GT = torch.tensor(list(Train_GT))
Train_Input = torch.tensor(list(Train_Input))
Test_Input = torch.tensor(list(Test_Input))
Test_GT = torch.tensor(list(Test_GT))

#这里取size-fit作为训练标签，实际上Train_GT和Test_GT是三元的
Train_GT = torch.unsqueeze(Train_GT[:,0], dim=0)
Test_GT = torch.unsqueeze(Test_GT[:,0], dim=0)

accuracy=DNN(Train_Input.T, Train_GT, Test_Input.T,Test_GT,[5,128,64,32,1],0.001,2000)
print(accuracy)