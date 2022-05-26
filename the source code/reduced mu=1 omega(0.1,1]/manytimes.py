# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 18:09:36 2022

@author: likai
"""

import numpy as np
import matplotlib.pyplot as pl
import os
from dataprocess import datacollect
#import network
import network3
import torch 
from torch.autograd import Variable 
import torch.optim as optim 
from torch.utils.data import DataLoader 
from torchvision import transforms 
from torch.utils.data.dataset import Dataset 
from moviepy.editor import ImageSequenceClip
import matplotlib.font_manager as font_manager
from matplotlib.pyplot import MultipleLocator
import time
font = font_manager.FontProperties(family='Times New Roman',
                                   weight='normal',
                                  #style='normal',
                                  size=20)
start=time.process_time()#######record the start time#########
def mse(a,b):
    return np.sum((np.array(a)-np.array(b))**2)/len(a)
def mae(a,b):
    return np.sum(np.absolute(np.array(a)-np.array(b)))/len(a)
def train(net,train_d_loader,total_epoch_penalty_1,show_epoch_each_1,regp1,regp2,count,iterition,loss_data,j,typ):
    flag=1
    epochloss=[]
    if typ==1: 
        LR = 1e-3
        optimizer= optim.RMSprop(net.parameters(),alpha=0.9,lr=LR)  
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[1000,1500],gamma = 0.9)     
    if typ==2: 
        LR = 1e-3 
        optimizer=optim.Adam(net.parameters(),lr=LR) 
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[500,1000,1250,1500,1750],gamma = 0.1)        
    net.train()
    for epoch in range(1,total_epoch_penalty_1+1):
        batches = iter(train_d_loader)
        train_loss = 0.0
        scheduler.step()
        if typ==1:
            regp1=50/(epoch/ 10)**1.5
            regularization_coeffs = []
            #train_loss_2=train_loss_1
            #print('train_loss_2:',train_loss_2)
            train_loss_1 = 0.0 
            for i in range(0, N_layer):
                regularization_coeffs.append(regp1/network3.eta(i, z_ini, z_fin, N_layer)**regp2)           
        for (Re_i1,Im_i1,Re_f1,Im_f1,omega1) in batches:
            Re_i2,Im_i2,Re_f2,Im_f2,omega2= Variable(Re_i1), Variable(Im_i1), Variable(Re_f1),Variable(Im_f1), Variable(omega1)
            # calclation of loss
            if typ==1:
                loss = net.loss(Re_i=Re_i2, Im_i=Im_i2,
                                  Re_f=Re_f2,Im_f=Im_f2,omega=omega2,reg_coef_list=regularization_coeffs,PiT=PiT)
                loss_1=net.loss_3(Re_i=Re_i2, Im_i=Im_i2,
                                  Re_f=Re_f2,Im_f=Im_f2,omega=omega2,PiT=PiT)
                train_loss_1 += loss_1.data
            
            if typ==2:
                loss = net.loss_2(Re_i=Re_i2, Im_i=Im_i2,
                                  Re_f=Re_f2,Im_f=Im_f2,omega=omega2,PiT=PiT)
            # update
            optimizer.zero_grad()
            loss.backward() 
            optimizer.step()
            train_loss += loss.data
            if torch.isnan(train_loss).any()==1 or (train_loss >torch.Tensor([1e8]))==1:
                flag=0
                break
        if flag==0:
                count=count
                break
            
        if typ==1 and epoch % show_epoch_each_1 == 1:
            f_p=[]
            for h in net.Bs:     # metric (h)
                f_p.append(h.weight.data.numpy()[0][0])              
            fig=pl.figure(figsize=(6,5))
            pl.ylim(-0.02,2.4)
            pl.xlim(-0.02,1.02)
            pl.plot(z1,B,c='#ff9900',label='True f',lw=10)
            pl.xlabel('z',fontsize=16)
            pl.ylabel('$f(z)$',fontsize=16)
            pl.plot(z1,np_f,c='#e06666',label='Initial f',lw=5,marker='o',markersize=10)
            pl.plot(z1,f_p,label='f with penalty',marker='D',lw=6,markersize=10)
            pl.legend()
            pl.xticks(fontsize=16)
            pl.yticks(fontsize=16)
            ax=pl.gca()
            y_major_locator = MultipleLocator(0.5)
            x_major_locator = MultipleLocator(0.2)
            ax.yaxis.set_major_locator(y_major_locator)
            ax.xaxis.set_major_locator(x_major_locator)
            pl.legend(fontsize=12,frameon=False)
            #pl.show()
            pl.close()
            print ("training epoch: {},   loss: {}".format(epoch, train_loss[0]))
            print ("training epoch: {},   loss: {}".format(epoch, train_loss_1))
            #print(train_loss_1)
            #iterition.append(epoch)
            #loss_data.append(np.array(train_loss).tolist())
            fig.savefig('./'+str(epoch)+str(typ)+str(j)+'.png')
            if epoch >800:
                epochloss.append(train_loss_1)
                if len(epochloss)>5:
                    if (epochloss[-1]>epochloss[-5])==1:
                        break
        if typ==2 and epoch % show_epoch_each_1 == 1:
            f_p=[]
            for h in net.Bs:     # metric (h)
                f_p.append(h.weight.data.numpy()[0][0])              
            fig=pl.figure(figsize=(6,5))
            pl.ylim(-0.02,2.4)
            pl.xlim(-0.02,1.02)
            pl.plot(z1,B,c='#ff9900',label='True f',lw=10)
            pl.xlabel('z',fontsize=16)
            pl.ylabel('$f(z)$',fontsize=16)
            pl.plot(z1,np_f,c='#e06666',label='Initial f',lw=5,marker='o',markersize=10)
            pl.plot(z1,fp,c='#24e9e9',label='f with penalty',marker='D',lw=6,markersize=10)
            pl.plot(z1,f_p,c='#674ea7',label='Finally predicted f',marker='*',lw=3,markersize=12)
            pl.legend()
            pl.xticks(fontsize=16)
            pl.yticks(fontsize=16)
            ax=pl.gca()
            y_major_locator = MultipleLocator(0.5)
            x_major_locator = MultipleLocator(0.2)
            ax.yaxis.set_major_locator(y_major_locator)
            ax.xaxis.set_major_locator(x_major_locator)
            pl.legend(fontsize=12,frameon=False)
            #pl.show()
            pl.close()
            print ("training epoch: {},   loss: {}".format(epoch, train_loss))
            #iterition.append(epoch)
            if epoch==total_epoch_penalty_1:
                fig.savefig('./'+str(epoch)+str(typ)+str(j)+'.png',dpi=600,bbox_inches='tight')
                fig.savefig('./'+str(epoch)+str(typ)+str(j)+'.pdf',format='pdf')
                loss_data.append(train_loss)
            else:
                fig.savefig('./'+str(epoch)+str(typ)+str(j)+'.png')
            
    if flag==1:
        count=count+1
    return net,count,iterition,loss_data,epoch
class Re_Im_DataSet(Dataset):
    ''' class for handling data '''
    def __init__(self, Re_i, Im_i, Re_f,Im_f,omega,transform=None):
        self.Re_i = Re_i
        self.Im_i= Im_i
        self.Re_f = Re_f
        self.Im_f = Im_f
        self.omega=omega
        self.transform = transform

    def __getitem__(self, index):
        Re_i=self.Re_i[index]
        Im_i=self.Im_i[index]
        Re_f=self.Re_f[index]
        Im_f=self.Im_f[index] 
        omega=self.omega[index]
       # if self.transform is not None:
        return Re_i,Im_i,Re_f,Im_f,omega
    def __len__(self):
        return len(self.Re_i)

####################the initial data obtaining of sigma(z=0.01)#####################
##read the data file##
filen1='AxImsubv8.dat'
filen2='AxResubv8.dat'
filen3='dAxImsubv8.dat'
filen4='dAxResubv8.dat'
axim01,omega1=datacollect(filen1)
axre01,omega2=datacollect(filen2)
daxim01,omega3=datacollect(filen3)
daxre01,omega4=datacollect(filen4)
##process data##
ax01=[]
dax01=[]

for i in range(len(omega1)):
    ax01.append(complex(axre01[i],axim01[i]))
    dax01.append(complex(daxre01[i],daxim01[i]))
sigmare01=[]###z=0.01 sigma's real part with bias
sigmare01n=[]###z=0.01 sigma's real part without bias
sigmaim01=[]
sigma01=[]

z_fin = 0.99
z_ini = 0.01
mu=1
PiT=(12-mu**2)/4
bias1=1/((3-mu**2/4)*(1-z_ini))
for i in range(len(omega1)):
    sigma=dax01[i]/(ax01[i]*1j*omega1[i])
    sigma01.append(sigma)
    sigmare01.append(sigma.real+bias1)
    sigmaim01.append(sigma.imag)
    sigmare01n.append(sigma.real)
    
##################use the network to generate the z=0.9 data###################
##data type transformation##
N=len(omega1)

a=np.array(sigmare01n).reshape((N,1))
b=np.array(sigmaim01).reshape((N,1))
c=np.array(omega1).reshape((N,1))
N_layer = 10

a=torch.Tensor(a)
b=torch.Tensor(b)
c=torch.Tensor(c)
# making network

del_z = (z_fin - z_ini)/N_layer

##difference network
Test = network3.MetricNet(Number_of_layers=N_layer, 
                         z_ini=z_ini, 
                         z_fin=z_fin,
                         del_z=del_z,
                         mu=mu)
network3.init_weights_3(Test,z_ini, z_fin,del_z,mu)
##continuum network

Re_f,Im_f=Test.forward_1(Re_s=a, Im_s=b,omega=c,PiT=PiT)


#####data preparation finished######

Re_i=np.array(a).reshape((N,1))
Im_i=np.array(b).reshape((N,1))
Re_f=np.array(Re_f.detach().numpy()).reshape((N,1))
Im_f=np.array(Im_f.detach().numpy()).reshape((N,1))
omega=np.array(c).reshape((N,1))
train_d_loader = DataLoader(Re_Im_DataSet( Re_i, Im_i, Re_f,Im_f,omega, 
                            transform=transforms.Compose([torch.from_numpy])), 
                            batch_size=200, shuffle=True)
##############################################################################################################


# For regularization terms
regularizationp1=50
regularizationp2=2
B=[]
z1=[]
for i in range(N_layer+1):
    z=z_ini+i*del_z
    B.append(1-z**3-mu**2*z**3/4+mu**2*z**4/4)
    z1.append(z_ini+i*del_z)

#total_epoch_penalty=500
show_epoch_each=10
total_epoch_penalty=[3001,2001]
typ=0
count=0
metricfp=[]
metricf=[]
maefp=[]
msefp=[]
maef=[]
msef=[]
iterition=[]
loss_data=[]
for j in range(10000):
    if count==2:
        break
   # print(j)
    Test2 = network3.MetricNet(Number_of_layers=N_layer, 
                         z_ini=z_ini, 
                         z_fin=z_fin,
                         del_z=del_z,
                         mu=mu)
    network3.init_weights(Test2,z_ini, z_fin,del_z)
    np_f=[]
    for h in Test2.Bs:     # metric (h)
        np_f.append(h.weight.data.numpy()[0][0])
    Test2,N,iterition,loss_data,nfig1=train(net=Test2,
                  train_d_loader=train_d_loader,
                  total_epoch_penalty_1=total_epoch_penalty[0],
                  show_epoch_each_1=show_epoch_each,
                  regp1=regularizationp1,
                  regp2=regularizationp2,
                  count=count,
                  iterition=iterition,
                  loss_data=loss_data,
                  j=j,
                  typ=1)
    if N==count:
        continue
    fp=[]
    for h in Test2.Bs:     # metric (h)
        metricfp.append(h.weight.data.numpy()[0][0])
        fp.append(h.weight.data.numpy()[0][0])
    Test2,N,iterition,loss_data,nfig2=train(net=Test2,
                  train_d_loader=train_d_loader,
                  total_epoch_penalty_1=total_epoch_penalty[1],
                  show_epoch_each_1=show_epoch_each,
                  regp1=regularizationp1,
                  regp2=regularizationp2,
                  count=N,
                  iterition=iterition,
                  loss_data=loss_data,
                  j=j,
                  typ=2)
    f=[]
    for h in Test2.Bs:     # metric (h)
        metricf.append(h.weight.data.numpy()[0][0])
        f.append(h.weight.data.numpy()[0][0])

    maefp.append(mae(B,fp))
    msefp.append(mse(B,fp))
    maef.append(mae(B,f))
    msef.append(mse(B,f))
    count=N
    img_names1 = ['./'+str(i*show_epoch_each+1)+str(1)+str(j)+'.png' for i in range(int(nfig1/show_epoch_each)+1)]
    img_names2 = ['./'+str(i*show_epoch_each+1)+str(2)+str(j)+'.png' for i in range(int(total_epoch_penalty[1]/show_epoch_each))]
    img_names=img_names1+img_names2
    clip = ImageSequenceClip(img_names,fps=20)
    clip.write_gif(str(j)+'.gif')
    for num in range(len(img_names)):
        os.remove(img_names[num])
    print('count:',count)
    end=time.process_time()
    print('Running time: %s Seconds'%(end-start))

####record the network parameters##########
fil= open("file_data_z=0.99_N=11.dat", "w")
print('detailed imformation data of f with penalty:',metricfp, '\n',
      "mae of f with penalty:",maefp,'\n',
      "mse of f with penalty:",msefp,'\n',
      "average mae of f with penalty:",np.average(maefp),'\n',
      "average mse of f with penalty:",np.average(msefp),'\n',
      'detailed imformation data of f:',metricf, '\n',
      "mae of f:",maef,'\n',
      "mse of f:",msef,'\n',
      "average mae of f:",np.average(maef),'\n',
      "average mse of f:",np.average(msef),'\n',
     "loss of output data:",loss_data,'\n',
      "average loss of output data:",np.average(loss_data),
      file=fil)
fil.close()

print( "average mae of f with penalty:",np.average(maefp),'\n',
      "average mse of f with penalty:",np.average(msefp),'\n',
      "average mae of f:",np.average(maef),'\n',
      "average mse of f:",np.average(msef),'\n',
      "average loss of output data:",np.average(loss_data),'\n',
      )

#end=time.process_time()
print('Running time: %s Seconds'%(end-start))

    
