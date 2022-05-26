import numpy as np
import torch.nn as nn 
from torch.autograd import Variable
import torch  

# discretised ODE for phi & Pi
def eta(s, eta_ini, eta_fin, N_layer):
    eta = eta_ini + (eta_fin - eta_ini)*s/N_layer
    return eta
##### 

class MetricNet(nn.Module):
    
    ''' class of model to be trained '''
    def __init__(self, Number_of_layers=None, z_ini=None, z_fin=None, del_z=None,mu=None):
        super(MetricNet, self).__init__()
        # trained parameters
        Bs = []
        Ds=[]
        for layer_index in range(Number_of_layers+1):
            Bs.append(nn.Linear(1, 1, bias=False))
        for layer_index in range(Number_of_layers+1):
            Ds.append(nn.Linear(1, 1, bias=False))
        self.Bs = nn.ModuleList(Bs)
        self.Ds = nn.ModuleList(Ds)
        # fixed parameters
        self.one = Variable(torch.ones(1)) # it would be better to use torch.nn.parameter.
        self.N_layers = Number_of_layers
        self.z_ini = z_ini
        self.z_fin = z_fin
        self.del_z = del_z
        self.mu=mu

    def penalty(self,coef_list):##only f
        
        penalty=0
        if coef_list==None:
            coefs = torch.ones(self.N_layers)
        else:
            n_coef_list = np.array(coef_list, dtype=np.float32)
            coefs = torch.from_numpy(n_coef_list)
        for i in range(self.N_layers+1):
            B = self.Bs[i]##or socalled f
            
            if i==0:
                bs = B(self.one)
                penalty=penalty+(bs-1)*(bs-1)
            else:
                # smoothing penalty
                penalty = penalty + coefs[i-1]*(B(self.one) - bs)**2
                bs = B(self.one)
        penalty=penalty#+ 10**(-2)*(bs-0)*(bs-0)#+ 10**(-4)*(bs-0.23456)*(bs-0.23456)##boundary condition
        return penalty
    def penalty_2(self,coef_list):##f and f'
        
        penalty=0
        if coef_list==None:
            coefs = torch.ones(self.N_layers)
        else:
            n_coef_list = np.array(coef_list, dtype=np.float32)
            coefs = torch.from_numpy(n_coef_list)
        for i in range(self.N_layers+1):
            B = self.Bs[i]##or socalled f
            D = self.Ds[i]
            if i==0:
                bs = B(self.one)
                ds = D(self.one)
                penalty=penalty+(bs-1)*(bs-1)+(ds-0)*(ds-0)
            else:
                # smoothing penalty
                penalty = penalty + coefs[i-1]*(B(self.one) - bs)**2 + coefs[i-1]*(D(self.one) - ds)**2
                bs = B(self.one)
                bs = D(self.one)
        penalty=penalty#+ 10**(-2)*(bs-0)*(bs-0)#+ 10**(-4)*(bs-0.23456)*(bs-0.23456)##boundary condition
        return penalty
    
    def forward_1(self,Re_s=None, Im_s=None,omega=None, PiT=None):##f forward
        
        for j in range(0,self.N_layers):
            B2=self.Bs[j+1]
            B1=self.Bs[j]
            z=self.z_ini + j*self.del_z
            Re_s=(Re_s+(1-B2(self.one)/B1(self.one))*(Re_s + 1/(PiT*(1-z)))
                  +self.del_z*(2*omega*Im_s*Re_s + 2*omega*Im_s/(PiT*(1-z))
                               -1/(PiT*(1-z)**2)) )
            Im_s=(Im_s+(1-B2(self.one)/B1(self.one))*Im_s
                  +self.del_z*( -omega/(PiT**2*(1-z)**2)-2*omega/(PiT*(1-z))*Re_s
                               +omega*Im_s*Im_s-omega*Re_s*Re_s
                               +omega/(B1(self.one)*B1(self.one))-z**2*self.mu**2/(B1(self.one)*omega)
                               ))   
        return Re_s,Im_s 
    
    def forward_2(self,Re_s=None, Im_s=None,omega=None):##lnf(z) middle
        f2=self.Bs[1]
        f1=self.Bs[0]
        z=self.z_ini
        Re_s=(Re_s+(torch.log(f1(self.one))-torch.log(f2(self.one)))*Re_s+
              self.del_z*(2*omega*Im_s*Re_s))
        Im_s=(Im_s+(torch.log(f1(self.one))-torch.log(f2(self.one)))*Im_s
        +self.del_z*( omega*Im_s*Im_s-omega*Re_s*Re_s
                     +omega/(f1(self.one)*f1(self.one))
                          -self.mu**2*z**2/(f1(self.one)*omega)))
        for j in range(1,self.N_layers):
            f2=self.Bs[j+1]
            f1=self.Bs[j-1]
            z=self.z_ini + j*self.del_z
            Re_s=(Re_s+(torch.log(f1(self.one))-torch.log(f2(self.one)))/2*Re_s+
                  self.del_z*(2*omega*Im_s*Re_s))
            Im_s=(Im_s+(torch.log(f1(self.one))-torch.log(f2(self.one)))/2*Im_s
            +self.del_z*( omega*Im_s*Im_s-omega*Re_s*Re_s
                         +omega/(f1(self.one)*f1(self.one))
                              -self.mu**2*z**2/(f1(self.one)*omega)))
        return Re_s,Im_s

    def forward_3(self,Re_s=None, Im_s=None,omega=None):##f(z) forward
        
        for j in range(self.N_layers):
            f2=self.Bs[j+1]
            f1=self.Bs[j]
            z=self.z_ini + j*self.del_z
            Re_s=(Re_s+(1-f2(self.one)/f1(self.one))*Re_s+
                  self.del_z*(2*omega*Im_s*Re_s))
            
            Im_s=(Im_s+(1-f2(self.one)/f1(self.one))*Im_s
            +self.del_z*( omega*Im_s*Im_s-omega*Re_s*Re_s
                         +omega/(f1(self.one)*f1(self.one))
                              -self.mu**2*z**2/(f1(self.one)*omega)))
        return Re_s,Im_s
    
    def forward_4(self,Re_s=None, Im_s=None,omega=None):##f(z) middle
        f2=self.Bs[1]
        f1=self.Bs[0]
        z=self.z_ini
        Re_s=(Re_s+(1-f2(self.one)/f1(self.one))*Re_s+
              self.del_z*(2*omega*Im_s*Re_s))
        Im_s=(Im_s+(1-f2(self.one)/f1(self.one))*Im_s
        +self.del_z*( omega*Im_s*Im_s-omega*Re_s*Re_s
                     +omega/(f1(self.one)*f1(self.one))
                          -self.mu**2*z**2/(f1(self.one)*omega)))
        for j in range(1,self.N_layers):
            f2=self.Bs[j]
            f1=self.Bs[j-1]
            f3=self.Bs[j+1]
            z=self.z_ini + j*self.del_z
            Re_s=(Re_s+((f1(self.one)-f3(self.one))/f2(self.one))/2*Re_s+
                  self.del_z*(2*omega*Im_s*Re_s))
            Im_s=(Im_s+((f1(self.one)-f3(self.one))/f2(self.one))/2*Im_s
            +self.del_z*( omega*Im_s*Im_s-omega*Re_s*Re_s
                         +omega/(f1(self.one)*f1(self.one))
                              -self.mu**2*z**2/(f1(self.one)*omega)))
        return Re_s,Im_s
    
    def forward_5(self,Re_s=None, Im_s=None,omega=None):##f'(z) and f(z)
        for j in range(self.N_layers):
            f=self.Bs[j]
            fp=self.Ds[j]
            z=self.z_ini + j*self.del_z
            Re_s=(Re_s+self.del_z*(-fp(self.one)/f(self.one)*Re_s +2*omega*Im_s*Re_s))
            Im_s=(Im_s+self.del_z*(-fp(self.one)/f(self.one)*Im_s+ omega*Im_s*Im_s-omega*Re_s*Re_s
                                   +omega/(f(self.one)*f(self.one))
                              -self.mu**2*z**2/(f(self.one)*omega)))
        return Re_s,Im_s

    def forward_6(self,Re_s=None, Im_s=None,omega=None):##f and f'/f=D
        for j in range(1,self.N_layers+1):
            f=self.Bs[j]
            DoverB=self.Ds[j]
            z=self.z_ini + j*self.del_z
            Re_s=(Re_s+self.del_z*(-10*DoverB(self.one)*Re_s +2*omega*Im_s*Re_s))
            Im_s=(Im_s+self.del_z*(-10*DoverB(self.one)*Im_s+ omega*Im_s*Im_s-omega*Re_s*Re_s
                                   +omega/(f(self.one)*f(self.one))
                              -self.mu**2*z**2/(f(self.one)*omega)))
            '''
            Re_s=Re_s+self.del_z*(B(Re_s)/z**2 - 2*Re_s/z + 2*omega*Im_s*Re_s-E-D(self.one)/z**4)
            Im_s=Im_s+self.del_z*(B(Im_s)/z**2 - 2*Im_s/z + omega*Im_s*Im_s-omega*Re_s*Re_s)
            '''
        return Re_s,Im_s
    def loss(self, Re_i=None, Im_i=None, Re_f=None,Im_f=None,omega=None, reg_coef_list=None,PiT=None):
        
        criterion=torch.nn.L1Loss()
        #criterion = torch.nn.MSELoss(reduction='sum')
        model_outputRe,  model_outputIm= self.forward_1(Re_s=Re_i, Im_s=Im_i,omega=omega,PiT=PiT) 
        #print('model_outputRe:',model_outputRe[0][0])
        loss = criterion(model_outputRe, Re_f)+criterion(model_outputIm, Im_f)
        loss = loss + self.penalty(reg_coef_list)
        return loss
    
    def loss_2(self, Re_i=None, Im_i=None, Re_f=None,Im_f=None,omega=None, reg_coef_list=None,PiT=None):
        
        criterion = torch.nn.MSELoss(reduction='sum')
        model_outputRe,  model_outputIm= self.forward_1(Re_s=Re_i, Im_s=Im_i,omega=omega,PiT=PiT) 
        #print('model_outputRe:',model_outputRe[0][0])
        loss = criterion(model_outputRe, Re_f)+criterion(model_outputIm, Im_f)
        loss = loss
        return loss
        
    def loss_3(self, Re_i=None, Im_i=None, Re_f=None,Im_f=None,omega=None,PiT=None):
        criterion = torch.nn.L1Loss()
        model_outputRe,  model_outputIm= self.forward_1(Re_s=Re_i, Im_s=Im_i,omega=omega,PiT=PiT) 
        #print('model_outputRe:',model_outputRe[0][0])
        loss = criterion(model_outputRe, Re_f)+criterion(model_outputIm, Im_f)
        loss = loss
        return loss

#############################
    
def init_weights(Model, z_ini, z_fin, del_z):
    
    ''' for initializations of parameters '''
    # metric (H[layer])
    for i in range(len(Model.Bs)):
        wb=np.random.uniform(0,2)
        wb = round(wb, 20)
        Model.Bs[i].weight.data.fill_(wb) # initialization excuted 
        #wd=np.random.uniform(-1,0)
        #wd = round(wd, 20)
        #Model.Ds[i].weight.data.fill_(wd) # initialization excuted 
    
def init_weights_2(Model, z_ini, z_fin, del_z,mu):
    
    ''' for initializations of parameters '''
    # metric (H[layer])
    for i in range(len(Model.Bs)):
        z=z_ini + i*del_z
        wb=1-z**3-mu**2*z**3/4+mu**2*z**4/4
        wb = round(wb, 20)
        Model.Bs[i].weight.data.fill_(wb) # initialization excuted 
        
def init_weights_3(Model, z_ini, z_fin, del_z,mu):
    
    ''' for initializations of parameters '''
    # metric (H[layer])
    for i in range(len(Model.Bs)):
        z=z_ini + i*del_z
        wb=1-z**3-mu**2*z**3/4+mu**2*z**4/4
        wd=-3*z**2 - 3/4*mu**2*z**2+mu**2*z**3
        wb = round(wb, 20)
        wd = round(wd, 20)
        Model.Bs[i].weight.data.fill_(wb) # initialization excuted 
        Model.Ds[i].weight.data.fill_(wd)
def init_weights_4(Model, z_ini, z_fin, del_z,mu):

    for i in range(len(Model.Bs)):
        z=z_ini + i*del_z
        wb=1-z**3-mu**2*z**3/4+mu**2*z**4/4
        wd=(-3*z**2 - 3/4*mu**2*z**2+mu**2*z**3)/(1-z**3-mu**2*z**3/4+mu**2*z**4/4)/10
        wb = round(wb, 20)
        wd = round(wd, 20)
        Model.Bs[i].weight.data.fill_(wb) # initialization excuted 
        Model.Ds[i].weight.data.fill_(wd)