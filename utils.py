import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import torch
def plot_rewards(rewards,ma_rewards,tag="train",env='CartPole-v0',algo = "DQN",save=True,path='./'):
    sns.set()
    plt.title("average learning curve of {} for {}".format(algo,env))
    plt.xlabel('epsiodes')
    plt.plot(rewards,label='rewards')
    plt.plot(ma_rewards,label='ma rewards')
    plt.legend()
    if save:
        plt.savefig(path+"{}_rewards_curve".format(tag))
    plt.show()

def plot_loss(loss_arr,path,save = 1):
    sns.set()
    plt.title("loss curve")
    plt.xlabel('epsiodes')
    plt.plot(loss_arr['train'],label='train')
    plt.plot(loss_arr['val'],label='val')
    # plt.axhline(y=loss_arr['test'],label='test',c='g')
    plt.legend()
    if save:
        plt.savefig(os.path.join(path,"losses_curve"))
    # plt.show()
def plot_xy(y_pred,y_gnd,path,dataset,save = 1):
    sns.set()
    fig, ax = plt.subplots(1, 1)
    fig.suptitle('predict_result')
    ax.set_xlabel('y_pred')
    ax.set_ylabel('y_gnd')
    y_gnd = torch.tensor(y_gnd,device='cpu')
    y_pred = torch.tensor(y_pred,device='cpu')
    ax.scatter(y_pred,y_gnd,s=3)
    mx = max(torch.max(y_pred),torch.max(y_gnd)) + 10


    xx = [0,mx]
    yy = [0,mx]
    ax.plot(xx,yy,label='y=x',c = 'r')
    ax.legend()
    if save:
        fig.savefig(os.path.join(path , '{}_pred'.format(dataset)))
    # plt.show()



def save_loss(loss_arr,path):
    np.save(os.path.join(path,'losses.npy'),loss_arr)
    json.dump(loss_arr,open(os.path.join(path,'losses.json'),'w+'), indent = 2)
    print('results saved!')

def load_loss(path):
    loss_arr = np.load(os.path.join(path,'losses.npy'),allow_pickle=True)
    return loss_arr

def check_path(path):
    if(not os.path.exists(path)):
        os.makedirs(path)