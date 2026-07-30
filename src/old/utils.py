import torch
from tensorboardX import SummaryWriter
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import networkx as nx
import random

plt.rc('font',family='Times New Roman') 
plt.rcParams['axes.unicode_minus'] = False
from pylab import mpl
mpl.rcParams['font.size'] = 18

def make_dir(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)

def set_seed(seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False

def train(epoch,model,train_loader,optimizer,criterion,device,adj=None, writer=None):
    model.train()
    running_loss = 0.0
    acc = 0.0
    init_signal = torch.zeros((1, 12, 65), device=device)
    confusion_matrix = torch.zeros(2, 2)
    # writer.add_graph(model, init_signal)
    for data, label in tqdm(train_loader):
        label=label.to(device)
        # print(data.shape)
        
        if adj is not None:
            adj=adj.to(device)
            data=data.to(device)
            outputs=model(data,adj)
        else:
            data=data.to(device)
            outputs=model(data)

        # print(outputs.size(), label.size())
        print(outputs, label)
        loss = criterion(outputs, label)
        loss = loss.requires_grad_()
        loss=loss.mean()
        # writer.add_scalar(tag="loss/train", scalar_value=loss,
        #                   global_step=epoch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss+=loss.item()
        predict_y = torch.max(outputs, dim=1)[1]
        label_y = torch.argmax(label)

        # 🌟 解决方案：将索引张量移至 CPU
        predict_y_cpu = predict_y.cpu()
        label_y_cpu = label_y.cpu()
        confusion_matrix[label_y_cpu.long(), predict_y_cpu.long()] += 1
        if predict_y_cpu[0] == label_y_cpu: 
            acc += 1

        # confusion_matrix[label_y.long(), predict_y.long()] += 1
        # if predict_y[0] == label_y:
        #     acc += 1

    train_acc = acc / len(train_loader)
    confusion_matrix = confusion_matrix.detach().cpu().numpy()
    confusion_matrix = np.rint(100 * confusion_matrix / confusion_matrix.sum(axis=1)[:, np.newaxis])

    mean_loss = running_loss/len(train_loader)
    print("train epoch[{}] loss:{:.3f}".format(epoch+1, mean_loss))
    return mean_loss, train_acc


def test(epoch,model,test_loader,criterion,n_class,device,adj=None):
    t_start=time.perf_counter()
    model.eval()
    acc=0.0
    running_loss = 0.0
    confusion_matrix = torch.zeros(n_class, n_class)
    with torch.no_grad():
        for data, label in tqdm(test_loader):
            label=label.to(device)
            if adj is not None:
                adj=adj.to(device)
                data=data.to(device)
                outputs=model(data,adj)
            else:
                data=data.to(device)
                outputs=model(data)
            
            loss = criterion(outputs, label)
            loss=loss.mean()
            running_loss+=loss.item()
        
            predict_y=torch.max(outputs,dim=1)[1]
            label_y=torch.argmax(label)

            # 🌟 解决方案：将索引张量移至 CPU
            predict_y_cpu = predict_y.cpu()
            label_y_cpu = label_y.cpu()
            
            confusion_matrix[label_y_cpu.long(), predict_y_cpu.long()] += 1
            if predict_y_cpu[0] == label_y_cpu:
                acc += 1
            
            # confusion_matrix[label_y.long(), predict_y.long()] += 1
            # if predict_y[0] == label_y:
            #     acc += 1
    t_end=time.perf_counter()
    t_mean=(t_end-t_start)/len(test_loader)

    val_acc=acc/len(test_loader)
    confusion_matrix=confusion_matrix.detach().cpu().numpy()
    confusion_matrix=np.rint(100*confusion_matrix/confusion_matrix.sum(axis=1)[:, np.newaxis])

    print("test epoch[{}] loss:{:.3f}".format(epoch+1,running_loss/len(test_loader)))
    
    return confusion_matrix,val_acc,t_mean,running_loss


def v_confusion_matrix(cm,class_list,title=None,save_path=None):
    cm=pd.DataFrame(cm,index=class_list,columns=class_list)

    plt.figure(figsize=(6,6))
    sns_plot=sns.heatmap(cm,annot=True,linewidth=0.5,fmt=".4g",cmap="binary",cbar=False)#cmap:'Reds/Blues','binary',YlGnBu','RdBu_r'
    sns_plot.tick_params(labelsize=16,direction='out')
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    if title is not None:
        plt.title(title)
    # if save_path is not None:
    #     plt.savefig(save_path,bbox_inches="tight",dpi=300)

def select_channel(n,items):
    N = len(items)
    set_all=[]
    for i in range(2**N):
        combo = []  
        for j in range(N):   
            if(i >> j ) % 2 == 1:  
                combo.append(items[j])
        if len(combo) == n:
            set_all.append(combo)
    return set_all

def get_adjmatrix(args):

    # an adjacency matrix for the given physiological signals with medical knowledge of cardiopulmonary exercise tests created by chat-gpt
    # 该邻接矩阵包括以下关系:
    # METS、VO2、VO2 / kg、VCO2、RER都与运动时的能量代谢有关，因此相互联系。
    # 心率与心排血量有关，因此与METS、VO2和VO2 / kg有关。
    # VE、VE / VO2、VE / VCO2与运动时的肺通气有关，因此相互联系。
    # RR、VTex和VTin与呼吸力学有关，因此相互联系。

    adj = np.array(
        [
            [0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0],
            [1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1],
            [1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
            [1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
            [1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0],
            [0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1],
            [0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
        ]
    )
    # adj = np.ones((12, 12))
    adj = preprocess_adj(adj)
    return adj


#
# def get_adjmatrix(args):
#     sensor_loc_list=np.array([3,11,2,12,10,9,15,16,8,5,7,1,4,6,13,14])-1#minus 1:crresponding to index
#     group_list=[sensor_loc_list[:6],sensor_loc_list[6:12],sensor_loc_list[12:]]
#
#     graph_dict={}
#     for group in  group_list:
#         for i in group:
#             if i in args.channels:
#                 connect_node=[]
#                 for j in group:
#                     if j!=i and j in args.channels:#no circle
#                         connect_node.append(j)
#                 graph_dict[i]=connect_node
#     graph_dict=dict(sorted(graph_dict.items(),key=lambda item:item[0]))#sort the graph_dict by key
#     # print(graph_dict)
#
#     adj=nx.adjacency_matrix(nx.from_dict_of_lists(graph_dict)).toarray()
#     adj=preprocess_adj(adj)
#     # print(adj)
#
#     return adj

def preprocess_adj(adj):
    '''
    Pre-process adjacency matrix
    :param A: adjacency matrix
    :return:
    '''
    I = np.eye(adj.shape[0])
    A_hat = adj + I # add self-loops
    D_hat_diag = np.sum(A_hat, axis=1)
    D_hat_diag_inv_sqrt = np.power(D_hat_diag, -0.5)
    D_hat_diag_inv_sqrt[np.isinf(D_hat_diag_inv_sqrt)] = 0.
    D_hat_inv_sqrt = np.diag(D_hat_diag_inv_sqrt)
    A=np.dot(np.dot(D_hat_inv_sqrt, A_hat), D_hat_inv_sqrt)
    return torch.from_numpy(A).float()
    
def sparse_dropout(x, rate, noise_shape):
    """
    :param x:
    :param rate:
    :param noise_shape: int scalar
    :return:
    """
    random_tensor = 1 - rate
    random_tensor += torch.rand(noise_shape).to(x.device)
    dropout_mask = torch.floor(random_tensor).byte()
    dropout_mask=dropout_mask.bool()
    i = x._indices() 
    v = x._values() 

    i = i[:, dropout_mask]
    v = v[dropout_mask]

    out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
    out = out * (1./ (1-rate))

    return out


def calculate_metrics(confusion_matrix):
    """
    计算详细的评估指标
    :param confusion_matrix: 混淆矩阵 (numpy array)
    :return: 字典包含各项指标
    """
    # 假设是二分类问题，混淆矩阵格式为:
    # [[TN, FP],
    #  [FN, TP]]
    
    TN = confusion_matrix[0, 0]
    FP = confusion_matrix[0, 1]
    FN = confusion_matrix[1, 0]
    TP = confusion_matrix[1, 1]
    
    # 准确率 (Accuracy)
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
    
    # 灵敏度 (Sensitivity/Recall/True Positive Rate)
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
    
    # 特异性 (Specificity/True Negative Rate)
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    
    # 精确率 (Precision/Positive Predictive Value)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    
    # F1分数
    f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
    
    metrics = {
        'accuracy': accuracy,
        'sensitivity': sensitivity,  # 灵敏度
        'specificity': specificity,  # 特异性
        'precision': precision,
        'f1_score': f1_score,
        'TP': TP,
        'TN': TN,
        'FP': FP,
        'FN': FN
    }
    
    return metrics


def print_metrics(metrics, fold=None):
    """
    打印评估指标
    :param metrics: 指标字典
    :param fold: 折数（可选）
    """
    if fold is not None:
        print(f"\n{'='*80}")
        print(f"第 {fold} 折 - 评估指标")
        print(f"{'='*80}")
    else:
        print(f"\n{'='*80}")
        print(f"评估指标")
        print(f"{'='*80}")
    
    print(f"准确率 (Accuracy):    {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"灵敏度 (Sensitivity): {metrics['sensitivity']:.4f} ({metrics['sensitivity']*100:.2f}%)")
    print(f"特异性 (Specificity): {metrics['specificity']:.4f} ({metrics['specificity']*100:.2f}%)")
    print(f"精确率 (Precision):   {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
    print(f"F1分数 (F1-Score):    {metrics['f1_score']:.4f} ({metrics['f1_score']*100:.2f}%)")
    print(f"\n混淆矩阵详情:")
    print(f"  真阴性 (TN): {metrics['TN']:.0f}  |  假阳性 (FP): {metrics['FP']:.0f}")
    print(f"  假阴性 (FN): {metrics['FN']:.0f}  |  真阳性 (TP): {metrics['TP']:.0f}")
    print(f"{'='*80}\n")


def calculate_robustness(all_metrics):
    """
    计算心肺整体功能评测算法的鲁棒性测试
    基于F1分数的变化来评估模型的稳定性
    :param all_metrics: 所有折的指标列表
    :return: 鲁棒性指标字典
    """
    f1_scores = [m['f1_score'] for m in all_metrics]
    accuracies = [m['accuracy'] for m in all_metrics]
    sensitivities = [m['sensitivity'] for m in all_metrics]
    specificities = [m['specificity'] for m in all_metrics]
    
    robustness = {
        # F1分数统计
        'f1_mean': np.mean(f1_scores),
        'f1_std': np.std(f1_scores),
        'f1_min': np.min(f1_scores),
        'f1_max': np.max(f1_scores),
        'f1_cv': np.std(f1_scores) / np.mean(f1_scores) if np.mean(f1_scores) > 0 else 0,  # 变异系数
        
        # 准确率统计
        'acc_mean': np.mean(accuracies),
        'acc_std': np.std(accuracies),
        
        # 灵敏度统计
        'sens_mean': np.mean(sensitivities),
        'sens_std': np.std(sensitivities),
        
        # 特异性统计
        'spec_mean': np.mean(specificities),
        'spec_std': np.std(specificities),
        
        # 鲁棒性评分 (1 - CV，越接近1越稳定)
        'robustness_score': 1 - (np.std(f1_scores) / np.mean(f1_scores) if np.mean(f1_scores) > 0 else 1)
    }
    
    return robustness


def print_robustness_report(robustness):
    """
    打印鲁棒性测试报告
    :param robustness: 鲁棒性指标字典
    """
    print(f"\n{'#'*80}")
    print(f"{'心肺整体功能评测算法鲁棒性测试报告':^76}")
    print(f"{'#'*80}\n")
    
    print(f"{'='*80}")
    print(f"【F1分数分析】")
    print(f"{'='*80}")
    print(f"平均F1分数:         {robustness['f1_mean']:.4f} ± {robustness['f1_std']:.4f}")
    print(f"F1分数范围:         [{robustness['f1_min']:.4f}, {robustness['f1_max']:.4f}]")
    print(f"F1变异系数 (CV):    {robustness['f1_cv']:.4f} ({robustness['f1_cv']*100:.2f}%)")
    
    print(f"\n{'='*80}")
    print(f"【其他指标统计】")
    print(f"{'='*80}")
    print(f"平均准确率:         {robustness['acc_mean']:.4f} ± {robustness['acc_std']:.4f} ({robustness['acc_mean']*100:.2f}%)")
    print(f"平均灵敏度:         {robustness['sens_mean']:.4f} ± {robustness['sens_std']:.4f} ({robustness['sens_mean']*100:.2f}%)")
    print(f"平均特异性:         {robustness['spec_mean']:.4f} ± {robustness['spec_std']:.4f} ({robustness['spec_mean']*100:.2f}%)")
    
    print(f"\n{'='*80}")
    print(f"【鲁棒性综合评估】")
    print(f"{'='*80}")
    print(f"鲁棒性评分:         {robustness['robustness_score']:.4f}")
    
    # 评级系统
    if robustness['robustness_score'] >= 0.95:
        grade = "优秀 (Excellent)"
        desc = "模型在不同数据集上表现非常稳定，鲁棒性极佳"
    elif robustness['robustness_score'] >= 0.90:
        grade = "良好 (Good)"
        desc = "模型表现稳定，具有良好的鲁棒性"
    elif robustness['robustness_score'] >= 0.85:
        grade = "中等 (Fair)"
        desc = "模型表现尚可，但在不同数据集上存在一定波动"
    else:
        grade = "较差 (Poor)"
        desc = "模型表现不稳定，鲁棒性需要改进"
    
    print(f"鲁棒性等级:         {grade}")
    print(f"评估说明:           {desc}")
    
    print(f"\n{'='*80}")
    print(f"【变异系数解读】")
    print(f"{'='*80}")
    print(f"变异系数 (CV) = 标准差 / 平均值")
    print(f"  • CV < 0.10:  模型非常稳定")
    print(f"  • 0.10 ≤ CV < 0.20:  模型较为稳定")
    print(f"  • 0.20 ≤ CV < 0.30:  模型稳定性一般")
    print(f"  • CV ≥ 0.30:  模型不稳定，需要优化")
    
    if robustness['f1_cv'] < 0.10:
        cv_comment = "非常稳定"
    elif robustness['f1_cv'] < 0.20:
        cv_comment = "较为稳定"
    elif robustness['f1_cv'] < 0.30:
        cv_comment = "稳定性一般"
    else:
        cv_comment = "不稳定"
    
    print(f"\n当前模型CV = {robustness['f1_cv']:.4f}，评价: {cv_comment}")
    
    print(f"\n{'#'*80}\n")

