"""
@project:Bearing_Fault_Diagnosis
@Author: Phantom
@Time:2024/3/5 上午11:09
@Email: 2909981736@qq.com
"""
import random
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils.set_seed import set_seed
from models.PhantomCNN import PhantomCNN
from utils.add_noise import batch_add_noise
from evaluation.TSNE_Map import plot_tsne_map
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
# from evaluation.GradCAM_plusplus_1d import plot_grad_CAM_plusplus_1d
from evaluation.Three_Dim_Channel_Spatial_Feature_Plot import plot_3dcsfp
from evaluation.Heartmap_Feature_Map_Plot import plot_heartmap_feature_map
from evaluation.Feature_Signal_Diagram_Plot import plot_feature_signal_diagram
from utils.hook_fn import save_feature_map_hook_fn, append_feature_map_hook_fn

# 超参数设置
# 随机数
set_seed(42)
# 设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 信噪比
snr_db = None
# 模型参数
model_params = {'channels': 128, 'kernel_size': 63, 'dilation': [1, 2, 5, 1], 'num_classes': 10, 'phantomlayer_num': 4}
# 模型权重
model_weights = "log/weights/PhantomCNN.pth"
# 数据路径
data_path = "data/CWRU/packaged/12kDE_2_data.npy"
# 数据标签路径
labels_path = "data/CWRU/packaged/12kDE_2_labels.npy"


# 测试集准确率
def test_acc_fn(model, test_data, test_labels, device="cuda:0"):
    test_dataset = TensorDataset(test_data, test_labels)
    test_num = len(test_dataset)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    acc = 0.0
    with torch.no_grad():
        for data in test_loader:
            tdata, tlabels = data
            outputs = model(tdata.to(device))
            predict_y = torch.max(outputs, dim=1)[1]
            acc += torch.eq(predict_y, tlabels.to(device)).sum().item()

        test_accurate = acc / test_num
        print(f'test_data count: {test_num}  test_accuracy: {test_accurate:.3f}')


# 3D 中间层特征图可视化
def feature_map_visualization_by_3D(model, data, labels,
                                    save_path="./log/plot_test/3D_feature_map_visualization.png",
                                    device="cuda:0"):
    data = data.unsqueeze(0).to(device)
    # model.big_kernel_conv.register_forward_hook(save_feature_map_hook_fn)
    model.phantomlayers[0].register_forward_hook(save_feature_map_hook_fn)
    model.phantomlayers[1].register_forward_hook(save_feature_map_hook_fn)
    model.phantomlayers[2].register_forward_hook(save_feature_map_hook_fn)
    model.phantomlayers[3].register_forward_hook(save_feature_map_hook_fn)
    # model.phantomlayer5.register_forward_hook(save_feature_map_hook_fn)
    model.phantomlayers[3].phantom_conv1.primary_conv.register_forward_hook(save_feature_map_hook_fn)
    model.phantomlayers[3].phantom_conv1.cheap_conv.register_forward_hook(save_feature_map_hook_fn)
    outputs = model(data)
    predict = torch.max(outputs, dim=1)[1].item()
    print(f'labels:{labels}  predict:{predict}')
    # model.big_kernel_conv.feature_map.detach().cpu().numpy(),
    feature_map = [
        model.phantomlayers[0].feature_map.detach().cpu().numpy(),
        model.phantomlayers[1].feature_map.detach().cpu().numpy(),
        model.phantomlayers[2].feature_map.detach().cpu().numpy(),
        model.phantomlayers[3].feature_map.detach().cpu().numpy(),
        #    model.phantomlayer5.feature_map.detach().cpu().numpy(),
        model.phantomlayers[3].phantom_conv1.primary_conv.feature_map.detach().cpu().numpy(),
        model.phantomlayers[3].phantom_conv1.cheap_conv.feature_map.detach().cpu().numpy()]
    plot_3dcsfp(feature_map, cmap="jet", savename=save_path, show=True)


# 2D heartmap 中间层(小核和扩张大核层)特征图可视化
def feature_map_visualization_by_heartmap(model, data, labels, downsample_factor=1,
                                          save_path="./log/plot_test/heartmap_feature_map_visualization.png",
                                          device="cuda:0"):
    data = data.unsqueeze(0).to(device)
    # model.phantomlayers[0].phantom_conv1.primary_conv.register_forward_hook(save_feature_map_hook_fn)
    # model.phantomlayers[0].phantom_conv1.cheap_conv.register_forward_hook(save_feature_map_hook_fn)
    # model.phantomlayers[1].phantom_conv1.primary_conv.register_forward_hook(save_feature_map_hook_fn)
    # model.phantomlayers[1].phantom_conv1.cheap_conv.register_forward_hook(save_feature_map_hook_fn)
    # model.phantomlayers[2].phantom_conv1.primary_conv.register_forward_hook(save_feature_map_hook_fn)
    # model.phantomlayers[2].phantom_conv1.cheap_conv.register_forward_hook(save_feature_map_hook_fn)
    model.phantomlayers[3].phantom_conv1.primary_conv.register_forward_hook(save_feature_map_hook_fn)
    model.phantomlayers[3].phantom_conv1.cheap_conv.register_forward_hook(save_feature_map_hook_fn)
    # model.phantomlayer5.phantom_conv1.primary_conv.register_forward_hook(save_feature_map_hook_fn)
    # model.phantomlayer5.phantom_conv1.cheap_conv.register_forward_hook(save_feature_map_hook_fn)
    outputs = model(data)
    predict = torch.max(outputs, dim=1)[1].item()
    print(f'labels:{labels}  predict:{predict}')
    # 调整张量形状为二维矩阵存入feature_map
    feature_map = [
        model.phantomlayers[3].phantom_conv1.primary_conv.feature_map.reshape(64, 256).detach().cpu().numpy(),
        model.phantomlayers[3].phantom_conv1.cheap_conv.feature_map.reshape(64, 256).detach().cpu().numpy()]
    # model.phantomlayers[0].phantom_conv1.primary_conv.feature_map.reshape(64, 256).detach().cpu().numpy(),
    #                    model.phantomlayers[0].phantom_conv1.cheap_conv.feature_map.reshape(64, 256).detach().cpu().numpy(),
    #                    model.phantomlayers[1].phantom_conv1.primary_conv.feature_map.reshape(64, 256).detach().cpu().numpy(),
    #                    model.phantomlayers[1].phantom_conv1.cheap_conv.feature_map.reshape(64, 256).detach().cpu().numpy(),
    #                    model.phantomlayers[2].phantom_conv1.primary_conv.feature_map.reshape(64, 256).detach().cpu().numpy(),
    #                    model.phantomlayers[2].phantom_conv1.cheap_conv.feature_map.reshape(64, 256).detach().cpu().numpy(),

    plot_heartmap_feature_map(feature_map, downsample_factor=downsample_factor, cmap="jet", show=True,
                              savename=save_path)


# 绘制指定层权重分布直方图
def plot_weights_distribution_histogram(model, target_layer,
                                        save_path="./log/plot_test/weights_distribution_histogram.png"):
    # print(model.state_dict())
    weights = []
    for index in range(len(target_layer)):
        having = False
        for name, param in model.state_dict().items():
            if target_layer[index] in name:
                weights.append(param)
                having = True
        if not having:
            raise ValueError(f"没有 {target_layer[index]}")

    # 将权重展平为一维数组
    weights_flat = np.concatenate([w.cpu().flatten() for w in weights])

    weight_min = np.min(weights_flat)
    weight_max = np.max(weights_flat)
    weights_normalized = (weights_flat - weight_min) / (weight_max - weight_min)
    # 绘制直方图
    plt.hist(weights_normalized, bins=20)
    plt.xlabel('Weight Value')
    plt.ylabel('Frequency')
    plt.title('Weight Distribution Histogram')
    plt.savefig(save_path)
    plt.show()
    plt.close()


# 绘制Grad-CAM++ heartmap
# def grad_cam_heartmap_visualization(model, target_layer, data, labels, camp='jet',
#                                     save_path="./log/plot_test/grad_cam_plusplus_heartmap.png",
#                                     device="cuda:0"):
#     data = data.unsqueeze(0).to(device)
#     outputs = model(data)
#     predict = torch.max(outputs, dim=1)[1].item()
#     print(f'labels:{labels}  predict:{predict}')
#     plot_grad_CAM_plusplus_1d(model, target_layer, data, save_path, camp=camp, device=device)


# 绘制中间层t-SNE降维图
def tsne_visualization(model, test_data, test_labels, camp='rainbow', n_components=2, verbose=1, perplexity=100,
                       n_iter=1000,
                       save_path="./log/plot_test/tsne_visualization.png",
                       device="cuda:0"):
    sample_num = len(test_data)
    test_dataset = TensorDataset(test_data, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    # model.big_kernel_conv.register_forward_hook(append_feature_map_hook_fn)
    model.phantomlayers[0].register_forward_hook(append_feature_map_hook_fn)
    model.phantomlayers[1].register_forward_hook(append_feature_map_hook_fn)
    model.phantomlayers[2].register_forward_hook(append_feature_map_hook_fn)
    model.phantomlayers[3].register_forward_hook(append_feature_map_hook_fn)
    # model.phantomlayer5.register_forward_hook(append_feature_map_hook_fn)
    labels = []
    with torch.no_grad():
        for data in test_loader:
            tdata, tlabels = data
            outputs = model(tdata.to(device))
            labels.append(tlabels)
    labels = torch.cat(labels).numpy()
    feature_map = [
        torch.cat(model.phantomlayers[0].feature_map_list).detach().view(sample_num, -1).cpu().numpy(),
        torch.cat(model.phantomlayers[1].feature_map_list).detach().view(sample_num, -1).cpu().numpy(),
        torch.cat(model.phantomlayers[2].feature_map_list).detach().view(sample_num, -1).cpu().numpy(),
        torch.cat(model.phantomlayers[3].feature_map_list).detach().view(sample_num, -1).cpu().numpy()]
    plot_tsne_map(feature_map, labels, camp=camp, n_components=n_components, verbose=verbose, perplexity=perplexity,
                  n_iter=n_iter, savename=save_path)


# def tsne_visualization(model, test_data, test_labels, camp='rainbow', n_components=2, verbose=1, perplexity=100,
#                        n_iter=1000,
#                        save_path="./log/plot_test/tsne_visualization2.png",
#                        device="cuda:0"):
#     sample_num = len(test_data)
#     test_dataset = TensorDataset(test_data, test_labels)
#     test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
#     model.conv_1.register_forward_hook(append_feature_map_hook_fn)
#     model.mix_1.register_forward_hook(append_feature_map_hook_fn)
#     model.mix_2.register_forward_hook(append_feature_map_hook_fn)
#     model.mix_3.register_forward_hook(append_feature_map_hook_fn)
#     labels = []
#     with torch.no_grad():
#         for data in test_loader:
#             tdata, tlabels = data
#             outputs = model(tdata.to(device))
#             labels.append(tlabels)
#     labels = torch.cat(labels).numpy()
#     feature_map = [torch.cat(model.conv_1.feature_map).detach().view(sample_num, -1).cpu().numpy(),
#                    torch.cat(model.mix_1.feature_map).detach().view(sample_num, -1).cpu().numpy(),
#                    torch.cat(model.mix_2.feature_map).detach().view(sample_num, -1).cpu().numpy(),
#                    torch.cat(model.mix_3.feature_map).detach().view(sample_num, -1).cpu().numpy()]
#     plot_tsne_map(feature_map, labels, camp=camp, n_components=n_components, verbose=verbose, perplexity=perplexity,
#                   n_iter=n_iter, savename=save_path)

# def feature_signal_diagram_visualization(model, data, labels,
#                                          save_path="./log/plot_test/feature_signal_diagram_visualization.png",
#                                          device="cuda:0"):
# data = data.unsqueeze(0).to(device)
# model.phantomlayer5.phantom_conv2.primary_conv.register_forward_hook(save_feature_map_hook_fn)
# model.phantomlayer5.phantom_conv2.cheap_conv.register_forward_hook(save_feature_map_hook_fn)
# outputs = model(data)
# predict = torch.max(outputs, dim=1)[1].item()
# print(f'labels:{labels}  predict:{predict}')
# # 调整张量形状为二维矩阵存入feature_map
# feature_map = [model.phantomlayer5.phantom_conv2.primary_conv.feature_map.reshape(48, 256).detach().cpu().numpy(),
#                model.phantomlayer5.phantom_conv2.cheap_conv.feature_map.reshape(48, 256).detach().cpu().numpy()]

# print(model.phantomlayer5.phantom_conv2.primary_conv.pw_conv.weight.shape)
# print(model.phantomlayer5.phantom_conv2.cheap_conv.dw_conv.weight.shape)
# plot_feature_signal_diagram(feature_map, savename=save_path)


if __name__ == '__main__':
    # 模型载入
    model = PhantomCNN(**model_params)
    model.load_state_dict(torch.load(model_weights))
    model.to(device)
    model.eval()

    # 加载原始数据
    data = np.load(data_path)
    data = batch_add_noise(data, snr_db)
    labels = np.load(labels_path)

    # 划分数据集,获取测试集
    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.1, random_state=42)
    test_data = torch.from_numpy(test_data).float()
    test_data = test_data.unsqueeze(1)
    test_labels = torch.from_numpy(test_labels).long()

    train_data = torch.from_numpy(train_data).float()
    train_data = train_data.unsqueeze(1)
    train_labels = torch.from_numpy(train_labels).long()
    # eval评价

    for label in [3]:
        flag = True
        while flag:

            rand = random.randint(0, 500)  # 321

            if test_labels[rand] == label:
                print(rand)
                # test_acc_fn(model, test_data, test_labels)
                print("-----------------------------------------------------------")
                feature_map_visualization_by_3D(model, test_data[rand], test_labels[rand],
                                                save_path=f"./log/plot_test/3D_feature_map_visualization_rand{rand}_{label}.png")
                print("-----------------------------------------------------------")
                feature_map_visualization_by_heartmap(model, test_data[rand], test_labels[rand], downsample_factor=2,
                                                      save_path=f"./log/plot_test/heartmap_feature_map_visualization__rand{rand}_{label}.png")
                print("-----------------------------------------------------------")
                plot_weights_distribution_histogram(model, ["phantomlayers.3.phantom_conv1.primary_conv.pw_conv.weight",
                                                            "phantomlayers.3.phantom_conv1.cheap_conv.dw_conv.weight"])
                print("-----------------------------------------------------------")
                # grad_cam_heartmap_visualization(model, model.phantomlayers[3], test_data[rand], test_labels[rand],
                #                                 save_path=f"./log/plot_test/10heartmap/grad_cam_plusplus_heartmap_rand_{rand}_{label}.png", )
                print("-----------------------------------------------------------")
                tsne_visualization(model, train_data[:4000, :], train_labels[:4000], camp='rainbow', perplexity=300)
                print("-----------------------------------------------------------")
                # feature_signal_diagram_visualization(model, test_data[rand], test_labels[rand])
                flag = False
