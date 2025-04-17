
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections as mcoll
import cv2


def target_category_loss(x, category_index, nb_classes):
    return torch.mul(x, F.one_hot(category_index, nb_classes))


def target_category_loss_output_shape(input_shape):
    return input_shape


def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (torch.sqrt(torch.mean(torch.square(x))) + 1e-5)


def resize_1d(array, shape):
    res = np.zeros(shape)
    if array.shape[0] >= shape:
        ratio = array.shape[0] / shape
        for i in range(array.shape[0]):
            res[int(i / ratio)] += array[i] * (1 - (i / ratio - int(i / ratio)))
            if int(i / ratio) != shape - 1:
                res[int(i / ratio) + 1] += array[i] * (i / ratio - int(i / ratio))
            else:
                res[int(i / ratio)] += array[i] * (i / ratio - int(i / ratio))
        res = res[::-1]
        array = array[::-1]
        for i in range(array.shape[0]):
            res[int(i / ratio)] += array[i] * (1 - (i / ratio - int(i / ratio)))
            if int(i / ratio) != shape - 1:
                res[int(i / ratio) + 1] += array[i] * (i / ratio - int(i / ratio))
            else:
                res[int(i / ratio)] += array[i] * (i / ratio - int(i / ratio))
        res = res[::-1] / (2 * ratio)
        array = array[::-1]
    else:
        ratio = shape / array.shape[0]
        left = 0
        right = 1
        for i in range(shape):
            if left < int(i / ratio):
                left += 1
                right += 1
            if right > array.shape[0] - 1:
                res[i] += array[left]
            else:
                res[i] += array[right] * \
                          (i - left * ratio) / ratio + array[left] * (right * ratio - i) / ratio
        res = res[::-1]
        array = array[::-1]
        left = 0
        right = 1
        for i in range(shape):
            if left < int(i / ratio):
                left += 1
                right += 1
            if right > array.shape[0] - 1:
                res[i] += array[left]
            else:
                res[i] += array[right] * \
                          (i - left * ratio) / ratio + array[left] * (right * ratio - i) / ratio
        res = res[::-1] / 2
        array = array[::-1]
    return res


class ActivationsAndGradients:
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = []
        self.activations = []

        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations.append(output)

    def save_gradient(self, module, grad_input, grad_output):
        # Gradients are computed in reverse order
        self.gradients = [grad_output[0]] + self.gradients

    def __call__(self, x):
        self.gradients = []
        self.activations = []
        return self.model(x)


class BaseCAM:
    def __init__(self, model, target_layer, use_cuda=False):
        self.model = model.eval()
        self.target_layer = target_layer
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.activations_and_grads = ActivationsAndGradients(self.model, target_layer)

    def forward(self, input_img):
        return self.model(input_img)

    def get_cam_weights(self,
                        input_tensor,
                        target_category,
                        activations,
                        grads):
        raise Exception("Not Implemented")

    def get_loss(self, output, target_category):
        print(output.size())
        return output[target_category]

    def __call__(self, input_tensor, target_category=None):
        if self.cuda:
            input_tensor = input_tensor.cuda()

        output = self.activations_and_grads(input_tensor)

        if target_category is None:
            output = output.squeeze()
            target_category = np.argmax(output.cpu().data.numpy())
            print(output)
            print(target_category)
        self.model.zero_grad()
        loss = self.get_loss(output, target_category)
        loss.backward(retain_graph=True)

        activations = self.activations_and_grads.activations[-1].cpu().data.numpy()[0, :]
        grads = self.activations_and_grads.gradients[-1].cpu().data.numpy()[0, :]
        weights = self.get_cam_weights(input_tensor, target_category, activations, grads)
        cam = activations.T.dot(weights)  # maybe better
        cam = resize_1d(cam, (input_tensor.shape[2]))
        heatmap = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-10)
        print(heatmap.shape)
        return heatmap


class GradCAM(BaseCAM):
    def __init__(self, model, target_layer, use_cuda=False):
        super(GradCAM, self).__init__(model, target_layer, use_cuda)

    def get_cam_weights(self, input_tensor,
                        target_category,
                        activations,
                        grads):
        grads_power_2 = grads ** 2
        grads_power_3 = grads_power_2 * grads
        sum_activations = np.sum(activations, axis=1)
        eps = 0.000001
        aij = grads_power_2 / (2 * grads_power_2 + sum_activations[:, None] * grads_power_3 + eps)
        aij = np.where(grads != 0, aij, 0)

        weights = np.maximum(grads, 0) * aij
        weights = np.sum(weights, axis=1)
        return weights


def colorline(x, y, heatmap, cmap='rainbow'):
    z = np.array(heatmap)
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = mcoll.LineCollection(segments, array=z, cmap=cmap)
    ax = plt.gca()
    ax.add_collection(lc)
    return lc


def multicolored_lines(x, y, heatmap, title_name, cmap='jet', save_path=None):
    fig, ax = plt.subplots()
    lc = colorline(x, y, heatmap, cmap=cmap)
    plt.colorbar(lc)
    lc.set_linewidth(2)
    lc.set_alpha(0.8)
    plt.xlim(x.min(), x.max())
    plt.ylim(y.min(), y.max())
    plt.title(title_name)
    plt.grid(False)
    fig.set_size_inches(7, 2)
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()
    plt.close()


def plot_grad_CAM_plusplus_1d(model, target_layer, data, save_path, camp='jet', device="cuda:0"):
    use_cuda = False
    if "cuda" in device:
        use_cuda = True
    net = GradCAM(model, target_layer, use_cuda=use_cuda)
    # input_tensor = data.unsqueeze(0)
    output = net(data)
    input_tensor = data.cpu().numpy().squeeze()
    # Normalization
    cam = np.maximum(output, 0)
    cam = cam / np.max(cam)
    heatmap = []
    heatmap.append(cam.tolist())
    big_heatmap = cv2.resize(np.array(heatmap), dsize=(input_tensor.shape[0], 500), interpolation=cv2.INTER_CUBIC)
    x = np.linspace(0, 1, input_tensor.shape[0])
    plt.style.use("seaborn-whitegrid")
    multicolored_lines(x, np.array([i for i in input_tensor]), big_heatmap[0], f"PhantomCNN GradCAM ++ Visualization",
                       cmap=camp, save_path=save_path)

    # plt.plot(x, np.array([i for i in input_tensor]), c="blue")
    # plt.plot(x, big_heatmap[0], c="red")
    # plt.show()
    # # 计算频谱
    # import scipy.signal as signal
    # freq, spectrum = signal.periodogram(np.array([i for i in input_tensor]))
    # # 绘制频谱图
    # plt.figure()
    # plt.plot(freq, spectrum)
    # plt.xlabel('Frequency')
    # plt.ylabel('Power Spectral Density')
    # plt.title('Power Spectrum')
    # plt.show()
    # # 傅里叶变换
    # fft_data = np.fft.fft(big_heatmap[0])
    # # 获取频率轴
    # freqs = np.fft.fftfreq(len(fft_data))
    # # 绘制频谱图
    # plt.plot(np.abs(freqs), np.abs(fft_data))
    # plt.xlabel('Frequency')
    # plt.ylabel('Amplitude')
    # plt.show()
