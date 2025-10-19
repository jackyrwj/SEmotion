import torch
import numpy as np

from torcheeg.datasets.constants.emotion_recognition.deap import \
    DEAP_CHANNEL_LOCATION_DICT
from torcheeg.transforms import ToGrid

ALL_FRONT = [[1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1],
             [1, 1, 1, 1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0]]

ALL_LEFT = [[1, 1, 1, 1, 0, 0, 0, 0, 0], [1, 1, 1, 1, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 0, 0, 0], [1, 1, 1, 1, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 0, 0, 0], [1, 1, 1, 1, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 0, 0, 0], [1, 1, 1, 1, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 0, 0, 0]]

FRONT_LEFT = [[1, 1, 1, 1, 0, 0, 0, 0, 0], [1, 1, 1, 1, 0, 0, 0, 0, 0],
              [1, 1, 1, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0]]

FRONT_RIGHT = [[0, 0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 0, 0, 1, 1, 1, 1],
               [0, 0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0]]

# ALL = [[1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1],
#             [1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1],
#             [1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1],
#             [1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1],
#             [1, 1, 1, 1, 1, 1, 1, 1, 1]]

ALL = [[0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0]]


# list = ['FP1', 'AF3', 'F3', 'F7', 'FC5', 'FC1',    'C3', 'T7',      'CP5', 'CP1', 'P3', 'P7', 'PO3', 'O1', 
#         'FP2', 'AF4', 'F4', 'F8', 'FC6', 'FC2',    'C4', 'T8',      'CP6', 'CP2', 'P4', 'P8', 'PO4', 'O2']
NEW_ALL_FRONT = [[1], [1], [1], [1], [1], [1],      [1], [1],        [0], [0], [0], [0], [0], [0], 
                    [1], [1], [1], [1], [1], [1],      [1], [1],        [0], [0], [0], [0], [0], [0]]


NEW_FRONT_LEFT = [[1], [1], [1], [1], [1], [1],      [1], [1],        [0], [0], [0], [0], [0], [0], 
                    [0], [0], [0], [0], [0], [0],      [0], [0],        [0], [0], [0], [0], [0], [0]]

ALL = [[1], [1], [1], [1], [1], [1],      [1], [1],        [0], [0], [0], [0], [0], [0],  [1], [1],
                    [0], [0], [0], [0], [0], [0],      [0], [0],        [0], [0], [0], [0], [0], [0],  [1], [1]]


MASK_MODES = {
    'ALL_FRONT': ALL_FRONT,
    'ALL_LEFT': ALL_LEFT,
    'FRONT_LEFT': FRONT_LEFT,
    'FRONT_RIGHT': FRONT_RIGHT,
    'NEW_ALL_FRONT': NEW_ALL_FRONT,
    'NEW_FRONT_LEFT': NEW_FRONT_LEFT,
    'ALL' : ALL
}


def mask_region(eeg, mode='ALL_FRONT'):
    assert mode in MASK_MODES
    mask_list = MASK_MODES[mode]
    if isinstance(eeg, np.ndarray):
        mask = np.array(mask_list, dtype=eeg.dtype)
        mask = mask[np.newaxis, ...]
    elif isinstance(eeg, torch.Tensor):
        mask = torch.tensor(mask_list, dtype=eeg.dtype, device=eeg.device)
        mask = mask.unsqueeze(0)
    else:
        raise NotImplementedError()
    return eeg * mask




# quick test
# eeg = ToGrid(DEAP_CHANNEL_LOCATION_DICT)(eeg=np.ones((32, 128)))['eeg']
# mask_region(eeg)


def grad_cam_loss_v2(model,
                     HEAD,
                     inputs,
                     labels,
                     mode='NEW_FRONT_LEFT',
                    #  reweight='clamp', 
                     reweight='sigmoid', 
                     layer= 'block3',
                     array = [],
                     counter = 0
                     ):
    intermediate_outputs = {}


    def hook(module, input, output):
        intermediate_outputs[module] = output

    if layer == 'block2':
        # handle = model.module.block2.register_forward_hook(hook)
        handle = model.block2.register_forward_hook(hook)
    elif layer == 'block3':
        # handle = model.module.block3.register_forward_hook(hook)
        handle = model.block3.register_forward_hook(hook)
    elif layer == 'conv1':
        handle = model.conv1.register_forward_hook(hook)
    elif layer == 'conv2':
        handle = model.conv2.register_forward_hook(hook)
    elif layer == 'conv3':
        handle = model.conv3.register_forward_hook(hook)
    elif layer == 'conv4':
        handle = model.conv4.register_forward_hook(hook)
    elif layer == 'BN_s':
        handle = model.BN_s.register_forward_hook(hook)
    elif layer == 'pp':
        handle = model.patch_embedding.shallownet.register_forward_hook(hook)
        

    # if counter == 10:
    #     print(10)



    # outputs, theta_yi, theta_j= HEAD(features, labels)

    outputs = model(inputs.requires_grad_())
    # outputs = HEAD(outputs, labels)

    # features = model(inputs.requires_grad_())
    # outputs, loss, intra_loss, inter_loss, WyiX, WjX, theta_yi, theta_j = HEAD(features, labels)


    intermediate_outputs = list(intermediate_outputs.values())

    handle.remove()
    # start here
    grad_output = torch.nn.functional.one_hot(labels)

    alpher_vector_list = []
    grad_intermediate_list = []

    for i in range(len(intermediate_outputs)):
        if i == 0:
            grad_intermediate = torch.autograd.grad(
                outputs=outputs,
                inputs=intermediate_outputs[i],
                grad_outputs=grad_output,
                retain_graph=True,
                create_graph=True
                )[0]
        else:
            grad_intermediate = torch.autograd.grad(
                outputs=intermediate_outputs[i - 1],
                inputs=intermediate_outputs[i],
                grad_outputs=alpher_vector_list[i - 1],
                retain_graph=True,
                create_graph=True,
                # allow_unused=True
                )[0]
            grad_intermediate = grad_intermediate * torch.sum(grad_intermediate_list[i - 1], dim=(1, 2, 3)).view(-1, 1, 1, 1)

        alpher_vector = torch.sum(grad_intermediate,axis=[2, 3]).view(grad_intermediate.shape[0],grad_intermediate.shape[1],1, 1)

        max_alpher = torch.amax(alpher_vector, dim=(1, 2, 3))
        min_alpher = torch.amin(alpher_vector, dim=(1, 2, 3))
        min_alpher = min_alpher.view(min_alpher.shape[0], 1, 1, 1)
        max_alpher = max_alpher.view(max_alpher.shape[0], 1, 1, 1)

        alpher_vector = ((alpher_vector - min_alpher) / (max_alpher - min_alpher))

        alpher_vector = torch.tile(alpher_vector,(1, 1, grad_intermediate.shape[2], grad_intermediate.shape[3]))
        # alpher_vector[torch.clamp(alpher_vector, min=0.9, max=1) == 0.9] = 0
        alpher_vector_list.append(alpher_vector)
        grad_intermediate_list.append(grad_intermediate)

    grad_input_tem = torch.autograd.grad(
        outputs=intermediate_outputs[len(intermediate_outputs) - 1],
        inputs=inputs,
        grad_outputs=alpher_vector_list[len(alpher_vector_list) - 1],
        retain_graph=True,
        create_graph=True)[0]

    des_grad_intermediate = torch.sum(grad_intermediate_list[len(grad_intermediate_list) - 1],dim=(1, 2, 3)).view(-1, 1, 1, 1)
    grad_input = des_grad_intermediate * grad_input_tem



    tem_data = torch.sum(grad_input, axis=1)

    max_matrix = torch.amax(tem_data, dim=(1, 2))
    min_matrix = torch.amin(tem_data, dim=(1, 2))
    min_matrix = min_matrix.view(min_matrix.shape[0], 1, 1)
    max_matrix = max_matrix.view(max_matrix.shape[0], 1, 1)
    tem_data = (tem_data - min_matrix) / (max_matrix - min_matrix)

# ---------------------------------------------
    # batch_size, height, width
    mul_matrix = mask_region(tem_data, mode=mode)
    mul_sample = mul_matrix.mean(dim=[-1, -2])
    if reweight == 'clamp':
        mul_sample = torch.clamp(mul_sample, min=0.0)
    elif reweight == 'sigmoid':
        mul_sample = torch.sigmoid(mul_sample)
    elif reweight == 'abs':
        mul_sample = torch.abs(mul_sample)
    else:
        raise NotImplementedError()
    print(counter)
    counter = counter + 1
    print(mul_sample)


    # # 读取保存的 NumPy 数组
    # tensor_array = np.load('./results/tensor.npy', allow_pickle=True)


    # # 将新张量转换为 NumPy 数组
    # new_tensor_array = mul_sample.cpu().detach().numpy()

    # # 将新数组添加到现有数组的末尾  
    # array_list = np.concatenate((tensor_array, new_tensor_array), axis=0)

    # # 保存整个数组到文件
    # np.save('./results/tensor.npy', array_list)
    
    re_array = np.concatenate((array, mul_sample.cpu().detach().numpy()), axis=0)



    # np.save('./results/classical.npy', mul_sample.cpu(),allow_pickle=False)
    
    # loss = torch.nn.functional.cross_entropy(outputs, labels,reduce=False) * mul_sample
    # loss = torch.nn.functional.cross_entropy(outputs, labels,reduction='none') * mul_sample
    
    # return loss.mean()
    return re_array,mul_sample,counter
    # return outputs, loss.mean(), intra_loss, inter_loss, WyiX, WjX, theta_yi, theta_j




# if __name__ == '__main__':
#     # quick test
#     from model import Model
#     from fbccnn import FBCCNN

#     # mock_model = Model(num_classes=2)
#     # mock_input = torch.randn(2, 128, 9, 9)
#     # mock_y = torch.ones(2, dtype=torch.long)

#     # print(grad_cam_loss_v2(mock_model, mock_input, mock_y))


    
#     mock_model = FBCCNN(
#                 num_classes=2, 
#                 in_channels=4, grid_size=(9, 9))
#     mock_input = torch.randn(64,4,9,9)
#     mock_y = torch.ones(64, dtype=torch.long)
#     print(grad_cam_loss_v2(mock_model, mock_input, mock_y))