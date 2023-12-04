# ip及对应节点位序
from Communicator import Communicator
import torch
from vgg import VGG
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import socket
import time

# CLIENTS_CONFIG = {"192.168.0.14": 0, "192.168.0.15": 1, "192.168.0.25": 2}
# CLIENTS_LIST = ["192.168.0.14", "192.168.0.15", "192.168.0.25"]
CLIENTS_CONFIG = {"127.0.0.1": 0, "127.0.0.1": 1, "127.0.0.1": 2}
CLIENTS_LIST = ["127.0.0.1", "127.0.0.1", "127.0.0.1"]
# Model configration
model_cfg = {
    # (Type, in_channels, out_channels, kernel_size, out_size(c_out*h*w), flops(c_out*h*w*k*k*c_in))
    "VGG5": [
        ("C", 3, 32, 3, 32 * 32 * 32, 32 * 32 * 32 * 3 * 3 * 3),
        ("M", 32, 32, 2, 32 * 16 * 16, 0),
        ("C", 32, 64, 3, 64 * 16 * 16, 64 * 16 * 16 * 3 * 3 * 32),
        ("M", 64, 64, 2, 64 * 8 * 8, 0),
        ("C", 64, 64, 3, 64 * 8 * 8, 64 * 8 * 8 * 3 * 3 * 64),
        ("D", 8 * 8 * 64, 128, 1, 64, 128 * 8 * 8 * 64),
        ("D", 128, 10, 1, 10, 128 * 10),
    ]
}

# 在每个节点上计算的第k层
split_layer = {0: [0, 1], 1: [2, 3], 2: [4, 5, 6]}

reverse_split_layer = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2}

host_port = 1999
host_node_num = 2
host_ip = CLIENTS_LIST[host_node_num]

info = "MSG_FROM_NODE(%d), host= %s" % (host_node_num, host_ip)

loss_list = []
acc_list = []

model_name = "VGG5"
model_len = len(model_cfg[model_name])

N = 10000 # data length
B = 256 # Batch size
### 假设本节点为节点2
class node_end(Communicator):
    def __init__(self,host_ip,host_port):
        super(node_end, self).__init__(host_ip,host_port)

    def add_addr(self, node_addr, node_port):
        while True:
            try:
                self.sock.connect((node_addr, node_port))
                break  # If the connection is successful, break the loop
            except socket.error as e:
                print(f"Failed to connect to {node_addr}:{node_port}, retrying...")
                time.sleep(1)  # Wait for a while before retrying

# TODO:理解这个函数
def calculate_accuracy(fx, y):
    preds = fx.max(1, keepdim=True)[1]
    #print("preds={}, y.view_as(preds)={}".format(preds, y.view_as(preds)))
    correct = preds.eq(y.view_as(preds)).sum()
    acc = 100.00 * correct.float() / preds.shape[0]
    return acc

def node_inference(node, model):
    while 1:
        last_send_ips=[]
        iteration = int(N / B)
        node_socket, node_addr = node.wait_for_connection()
        for i in range(iteration):
            print(f"node{host_node_num} get connection from node{node_addr}")
            msg = node.recv_msg(node_socket)
            data = msg[1]
            target = msg[2]
            start_layer = msg[3]
            data, next_layer, split = calculate_output(model, data, start_layer)
            if split + 1 < model_len:
                last_send_ip=CLIENTS_LIST[reverse_split_layer[split + 1]]
                if last_send_ip not in last_send_ips:
                    node.add_addr(last_send_ip, 2000)
                last_send_ips.append(last_send_ip)
                msg = [info, data.cpu(), target.cpu(), next_layer]
                node.send_msg(node.sock, msg)
                print(
                    f"node{host_node_num} send msg to node{CLIENTS_LIST[reverse_split_layer[split + 1]]}"
                )
            else:
                # 到达最后一层，计算损失
                loss = torch.nn.functional.cross_entropy(data, target)
                acc = calculate_accuracy(data, target)
                loss_list.append(loss)
                acc_list.append(acc)

        print("loss :{:.4}".format(sum(loss_list) / len(loss_list)))
        print("acc :{:.4}%".format(sum(acc_list) / len(acc_list)))
        
        node_socket.close()


def get_model(model, type, in_channels, out_channels, kernel_size, start_layer):
    # for name, module in model.named_children():
    #   print(f"Name: {name} | Module: {module}")
    # print(model)
    feature_s = []
    dense_s = []
    if type == "M":
        feature_s.append(model.features[start_layer])
        start_layer += 1
    if type == "D":
        dense_s.append(model.denses[start_layer-11])
        start_layer += 1
    if type == "C":
        for i in range(3):
            feature_s.append(model.features[start_layer])
            start_layer += 1
    next_layer = start_layer
    return nn.Sequential(*feature_s), nn.Sequential(*dense_s), next_layer


def calculate_output(model, data, start_layer):
    for split in split_layer[host_node_num]:
        # TODO:如果节点上的层不相邻，需要兼容
        type = model_cfg[model_name][split][0]
        in_channels = model_cfg[model_name][split][1]
        out_channels = model_cfg[model_name][split][2]
        kernel_size = model_cfg[model_name][split][3]
        # print("type,in_channels,out_channels,kernel_size",type,in_channels,out_channels,kernel_size)
        #print("start_layer", start_layer)
        features, dense, next_layer = get_model(
            model, type, in_channels, out_channels, kernel_size, start_layer
        )
        if len(features) > 0:
            model_layer = features
        else:
            model_layer = dense

        #print("data", data.shape)
        #print("model_layer", model_layer)
        if type=="D":
            data = data.view(data.size(0), -1)
        data = model_layer(data)
        start_layer = next_layer
        #print("next_layer", next_layer)
    return data, next_layer, split


def start_inference():
    include_first = False
    node = node_end(host_ip, host_port)
    # 修改VGG的配置，模型载入改为逐层载入；或者是直接调用载入的模型就行？
    # model= VGG('Unit', 'VGG5',split_layer[host_node_num] , model_cfg)
    # model= VGG('Client', 'VGG5', len(model_cfg[model_name]), model_cfg)

    model = VGG("Client", model_name, 6, model_cfg)
    model.load_state_dict(torch.load("model.pth"))

    # moddel layer Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    # print("moddel layer",model)

    # 如果含第一层，载入数据
    if include_first:
        # TODO:modify the data_dir
        start_layer = 0
        data_dir = "/home/whang1234/Downloads/githubFiles/dataset"
        test_dataset = datasets.CIFAR10(
            data_dir,
            train=False,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                ]
            ),
        )
        test_loader = DataLoader(
            test_dataset, batch_size=256, shuffle=False, num_workers=4
        )
        # TODO:modify the port
        last_send_ips=[] 
        for data, target in test_loader:
            #print(len(data))
            # split:当前节点计算的层
            # next_layer:下一个权重层
            data, next_layer, split = calculate_output(model, data, start_layer)

            # TODO:modify the port
            last_send_ip=CLIENTS_LIST[reverse_split_layer[split + 1]]
            if last_send_ip not in last_send_ips:
                node.add_addr(last_send_ip, 2000)
            last_send_ips.append(last_send_ip)

            # TODO:是否发送labels
            msg = [info, data.cpu(), target.cpu(), next_layer]
            print(
                f"node{host_node_num} send msg to node{CLIENTS_LIST[reverse_split_layer[split + 1]]}"
            )
            node.send_msg(node.sock, msg)
            include_first = False

    node_inference(node, model)


start_inference()
