# ip及对应节点位序
from Communicator import Communicator
import torch
from vgg import VGG
from torchvision import datasets,transforms
from torch.utils.data import DataLoader


CLIENTS_CONFIG= {'192.168.0.14':0, '192.168.0.15':1, '192.168.0.25':2} 
CLIENTS_LIST= ['192.168.0.14', '192.168.0.15', '192.168.0.25'] 
# Model configration
model_cfg = {
	# (Type, in_channels, out_channels, kernel_size, out_size(c_out*h*w), flops(c_out*h*w*k*k*c_in))
	'VGG5' : [('C', 3, 32, 3, 32*32*32, 32*32*32*3*3*3), ('M', 32, 32, 2, 32*16*16, 0), 
	('C', 32, 64, 3, 64*16*16, 64*16*16*3*3*32), ('M', 64, 64, 2, 64*8*8, 0), 
	('C', 64, 64, 3, 64*8*8, 64*8*8*3*3*64), 
	('D', 8*8*64, 128, 1, 64, 128*8*8*64), 
	('D', 128, 10, 1, 10, 128*10)]
}

# 在每个节点上计算的第k层
split_layer={0:[1,2],1:[3,4],2:[5,6]}

reverse_split_layer={1:0,2:0,3:1,4:1,5:2,6:2}

### 假设本节点为节点0
class node_end(Communicator):
    def __init__(self,node_num,ip_address):
        super(node_end,self).__init__(node_num,ip_address)
        self.node_num=node_num
    def add_addr(self,node_addr,node_port):
        self.sock.connect((node_addr,node_port))

def start_inference():
    host_node_num=0
    include_first=True
    node= node_end(host_node_num)

    # 修改VGG的配置，模型载入改为逐层载入；或者是直接调用载入的模型就行？
    # model= VGG('Unit', 'VGG5',split_layer[host_node_num] , model_cfg)
    model= VGG('Unit', 'VGG5', 0, model_cfg)
    model.load_state_dict(torch.load('model.pth'))
    layer_weight=[]
    for layer in model.layers:
        weight=layer.weight
        layer_weight.append(weight)

    # 如果含第一层，载入数据
    if include_first:
        #TODO:modify the data_dir
        data_dir = '/home/dataset'
        test_dataset = datasets.CIFAR10(
            data_dir,
            train=False,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                ]
            ),
        )
        test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4)
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            for split in split_layer[host_node_num]:
                # TODO:如果节点上的层不相邻，需要兼容
                layer_i=layer_weight[split].cuda()
                data = layer_i(data)

            # TODO:modify the port
            node.add_addr(CLIENTS_LIST[reverse_split_layer[split+1]], 6666)           

            info = "MSG_FROM_NODE(%d), host= %s, port= %d" %(host_node_num, CLIENTS_LIST[reverse_split_layer[split+1]], 6666)

            # TODO:是否发送labels
            msg=[info,data.cpu().state_dict()]
            node.send_msg(node.sock, msg)            
            include_first=False
    node_inference(node)

def node_inference(node):
        while(1):
            node.recv_msg(node.sock, 'Finish')

            node.send_msg(node.sock, ('Finish', 'Finish'))
