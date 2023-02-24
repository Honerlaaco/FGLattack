import os
os.environ['CUDA_VISIBLE_DEVICES']='4'

from torch_geometric.data import DataLoader
from FedGNNattack.Graph_AE.utils.CustomDataSet import SelectGraph
from FedGNNattack.Graph_AE.utils.train_utils import train_cp
import torch
import argparse
from torch_geometric.datasets import TUDataset



def main(arg):
    device = torch.device(arg.device)

    num_epoch = arg.e
    batch_size = arg.batch

    SelectGraph.data_name = args.d


    shapes = list(map(int, arg.shapes.split(",")))

    dataset_name = args.d  # PROTEINS, NCI1, AIDS, MUTAG, ENZYMES
    path = './data/TU'
    dataset = TUDataset(path, name=dataset_name).shuffle()
    input_size = dataset.num_features
    print(input_size)

    train_dataset = dataset[7 * len(dataset) // 10:]
    test_dataset = dataset[:3 * len(dataset) // 10]
    train_set = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_set = DataLoader(test_dataset, batch_size=batch_size)


    if arg.m == "MIAGAE":
        from classification.Graph_AE import Net
        model = Net(input_size, arg.k, arg.depth, [arg.c_rate] * arg.depth, shapes, device).to(device)
    elif arg.m == "UNet":
        from classification.UNet import Net
        model = Net(input_size, arg.depth, arg.c_rate, shapes, device).to(device)
    elif arg.m == "Gpool":
        from classification.Gpool_model import Net
        model = Net(input_size, arg.depth, arg.c_rate, shapes, device).to(device)
    elif arg.m == "SAGpool":
        from classification.SAG_model import Net
        model = Net(input_size, arg.depth, [arg.c_rate] * arg.depth, shapes, device).to(device)
    else:
        print("model not found")
        return

    optimizer = torch.optim.Adam(model.parameters(), lr=arg.lr)
    import timeit
    start = timeit.default_timer()
    train_cp(model, optimizer, device, train_set, test_set, num_epoch, arg.model_dir, arg.m)
    end = timeit.default_timer()
    print("part1:" + str(end - start))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Global_Dict generator')
    parser.add_argument('--d', type=str, default='AIDS', help="dataset name")
    parser.add_argument('--m', type=str, default='SAGpool', help="model name")
    parser.add_argument('--device', type=str, default='cuda', help="cuda / cpu")
    parser.add_argument('--batch', type=int, default=16, help="batch size")
    parser.add_argument('--e', type=int, default=100, help="number of epochs")
    parser.add_argument('--lr', type=float, default=1e-3, help="learning rate")
    parser.add_argument('--model_dir', type=str, default="data/model/", help="path to save model")
    parser.add_argument('--n_train', type=int, default=300, help="number of samples for train set")
    parser.add_argument('--n_test', type=int, default=100, help="number of samples for test set")
    parser.add_argument('--k', type=int, default=2, help="number of kernels")
    parser.add_argument('--depth', type=int, default=1, help="depth of encoder and decoder")
    parser.add_argument('--c_rate', type=float, default=0.8, help="compression ratio for each layer of encoder")
    parser.add_argument('--shapes', type=str, default="8", help="shape of each layer in encoder")
    args = parser.parse_args()

    main(args)

