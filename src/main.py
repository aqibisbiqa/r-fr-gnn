
import os
import random
from sys import argv
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import DataLoader
from torch_scatter import scatter_mean
from tqdm import tqdm

from gnn import *
from utils.argparser import argument_parser
from utils.util import load_data


def __loss_aux(output, loss, data, binary_prediction):
    if binary_prediction:
        labels = torch.zeros_like(output).scatter_(
            1, torch.maximum(data.node_labels,torch.zeros_like(data.node_labels)).unsqueeze(1), 1.)
    else:
        raise NotImplementedError()
    mask=torch.where(data.node_labels>=0,1,0).unsqueeze(1)
    return nn.BCEWithLogitsLoss(reduction='mean',weight=mask)(output, labels)


def train(
        model,
        device,
        training_data,
        optimizer,
        criterion,
        scheduler,
        binary_prediction=True) -> float:
    model.train()

    loss_accum = []

    for data in tqdm(training_data):
        #data = data.to(device)
        for i in range(len(data)):
            data[i]=data[i].to(device)
        edge_indexes=[]
        edge_attrs=[]
        batches=[]
        for i in range(len(data)):
            edge_indexes.append(data[i].edge_index)
            edge_attrs.append(data[i].edge_attr)
            batches.append(data[i].batch)
        output = model(x=data[0].x,
                       edge_index=edge_indexes,
                       edge_attr=edge_attrs,
                       batch=batches)

        loss = __loss_aux(
            output=output,
            loss=criterion,
            data=data[-1],
            binary_prediction=binary_prediction)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        loss_accum.append(loss.detach().cpu().numpy())

    average_loss = np.mean(loss_accum)

    print(f"Train loss: {average_loss}")

    return average_loss, loss_accum


def __accuracy_aux(node_labels, predicted_labels, batch, device):

    mask=torch.where(node_labels>=0,1,0)
    ind=torch.arange(1,node_labels.shape[0]+1,1,dtype=torch.int32).to(mask.device)
    mask=mask*ind
    mask=mask[mask>0]-1
    node_labels_filtered=node_labels[mask]
    predicted_labels_filtered=predicted_labels[mask]
    results = torch.eq(
        predicted_labels_filtered,
        node_labels_filtered).type(
        torch.FloatTensor).to(device)

    # micro average -> mean between all nodes
    micro = torch.sum(results)

    # macro average -> mean between the mean of nodes for each graph
    macro = micro

    return micro, macro


def test(
        model,
        device,
        criterion,
        epoch,
        train_data,
        test_data1,
        binary_prediction=True):
    model.eval()

    # ----- TRAIN ------
    train_micro_avg = 0.
    train_macro_avg = 0.

    if train_data is not None:
        n_nodes = 0
        n_graphs = 0
        for data in train_data:
            for i in range(len(data)):
                data[i]=data[i].to(device)
            edge_indexes=[]
            edge_attrs=[]
            batches=[]
            for i in range(len(data)):
                edge_indexes.append(data[i].edge_index)
                edge_attrs.append(data[i].edge_attr)
                batches.append(data[i].batch)

            with torch.no_grad():
                output = model(
                    x=data[0].x,
                    edge_index=edge_indexes,
                    edge_attr=edge_attrs,
                    batch=batches)

            output = torch.sigmoid(output)
            _, predicted_labels = output.max(dim=1)

            micro, macro = __accuracy_aux(
                node_labels=data[-1].node_labels,
                predicted_labels=predicted_labels,
                batch=data[-1].batch, device=device)

            train_micro_avg += micro.cpu().numpy()
            train_macro_avg += macro.cpu().numpy()
            n_nodes += data[-1].num_nodes
            n_graphs += data[-1].num_graphs

        train_micro_avg = train_micro_avg / n_nodes
        train_macro_avg = train_macro_avg / n_graphs

    # ----- /TRAIN ------

    # ----- TEST ------
    test_micro_avg = 0.
    test_macro_avg = 0.
    test_loss = []
    test_avg_loss = 0.

    if test_data1 is not None:
        n_nodes = 0
        n_graphs = 0
        for data in test_data1:
            for i in range(len(data)):
                data[i]=data[i].to(device)
            edge_indexes=[]
            edge_attrs=[]
            batches=[]
            for i in range(len(data)):
                edge_indexes.append(data[i].edge_index)
                edge_attrs.append(data[i].edge_attr)
                batches.append(data[i].batch)


            with torch.no_grad():
                output = model(
                    x=data[0].x,
                    edge_index=edge_indexes,
                    edge_attr=edge_attrs,
                    batch=batches)

            loss = __loss_aux(
                output=output,
                loss=criterion,
                data=data[-1],
                binary_prediction=binary_prediction)

            test_loss.append(loss.detach().cpu().numpy())

            output = torch.sigmoid(output)
            _, predicted_labels = output.max(dim=1)

            micro, macro = __accuracy_aux(
                node_labels=data[-1].node_labels,
                predicted_labels=predicted_labels,
                batch=data[-1].batch, device=device)

            test_micro_avg += micro.cpu().numpy()
            test_macro_avg += macro.cpu().numpy()
            n_nodes += data[-1].num_nodes
            n_graphs += data[-1].num_graphs

        test_avg_loss = np.mean(test_loss)

        test_micro_avg = test_micro_avg / n_nodes
        test_macro_avg = test_macro_avg / n_graphs

    # ----- /TEST ------

    print(
        f"Train accuracy: micro: {train_micro_avg}\tmacro: {train_macro_avg}")
    print(f"test loss: {test_avg_loss}")
    print(f"Test accuracy: micro: {test_micro_avg}\tmacro: {test_macro_avg}")

    return (train_micro_avg, train_macro_avg), \
        (test_avg_loss, test_micro_avg, test_macro_avg)


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main(
        args,
        manual,
        train_data=None,
        test_data=None,
        n_classes=None,
        save_model=None,
        load_model=None,
        train_model=True,
        plot=None,
        truncated_fn=None):
    # set up seeds and gpu device
    seed_everything(args.seed)

    if torch.cuda.is_available():
        device = torch.device("cuda:" + str(args.device))
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        # device = torch.device("mps")
        device = torch.device("cpu")
    else:
        device = torch.device("cpu")

    print(f"##### Using Device: {device} #####")

    if not manual:
        raise NotImplementedError()

    else:
        assert train_data is not None
        assert test_data is not None
        assert n_classes is not None
        # manual settings
        print("Using preloaded data")
        train_graphs = train_data
        test_graphs = test_data

        if args.task_type == "node":
            num_classes = n_classes
        else:
            raise NotImplementedError()

    # np.random.shuffle(train_graphs)
    pin=True
    train_loader = DataLoader(
        train_graphs,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=pin,
        num_workers=0)
    test_loader = DataLoader(
        test_graphs,
        batch_size=512,
        pin_memory=pin,
        num_workers=0)

    if args.network == "acgnn":
        _model = ACGNN
    elif args.network == "acrgnn":
        _model = ACRGNN
    elif args.network == "acrgnn-single":
        _model = SingleACRGNN
    else:
        raise ValueError()

    model = _model(
        input_dim=train_graphs[0][0].num_features,
        hidden_dim=args.hidden_dim,
        output_dim=num_classes,
        num_layers=args.num_layers,
        aggregate_type=args.aggregate,
        readout_type=args.readout,
        combine_type=args.combine,
        combine_layers=args.combine_layers,
        num_mlp_layers=args.num_mlp_layers,
        task=args.task_type,
        time_range=args.time_range,
        num_relation=args.num_relation,
        truncated_fn=truncated_fn)

    if load_model is not None:
        print("Loading Model")
        model.load_state_dict(torch.load(load_model))

    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss(reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0005)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    if not args.filename == "":
        with open(args.filename, 'w') as f:
            f.write(
                "train_loss,test_loss,train_micro,train_macro,test_micro,test_macro\n")

            with open(args.filename + ".train", 'w') as f:
                f.write(
                    "train_loss\n")
            with open(args.filename + ".test", 'w') as f:
                f.write(
                    "test_loss\n")

    if train_model:
        start = time.time()
        # stop training if for stop_times_max epochs, loss changes less than stop_thresh
        stop_times_max = 5
        stop_times = 0
        stop_thresh = 5e-4
        prev_loss = float('inf')

        # `epoch` is only for printing purposes
        for epoch in range(1, args.epochs + 1):

            print(f"Epoch {epoch}/{args.epochs}")

            # TODO: binary prediction
            avg_loss, loss_iter = train(
                model=model,
                device=device,
                training_data=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                scheduler=scheduler,
                binary_prediction=True)

            (train_micro, train_macro), (test_loss, test_micro, test_macro) = test(
                model=model, device=device, train_data=train_loader, test_data1=test_loader, epoch=epoch, criterion=criterion)

            file_line = f"{avg_loss: .10f}, {test_loss: .10f}, {train_micro: .8f}, {train_macro: .8f}, {test_micro: .8f}, {test_macro: .8f}"

            if not args.filename == "":
                with open(args.filename, 'a') as f:
                    f.write(file_line + "\n")

            if not args.filename == "":
                with open(args.filename + ".train", 'a') as f:
                    for l in loss_iter:
                        f.write(f"{l: .15f}\n")

                with open(args.filename + ".test", 'a') as f:
                    f.write(f"{test_loss: .15f}\n")

            # print(f"###### {stop_times} {avg_loss-prev_loss} ######")

            if abs(avg_loss - prev_loss) < stop_thresh:
                stop_times += 1
            else:
                stop_times = 0
            if stop_times >= stop_times_max:
                break
            prev_loss = avg_loss

        if save_model is not None:
            torch.save(model.state_dict(), save_model)

        if plot is not None:
            iter_losses = np.loadtxt(args.filename + ".train", skiprows=1)
            epoch_t1_losses, epoch_t2_losses = np.loadtxt(
                args.filename + ".test", delimiter=",", skiprows=1).T

            iters = np.arange(len(iter_losses))

            batch = (len(iter_losses) / len(epoch_t1_losses))
            epochs = np.arange(len(epoch_t1_losses)) * batch + batch

            plt.figure(figsize=(16, 10))
            plt.plot(
                iters,
                iter_losses,
                color="#377eb8",
                marker="*",
                linestyle="-",
                label="Train")
            plt.plot(
                epochs,
                epoch_t1_losses,
                color="#ff7f00",
                marker="o",
                linestyle="-",
                label="test")
            plt.plot(
                epochs,
                epoch_t2_losses,
                color="#4daf4a",
                marker="x",
                linestyle="-",
                label="Tets2")

            plt.title(
                f"{plot.split('/')[-1].split('.')[0]} - H{args.hidden_dim} - B{args.batch_size} - L{args.num_layers} - Epochs{args.epochs}")

            plt.ylim(bottom=0)
            plt.legend(loc='upper right')
            plt.savefig(plot, dpi=150, bbox_inches='tight')
            plt.close()

        end = time.time()
        return file_line + f"\nTotal time: {end-start: .10f}, Avg Time per Epoch: {(end-start)/epoch: .10f}\n"

    else:

        (train_micro, train_macro), (test_loss, test_micro, test_macro) = test(
            model=model, device=device, train_data=train_loader, test_data1=test_loader, epoch=-1, criterion=criterion)

        file_line = f" {-1: .8f}, {test_loss: .10f}, {train_micro: .8f}, {train_macro: .8f}, {test_micro: .8f}, {test_macro: .8f}"

        if not args.filename == "":
            with open(args.filename, 'a') as f:
                f.write(file_line + "\n")

        return file_line + ","


if __name__ == '__main__':

    # agg, read, comb
    _networks = [
        # [{"mean": "A"}, {"mean": "A"}, {"simple": "T"}],
        # [{"mean": "A"}, {"mean": "A"}, {"mlp": "MLP"}],
        # [{"mean": "A"}, {"max": "M"}, {"simple": "T"}],
        # [{"mean": "A"}, {"max": "M"}, {"mlp": "MLP"}],
        # [{"mean": "A"}, {"add": "S"}, {"simple": "T"}],
        # [{"mean": "A"}, {"add": "S"}, {"mlp": "MLP"}],

        # [{"max": "M"}, {"mean": "A"}, {"simple": "T"}],
        # [{"max": "M"}, {"mean": "A"}, {"mlp": "MLP"}],
        # [{"max": "M"}, {"max": "M"}, {"simple": "T"}],
        # [{"max": "M"}, {"max": "M"}, {"mlp": "MLP"}],
        # [{"max": "M"}, {"add": "S"}, {"simple": "T"}],
        # [{"max": "M"}, {"add": "S"}, {"mlp": "MLP"}],

        # [{"add": "S"}, {"mean": "A"}, {"simple": "T"}],
        # [{"add": "S"}, {"mean": "A"}, {"mlp": "MLP"}],
        # [{"add": "S"}, {"max": "M"}, {"simple": "T"}],
        # [{"add": "S"}, {"max": "M"}, {"mlp": "MLP"}],
        # [{"add": "S"}, {"add": "S"}, {"simple": "T"}],
        [{"add": "S"}, {"add": "S"}, {"mlp": "MLP"}],
    ]

    comb_arr = [1]
    h_arr = [64]
    num_layers_arr = [1, 2, 3]

    file_path = "data"
    data_path = "datasets"
    extra_name = "results/"

    print("Start running")
    data_dir = '.'
    import sys
    for key in [sys.argv[1]]:
        for enum, _set in enumerate([
            [(f"{data_dir}/{key}/train-random-erdos-5000-40-50",
              f"{data_dir}/{key}/test-random-erdos-500-40-50",)
             ],
        ]):

            for index, (_train, _test) in enumerate(_set):

                print(f"Start for dataset {_train}-{_test}")

                _train_graphs, (_, _, _n_node_labels) = load_data(
                    dataset=f"{file_path}/{data_path}/{_train}.txt",
                    degree_as_node_label=False)

                _test_graphs, _ = load_data(
                    dataset=f"{file_path}/{data_path}/{_test}.txt",
                    degree_as_node_label=False)

                for _net_class in [
                    "acgnn",
                    "acrgnn",
                    "acrgnn-single"
                ]:

                    filename = f"./logging/{extra_name}{key}-{enum}-{index}.mix"

                    for a, r, c in _networks:
                        (_agg, _agg_abr) = list(a.items())[0]
                        (_read, _read_abr) = list(r.items())[0]
                        (_comb, _comb_abr) = list(c.items())[0]

                        for comb_layers in comb_arr:
                            if _comb == "mlp" and comb_layers > 1:
                                continue
                            for l in num_layers_arr:
                                for lr in [0.01]:
                                    for h in h_arr:
                                        print(a, r, c, _net_class, l, comb_layers)

                                        run_filename = f"./logging/{extra_name}{key}-{enum}-{index}-{_net_class}-agg{_agg_abr}-read{_read_abr}-comb{_comb_abr}-cl{comb_layers}-L{l}.log"
                                        time_range=int(sys.argv[2])
                                        num_relation=int(sys.argv[3])
                                        _args = argument_parser().parse_args([
                                            f"--readout={_read}",
                                            f"--aggregate={_agg}",
                                            f"--combine={_comb}",
                                            f"--network={_net_class}",
                                            f"--filename={run_filename}",
                                            "--epochs=100",
                                            f"--batch_size=128",
                                            f"--hidden_dim={h}",
                                            f"--num_layers={l}",
                                            f"--combine_layers={comb_layers}",
                                            f"--num_mlp_layers=2",
                                            "--device=0",
                                            f"--lr={lr}",
                                            f"--time_range={time_range}",
                                            f"--num_relation={num_relation}"
                                        ])

                                        line = main(
                                            _args,
                                            manual=True,
                                            train_data=_train_graphs,
                                            test_data=_test_graphs,
                                            n_classes=_n_node_labels,
                                            # save_model=f"{file_path}/saved_models/{extra_name}{key}/MODEL-{_net_class}-{enum}-agg{_agg_abr}-read{_read_abr}-comb{_comb_abr}-cl{comb_layers}-L{l}-H{h}.pth",
                                            train_model=True,
                                            # load_model=f"saved_models/h32/MODEL-{_net_class}-{key}-{enum}-agg{_agg_abr}-read{_read_abr}-comb{_comb_abr}-L{l}.pth",
                                            # plot=f"plots/{run_filename}.png",
                                            truncated_fn=None
                                        )

                                # append results per layer
                                        with open(filename, 'a') as f:
                                            f.write(_net_class+' '+str(lr)+' '+str(h)+' '+str(l)+' '+str(comb_layers)+':'+line+'\n')
