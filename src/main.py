#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from dataloader.loader import MNIST,CriticalMNIST
from models.lenet import LeNet5
from redishelper.redishelper import RedisHelper

import sys
import time
import argparse


parser = argparse.ArgumentParser(description="Parse argument for training GoSGD with PyTorch")

# arguments for training data
parser.add_argument("-d","--dataset", default=".", help="Training/test data directory")
parser.add_argument("--classes",default=10,type=int,help="The number of classification")
parser.add_argument("--labels",default="0-9",help="The select labels")

# model related
parser.add_argument("--model",default="lenet5",help="The name of neural networks",choices=["lenet5","alexnet"])

# optimizer related
parser.add_argument("-b","--batchsize",default=64,type=int,help="Batch size of a training batch at each iteration")
parser.add_argument("--optim",default="sgd",help="The optimization algorithms", choices=["sgd","adam"])
parser.add_argument("--lr",default=0.001,type=float,help="Learning rate of optimization algorithms")
parser.add_argument("--epoch",default=100,type=int,help="The number of trainng episode")
parser.add_argument("--iteration",default=10000,type=int,help="The number of training iteration")

# Redis related
parser.add_argument("--edgenum",default=1,type=int,help="The number of edge")
parser.add_argument("--host",default="localhost",help="The ip of Redis server")
parser.add_argument("--port",default=6379,type=int,help="The port which Redis server listen to")

# Non-critical remove related
parser.add_argument("--noncriticalremove",action="store_true",help="if remove non-critical training samples")
parser.add_argument("--strategy",default="fixed",help="The strategy of identify non-critical samples",
choices=["fixed","mean","sampler"])
parser.add_argument("--fixedratio",default=0.5,type=float,help="The ratio of selected critical samples")

# tensorboard
parser.add_argument("--tensorboard",action="store_true",help="if user tensorboard to show training process")
parser.add_argument("--summary",default=".",help="The path of summary")

# GPU
parser.add_argument("--gpu",action="store_true",help="if use gpu for training")

def main(args,*k,**kw):
    # if use gpus
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")
    print("user device: {}".format(device))

    # redis helper related
    redis_helper = RedisHelper(host=args.host,port=args.port)
    redis_helper.signin()
    while redis_helper.cur_edge_num() < args.edgenum:
        time.sleep(1) # sleep 1 second

    model_score = 1.0 / args.edgenum # the initial model parameters score
    
    # log_file and summary path

    log_file = "{}-lenet-edge-{}.log".format(time.strftime('%Y%m%d-%H%M%S',time.localtime(time.time())),redis_helper.ID)
    log_dir = "tbruns/lenet-mnist-edge-{}".format(redis_helper.ID)

    logger = open(log_file,'w')
    swriter = SummaryWriter(log_dir)

    # load traing data
    trainset = MNIST(root=args.dataset, train=True, download=False, transform=None)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batchsize, shuffle=True, num_workers=0)

    testset = MNIST(root=args.dataset, train=False, download=False, transform=None)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batchsize, shuffle=False, num_workers=0)

    # construct neural network
    lenet = LeNet5()
    lenet.to(device)

    # define optimizer
    criterion = nn.CrossEntropyLoss()
    criterion_loss = nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.SGD(lenet.parameters(), lr=args.lr, momentum=0.9)

    SIZE = (1,28,28)
    # start training
    for epoch in range(0,args.epoch,1):
        iteration = 0
        # merge parameters of other edge
        if epoch > 0:
            mintime,maxtime,param_list = redis_helper.min2max_time_params()
            print("The min/max time cost of last epoch: {}/{}".format(mintime,maxtime))
            for item in param_list:
                w1 = model_score / (model_score + item[0])
                w2 = item[0] / (model_score + item[0])
                
                for local,other in zip(lenet.parameters(),item[1]):
                    ldev = local.get_device()
                    rdev = other.get_device()
                    if ldev < 0:
                        local.data = local.data * w1 + other.data.cpu() * w2
                    else:
                        local.data = local.data * w1 + other.data.cuda() * w2
                model_score = model_score + item[0]
            
            while redis_helper.finish_update() == False:
                time.sleep(1.0)



        critical_extra_start = time.time()
        # non-critical samples identigy and removal
        train_data = trainset.train_data
        train_labels = trainset.train_labels
        # data: n x zip x features
        # labels: n x zip
        if args.noncriticalremove == False:
            ndata = train_data[:,1:,:]
            ndata = ndata.reshape(-1,1,28,28)
            nlabels = train_labels[:,1:]
            nlabels = nlabels.reshape(-1,1)

            critrainset = CriticalMNIST(datax=ndata,targetsx=nlabels)
        else:
            cri_data = train_data[:,0,:] # n x feature
            cri_data = cri_data.reshape(-1,1,28,28)
            cri_label = train_labels[:,0] # n
            cri_lable = cri_label.reshape(-1,1)

            # calculate the loss of each aggregated points
            loss_info = []
            with torch.no_grad():
                offsetl = 0
                while offsetl < len(cri_data):
                    endl = offsetl + args.batchsize
                    images = cri_data[offsetl:endl]
                    labels = cri_label[offsetl:endl].squeeze()

                    # move data and label to `device`
                    images = images.to(device)
                    labels = labels.to(device)

                    outputs = lenet(images)
                    loss = criterion_loss(outputs, labels)

                    for ix in range(len(loss)):
                        loss_info.append((offsetl+ix,loss[ix]))

                    offsetl += args.batchsize # move to next batch


                #for idx in range(0,len(cri_data),1):
                    # insert a dimension at 0
                    #images = cri_data[idx].unsqueeze(0)
                    #labels = cri_label[idx].unsqueeze(0)

                    # move data and label to `device`
                    #images = images.to(device)
                    #labels = labels.to(device)

                    #outputs = lenet(images)
                    #loss = criterion(outputs, labels)

                   # loss_info.append((idx,loss.item()))
            
            # sort loss info from large to small according loss
            loss_info = sorted(loss_info,key=lambda x: x[1],reverse=True)
            sel_index = []
            offset = int(args.fixedratio * len(loss_info) + 0.5 )
            for i in range(0,offset):
                sel_index.append(loss_info[i][0])
            sel_index = sorted(sel_index)
            print("{} select {} critical agg points".format(redis_helper.NAME,len(sel_index)))
            
            #cri_data = cri_data[sel_index]
            #cri_label = cri_label[sel_index]

            ndata = train_data[sel_index][:,1:,:]
            ndata = ndata.reshape(-1,1,28,28)
            nlabels = train_labels[sel_index][:,1:]
            nlabels = nlabels.reshape(-1,1)

            critrainset = CriticalMNIST(datax=ndata,targetsx=nlabels)

        # The data loader of critical samples
        critrainloader = torch.utils.data.DataLoader(critrainset, batch_size=args.batchsize, shuffle=True, num_workers=0)

        critical_extra_cost = time.time() - critical_extra_start
        training_start = time.time()

        running_loss = 0.0
        record_running_loss = 0.0
        for i, data in enumerate(critrainloader, 0):
            iteration += 1
            # get the inputs
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.squeeze().to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = lenet(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            record_running_loss += loss.item()
            if i % 10 == 9:
                swriter.add_scalar("training loss",record_running_loss / 10,epoch*len(critrainloader)+i)
                record_running_loss = 0.0

            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

        training_cost = time.time() - training_start

        # push time and parameters to Redis
        model_score = model_score / 2
        sel_edge_id = redis_helper.random_edge_id(can_be_self=True)
        redis_helper.ins_time_params(sel_edge_id,training_cost,model_score,list(lenet.parameters()))
        while redis_helper.finish_push() == False:
            time.sleep(1.0)
        
        # test on test data
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images = images.reshape(-1,1,28,28).to(device)
                labels = labels.squeeze().to(device)

                outputs = lenet(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        _header="[ {} Epoch {} /Iteration {} ]".format(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())),epoch+1,iteration)
        print('{} Accuracy of the network on the 10000 test images: {} %%'.format(_header,100 * correct / total))
        logger.write('{} Accuracy {} %%\n'.format(_header,100 * correct / total))

        swriter.add_scalar("accuracy", 100 * correct / total, epoch)

    print('Finished Training')

    redis_helper.register_out()
    logger.close() # close log file writer

    return lenet

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
