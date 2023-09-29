import math
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
def cosine_lr(opt, base_lr, e, epochs):
    lr = 0.5 * base_lr * (math.cos(math.pi * e / epochs) + 1)
    for param_group in opt.param_groups:
        param_group["lr"] = lr
    return lr

def constant(opt, base_lr, e, epochs):
    return base_lr

def train(batch, model, criterion, optimizer):

    # switch to train mode
    model.train()

    points_x, points_y = batch
    points_x, points_y = (points_x.float()).cuda(), (points_y.float()).cuda()

    # compute output
    outputs_y = model(points_x)
#     print('train!!!!!!',points_x.size(), outputs_y.size(), points_y.size())
    loss = criterion(outputs_y, points_y)

    # compute gradient and do SGD step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_item = loss.item()
    return loss_item

def test(batch, model, criterion):
    ####################################################### Xin Wang's code ############################################
    # # check DNN model by R2 score
    # x_check = x_test[:1024 * 1024, :]
    # y_pred = model.predict(x_check)
    # y_target = y_test[:1024 * 1024, :]
    #
    # dQ_R2 = R2score(y_target[:, :30], y_pred[:, :30])
    # dT_R2 = R2score(y_target[:, 30:60], y_pred[:, 30:60])
    # print('dQ R2: %.6f \ndT R2: %.6f ' % (dQ_R2, dT_R2))
    #
    # print(R2score(y_target[:, 60], y_pred[:, 60]))
    # print(R2score(y_target[:, 61], y_pred[:, 61]))
    # print(R2score(y_target[:, 62], y_pred[:, 62]))
    # print(R2score(y_target[:, 63], y_pred[:, 63]))
    # print(R2score(y_target[:, 64], y_pred[:, 64]))
    ####################################################################################################################

    # switch to eval mode
    model.eval()
    with torch.no_grad():
        points_x, points_y = batch
        points_x, points_y = (points_x.float()).cuda(), (points_y.float()).cuda()

        # compute output
        outputs_y = model(points_x)
        loss0_30 = criterion(outputs_y[:,:30], points_y[:,:30])
        loss30_60 = criterion(outputs_y[:,30:60], points_y[:,30:60])
        loss60 = criterion(outputs_y[:, 60:61], points_y[:, 60:61])
        loss61 = criterion(outputs_y[:, 61:62], points_y[:, 61:62])
        loss62 = criterion(outputs_y[:, 62:63], points_y[:, 62:63])
        loss63 = criterion(outputs_y[:, 63:64], points_y[:, 63:64])
        loss64 = criterion(outputs_y[:, 64:65], points_y[:, 64:65])

    return loss0_30.item(), loss30_60.item(), loss60.item(), loss61.item(), loss62.item(), loss63.item(), loss64.item()

def test_de(batch, model, criterion):
    ####################################################### Xin Wang's code ############################################
    # # check DNN model by R2 score
    # x_check = x_test[:1024 * 1024, :]
    # y_pred = model.predict(x_check)
    # y_target = y_test[:1024 * 1024, :]
    #
    # dQ_R2 = R2score(y_target[:, :30], y_pred[:, :30])
    # dT_R2 = R2score(y_target[:, 30:60], y_pred[:, 30:60])
    # print('dQ R2: %.6f \ndT R2: %.6f ' % (dQ_R2, dT_R2))
    #
    # print(R2score(y_target[:, 60], y_pred[:, 60]))
    # print(R2score(y_target[:, 61], y_pred[:, 61]))
    # print(R2score(y_target[:, 62], y_pred[:, 62]))
    # print(R2score(y_target[:, 63], y_pred[:, 63]))
    # print(R2score(y_target[:, 64], y_pred[:, 64]))
    ####################################################################################################################

    # switch to eval mode
    model.eval()
    with torch.no_grad():
        points_x, points_y = batch
        # print(points_x.size(), points_y.size())
        points_x, points_y = (points_x.float()).cuda(), (points_y.float()).cuda()

        # compute output
        outputs_y = model(points_x)
        
#         print('####max',torch.max(points_y[:,1:2,:,:]))
#         print('####min',torch.min(points_y[:,1:2,:,:]))
#         print('####truemax',torch.max(outputs_y[:,1:2,:,:]))
#         print('####truemin',torch.min(outputs_y[:,1:2,:,:]))
#         print('####truemean',torch.mean(outputs_y[:,1:2,:,:]))
#         np.savez('debug_data',out = points_y.cpu().numpy(), true = outputs_y.cpu().numpy())
#         plt.imshow(points_y[0][0].cpu().numpy())
#         plt.savefig('points_y.jpg')
#         plt.show()
#         plt.imshow(outputs_y[0][0].cpu().numpy())
#         plt.savefig('outputs_y.jpg')
#         plt.show()
    
    
        loss = criterion(outputs_y, points_y)
        loss_chidao = criterion(outputs_y[:,:,36:60,:], points_y[:,:,36:60,:])
#         print(outputs_y.size(), points_y.size())
#         print('loss',loss)

    return loss.item(), loss_chidao.item()

def test_de_61_64(batch, model, criterion):
    ####################################################### Xin Wang's code ############################################
    # # check DNN model by R2 score
    # x_check = x_test[:1024 * 1024, :]
    # y_pred = model.predict(x_check)
    # y_target = y_test[:1024 * 1024, :]
    #
    # dQ_R2 = R2score(y_target[:, :30], y_pred[:, :30])
    # dT_R2 = R2score(y_target[:, 30:60], y_pred[:, 30:60])
    # print('dQ R2: %.6f \ndT R2: %.6f ' % (dQ_R2, dT_R2))
    #
    # print(R2score(y_target[:, 60], y_pred[:, 60]))
    # print(R2score(y_target[:, 61], y_pred[:, 61]))
    # print(R2score(y_target[:, 62], y_pred[:, 62]))
    # print(R2score(y_target[:, 63], y_pred[:, 63]))
    # print(R2score(y_target[:, 64], y_pred[:, 64]))
    ####################################################################################################################

    # switch to eval mode
    model.eval()
    with torch.no_grad():
        points_x, points_y = batch
        points_x, points_y = (points_x.float()).cuda(), (points_y.float()).cuda()

        # compute output
        outputs_y = model(points_x)
        loss = criterion(outputs_y, points_y)

        loss61 = criterion(outputs_y[:, 0:1], points_y[:, 0:1])
        loss62 = criterion(outputs_y[:, 1:2], points_y[:, 1:2])
        loss63 = criterion(outputs_y[:, 2:3], points_y[:, 2:3])
        loss64 = criterion(outputs_y[:, 3:4], points_y[:, 3:4])

    return loss61.item(), loss62.item(), loss63.item(), loss64.item()

def save_checkpoint(state, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)