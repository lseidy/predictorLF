from argparse import Namespace

import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import save_image
import wandb
from PIL import Image
from torch.utils.data import DataLoader
from skimage.measure import shannon_entropy

from DataSet import DataSet, LensletBlockedReferencer

import LightField as LF

from customLearningRateScaler import CustomExpLr as lrScaler

from scipy.linalg import hadamard


class CustomMSELoss(nn.Module):
    def __init__(self):
        super(CustomMSELoss, self).__init__()

    def hadamard_transform(self, block):
        hadamard_transform = torch.from_numpy(hadamard(block.shape[-1])).to(block.device, dtype=torch.float32)
        return torch.matmul(hadamard_transform, block)


    def satd_loss(self, original, pred):
        transformed_original = self.hadamard_transform(original)
        transformed_pred = self.hadamard_transform(pred)
        return torch.sum(torch.abs(transformed_original - transformed_pred))


    def forward(self, original, pred):
        return self.satd_loss(original, pred)
    



class Trainer:

   

    def __init__(self, dataset: DataSet, config_name: str, params: Namespace):
        self.model_name = params.model
        # TODO make loss GREAT AGAIN, nope, make it a param.

        self.params = params
        self.predictor_size_v = self.params.num_views_ver * self.params.predictor_size
        self.predictor_size_h = self.params.num_views_hor * self.params.predictor_size
        self.dataset = dataset
        self.best_loss = 1000.1
        if self.params.loss == 'mse':
            self.loss = nn.MSELoss()
            print("Using MSE")
        elif self.params.loss == 'satd':
            self.loss = CustomMSELoss()
            print("Using SATD")
        else:
            print("Unknown Loss Metric")
            exit(404)


        torch.manual_seed(42)
        torch.cuda.manual_seed(42)


        # TODO REMOVE
        self.count_blocks = 0

        # TODO after everything else is done, adapt for other models
        self.model = ModelOracle(params.model).get_model(config_name, params)


        if params.resume != '':
                    #TODO FINISH RESUMING TRAINING
                    try:
                        checkpoint = torch.load(params.resume, map_location=torch.device('cuda'))
                        self.model.load_state_dict(checkpoint)
                    except RuntimeError:
                        print("Failed to resume model")


        # TODO make AMERICA GREAT AGAIN, nope.... Num works be a parameter too
        # TODO test prefetch_factor and num_workers to optimize
        self.train_set = DataLoader(dataset.list_train, shuffle=True, num_workers=8,
                                    pin_memory=True, prefetch_factor=2)
        self.val_set = DataLoader(dataset.list_test, shuffle=False, num_workers=8,
                                  pin_memory=True)
        self.test_set = DataLoader(dataset.list_test, shuffle=False, num_workers=8,
                                   pin_memory=True)

        if torch.cuda.is_available():
            self.model.cuda()
            device = torch.device("cuda")
        else:
            print("Running on CPU!")
            device = torch.device("cpu")

        self.loss = self.loss.to(device)

        
        
        self.optimizer = torch.optim.Adam(
        self.model.parameters(), lr=params.lr, betas=(0.9, 0.999))


        if params.lr_scheduler == 'custom':
            self.scheduler = lrScaler(optimizer=self.optimizer, initial_learning_rate=params.lr,
                                  decay_steps=params.epochs, decay_rate=params.lr_gamma)
            print("Using Custom Scheduler")
       

        


        for epoch in range(params.resume_epoch, 1 + params.epochs):
            #0 for validation off
            loss, entropy = self.train(epoch, 0, params.wandb_active)
            print(f"Epoch {epoch}: {loss}, {entropy}")

            if params.wandb_active:
                wandb.log({f"Epoch": epoch},commit=False)
                wandb.log({f"Loss_epoch": loss}, commit=False)
                wandb.log({f"Entropy_epoch": entropy}, commit=False)

            # if epoch == 5:#1 to signalize that the validations is On
            loss, entropy = self.train(epoch, 1, params.wandb_active)
            print(f"Validation: {loss}, {entropy}")

            if self.params.lr_scheduler == 'custom':
                self.scheduler.step()

            if params.wandb_active:
                wandb.log({f"Loss_VAL_epoch": loss}, commit=False)
                wandb.log({f"Entropy_VAL_epoch": entropy})

            check = {
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            }



            if loss < self.best_loss:
                torch.save(check, f"/home/machado/saved_models/{params.output_path}/bestMSE_{config_name}.pth.tar")
                self.best_loss = loss


            torch.save(check, f"/home/machado/saved_models/{params.output_path}/{config_name}_{epoch}.pth.tar")

    def train(self, current_epoch, val, wandb_active):

        acc = 0
        acc_entropy = 0
        batches_now = 0
        if val == 0:
            self.model.train()
            set = self.train_set
        else:
            # TODO validation set
            set = self.test_set
            self.model.eval()
        resol_ver = self.params.resol_ver
        resol_hor = self.params.resol_hor
        pred_size = self.params.predictor_size * self.params.num_views_hor


        for i, data in enumerate(set):

            it_i = 0
            it_j = 0
            output_lf = torch.zeros((1, resol_ver, resol_hor))

            self.count_blocks = 0


            # possible TODO: make MI_Size take a tuple
            referencer = LensletBlockedReferencer(data, MI_size=self.params.num_views_ver,
                                                  predictor_size=self.params.predictor_size,context_size=self.params.context_size)
            loader = DataLoader(referencer, batch_size=self.params.batch_size)

            for neighborhood, actual_block in loader:
                current_batch_size = actual_block.shape[0]
                if torch.cuda.is_available():
                    neighborhood, actual_block = (neighborhood.cuda(), actual_block.cuda())
                predicted = self.model(neighborhood)
                predicted = predicted[:, :, -self.predictor_size_v:, -self.predictor_size_h:]
                #print(predicted)
                

                #actual_block = actual_block[:, :, -self.predictor_size_v:, -self.predictor_size_h:]
                if (val == 1) or (self.params.save_train == True):
                    cpu_pred = predicted.cpu().detach()
                    cpu_orig = actual_block.cpu().detach()
                    cpu_ref = neighborhood.cpu().detach()

                    # print(cpu_pred)
                    # print(cpu_orig)
                    # print(LF.denormalize_image(cpu_orig, 8))
                    # print(LF.denormalize_image(cpu_pred, 8))


                    for bs_sample in range(0, cpu_pred.shape[0]):
                        try:
                            block_pred = cpu_pred[bs_sample]
                            block_orig = cpu_orig[bs_sample]
                            block_ref = cpu_ref[bs_sample]
                        except IndexError as e:
                            print("counts ", it_i, it_j)
                            print(block_pred.shape)
                            print(cpu_pred.shape)
                            print(e)
                            exit()

                        if self.count_blocks < 10 and (current_epoch == 1):
                            save_image(block_pred, f"/home/machado/blocks_tests/{self.count_blocks}_predicted.png")
                            save_image(block_orig, f"/home/machado/blocks_tests/{self.count_blocks}_original.png")
                            save_image(block_ref, f"/home/machado/blocks_tests/{self.count_blocks}_reference.png")
                        self.count_blocks += 1

                        try:
                            output_lf[:, it_j:it_j + self.predictor_size_v, it_i:it_i + self.predictor_size_h] = block_pred
                        except RuntimeError as e:
                            print("counts error", it_i, it_j)
                            print(e)
                            exit()



                        it_j += self.predictor_size_v
                        if it_j >= resol_ver - self.predictor_size_v-1:
                            it_j = 0
                            it_i += self.predictor_size_h


                        if it_i > resol_hor - self.predictor_size_h-1 and it_j == 0:
                            # print("counts save", it_j, it_i)
                            if val == 0:
                                save_image(output_lf, f"/home/machado/saved_LFs/{self.params.output_path}/train/allBlocks_{i}.png")
                            elif val == 1:
                                save_image(output_lf, f"/home/machado/saved_LFs/{self.params.output_path}/validation/allBlocks_{i}_{current_epoch}.png")


                # predicted = predicted*255
                # actual_block = actual_block*255

                # loss = self.loss(predicted, actual_block)
                # cpu_pred=LF.denormalize_image(cpu_pred, self.params.bit_depth)
                # cpu_orig=LF.denormalize_image(cpu_orig, self.params.bit_depth)

                #print(predicted.shape, actual_block.shape)
                loss = self.loss(predicted, actual_block)

                
                #print(predicted.squeeze().shape)
                res = (torch.abs(LF.denormalize_image(predicted.cpu().detach(),8)-LF.denormalize_image(actual_block.cpu().detach(), 8))).int()
                #print(res.int())
                result_entropy=0
                count_batchs=0
                for batch in res:
                    count_batchs+=1
                    #print(batch)
                    result_entropy += shannon_entropy(batch)
                result_entropy = result_entropy/count_batchs
                #print("result final:", result_entropy)
                




                if val == 0:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    

                # loss = Mean over batches... so we weight the loss by the batch
                loss = loss.cpu().item()
                acc += loss * current_batch_size
                acc_entropy += result_entropy * count_batchs
                batches_now += current_batch_size
                # if wandb_active:
                #     if val == 0:
                #         wandb.log({f"Batch_MSE_era_{current_epoch}": loss})
                #         wandb.log({f"Batch_MSE_global": loss})
                #     else:
                #         wandb.log({f"Batch_MSE_VAL_global_{current_epoch}": loss})

        return acc/batches_now, acc_entropy/batches_now


class ModelOracle:
    def __init__(self, model_name):
        if model_name == 'Unet3k':
            from Models.gabriele_k3 import UNetSpace
            # talvez faça mais sentido sò passar as variaveis necessarias do dataset
            print("gabri_like")
            self.model = UNetSpace
        elif model_name == 'Unet4k':
            from Models.gabriele_k4 import UNetSpace
            self.model = UNetSpace
            print("keras_like")
        else:
            print("Model not Found.")
            exit(404)

    def get_model(self, config_name, params):

       
        try:
            return self.model(config_name, params)

        except RuntimeError as e:
            print("Failed to import model: ", e)
