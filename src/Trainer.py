from argparse import Namespace

import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import save_image
import wandb
from PIL import Image
from torch.utils.data import DataLoader

from DataSet import DataSet, LensletBlockedReferencer

import LightField as LF

class Trainer:

    def __init__(self, dataset: DataSet, config_name: str, params: Namespace):
        self.model_name = params.model
        # TODO make loss GREAT AGAIN, nope, make it a param.
        self.loss = nn.MSELoss()
        self.params = params
        self.predictor_size_v = self.params.num_views_ver * self.params.predictor_size
        self.predictor_size_h = self.params.num_views_hor * self.params.predictor_size
        self.dataset = dataset
        self.best_loss = 1000.1

        torch.manual_seed(42)
        torch.cuda.manual_seed(42)


        # TODO REMOVE
        self.count_blocks = 0

        # TODO after everything else is done, adapt for other models
        self.model = ModelOracle(params.model).get_model(config_name, params)



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

        # TODO check betas
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=params.lr, betas=(0.9, 0.999))

        for epoch in range(params.resume_epoch, 1 + params.epochs):
            loss = self.train(epoch, 0, params.wandb_active)
            print(f"Epoch {epoch}: {loss}")

            if params.wandb_active:
                wandb.log({f"Epoch": epoch})
                wandb.log({f"MSE_epoch": loss})

            # if epoch == 5:
            loss = self.train(epoch, 1, params.wandb_active)
            print(f"Validation: {loss}")

            if params.wandb_active:
                wandb.log({f"MSE_VAL_epoch": loss})

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
                                                  predictor_size=self.params.predictor_size)
            loader = DataLoader(referencer, batch_size=self.params.batch_size)

            for neighborhood, actual_block in loader:
                current_batch_size = actual_block.shape[0]
                if torch.cuda.is_available():
                    neighborhood, actual_block = (neighborhood.cuda(), actual_block.cuda())
                predicted = self.model(neighborhood)
                predicted = predicted[:, :, -self.predictor_size_v:, -self.predictor_size_h:]
                actual_block = actual_block[:, :, -self.predictor_size_v:, -self.predictor_size_h:]

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

                        # if self.count_blocks < 500 and (current_epoch == 1 or current_epoch == 14):
                        #     save_image(block_pred, f"/home/machado/blocks_tests/{self.count_blocks}_predicted.png")
                        #     save_image(block_orig, f"/home/machado/blocks_tests/{self.count_blocks}_original.png")
                        #     save_image(block_ref, f"/home/machado/blocks_tests/{self.count_blocks}_reference.png")
                        # self.count_blocks += 1

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

                loss = self.loss(predicted, actual_block)

                if val == 0:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                # loss = Mean over batches... so we weight the loss by the batch
                loss = loss.cpu().item()
                acc += loss * current_batch_size
                batches_now += current_batch_size
                # if wandb_active:
                #     if val == 0:
                #         wandb.log({f"Batch_MSE_era_{current_epoch}": loss})
                #         wandb.log({f"Batch_MSE_global": loss})
                #     else:
                #         wandb.log({f"Batch_MSE_VAL_global_{current_epoch}": loss})


        return acc / batches_now


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

        if params.resume != '':
            #TODO FINISH RESUMING TRAINING
            try:
                self.model(config_name, params)
                self.model.load_state_dict(torch.load(params.resume))

                return
            except RuntimeError:
                print("Failed to resume model")
        try:
            return self.model(config_name, params)

        except RuntimeError as e:
            print("Failed to import model: ", e)
