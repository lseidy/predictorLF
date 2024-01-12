
import wandb

from DataSet import DataSet
from Params import get_args
from Trainer import Trainer

import os

def main():
    params = get_args()

    config_name = f"{params.run_name}{params.model}_{params.batch_size}_{params.lr}"

    try:
        os.mkdir(f"/home/machado/saved_LFs/{params.output_path}")
        os.mkdir(f"/home/machado/saved_LFs/{params.output_path}/validation/")

        os.mkdir(f"/home/machado/saved_models/{params.output_path}")
        if params.save_train:
            os.mkdir(f"/home/machado/saved_LFs/{params.output_path}/train/")
    except FileExistsError:
        print("Using Existent folder!!")




    dataset = DataSet(params)
    dataset.split()



    if params.wandb_active:
        wandb.init(
            # set the wandb project where this run will be logged
            project="predictorUnet",
            # track hyperparameters and run metadata
            name=config_name,
            config={
                "architecture": params.model,
                "dataset": params.dataset_name,
                "views ver": params.num_views_ver,
                "views hor": params.num_views_hor,
                "epochs": params.epochs,
                "batch size": params.batch_size,
                "learning_rate": params.lr,
                "name": config_name,
                "Training Size": len(dataset.list_train),
                "Test Size": len(dataset.list_test),
                "Train list": dataset.list_train
            }
        )



    # for lf in dataset.list_test.inner_storage:
    #     print(lf.name)


    Trainer(dataset, config_name, params)

    if params.wandb_active:
        wandb.finish()








if __name__ == '__main__':
    main()



