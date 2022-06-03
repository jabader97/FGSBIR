import torch
import time
from model import FGSBIR_Model
from dataset import get_dataloader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from options import Options
import os


if __name__ == "__main__":
    hp = Options().parse()
    dataloader_Train, dataloader_Test = get_dataloader(hp)
    print(hp)
    if hp.savename == 'group_plus_seed':
        if hp.log_online:
            hp.savename = hp.group + '_s{}'.format(hp.seed)
        else:
            hp.savename = ''

    model = FGSBIR_Model(hp)
    model.to(device)
    # model.load_state_dict(torch.load('VGG_ShoeV2_model_best.pth', map_location=device))
    step_count, top1, top10 = -1, 0, 0
    torch.manual_seed(hp.seed)

    if hp.log_online:
        import wandb
        _ = os.system('wandb login {}'.format(hp.wandb_key))
        os.environ['WANDB_API_KEY'] = hp.wandb_key
        save_path = os.path.join(hp.path_aux, 'CheckPoints', 'wandb')
        wandb.init(project=hp.project, group=hp.group, name=hp.savename, dir=save_path,
                   settings=wandb.Settings(start_method='fork'))
        wandb.config.update(vars(hp))

    for i_epoch in range(hp.max_epoch):
        for batch_data in dataloader_Train:
            step_count = step_count + 1
            start = time.time()
            model.train()
            loss = model.train_model(batch=batch_data)

            if step_count % hp.print_freq_iter == 0:
                print('Epoch: {}, Iteration: {}, Loss: {:.5f}, Top1_Accuracy: {:.5f}, Top10_Accuracy: {:.5f}, Time: {}'.format
                      (i_epoch, step_count, loss, top1, top10, time.time()-start))

            if step_count % hp.eval_freq_iter == 0:
                with torch.no_grad():
                    top1_eval, top10_eval = model.evaluate(dataloader_Test)
                    print('results : ', top1_eval, ' / ', top10_eval)

                if top1_eval > top1:
                    torch.save(model.state_dict(), hp.backbone_name + '_' + hp.dataset_name + '_model_best.pth')
                    top1, top10 = top1_eval, top10_eval
                    print('Model Updated')
        if hp.log_online:
            valid_data = {'top1_eval': top1_eval, 'top10_eval': top10_eval}
            wandb.log(valid_data)
