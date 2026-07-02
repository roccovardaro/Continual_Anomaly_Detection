import os
import torch
import time
import csv
from tqdm import tqdm

from argument import get_args
from datasets import get_dataloaders
from eval import eval_model
from methods import get_model
from models import get_net_optimizer_scheduler
from utils.density import GaussianDensityTorch
import warnings
import logging
warnings.filterwarnings("ignore")
logging.getLogger("codecarbon").disabled = True

#task_wise_mean: è semplicemente una lista che tiene traccia della media degli embedding per ciascun task nel contesto del continual learning.
#task_wise_cov: contiene la matrice di covarianza degli embedding di ogni task.
# task_wise_train_data_nums: contiene il numero di dati di training per ogni task.

def get_inputs_labels(data):
    if isinstance(data, list):
        inputs = [x.to(args.device) for x in data]
        labels = torch.arange(len(inputs), device=args.device)
        labels = labels.repeat_interleave(inputs[0].size(0)) #ripete ogni etichetta tante volte quanto il numero di righe (batch size) del primo tensore della lista.
        inputs = torch.cat(inputs, dim=0)
    else:
        inputs = data.to(args.device)
        labels = torch.zeros(inputs.size(0), device=args.device).long()
    return inputs, labels
    
def main(args):
    
    # Track the start time of training
    start_time = time.time()

    #PREPARAZIONE DEL MODELLO
    net, optimizer, scheduler = get_net_optimizer_scheduler(args)
    density = GaussianDensityTorch()
    net.to(args.device)

    model = get_model(args, net, optimizer, scheduler)

    dataloaders_train, dataloaders_test, learned_tasks, all_test_filenames = [], [], [], []
    task_wise_mean, task_wise_cov, task_wise_train_data_nums = [], [], []
    history_auc = []
    
    # List to store timing statistics
    timing_stats = []

    for t in range(args.dataset.n_tasks): #CICLO SUI TASK
        print('---' * 10, f'Task:{t}', '---' * 10)
        train_dataloader, dataloaders_train, dataloaders_test, learned_tasks, data_train_nums, all_test_filenames = get_dataloaders(args, t, dataloaders_train, dataloaders_test, learned_tasks, all_test_filenames)
        task_wise_train_data_nums.append(data_train_nums)

        extra_para = None
        if args.model.method == 'panda':
            extra_para = model.get_center(train_dataloader)

        net.train()
        task_train_time = 0

        for epoch in tqdm(range(args.train.num_epochs)):
            epoch_start_time = time.time()
            print('---' * 2, f'Epoch:{epoch}', '---' * 2)
            one_epoch_embeds = [] # RACCOGLIE GLI EMBEDDING GENERATI DURANTE L'EPOCA
            if args.model.method == 'upper':
                for dataloader_train in dataloaders_train:
                    for batch_idx, (data) in enumerate(dataloader_train):
                        inputs, labels = get_inputs_labels(data)
                        model(epoch, inputs, labels, one_epoch_embeds, t, extra_para)
            else:
                print('Inizio Addestramento')
                for batch_idx, (data) in enumerate(train_dataloader):
                    inputs, labels = get_inputs_labels(data)

                    model(epoch, inputs, labels, one_epoch_embeds, t, extra_para) # va ad aggiornare direttamente le liste
                print('Fine Addestramento')

            # Calculate and store training time for this epoch before any evaluation begins
            epoch_train_time = time.time() - epoch_start_time
            task_train_time += epoch_train_time
            timing_stats.append({"Task": t, "Epoch": epoch, "Time_Type": "Epoch_Train", "Duration_Seconds": round(epoch_train_time, 2)})

            #SE La valutazione periodica è attiva, aggiorna lo stimatore densità e valuta il modello
            if args.train.test_epochs > 0 and (epoch+1) % args.train.test_epochs == 0:
                print("Inizio Valutazione")
                net.eval()
                density = model.training_epoch(density, one_epoch_embeds, task_wise_mean, task_wise_cov, task_wise_train_data_nums, t)
                last_auc = eval_model(args, epoch, dataloaders_test, learned_tasks, net, density)
                print("Fine Valutazione")
                net.train()

        if hasattr(model, 'end_task'):
            end_task_start = time.time()
            model.end_task(train_dataloader)
            task_train_time += time.time() - end_task_start
            
        timing_stats.append({"Task": t, "Epoch": "N/A", "Time_Type": "Task_Train_Total", "Duration_Seconds": round(task_train_time, 2)})

        if args.train.test_epochs > 0:
            history_auc.append(last_auc)

    if args.save_checkpoint:
        torch.save(net,  f'{args.save_path}/net.pth')
        torch.save(density, f'{args.save_path}/density.pth')

    # Save timing stats to CSV
    csv_file = 'training_times.csv'
    try:
        with open(csv_file, mode='w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["Task", "Epoch", "Time_Type", "Duration_Seconds"])
            writer.writeheader()
            writer.writerows(timing_stats)
        print(f"\n[+] Statistiche sui tempi salvate in {csv_file}")
    except Exception as e:
        print(f"\n[-] Errore nel salvataggio del CSV: {e}")
    
    # Calculate total training time
    total_time = time.time() - start_time
    hours, rem = divmod(total_time, 3600)
    minutes, seconds = divmod(rem, 60)

    T = args.dataset.n_tasks
    if args.train.test_epochs > 0 and len(history_auc) == T:
        print("\n" + "="*40)
        print("Continual Learning Metrics (Final):")
        
        if T > 0 and len(history_auc[T-1]) > 0:
            final_aucs = history_auc[T-1]
            ACC_auc = sum(final_aucs) / len(final_aucs)
            print(f"Final Average Accuracy (ACC): {ACC_auc * 100:.2f}%")
            
            if T > 1:
                forgetting_sum_auc = 0
                for i in range(T - 1):
                    max_auc = max(history_auc[tau][i] for tau in range(i, T))
                    forgetting_sum_auc += (max_auc - final_aucs[i])
                FM_auc = forgetting_sum_auc / (T - 1)
                print(f"Final Forgetting Measure (FM): {FM_auc * 100:.2f}%")
                
        print("="*40 + "\n")


if __name__ == "__main__":
    #os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    args = get_args()
    main(args)
