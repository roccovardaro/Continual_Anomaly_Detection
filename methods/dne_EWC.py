import warnings
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from .utils.base_method import BaseMethod
import copy

class DNE_EWC(BaseMethod):
    def __init__(self, args, net, optimizer, scheduler):
        super(DNE_EWC, self).__init__(args, net, optimizer, scheduler)
        self.loss_fn = nn.CrossEntropyLoss()

        # EWC (Paper: Kirkpatrick et al., PNAS 2017)
        # fisher= matrice di Fisher; optar= parametri ottimali
        # Lista di dizionari {fisher, optpar} — uno per ogni task completato.
        # Nel paper, la penalizzazione somma su TUTTI i task passati,
        # quindi serve mantenere una Fisher separata per ogni task.
        self.ewc_tasks = []
        self.ewc_lambda = getattr(args.train, 'ewc_lambda', 5000.0)


    def forward(self, epoch, inputs, labels, one_epoch_embeds, t, *args):
        if self.args.dataset.strong_augmentation:
            half_num = int(len(inputs) / 2)
            no_strongaug_inputs = inputs[:half_num]
        else:
            no_strongaug_inputs = inputs

        if self.args.model.fix_head:
            if t >= 1: #Congelamento della Testa Questo è il passaggio fondamentale per evitare il catastrophic forgetting
                for param in self.net.head.parameters():
                    param.requires_grad = False

        self.optimizer.zero_grad()
        with torch.no_grad():
            #CALCOLA GLI EMBEDDING DEL MODELLO SENZA AGGIORNARE I PESI
            noaug_embeds = self.net.forward_features(no_strongaug_inputs)
            #GLI EMBEDDING VENGONO SALVATI IN ONE_EPOCH_EMBEDS PER AGGIORNARE LA DENSITA' PIU TARDI
            one_epoch_embeds.append(noaug_embeds.cpu())
        
        # L'Addestramento Vero e Proprio (se task > 1 e head è fixato, non aggiorna la testa ma solo il backbone)
        out, current_embeds = self.net(inputs)
        loss = self.loss_fn(out, labels)
        
        # EWC LOSS (Paper: Eq. 3, Kirkpatrick et al. 2017)
        # L_total = L_B(θ) + Σ_{k} (λ/2) * Σ_i F_{k,i} * (θ_i - θ*_{k,i})^2
        # Somma la penalizzazione su TUTTI i task passati, non solo l'ultimo.
        if len(self.ewc_tasks) > 0 and t >= 1:
            ewc_loss = torch.tensor(0.0, device=inputs.device)
            for task_data in self.ewc_tasks:
                fisher = task_data['fisher']
                optpar = task_data['optpar']
                for name, param in self.net.named_parameters():
                    if param.requires_grad and name in fisher:
                        ewc_loss = ewc_loss + (fisher[name] * (param - optpar[name]).pow(2)).sum()
            
            loss += (self.ewc_lambda / 2.0) * ewc_loss
            
        loss.backward()
        self.optimizer.step()
        if self.scheduler:
            self.scheduler.step(epoch)


    def training_epoch(self, density, one_epoch_embeds, task_wise_mean, task_wise_cov, task_wise_train_data_nums, t):
        if self.args.eval.eval_classifier == 'density':

            #PRIMA PARTE
            #serve a calcolare le statistiche media e varianza per il task corrente
            one_epoch_embeds = torch.cat(one_epoch_embeds)
            one_epoch_embeds = F.normalize(one_epoch_embeds, p=2, dim=1)
            mean, cov = density.fit(one_epoch_embeds) #STIMA MEDIA E COVARIANZA DEGLI EMBEDDING

            #Verifichiamo se abbiamo gia delle statistiche per questo task
            if len(task_wise_mean) < t + 1:# se non ci sono le aggiorna
                task_wise_mean.append(mean)
                task_wise_cov.append(cov)
            else:#altrimenti le sovrascrive
                task_wise_mean[-1] = mean
                task_wise_cov[-1] = cov

            #SECONDA PARTE
            # Questa parte serve a preparare il modello per la fase di test (inferenza)
            task_wise_embeds = []
            for i in range(t + 1):
                if i < t: # task passati
                    past_mean, past_cov, past_nums = task_wise_mean[i], task_wise_cov[i], task_wise_train_data_nums[i]
                    past_embeds = np.random.multivariate_normal(past_mean, past_cov, size=int(past_nums * (1 - self.args.noise_ratio))) # genera vettori fittizi (dati ricostruiti puliti)
                    task_wise_embeds.append(torch.FloatTensor(past_embeds))
                    noise_mean, noise_cov = np.random.rand(past_mean.shape[0]), np.random.rand(past_cov.shape[0], past_cov.shape[1])
                    noise = np.random.multivariate_normal(noise_mean, noise_cov, size=int(past_nums * self.args.noise_ratio)) #genera vettori fittizi casuali (rumore)
                    task_wise_embeds.append(torch.FloatTensor(noise))
                else:
                    task_wise_embeds.append(one_epoch_embeds)
            for_eval_embeds = torch.cat(task_wise_embeds, dim=0)
            for_eval_embeds = F.normalize(for_eval_embeds, p=2, dim=1)
            _, _ = density.fit(for_eval_embeds)
            return density
        else:
            pass

        # CREAZIONE DI UNA DENSITA BASATA SU TUTTI I VETTORI RICOSTRUTI DA OGNI DISTRIBUZIONE UNA PER OGNI TASK

    def end_task(self, train_dataloader):
        # Salva una copia congelata del modello alla fine del task per eventuali altri usi
        self.old_net = copy.deepcopy(self.net)
        self.old_net.eval()
        for param in self.old_net.parameters():
            param.requires_grad = False
            
        # CALCOLO DELLA MATRICE DI FISHER (EWC - Paper: Kirkpatrick et al. 2017)
        # A differenza della versione precedente dove self.fisher veniva sovrascritta,
        # qui calcoliamo la Fisher per questo task e la AGGIUNGIAMO alla lista.
        # Così la penalizzazione nel forward può sommare su TUTTI i task passati.
        task_fisher = {}
        task_optpar = {}
        
        device = next(self.net.parameters()).device

        # Salva i parametri ottimali θ* di questo task
        for name, param in self.net.named_parameters():
            if param.requires_grad:
                task_optpar[name] = param.data.clone().detach()
                task_fisher[name] = torch.zeros_like(param.data)
                
        self.net.eval()

        #LOOP SU DATASET
        for batch_idx, data in enumerate(train_dataloader):
            if isinstance(data, list):
                inputs = [x.to(device) for x in data]
                labels = torch.arange(len(inputs), device=device)
                labels = labels.repeat_interleave(inputs[0].size(0))
                inputs = torch.cat(inputs, dim=0)
            else:
                inputs = data.to(device)
                labels = torch.zeros(inputs.size(0), device=device).long()

            #FORWARD + BACKWARD
            # - CALCOLA OUT
            # - CALCOLA LOSS
            # FA BACKWARD per ottenere i gradienti
            self.optimizer.zero_grad()
            out, _ = self.net(inputs)
            loss = self.loss_fn(out, labels)
            loss.backward()

            # cuore del calcolo della Fisher Information
            for name, param in self.net.named_parameters():
                # param.requires_grad esclude parametri congelati (es. head bloccata)
                # param.grad is not None il gradiente è stato calcolato evita errori su layer non coinvolti nel forward
                if param.requires_grad and param.grad is not None:
                    # param.grad.data = ∂θ/∂L
                    task_fisher[name] += param.grad.data.pow(2) / len(train_dataloader)
                    
        # AGGIUNGI alla lista dei task (NON sovrascrivere)
        self.ewc_tasks.append({
            'fisher': task_fisher,
            'optpar': task_optpar
        })
                    
        self.net.train()
