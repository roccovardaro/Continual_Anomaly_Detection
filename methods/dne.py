import warnings
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from .utils.base_method import BaseMethod



class DNE(BaseMethod):
    def __init__(self, args, net, optimizer, scheduler):
        super(DNE, self).__init__(args, net, optimizer, scheduler)
        self.cross_entropy = nn.CrossEntropyLoss()


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
        out, _ = self.net(inputs)
        loss = self.cross_entropy(out, labels)
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
            with warnings.catch_warnings(), np.errstate(all='ignore'):
                warnings.simplefilter("ignore")
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


