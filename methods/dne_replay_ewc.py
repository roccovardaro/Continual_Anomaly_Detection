import warnings
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from .utils.base_method import BaseMethod
import copy
from torch.utils.data import DataLoader, Subset
from datasets.transforms import no_aug_transformation


class DNE_Replay_EWC(BaseMethod):
    """
    DNE con Image Replay Buffer + EWC (Kirkpatrick et al., PNAS 2017).
    """

    def __init__(self, args, net, optimizer, scheduler):
        super(DNE_Replay_EWC, self).__init__(args, net, optimizer, scheduler)
        self.loss_fn = nn.CrossEntropyLoss()

        # EWC (Paper: Kirkpatrick et al., PNAS 2017)
        # fisher= matrice di Fisher; optpar= parametri ottimali
        # Lista di dizionari {fisher, optpar} — uno per ogni task completato.

        self.ewc_tasks = []
        self.ewc_lambda = getattr(args.train, 'ewc_lambda', 5000.0)

        # Percentuale di immagini da salvare per replay nella stima di densità
        self.real_embed_ratio = getattr(args.train, 'real_embed_ratio', 0.05)

        # Buffer di immagini raw salvate alla fine di ogni task (lista di tensori CPU)
        self.past_replay_images = []

    def forward(self, epoch, inputs, labels, one_epoch_embeds, t, *args):
        if self.args.dataset.strong_augmentation:
            half_num = int(len(inputs) / 2)
            no_strongaug_inputs = inputs[:half_num]
        else:
            no_strongaug_inputs = inputs

        if self.args.model.fix_head:
            if t >= 1: #Congelamento della Testa
                for param in self.net.head.parameters():
                    param.requires_grad = False

        self.optimizer.zero_grad()
        with torch.no_grad():
            #CALCOLA GLI EMBEDDING DEL MODELLO SENZA AGGIORNARE I PESI
            noaug_embeds = self.net.forward_features(no_strongaug_inputs)
            #GLI EMBEDDING VENGONO SALVATI IN ONE_EPOCH_EMBEDS PER AGGIORNARE LA DENSITA' PIU TARDI
            one_epoch_embeds.append(noaug_embeds.cpu())
        
        # L'Addestramento Vero e Proprio
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
            # IBRIDO: per i task passati usa un mix di embedding reali + sintetici + rumore
            with warnings.catch_warnings(), np.errstate(all='ignore'):
                warnings.simplefilter("ignore")
                device = next(self.net.parameters()).device
                task_wise_embeds = []
                for i in range(t + 1):
                    if i < t: # task passati
                        past_mean, past_cov, past_nums = task_wise_mean[i], task_wise_cov[i], task_wise_train_data_nums[i]
                        
                        # Calcola il numero di embedding per ogni tipo
                        n_noise = int(past_nums * self.args.noise_ratio)
                        
                        # 1. Embedding REALI ricalcolati on-the-fly con la backbone CORRENTE
                        #    Questo elimina il problema degli embedding stale
                        n_saved_real = 0
                        if i < len(self.past_replay_images) and self.past_replay_images[i].numel() > 0:
                            replay_imgs = self.past_replay_images[i]
                            with torch.no_grad():
                                real_embeds = self.net.forward_features(replay_imgs.to(device))
                                real_embeds = F.normalize(real_embeds, p=2, dim=1).cpu()
                            n_saved_real = real_embeds.size(0)
                            task_wise_embeds.append(real_embeds)

                        n_synth = max(0, past_nums - n_noise - n_saved_real)

                        # 2. Embedding SINTETICI dalla distribuzione gaussiana
                        if n_synth > 0:
                            past_embeds = np.random.multivariate_normal(past_mean, past_cov, size=n_synth)
                            task_wise_embeds.append(torch.FloatTensor(past_embeds))

                        # 3. Rumore casuale (invariato, controllato da noise_ratio)
                        if n_noise > 0:
                            noise_mean = np.random.rand(past_mean.shape[0])
                            noise_cov = np.random.rand(past_cov.shape[0], past_cov.shape[1])
                            noise = np.random.multivariate_normal(noise_mean, noise_cov, size=n_noise)
                            task_wise_embeds.append(torch.FloatTensor(noise))
                    else:
                        task_wise_embeds.append(one_epoch_embeds)
                for_eval_embeds = torch.cat(task_wise_embeds, dim=0)
                for_eval_embeds = F.normalize(for_eval_embeds, p=2, dim=1)
                _, _ = density.fit(for_eval_embeds)
            return density
        else:
            pass


    def end_task(self, train_dataloader):
        # Salva una copia congelata del modello alla fine del task per eventuali altri usi
        self.old_net = copy.deepcopy(self.net)
        self.old_net.eval()
        for param in self.old_net.parameters():
            param.requires_grad = False

        # Salva un buffer di immagini raw per replay nei task futuri
        self._save_replay_images(train_dataloader)
            
        # CALCOLO DELLA MATRICE DI FISHER (EWC - Paper: Kirkpatrick et al. 2017)
        # Calcoliamo la Fisher per questo task e la AGGIUNGIAMO alla lista.
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

        # LOOP SU DATASET
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
                    task_fisher[name] += param.grad.data.pow(2) / len(train_dataloader)
                    
        # AGGIUNGE alla lista dei task
        self.ewc_tasks.append({
            'fisher': task_fisher,
            'optpar': task_optpar
        })
                    
        self.net.train()

    def _save_replay_images(self, train_dataloader):
        """
        Salva un buffer di immagini raw (senza augmentation) alla fine del task.
        """

        dataset = train_dataloader.dataset
        total_samples = len(dataset)
        
        # Calcola quante immagini salvare
        n_real = int(total_samples * self.real_embed_ratio)
        if n_real <= 0:
            print(f"[DNE_Replay_EWC] real_embed_ratio troppo basso, nessuna immagine salvata per il task {len(self.past_replay_images)}")
            self.past_replay_images.append(torch.empty(0))
            return

        # 1. Campiona gli indici
        indices = torch.randperm(total_samples)[:n_real].tolist()

        # 2. Crea un subset con solo le immagini campionate
        subset = Subset(dataset, indices)

        # 3. Salva la transform originale e sostituisci con no_aug
        #    per ottenere immagini pulite (senza augmentation)
        original_transform = dataset.transform
        dataset.transform = no_aug_transformation(self.args)
        
        # 4. Crea un dataloader temporaneo per il subset campionato
        sample_loader = DataLoader(
            subset,
            batch_size=train_dataloader.batch_size,
            shuffle=False,
            num_workers=0
        )

        # 5. Raccoglie le immagini raw
        sampled_images = []
        for data in sample_loader:
            if isinstance(data, list):
                imgs = torch.cat(data, dim=0)
            else:
                imgs = data
            sampled_images.append(imgs.cpu())

        # 6. Ripristina la transform originale
        dataset.transform = original_transform

        sampled_images = torch.cat(sampled_images, dim=0)
            
        self.past_replay_images.append(sampled_images)
        print(f"[DNE_Replay_EWC] Salvate {sampled_images.size(0)} immagini replay "
              f"({sampled_images.shape}, {sampled_images.nelement() * sampled_images.element_size() / 1024:.0f} KB) "
              f"per il task {len(self.past_replay_images)-1}")
