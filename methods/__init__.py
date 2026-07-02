from .cutpaste import CutPaste
from .dne_EWC import DNE_EWC
from .dne_EWC2 import DNE_EWC2
from .dne_hybrid import DNE_Hybrid
from .csflow import CSFlow
from .panda import PANDA
from .revdis import RevDis
from .der import DER
from .er import ER
from .derpp import DERpp
from .fdr import FDR
from .agem import AGEM


def get_model(args, net, optimizer, scheduler):
    if args.model.method == 'dne_ewc':
        model = DNE_EWC(args, net, optimizer, scheduler)
        args.dataset.strong_augmentation = True
    elif args.model.method == 'dne_ewc2':
        model = DNE_EWC2(args, net, optimizer, scheduler)
        args.dataset.strong_augmentation = True
    elif args.model.method == 'dne_hybrid':
        model = DNE_Hybrid(args, net, optimizer, scheduler)
        args.dataset.strong_augmentation = True
    elif args.model.method == 'upper':
        model = CutPaste(args, net, optimizer, scheduler)
    elif args.model.method == 'cutpaste':
        model = CutPaste(args, net, optimizer, scheduler)
        args.dataset.strong_augmentation = True
        args.dataset.strong_augmentation = True
    elif args.model.method == 'csflow':
        model = CSFlow(args, net, optimizer, scheduler)
        args.dataset.strong_augmentation = False
    elif args.model.method == 'panda':
        model = PANDA(args, net, optimizer, scheduler)
        args.dataset.strong_augmentation = False
    elif args.model.method == 'revdis':
        model = RevDis(args, net, optimizer, scheduler)
        args.dataset.strong_augmentation = False
    elif args.model.method == 'er':
        model = ER(args, net, optimizer, scheduler)
    elif args.model.method == 'der':
        model = DER(args, net, optimizer, scheduler)
    elif args.model.method == 'derpp':
        model = DERpp(args, net, optimizer, scheduler)
    elif args.model.method == 'fdr':
        model = FDR(args, net, optimizer, scheduler)
    elif args.model.method == 'agem':
        model = AGEM(args, net, optimizer, scheduler)
    return model