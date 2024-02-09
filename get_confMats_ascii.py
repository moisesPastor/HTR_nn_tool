#!/usr/bin/env python3

# Standard packages
import sys, os, argparse
import torch
#from torchvision.utils import save_image
from multiprocessing import cpu_count

from tqdm import tqdm
from termcolor import colored

# Local packages
import procImg
from dataset import HTRDataset, ctc_collate


def test(model, htr_dataset, test_loader, device, numClasses, batch_size=20, confMatrix=None, verbosity=True):
    # https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
    # https://pytorch.org/docs/stable/data.html
    # DataLoader wraps an iterable around the Dataset to enable easy 
    # access to the samples. It typically pass samples in "minibatches",
    # reshuffle the data at every epoch, and use Python’s multiprocessing 
    # to speed up data retrieval.

    # For more details about how to set up the "num_workers" parameter, see at:
    # https://chtalhaanwar.medium.com/pytorch-num-workers-a-tip-for-speedy-training-ed127d825db7#:~:text=Num_workers%20tells%20the%20data%20loader,the%20GPU%20has%20to%20wait.


    # There are some specific layers/parts of the model that behave
    # differently during training and evaluation time (Dropouts,
    # BatchNorm, etc.). To turn off them during model evaluation:
    model.eval()    
    
    # Deactivate the autograd engine
    # It's not required for inference phase
    with torch.no_grad():

        f_out = open(confMatrix, 'w')
        
        # Main Mini-batch loop
        for ((x, input_lengths),(y,target_lengths), bIdxs) in tqdm(test_loader, colour='cyan', desc='Test'):
            #To GPU (or CPU)
            x = x.to(device)
            #save_image(x, f"out.png", nrow=1); break

            # Run forward pass (equivalent to: outputs = model(x))
            outputs = model.forward(x)
            # outputs ---> W,N,K    K=number of different chars + 1
            outputs = outputs.permute(1, 0, 2)
            # outputs ---> N,W,K    (BATCHSIZE, #TIMESTEPS, #LABELS)

            for i, batch in enumerate(input_lengths):
                line_id = htr_dataset.items[bIdxs[i]]
                f_out.write(line_id+" [\n")
                
                for temps in range(outputs.size(dim=1)):
                    line=""
                    for char_prob in range(outputs.size(dim=2)):
                         line = line + "{: 5.5f} ".format(outputs[i][temps][char_prob])
                    if temps < outputs.size(dim=1) - 1:
                        f_out.write(line+"\n")  
                    else: 
                        f_out.write(line);


                f_out.write("]\n")
    f_out.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Testing a HTR model on a given datset.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--gpu', type=int, default=[0,1], nargs='+', help='used gpu')
    parser.add_argument('--batch_size', type=int, help='image batch-size', default=24)
    parser.add_argument('--conf_matrix', type=str, help='fileConfMat', default="ConfMat.txt")
    parser.add_argument('model', type=str, help='PyTorch model file')
    parser.add_argument('dataset', type=str, help='dataset path')

    args = parser.parse_args()
    print ("\n"+str(sys.argv))
    
    if args.gpu:
        if args.gpu[0] > torch.cuda.device_count() or args.gpu[0] < 0:
            sys.exit("ERROR: gpu must be in the rang of [0:%i]"%(torch.cuda.device_count()))
        torch.cuda.set_device(args.gpu[0])
        
    print("selected GPU %i"%(torch.cuda.current_device()))

    # Check if CUDA is available
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # Load model in memory
    state = torch.load(args.model, map_location=device)
    model = state['model']
   
    # Get the sequence of transformations to apply to images 
    img_transforms = procImg.get_tranform(state['line_height'])

    htr_dataset = HTRDataset(args.dataset,
                            state['spaceSymbol'],
                            transform=img_transforms,
                            charVoc=state['codec'])

    numClasses = htr_dataset.get_num_classes() 
    nwrks = int(cpu_count() * 0.70) # use ~2/3 of avilable cores
    test_loader = torch.utils.data.DataLoader(htr_dataset,
                                            batch_size = args.batch_size,
                                            num_workers=nwrks,
                                            pin_memory=True,
                                            shuffle = False,
                                            collate_fn = ctc_collate)

    print("models= %s %i models"%(htr_dataset.get_charVoc(),htr_dataset.get_num_classes()))
    print('Space symbol= \"%s\"'%(htr_dataset.get_spaceChar()))  

    test(model, htr_dataset, test_loader, device,
            batch_size=args.batch_size,
            numClasses=numClasses,
            confMatrix=args.conf_matrix, verbosity=True)

    sys.exit(os.EX_OK)
