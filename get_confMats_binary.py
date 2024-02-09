#!/usr/bin/env python3

# Standard packages
import sys, os, argparse
import torch
#from torchvision.utils import save_image
from multiprocessing import cpu_count
import numpy as np
from tqdm import tqdm
from termcolor import colored

# Local packages
import procImg
from dataset import HTRDataset, ctc_collate


def test(model, htr_dataset, test_loader, device, numClasses, batch_size=20, confMatrix=None, verbosity=True):
   
    # There are some specific layers/parts of the model that behave
    # differently during training and evaluation time (Dropouts,
    # BatchNorm, etc.). To turn off them during model evaluation:
    model.eval()    
    
    # Deactivate the autograd engine
    # It's not required for inference phase
    with torch.no_grad():

        f_out = open(confMatrix, 'wb')
        
        # Main batch loop
        for ((x, input_lengths),(y,target_lengths), bIdxs) in tqdm(test_loader, colour='cyan', desc='Test'):
            #To GPU (or CPU)
            x = x.to(device)
            
            #save_image(x, f"out.png", nrow=1); break

            # Run forward pass (equivalent to: outputs = model(x))
            outputs = model.forward(x) # outputs ---> T,B,L    K=number of different chars + 1
           
            outputs = outputs.permute(1, 0, 2)  # outputs ---> B,T,L    (BATCHSIZE, #TIMESTEPS, #LABELS)

            for i, batch in enumerate(input_lengths):
                line_id = htr_dataset.items[bIdxs[i]]                
                f_out.write(bytes(line_id+" ", encoding='utf8'))
                f_out.write(b"\x00B")
                
                rows=outputs.size(dim=1).to_bytes(length=4, byteorder=sys.byteorder)
                cols=outputs.size(dim=2).to_bytes(length=4, byteorder=sys.byteorder)
                f_out.write("FM ".encode('utf8') + b"\x04" + rows + b"\x04" + cols)
                
                line = np.array([],np.float32)
                for temps in range(outputs.size(dim=1)):
                    a = np.array(outputs[i][temps].cpu(),'float32')
                    line=np.append(line,a)
              
                line.tofile(f_out)
    f_out.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='gets matrix confidence for a HTR model on a given datset.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--gpu', type=int, default=[0,1], nargs='+', help='gpu to be used')
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
        
    print("\nselected GPU %i\n"%(torch.cuda.current_device()))

    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
