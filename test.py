#!/usr/bin/env python3

# Standard packages
import sys, os, argparse
import torch
import fastwer
import numpy as np

#from torchvision.utils import save_image
from multiprocessing import cpu_count
from tqdm import tqdm

# Local packages
import procImg
from dataset import HTRDataset, ctc_collate
from ctcdecode import CTCBeamDecoder

def test(model, htr_dataset, test_loader, device, bs=20, lmFile=None, beam=200, gsf=1.0, wip=0.0, verbosity=True):
    
    CTC_BLANK = 1
    
    if (verbosity):
        charVoc=htr_dataset.get_charVoc();
        print("models= %s %i models"%(htr_dataset.get_charVoc(),htr_dataset.get_num_classes()))
        print("CTC_BLANK= %s"%(charVoc[CTC_BLANK]))
        print('Space symbol= \"%s\"'%(htr_dataset.get_spaceChar()))
    
    # There are some specific layers/parts of the model that behave
    # differently during training and evaluation time (Dropouts,
    # BatchNorm, etc.). To turn off them during model evaluation:
    model.eval()    
    
    # Deactivate the autograd engine
    # It's not required for inference phase
    with torch.no_grad():

        # https://github.com/parlance/ctcdecode
        # In this case, parameters alpha and beta:
        #     alpha --> grammar scale factor
        #      beta --> word insertion penalty
        decoder = CTCBeamDecoder(htr_dataset.get_charVoc(),
                                model_path=lmFile,
                                alpha=gsf if lmFile else 0.0,
                                beta=wip if lmFile else 0.0,
                                beam_width=beam,
                                #cutoff_top_n=200,
                                #cutoff_prob=1.0,
                                num_processes=int(cpu_count() * 0.8),
                                blank_id=CTC_BLANK,
                                log_probs_input=True)
        

        # To store the reference and hypothesis strings
        ref, hyp = list(), list()
        
        for ((x, input_lengths),(y,target_lengths), bIdxs) in tqdm(test_loader,colour='cyan', desc='Test'):
            x = x.to(device)
            #save_image(x, f"out.png", nrow=1); break

            # Run forward pass (equivalent to: outputs = model(x))
            outputs = model.forward(x)
            # outputs ---> W,N,K    K=number of different chars + 1

            outputs = outputs.permute(1, 0, 2)
            # outputs ---> N,W,K    (BATCHSIZE, #TIMESTEPS, #LABELS)

            output, scores, ts, out_seq_len = decoder.decode(outputs.data, torch.IntTensor(input_lengths))
            # output ---> N,N_BEAMS,N_TIMESTEPS   
            #             default: N_BEAMS=100 (beam_width)
            
            # Decode outputted tensors into text lines
            assert len(output) == len(target_lengths)
           
            ptr = 0
            for i, batch in enumerate(output):
                dec, size =  batch[0], out_seq_len[i][0]
                
                yi = y[ptr:ptr+target_lengths[i]]
                refText = htr_dataset.get_decoded_label(yi)
               
                ref.append(refText)
                
                hypText = htr_dataset.get_decoded_label(dec[0:size]) if size > 0 else ''
                hyp.append(hypText)
                
                

                if (verbosity):
                    line_id = htr_dataset.items[bIdxs[i]]
                    print(f'Line-ID: {line_id}\nREF: {refText}\nHYP: {hypText}\n', file=sys.stdout)
                ptr += target_lengths[i]
            

    # Compute the total number of parameters of the trained model
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    cer = fastwer.score(hyp, ref, char_level=True)
    wer = fastwer.score(hyp, ref, char_level=False)

   
    print('\n'+'#'*30, file=sys.stderr)
    print(f'# Model\'s params: {total_params}', file=sys.stderr)
    print("             CER: {:5.2f}%".format(cer), file=sys.stderr)
    print("             WER: {:5.2f}%".format(wer), file=sys.stderr)
    print('#'*30, file=sys.stderr)

    return cer, wer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Testing a HTR model.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--lm-model', type=str, help='laguage model file (Arpa/Kenlm format)', default=None)
    parser.add_argument('--beam-width', type=int, help='this sets up how broad the beam search is', default=200)
    parser.add_argument('--gsf', type=float, help='this sets up Grammar scale factor', default=1.0)
    parser.add_argument('--wip', type=float, help='this sets up Word insertion penalty', default=0.0)
    parser.add_argument('--batch_size', type=int, help='image batch-size', default=24)
    parser.add_argument('--gpu', type=int, default=[0,1], nargs='+', help='used gpu')
    parser.add_argument("--verbosity", action="store_true",  help="increase output verbosity",default=False) 
    parser.add_argument('model', type=str, help='PyTorch model file')
    parser.add_argument('dataset', type=str, help='dataset path')

    args = parser.parse_args()
    print ("\n"+str(sys.argv) )

    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.gpu:
        if args.gpu[0] > torch.cuda.device_count() or args.gpu[0] < 0:
            sys.exit("ERROR: gpu must be in the rang of [0:%i]"%(torch.cuda.device_count()))
        torch.cuda.set_device(args.gpu[0])
        
    print("selected GPU %i"%(torch.cuda.current_device()))
          
    # Load model in memory
    state = torch.load(args.model, map_location=device)
    model = state['model']
   
    # Get the sequence of transformations to apply to images 
    img_transforms = procImg.get_tranform(state['line_height'])
          
    htr_dataset = HTRDataset(args.dataset,
                            state['spaceSymbol'],
                            transform=img_transforms,
                            charVoc=state['codec'])

    nwrks = int(cpu_count() * 0.70) # use ~2/3 of avilable cores
    test_loader = torch.utils.data.DataLoader(htr_dataset,
                                            batch_size = args.batch_size,
                                            num_workers=nwrks,
                                            pin_memory=True,
                                            shuffle = False,
                                            collate_fn = ctc_collate)

    test(model, htr_dataset, test_loader, device,
         bs=args.batch_size,
         lmFile=args.lm_model,
         beam=args.beam_width,
         gsf=args.gsf,
         wip=args.wip,
         verbosity=args.verbosity)
  
    
    sys.exit(os.EX_OK)
