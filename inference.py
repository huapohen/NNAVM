import torch
import warnings


warnings.filterwarnings("ignore")

''' add this line, because GTX30 serials' default torch.matmul() on cuda is uncorrected '''
torch.backends.cuda.matmul.allow_tf32 = False