import sys
sys.path.append('.')
import torch
from Unet_organized.Unet_SA_Claude_SameRes.model_attention import create_model, count_parameters

def main():
    m = create_model(5,5,scale=1)
    x = torch.randn(1,5,64,64)
    y = m(x)
    print('ok', tuple(y.shape), count_parameters(m))

if __name__ == '__main__':
    main()

