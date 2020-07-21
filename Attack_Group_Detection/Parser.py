import argparse

def Define_Params():
    parser = argparse.ArgumentParser(description= 'Attack Detection Model')
    parser.add_argument('--file_path', type=str, default='./dataset/AMAZON.txt', help='dataset path')
    parser.add_argument('--batch_size', type=int, default=512, metavar='N', help='input batch size for training')
    parser.add_argument('--embed_dim', type=int, default=32, metavar='N', help='embedding size')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate')
    parser.add_argument('--test_batch_size', type=int, default=1000, metavar='N', help='input batch size for testing')
    parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train')
    args = parser.parse_args()
    return args