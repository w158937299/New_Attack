import argparse

def Define_Params():
    parser = argparse.ArgumentParser(description= 'Attack Detection Model')
    parser.add_argument('--file_path', type=str, default='./dataset/AMAZON.txt', help='dataset path')
    parser.add_argument('--batch_size', type=int, default=512, metavar='N', help='input batch size for training')
    parser.add_argument('--embed_dim', type=int, default=128, metavar='N', help='embedding size')
    parser.add_argument('--lr', type=float, default=0.002, metavar='LR', help='learning rate')
    parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train')
    parser.add_argument('--UUedge_weight', type=str, default= './edges/UUedge.txt')
    parser.add_argument('--UIedge_weight', type=str, default= './edges/UIedge.txt')
    parser.add_argument('--format_UUedge_weight', type=str, default= './edges/format_UUedge.txt')
    parser.add_argument('--uu_neigh', type=str, default= './edges/uu_neigh.txt')
    parser.add_argument('--ui_neigh', type=str, default= './edges/ui_neigh.txt')
    parser.add_argument('--iu_neigh', type=str, default= './edges/iu_neigh.txt')
    parser.add_argument('--windows_len', type=int, default= '5')
    parser.add_argument('--het_random_walk_fliepath', type=str, default= './edges/het_random_walk.txt')
    parser.add_argument('--start_node_embedding', type=str, default= './edges/start_node_embedding.txt')
    parser.add_argument('--new_node_embedding', type=str, default= './edges/new_node_embedding.txt')
    parser.add_argument('--OJLD', type=str, default= './edges/OJLD_Similarity_matrix.txt')
    parser.add_argument('--MHD', type=str, default= './edges/MHD_Similarity_matrix.txt')
    parser.add_argument('--COS', type=str, default= './edges/COS_Similarity_matrix.txt')
    parser.add_argument('--u_l', type=int, default= '5055')
    parser.add_argument('--i_l', type=int, default= '17610')
    parser.add_argument('--walk_length', type=int, default= '30')
    parser.add_argument('--iter_n', type=int, default= '3600')

    args = parser.parse_args()
    return args