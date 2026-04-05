import torch
import torch.nn as nn
import os
import sys
from scipy import stats
from tqdm import tqdm
import numpy as np
from PIL import Image
from models import InceptionI3d,DAE
from dataloader import load_image_train,load_image,VideoDataset,get_dataloaders
from config import get_parser
from util import get_logger,log_and_print,loss_function,loss_function_v2
from transformers import BertModel,BertTokenizer
from moe import MOE
from text import text_prompt

sys.path.append('../')
torch.backends.cudnn.enabled = True
i3d_pretrained_path = './rgb_i3d_pretrained.pt'
feature_dim = 1024



class Join_model(nn.Module):
    def __init__(self, ):
        super(Join_model, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained('./bert/')
        self.bert = BertModel.from_pretrained("./bert/")
        self.mseloss = nn.MSELoss()
        self.moe = MOE(input_dim=768, num_experts=2, hidden_dim=768)
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=768,  
                                                            nhead=8,  
                                                            dim_feedforward=2048 
                                                            )
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_layer, num_layers=1)
        self.gate=nn.Linear(768,1)
        self.layer1 = nn.Linear(1024, 768)
        self.layer2 = nn.Linear(768, 1024)

    def forward(self, clip_feats, text):
        text_input = text_prompt(text)
        text_feature = self.bert(**text_input)[0]
        text_feats = text_feature.mean(1)

        clip_feats = self.layer1(clip_feats.mean(1))

        gate_value = torch.sigmoid(self.gate(text_feats))
        modulated_clip_feats = clip_feats * gate_value

        modulated_feats = self.transformer_encoder(modulated_clip_feats.unsqueeze(1)).squeeze(1)
        joint_feats = self.layer2(modulated_feats)

        text_modulation = self.moe(text_feats) 
        loss = self.mseloss(text_modulation, clip_feats)  

        return joint_feats, loss




if __name__ == '__main__':

    args = get_parser().parse_known_args()[0]

    if not os.path.exists('./exp'):
        os.mkdir('./exp')
    if not os.path.exists('./ckpts'):
        os.mkdir('./ckpts')

    base_logger = get_logger(f'exp/DAE+LVFL.log', args.log_info)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    i3d = InceptionI3d().cuda()
    i3d.load_state_dict(torch.load(i3d_pretrained_path))
    join_model = Join_model().cuda()
    dae = DAE().cuda()
    dataloaders = get_dataloaders(args)

    optimizer = torch.optim.Adam([*i3d.parameters()] + [*join_model.parameters()] + [*dae.parameters()],
                                 lr=args.lr, weight_decay=args.weight_decay)

    epoch_best = 0
    rho_best = 0
    for epoch in range(args.num_epochs):
        log_and_print(base_logger, f'Epoch: {epoch}')

        for split in ['train', 'test']:
            true_scores = []
            pred_scores = []
            sigma = []

            if split == 'train':
                i3d.eval()
                join_model.train()
                dae.train()
                torch.set_grad_enabled(True)
            else:
                i3d.eval()
                join_model.eval()
                dae.eval()
                torch.set_grad_enabled(False)

            for data in tqdm(dataloaders[split]):
                true_scores.extend(data['final_score'].numpy())
                videos = data['video'].cuda()
                videos.transpose_(1, 2)  

                batch_size, C, frames, H, W = videos.shape
                clip_feats = torch.empty(batch_size, 10, feature_dim).cuda()
                for i in range(9):
                    clip_feats[:, i] = i3d(videos[:, :, 10 * i:10 * i + 16, :, :]).squeeze(2)
                clip_feats[:, 9] = i3d(videos[:, :, -16:, :, :]).squeeze(2)

                text = data["text"]
                join_feats, mseloss = join_model(clip_feats, text)
                preds, mu, sigmas = dae(join_feats)
                preds = preds.view(-1)
                sigmas = sigmas.view(-1)
                mu = mu.view(-1)               
                pred_scores.extend([i.item() for i in preds])
                sigma.extend([i.item()**2 for i in sigmas])

                if split == 'train':
                    loss1 = loss_function(preds, data['final_score'].float().cuda(), mu)
                    loss = loss1 +  mseloss
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            rho, p = stats.spearmanr(pred_scores, true_scores)
            pred_scores = np.array(pred_scores)
            true_scores = np.array(true_scores)
            RL2 = 100 * np.power((pred_scores - true_scores) / (true_scores.max() - true_scores.min()), 2).sum() / \
                  true_scores.shape[0]

            log_and_print(base_logger, f'{split} correlation: {rho} R-ℓ2: {RL2}')

        if rho > rho_best:
            rho_best = rho
            epoch_best = epoch
            log_and_print(base_logger, '##### New best correlation #####')
            log_and_print(base_logger, f'*******************pred_scores：{pred_scores}')
            log_and_print(base_logger, f'*******************true_scores：{true_scores}')

            path = './ckpts/' + str(rho) + '.pt'
            torch.save({'epoch': epoch,
                        'i3d': i3d.state_dict(),
                        'dae': dae.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'rho_best': rho_best}, path)
