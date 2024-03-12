import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR
from collections import defaultdict
from tqdm import tqdm
from topmost.utils import static_utils


class BasicTrainer:
    def __init__(self, model, dataset_handler, epochs=250, learning_rate=0.002, batch_size=200, lr_scheduler=None, lr_step_size=125, log_interval=5):
        self.model = model
        self.dataset_handler = dataset_handler
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.lr_scheduler = lr_scheduler
        self.lr_step_size = lr_step_size
        self.log_interval = log_interval

    def make_optimizer(self,):
        args_dict = {
            'params': self.model.parameters(),
            'lr': self.learning_rate,
        }

        optimizer = torch.optim.Adam(**args_dict)
        return optimizer


# 用于创建一个学习率调度器（LR scheduler），它基于步骤（step）的方式来调整学习率。
# 具体来说，它创建了一个 StepLR 调度器，
# 其中 step_size 参数表示学习率调整的频率（即每经过多少个 epoch 调整一次），
# gamma 参数表示每次调整时学习率的缩放因子。

    def make_lr_scheduler(self, optimizer):
        lr_scheduler = StepLR(optimizer, step_size=self.lr_step_size, gamma=0.5, verbose=False)
        return lr_scheduler

    def train(self):
        optimizer = self.make_optimizer()

        if self.lr_scheduler:
            print("===>Warning: use lr_scheduler")
            lr_scheduler = self.make_lr_scheduler(optimizer)

        data_size = len(self.dataset_handler.train_dataloader.dataset) #NG：57

        for epoch in tqdm(range(1, self.epochs + 1)):
            # 初步理解为将模型设置为训练模式，这样模型中的一些层（如 dropout 层）会以训练模式运行，以便在训练中产生期望的效果
            self.model.train()
            #在训练过程中，每个批次的损失值会被累加到这个字典中，以便最后计算平均损失。
            loss_rst_dict = defaultdict(float)  

            for batch_data in self.dataset_handler.train_dataloader:
                #单一输出：rst_dict{'loss': tensor(846.8218, device='cuda:0', grad_fn=<AddBackward0>), 
                # 'loss_TM': tensor(834.3723, device='cuda:0', grad_fn=<AddBackward0>), 
                # 'loss_ECR': tensor(12.4495, device='cuda:0', grad_fn=<MulBackward0>)}
                rst_dict = self.model(batch_data)
                batch_loss = rst_dict['loss']

                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

                for key in rst_dict:
                    loss_rst_dict[key] += rst_dict[key] * len(batch_data)

            if self.lr_scheduler:
                lr_scheduler.step()

            if epoch % self.log_interval == 0:
                output_log = f'Epoch: {epoch:03d}'
                for key in loss_rst_dict:
                    output_log += f' {key}: {loss_rst_dict[key] / data_size :.3f}'

                print(output_log)

    def test(self, input_data):
        data_size = input_data.shape[0]
        theta = list()
        all_idx = torch.split(torch.arange(data_size), self.batch_size)

        with torch.no_grad():
            self.model.eval()
            for idx in all_idx:
                batch_input = input_data[idx]
                batch_theta = self.model.get_theta(batch_input)
                theta.extend(batch_theta.cpu().tolist())

        theta = np.asarray(theta)
        return theta

    def export_beta(self):
        beta = self.model.get_beta().detach().cpu().numpy()
        return beta

    def export_top_words(self, num_top_words=15):
        beta = self.export_beta()
        top_words = static_utils.print_topic_words(beta, vocab=self.dataset_handler.vocab, num_top_words=num_top_words)
        return top_words

    def export_theta(self):
        train_theta = self.test(self.dataset_handler.train_bow)
        test_theta = self.test(self.dataset_handler.test_bow)
        return train_theta, test_theta
