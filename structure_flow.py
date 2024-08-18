import numpy as np
import torch
import os
import math
from torch.utils.data import DataLoader
from data import Dataset
from utils import Progbar, write_2images, create_dir, imsave, Logger
from models import StructureFlowModel
from itertools import islice
from skimage.measure import compare_psnr

class StructureFlow():
    def __init__(self, config):
        self.config = config
        self.debug = False
        self.flow_model = StructureFlowModel(config).to(config.DEVICE)

        self.samples_path = os.path.join(config.PATH, config.NAME, 'images')
        self.checkpoints_path = os.path.join(config.PATH, config.NAME, 'checkpoints')
        self.test_image_path = os.path.join(config.PATH, config.NAME, 'test_result')
        self.logger_path = os.path.join(config.PATH, config.NAME)
        self.writer = Logger(self.logger_path)

        if self.config.MODE == 'train' and not self.config.RESUME_ALL:
            pass
        else:
            self.flow_model.load(self.config.WHICH_ITER)

        
    def train(self):
        train_dataset = Dataset(self.config.DATA_TRAIN_GT, self.config, self.config.DATA_MASK_FILE)
        train_loader = DataLoader(dataset=train_dataset, batch_size=self.config.TRAIN_BATCH_SIZE, 
                                  shuffle=True, drop_last=True, num_workers=8)

        val_dataset = Dataset(self.config.DATA_VAL_GT,
                              self.config, self.config.DATA_MASK_FILE)
        
        sample_iterator = val_dataset.create_iterator(self.config.SAMPLE_SIZE)

        iterations = self.flow_model.iterations
        total = len(train_dataset)
        epoch = math.floor(iterations*self.config.TRAIN_BATCH_SIZE/total)
        keep_training = True

        while(keep_training):
            epoch += 1
            print('\n\nTraining epoch: %d' % epoch)

            progbar = Progbar(total, width=20, stateful_metrics=['epoch', 'iter'])

            for items in train_loader:
                inputs, gts, maps = self.cuda(*items)

                # structure model
                logs = self.flow_model.update_structure(inputs, gts, maps)
                iterations = self.flow_model.iterations

                # print(logs)
                logs = [
                    ("epoch", epoch),
                    ("iter", iterations)
                ] + logs

                progbar.add(len(inputs), values=logs if self.config.VERBOSE else [x for x in logs if not x[0].startswith('l_')])
                
                # 测试模型
                if self.config.SAMPLE_INTERVAL and iterations % self.config.SAMPLE_INTERVAL == 0:
                    items = next(sample_iterator)
                    inputs, gts, maps = self.cuda(*items)
                    result = self.flow_model.sample(inputs, gts, maps)
                    self.write_image(result, iterations, 'image')
                    
                # 计算模型
                if self.config.EVAL_INTERVAL and iterations % self.config.EVAL_INTERVAL == 0:
                    self.flow_model.eval()
                    print('\nstart eval...\n')
                    self.eval()
                    self.flow_model.train()

                # 保存模型
                if self.config.SAVE_LATEST and iterations % self.config.SAVE_LATEST == 0:
                    print('\nsaving the latest model (total_steps %d)\n' % (iterations))
                    self.flow_model.save('latest')

                if self.config.SAVE_INTERVAL and iterations % self.config.SAVE_INTERVAL == 0:
                    print('\nsaving the model of iterations %d\n' % iterations)
                    self.flow_model.save(iterations)
        print('\nEnd training....')

    def eval(self):
        # 加载数据
        val_dataset = Dataset(self.config.DATA_VAL_GT, self.config, self.config.DATA_VAL_MASK)
        val_loader = DataLoader(
            dataset=val_dataset,
            batch_size = self.config.TRAIN_BATCH_SIZE,
            shuffle=False
        )
        total = len(val_dataset)
        iterations = self.flow_model.iterations

        # 定义进度条
        progbar = Progbar(total, width=20, stateful_metrics=['it'])
        iteration = 0
        psnr_list = []

         # TODO: add fid score to evaluate
        with torch.no_grad():
            for j, items in enumerate(islice(val_loader, 50)):
                logs = []
                iteration += 1
                inputs, gts, maps = self.cuda(*items)
                outputs_structure = self.flow_model.structure_forward(inputs, maps)
                psnr = self.metrics(outputs_structure, gts)
                logs.append(('psnr', psnr.item()))
                psnr_list.append(psnr.item())

                logs = [("it", iteration), ] + logs
                progbar.add(len(inputs), values=logs)

        avg_psnr = np.average(psnr_list)
        self.writer.add_scalar("val.psnr", avg_psnr, iterations)
        self.writer.write_scalar("val.psnr", avg_psnr, iterations)
        print('model eval at iterations:%d'%iterations)
        print('average psnr:%f'%avg_psnr)

    def test(self):
        self.flow_model.eval()

        print(self.config.DATA_TEST_RESULTS)
        create_dir(self.config.DATA_TEST_RESULTS)
        test_dataset = Dataset(self.config.DATA_TEST_GT, self.config, self.config.DATA_TEST_MASK)
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=1,
        )

        index = 0
        with torch.no_grad(): 
            for items in test_loader:
                inputs, gts, maps = self.cuda(*items)

                # structure model
                outputs = self.flow_model.structure_forward(inputs, maps)
                outputs_merged = (outputs * maps) + (gts * (1 - maps))
                
                # inputs = self.postprocess(inputs)*255.0
            
                outputs_merged = self.postprocess(outputs_merged)*255.0
                
                for i in range(outputs_merged.size(0)):
                    name = test_dataset.load_name(index, self.debug)
                    # print(index, name)
                    path = os.path.join(self.config.DATA_TEST_RESULTS, name)
                    # path1 = os.path.join(self.config.DATA_TEST_RESULTS, 'name'+str(index)+'.jpg')
                    imsave(outputs_merged[i,:,:,:].unsqueeze(0), path)
                    # imsave(inputs, path1)
                    index += 1

        print('\nEnd test....')

        

    def cuda(self, *args):
        return (item.to(self.config.DEVICE) for item in args)

    def postprocess(self, x):
        x = (x + 1) / 2
        x.clamp_(0, 1)
        return x

    def metrics(self, inputs, gts):
        inputs = self.postprocess(inputs)
        gts = self.postprocess(gts)
        psnr_value = []

        inputs = (inputs*255.0).int().float()/255.0
        gts    = (gts*255.0).int().float()/255.0

        for i in range(inputs.size(0)):
            inputs_p = inputs[i,:,:,:].cpu().numpy().astype(np.float32).transpose(1,2,0)    # 取第i个batch,结果是3维
            gts_p = gts[i,:,:,:].cpu().numpy().astype(np.float32).transpose(1,2,0)
            psnr_value.append(compare_psnr(inputs_p, gts_p, data_range=1))         

        psnr_value = np.average(psnr_value)
        return psnr_value    

    def write_image(self, result, iterations, label):
        if result:
            name = '%s/model%d_sample_%08d'%(self.samples_path, self.config.MODEL, iterations) + label + '.jpg' 
            write_2images(result, self.config.SAMPLE_SIZE, name)