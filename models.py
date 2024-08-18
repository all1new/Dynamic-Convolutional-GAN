import torch
import torch.nn as nn
from base_model import BaseModel
from networks import StructureGen, MultiDiscriminator, Basic_dcd, StructureGen_condconv, StructureGen_dcd, StructureGen_dcd_v2, StructureGen_dcd_v3, StructureGen_drconv
from loss import AdversarialLoss, PerceptualLoss, StyleLoss, SmoothLoss

class StructureFlowModel(BaseModel):
    def __init__(self, config):
        super(StructureFlowModel, self).__init__(config)
        # 初始化参数
        self.config = config
        self.net_name = ['s_gen', 's_dis', 'f_gen', 'f_dis']

        self.structure_param = {'input_dim':3, 'dim':64, 'n_res':4, 'activ':'relu', 
                         'norm':'in', 'pad_type':'reflect', 'use_sn':True}
        self.dis_param = {'input_dim':3, 'dim':64, 'n_layers':3, 
                         'norm':'none', 'activ':'lrelu', 'pad_type':'reflect', 'use_sn':True}

        # 初始化损失函数
        l1_loss = nn.L1Loss()
        adversarial_loss = AdversarialLoss(type=config.DIS_GAN_LOSS)
        vgg_style = StyleLoss()
        vgg_content = PerceptualLoss()
        # smooth_loss = SmoothLoss(weight=config.REGULARIZE_WEIGHT)
        
        self.add_module('l1_loss', l1_loss)
        self.add_module('adversarial_loss', adversarial_loss)
        self.add_module('vgg_style', vgg_style)
        self.add_module('vgg_content', vgg_content)
        # self.add_module('smooth_loss', smooth_loss)
        

        # 初始化网络模型
        self.build_model()
    
    def build_model(self):
        self.iterations = 0
        # structure model
        # self.s_gen = StructureGen(**self.structure_param)
        # self.s_gen = StructureGen_dcd(**self.structure_param)
        self.s_gen = StructureGen_dcd_v3(**self.structure_param)
        # self.s_gen = StructureGen_drconv(**self.structure_param)
        # self.s_gen = StructureGen_condconv(**self.structure_param)
        self.s_gen = nn.DataParallel(self.s_gen)
        self.s_dis = MultiDiscriminator(**self.dis_param)
        self.s_dis = nn.DataParallel(self.s_dis)

        # 定义优化函数的学习率并初始化
        self.define_optimizer()
        self.init()

    def structure_forward(self, inputs, maps):
        outputs = self.s_gen(torch.cat((inputs, maps), dim=1))
        return outputs

    # 架构生成网络有两个损失函数没有使用
    def update_structure(self, inputs, gts, maps):
        self.iterations += 1

        self.s_gen.zero_grad()
        self.s_dis.zero_grad()
        outputs = self.structure_forward(inputs, maps)

        # 鉴别器损失
        dis_loss = 0
        dis_fake_input = outputs.detach()
        dis_real_input = gts
        fake_labels = self.s_dis(dis_fake_input)
        real_labels = self.s_dis(dis_real_input)
        for i in range(len(fake_labels)):
            dis_real_loss = self.adversarial_loss(real_labels[i], True, True)
            dis_fake_loss = self.adversarial_loss(fake_labels[i], False, True)
            dis_loss += (dis_real_loss + dis_fake_loss) / 2
        # 鉴别器的对抗损失
        self.structure_adv_dis_loss = dis_loss / len(fake_labels)

        self.structure_adv_dis_loss.backward()
        self.s_dis_opt.step()
        if self.s_dis_scheduler is not None:
            self.s_dis_scheduler.step()

        dis_gen_loss = 0
        # 生成模型的输出结果输入鉴别器中
        fake_labels = self.s_dis(outputs)
        for i in range(len(fake_labels)):
            dis_fake_loss = self.adversarial_loss(fake_labels[i], True, False)  # 感觉最后那位是True还是False都不打紧
            dis_gen_loss += dis_fake_loss

        self.structure_adv_gen_loss = dis_gen_loss / len(fake_labels) * self.config.STRUCTURE_ADV_GEN
        self.structure_l1_loss = self.l1_loss(outputs, gts) * self.config.STRUCTURE_L1

        self.vgg_loss_style = self.vgg_style(outputs * maps, gts * maps) * self.config.VGG_STYLE
        self.vgg_loss_content = self.vgg_content(outputs, gts) * self.config.VGG_CONTENT
        self.vgg_loss = self.vgg_loss_style + self.vgg_loss_content

        # 生成器的对抗损失
        self.structure_gen_loss = self.structure_l1_loss + self.structure_adv_gen_loss + self.vgg_loss
        
        self.structure_gen_loss.backward()
        self.s_gen_opt.step()
        if self.s_gen_scheduler is not None:
            self.s_gen_scheduler.step()
            

        # 统计损失函数
        logs = [
            ("l_s_adv_dis", self.structure_adv_dis_loss.item()),
            ("l_s_l1", self.structure_l1_loss.item()),
            ("l_s_adv_gen", self.structure_adv_gen_loss.item()),
            ("l_s_gen", self.structure_gen_loss.item()),
        ]
        return logs

    def sample(self, inputs, gts, maps):
        with torch.no_grad():
            outputs = self.structure_forward(inputs, maps)
            result = [inputs, gts, outputs]
        return result