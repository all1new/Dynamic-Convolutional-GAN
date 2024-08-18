import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn.init as init
import math
import torch
import torchvision.utils as vutils
from natsort import natsorted

def create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def imsave(img, path):
    img = img.permute(0, 2, 3, 1)
    im = Image.fromarray(img.cpu().detach().numpy().astype(np.uint8).squeeze())
    im.save(path)

# 进度条
class Progbar(object):
    """Displays a progress bar.

    Arguments:
        target: Total number of steps expected, None if unknown.
        width: Progress bar width on screen.
        verbose: Verbosity mode, 0 (silent), 1 (verbose), 2 (semi-verbose)
        stateful_metrics: Iterable of string names of metrics that
            should *not* be averaged over time. Metrics in this list
            will be displayed as-is. All others will be averaged
            by the progbar before display.
        interval: Minimum visual progress update interval (in seconds).
    """

    def __init__(self, target, width=25, verbose=1, interval=0.05,
                 stateful_metrics=None):
        self.target = target
        self.width = width
        self.verbose = verbose
        self.interval = interval
        if stateful_metrics:
            self.stateful_metrics = set(stateful_metrics)
        else:
            self.stateful_metrics = set()

        self._dynamic_display = ((hasattr(sys.stdout, 'isatty') and
                                  sys.stdout.isatty()) or
                                 'ipykernel' in sys.modules or
                                 'posix' in sys.modules)
        self._total_width = 0
        self._seen_so_far = 0
        # We use a dict + list to avoid garbage collection
        # issues found in OrderedDict
        self._values = {}
        self._values_order = []
        self._start = time.time()
        self._last_update = 0

    def update(self, current, values=None):
        """Updates the progress bar.

        Arguments:
            current: Index of current step.
            values: List of tuples:
                `(name, value_for_last_step)`.
                If `name` is in `stateful_metrics`,
                `value_for_last_step` will be displayed as-is.
                Else, an average of the metric over time will be displayed.
        """
        values = values or []
        for k, v in values:
            if k not in self._values_order:
                self._values_order.append(k)
            if k not in self.stateful_metrics:
                if k not in self._values:
                    self._values[k] = [v * (current - self._seen_so_far),
                                       current - self._seen_so_far]
                else:
                    self._values[k][0] += v * (current - self._seen_so_far)
                    self._values[k][1] += (current - self._seen_so_far)
            else:
                self._values[k] = v
        self._seen_so_far = current

        now = time.time()
        info = ' - %.0fs' % (now - self._start)
        if self.verbose == 1:
            if (now - self._last_update < self.interval and
                    self.target is not None and current < self.target):
                return

            prev_total_width = self._total_width
            if self._dynamic_display:
                sys.stdout.write('\b' * prev_total_width)
                sys.stdout.write('\r')
            else:
                sys.stdout.write('\n')

            if self.target is not None:
                numdigits = int(np.floor(np.log10(self.target))) + 1
                barstr = '%%%dd/%d [' % (numdigits, self.target)
                bar = barstr % current
                prog = float(current) / self.target
                prog_width = int(self.width * prog)
                if prog_width > 0:
                    bar += ('=' * (prog_width - 1))
                    if current < self.target:
                        bar += '>'
                    else:
                        bar += '='
                bar += ('.' * (self.width - prog_width))
                bar += ']'
            else:
                bar = '%7d/Unknown' % current

            self._total_width = len(bar)
            sys.stdout.write(bar)

            if current:
                time_per_unit = (now - self._start) / current
            else:
                time_per_unit = 0
            if self.target is not None and current < self.target:
                eta = time_per_unit * (self.target - current)
                if eta > 3600:
                    eta_format = '%d:%02d:%02d' % (eta // 3600,
                                                   (eta % 3600) // 60,
                                                   eta % 60)
                elif eta > 60:
                    eta_format = '%d:%02d' % (eta // 60, eta % 60)
                else:
                    eta_format = '%ds' % eta

                info = ' - ETA: %s' % eta_format
            else:
                if time_per_unit >= 1:
                    info += ' %.0fs/step' % time_per_unit
                elif time_per_unit >= 1e-3:
                    info += ' %.0fms/step' % (time_per_unit * 1e3)
                else:
                    info += ' %.0fus/step' % (time_per_unit * 1e6)

            for k in self._values_order:
                info += ' - %s:' % k
                if isinstance(self._values[k], list):
                    avg = np.mean(self._values[k][0] / max(1, self._values[k][1]))
                    if abs(avg) > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                else:
                    info += ' %s' % self._values[k]

            self._total_width += len(info)
            if prev_total_width > self._total_width:
                info += (' ' * (prev_total_width - self._total_width))

            if self.target is not None and current >= self.target:
                info += '\n'

            sys.stdout.write(info)
            sys.stdout.flush()

        elif self.verbose == 2:
            if self.target is None or current >= self.target:
                for k in self._values_order:
                    info += ' - %s:' % k
                    avg = np.mean(self._values[k][0] / max(1, self._values[k][1]))
                    if avg > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                info += '\n'

                sys.stdout.write(info)
                sys.stdout.flush()

        self._last_update = now

    def add(self, n, values=None):
        self.update(self._seen_so_far + n, values)

# network init function
def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'gaussian':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
    return init_fun

# get model list for resume
def get_mdoel_list(dirname, key):
    if os.path.exists(dirname) is False:
        return None 
    gen_models = [os.path.join(dirname, f) for f in os.listdir(dirname) if 
                  os.path.isfile(os.path.join(dirname, f)) and key in f and ".pt" in f]
    if gen_models == []:
        return None
    gen_models.sort()
    last_models_name = gen_models[-1]
    return last_models_name

# 获得迭代的次数
def get_iteration(dir_name, file_name, net_name):
    if os.path.exists(os.path.join(dir_name, file_name)) is False:
        return None
    if 'latest' in file_name:
        gen_models = [os.path.join(dir_name, f) for f in os.listdir(dir_name) if
                      os.path.isfile(os.path.join(dir_name, f)) and (not 'latest' in f) and (".pt" in f) and (net_name in f)]
        if gen_models == []:
            return 0
        model_name = os.path.basename(natsorted(gen_models)[-1])
    else:
        model_name = file_name
    iterations = int(model_name.replace('_net_' + net_name + '.pth', ''))
    return iterations

# 归一化图片数据
def __denorm(x):
    x = (x + 1) / 2
    return x.clamp_(0, 1)

# 保存图像数据结果
def __write_images(image_outputs, display_image_num, file_name):
    image_outputs = [images.expand(-1, 3, -1, -1) for images in image_outputs] # expand gray-scale images to 3 channels
    image_tensor = torch.cat([images[:display_image_num] for images in image_outputs], 0)
    image_tensor = __denorm(image_tensor)
    image_grid = vutils.make_grid(image_tensor.data, nrow=display_image_num, padding=0, normalize=False)
    vutils.save_image(image_grid, file_name, nrow=1)


def write_2images(image_outputs, display_image_num, name):
    n = len(image_outputs)
    __write_images(image_outputs[0:n], display_image_num, name) 

class Logger:
    def __init__(self, log_dir=None, clear=False):
        if log_dir is not None:
            # color palette
            #colors = loadmat('color150.mat')['colors']
            #palette = colors.reshape(-1)
            #palette = list(palette)
            #palette += ([0] * (256*3-len(palette)))
            #pdb.set_trace()
            # self.palette = palette

            self.log_dir = log_dir
            if not os.path.exists(log_dir):
                os.mkdir(log_dir)
            self.log_dir = log_dir
            self.plot_dir = os.path.join(log_dir, "plot")
            if not os.path.exists(self.plot_dir):
                os.mkdir(self.plot_dir)
            elif clear:
                os.system("rm {}/plot/*".format(log_dir))
            self.image_dir = os.path.join(log_dir, "image")
            if not os.path.exists(self.image_dir):
                os.mkdir(self.image_dir)
            elif clear:
                os.system("rm -rf {}/image/*".format(log_dir))
            if not os.path.exists(os.path.join(log_dir, "image_ticks")):
                os.mkdir(os.path.join(log_dir, "image_ticks"))
            elif clear:
                os.system("rm -rf {}/image_ticks/*".format(log_dir))
            self.plot_vals = {}
            self.plot_times = {}
            #def http_server():
            #    Handler = QuietHandler
            #    with socketserver.TCPServer(("", port), Handler) as httpd:
            #        #print("serving at port", PORT)
            #        httpd.serve_forever()
            #x=threading.Thread(target=http_server)
            #x.start()
            #print("==============================================")
            #print("visualize at http://host ip:{}/{}.html".format(port, self.log_dir))
            #print("==============================================")

    def batch_plot_landmark(self, name, batch_img, dict_label_mark):
        bsize, c, h, w = batch_img.shape
        batch_img = batch_img.detach().cpu().numpy().transpose((0, 2, 3, 1))
        cat_image = np.concatenate(list(batch_img), 1)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(cat_image)
        for label, batch_land in dict_label_mark.items():
            batch_land2d = batch_land[:, :, :2]
            _, n_points, _ = batch_land2d.shape
            batch_land2d[:, :, 1] = h - batch_land2d[:, :, 1]
            batch_land2d = batch_land2d.detach().cpu().numpy()
            offset = np.arange(0, bsize * w, w)
            batch_land2d[:, :, 0] += offset[..., None]
            batch_land2d = batch_land2d.reshape(-1, 2)

            ax.scatter(batch_land2d[:, 0], batch_land2d[:, 1], s=0.5, label=label)
        ax.axis("off")
        # Put a legend below current axis
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
                  fancybox=True, shadow=True, ncol=5)
        fig.savefig(os.path.join(self.plot_dir, '%s.png'%name))
        plt.close()


    def add_scalar(self, name, value, t_iter):
        if not name in self.plot_vals:
            self.plot_vals[name] = [value]
            self.plot_times[name] = [t_iter]
        else:
            self.plot_vals[name].append(value)
            self.plot_times[name].append(t_iter)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(self.plot_times[name], self.plot_vals[name])
        fig.savefig(os.path.join(self.plot_dir, '%s.png'%name))
        plt.close()
    #add_image('image', torchvision.utils.make_grid(img), num_iter)

    def add_single_image(self, name, image, t_iter=None):
       image = image.detach().cpu().numpy()
       image = image.transpose((1, 2, 0))
       image = Image.fromarray((image*255).astype(np.uint8))
       image.save(os.path.join(self.plot_dir, "%s.png"%name))

    def add_image(self, name, image, t_iter):
       path_name = os.path.join(self.image_dir, name)
       if not os.path.exists(path_name):
           os.mkdir(path_name)
       image = image.detach().cpu().numpy()
       image = image.transpose((1, 2, 0))
       image = Image.fromarray((image*255).astype(np.uint8))
       image.save(os.path.join(path_name, "%d.png"%t_iter))
       with open(os.path.join(self.log_dir, "image_ticks", name+".txt"), "a") as f:
           f.write(str(t_iter)+'\n')

    def add_single_label(self, name, image, t_iter):
       image = image.detach().cpu().numpy()
       image = Image.fromarray(image.astype(np.uint8)).convert("P")
       image.putpalette(self.palette)
       image.save(os.path.join(self.plot_dir, "%s.png"%name))

    def add_label(self, name, image, t_iter):
       path_name = os.path.join(self.image_dir, name)
       if not os.path.exists(path_name):
           os.mkdir(path_name)
       image = image.detach().cpu().numpy()
       image = Image.fromarray(image.astype(np.uint8)).convert("P")
       image.putpalette(self.palette)
       image.save(os.path.join(path_name, "%d.png"%t_iter))
       with open(os.path.join(self.log_dir, "image_ticks", name+".txt"), "a") as f:
           f.write(str(t_iter)+'\n')

    def write_html_eval(self, base_dir):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        # file_loader = FileSystemLoader(os.path.join(dir_path, "templates"))
        # env = Environment(loader=file_loader)
        # template = env.get_template('val.html')

        list_exp_ticks = []
        list_exp_paths = []
        list_img_names = []
        list_dirs = os.listdir(base_dir)
        if "mask" in list_dirs:
            list_dirs.remove("mask")
        list_dirs = sorted(list_dirs)
        list_dirs = ["mask"] + list_dirs
        k=0
        for i, exp in enumerate(list_dirs):
            if os.path.isdir(os.path.join(base_dir, exp)) and not exp.startswith("."):
                list_exp_paths.append( os.path.join(exp, "image"))
                list_exp_ticks.append( os.listdir(os.path.join(base_dir, exp, "image")))
                if k == 0:
                    _l = list(filter(lambda x: not x.startswith("."), list_exp_ticks[0]))
                    list_img_names += \
                            os.listdir(
                                os.path.join(base_dir, exp, "image", str(_l[0]))
                                )
                    k += 1
        # output = template.render( list_exp_ticks=list_exp_ticks, list_exp_paths=list_exp_paths, list_img_names=list_img_names)
        #print(output)
        # with open("{}/validation.html".format(base_dir), "w") as f:
        #     f.writelines(output)

    def write_console(self, epoch, i, t):
        message = '(epoch: %d, iters: %d, time: %.3f) ' % (epoch, i, t)
        for k,v in self.plot_vals.items():
            #print(v)
            #if v != 0:
            #v = v.mean().float()
            v = v[-1]
            message += '%s: %.4f ' % (k, v)

        print(message)
        prefix = self.log_dir
        with open("{}/logs.txt".format(prefix), "a") as log_file:
            log_file.write('%s\n' % message)

    def write_scalar(self, name, value, t_iter):
        prefix = self.log_dir
        with open("{}/{}.txt".format(prefix, name), "a") as log_file:
            message = '%s %d: %.4f' % (name, t_iter, value)
            print(message)
            log_file.write(message+"\n")


    def write_html(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        # file_loader = FileSystemLoader(os.path.join(dir_path, "templates"))
        # env = Environment(loader=file_loader)
        # template = env.get_template('train.html')

        prefix = self.log_dir
        re_prefix = self.log_dir.split("/")[-1]
        plotpath = os.path.join(prefix, "plot")
        plotfiles = os.listdir(plotpath)
        plotfiles = list(map(lambda x: os.path.join(re_prefix, "plot", x), plotfiles))

        image_tick_path = []
        imagepath = os.path.join(prefix, "image")
        for folder in os.listdir(imagepath):
            ticks = open("{}/image_ticks/{}.txt".format(prefix, folder), "r").read()
            ticks = ticks.split('\n')[:-1]
            ticks = list(map(lambda x:int(x), ticks))
            folderpath = os.path.join(re_prefix, "image", folder)
            image_tick_path.append({"tick":ticks, "path":folderpath})

