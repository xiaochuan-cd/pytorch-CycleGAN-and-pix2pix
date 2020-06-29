import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from PIL import Image
from data.base_dataset import BaseDataset, get_params, get_transform
import util.util as util
import torchvision.transforms as transforms

def getitem(opt):

    IM = Image.open('20.jpg').convert('RGB')

    # transform_params = get_params(opt, IM.size)
    # IM_transform = get_transform(opt, transform_params, grayscale=False)

    IM_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    IM = IM_transform(IM)

    return IM.reshape([1]+list(IM.shape))

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    im = getitem(opt)
    im = model.infer(im)
    print(im)
    im = util.tensor2im(im)
    util.save_image(im, '20f.png')