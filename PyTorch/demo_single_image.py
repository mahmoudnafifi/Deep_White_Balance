"""
 Demo single image
 Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
 If you use this code, please cite the following paper:
 Mahmoud Afifi and Michael S Brown. Deep White-Balance Editing. In CVPR, 2020.
"""
__author__ = "Mahmoud Afifi"
__credits__ = ["Mahmoud Afifi"]

import argparse
import logging
import os
import torch
from PIL import Image
from arch import deep_wb_model
import utilities.utils as utls
from utilities.deepWB import deep_wb
import arch.splitNetworks as splitter
from arch import deep_wb_single_task


def get_args():
    parser = argparse.ArgumentParser(description='Changing WB of an input image.')
    parser.add_argument('--model_dir', '-m', default='./models',
                        help="Specify the directory of the trained model.", dest='model_dir')
    parser.add_argument('--input', '-i', help='Input image filename', dest='input',
                        default='../example_images/00.JPG')
    parser.add_argument('--output_dir', '-o', default='../result_images',
                        help='Directory to save the output images', dest='out_dir')
    parser.add_argument('--task', '-t', default='all',
                        help="Specify the required task: 'AWB', 'editing', or 'all'.", dest='task')
    parser.add_argument('--target_color_temp', '-tct', default=None, type=int,
                        help="Target color temperature [2850 - 7500]. If specified, the --task should be 'editing'",
                        dest='target_color_temp')
    parser.add_argument('--mxsize', '-S', default=656, type=int,
                        help="Max dim of input image to the network, the output will be saved in its original res.",
                        dest='S')
    parser.add_argument('--show', '-v', action='store_true', default=True,
                        help="Visualize the input and output images",
                        dest='show')
    parser.add_argument('--save', '-s', action='store_true',
                        help="Save the output images",
                        default=True, dest='save')
    parser.add_argument('--device', '-d', default='cuda',
                        help="Device: cuda or cpu.", dest='device')

    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    if args.device.lower() == 'cuda':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    fn = args.input
    out_dir = args.out_dir
    S = args.S
    target_color_temp = args.target_color_temp
    tosave = args.save

    if target_color_temp:
        assert 2850 <= target_color_temp <= 7500, (
                'Color temperature should be in the range [2850 - 7500], but the given one is %d' % target_color_temp)

        if args.task.lower() != 'editing':
            raise Exception('The task should be editing when a target color temperature is specified.')

    logging.info(f'Using device {device}')

    if tosave:
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

    if args.task.lower() == 'all':
        if os.path.exists(os.path.join(args.model_dir, 'net_awb.pth')) and \
                os.path.exists(os.path.join(args.model_dir, 'net_t.pth')) and \
                os.path.exists(os.path.join(args.model_dir, 'net_s.pth')):
            # load awb net
            net_awb = deep_wb_single_task.deepWBnet()
            logging.info("Loading model {}".format(os.path.join(args.model_dir, 'net_awb.pth')))
            net_awb.to(device=device)
            net_awb.load_state_dict(torch.load(os.path.join(args.model_dir, 'net_awb.pth'),
                                               map_location=device))
            net_awb.eval()
            # load tungsten net
            net_t = deep_wb_single_task.deepWBnet()
            logging.info("Loading model {}".format(os.path.join(args.model_dir, 'net_t.pth')))
            net_t.to(device=device)
            net_t.load_state_dict(
                torch.load(os.path.join(args.model_dir, 'net_t.pth'), map_location=device))
            net_t.eval()
            # load shade net
            net_s = deep_wb_single_task.deepWBnet()
            logging.info("Loading model {}".format(os.path.join(args.model_dir, 'net_s.pth')))
            net_s.to(device=device)
            net_s.load_state_dict(
                torch.load(os.path.join(args.model_dir, 'net_s.pth'), map_location=device))
            net_s.eval()
            logging.info("Models loaded !")
        elif os.path.exists(os.path.join(args.model_dir, 'net.pth')):
            net = deep_wb_model.deepWBNet()
            logging.info("Loading model {}".format(os.path.join(args.model_dir, 'net.pth')))
            net.load_state_dict(torch.load(os.path.join(args.model_dir, 'net.pth')))
            net_awb, net_t, net_s = splitter.splitNetworks(net)
            net_awb.to(device=device)
            net_awb.eval()
            net_t.to(device=device)
            net_t.eval()
            net_s.to(device=device)
            net_s.eval()
        else:
            raise Exception('Model not found!')
    elif args.task.lower() == 'editing':
        if os.path.exists(os.path.join(args.model_dir, 'net_t.pth')) and \
                os.path.exists(os.path.join(args.model_dir, 'net_s.pth')):
            # load tungsten net
            net_t = deep_wb_single_task.deepWBnet()
            logging.info("Loading model {}".format(os.path.join(args.model_dir, 'net_t.pth')))
            net_t.to(device=device)
            net_t.load_state_dict(
                torch.load(os.path.join(args.model_dir, 'net_t.pth'), map_location=device))
            net_t.eval()
            # load shade net
            net_s = deep_wb_single_task.deepWBnet()
            logging.info("Loading model {}".format(os.path.join(args.model_dir, 'net_s.pth')))
            net_s.to(device=device)
            net_s.load_state_dict(
                torch.load(os.path.join(args.model_dir, 'net_s.pth'), map_location=device))
            net_s.eval()
            logging.info("Models loaded !")
        elif os.path.exists(os.path.join(args.model_dir, 'net.pth')):
            net = deep_wb_model.deepWBNet()
            logging.info("Loading model {}".format(os.path.join(args.model_dir, 'net.pth')))
            logging.info(f'Using device {device}')
            net.load_state_dict(torch.load(os.path.join(args.model_dir, 'net.pth')))
            _, net_t, net_s = splitter.splitNetworks(net)
            net_t.to(device=device)
            net_t.eval()
            net_s.to(device=device)
            net_s.eval()
        else:
            raise Exception('Model not found!')
    elif args.task.lower() == 'awb':
        if os.path.exists(os.path.join(args.model_dir, 'net_awb.pth')):
            # load awb net
            net_awb = deep_wb_single_task.deepWBnet()
            logging.info("Loading model {}".format(os.path.join(args.model_dir, 'net_awb.pth')))
            logging.info(f'Using device {device}')
            net_awb.to(device=device)
            net_awb.load_state_dict(torch.load(os.path.join(args.model_dir, 'net_awb.pth'),
                                               map_location=device))
            net_awb.eval()
        elif os.path.exists(os.path.join(args.model_dir, 'net.pth')):
            net = deep_wb_model.deepWBNet()
            logging.info("Loading model {}".format(os.path.join(args.model_dir, 'net.pth')))
            logging.info(f'Using device {device}')
            net.load_state_dict(torch.load(os.path.join(args.model_dir, 'net.pth')))
            net_awb, _, _ = splitter.splitNetworks(net)
            net_awb.to(device=device)
            net_awb.eval()
        else:
            raise Exception('Model not found!')
    else:
        raise Exception("Wrong task! Task should be: 'AWB', 'editing', or 'all'")

logging.info("Processing image {} ...".format(fn))
img = Image.open(fn)
_, fname = os.path.split(fn)
name, _ = os.path.splitext(fname)
if args.task.lower() == 'all':  # awb and editing tasks
    out_awb, out_t, out_s = deep_wb(img, task=args.task.lower(), net_awb=net_awb, net_s=net_s, net_t=net_t,
                                    device=device, s=S)
    out_f, out_d, out_c = utls.colorTempInterpolate(out_t, out_s)
    if tosave:
        result_awb = utls.to_image(out_awb)
        result_t = utls.to_image(out_t)
        result_s = utls.to_image(out_s)
        result_f = utls.to_image(out_f)
        result_d = utls.to_image(out_d)
        result_c = utls.to_image(out_c)
        result_awb.save(os.path.join(out_dir, name + '_AWB.png'))
        result_s.save(os.path.join(out_dir, name + '_S.png'))
        result_t.save(os.path.join(out_dir, name + '_T.png'))
        result_f.save(os.path.join(out_dir, name + '_F.png'))
        result_d.save(os.path.join(out_dir, name + '_D.png'))
        result_c.save(os.path.join(out_dir, name + '_C.png'))

    if args.show:
        logging.info("Visualizing results for image: {}, close to continue ...".format(fn))
        utls.imshow(img, result_awb, result_t, result_f, result_d, result_c, result_s)

elif args.task.lower() == 'awb':  # awb task
    out_awb = deep_wb(img, task=args.task.lower(), net_awb=net_awb, device=device, s=S)
    if tosave:
        result_awb = utls.to_image(out_awb)
        result_awb.save(os.path.join(out_dir, name + '_AWB.png'))

    if args.show:
        logging.info("Visualizing result for image: {}, close to continue ...".format(fn))
        utls.imshow(img, result_awb)

else:  # editing
    out_t, out_s = deep_wb(img, task=args.task.lower(), net_s=net_s, net_t=net_t, device=device, s=S)

    if target_color_temp:
        out = utls.colorTempInterpolate_w_target(out_t, out_s, target_color_temp)
        if tosave:
            out = utls.to_image(out)
            out.save(os.path.join(out_dir, name + '_%d.png' % target_color_temp))

        if args.show:
            logging.info("Visualizing result for image: {}, close to continue ...".format(fn))
            utls.imshow(img, out, colortemp=target_color_temp)

    else:
        out_f, out_d, out_c = utls.colorTempInterpolate(out_t, out_s)
        if tosave:
            result_t = utls.to_image(out_t)
            result_s = utls.to_image(out_s)
            result_f = utls.to_image(out_f)
            result_d = utls.to_image(out_d)
            result_c = utls.to_image(out_c)
            result_s.save(os.path.join(out_dir, name + '_S.png'))
            result_t.save(os.path.join(out_dir, name + '_T.png'))
            result_f.save(os.path.join(out_dir, name + '_F.png'))
            result_d.save(os.path.join(out_dir, name + '_D.png'))
            result_c.save(os.path.join(out_dir, name + '_C.png'))

        if args.show:
            logging.info("Visualizing results for image: {}, close to continue ...".format(fn))
            utls.imshow(img, result_t, result_f, result_d, result_c, result_s)
