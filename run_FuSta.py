import sys

sys.path.append('core')

import argparse
import os
import glob
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image as imwrite
# from raft import RAFT
from sparsenet import SparseNet
from utils.utils import InputPadder

import torch.nn.functional as F
import softsplat

import models_arbitrary

DEVICE = 'cuda'


def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img


def load_image_list(image_files):
    images = []
    for imfile in image_files:
        images.append(load_image(imfile))

    images = torch.stack(images, dim=0)
    images = images.to(DEVICE)

    padder = InputPadder(images.shape)
    return padder.pad(images)[0]


def calc_flow(img1, img2):
    with torch.no_grad():
        images = load_image_list([img1, img2])
        # flow_low_res not used
        flow_up = flow_model(images[0, None], images[1, None], iters=20)
    return flow_up.detach()


backwarp_tenGrid = {}
backwarp_tenPartial = {}


def backwarp(ten_input, ten_flow):
    if str(ten_flow.shape) not in backwarp_tenGrid:
        ten_hor = torch.linspace(-1.0 + (1.0 / ten_flow.shape[3]), 1.0 - (1.0 / ten_flow.shape[3]),
                                 ten_flow.shape[3]).view(1, 1, 1, -1).expand(-1, -1, ten_flow.shape[2], -1)
        ten_ver = torch.linspace(-1.0 + (1.0 / ten_flow.shape[2]), 1.0 - (1.0 / ten_flow.shape[2]),
                                 ten_flow.shape[2]).view(1, 1, -1, 1).expand(-1, -1, -1, ten_flow.shape[3])

        backwarp_tenGrid[str(ten_flow.shape)] = torch.cat([ten_hor, ten_ver], 1).cuda()
    # end
    ten_flow = torch.cat([ten_flow[:, 0:1, :, :] / ((ten_input.shape[3] - 1.0) / 2.0),
                          ten_flow[:, 1:2, :, :] / ((ten_input.shape[2] - 1.0) / 2.0)], 1)
    return F.grid_sample(input=ten_input, grid=(backwarp_tenGrid[str(ten_flow.shape)] + ten_flow).permute(0, 2, 3, 1),
                         mode='bilinear', padding_mode='zeros', align_corners=False)


# end


def read_homography(h_path):
    xv, yv = np.meshgrid(np.linspace(0, 832 + 2 * 64 - 1, 832 + 2 * 64), np.linspace(0, 448 + 2 * 64 - 1, 448 + 2 * 64))
    h_inv = np.load(h_path)
    if np.sum(np.abs(h_inv)) == 0.0:
        h_inv[0, 0] = 1.0
        h_inv[1, 1] = 1.0
        h_inv[2, 2] = 1.0
    xv_prime = (h_inv[0, 0] * xv + h_inv[0, 1] * yv + h_inv[0, 2]) / (h_inv[2, 0] * xv + h_inv[2, 1] * yv + h_inv[2, 2])
    yv_prime = (h_inv[1, 0] * xv + h_inv[1, 1] * yv + h_inv[1, 2]) / (h_inv[2, 0] * xv + h_inv[2, 1] * yv + h_inv[2, 2])
    flow = np.stack((xv_prime - xv, yv_prime - yv), -1)
    return flow


def read_flo(flo_path):
    print(flo_path)
    xv, yv = np.meshgrid(np.linspace(-1, 1, 832 + 2 * 64), np.linspace(-1, 1, 448 + 2 * 64))
    flow = np.load(flo_path)
    flow_u = ((flow[:, :, 0] + xv) + 1.0) / 2.0 * float(832 + 2 * 64 - 1)
    flow_v = ((flow[:, :, 1] + yv) + 1.0) / 2.0 * float(448 + 2 * 64 - 1)
    flow_u -= ((xv + 1.0) / 2.0 * float(832 + 2 * 64 - 1))
    flow_v -= ((yv + 1.0) / 2.0 * float(448 + 2 * 64 - 1))
    flow = np.stack((flow_u, flow_v), -1)
    return flow


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='FuSta')
    # parameters
    # Model Selection
    parser.add_argument('--model', type=str, default='adacofnet')

    # Hardware Setting
    parser.add_argument('--gpu_id', type=int, default=0)

    # Directory Setting
    parser.add_argument('--load', type=str, default='FuSta_model/checkpoint/model_epoch050.pth')
    # Learning Options
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    # Options for AdaCoF (none)

    # Options for network
    parser.add_argument('--pooling_with_mask', type=int, default=1)
    parser.add_argument('--decoder_with_mask', type=int, default=1)
    parser.add_argument('--softargmax_with_mask', type=int, default=0)
    parser.add_argument('--decoder_with_gated_conv', type=int, default=1)
    parser.add_argument('--residual_detail_transfer', type=int, default=1)
    parser.add_argument('--beta_learnable', type=int, default=0)
    parser.add_argument('--splatting_type', type=str, default='softmax')
    parser.add_argument('--concat_proxy', type=int, default=0)
    parser.add_argument('--center_residual_detail_transfer', type=int, default=0)
    parser.add_argument('--pooling_with_center_bias', type=int, default=1)
    parser.add_argument('--pooling_type', type=str, default='CNN_flowError')
    parser.add_argument('--no_pooling', type=int, default=0)
    parser.add_argument('--single_decoder', type=int, default=0)
    parser.add_argument('--noDL_CNNAggregation', type=int, default=0)
    parser.add_argument('--gumbel', type=int, default=0)
    parser.add_argument('--inference_with_frame_selection', type=int, default=0)
    parser.add_argument('--FOV_expansion', type=int, default=1)
    parser.add_argument('--seamless', type=int, default=1)
    parser.add_argument('--all_backward', type=int, default=0)
    parser.add_argument('--bundle_forward_flow', type=int, default=0)

    parser.add_argument('--input_frames_path', type=str)
    parser.add_argument('--warping_field_path', type=str)
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--temporal_width', type=int, default=41)
    parser.add_argument('--temporal_step', type=int, default=4)
    parser.add_argument('--num_k', type=int, default=8,
                        help='number of hypotheses to compute for knn Faiss')
    parser.add_argument('--mixed_precision', default=True, help='use mixed precision')
    args = parser.parse_args()

    model = models_arbitrary.Model(args)

    checkpoint = torch.load(args.load)
    model.load(checkpoint['state_dict'])

    model.eval()
    transform = transforms.Compose([transforms.ToTensor()])

    # RAFT
    flow_model = torch.nn.DataParallel(SparseNet(args))
    flow_model.load_state_dict(torch.load('checkpoints/scv-things.pth'))
    flow_model = flow_model.module
    flow_model.to('cuda')
    flow_model.eval()

    INPUT_FRAMES_PATH = args.input_frames_path
    CVPR2020_warping_field_path = args.warping_field_path
    OUTPUT_PATH = args.output_path

    GAUSSIAN_FILTER_KSIZE = args.temporal_width
    gaussian_filter = cv2.getGaussianKernel(GAUSSIAN_FILTER_KSIZE, -1)

    assert (GAUSSIAN_FILTER_KSIZE - 1) // 2 % args.temporal_step == 0

    with torch.no_grad():
        import itertools

        os.makedirs(os.path.join(OUTPUT_PATH), exist_ok=True)
        # put all pngs in a sequence
        all_imgs = sorted(glob.glob(os.path.join(INPUT_FRAMES_PATH, '*.png')))

        tmp_img = cv2.imread(all_imgs[0])
        H = tmp_img.shape[0]
        W = tmp_img.shape[1]

        # temporal padding frames for Gaussian filter
        original_length = len(all_imgs)
        assert original_length > 0

        first_frame = all_imgs[0]
        last_frame = all_imgs[-1]
        all_imgs = [first_frame] * (GAUSSIAN_FILTER_KSIZE // 2) + all_imgs + [last_frame] * (GAUSSIAN_FILTER_KSIZE // 2)

        large_mask_chain = []

        output_frames = []

        for idx in range(GAUSSIAN_FILTER_KSIZE // 2, (GAUSSIAN_FILTER_KSIZE // 2) + original_length):
            keyframe = all_imgs[idx]
            img_name = os.path.split(keyframe)[-1]
            print(img_name)
            tenSecond = torch.FloatTensor(np.ascontiguousarray(
                cv2.imread(filename=keyframe, flags=-1)[..., ::-1].transpose(2, 0, 1)[None, :, :, :].astype(
                    np.float32) * (1.0 / 255.0))).cuda()

            if int(img_name[:-4]) == 0:
                tenH_inv = torch.zeros((1, 2, 448 + 2 * 64, 832 + 2 * 64)).cuda()
                tenFlow = torch.zeros((1, 2, 448 + 2 * 64, 832 + 2 * 64)).cuda()
            else:
                if os.path.isfile(os.path.join(CVPR2020_warping_field_path,
                                               str(int(img_name[:-4]) - 1).zfill(5) + '_H_inv.npy')) and os.path.isfile(
                        os.path.join(CVPR2020_warping_field_path, str(int(img_name[:-4]) - 1).zfill(5) + '.npy')):
                    tenH_inv = torch.FloatTensor(np.ascontiguousarray(read_homography(
                        os.path.join(CVPR2020_warping_field_path,
                                     str(int(img_name[:-4]) - 1).zfill(5) + '_H_inv.npy')).transpose(2, 0, 1)[None, :,
                                                                      :, :])).cuda()
                    if int(img_name[:-4]) == 1:
                        tenFlow = torch.zeros((1, 2, 448 + 2 * 64, 832 + 2 * 64)).cuda()
                    else:
                        tenFlow = torch.FloatTensor(np.ascontiguousarray(read_flo(
                            os.path.join(CVPR2020_warping_field_path,
                                         str(int(img_name[:-4]) - 1).zfill(5) + '.npy')).transpose(2, 0, 1)[None, :, :,
                                                                         :])).cuda()
                else:
                    print('no flow data')
                    continue

            # calculate backward flow using inv_H and backward_flow
            tenBackFlow = backwarp(ten_input=tenH_inv, ten_flow=tenFlow)
            totalFlowIn832 = (tenBackFlow + tenFlow)[:, :, 64:-64, 64:-64]
            """second backward warping in full resolution"""
            W_ratio = W / 832.0
            H_ratio = H / 448.0
            totalFlow = F.upsample(totalFlowIn832, size=(H, W), mode='bilinear')
            F_kprime_to_k = torch.stack((totalFlow[:, 0] * W_ratio, totalFlow[:, 1] * H_ratio), dim=1)

            sum_color = []
            sum_alpha = []
            input_frames = []
            input_flows = []
            forward_flows = []
            backward_flows = []
            for frame_shift in range(-(GAUSSIAN_FILTER_KSIZE // 2), (GAUSSIAN_FILTER_KSIZE // 2) + 1,
                                     int(args.temporal_step)):
                # for frame_shift in [-5, 0, 5]:
                ref_frame = all_imgs[idx + frame_shift]
                ref_frame_name = os.path.split(ref_frame)[-1]
                forward_flow = calc_flow(ref_frame, keyframe)

                # somtimes flow encounters nan or very large values
                """forward_flow[forward_flow != forward_flow] = 0
                forward_flow[forward_flow > 448] = 0
                forward_flow[forward_flow < (-448)] = 0"""
                # cut off padding pixels done by RAFT
                flow_H = list(forward_flow.size())[2]
                flow_W = list(forward_flow.size())[3]
                if H != flow_H or W != flow_W:
                    top = (flow_H - H) // 2
                    forward_flow = forward_flow[:, :, top:top + H]
                    left = (flow_W - W) // 2
                    forward_flow = forward_flow[:, :, :, left:left + W]
                print(forward_flow.shape)
                forward_flows.append(forward_flow)

                backward_flow = calc_flow(keyframe, ref_frame)
                """backward_flow[backward_flow != backward_flow] = 0
                backward_flow[backward_flow > 448] = 0
                backward_flow[backward_flow < (-448)] = 0"""
                backward_flows.append(backward_flow)
                input_frames.append(torch.FloatTensor(np.ascontiguousarray(
                    cv2.imread(filename=ref_frame, flags=-1)[..., ::-1].transpose(2, 0, 1)[None, :, :, :].astype(
                        np.float32) * (1.0 / 255.0))).cuda())

            BOUND_CROP_HT = 3 + int(H % 4 == 0)
            BOUND_CROP_WD = 3 + int(W % 4 == 0)

            input_frames = [x[:, :, BOUND_CROP_HT:-BOUND_CROP_HT, BOUND_CROP_WD:-BOUND_CROP_WD] for x in input_frames]
            F_kprime_to_k = F_kprime_to_k[:, :, BOUND_CROP_HT:-BOUND_CROP_HT, BOUND_CROP_WD:-BOUND_CROP_WD]
            forward_flows = [x[:, :, BOUND_CROP_HT:-BOUND_CROP_HT, BOUND_CROP_WD:-BOUND_CROP_WD] for x in forward_flows]
            backward_flows = [x[:, :, BOUND_CROP_HT:-BOUND_CROP_HT, BOUND_CROP_WD:-BOUND_CROP_WD] for x in
                              backward_flows]

            frame_out = model(input_frames, F_kprime_to_k, forward_flows, backward_flows)
            """output_frames.append(frame_out.detach().cpu())"""
            # if OOM
            os.makedirs('tmp/', exist_ok=True)
            np.save('tmp/' + str(len(large_mask_chain)).zfill(5), frame_out.detach().cpu().numpy())
            output_frames.append('tmp/' + str(len(large_mask_chain)).zfill(5) + '.npy')

            # blending methods
            WWW = 256
            HHH = 256
            tenOnes = torch.ones_like(input_frames[0])[:, 0:1, :, :]
            tenOnes = torch.nn.ZeroPad2d((WWW, WWW, HHH, HHH))(tenOnes).detach()
            F_kprime_to_k_pad = torch.nn.ReplicationPad2d((WWW, WWW, HHH, HHH))(F_kprime_to_k)
            tenWarpedFeat = []
            tenWarpedMask = []
            for iii, feat in enumerate(input_frames):
                # padding for forward warping
                ref_frame_flow = torch.nn.ReplicationPad2d((WWW, WWW, HHH, HHH))(forward_flows[iii])
                # first forward warping
                tenMaskFirst = softsplat.FunctionSoftsplat(tenInput=tenOnes, tenFlow=ref_frame_flow, tenMetric=None,
                                                           strType='average')
                # second backward warping
                tenMaskSecond = backwarp(ten_input=tenMaskFirst, ten_flow=F_kprime_to_k_pad)
                # back to original resolution
                tenMask = tenMaskSecond
                tenWarpedMask.append(tenMask)
            weight_tensor = torch.stack(tenWarpedMask, 0)
            output_mask = torch.sum(weight_tensor, dim=0)
            output_mask = torch.clamp(output_mask, max=1.0)

            large_mask_chain.append(output_mask.detach().cpu())

        WWW -= BOUND_CROP_WD
        HHH -= BOUND_CROP_HT

        from maxflow.fastmin import aexpansion_grid

        accumulated_motion_vectors = np.zeros((original_length, 2), np.int32)
        maximum_movement = 3
        for level in range(5, -1, -1):
            # data term / 2^3 level
            data_term = np.zeros((original_length, (2 * maximum_movement + 1) * (2 * maximum_movement + 1)))
            print('data term')
            for index, obj in enumerate(large_mask_chain):
                print(index)
                for uu, vv in itertools.product(*itertools.repeat(range(-maximum_movement, maximum_movement + 1), 2)):
                    print(uu)
                    # converage term
                    motion_vector_u = int(accumulated_motion_vectors[index, 0] + uu * (2 ** level))
                    motion_vector_v = int(accumulated_motion_vectors[index, 1] + vv * (2 ** level))

                    cropped_mask = obj[:, :, HHH + motion_vector_u:-HHH + motion_vector_u,
                                   WWW + motion_vector_v:-WWW + motion_vector_v]

                    summed_mask = (torch.sum(1.0 - cropped_mask)).cpu().numpy()

                    # fidelity term
                    data_term[index, (uu + maximum_movement) * (
                                2 * maximum_movement + 1) + vv + maximum_movement] = summed_mask

            print('smoothness term')
            smoothness_term = np.zeros(((2 * maximum_movement + 1) * (2 * maximum_movement + 1),
                                        (2 * maximum_movement + 1) * (2 * maximum_movement + 1)))
            for uu, vv, uuuu, vvvv in itertools.product(
                    *itertools.repeat(range(-maximum_movement, maximum_movement + 1), 4)):
                print(uu, vv)
                smoothness_term[(uu + maximum_movement) * (2 * maximum_movement + 1) + vv + maximum_movement, (
                            uuuu + maximum_movement) * (2 * maximum_movement + 1) + vvvv + maximum_movement] = (
                            ((uu - uuuu) * (2 ** level)) ** 2 + ((vv - vvvv) * (2 ** level)) ** 2)

            alpha = 100.0
            labels = aexpansion_grid(data_term, smoothness_term * alpha)  # [H, W]
            for iiii in range(labels.shape[0]):
                accumulated_motion_vectors[iiii, 0] += (labels[iiii] // (
                            2 * maximum_movement + 1) - maximum_movement) * (2 ** level)
                accumulated_motion_vectors[iiii, 1] += (labels[iiii] % (
                            2 * maximum_movement + 1) - maximum_movement) * (2 ** level)
            print(accumulated_motion_vectors)
        print(accumulated_motion_vectors)
        np.savetxt("motion_vector.csv", accumulated_motion_vectors, delimiter=",")

        loss = 0.0
        for index, obj in enumerate(large_mask_chain):
            motion_vector_u = int(accumulated_motion_vectors[index, 0])
            motion_vector_v = int(accumulated_motion_vectors[index, 1])
            cropped_mask = obj[:, :, HHH + motion_vector_u:-HHH + motion_vector_u,
                           WWW + motion_vector_v:-WWW + motion_vector_v]
            print(motion_vector_u)
            print(motion_vector_v)
            print(cropped_mask.shape)
            # if OOM
            imwrite(torch.from_numpy(np.load(output_frames[index]))[:, :, HHH + motion_vector_u:-HHH + motion_vector_u,
                    WWW + motion_vector_v:-WWW + motion_vector_v],
                    os.path.join(OUTPUT_PATH, str(index + 1).zfill(5) + '.png'), range=(0, 1))

            summed_mask = (torch.sum(1.0 - cropped_mask)).cpu().numpy()
            loss += summed_mask
        print(loss)

        # loss without adjustment
        loss = 0.0
        for _, obj in enumerate(large_mask_chain):
            cropped_mask = obj[:, :, HHH:-HHH, WWW:-WWW]
            summed_mask = (torch.sum(1.0 - cropped_mask)).cpu().numpy()
            loss += summed_mask
        print(loss)



