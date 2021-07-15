import sys
import warnings
import glob
import os
import numpy as np
import cv2
import models_flownet
import mpi_net35
import torch
from torch.nn.functional import conv2d, grid_sample
from torch.autograd import Variable
from flow_vis import flow_to_color
from torchvision import transforms
from lib.nn import async_copy_to
from lib.utils import as_numpy
from models import ModelBuilder, SegmentationModule
from scipy.ndimage.filters import gaussian_filter

warnings.filterwarnings("ignore")

input_file = sys.argv[1]
out_file = sys.argv[2]
out_warping_field_path = sys.argv[3]
SEG = 300

RHO = 0.1
N_FRAME = 20
N_KEEP = 5
BATCH_SIZE = 8
MARGIN = 64


# Network Builders
builder = ModelBuilder()
net_encoder = builder.build_encoder(
    arch='resnet50dilated',
    fc_dim=2048,
    weights='baseline-resnet50dilated-ppm_deepsup/encoder_epoch_20.pth')
net_decoder = builder.build_decoder(
    arch='ppm_deepsup',
    fc_dim=2048,
    num_class=150,
    weights='baseline-resnet50dilated-ppm_deepsup/decoder_epoch_20.pth',
    use_softmax=True)
crit = torch.nn.NLLLoss(ignore_index=-1)
segmentation_module = SegmentationModule(net_encoder, net_decoder, crit).cuda()
segmentation_module.eval()
normalize = transforms.Normalize(
    mean=[102.9801, 115.9465, 122.7717],
    std=[1., 1., 1.])


def transform1(x):
    return x.unsqueeze(0).permute(0, 2, 3, 1)


def float_cuda(x):
    return x.unsqueeze(0).float().cuda()


def img_transform(img):
    # image to float
    img = img.astype(np.float32)
    img = img.transpose((2, 0, 1))
    img = normalize(torch.from_numpy(img.copy()))
    return img


# Round x to the nearest multiple of p and x' >= x
def round2nearest_multiple(x, p):
    return ((x - 1) // p + 1) * p


def compute_mask(img):
    ori_height, ori_width, _ = img.shape
    img_size = [300, 400, 500, 600]  #
    img_resized_list = []
    for this_short_size in img_size:
        # calculate target height and width
        scale = min(this_short_size / float(min(ori_height, ori_width)),
                    1000 / float(max(ori_height, ori_width)))
        target_height, target_width = int(ori_height * scale), int(ori_width * scale)

        # to avoid rounding in network
        target_height = round2nearest_multiple(target_height, 8)
        target_width = round2nearest_multiple(target_width, 8)

        # resize
        img_resized = cv2.resize(img.copy(), (target_width, target_height))

        # image transform
        img_resized = img_transform(img_resized)
        img_resized = torch.unsqueeze(img_resized, 0)
        img_resized_list.append(img_resized)

    batch_data = dict()
    batch_data['img_ori'] = img.copy()
    batch_data['img_data'] = [x.contiguous() for x in img_resized_list]

    seg_size = (batch_data['img_ori'].shape[0],
               batch_data['img_ori'].shape[1])
    img_resized_list = batch_data['img_data']

    with torch.no_grad():
        scores = torch.zeros(1, 150, seg_size[0], seg_size[1])
        scores = async_copy_to(scores, 0)

        for img in img_resized_list:
            feed_dict = batch_data.copy()
            feed_dict['img_data'] = img
            del feed_dict['img_ori']
            feed_dict = async_copy_to(feed_dict, 0)

            # forward pass
            pred_tmp = segmentation_module(feed_dict, segSize=seg_size)
            scores = scores + pred_tmp / len(img_size)

        _, pred = torch.max(scores, dim=1)
        pred = as_numpy(pred.squeeze(0).cpu())
        mask = 1 - torch.from_numpy(np.logical_or.reduce((pred == 12, pred == 20, pred == 76, pred == 80,
                                                          pred == 83, pred == 90, pred == 102, pred == 103,
                                                          pred == 116, pred == 126, pred == 127)).astype(np.float32))

        mask_person = 1 - torch.from_numpy((pred == 12).astype(np.float32))

    return mask, mask_person


def movingstd2(A, k):
    A = A - torch.mean(A)
    Astd = torch.std(A)
    A = A / Astd
    A2 = A * A

    wuns = torch.ones(A.shape).cuda()

    kernel = torch.ones(2 * k + 1, 2 * k + 1).cuda()

    N = conv2d(wuns.unsqueeze(0).unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0), padding=5)
    s = torch.sqrt((conv2d(A2.unsqueeze(0).unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0), padding=5) - ((conv2d(A.unsqueeze(0).unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0), padding=5)) ** 2) / N) / (N - 1))
    s = s * Astd

    return s


def moving_average(b, n=3):
    res = gaussian_filter(b, sigma=n)
    return res


def grad_image(x):
    A = torch.Tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]]).cuda()
    A = A.view((1, 1, 3, 3))
    G_x = conv2d(x.unsqueeze(0).unsqueeze(0), A, padding=1)

    B = torch.Tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).cuda()
    B = B.view((1, 1, 3, 3))
    G_y = conv2d(x.unsqueeze(0).unsqueeze(0), B, padding=1)

    G = torch.sqrt(torch.pow(G_x[0, 0, :, :], 2) + torch.pow(G_y[0, 0, :, :], 2))
    return G


PERSON_COUNTER = 1


def compute_flow_seg(video, H, start):
    frm_count, wd, ht, _ = video.shape
    new_wd, new_ht = wd + 2 * MARGIN, ht + 2 * MARGIN
    inp = torch.zeros(1, 3, 2, new_wd, new_ht).cuda()
    optic = torch.zeros(frm_count - 1 + N_FRAME, 3, new_wd, new_ht)
    mask_object = torch.zeros(frm_count - 1 + N_FRAME, new_wd, new_ht)

    warped_video = torch.ones(frm_count, new_wd, new_ht, 3)
    mask = torch.ones(frm_count, new_wd, new_ht)
    mask_towarp = np.zeros((new_wd, new_ht))
    mask_towarp[MARGIN:MARGIN + wd, MARGIN:MARGIN + ht] = 1

    optic[:, 2, :, :] = torch.from_numpy(mask_towarp)

    des_wd, des_ht = 832 + 2 * MARGIN, 448 + 2 * MARGIN

    for i in range(video.shape[0]):
        im1 = video[i, :, :, :]
        im1 = np.concatenate((np.zeros((MARGIN, 832, 3)), im1, np.zeros((MARGIN, 832, 3))), axis=0)
        im1 = np.concatenate((np.zeros((des_ht, MARGIN, 3)), im1, np.zeros((des_ht, MARGIN, 3))), axis=1)
        im1 = im1.astype(np.uint8)
        warped_video[i, :, :, :] = torch.from_numpy(cv2.warpPerspective(im1, H[start + i, :, :], (im1.shape[1], im1.shape[0])))
        mask[i, :, :] = torch.from_numpy(cv2.warpPerspective(mask_towarp, H[start + i, :, :], (im1.shape[1], im1.shape[0])))
        ss, ss_person = compute_mask(video[i, :, :, :])
        ss = np.concatenate((np.zeros((MARGIN, 832)), ss.numpy(), np.zeros((MARGIN, 832))), axis=0)
        ss_person = np.concatenate((np.zeros((MARGIN, 832)), ss_person.numpy(), np.zeros((MARGIN, 832))), axis=0)
        ss = np.concatenate((np.zeros((des_ht, MARGIN)), ss, np.zeros((des_ht, MARGIN))), axis=1)
        ss_person = np.concatenate((np.zeros((des_ht, MARGIN)), ss_person, np.zeros((des_ht, MARGIN))), axis=1)
        optic[i, 2, :, :] = torch.from_numpy(cv2.warpPerspective(ss, H[start + i, :, :], (im1.shape[1], im1.shape[0])))
        mask_object[i, :, :] = torch.from_numpy(cv2.warpPerspective(ss_person, H[start + i, :, :], (im1.shape[1], im1.shape[0])))

    warped_video = np.concatenate((warped_video.numpy(), np.tile(np.expand_dims(warped_video[-1, :, :, :].numpy(), 0), (N_FRAME, 1, 1, 1))), 0)
    mask = np.concatenate((mask.numpy(), np.tile(np.expand_dims(mask[-1, :, :].numpy(), 0), (N_FRAME, 1, 1))), 0)

    for i in range(video.shape[0] - 1):
        inp[:, :, 0, :, :] = torch.from_numpy(warped_video[i + 1, :, :, :]).float().cuda().permute((2, 0, 1)).unsqueeze(0)
        inp[:, :, 1, :, :] = torch.from_numpy(warped_video[i, :, :, :]).float().cuda().permute((2, 0, 1)).unsqueeze(0)
        out = flow_model(inp) * 2
        flow = out[0].cpu().permute(1, 2, 0).numpy()

        flow_color = flow_to_color(flow, convert_to_bgr=True)

        cv2.imwrite(f'./data/flow{i}.png', flow_color)

        out[:, 0, :, :] = out[:, 0, :, :] / des_wd
        out[:, 1, :, :] = out[:, 1, :, :] / des_ht
        meanx = torch.mean(out[0, 0, :, :][torch.from_numpy(mask[i, :, :]) > 0])
        meany = torch.mean(out[0, 1, :, :][torch.from_numpy(mask[i, :, :]) > 0])
        optic[i, 0:2, :, :] = out

        u2 = out[0, 0, :, :]
        v2 = out[0, 1, :, :]
        A1 = movingstd2(u2, 5).squeeze()
        A2 = movingstd2(v2, 5).squeeze()
        im1_gray = cv2.cvtColor(warped_video[i, :, :, :], cv2.COLOR_BGR2GRAY).astype(np.float32)
        # skin color detection removed since not used

        grad = grad_image(torch.from_numpy(im1_gray).cuda())
        ss = optic[i, 2, :, :]
        valid = (torch.abs(out[0, 0, :, :] - meanx) < (5.0 / des_wd)) * (torch.abs(out[0, 1, :, :] - meany) < (5.0 / des_ht))
        invalid = (torch.abs(out[0, 0, :, :] - meanx) > (20.0 / des_wd)) + (torch.abs(out[0, 1, :, :] - meany) > (20.0 / des_ht))
        ss[valid == 1] = 1
        ss[invalid == 1] = 0
        global PERSON_COUNTER
        PERSON_COUNTER += 1

        ss[grad < 8] = 0
        ss[A1 > 3 * torch.mean(A1)] = 0
        ss[A2 > 3 * torch.mean(A2)] = 0
        ss[torch.from_numpy(mask[i, :, :]) == 0] = 0
        optic[i, 2, :, :] = ss

        oup = optic[i, 0:2, :, :].permute((1, 2, 0)).view(des_ht * des_wd, 2)
        mask1 = optic[i, 2, :, :].view(-1)
        Q = base[mask1 > 0]
        c = torch.mm(torch.mm((torch.mm(Q.t(), Q) + RHO * torch.eye(2 * N_KEEP)).inverse(), Q.t()), oup[mask1 > 0].cpu())
        oup2 = torch.mm(base, c)
        fill = ((mask1 == 0) * (torch.from_numpy(mask[i, :, :]).view(-1) > 0))
        oup[fill] = oup2[fill]
        optic[i, 0:2, :, :] = oup.view(des_ht, des_wd, 2).permute((2, 0, 1)).unsqueeze(0)

    return optic, warped_video, mask, mask_object


def modify_flow(optic, warp, grid_large):
    for i in range(warp.shape[0]):
        if i == 0:
            flow = grid_large.cpu() + optic[i, 0:2, :, :].unsqueeze(0).permute((0, 2, 3, 1))
            optic[i, 0:2, :, :] = optic[i, 0:2, :, :] + grid_sample(warp[i, :, :, :].unsqueeze(0), flow).squeeze()
        elif i == warp.shape[0] - 1:
            optic[i, 0:2, :, :] = optic[i, 0:2, :, :] - warp[i - 1, :, :, :]
        else:
            flow = grid_large.cpu() + optic[i, 0:2, :, :].unsqueeze(0).permute((0, 2, 3, 1))
            optic[i, 0:2, :, :] = optic[i, 0:2, :, :] - warp[i - 1, :, :, :] + grid_sample(warp[i, :, :, :].unsqueeze(0), flow).squeeze()
    return optic


def detect_points(im1, im2):
    im1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(2500)
    keypoints1, descriptors1 = orb.detectAndCompute(im1_gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2_gray, None)
    #    sift = cv2.xfeatures2d.SURF_create(2500)
    #    keypoints1, descriptors1 = sift.detectAndCompute(im1Gray,None)
    #    keypoints2, descriptors2 = sift.detectAndCompute(im2Gray,None)
    if len(keypoints1) == 0 or len(keypoints2) == 0:
        return None, None

    matcher = cv2.BFMatcher(cv2.NORM_L1, crossCheck=False)
    matches = matcher.match(descriptors1, descriptors2, None)

    matches.sort(key=lambda x: x.distance, reverse=False)

    num_good_matches = int(len(matches) * 0.15)
    matches = matches[:num_good_matches]
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)
    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt
    return points1, points2


def rigid_transform_2d(im1, im2):
    A, B = detect_points(im1, im2)
    if A is None:
        return np.eye(2), np.zeros((2, 1)), 1
    if B is None:
        return np.eye(2), np.zeros((2, 1)), 1

    N = A.shape[0]
    if N < 10:
        return np.eye(2), np.zeros((2, 1)), 1

    centroid_a = np.mean(A, axis=0)
    centroid_b = np.mean(B, axis=0)

    AA = A - np.tile(centroid_a, (N, 1))
    BB = B - np.tile(centroid_b, (N, 1))

    H = np.matmul(np.transpose(AA), BB)

    U, _, Vt = np.linalg.svd(H) # second variable S unused

    R = np.matmul(Vt.T, U.T)

    if np.linalg.det(R) < 0:
        Vt[1, :] *= -1
        R = np.matmul(Vt.T, U.T)
        print(np.linalg.det(R))

    t = centroid_b.T - centroid_a.T

    pt = np.matmul(AA, R) / BB
    pt = pt[np.abs(pt - 1) < 0.1]
    ss = np.mean(pt)
    if ss < 0:
        print('error')

    return R, t, ss


def compute_h(path):
    print(path)
    img_mask = '*.png'
    all_images = sorted(glob.glob(os.path.join(path, img_mask)))
    numframe = len(all_images)
    H = np.zeros((numframe + N_FRAME - 1, 3, 3))
    H[:, 0, 0] = 1.0
    H[:, 1, 1] = 1.0
    H[:, 2, 2] = 1.0
    h_inv = np.zeros((numframe + N_FRAME - 1, 3, 3))
    h_inv[:, 0, 0] = 1.0
    h_inv[:, 1, 1] = 1.0
    h_inv[:, 2, 2] = 1.0
    R = np.zeros((numframe + N_FRAME - 1, 1))
    tx = np.zeros((numframe + N_FRAME - 1, 1))
    ty = np.zeros((numframe + N_FRAME - 1, 1))
    s = np.ones((numframe + N_FRAME - 1, 1))

    flag = 0
    i = 0
    all_frames = sorted(glob.glob(os.path.join(path, img_mask)))
    for iiiii in all_frames:
        frame = cv2.imread(iiiii)
        if flag == 0:
            im1_ori = cv2.resize(frame, (832, 448))
            if np.max(im1_ori) != 0:
                flag = 1
        else:

            im2_ori = cv2.resize(frame, (832, 448))

            rr, tt, ss = rigid_transform_2d(im1_ori, im2_ori)

            if np.isnan(np.min(rr)) or np.isnan(np.min(tt)) or np.isnan(np.min(ss)):
                tx[i + 1] = tx[i]
                ty[i + 1] = ty[i]
                R[i + 1] = R[i]
                s[i + 1] = s[i]
            else:
                tx[i + 1] = tx[i] + tt[0]
                ty[i + 1] = ty[i] + tt[1]
                R[i + 1] = R[i] + np.arctan2(rr[1, 0], rr[0, 0])
                s[i + 1] = s[i] * ss

            i += 1
            im1_ori = im2_ori

    H = np.concatenate((H[:i, :, :], np.zeros((N_FRAME, 3, 3))), 0)
    R = np.concatenate((R[:i], R[i - 1] * np.ones((N_FRAME, 1))), 0)
    tx = np.concatenate((tx[:i], tx[i - 1] * np.ones((N_FRAME, 1))), 0)
    ty = np.concatenate((ty[:i], ty[i - 1] * np.ones((N_FRAME, 1))), 0)
    s = np.concatenate((s[:i], s[i - 1] * np.ones((N_FRAME, 1))), 0)

    r_smooth = moving_average(R, 3)
    tx_smooth = moving_average(tx, 20)
    ty_smooth = moving_average(ty, 5)
    s_smooth = moving_average(s, 3)

    tx_move = (tx_smooth - tx)
    ty_move = (ty_smooth - ty)

    for i in range(H.shape[0] - 1):
        RR = r_smooth[i + 1] - R[i + 1]

        M1 = np.float32([[1, 0, -tx[i + 1] - MARGIN - 416], [0, 1, -ty[i + 1] - MARGIN - 224], [0, 0, 1]])
        M11 = np.float32([[s[i + 1] / s_smooth[i + 1], 0, 0], [0, s[i + 1] / s_smooth[i + 1], 0], [0, 0, 1]])
        M2 = np.float32([[np.cos(RR), -np.sin(RR), 0], [np.sin(RR), np.cos(RR), 0], [0, 0, 1]])
        M3 = np.float32([[1, 0, tx[i + 1] + MARGIN + 416], [0, 1, ty[i + 1] + MARGIN + 224], [0, 0, 1]])
        M4 = np.float32([[1, 0, tx_move[i + 1]], [0, 1, ty_move[i + 1]], [0, 0, 1]])
        H[i, :, :] = np.matmul(M4, np.matmul(M3, np.matmul(M2, np.matmul(M11, M1))))
        if np.isnan(np.min(H[i, :, :])):
            H[i, :, :] = np.zeros((3, 3))
            H[i, 0, 0] = 1.0
            H[i, 1, 1] = 1.0
            H[i, 2, 2] = 1.0
        h_inv[i, :, :] = np.linalg.inv(H[i, :, :])
    return np.concatenate((np.expand_dims(np.eye(3), 0), H), 0), np.concatenate((np.expand_dims(np.eye(3), 0), h_inv), 0)


with torch.no_grad():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    flow_model = models_flownet.FlowNet2().cuda()
    checkpoint = torch.load('FlowNet2_checkpoint.pth.tar')  #
    flow_model.load_state_dict(checkpoint['state_dict'])
    flow_model.eval()

    xv, yv = np.meshgrid(np.linspace(-1, 1, 832 + 2 * MARGIN), np.linspace(-1, 1, 448 + 2 * MARGIN))
    xv = np.expand_dims(xv, axis=2)
    yv = np.expand_dims(yv, axis=2)
    grid = np.expand_dims(np.concatenate((xv, yv), axis=2), axis=0)
    grid_large = np.repeat(grid, 1, axis=0)
    grid_large = Variable(torch.from_numpy(grid_large).float().cuda(), requires_grad=False)

    xv, yv = np.meshgrid(np.linspace(1, 832, 832), np.linspace(1, 448, 448))
    xv = np.expand_dims(xv, axis=2)
    yv = np.expand_dims(yv, axis=2)
    grid_full = np.concatenate((xv, yv), axis=2)
    grid_full = np.reshape(grid_full, (448 * 832, 2))

    U = np.load('PC_U.npy')
    V = np.load('PC_V.npy')
    U1 = np.zeros((N_KEEP, (832 + 2 * MARGIN) * (448 + 2 * MARGIN)))
    V1 = np.zeros((N_KEEP, (832 + 2 * MARGIN) * (448 + 2 * MARGIN)))
    for i in range(0, N_KEEP):
        temp = U[i, :].reshape((256, 512))
        temp = cv2.resize(temp, ((832 + 2 * MARGIN), (448 + 2 * MARGIN)))
        U1[i, :] = temp.reshape(1, (832 + 2 * MARGIN) * (448 + 2 * MARGIN))
        temp = V[i, :].reshape((256, 512))
        temp = cv2.resize(temp, ((832 + 2 * MARGIN), (448 + 2 * MARGIN)))
        V1[i, :] = temp.reshape(1, (832 + 2 * MARGIN) * (448 + 2 * MARGIN))
    U = torch.from_numpy(U1).float()
    V = torch.from_numpy(V1).float()
    base = torch.cat((U, V), 0).t()

    model = torch.nn.DataParallel(mpi_net35.mpinet(in_channels=60, out_channels=38).cuda())
    if os.path.isfile('checkpt.pt'):
        model.load_state_dict(torch.load('checkpt.pt'))
        print('loaded!!!!!!!!!!!!!!!!!')
    model.eval()

    video_path = input_file
    H, H_inv = compute_h(video_path)
    img_mask = '*.png'
    if not os.path.exists(out_warping_field_path):
        os.makedirs(out_warping_field_path)
    for iii in range(H_inv.shape[0]):
        np.save(out_warping_field_path + str(iii).zfill(5) + '_H_inv.npy', H_inv[iii, :, :])

    all_images = sorted(glob.glob(os.path.join(video_path, img_mask)))
    numframe = len(all_images)
    tmp_img = cv2.imread(all_images[0])
    height = tmp_img.shape[0]
    width = tmp_img.shape[1]
    nseg = np.ceil(float(numframe) / float(SEG))
    TOTAL_MAS = np.ones((448 + 2 * MARGIN, 832 + 2 * MARGIN))

    writer = cv2.VideoWriter(out_file + '.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 29, (832 + 2 * MARGIN, 448 + 2 * MARGIN))  # vid_reader.get(cv2.CAP_PROP_FPS)
    writer_mask = cv2.VideoWriter(out_file + '_mask.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 29, (832 + 2 * MARGIN, 448 + 2 * MARGIN))  # vid_reader.get(cv2.CAP_PROP_FPS)
    COUNTER = 1
    COUNTER2 = 1
    if not os.path.exists(out_warping_field_path):
        os.makedirs(out_warping_field_path)
    np.save(out_warping_field_path + str(0).zfill(5) + '.npy', torch.zeros(2, height, width).permute(1, 2, 0).cpu().numpy())
    for ite2 in range(int(nseg)):
        all_frames = sorted(glob.glob(os.path.join(video_path, img_mask)))

        SKIP = 0
        cu = 0
        video = np.zeros((SEG, height, width, 3))
        for iiiii in all_frames:
            frame = cv2.imread(iiiii)
            SKIP = SKIP + 1
            if SKIP >= ite2 * (SEG - 1) + 1:
                video[cu, :, :, :] = frame
                cu = cu + 1
            if cu == SEG:
                break
        video = video[:cu, :, :, :]
        if video.shape[0] == 0:
            break
        while np.max(video[0, :, :, :]) == 0:
            video = video[1:, :, :, :]

        res = np.zeros((video.shape[0], 448, 832, 3))
        for i in range(0, video.shape[0]):
            res[i, :, :, :] = cv2.resize(video[i, :, :, :], (832, 448))
        video = res.copy()

        realnframe = video.shape[0]

        del res

        optic, video, mask, mask_object = compute_flow_seg(video, H, ite2 * (SEG - 1))
        MASK = np.ones((video.shape[0], 1, 448 + 2 * MARGIN, 832 + 2 * MARGIN))

        if ite2 > 0:
            optic[0, 0:2, :, :] = optic[0, 0:2, :, :] - lastwarp.cpu()
            MASK[0, :, :, :] = lastmask

        ROU = 2
        warp_acc = torch.zeros(video.shape[0] - 2, 2, video.shape[1], video.shape[2]).cuda()
        for ITE1 in range(ROU):
            print(ITE1)
            warp = torch.zeros(video.shape[0] - 2, 2, video.shape[1], video.shape[2])
            result = torch.zeros(video.shape[0], video.shape[1], video.shape[2], 3)
            print(result.shape)
            warping_fields = torch.zeros(video.shape[0], 2, video.shape[1], video.shape[2])
            H_invs = np.zeros((video.shape[0], 3, 3))
            for i in range(0, video.shape[0] - N_FRAME):
                if i == 0:
                    optic_temp = optic[i:i + N_FRAME, :, :, :].cuda()
                    optic_temp[:, 2, :, :] = torch.from_numpy(mask[i:i + N_FRAME, :, :]).cuda()
                    optic_temp = optic_temp.view(optic_temp.shape[0] * optic_temp.shape[1], optic_temp.shape[2], optic_temp.shape[3]).unsqueeze(0)
                    oup = model(optic_temp)

                    oup = oup[0, 0:2, :, :].permute((1, 2, 0)).view((448 + 2 * MARGIN) * (832 + 2 * MARGIN), 2)
                    mask1 = optic[i + 1, 2, :, :].view(-1)
                    Q = base[mask1 > 0]
                    c = torch.mm(torch.mm((torch.mm(Q.t(), Q) + RHO * torch.eye(2 * N_KEEP)).inverse(), Q.t()), oup[mask1 > 0].cpu())
                    oup2 = torch.mm(base, c).cuda()
                    oup = oup2
                    oup = oup.view(448 + 2 * MARGIN, 832 + 2 * MARGIN, 2).permute((2, 0, 1)).unsqueeze(0)  # .cuda()

                    warp_acc[i, :, :, :] = warp_acc[i, :, :, :] + oup[:, 0:2, :, :].data
                    warp[i, :, :, :] = oup[:, 0:2, :, :].data.cpu()
                    warping_fields[i, :, :, :] = warp_acc[i, :, :, :]

                    if ITE1 == ROU - 1:
                        warp_acc[i, 0, :, :] = torch.from_numpy(gaussian_filter(warp_acc[i, 0, :, :].data.cpu().numpy(), sigma=3))
                        warp_acc[i, 1, :, :] = torch.from_numpy(gaussian_filter(warp_acc[i, 1, :, :].data.cpu().numpy(), sigma=3))
                        warping_fields[i, :, :, :] = warp_acc[i, :, :, :]

                        newimages = grid_sample(float_cuda(torch.from_numpy(np.transpose(video[i + 1, :, :, :], (2, 0, 1)))), grid_large + transform1(warp_acc[i, :, :, :])).data.cpu()
                        newmasks = grid_sample(float_cuda(torch.from_numpy(np.transpose(np.expand_dims(mask[i + 1, :, :], -1), (2, 0, 1)))), grid_large + transform1(warp_acc[i, :, :, :])).data.cpu()
                        MAS = grid_sample(float_cuda(torch.from_numpy(mask[i + 1, :, :]).unsqueeze(0)), grid_large + transform1(warp_acc[i, :, :, :])).data.cpu()
                        if ite2 > 0:
                            result[i, :, :, :] = grid_sample(float_cuda(torch.from_numpy(np.transpose(video[i, :, :, :], (2, 0, 1)))), grid_large + transform1(lastwarp.cuda())).data.cpu().permute(0, 2, 3, 1)
                            warping_fields[i, :, :, :] = warp_acc[i, :, :, :]
                        else:
                            result[i, :, :, :] = torch.from_numpy(video[i, :, :, :])
                            warping_fields[i, :, :, :] = warp_acc[i, :, :, :]
                        result[i + 1, :, :, :] = newimages.permute(0, 2, 3, 1)
                        MASK[i + 1, :, :, :] = MAS

                elif i == video.shape[0] - N_FRAME - 1:
                    optic_temp = optic[i:i + N_FRAME, :, :, :].cuda()
                    optic_temp[:, 2, :, :] = torch.from_numpy(mask[i:i + N_FRAME, :, :]).cuda()
                    optic_temp[0, 0:2, :, :] = optic_temp[0, 0:2, :, :] - oup[0, 0:2, :, :]
                    optic_temp = optic_temp.view(optic_temp.shape[0] * optic_temp.shape[1], optic_temp.shape[2], optic_temp.shape[3]).unsqueeze(0)
                    oup = model(optic_temp)

                    if ITE1 == ROU - 1:
                        print(video.shape)
                        newimages = np.transpose(video[i:i + N_FRAME + 1, :, :, :], (0, 3, 1, 2))
                        newmasks = np.transpose((np.expand_dims(mask, -1))[i:i + N_FRAME + 1, :, :, :], (0, 3, 1, 2))
                    for ite in range(0, N_FRAME - 1):

                        oup1 = oup[0, 2 * ite:2 * ite + 2, :, :].permute((1, 2, 0)).view((448 + 2 * MARGIN) * (832 + 2 * MARGIN), 2)
                        mask1 = optic[i + ite + 1, 2, :, :].view(-1)  # test_mask[i+1,0,:,:].view(-1)#
                        Q = base[mask1 > 0]
                        c = torch.mm(torch.mm((torch.mm(Q.t(), Q) + RHO * torch.eye(2 * N_KEEP)).inverse(), Q.t()), oup1[mask1 > 0].cpu())
                        oup2 = torch.mm(base, c).cuda()
                        oup1 = oup2
                        oup1 = oup1.view(448 + 2 * MARGIN, 832 + 2 * MARGIN, 2).permute((2, 0, 1)).unsqueeze(0)
                        warp[i + ite, :, :, :] = oup1[:, 0:2, :, :].data
                        warp_acc[i + ite, :, :, :] = warp_acc[i + ite, :, :, :] + oup1[:, 0:2, :, :].data
                        warping_fields[i + ite, :, :, :] = warp_acc[i + ite, :, :, :]

                        if ITE1 == ROU - 1:
                            warp_acc[i + ite, 0, :, :] = torch.from_numpy(gaussian_filter(warp_acc[i + ite, 0, :, :].data.cpu().numpy(), sigma=3))
                            warp_acc[i + ite, 1, :, :] = torch.from_numpy(gaussian_filter(warp_acc[i + ite, 1, :, :].data.cpu().numpy(), sigma=3))
                            MAS = grid_sample(float_cuda(torch.from_numpy(mask[i + ite + 1, :, :]).unsqueeze(0)), grid_large + transform1(warp_acc[i + ite, :, :, :])).data.cpu()
                            lastwarp = warp_acc[i - 1, :, :, :]
                            newimages[ite + 1, :, :, :] = grid_sample(float_cuda(torch.from_numpy(newimages[ite + 1, :, :, :])), grid_large + transform1(warp_acc[i + ite, :, :, :])).data.cpu()
                            MASK[i + 1, :, :, :] = MAS
                            newmasks[ite + 1, :, :, :] = grid_sample(float_cuda(torch.from_numpy(newmasks[ite + 1, :, :, :])), grid_large + transform1(warp_acc[i + ite, :, :, :])).data.cpu()

                            warping_fields[i, :, :, :] = warp_acc[i, :, :, :]
                    if ITE1 == ROU - 1:
                        result[i + 1:i + N_FRAME + 1, :, :, :] = torch.from_numpy(np.transpose(newimages[1:, :, :, :], (0, 2, 3, 1)))
                        MASK[i + 1:i + N_FRAME + 1, :, :, :] = newmasks[1:, :, :, :]
                        warping_fields[i + 1:i + N_FRAME + 1, :, :, :] = warp_acc[1, :, :, :]
                else:
                    optic_temp = optic[i:i + N_FRAME, :, :, :].cuda()
                    optic_temp[:, 2, :, :] = torch.from_numpy(mask[i:i + N_FRAME, :, :]).cuda()
                    optic_temp[0, 0:2, :, :] = optic_temp[0, 0:2, :, :] - oup[0, 0:2, :, :]
                    optic_temp = optic_temp.view(optic_temp.shape[0] * optic_temp.shape[1], optic_temp.shape[2], optic_temp.shape[3]).unsqueeze(0)
                    oup = model(optic_temp)

                    oup = oup[0, 0:2, :, :].permute((1, 2, 0)).view((448 + 2 * MARGIN) * (832 + 2 * MARGIN), 2)
                    mask1 = optic[i + 1, 2, :, :].view(-1)  # test_mask[i+1,0,:,:].view(-1)#
                    Q = base[mask1 > 0]
                    c = torch.mm(torch.mm((torch.mm(Q.t(), Q) + RHO * torch.eye(2 * N_KEEP)).inverse(), Q.t()), oup[mask1 > 0].cpu())
                    oup2 = torch.mm(base, c).cuda()
                    oup = oup2
                    oup = oup.view(448 + 2 * MARGIN, 832 + 2 * MARGIN, 2).permute((2, 0, 1)).unsqueeze(0)

                    warp[i, :, :, :] = oup[:, 0:2, :, :].data.cpu()
                    warp_acc[i, :, :, :] = warp_acc[i, :, :, :] + oup[:, 0:2, :, :].data

                    warping_fields[i, :, :, :] = warp_acc[i, :, :, :]

                    if ITE1 == ROU - 1:
                        warp_acc[i, 0, :, :] = torch.from_numpy(gaussian_filter(warp_acc[i, 0, :, :].data.cpu().numpy(), sigma=3))
                        warp_acc[i, 1, :, :] = torch.from_numpy(gaussian_filter(warp_acc[i, 1, :, :].data.cpu().numpy(), sigma=3))

                        MAS = grid_sample(float_cuda(torch.from_numpy(mask[i + 1, :, :]).unsqueeze(0)), grid_large + transform1(warp_acc[i, :, :, :])).data.cpu()
                        newimages = grid_sample(float_cuda(torch.from_numpy(np.transpose(video[i + 1, :, :, :], (2, 0, 1)))), grid_large + transform1(warp_acc[i, :, :, :])).data.cpu()
                        newmasks = grid_sample(float_cuda(torch.from_numpy(np.transpose(np.expand_dims(mask[i + 1, :, :], -1), (2, 0, 1)))), grid_large + transform1(warp_acc[i, :, :, :])).data.cpu()

                        warping_fields[i, :, :, :] = warp_acc[i, :, :, :]
                        H_invs[i, :, :] = H_inv[i + ite2 * (SEG - 1) - 1, :, :]
                        result[i + 1, :, :, :] = newimages.permute(0, 2, 3, 1)
                        MASK[i + 1, :, :, :] = MAS

            optic = modify_flow(optic, warp, grid_large)

        del optic, warp, mask, video

        original_shape = result.shape[0]

        result = result[:result.shape[0] - N_FRAME - 1, :, :, :].numpy()

        H_invs = H_invs[1:]

        MASK = MASK[:original_shape - N_FRAME - 1, :, :, :]
        totalmask = np.prod(MASK, axis=0)
        TOTAL_MAS = TOTAL_MAS * totalmask[0, :, :]

        if ite2 < int(nseg) - 1:
            lastframe = result[-1, :, :, :].copy()
            lastmask = MASK[-1, :, :, :].copy()
        else:
            totalmask = np.prod(MASK, axis=0)
            TOTAL_MAS = TOTAL_MAS * totalmask[0, :, :]

        for i in range(result.shape[0]):
            writer.write(result[i, :, :, :].astype(np.uint8))
            if not os.path.exists(out_warping_field_path):
                os.makedirs(out_warping_field_path)
            np.save(out_warping_field_path + str(COUNTER).zfill(5) + '.npy', warping_fields[i, :, :, :].permute(1, 2, 0).cpu().numpy())
            COUNTER += 1

        del result
    writer.release()
    cv2.imwrite(out_file + '.png', TOTAL_MAS * 255)
