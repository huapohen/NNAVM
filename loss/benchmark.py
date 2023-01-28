import torch


def calc_point_indicator(params, data, indicator):
    offset = data['offset']
    output = data['offset_pred'].tanh() * params.max_shift_pixels
    batch_size = int(data['image'].shape[0] / len(params.camera_list))

    m_pixel_err = []
    for i, cam in enumerate(params.camera_list):
        pd = output[i * batch_size : (i + 1) * batch_size]
        gt = offset[i * batch_size : (i + 1) * batch_size]
        x2 = torch.square(pd[:, :, 0] - gt[:, :, 0])
        y2 = torch.square(pd[:, :, 1] - gt[:, :, 1])
        pix_err = torch.mean(torch.sqrt(x2 + y2))
        m_pixel_err.append(pix_err.unsqueeze(0))

    m_pixel_err = torch.cat(m_pixel_err).mean()

    hpr = m_pixel_err / params.max_shift_pixels
    homing_point_ratio = torch.tensor(1.0 - min(hpr, 1.0)).to(hpr)

    indicator["mean_pixel_err"] = m_pixel_err
    indicator["homing_point_ratio"] = homing_point_ratio

    return indicator


def cacl_psnr_indicator(params, data, indicator):
    '''
    # im1 和 im2 都为灰度图像, uint8 类型
    # method 1
    diff = im1 - im2
    mse = np.mean(np.square(diff))
    psnr = 10 * np.log10(255 * 255 / mse)
    # method 2
    psnr = skimage.measure.compare_psnr(im1, im2, 255)
    '''
    gt = data['bev_origin']
    pd = data['bev_origin_pred']
    m_psnr = []
    for i, cam in enumerate(params.camera_list):
        mse = torch.square(pd[i] / 255.0 - gt[i] / 255.0).float().mean()
        if mse != torch.zeros_like(mse):
            psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
            # psnr = 10 * torch.log10(1.0 / mse)
        else:
            psnr = 100 * torch.ones_like(mse)
        m_psnr.append(psnr.unsqueeze(0))
    m_psnr = torch.cat(m_psnr).mean()
    indicator['psnr'] = m_psnr
    return indicator


def indicator_supervised(params, data, indicator):

    indicator = calc_point_indicator(params, data, indicator)
    indicator = cacl_psnr_indicator(params, data, indicator)

    return indicator


def indicator_unsupervised(params, data, indicator):

    indicator = calc_point_indicator(params, data, indicator)
    indicator = cacl_psnr_indicator(params, data, indicator)

    return indicator


def benchmark_indicator(params, data):
    indicator = {}
    if params.model_train_type == 'supervised':
        indicator = indicator_supervised(params, data, indicator)
    elif params.model_train_type == 'unsupervised':
        indicator = indicator_unsupervised(params, data, indicator)
    else:
        raise ValueError

    return indicator
