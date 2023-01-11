def to_cuda(params, data_batch):
    task_mode = params.dataloader_task_mode
    if params.cuda:
        data_batch["image"] = data_batch["image"].cuda()
        if 'offset' in task_mode:
            data_batch['offset'] = data_batch['offset'].cuda()
        if 'undist' in task_mode:
            data_batch['undist'] = [ele.cuda() for ele in data_batch['undist']]
        if 'bev_origin' in task_mode:
            data_batch['bev_origin'] = [ele.cuda() for ele in data_batch['bev_origin']]
        if 'bev_perturbed' in task_mode:
            data_batch['bev_perturbed'] = [
                ele.cuda() for ele in data_batch['bev_perturbed']
            ]
        if 'coords' in task_mode:
            data_batch["coords_undist"] = data_batch["coords_undist"].cuda()
            data_batch["coords_bev_origin"] = data_batch["coords_bev_origin"].cuda()
            data_batch["coords_bev_perturbed"] = data_batch[
                "coords_bev_perturbed"
            ].cuda()

    return data_batch
