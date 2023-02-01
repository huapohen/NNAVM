def to_cuda(params, data_batch):
    task_mode = params.dataloader_task_mode
    if params.cuda:
        data_batch["image"] = data_batch["image"].cuda()
        for item in ['fev', 'undist', 'bev_origin', 'bev_perturbed']:
            if item in task_mode:
                try:
                    data_batch[item] = [ele.cuda() for ele in data_batch[item]]
                except:
                    pass  # input undist, BEVTransformer
        if 'offset' in task_mode:
            data_batch['offset'] = data_batch['offset'].cuda()
        if 'coords' in task_mode:
            for item in ["coords_undist", "coords_bev_origin", "coords_bev_perturbed"]:
                data_batch[item] = data_batch[item].cuda()

    return data_batch
