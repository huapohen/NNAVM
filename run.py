from dataset.datamaker.bev_from_multi_fev import *
from util.torch_func_op import unit_test_warp_image


def run_bev_from_multi_fev():
    generator = DataMakerTorch(enable_cuda=True)
    # for mode in ['gt_bev']:
    # for mode in ['test']:
    # for mode in ['gt_bev', 'test']:
    # for mode in ['train', 'test']:
    for mode in ['gt_bev', 'train', 'test']:
        generator.init_dataset_mode_info(mode)
        pts = generator.generate_perturbed_points()
        for i in tqdm(range(generator.iteration)):
            src, name = generator.get_src_images(i)
            generator.warp_perturbed_image(pts, src, name)
        generator.shutil_copy()


if __name__ == "__main__":

    # run_bev_from_multi_fev()
    unit_test_warp_image()
