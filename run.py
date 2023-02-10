from dataset.datamaker.bev_from_multi_fev import run_bev_from_multi_fev
from util.torch_func_op import unit_test_warp_image
from segment.inference import segment_inference


if __name__ == "__main__":

    # run_bev_from_multi_fev()
    # unit_test_warp_image()
    segment_inference()
