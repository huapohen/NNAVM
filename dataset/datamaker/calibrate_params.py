import ipdb
from easydict import EasyDict


class CalibrateParameter:
    '''
    calib_param = EasyDict(CalibrateParameter().__dict__)
    other_param.__dict__.update(calib_param)
    fish_scale: previous 0.5, current 1.0
    '''

    def __init__(self, fish_scale=1.0):
        self.fish_scale = fish_scale
        self._init_undistored_calibrate_parameters()
        self._init_avm_calibrate_paramters()

    def _init_avm_calibrate_paramters(self):
        self.calibrate = {
            "shift_w": 310,
            "shift_h": 170,
            "spread_w": 460,
            "spread_h": 740,
            "inn_shift_w": 30,
            "inn_shift_h": 20,
            "rectangle_length": 100,
            "detected_points_json_name": "detected_points.json",
            "detected_points_write_flag": False,
        }
        self.bev_wh_fblr = {
            "front": [1078, 336],
            "back": [1078, 336],
            "left": [1172, 439],
            "right": [1172, 439],
        }
        self.avm_resolution = {"w": 616, "h": 670, "scale": 1.75}

    def _init_undistored_calibrate_parameters(self):
        scale = self.new_scale_for_undist = self.fish_scale
        self.scale_previous_value = 0.5
        fish = {"scale": scale, "width": 1280, "height": 960}
        hardware = {"focal_length": 950, "dx": 3, "dy": 3, "cx": 640, "cy": 480}
        distort = {
            "Opencv_k0": 0.117639891128,
            "Opencv_k1": -0.0264845591715,
            "Opencv_k2": 0.0064761037844,
            "Opencv_k3": -0.0012833025037,
            "undis_scale": 3.1,
        }
        focal_len = hardware["focal_length"]
        self.dx = dx = hardware["dx"] / fish["scale"]
        self.dy = dy = hardware["dy"] / fish["scale"]
        self.fish_width = distort_width = int(fish["width"] * fish["scale"])
        self.fish_height = distort_height = int(fish["height"] * fish["scale"])
        undis_scale = distort["undis_scale"]
        self.distort = distort
        self.center_w = center_w = distort_width / 2
        self.center_h = center_h = distort_height / 2
        self.intrinsic = intrinsic = [
            [focal_len / dx, 0, center_w],
            [0, focal_len / dy, center_h],
            [0, 0, 1],
        ]
        self.intrinsic_undis = intrinsic_undis = [
            [focal_len / dx, 0, center_w * undis_scale],
            [focal_len / dy, 0, center_h * undis_scale],
            [0, 0, 1],
        ]
        self.undist_w = u_w = int(distort_width * undis_scale)
        self.undist_h = u_h = int(distort_height * undis_scale)
        check1 = u_w == 1984 and u_h == 1488 and scale == 0.5
        check2 = u_w == 3968 and u_h == 2976 and scale == 1.0
        assert check1 or check2
        self.d = pow(
            pow(intrinsic_undis[0][2] / intrinsic[0][0], 2)
            + pow(intrinsic_undis[1][2] / intrinsic[1][1], 2),
            0.5,
        )


if __name__ == '__main__':

    params = {}
    calib_param = EasyDict(CalibrateParameter().__dict__)
    ipdb.set_trace()
    params.update(calib_param)
    print(params)
