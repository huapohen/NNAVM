import os
import cv2
import tqdm
import torch
import threading
from easydict import EasyDict


class ThreadsOP(threading.Thread):
    def __init__(self, func, args):
        super().__init__()
        self.func = func
        self.args = args

    def run(self):
        self.result = self.func(*self.args)

    def get_result(self):
        '''DIY function'''
        try:
            return self.result
        except Exception:
            return None


class MultiThreadsProcess:
    '''
    output: result_list
    class `ThreadsOP`:
        which is used for single thread process

    func_preprocess:
        input: tuple(i, j, *input_values)
        output: tuple(v1, ...) at least one return value
    func_thread_kernel:
        input: tuple(v1, ...) at least one input value
        output: None or tuple(v1, ...)
    func_postprocess:
        input: None or tuple(v1, ...)
        output: None

    method of applicatin:
        step 1. init class
            self.mtp = MultiThreadsProcess()
        step 2. write template pipeline function
            (a) def _threads_kernel; (b) def _preprocess;
            (c) def _postprocess (if need)
        step 3. call multiprocessing function
            input_values = (v1, v2, ...) # at least one value
            self.mtp.multi_threads_process(
                input_values=input_values,
                batch_size=len(dst),
                threads_num=self.threads_num,
                func_thread_kernel=_threads_kernel,
                func_preprocess=_preprocess,
                func_postprocess=_postprocess,
            )
    '''

    def __init__(self):
        super().__init__()

    def multi_threads_process(
        self,
        input_values,
        batch_size,
        threads_num,
        func_thread_kernel,
        func_preprocess,
        func_postprocess=None,
    ):
        # multiprocessing
        if batch_size < threads_num:
            loop_num = 1
            threads_num = max(1, batch_size)
        else:
            assert batch_size % threads_num == 0
            loop_num = int(batch_size / threads_num)
        # pipeline
        result_list = []
        for i in range(loop_num):
            threads = []
            for j in range(threads_num):
                # preprocess
                args = func_preprocess(i, j, *input_values)
                # input
                thr = ThreadsOP(func_thread_kernel, args)
                threads.append(thr)
            # process
            for thr in threads:
                thr.start()
            for thr in threads:
                thr.join()
            for thr in threads:
                # postprocess
                res = thr.get_result()
                if func_postprocess is not None:
                    result_list.append(func_postprocess(res))
                else:
                    result_list.append(res)

        return result_list


class ExampleMTP:
    def __init__(self, params=None):
        super().__init__()
        assert params and len(params)
        # ['front', 'back', 'left', 'right']
        self.camera_list = params.camera_list
        assert len(self.camera_list) > 0
        self.threads_num = 4
        self.mode = params.mode
        assert self.mode in ['read', 'write']
        self.save_dir = params.save_dir
        if self.mode == 'read':
            self.read_dir = params.read_dir
            self.scale_ratio = params.scale_ratio
        self.enable_cuda = params.enable_cuda
        self.mtp = MultiThreadsProcess()

    def forward(self, dst=None, names=None):
        assert names is not None

        def _threads_kernel(args):
            if args[0] == 'read':
                # mode, read_path, scale_ratio
                assert len(args) == 3
                img = cv2.imread(args[1])
                if args[2] != 1.0:
                    wh = tuple([int(x * args[2]) for x in img.shape[:2]][::-1])
                    img = cv2.resize(img, wh, interpolation=cv2.INTER_AREA)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = torch.from_numpy(img).unsqueeze(0)
                return tuple(args[0], img)
            elif args[0] == 'write':
                # mode, sv_path, img
                assert len(args) == 3
                is_success = cv2.imwrite(args[1], args[2])
                return tuple(args[0], is_success)

        def _preprocess(i, j, args):
            if args[0] == 'read':
                # mode, threads_num, names, read_dir, scale_ratio
                assert len(args) == 5
                name = args[2][i * args[1] + j]
                read_path = os.path.join(args[3], name)
                img = cv2.imread(read_path)
                return tuple(args[0], img, args[4])
            elif args[0] == 'write':
                # mode, threads_num, dst, names, save_dir
                assert len(args) == 5
                img = args[2][i * args[1] + j]
                name = args[3][i * args[1] + j]
                sv_path = os.path.join(args[4], name)
                return tuple(args[0], sv_path, img)
            else:
                raise ValueError

        def _postprocess(args):
            if args[0] == 'read':
                # mode, images
                assert len(args) == 2
                return args[1]
            elif args[0] == 'write':
                # mode, is_success
                assert len(args) == 2
                if not args[1]:
                    raise Exception("imwrite error")
                return None

        result = {}
        with tqdm(total=len(self.camera_list)) as t:
            for camera in tqdm(self.camera_list):
                print(camera)
                if self.mode == 'read':
                    input_values = (self.threads_num, names, self.read_dir)
                elif self.mode == 'write':
                    input_values = (self.threads_num, dst, names, self.save_dir)
                mtp_result = self.mtp.multi_threads_process(
                    input_values=input_values,
                    batch_size=len(names) if dst is None else len(dst),
                    threads_num=self.threads_num,
                    func_thread_kernel=_threads_kernel,
                    func_preprocess=_preprocess,
                    func_postprocess=_postprocess,
                )
                if self.mode == 'read':
                    img_batch = torch.stack(mtp_result, dim=0)
                    if self.enable_cuda:
                        img_batch = img_batch.cuda()
                    result[camera] = img_batch
                else:
                    result = None
                t.update()

        return result


if __name__ == '__main__':

    # for multiprocessing
    params = EasyDict(dict())
    mtp = ExampleMTP(params)
    mtp.forward()
