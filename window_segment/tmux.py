import os
import time
import ipdb

from subprocess import check_call


class TmuxOps:
    @classmethod
    def new_session(cls, session_name, first_name):
        # '''  tmux new-session -s a -n editor -d
        # test:  new_session('a','b')
        # '''
        print(f'session_name: {session_name}, first_name: {first_name}')
        os.system("tmux new-session -s {} -n {} -d".format(session_name, first_name))

    @classmethod
    def new_window(cls, session_name, window_name):
        # '''  tmux neww -a -n tool -t init
        # test:  new_session('a','b')  & new_window('a', 'c')
        # '''
        # os.system("tmux neww -a -n {} -t {}".format(window_name, session_name))
        cmd = "tmux neww -a -n {} -t {}".format(window_name, session_name)
        print(cmd)
        check_call(cmd, shell=True)

    @classmethod
    def switch_window(cls, session_name, window_name):
        # ''' tmux attach -t [session_name]
        # test:  new_session('a','b')  & new_window('a', 'c') & new_window('a', 'd') & switch_window('a', 'b')
        # '''
        os.system("tmux attach -t %s:%s" % (session_name, window_name))

    @classmethod
    def split_window(cls, session_name, window_name, h_v='h', panel_number=0):
        # ''' tmux split-window -h -t development
        # test:  new_session('a','b')  & new_window('a', 'c') & split_window('a', 'b', h_v='h', panel_number=0)
        # '''
        assert h_v in ['h', 'v']
        os.system(
            "tmux split-window -%s -t %s:%s.%s"
            % (h_v, session_name, window_name, panel_number)
        )

    @classmethod
    def split_window_by_2(cls, session_name, window_name):

        cls.split_window(session_name, window_name, h_v='v', panel_number=0)

    @classmethod
    def split_window_by_4(cls, session_name, window_name):

        cls.split_window(session_name, window_name, h_v='h', panel_number=0)
        cls.split_window(session_name, window_name, h_v='v', panel_number=0)
        cls.split_window(session_name, window_name, h_v='v', panel_number=2)

    @classmethod
    def split_window_by_6(cls, session_name, window_name):

        cls.split_window(session_name, window_name, h_v='h', panel_number=0)
        cls.split_window(session_name, window_name, h_v='h', panel_number=2)
        cls.split_window(session_name, window_name, h_v='v', panel_number=0)
        cls.split_window(session_name, window_name, h_v='v', panel_number=2)
        cls.split_window(session_name, window_name, h_v='v', panel_number=4)

    @classmethod
    def split_window_by_8(cls, session_name, window_name):

        cls.split_window_by_4(session_name, window_name)
        for i in range(4):
            cls.split_window(session_name, window_name, h_v='v', panel_number=i * 2)

    @classmethod
    def split_window_by_16(cls, session_name, window_name):

        cls.split_window_by_8(session_name, window_name)
        for i in range(8):
            cls.split_window(session_name, window_name, h_v='h', panel_number=i * 2)

    @classmethod
    def run_command(cls, session_name, window_name, panel_number=0, command_line='ls'):
        cmd = "tmux send-keys -t {}:{}.{} '{}' C-m".format(
            session_name, window_name, panel_number, command_line
        )
        print(cmd)
        check_call(cmd, shell=True)

    @classmethod
    def demo(cls):
        # tmux kill-session -t a
        # demo()
        session_name = 'a'
        window_name = 'c'
        cls.new_session(session_name, '1234567')
        cls.new_window(session_name, window_name)
        cls.split_window_by_2(session_name, window_name)
        for i in range(2):
            time.sleep(0.1)
            cls.run_command(session_name, window_name, i, command_line='ls')

    @classmethod
    def run_command_v2(
        cls, session_name, window_name, panel_number=0, command_line='ls', **kwargs
    ):
        for i in kwargs.keys():
            command_line += ' --%s %s' % (i, kwargs[i])
        cls.run_command(
            session_name,
            window_name,
            panel_number=panel_number,
            command_line=command_line,
        )

    @classmethod
    def run_task(cls, task_ls, task_name='demo', session_name='exp', start_id=0):
        N = len(task_ls)
        session_name = session_name
        window_number = 0
        ind = -1
        # new session -> new window -> split window
        end_id = start_id + len(task_ls) - 1
        first_name = f'{start_id}-{end_id}'
        cls.new_session(session_name, first_name)

        def create_window(window_number, panel_number=16):
            window_name = task_name + '_%s' % window_number
            cls.new_window(session_name, window_name)
            if panel_number == 16:
                print('create a window with %s panels' % panel_number)
                cls.split_window_by_16(session_name, window_name)
            elif panel_number == 8:
                print('create a window with %s panels' % panel_number)
                cls.split_window_by_8(session_name, window_name)
            elif panel_number == 6:
                print('create a window with %s panels' % panel_number)
                cls.split_window_by_6(session_name, window_name)
            elif panel_number == 4:
                print('create a window with %s panels' % panel_number)
                cls.split_window_by_4(session_name, window_name)
            elif panel_number == 2:
                print('create a window with %s panels' % panel_number)
                cls.split_window_by_2(session_name, window_name)
            elif panel_number == 1:
                print('create a window with %s panels' % panel_number)
            else:
                pass
            window_number += 1
            return window_number, window_name

        def run_16(data_ls, cnt, window_number):
            for _ in range(len(data_ls) // 16):
                # create window
                window_number, window_name = create_window(
                    window_number, panel_number=16
                )
                print(window_name)
                for j in range(16):
                    cnt += 1
                    if cnt >= N:
                        return cnt, window_number
                    cls.run_command(
                        session_name=session_name,
                        window_name=window_name,
                        panel_number=j,
                        command_line=data_ls[cnt],
                    )
                    print(window_name, data_ls[cnt])
            return cnt, window_number

        def run_one_window(data_ls, cnt, window_number, panel_number):
            window_number, window_name = create_window(
                window_number, panel_number=panel_number
            )
            print(window_name)
            for i in range(panel_number):
                cnt += 1
                if cnt >= N:
                    return cnt, window_number
                cls.run_command(
                    session_name=session_name,
                    window_name=window_name,
                    panel_number=i,
                    command_line=data_ls[cnt],
                )
                print(window_name, data_ls[cnt])
            return cnt, window_number

        if N > 16:
            ind, window_number = run_16(task_ls, cnt=ind, window_number=window_number)
        rest_number = N - ind - 1
        if rest_number > 8:
            ind, window_number = run_one_window(
                task_ls, cnt=ind, window_number=window_number, panel_number=16
            )
        elif rest_number > 4:
            ind, window_number = run_one_window(
                task_ls, cnt=ind, window_number=window_number, panel_number=8
            )
        elif rest_number > 2:
            ind, window_number = run_one_window(
                task_ls, cnt=ind, window_number=window_number, panel_number=4
            )
        elif rest_number > 0:
            ind, window_number = run_one_window(
                task_ls, cnt=ind, window_number=window_number, panel_number=2
            )
        else:
            pass


if __name__ == '__main__':
    t = TmuxOps()
    t.demo()
