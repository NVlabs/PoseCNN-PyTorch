# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

import subprocess
import os


def get_available_devices():
    executable_path = os.path.join(os.path.dirname(__file__), 'build')

    num_devices = int(subprocess.check_output(
        ["{}/query_devices".format(executable_path)]))

    available_devices = []
    for i in range(num_devices):
        try:
            if b"NVIDIA" in subprocess.check_output(["{}/test_device".format(executable_path),  str(i)]):
                available_devices.append(i)
        except subprocess.CalledProcessError as e:
            print(e)
    return(available_devices)


if __name__ == '__main__':
    print(get_available_devices())
