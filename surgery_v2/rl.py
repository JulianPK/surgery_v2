import copy
import os
from types import SimpleNamespace

import numpy as np
import torch
import rlpyt.envs.base
from rlpyt import *
import torchvision.transforms as tf
import torchvision.transforms.functional as tff

from surgery_v2.designer import Patient
from surgery_v2.designer.level import wound_idx_2_str


class OperatingRoom(rlpyt.envs.base.Env):
    tensor_size = [256, 256]
    img_size = []

    def step(self, action: torch.Tensor):
        # [X, Y] coordinates of the click. [[0, 1], [0, 1]] intervals
        mouse = SimpleNamespace(x=int(action[0] * self.img_size[0]), y=int(action[1] * self.img_size[1]))
        done = all(not bp.wound_files for bp in self.level.body_parts)
        if done:
            return rlpyt.envs.base.EnvStep(self.state, 100, True, rlpyt.envs.base.EnvInfo())
        if self.body_part is None:
            return self.step_patient(mouse)
        else:
            return self.step_body(mouse)

    def step_patient(self, event):
        i = 0
        inbbox = False
        for i, bbox in enumerate(self.level.bbox):
            if (bbox[0] <= event.x <= (bbox[0] + bbox[2])) and (bbox[1] <= event.y <= (bbox[1] + bbox[3])) and \
                    self.level.body_parts[i].wound_files:
                # print(self.level.p.patches_files[i])
                print(f"Clicked on patch '{os.path.split(self.level.patches_files[i])[1]}'. "
                      f"BBox: {bbox}, Coords: [{event.x}, {event.y}]")
                inbbox = True
                break
        if not inbbox:
            return rlpyt.envs.base.EnvStep(self.state, 0, False, rlpyt.envs.base.EnvInfo())

        self.body_part = self.level.body_parts[i]
        self.img = self.body_part.render()
        self.state = self.img2tensor(self.img)
        return rlpyt.envs.base.EnvStep(self.state, 1, False, rlpyt.envs.base.EnvInfo())

    def step_body(self, event):
        mask = self.body_part.canvas_mask_to_ndarray()
        name = wound_idx_2_str(np.argmax(mask[event.y, event.x]))
        if name:
            print(f"Clicked on wound {name} - advancing wound.")
            for i, data in enumerate(self.body_part.wound_data):
                bbox = data["coords"]
                if (bbox[0] <= event.x <= (bbox[0] + bbox[2])) and (bbox[1] <= event.y <= (bbox[1] + bbox[3])):
                    break
            self.body_part.advance_wound(i)
            self.img = self.body_part.render()
            self.state = self.img2tensor(self.img)
            return rlpyt.envs.base.EnvStep(self.state, 1, False, rlpyt.envs.base.EnvInfo())
        return rlpyt.envs.base.EnvStep(self.state, 0, False, rlpyt.envs.base.EnvInfo())

    def reset(self):
        self.level = copy.deepcopy(self.level)
        self.img = self.level.render()
        self.state = self.img2tensor(self.img)
        self.body_part = None
        return rlpyt.envs.base.EnvStep(self.state, 0, False, rlpyt.envs.base.EnvInfo())
        # TODO Return state and reward etc. look up docs!

    @property
    def horizon(self):
        pass

    def __init__(self):
        super(OperatingRoom, self).__init__()
        self.initial = Patient()
        self.level = copy.deepcopy(self.initial)
        self.img_size = list(self.level.render().size)
        self.img = self.level.render()
        self.body_part = None
        self.img2tensor = tf.Compose([
            tf.Resize(self.img_size),
            tf.ToTensor()
        ])
        self.state = self.img2tensor(self.img)


def main():
    op = OperatingRoom()
    steps = torch.rand([10000, 2])
    for step in steps:
        ret = op.step(step)

        ...


if __name__ == '__main__':
    main()
