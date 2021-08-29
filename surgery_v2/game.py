import copy

import numpy as np
import os
from .designer import LevelGenerator, Patient
from .designer.level import wound_idx_2_str
import tkinter
from PIL import Image, ImageTk


class Environment:
    def __init__(self):
        self.initial = Patient()
        self.level = copy.deepcopy(self.initial)

    def step(self, action, params):
        # action is array already, params depending on action.
        # action is one hot, params is regression
        ...

    def reset(self):
        self.level = copy.deepcopy(self.level)


class Game:
    def __init__(self):
        self.level = LevelGenerator()
        self.tk_root = tkinter.Tk()
        self.tk_root.title("Surgery V2")
        self.tk_canvas = tkinter.Canvas(self.tk_root, width=702, height=510)
        self.tk_canvas.pack()
        self.img_tk = ImageTk.PhotoImage(self.level.p.render())
        self.tk_canvas.create_image(self.img_tk.width() // 2, self.img_tk.height() // 2, image=self.img_tk)
        self.tk_canvas.update()
        self.tk_canvas.bind("<Button-1>", self._on_body_click)

    def spin(self):
        self.tk_root.mainloop()

    def continue_wound(self, event):
        print(self.body_part.wound_files)
        ret = self.body_part.advance_wound(0)
        if ret:
            self.img_tk = ImageTk.PhotoImage(self.level.p.render())
            self.tk_canvas.create_image(self.img_tk.width() // 2, self.img_tk.height() // 2, image=self.img_tk)
            self.tk_canvas.update()
            self.tk_canvas.bind("<Button-1>", self._on_body_click)
            return
        self.img_tk = ImageTk.PhotoImage(self.body_part.render())
        self.tk_canvas.create_image(self.img_tk.width() // 2, self.img_tk.height() // 2, image=self.img_tk)
        self.tk_canvas.update()

    def _on_body_click(self, event):
        i = 0
        inbbox = False
        for i, bbox in enumerate(self.level.p.bbox):
            if (bbox[0] <= event.x <= (bbox[0] + bbox[2])) and (bbox[1] <= event.y <= (bbox[1] + bbox[3])) and \
                    self.level.p.body_parts[i].wound_files:
                # print(self.level.p.patches_files[i])
                print(f"Clicked on patch '{os.path.split(self.level.p.patches_files[i])[1]}'. "
                      f"BBox: {bbox}, Coords: [{event.x}, {event.y}]")
                inbbox = True
                break
        if not inbbox:
            return

        self.tk_canvas.bind("<Button-1>", self._on_body_part_click)
        self.img_tk = ImageTk.PhotoImage(self.level.p.body_parts[i].render())
        self.body_part = self.level.p.body_parts[i]
        self.tk_canvas.create_image(self.img_tk.width() // 2, self.img_tk.height() // 2, image=self.img_tk)
        self.tk_canvas.update()

    def _on_body_part_click(self, event):
        # Test for tool selection
        # Use different brush for different tool
        mask = self.body_part.canvas_mask_to_ndarray()
        name = wound_idx_2_str(np.argmax(mask[event.y, event.x]))
        if name:
            print(f"Clicked on wound {name} - advancing wound.")
            for i, bbox in enumerate(self.level.p.bbox):
                if (bbox[0] <= event.x <= (bbox[0] + bbox[2])) and (bbox[1] <= event.y <= (bbox[1] + bbox[3])):
                    break
            self.body_part.advance_wound(i)

