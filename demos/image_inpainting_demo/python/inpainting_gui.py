"""
 Copyright (c) 2019-2024 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import cv2
import numpy as np


class InpaintingGUI:
    def __init__(self, srcImg, inpainter):
        self.wnd_name = "Inpainting demo (press H for help)"
        self.mask_color = (255, 0, 0)
        self.radius = 10
        self.old_point = None

        self.inpainter = inpainter

        self.img = cv2.resize(srcImg, (self.inpainter.input_width, self.inpainter.input_height))
        self.original_img = self.img.copy()
        self.label = ""
        self.mask = np.zeros((self.inpainter.input_height, self.inpainter.input_width, 1), dtype=np.float32)

        cv2.namedWindow(self.wnd_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.wnd_name, self.on_mouse)
        cv2.createTrackbar("Brush size", self.wnd_name, self.radius, 30, self.on_trackbar)
        cv2.setTrackbarMin("Brush size", self.wnd_name, 1)

        self.is_help_shown = False
        self.is_original_shown = False
        self.is_inpainted = False


    def on_mouse(self, _event, x, y, flags, _param):
        if flags == cv2.EVENT_FLAG_LBUTTON and not self.is_original_shown:
            if self.old_point is not None:
                cv2.line(self.mask, self.old_point, (x, y), 1, self.radius*2)
            cv2.circle(self.mask, (x, y), self.radius, 1, cv2.FILLED)
            self.old_point = (x, y)

            self.update_window()
        else:
            self.old_point = None


    def run(self):
        self.update_window()

        key = ""
        while key not in (27, ord('q'), ord('Q')):
            if key in (ord(" "), ord("\r")):
                self.is_original_shown = False
                self.show_info("Processing...")

                self.img[np.squeeze(self.mask, -1) > 0] = 0
                self.img = self.inpainter.process(self.img, self.mask)

                self.show_info("")
                self.mask[:, :, :] = 0
                self.is_inpainted=True
                self.update_window()
            elif key in (8, ord('c'), ord('C')): # Backspace or c
                self.is_original_shown = False
                self.mask[:, :, :] = 0
                self.update_window()
            elif key in (ord('r'), ord('R')):
                self.is_original_shown = False
                self.mask[:, :, :] = 0
                self.img = self.original_img.copy()
                self.is_inpainted=False
                self.update_window()
            elif key == ord('\t'):
                self.is_original_shown = not self.is_original_shown
                self.update_window()
            elif key in (ord('h'), ord('H')):
                if not self.is_help_shown:
                    self.show_info("Use mouse with LMB to paint\n"
                                   "Bksp or C to clear current mask\n"
                                   "Space or Enter to inpaint\n"
                                   "R to reset all changes\n"
                                   "Tab to show original image\n"
                                   "Esc or Q to quit")
                    self.is_help_shown = True
                else:
                    self.show_info("")
                    self.is_help_shown = False

            key = cv2.waitKey()


    def on_trackbar(self, x):
        self.radius = x


    def show_info(self, text):
        self.label = text
        self.update_window()
        cv2.waitKey(1) # This is required to actually paint window contents right away
        self.is_help_shown = False # Any other label removes help from the screen


    def update_window(self):
        pad = 10
        margin = 10

        if self.is_original_shown:
            backbuffer = self.original_img.copy()
            lbl_txt = "Original"
        else:
            backbuffer = self.img.copy()
            backbuffer[np.squeeze(self.mask, -1) > 0] = self.mask_color
            lbl_txt = ("Editing" if np.any(self.mask) else "Result") if self.is_inpainted else "Original"

        if self.label != "Processing...":
            sz = cv2.getTextSize(lbl_txt, cv2.FONT_HERSHEY_COMPLEX, 0.75, 1)[0]
            img_width = backbuffer.shape[1]
            label_area = backbuffer[margin:sz[1]+pad*2+margin, img_width-margin-(sz[0]+pad*2):img_width-margin]
            label_area //= 2
            cv2.putText(backbuffer, lbl_txt, (img_width-margin-sz[0]-pad, margin+sz[1]+pad), cv2.FONT_HERSHEY_COMPLEX, 0.75, (128, 255, 128))

        if self.label is not None and self.label != "":
            lines = self.label.split("\n")
            count = len(lines)
            w = max(cv2.getTextSize(line, cv2.FONT_HERSHEY_COMPLEX, 0.75, 1)[0][0] for line in lines) + pad*2
            line_h = cv2.getTextSize(lines[0], cv2.FONT_HERSHEY_COMPLEX, 0.75, 1)[0][1] + pad
            label_area = backbuffer[margin:line_h*count+pad*2+margin, margin:w+margin]
            label_area //= 2
            for i, line in enumerate(lines):
                cv2.putText(backbuffer, line, (pad+margin, margin+(i+1)*line_h), cv2.FONT_HERSHEY_COMPLEX, 0.75, (192, 192, 192))

        cv2.imshow(self.wnd_name, backbuffer)
