import os

import threading

from utils.utils import ocr, RMZ_crop_save, logger


class ShotImageProcess(threading.Thread):
    def __init__(self, task_queue, save_path, ocr_crop_zone, RMZ_crop_zone):
        threading.Thread.__init__(self)
        self._stop_event = threading.Event()
        self.save_path = save_path
        self.task_queue = task_queue
        self.ocr_crop_zone = ocr_crop_zone
        self.RMZ_crop_zone = RMZ_crop_zone
        self.logger = logger()

    def set_save_path(self, save_path):
        self.save_path = save_path

    def grab_save(self, grab_image):
        stock_num = ocr(grab_image, self.ocr_crop_zone)
        if not stock_num.isdigit():
            return
        image_name = "{}.jpg".format(stock_num)
        RMZ_crop_save(grab_image, os.path.join(self.save_path, image_name), self.RMZ_crop_zone)

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()

    def run(self):
        while True:
            if self.stopped():
                break
            grad_image = self.task_queue.get()
            self.grab_save(grad_image)
            self.task_queue.task_done()

    def get_enumerate(self):
        return len(threading.enumerate())


if __name__ == '__main__':
    pass
