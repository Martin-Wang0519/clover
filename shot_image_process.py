import os

from threading import Thread

from utils.utils import ocr, RMZ_crop_save, logger


class ShotImageProcess(Thread):
    def __init__(self, task_queue, save_path, ocr_crop_zone, RMZ_crop_zone):
        Thread.__init__(self)
        self.save_path = save_path
        self.index_queue = task_queue
        self.ocr_crop_zone = ocr_crop_zone
        self.RMZ_crop_zone = RMZ_crop_zone
        self.logger = logger()

    def grab_save(self, grab_image):
        stock_num = ocr(grab_image, self.ocr_crop_zone)
        if not stock_num.isdigit():
            return
        image_name = "{}.jpg".format(stock_num)
        RMZ_crop_save(grab_image, os.path.join(self.save_path, image_name), self.RMZ_crop_zone)

    def run(self):
        while True:
            grad_image = self.index_queue.get()
            self.grab_save(grad_image)
            self.index_queue.task_done()


if __name__ == '__main__':
    pass
