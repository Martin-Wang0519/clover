import os
import shutil
import zipfile
from utils.utils import sha256_str, clear_folder, folder_RMZ_crop_save


class ImportImages(object):
    def __init__(self, src_path, dst_path, RMZ_crop_zone):
        super().__init__()
        self.src_path = src_path
        self.dst_path = dst_path
        self.RMZ_crop_zone = RMZ_crop_zone
        self.dst_name_set = self.__get_name_set()

    def __get_name_set(self):
        name_set = set()
        for sub_path in os.listdir(self.dst_path):
            image_name = sub_path.split(".")[0]
            name_set.add(image_name)
        return name_set

    def __unzip_folder(self):
        for sub_path in os.listdir(self.src_path):
            path = os.path.join(self.src_path, sub_path)
            if path.endswith(".zip"):
                with zipfile.ZipFile(path, 'r') as fz:
                    new_dir = path.split('.')[0]
                    os.mkdir(new_dir)
                    for file in fz.namelist():
                        fz.extract(file, new_dir)
                    folder_RMZ_crop_save(new_dir, self.RMZ_crop_zone)
                    self.__move_from(new_dir)

        clear_folder(self.src_path)

    def __move_from(self, src_path):
        for sub_path in os.listdir(src_path):
            path = os.path.join(src_path, sub_path)
            if os.path.isdir(path):
                self.__move_from(path)
            elif os.path.isfile(path):
                sha_str = sha256_str(path)
                if sha_str in self.dst_name_set:
                    os.remove(path)
                    continue
                else:
                    self.dst_name_set.add(sha_str)
                    new_name = "{}.jpg".format(sha_str)
                    new_path = os.path.join(src_path, new_name)
                    os.rename(path, new_path)
                    shutil.move(new_path, self.dst_path)

        os.rmdir(src_path)

    def import_from_image_folder(self):
        folder_RMZ_crop_save(self.src_path, self.RMZ_crop_zone)
        self.__move_from(self.src_path)

    def import_from_zip_folder(self):
        self.__unzip_folder()


if __name__ == '__main__':
    pass
