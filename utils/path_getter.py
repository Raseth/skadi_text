import logging
import os


class PathGetter:
    @staticmethod
    def inception_v3_model_path(): 
        base = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(base, "jj_model/files/inceptionV3")

    @staticmethod
    def setup_file_path(category_id, product_id, destination, main_directory):
        main_directory_path = os.path.join(destination, main_directory)
        if not os.path.exists(main_directory_path):
            os.makedirs(main_directory_path)

        category_id = str(category_id)
        category_path = os.path.join(main_directory_path, category_id)
        if not os.path.exists(category_path):
            os.makedirs(category_path)

        file_name = product_id
        file_path = os.path.join(category_path, file_name)
        if os.path.isfile(file_path):
            logging.info("[blending] File %s already exists!", file_name)
            return None
        return file_path
