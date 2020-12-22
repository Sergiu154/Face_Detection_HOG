import os


class Parameters:
    def __init__(self):
        self.base_dir = '../data'
        self.dir_pos_examples = os.path.join(self.base_dir, 'aug_data')
        self.dir_neg_examples = os.path.join(self.base_dir, 'exempleNegative')
        self.dir_test_examples = os.path.join(self.base_dir,
                                              'exempleTest/CMU+MIT')  # 'exempleTest/CursVA'   'exempleTest/CMU+MIT'
        self.path_annotations = os.path.join(self.base_dir, 'exempleTest/CMU+MIT_adnotari/ground_truth_bboxes.txt')
        self.dir_save_files = os.path.join(self.base_dir, 'salveazaFisiere')
        if not os.path.exists(self.dir_save_files):
            os.makedirs(self.dir_save_files)
            print('directory created: {} '.format(self.dir_save_files))
        else:
            print('directory {} exists '.format(self.dir_save_files))

        # set the parameters
        self.dim_window = 36  # exemplele pozitive (fete de oameni cropate) au 36x36 pixeli
        self.dim_hog_cell = 4  # dimensiunea celulei
        self.dim_descriptor_cell = 36  # dimensiunea descriptorului unei celule
        self.overlap = 0.3
        self.number_positive_examples = 20138 # numarul exemplelor pozitive
        self.number_negative_examples = 29866  # numarul exemplelor negative
        self.threshold = 0.8  # toate ferestrele cu scorul > threshold si maxime locale devin detectii
        self.has_annotations = False
        self.scaling_ratio = 0.9
        self.use_hard_mining = False  # (optional)antrenare cu exemple puternic negative
        self.use_flip_images = False  # adauga imaginile cu fete oglindite
        self.hard_negative_num = 15000
        self.small_neg_images = False
        self.name_MAP = "precizie_medie_" + str(self.number_positive_examples) + "_" + str(
            self.number_negative_examples) + "_" + str(self.threshold) + "_" + str(self.dim_hog_cell)

        self.upscale_ratio = 1
