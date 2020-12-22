from datetime import time

from skimage.transform import pyramid_gaussian

from cod.Parameters import *
import numpy as np
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
import glob
import cv2 as cv
import pdb
import pickle
import ntpath
from copy import deepcopy
import time
import timeit
from skimage.feature import hog


class FacialDetector:
    def __init__(self, params: Parameters):
        self.params = params
        self.best_model = None

    def get_positive_descriptors(self):
        # in aceasta functie calculam descriptorii pozitivi
        # vom returna un numpy array de dimensiuni NXD
        # unde N - numar exemplelor pozitive
        # iar D - dimensiunea descriptorului
        # D = (params.dim_window/params.dim_hog_cell - 1) ^ 2 * params.dim_descriptor_cell (fetele sunt patrate)

        images_path = os.path.join(self.params.dir_pos_examples, '*.jpg')
        files = glob.glob(images_path)
        files.extend(glob.glob(os.path.join(self.params.dir_pos_examples, '*.jpeg')))
        num_images = len(files)
        positive_descriptors = []
        print('Calculam descriptorii pt %d imagini pozitive...' % num_images)
        for i in range(num_images):
            print('Procesam exemplul pozitiv numarul %d...' % i)
            img = cv.imread(files[i], cv.IMREAD_GRAYSCALE)

            # daca vream sa adaugam imaginea cu flip
            if self.params.use_flip_images:
                img_flip = cv.flip(img, 1)
                new_name = '..' + files[i].split('.')[2] + '_flip.' + files[i].split('.')[-1]
                cv.imwrite(new_name, img_flip)
                positive_descriptors.append(
                    hog(img_flip, pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell),
                        cells_per_block=(2, 2,))
                )

            descriptor_img = hog(img, pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell),
                                 cells_per_block=(2, 2,))

            positive_descriptors.append(descriptor_img)

            print("am extras descriptorul pentru imaginea ", i, " care are dimensiunea de ", descriptor_img.shape)

        positive_descriptors = np.array(positive_descriptors)
        self.params.number_positive_examples = len(positive_descriptors)
        return positive_descriptors

    def get_negative_descriptors(self):
        # in aceasta functie calculam descriptorii negativi
        # vom returna un numpy array de dimensiuni NXD
        # unde N - numar exemplelor negative
        # iar D - dimensiunea descriptorului
        # avem 274 de imagini negative, vream sa avem self.params.number_negative_examples (setat implicit cu 10000)
        # de exemple negative, din fiecare imagine vom genera aleator self.params.number_negative_examples // 274
        # patch-uri de dimensiune 36x36 pe care le vom considera exemple negative

        images_path = os.path.join(self.params.dir_neg_examples, '*.jpg')
        files = glob.glob(images_path)
        print("")
        num_images = len(files)
        # daca incarcam deja imagini mici nu avem nevoie de num_neg_per_image
        num_negative_per_image = self.params.number_negative_examples // num_images if not self.params.small_neg_images else 0
        negative_descriptors = []
        print('Calculam descriptorii pt %d imagini negative' % num_images)
        for i in range(num_images):
            print('Procesam exemplul negativ numarul %d...' % i)
            img = cv.imread(files[i], cv.IMREAD_GRAYSCALE)
            H, W = img.shape[:2]

            # daca nu foloesc imagini de 36x36 si extrag dintr-o imagine
            # mare ferestre
            if not self.params.small_neg_images:
                # coltul din stanga sus al unei ferestre il notez cu (xmin,ymin)
                # coltul din stanga jos al unei ferestre il notez cu (xman,ymax) , xmax = xmin + 35,ymax = ymin + 35

                xmin = np.random.randint(0, W - self.params.dim_window, num_negative_per_image)
                xmax = xmin + self.params.dim_window
                ymin = np.random.randint(0, H - self.params.dim_window, num_negative_per_image)
                ymax = ymin + self.params.dim_window

                # fiecare fereastra va fo de forma [ymin[i]:ymax[i],xmin[i]:xmax[i]]
                for idx in range(len(xmin)):
                    window = img[ymin[idx]:ymax[idx], xmin[idx]:xmax[idx]]
                    descriptor_window = hog(window,
                                            pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell),
                                            cells_per_block=(2, 2,))

                    # daca am imagini deja de
                    negative_descriptors.append(descriptor_window)

            else:
                # daca avem imagini mici calculam hog pe ea si o adaugam la lista
                descriptor_window = hog(img, pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell),
                                        cells_per_block=(2, 2,))
                negative_descriptors.append(descriptor_window)

        negative_descriptors = np.array(negative_descriptors)
        self.params.number_negative_examples = len(negative_descriptors)
        print("Dim descriptori negativi ", negative_descriptors.shape)
        return negative_descriptors

    def train_classifier(self, training_examples, train_labels):
        svm_file_name = os.path.join(self.params.dir_save_files, 'best_model_%d_%d_%d' %
                                     (self.params.dim_hog_cell, self.params.number_negative_examples,
                                      self.params.number_positive_examples))
        if os.path.exists(svm_file_name):
            self.best_model = pickle.load(open(svm_file_name, 'rb'))
            print(self.best_model.score(training_examples, train_labels))
            return

        best_accuracy = 0
        best_c = 0
        best_model = None
        # Cs = [10 ** -5, 10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1, 10 ** 0]
        Cs = [10 ** -5, 10 ** -4, 10 ** -3, 10 ** -2]
        # Cs = [10 ** -5, 10 ** -4,]

        for c in Cs:
            print('Antrenam un clasificator pentru c=%f' % c)
            model = LinearSVC(C=c)
            print()
            model.fit(training_examples, train_labels)
            acc = model.score(training_examples, train_labels)
            if acc > best_accuracy:
                best_accuracy = acc
                best_c = c
                best_model = deepcopy(model)

        print('Performanta clasificatorului optim pt c = %f' % best_c)
        print('Acuratetea ', best_accuracy)
        # salveaza clasificatorul
        pickle.dump(best_model, open(svm_file_name, 'wb'))

        # vizualizeaza cat de bine sunt separate exemplele pozitive de cele negative dupa antrenare
        # ideal ar fi ca exemplele pozitive sa primeasca scoruri > 0, iar exemplele negative sa primeasca scoruri < 0
        scores = best_model.decision_function(training_examples)
        self.best_model = best_model
        positive_scores = scores[train_labels > 0]
        negative_scores = scores[train_labels <= 0]

        plt.plot(np.sort(positive_scores))
        plt.plot(np.zeros(len(negative_scores) + 20))
        plt.plot(np.sort(negative_scores))
        plt.xlabel('Nr example antrenare')
        plt.ylabel('Scor clasificator')
        plt.title('Distributia scorurilor clasificatorului pe exemplele de antrenare')
        plt.legend(['Scoruri exemple pozitive', '0', 'Scoruri exemple negative'])
        plt.show()

    def intersection_over_union(self, bbox_a, bbox_b):
        x_a = max(bbox_a[0], bbox_b[0])
        y_a = max(bbox_a[1], bbox_b[1])
        x_b = min(bbox_a[2], bbox_b[2])
        y_b = min(bbox_a[3], bbox_b[3])

        inter_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)

        box_a_area = (bbox_a[2] - bbox_a[0] + 1) * (bbox_a[3] - bbox_a[1] + 1)
        box_b_area = (bbox_b[2] - bbox_b[0] + 1) * (bbox_b[3] - bbox_b[1] + 1)

        iou = inter_area / float(box_a_area + box_b_area - inter_area)

        return iou

    def non_maximum_suppression(self, image_detections, image_scores, file_names, image_size):
        """
        Detectiile cu scor mare suprima detectiile ce se suprapun cu acestea dar au scor mai mic.
        Detectiile se pot suprapune partial, dar centrul unei detectii nu poate
        fi in interiorul celeilalte detectii.
        :param image_detections:  numpy array de dimensiune NX4, unde N este numarul de detectii.
        :param image_scores: numpy array de dimensiune N
        :param image_size: tuplu, dimensiunea imaginii
        :return: image_detections si image_scores care sunt maximale.
        """

        # xmin, ymin, xmax, ymax
        x_out_of_bounds = np.where(image_detections[:, 2] > image_size[1])[0]
        y_out_of_bounds = np.where(image_detections[:, 3] > image_size[0])[0]

        image_detections[x_out_of_bounds, 2] = image_size[1]
        image_detections[y_out_of_bounds, 3] = image_size[0]
        sorted_indices = np.flipud(np.argsort(image_scores))
        sorted_image_detections = image_detections[sorted_indices]
        sorted_scores = image_scores[sorted_indices]

        is_maximal = np.ones(len(image_detections)).astype(bool)
        iou_threshold = 0.3
        for i in range(len(sorted_image_detections) - 1):
            if is_maximal[i] == True:  # don't change to 'is True' because is a numpy True and is not a python True :)
                for j in range(i + 1, len(sorted_image_detections)):
                    if is_maximal[
                        j] == True:  # don't change to 'is True' because is a numpy True and is not a python True :)
                        if self.intersection_over_union(sorted_image_detections[i],
                                                        sorted_image_detections[j]) > iou_threshold:
                            is_maximal[j] = False
                        else:  # verificam daca centrul detectiei este in mijlocul detectiei cu scor mai mare
                            c_x = (sorted_image_detections[j][0] + sorted_image_detections[j][2]) / 2
                            c_y = (sorted_image_detections[j][1] + sorted_image_detections[j][3]) / 2
                            if sorted_image_detections[i][0] <= c_x <= sorted_image_detections[i][2] and \
                                    sorted_image_detections[i][1] <= c_y <= sorted_image_detections[i][3]:
                                is_maximal[j] = False

        print("Am mai ramas cu", sorted_image_detections[is_maximal].shape[0], "detectii")
        return sorted_image_detections[is_maximal], sorted_scores[is_maximal], file_names[is_maximal]

    # genereaza toate imaginile de dimeniune image ---> minim cat o fereastra
    # la fiecare pas redusa cu scala scale
    def get_image_pyramid(self, image, scale, min_size=(36, 36)):

        images = []
        image = cv.resize(image, (
        int(image.shape[1] * self.params.upscale_ratio), int(image.shape[0] * self.params.upscale_ratio)))
        while image.shape[0] >= min_size[0] and image.shape[1] >= min_size[1]:
            images.append(image)
            image = cv.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale)))

        return images

    def get_hogs(self, img, scale):

        cell_size = self.params.dim_hog_cell
        cells_per_block = 2
        win_size = self.params.dim_window
        no_cell = win_size // cell_size

        H, W = img.shape[:2]

        cells_per_row = W // cell_size
        cells_per_column = H // cell_size

        block_size = cell_size * cells_per_block
        blocks_per_row = cells_per_row - cells_per_block + 1
        blocks_per_col = cells_per_column - cells_per_block + 1

        # fac hog pe toate imaginea --> o matrice in care fiecare element este un block ce contine 2x2 cells
        # blocurle avanad o suprapunere de 1 cell
        image_hog = hog(img, pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell),
                        cells_per_block=(cells_per_block, cells_per_block,), feature_vector=False)

        # imi redimensionez vectorul de blocks ca matrice pentru a manipula mai usor ferestrele
        image_hog = image_hog.reshape(
            (blocks_per_col, blocks_per_row, image_hog.shape[2] * image_hog.shape[3] * image_hog.shape[4]))

        # print(image_hog.shape)

        # acum doar simulam ultima parte din obtinea hog
        # in care luam blocurile aferente unei ferestre
        # si le facem flat pentru a obtine features pentru acea fereastra
        # i for y and j for x

        # cum blocurile din algoritmul hog se deplaseaza cu pass cell
        # atunci pentru o ferestra de 36x36 vom avea 5x5 blocuri de dim 2x2
        step_size = no_cell - 1
        for i in range(0, blocks_per_col - step_size + 1):
            for j in range(0, blocks_per_row - step_size + 1):
                # selecteaza blocurile ferestrei curente
                curr_hog_window = image_hog[i:i + step_size, j:j + step_size, :].flatten()

                # transforma coorodnata unui block intr-o coordonata imagine
                # care trebuie scalata la dimensiunea imaginii date ca parametru
                # pentru festrele din piramida de imagini
                x_min = (j * cell_size) / scale
                y_min = (i * cell_size) / scale
                # pentru a ajunge la coltul din dreapta trebuie
                # mai intai sa ne mutam no_cell blocuri la dreapta si in jos
                x_max = ((j + no_cell) * cell_size) / scale
                y_max = ((i + no_cell) * cell_size) / scale

                yield curr_hog_window, int(x_min), int(y_min), int(x_max), int(y_max)

    def run(self, return_descriptors=False):
        """
        Aceasta functie returneaza toate detectiile ( = ferestre) pentru toate imaginile din self.params.dir_test_examples
        Directorul cu numele self.params.dir_test_examples contine imagini ce
        pot sau nu contine fete. Aceasta functie ar trebui sa detecteze fete atat pe setul de
        date MIT+CMU dar si pentru alte imagini (imaginile realizate cu voi la curs+laborator).
        Functia 'non_maximum_suppression' suprimeaza detectii care se suprapun (protocolul de evaluare considera o detectie duplicata ca fiind falsa)
        Suprimarea non-maximelor se realizeaza pe pentru fiecare imagine.
        :return:
        detections: numpy array de dimensiune NX4, unde N este numarul de detectii pentru toate imaginile.
        detections[i, :] = [x_min, y_min, x_max, y_max]
        scores: numpy array de dimensiune N, scorurile pentru toate detectiile pentru toate imaginile.
        file_names: numpy array de dimensiune N, pentru fiecare detectie trebuie sa salvam numele imaginii.
        (doar numele, nu toata calea).
        """

        if return_descriptors:
            test_images_path = os.path.join(os.path.join(self.params.base_dir, 'exempleNegative'), '*.jpg')
        else:
            test_images_path = os.path.join(self.params.dir_test_examples, '*.jpg')

        test_files = glob.glob(test_images_path)
        detections = np.array([])  # array cu toate detectiile pe care le obtinem
        scores = np.array([])  # array cu toate scorurile pe care le optinem
        file_names = np.array(
            [])  # array cu fisiele, in aceasta lista fisierele vor aparea de mai multe ori, pentru fiecare
        # detectie din imagine, numele imaginii va aparea in aceasta lista
        w = self.best_model.coef_.T
        w = w.reshape((w.shape[0]))
        bias = self.best_model.intercept_[0]
        num_test_images = len(test_files)
        descriptors_to_return = []

        for i in range(num_test_images):
            curr_detections = []
            curr_scores = np.array([])
            curr_file_names = np.array([])

            start_time = timeit.default_timer()
            print('Procesam imaginea de testare %d/%d..' % (i, num_test_images))
            img = cv.imread(test_files[i], cv.IMREAD_GRAYSCALE)

            # tine scale-ul curent
            scale = self.params.upscale_ratio
            # pentru fiecare imagine din piramida de imagini
            for index, p_image in enumerate(
                    self.get_image_pyramid(img, scale=self.params.scaling_ratio, min_size=(36, 36))):

                curr_scale = scale
                # pentru hard negatives cautam fereste in imaginile
                # negative dar doar la o singura scala
                if return_descriptors and index > 0:
                    continue

                # pentru fiecare window hog
                for hog_window, x_min, y_min, x_max, y_max in self.get_hogs(p_image, curr_scale):

                    descriptor = hog_window
                    # calculeaza scorul
                    cls = np.dot(descriptor, w) + bias

                    if cls > self.params.threshold:

                        # daca e hard negative cu scorul pozitiv
                        # atunci salveaza window-ul
                        if return_descriptors:
                            descriptors_to_return.append((cls, descriptor))
                            print("Descriptorul " + str(len(descriptors_to_return)) + " cu scorul " + str(cls))

                        # salveaza scorurile,detectiile si fisiere aferente
                        curr_scores = np.append(curr_scores, cls)
                        curr_file_names = np.append(curr_file_names, test_files[i].split('/')[-1])
                        curr_detections.append([x_min, y_min, x_max, y_max])

                # reduc scale-ul
                scale = scale * self.params.scaling_ratio

            # daca am detectii pentru imaginea curenta
            if len(curr_detections) > 0:
                # aplica non_maximum_supression
                curr_detections, curr_scores, curr_file_names = self.non_maximum_suppression(
                    np.array(curr_detections),
                    curr_scores,
                    curr_file_names,
                    np.array([img.shape[0],
                              img.shape[1]]))

                curr_detections = list(curr_detections)

                # adauga detectiile ramase la lista finala
                if len(detections) == 0:
                    detections = np.copy(curr_detections)
                else:
                    detections = np.append(detections, curr_detections, axis=0)
                # salveza scorurile si numele fisierelor din care fac parte detectiile
                scores = np.append(scores, curr_scores, axis=0)
                file_names = np.append(file_names, curr_file_names, axis=0)

            end_time = timeit.default_timer()
            print('Timpul de procesarea al imaginii de testare %d/%d este %f sec.'
                  % (i, num_test_images, end_time - start_time))

        # detections = np.array(detections)
        # daca facem hard negative mining returnam descriptorii
        # altfe returneam detectiile si scorurile
        if return_descriptors:
            return descriptors_to_return
        return detections, scores, file_names

    def compute_average_precision(self, rec, prec):
        # functie adaptata din 2010 Pascal VOC development kit
        m_rec = np.concatenate(([0], rec, [1]))
        m_pre = np.concatenate(([0], prec, [0]))
        for i in range(len(m_pre) - 1, -1, 1):
            m_pre[i] = max(m_pre[i], m_pre[i + 1])
        m_rec = np.array(m_rec)
        i = np.where(m_rec[1:] != m_rec[:-1])[0] + 1
        average_precision = np.sum((m_rec[i] - m_rec[i - 1]) * m_pre[i])
        return average_precision

    def eval_detections(self, detections, scores, file_names):
        ground_truth_file = np.loadtxt(self.params.path_annotations, dtype='str')
        ground_truth_file_names = np.array(ground_truth_file[:, 0])
        ground_truth_detections = np.array(ground_truth_file[:, 1:], np.int)

        num_gt_detections = len(ground_truth_detections)  # numar total de adevarat pozitive
        gt_exists_detection = np.zeros(num_gt_detections)
        # sorteazam detectiile dupa scorul lor
        sorted_indices = np.argsort(scores)[::-1]
        file_names = file_names[sorted_indices]
        scores = scores[sorted_indices]
        detections = detections[sorted_indices]

        num_detections = len(detections)
        true_positive = np.zeros(num_detections)
        false_positive = np.zeros(num_detections)
        duplicated_detections = np.zeros(num_detections)

        for detection_idx in range(num_detections):
            indices_detections_on_image = np.where(ground_truth_file_names == file_names[detection_idx])[0]

            gt_detections_on_image = ground_truth_detections[indices_detections_on_image]
            bbox = detections[detection_idx]
            max_overlap = -1
            index_max_overlap_bbox = -1
            for gt_idx, gt_bbox in enumerate(gt_detections_on_image):
                overlap = self.intersection_over_union(bbox, gt_bbox)
                if overlap > max_overlap:
                    max_overlap = overlap
                    index_max_overlap_bbox = indices_detections_on_image[gt_idx]

            # clasifica o detectie ca fiind adevarat pozitiva / fals pozitiva
            if max_overlap >= 0.3:
                if gt_exists_detection[index_max_overlap_bbox] == 0:
                    true_positive[detection_idx] = 1
                    gt_exists_detection[index_max_overlap_bbox] = 1
                else:
                    false_positive[detection_idx] = 1
                    duplicated_detections[detection_idx] = 1
            else:
                false_positive[detection_idx] = 1

        cum_false_positive = np.cumsum(false_positive)
        cum_true_positive = np.cumsum(true_positive)

        rec = cum_true_positive / num_gt_detections
        prec = cum_true_positive / (cum_true_positive + cum_false_positive)
        average_precision = self.compute_average_precision(rec, prec)
        plt.plot(rec, prec, '-')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Average precision %.3f' % average_precision)
        plt.savefig(os.path.join(self.params.dir_save_files, self.params.name_MAP + '.png'))
        plt.show()
