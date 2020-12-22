from cod.Parameters import *
from cod.FacialDetector import *
import pdb
from cod.Visualize import *
from skimage.transform import pyramid_gaussian

params: Parameters = Parameters()
params.dim_window = 36  # exemplele pozitive (fete de oameni cropate) au 36x36 pixeli
params.dim_hog_cell = 4  # dimensiunea celulei
params.overlap = 0.3

params.number_positive_examples = 20138  # numarul exemplelor pozitive
params.number_negative_examples = 29866  # numarul exemplelor negative

params.threshold = 0.8  # toate ferestrele cu ooscorul > othreshold si maxime locale devin detectii
params.has_annotations = True

params.scaling_ratio = 0.9

params.use_hard_mining = False  # (optional)antrenare cu exemple puternic negative
params.use_flip_images = False  # adauga imaginile cu fete oglindite

facial_detector: FacialDetector = FacialDetector(params)

# Pasul 1. Incarcam exemplele pozitive (cropate) si exemple negative generate exemple pozitive
# verificam daca ii avem deja salvati
positive_features_path = os.path.join(params.dir_save_files,
                                      'descriptoriExemplePozitive_' + str(params.dim_hog_cell) + '_' +
                                      str(params.number_positive_examples) + '.npy')
if os.path.exists(positive_features_path):
    positive_features = np.load(positive_features_path)
    print('Am incarcat descriptorii pentru exemplele pozitive')
else:
    print('Construim descriptorii pentru exemplele pozitive:')
    positive_features = facial_detector.get_positive_descriptors()
    np.save(positive_features_path, positive_features)
    print('Am salvat descriptorii pentru exemplele pozitive in fisierul %s' % positive_features_path)

# exemple negative
negative_features_path = os.path.join(params.dir_save_files,
                                      'descriptoriExempleNegative_' + str(params.dim_hog_cell) + '_' +
                                      str(params.number_negative_examples) + '.npy')
if os.path.exists(negative_features_path):
    negative_features = np.load(negative_features_path)
    print('Am incarcat descriptorii pentru exemplele negative')
else:
    print('Construim descriptorii pentru exemplele negative:')
    negative_features = facial_detector.get_negative_descriptors()
    np.save(negative_features_path, negative_features)
    print('Am salvat descriptorii pentru exemplele negative in fisierul %s' % negative_features_path)

# Pasul 2. Invatam clasificatorul liniar
print("Positive features", positive_features.shape)
print("Negative features", negative_features.shape)
training_examples = np.concatenate((np.squeeze(positive_features), np.squeeze(negative_features)), axis=0)
train_labels = np.concatenate((np.ones(params.number_positive_examples), np.zeros(negative_features.shape[0])))
print("Training examples size", training_examples.shape)
facial_detector.train_classifier(training_examples, train_labels)

# Pasul 3. (optional) Antrenare cu exemple puternic negative (detectii cu scor >0 din cele 274 de imagini negative)
# Daca implementati acest pas ar trebui sa modificati functia FacialDetector.run()
# astfel incat sa va returneze descriptorii detectiilor cu scor > 0 din cele 274 imagini negative
# completati codul in continuare
# TODO:  (optional)  completeaza codul in continuare

if params.use_hard_mining:
    hard_negatives_path = os.path.join(params.dir_save_files,
                                       'descriptoriHardNegatives_' + str(params.dim_hog_cell) + '_' + str(
                                           params.number_positive_examples)
                                       + "_" + str(params.number_negative_examples) + "_" +
                                       str(params.hard_negative_num) + '.npy')

    if os.path.exists(hard_negatives_path):
        hard_negatives = np.load(hard_negatives_path)[:15000]
        print('Am incarcat descriptorii pentru exemplele hard negative')
        print(len(hard_negatives))

    else:
        print('Construim descriptorii pentru hard negatives:')
        hard_negatives = facial_detector.run(return_descriptors=True)
        # luam doar primele hard_negaive_nums exemple in ordine descrescatoare a scorului
        hard_negatives = np.array([x[1] for x in sorted(hard_negatives, key=lambda x: x[0], reverse=True)[
                                                 :facial_detector.params.hard_negative_num]])
        np.save(hard_negatives_path, hard_negatives)
        print("Avem ", len(hard_negatives), " hard negatives")

    # adaugam hard negative la multimea de atrenarea
    training_examples = np.concatenate((training_examples, np.squeeze(hard_negatives)), axis=0)
    # adaugam cu label-ul 0
    train_labels = np.concatenate((train_labels, np.zeros(hard_negatives.shape[0])))
    print(training_examples.shape)
    print(train_labels.shape)
    # crestem marimea setului de exemple negative
    facial_detector.params.number_negative_examples += hard_negatives.shape[0]
    facial_detector.train_classifier(training_examples, train_labels)

# Pasul 4. Ruleaza detectorul facial pe imaginile de test.
detections, scores, file_names = facial_detector.run()

# # Pasul 5. Evalueaza si vizualizeaza detectiile
# # Pentru imagini pentru care exista adnotari (cele din setul de date  CMU+MIT) folositi functia show_detection_with_ground_truth
# # pentru imagini fara adnotari (cele realizate la curs si laborator) folositi functia show_detection_without_ground_truth
if params.has_annotations:
    facial_detector.eval_detections(detections, scores, file_names)
    show_detections_with_ground_truth(detections, scores, file_names, params)
else:
    show_detections_without_ground_truth(detections, scores, file_names, params)
