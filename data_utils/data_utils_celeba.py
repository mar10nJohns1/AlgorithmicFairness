import numpy as np
import pandas as pd
import os

from PIL import Image, ImageOps
from skimage.io import imread
from skimage.transform import resize
from sklearn.preprocessing import LabelEncoder
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    #from sklearn.cross_validation import StratifiedShuffleSplit
    # cross_validation -> now called: model_selection
    # https://stackoverflow.com/questions/30667525/importerror-no-module-named-sklearn-cross-validation
    from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold


def onehot(t, num_classes):
    out = np.zeros((t.shape[0], num_classes))
    for row, col in enumerate(t):
        out[int(row), int(col)] = 1
    return out


class load_data():
    # data_train, data_test and le are public
    def __init__(self, train_path, valid_path, test_path, image_paths,  target_col, image_shape=(128, 128)):
        train_df = pd.read_csv(train_path)
        valid_df = pd.read_csv(valid_path)
        test_df = pd.read_csv(test_path)
        image_paths = image_paths
        image_shape = image_shape
        self._load(train_df, valid_df, test_df, image_paths, target_col, image_shape)
        
    def _load(self, train_df, valid_df, test_df, image_paths, target_col, image_shape):
        # load train.csv
        path_dict = self._path_to_dict(image_paths) # numerate image paths and make it a dict
        # merge image paths with data frame
        train_image_df = self._merge_image_df(train_df, path_dict)
        valid_image_df = self._merge_image_df(valid_df, path_dict)
        test_image_df = self._merge_image_df(test_df, path_dict)
        # label encoder-decoder (self. because we need it later)
        self.le_train = LabelEncoder().fit(train_image_df[target_col])
        # labels for train
        t_train = self.le_train.transform(train_image_df[target_col])
        # label encoder-decoder (self. because we need it later)
        self.le_valid = LabelEncoder().fit(valid_image_df[target_col])
        # labels for valid
        t_valid = self.le_valid.transform(valid_image_df[target_col])
        # label encoder-decoder (self. because we need it later)
        self.le_test = LabelEncoder().fit(test_image_df[target_col])
        # labels for test
        t_test = self.le_test.transform(test_image_df[target_col])
        # getting data
        print("Loading training data")
        train_data = self._make_dataset(train_image_df, image_shape, target_col, t_train)
        print("Loading validation data")
        valid_data = self._make_dataset(valid_image_df, image_shape, target_col, t_valid) 
        print("Loading test data")
        test_data = self._make_dataset(test_image_df, image_shape, target_col, t_test)    
        # need to reformat the train for validation split reasons in the batch_generator
        self.train = self._format_dataset(train_data, with_target=True)
        self.valid = self._format_dataset(valid_data, with_target=True)
        self.test = self._format_dataset(test_data, with_target=True)
        

    def _path_to_dict(self, image_paths):
        path_dict = dict()
        for image_path in image_paths:
            num_path = int(os.path.basename(image_path[:-4]))
            path_dict[num_path] = image_path
        return path_dict

    def _merge_image_df(self, df, path_dict):
        split_path_dict = dict()
        for index, row in df.iterrows():
            split_path_dict[row['im_id']] =  path_dict[int(row['im_id'][:-4])]
        image_frame = pd.DataFrame(list(split_path_dict.values()), columns=['image'])
        df_image =  pd.concat([image_frame, df], axis=1)
        return df_image
        
    def _make_dataset(self, df, image_shape, target_col, target=None):
        # make dataset
        data = dict()
        # merge image with 3x64 features
        for i, dat in enumerate(df.iterrows()):
            index, row = dat
            sample = dict()
            features = row.drop(['im_id', 'image', target_col], axis=0).values
            sample['attributes'] = features
            if target is not None:
                sample['t'] = np.asarray(target[i], dtype='int32')
            image = imread(row['image'])
            sample['image'] = image   
            data[row['im_id']] = sample
            if i % 10000 == 0:
                print("\t%d of %d" % (i, len(df)))
        return data

    def _format_dataset(self, df, with_target):
        # making arrays with all data in, is nessesary when doing validation split
        data = dict()
        value = list(df.values())[0]
        img_tot_shp = tuple([len(df)] + list(value['image'].shape))
        data['images'] = np.zeros(img_tot_shp, dtype='float32')
        feature_tot_shp = (len(df), 40)
        data['attributes'] = np.zeros(feature_tot_shp, dtype='float32')
        if with_target:
            data['ts'] = np.zeros((len(df),), dtype='int32')
        else:
            data['ids'] = np.zeros((len(df),), dtype='int32')
        for i, pair in enumerate(df.items()):
            key, value = pair
            data['images'][i] = value['image']
            data['attributes'][i] = value['attributes']
            if with_target:
                data['ts'][i] = value['t']
            else:
                data['ids'][i] = int(key[:-4])
        return data

    
class batch_generator():
    def __init__(self, data, batch_size=64, num_classes=2,
                 num_iterations=5e3, num_features=40, seed=42):
        self._train = data.train
        self._idcs_train = list(range(0,1000))
        self._valid = data.valid
        self._test = data.test
        # get image size
        value = self._train['images'][0]
        self._image_shape = list(value.shape)
        self._batch_size = batch_size
        self._num_classes = num_classes
        self._num_iterations = num_iterations
        self._num_features = num_features
        self._seed = seed

        
    def _shuffle_train(self):
        np.random.shuffle(self._idcs_train)

    def _batch_init(self, purpose):
        assert purpose in ['train', 'valid', 'test']
        batch_holder = dict()
        batch_holder['attributes'] = np.zeros((self._batch_size, self._num_features), dtype='float32')
        batch_holder['images'] = np.zeros(tuple([self._batch_size] + self._image_shape), dtype='float32')
        batch_holder['ts'] = np.zeros((self._batch_size, self._num_classes), dtype='float32')          
        return batch_holder

    def gen_valid(self):
        batch = self._batch_init(purpose='valid')
        i = 0
        for idx in range(len(self._valid['ts'])):
            batch['attributes'][i] = self._valid['attributes'][idx]
            batch['images'][i] = self._valid['images'][idx]
            batch['ts'][i] = onehot(np.asarray([self._valid['ts'][idx]], dtype='float32'), self._num_classes)
            i += 1
            if i >= self._batch_size:
                yield batch, i
                batch = self._batch_init(purpose='valid')
                i = 0
        if i != 0:
            batch['ts'] = batch['ts'][:i]
            batch['attributes'] = batch['attributes'][:i]
            batch['images'] = batch['images'][:i]
            yield batch, i

    def gen_test(self):
        batch = self._batch_init(purpose='test')
        i = 0
        for idx in range(len(self._test['ts'])):
            batch['attributes'][i] = self._test['attributes'][idx]
            batch['images'][i] = self._test['images'][idx]
            batch['ts'][i] = onehot(np.asarray([self._test['ts'][idx]], dtype='float32'), self._num_classes)
            i += 1
            if i >= self._batch_size:
                yield batch, i
                batch = self._batch_init(purpose='test')
                i = 0
        if i != 0:
            batch['ts'] = batch['ts'][:i]
            batch['attributes'] = batch['attributes'][:i]
            batch['images'] = batch['images'][:i]
            yield batch, i       

    def gen_train(self):
        batch = self._batch_init(purpose='train')
        iteration = 0
        i = 0
        while True:
            # shuffling all batches
            self._shuffle_train()
            for idx in self._idcs_train:
                # extract data from dict
                batch['attributes'][i] = self._train['attributes'][idx]
                batch['images'][i] = self._train['images'][idx]
                batch['ts'][i] = onehot(np.asarray([self._train['ts'][idx]], dtype='float32'), self._num_classes)
                i += 1
                if i >= self._batch_size:
                    yield batch
                    batch = self._batch_init(purpose='train')
                    i = 0
                    iteration += 1
                    if iteration >= self._num_iterations:
                        break