import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import glob
import re
import warnings
import time
import datetime
import logging
import subprocess
import multiprocessing

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import sklearn.pipeline
import sklearn.model_selection
import sklearn.preprocessing

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split

import central_arena_view as cav
import tqdm

model_directory = os.path.dirname(os.path.realpath(__file__))

DEFAULT_MODEL_SUFFIX = 'default'

log = logging.getLogger('model_base_pytorch')

# --------------------------------------------------------------------------------
# 1) ACTIONS & TRANSFORMS
# --------------------------------------------------------------------------------

class Transform():
    def __init__(self, logger, name='nop-transform'):
        self.logger = logger
        self.name = name

    def get_name(self):
        return self.name

    def in_game_transform(self, game_state, action):
        pass

    def batch_transform(self, id, use_cached=True, augment_with_penalty_moves=False):
        pass

    def batch_transform_X_y(self, in_df):
        pass


action_options = [
    ('WAIT',(1,0,0,0,0,0)),
    ('UP',(0,1,0,0,0,0)),
    ('LEFT',(0,0,1,0,0,0)),
    ('DOWN',(0,0,0,1,0,0)),
    ('RIGHT',(0,0,0,0,1,0)),
    ('BOMB',(0,0,0,0,0,1))
]

action_options_df = pd.DataFrame(
    [
        (1, 0, 0, 0, 0, 0),
        (0, 1, 0, 0, 0, 0),
        (0, 0, 1, 0, 0, 0),
        (0, 0, 0, 1, 0, 0),
        (0, 0, 0, 0, 1, 0),
        (0, 0, 0, 0, 0, 1)
    ],
    columns=cav.DFIDs.A_ONE_HOT
)

class BaseTransform(Transform):
    def __init__(self, logger, name='base-transform', size=5):
        super().__init__(logger, name=name)
        logger.debug('setup')
        self.size = size
        self.action_option_dict = dict(action_options)

        self.input_columns = (
            cav.DFIDs.A_ONE_HOT
            + self.transform_nearest_object_info_columns(cav.PACAV.nearest_other_agent_info_columns)
            + self.transform_nearest_object_info_columns(cav.PACAV.nearest_coin_info_columns)
            + self.transform_nearest_object_info_columns(cav.PACAV.nearest_crate_info_columns)
            + self.transform_nearest_object_info_columns(cav.PACAV.mid_of_map_info_columns)
            + cav.TC.get_transformation(size)
        )

        self.df0 = pd.DataFrame(columns=self.input_columns)

    def transform_nearest_object_info_columns(self, columns):
        """
        Example: columns might be ['agent','dist','direction'] 
        We'll rename to ['agentx','agenty','dist','direction']
        """
        e0   = columns[0]
        rest = columns[1:]
        r = [e0 + 'x', e0 + 'y'] + rest
        return r

    def get_name(self):
        return '{}-size{}'.format(self.name, self.size)

    def in_game_transform(self, game_state, action=None, validate=False):
        self.logger.debug('in_game_transform: start')

        # Create the augmented view as DataFrame
        av = cav.PandasAugmentedCentralArenaView(game_state)
        ldf = av.to_df()
        self.logger.debug('in_game_transform: PandasAugmentedCentralArenaView done')

        # Extract features
        t = cav.FeatureSelectionTransformation0(ldf, size=self.size)
        self.logger.debug('in_game_transform: FeatureSelectionTransformation0 created')
        out_npa = t.in_game_transform(av)
        self.logger.debug('in_game_transform: out_npa created')

        out_npa_ = out_npa.reshape(1,-1)
        out_df = pd.DataFrame(out_npa_, columns=self.input_columns)

        if validate:
            # Double-check with the transform on the entire DF
            out_df_ = t.transform()[self.input_columns].copy()
            lds = out_df.iloc[0,:] == out_df_.iloc[0,:]
            if not lds.all():
                raise Exception('Validation between in_game_transform and transform failed: {}'.format(lds))
        self.logger.debug('in_game_transform: FeatureSelectionTransformation0 done')

        # If an action is specified, set that single row. Otherwise, replicate for all possible actions.
        if action is not None:
            action_one_hot = self.action_option_dict[action]
            out_df.loc[0, cav.DFIDs.A_ONE_HOT] = action_one_hot
        else:
            out_df = pd.concat([out_df]*len(action_options_df), axis=0).reset_index(drop=True)
            out_df.loc[:, cav.DFIDs.A_ONE_HOT] = action_options_df.loc[:,:]

        self.logger.debug('in_game_transform: A_ONE_HOT done')
        return out_df, av

    def batch_transform(self, id, use_cached=True, augment_with_penalty_moves=False):
        """
        Uses PostProcessGame from central_arena_view to load and transform data from .h5 files.
        """
        file_name_pattern = './hdf5_training_data/{}-{}.h5'
        out_file_name = file_name_pattern.format(id, self.get_name())
        rdf  = None
        rdf_ = None
        if use_cached and os.path.isfile(out_file_name):
            self.logger.info('Loading transformed data from file: {}'.format(out_file_name))
            with pd.HDFStore(out_file_name, mode='r') as s:
                rdf = s['df'].copy()
        else:
            in_file_names = [file_name_pattern.format(id, i) for i in range(4)]
            postprocess = cav.PostProcessGame(in_file_names, out_file_name, size=self.size)
            time1 = time.time()
            postprocess.process()
            time2 = time.time()
            self.logger.debug(postprocess.time_info)
            del postprocess
            self.logger.debug('PostProcessGame took {:.3f} ms'.format((time2 - time1) * 1000.0))

            with pd.HDFStore(out_file_name) as s:
                rdf = s['df'].copy()

        if augment_with_penalty_moves:
            time1 = time.time()
            penalty_moves_transform = cav.AugmentGameDataWithPenaltyMoves(rdf)
            rdf_ = penalty_moves_transform.process()
            time2 = time.time()
            del penalty_moves_transform
            self.logger.debug('AugmentGameDataWithPenaltyMoves took {:.3f} ms'.format((time2 - time1) * 1000.0))

        return rdf, rdf_

    def batch_transform_X_y(self, in_df):
        X = in_df[self.input_columns]
        y = in_df['QQ']
        return X, y

# --------------------------------------------------------------------------------
# 2) MODEL LOADING UTILITY (REPLACES MXNET-SPECIFIC FILE PATTERNS)
# --------------------------------------------------------------------------------

def get_model_load_path(name, suffix=DEFAULT_MODEL_SUFFIX, selected_id=None, latest=True):
    """
    Looks for .pt files of the form:  <timeID>-<agentName>-<suffix>.pt
    e.g.  1677937795-VerySimpleMX5-default.pt
    """
    glob_list = glob.glob(f'{model_directory}/*.pt')

    model_options = []
    for g in glob_list:
        path_separator = '/'
        if os.sep == '\\':
            path_separator = '\\\\'

        pattern = rf'^.*{path_separator}(\d+)-(.*?)-(.*?).pt$'
        r = re.search(pattern, g)
        if r:
            id_         = r.group(1)
            agent_name  = r.group(2)
            suffix_     = r.group(3)
            model_options.append((id_, agent_name, suffix_))
        else:
            # not matching the pattern
            continue

    # Filter for the requested name + suffix
    if selected_id is not None:
        # user wants a specific ID
        matching = [(id_, an, sfx) for (id_, an, sfx) in model_options if an == name and sfx == suffix]
        if len(matching) == 0:
            return None
        # ensure selected_id is present
        if str(selected_id) not in [m[0] for m in matching]:
            raise Exception(
                f'The selected_id={selected_id} not found among existing model IDs: {matching}'
            )
        # return that single
        return f'{model_directory}/{selected_id}-{name}-{suffix}.pt'
    else:
        # user wants the latest or we have no id
        candidates = [ (int(id_), sfx) for (id_, an, sfx) in model_options if an == name and sfx == suffix ]
        if not candidates:
            return None
        if latest:
            best_id = np.max([c[0] for c in candidates])
            return f'{model_directory}/{best_id}-{name}-{suffix}.pt'
        else:
            # If not 'latest' but no selected_id given, do nothing special
            return None

# --------------------------------------------------------------------------------
# 3) A BASIC PYTORCH “REGRESSOR” WRAPPER (REPLACES GluonRegressor)
# --------------------------------------------------------------------------------

class PyTorchRegressor:
    """
    Roughly mimics the interface your GluonRegressor might have had:
      - model_fn: a function that returns an nn.Module
      - batch_size, device, epochs, auto_save, etc.
      - fit(X, y, ...)
      - predict(X)
      - evaluate(X, y)
      - save(path)
      - load(path)
    """
    def __init__(self,
                 model_fn,
                 batch_size=2048,
                 device=None,
                 epochs=2,
                 auto_save=True,
                 lr=1e-3):
        self.model = model_fn()  # This should be an nn.Module
        self.device = device if device else torch.device("cpu")
        self.model.to(self.device)

        self.batch_size = batch_size
        self.epochs = epochs
        self.auto_save = auto_save

        # Basic optimizer + MSE loss for demonstration
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def fit(self, X, y,
            batch_size=None,
            epochs=None,
            verbose=1,
            validation_split=0.1,
            model_save_path=None,
            **kwargs):

        if batch_size is None:
            batch_size = self.batch_size
        if epochs is None:
            epochs = self.epochs

        # Convert X, y to Tensors
        X_t = torch.tensor(X.values, dtype=torch.float32)
        y_t = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)

        dataset = TensorDataset(X_t, y_t)

        # Simple train/val split by ratio
        val_size = int(len(dataset) * validation_split)
        train_size = len(dataset) - val_size
        train_ds, val_ds = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        for ep in range(epochs):
            self.model.train()
            running_loss = 0.0

            for xb, yb in train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)

                self.optimizer.zero_grad()
                preds = self.model(xb)
                loss = self.criterion(preds, yb)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item() * xb.size(0)

            # Compute average train loss
            train_loss = running_loss / len(train_ds)

            # Validation
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(self.device)
                    yb = yb.to(self.device)
                    preds = self.model(xb)
                    loss = self.criterion(preds, yb)
                    val_loss += loss.item() * xb.size(0)
            val_loss = val_loss / len(val_ds) if val_size > 0 else 0.0

            if verbose:
                print(f"Epoch {ep+1}/{epochs} - train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}")

            # Auto-save (optional)
            if self.auto_save and model_save_path is not None:
                self.save(model_save_path)

        return {"train_loss": train_loss, "val_loss": val_loss}

    def predict(self, X):
        self.model.eval()
        X_t = torch.tensor(X.values, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            preds = self.model(X_t).cpu().numpy()
        # Return as a 1D array
        return preds.ravel()

    def evaluate(self, X, y):
        """
        Returns MSE for demonstration. 
        Adjust if you need other metrics (MAE, etc.).
        """
        preds = self.predict(X)
        mse = np.mean((preds - y.values)**2)
        return mse

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()

# --------------------------------------------------------------------------------
# 4) BASEMODEL: REPLACES THE OLD MXNET-BASED CLASS
# --------------------------------------------------------------------------------

class BaseModel(object):
    MODEL_NAME_PATTERN = '{}-{}-{}'  # We’ll add .pt at the end

    def __init__(self, name, fn, logger, transform, auto_save=True):
        self.logger = logger
        if '-' in name:
            raise Exception('Model names must not contain "-" characters: {}'.format(name))

        # Pytorch device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")

        self.transform = transform
        self.auto_save = auto_save

        # Our PyTorchRegressor (replacing GluonRegressor)
        if fn is not None:
            self.model = PyTorchRegressor(
                model_fn=fn,
                batch_size=8 * 256,
                device=self.device,
                epochs=2,
                auto_save=auto_save
            )
        else:
            self.model = None

        self.name = name
        self.path = self.create_model_save_path()
        self.action_options = action_options
        self.trainer = None  # Not used here, but kept for structural similarity

    def get_transform(self):
        return self.transform

    def create_model_save_path(self, suffix=DEFAULT_MODEL_SUFFIX):
        # generate a unique ID from timestamp
        id_ = int(time.mktime(datetime.datetime.now().timetuple()))
        file_name = BaseModel.MODEL_NAME_PATTERN.format(id_, self.name, suffix)
        return f'{model_directory}/{file_name}.pt'

    def fit(self, X_train, y_train, model_save_path=None, epochs=2, batchsize=8*256, **kwargs):
        if model_save_path is None:
            model_save_path = self.create_model_save_path()
        self.path = model_save_path
        self.logger.info('model_save_path: {}'.format(model_save_path))

        return self.model.fit(
            X_train, y_train,
            batch_size=batchsize,
            epochs=epochs,
            verbose=1,
            validation_split=0.1,
            model_save_path=model_save_path,
            **kwargs
        )

    def save(self):
        self.model.save(self.path)

    def load(self, id=None, suffix=None, latest=True):
        """
        Loads .pt file from disk using our get_model_load_path() utility.
        """
        model_load_path_ = get_model_load_path(
            self.name,
            suffix=DEFAULT_MODEL_SUFFIX if suffix is None else suffix,
            selected_id=id,
            latest=latest
        )
        if not model_load_path_ or not os.path.isfile(model_load_path_):
            self.logger.info("No model found to load.")
            return

        self.logger.info("Trying to load model: {}".format(model_load_path_))
        self.model.load(model_load_path_)
        self.logger.info("Model loaded: {}".format(model_load_path_))

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y):
        return self.model.evaluate(X, y)

# --------------------------------------------------------------------------------
# 5) NETWORK DEFINITION (REPLACES create_net_ / very_simple)
# --------------------------------------------------------------------------------

class VerySimpleNet(nn.Module):
    """
    Equivalent to:
      Dense(300, relu), Dense(100, relu), Dense(1, None)
    """
    def __init__(self):
        super(VerySimpleNet, self).__init__()
        self.fc1 = nn.Linear(in_features=300, out_features=100)
        # Because original code does Dense(300->100) after Flatten(??) 
        # YOU may need to adjust input size if your feature dimension differs
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(100, 1)

    def forward(self, x):
        # If your input dimension is not 300, adjust accordingly
        x = self.act(nn.Linear(x.size(1), 300)(x))  # Or do something that matches your input dimension
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return x

def create_net_():
    """
    A functional-style builder returning the PyTorch network. 
    If your real input dimension is different, adapt the # of in_features.
    """
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.flatten = nn.Flatten()
            self.fc0 = nn.Linear( 300, 300 )  # optional pass-through
            self.fc1 = nn.Linear(300, 100)
            self.fc2 = nn.Linear(100, 1)
            self.relu = nn.ReLU()

        def forward(self, x):
            # Flatten if needed
            x = self.flatten(x)
            # Example pipeline:
            x = self.relu(self.fc0(x))
            x = self.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    return Net()

def very_simple():
    """
    Replaces your MXNet 'very_simple()' function that returned 
    a HybridSequential with [Dense(300, relu), Dense(100, relu), Dense(1, None)].
    """
    return create_net_()

class VerySimple5(BaseModel):
    """
    If you've already defined this in your previous code snippet, 
    you may not need to redefine it. Otherwise, here's an example.
    """
    def __init__(self, logger):
        self.transform = BaseTransform(logger, size=5)
        # "very_simple" is your function returning a small PyTorch net 
        super().__init__("VerySimpleMX5", very_simple, logger, self.transform)


# -------------------------------------------------------------------
# 2) NCHWTransform
# -------------------------------------------------------------------
class NCHWTransform(Transform):
    def __init__(self, logger, name='nchw-transform', size=11):
        super().__init__(logger, name=name)
        logger.debug('setup')
        self.size = size
        self.action_option_dict = dict(action_options)

        # Re-use the base transform for data selection
        self.base_transform = BaseTransform(logger, size=size)
        self.input_columns  = self.base_transform.input_columns

    def get_name(self):
        return '{}-size{}'.format(self.name, self.size)

    def in_game_transform(self, game_state, action=None):
        """
        For single in-game transformations, we simply defer to the base transform
        and then do any needed NCHW modifications if necessary.
        """
        self.logger.debug('in_game_transform: start')

        out_df, av = self.base_transform.in_game_transform(game_state, action=action)
        return out_df  # or (out_df, av), if you need both

    def batch_transform(self, id, use_cached=True, augment_with_penalty_moves=False):
        """
        Returns an xarray Dataset (rxds) in NCHW shape from your FeatureSelectionTransformationNCHW.
        """
        rdf, rdf_ = self.base_transform.batch_transform(
            id, 
            use_cached=use_cached, 
            augment_with_penalty_moves=augment_with_penalty_moves
        )

        # Decide which DF to pass to the NCHW transformation
        df_to_transform = rdf_ if augment_with_penalty_moves else rdf

        # This transformation produces an xarray with dims like ['N','C','H','W']
        t = cav.FeatureSelectionTransformationNCHW(df_to_transform)
        rxds = t.transform()
        return rxds

    def batch_transform_iter(self, id, use_cached=True, augment_with_penalty_moves=False, batch_size=600000):
        """
        An iterator version if your data is very large, chunked by `batch_size`.
        """
        rdf, rdf_ = self.base_transform.batch_transform(
            id, 
            use_cached=use_cached, 
            augment_with_penalty_moves=augment_with_penalty_moves
        )

        if augment_with_penalty_moves:
            rdf = rdf_

        n = len(rdf)
        rnd_idx = np.random.permutation(np.arange(0, n))
        rdf = rdf.iloc[rnd_idx,:]

        for i in range(0, n, batch_size):
            e = min(n, i + batch_size)
            ldf = rdf.iloc[i:e,:]
            t = cav.FeatureSelectionTransformationNCHW(ldf)
            rxds = t.transform()
            yield rxds


# -------------------------------------------------------------------
# 3) A PyTorch Implementation of VGG
# -------------------------------------------------------------------
vgg_spec = {
    8:  ([3, 2], [32, 64]),  # layers, filters
    11: ([1, 1, 2, 2, 2], [64, 128, 256, 512, 512]),
    13: ([2, 2, 2, 2, 2], [64, 128, 256, 512, 512]),
}

class PyTorchVGG(nn.Module):
    """
    VGG layout:
    1) A sequence of conv-batchnorm-activation blocks + pooling
    2) Flatten
    3) Dense(512, relu) -> Dropout(0.5)
    4) Dense(512, relu) -> Dropout(0.5)
    5) Dense(1) 
    """
    def __init__(self, layers, filters, batch_norm=True):
        super().__init__()
        assert len(layers) == len(filters)

        # We'll build self.features as a sequential stack of conv/pool
        # then a couple of fully-connected layers as "classifier".
        # For a typical VGG, input dims might be NCHW. 
        # You must ensure the final shape fits 512 fully connected units 
        # if you want to replicate the original exactly. 
        self.features = nn.Sequential()
        in_channels   = None  # set this from data or pass as an arg, default is ? 
                              # Typically you might have in_channels = 8 or so 
                              # if your data has 8 channels

        # Because your original code references "N x 8 x 11 x 11", let's guess in_channels=8
        # You can adjust as needed:
        in_channels = 8  # or however many channels you actually have

        # Build the blocks
        current = in_channels
        for i, num_convs in enumerate(layers):
            for _ in range(num_convs):
                conv = nn.Conv2d(
                    in_channels=current,
                    out_channels=filters[i],
                    kernel_size=3,
                    padding=1,   # 'same' for 3x3
                    stride=1,
                    dilation=1
                )
                self.features.append(conv)
                if batch_norm:
                    self.features.append(nn.BatchNorm2d(filters[i]))
                self.features.append(nn.ReLU(inplace=True))

                current = filters[i]

            # MaxPool after each block
            self.features.append(nn.MaxPool2d(kernel_size=2, stride=2))

        # Flatten is handled in forward() or we can do an nn.Flatten() 
        # We'll do it in forward for clarity.

        # The "classifier" part
        self.fc = nn.Sequential(
            nn.Linear(512 * 1 * 1,  # <== You may need to adjust if your final spatial dims != (1,1)
                      512),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(0.5),
        )

        # The output layer
        self.output = nn.Linear(512, 1)

    def forward(self, x):
        # x shape: [N, 8, 11, 11] (if in_channels=8, H=W=11)
        # Pass through feature extractor
        x = self.features(x)

        # If your final feature map is [N, 512, 1, 1], then we flatten to [N, 512]
        # If your shape is different, adjust the in/out features in self.fc
        x = x.view(x.size(0), -1)

        # Now pass through the classifier 
        x = self.fc(x)

        # Finally
        x = self.output(x)
        return x


def vgg():
    """
    The old code used vgg_spec[8] => (layers=[3,2], filters=[32,64]).
    We create a PyTorchVGG with that config, hybridize, etc.
    """
    layers, filters = vgg_spec[8]
    net = PyTorchVGG(layers, filters, batch_norm=True)
    return net


# -------------------------------------------------------------------
# 4) VGGModel: extends BaseModel, uses the new VGG architecture 
# -------------------------------------------------------------------
class VGGModel(BaseModel):
    """
    Equivalent to your old 'VGGModel' but now in PyTorch.
    """
    def __init__(self, logger, auto_save=False):
        # The transform that yields NCHW data
        self.transform = NCHWTransform(logger, size=11)
        # pass our PyTorch vgg() to BaseModel
        super().__init__("VGG5", vgg, logger, self.transform, auto_save=auto_save)


# -------------------------------------------------------------------
# 5) VGGPlusBlock in PyTorch
#     - Multi-input: x_nf (N x F) and x_nchw (N x C x H x W)
#     - Two parallel streams that get concatenated 
#     - Then a fully-connected (512->512->1) 
# -------------------------------------------------------------------
class PyTorchVGGPlusBlock(nn.Module):
    """
    Replaces your old 'VGGPlusBlock(mx.gluon.nn.HybridBlock)'. 
    We'll do the same basic idea:
      - For x_nchw, we do two feature-extraction sub-blocks => combine => flatten
      - For x_nf, we pass it through directly or combine it with x_nchw features
      - Then we do some fully-connected layers => single output
    """

    def __init__(self):
        super().__init__()
        # In your old code, you had two smaller blocks (f1, f2) each with conv layers.
        # We replicate that with PyTorch:
        self.f1 = self._make_features(num_convs=4, out_channels=32, block_name="f1")
        self.f2 = self._make_features(num_convs=3, out_channels=64, block_name="f2")

        # Flatten after the second block
        # We'll just call `x = x.view(x.size(0), -1)` in forward()

        # Suppose x_nchw eventually becomes shape [N, ???], we then cat with x_nf => fully connected.
        # We'll guess the final flatten dimension is "some number" + the dimension of x_nf.
        # Let’s define that more precisely after two pools:
        #  - f1 => conv(3x3, pad=1) repeated, then pool => shape halved
        #  - f2 => conv repeated, then pool => shape halved again
        # If input is [N, 8, H, W], after f1 => [N, 32, H/2, W/2], after f2 => [N, 64, H/4, W/4].
        # If H=W=11 => after two 2x2 pools => [N,64, 11/4, 11/4], i.e. [N,64, 2, 2].
        # So flatten => 64*2*2=256 features from the conv path. 
        #
        # Meanwhile, x_nf has shape [N, F], so in forward we do cat([x_nf, conv_features], dim=1).
        # Then we do a few linear layers. Let's replicate the "512 => 512 => 1" approach.

        self.post_conv_fc = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(True)
        )

        # The final fully-connected portion after concatenation:
        # If x_nf has dimension F, and the conv path yields 512, then total = F + 512
        # We'll do two dense layers with dropout, etc.
        self.fc = nn.Sequential(
            nn.Linear(512 + 0, 512),  # we might add a +F if we want x_nf to skip the post_conv_fc
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(0.5),
        )

        self.output = nn.Linear(512, 1)

    def _make_features(self, num_convs, out_channels, block_name=""):
        """
        Creates a small block of conv+relu repeated `num_convs` times, 
        then a maxpool (stride=2).
        """
        block = nn.Sequential()
        in_c  = None  # figure out if we chain them or do them in separate steps
        # Because the first block might see an input of shape [N, 8, H, W], 
        # we'll define it here. For the second block, we'll see [N, out_channels_of_f1, ...].
        # But since we’re making them as separate modules, we either do that all in forward 
        # or define them carefully. For simplicity:
        #  - f1's first conv: in_c=8
        #  - f2's first conv: in_c=32 (the out of f1)

        # We'll pass in_c at construction time if we want each block independent. 
        # Another approach: you can define a single `_make_features(layers, filters)` 
        # if you want more dynamic.
        #
        # For this demonstration, let's do it more explicitly in forward().
        # We'll define the “first block” with in_c=8, second with in_c=32, etc.
        # We’ll handle that in forward with a “block1_in_c=8, block2_in_c=32,” etc. 
        return block

    def forward_block(self, x, block_id=1):
        """
        Actually build/run the conv layers on-the-fly for demonstration. 
        If you prefer, you can define them statically in __init__ like in the normal VGG.
        """
        # For the sake of matching your older approach:
        #  - block1: 4 conv layers (out_channels=32), then maxpool2d
        #  - block2: 3 conv layers (out_channels=64), then maxpool2d
        # We'll do it “inline” here. 
        if block_id == 1:
            out_channels = 32
            num_convs    = 4
        else:
            out_channels = 64
            num_convs    = 3

        in_c = x.size(1)  # dynamic (8 for block1, 32 for block2)
        for _ in range(num_convs):
            conv = nn.Conv2d(in_c, out_channels, kernel_size=3, padding=1)
            x = conv(x)
            x = nn.functional.relu(x, inplace=True)
            in_c = out_channels

        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        return x

    def forward(self, x_nf, x_nchw):
        """
        x_nf:   shape [N, F]
        x_nchw: shape [N, 8, H, W], e.g. [N, 8, 11, 11]
        """
        # Pass x_nchw through block1 => block2
        x_nchw = self.forward_block(x_nchw, block_id=1)  # => out_channels=32
        x_nchw = self.forward_block(x_nchw, block_id=2)  # => out_channels=64
        # shape might now be [N, 64, 2, 2] => flatten to [N, 256]
        x_nchw = x_nchw.view(x_nchw.size(0), -1)
        # Then a linear transform from 256 => 512
        x_nchw = self.post_conv_fc(x_nchw)  # => [N, 512]

        # If you want to combine x_nf with x_nchw, do so here:
        # e.g.  x_cat = torch.cat([x_nf, x_nchw], dim=1)
        # But your old code is slightly ambiguous if you want that or not. 
        # If you DO want to cat them, you must ensure the fc input dimension is (512 + F).
        # Let’s assume you do:
        x_cat = torch.cat([x_nf, x_nchw], dim=1)

        # For a real match, you must define how large x_nf is. 
        # Suppose x_nf has dimension F=0 or is simply not used. 
        # Then just remove x_nf or set F=some number and adjust linear layer dims. 
        # Example: if x_nf has dimension 10 => then we do 
        #   self.fc = nn.Linear(512 + 10, 512) 
        # in __init__.

        # Here, to keep it consistent, let's assume x_nf is empty 
        # or we let x_nf=0 dims. That means x_cat has dimension 512. 
        # We'll keep the code but note that you must adjust if you do have additional features.

        x_cat = self.fc(x_cat)
        out   = self.output(x_cat)
        return out

# -------------------------------------------------------------------
# 6) VGGPlusModel in PyTorch
# -------------------------------------------------------------------
class VGGPlusModel(BaseModel):
    """
    A PyTorch version of your old 'VGGPlusModel'.
    It handles multi-input: 
     - 'nf' = N x F (flat features)
     - 'nchw' = N x C x H x W (image-like features)
    Then merges them in a single forward pass.
    """

    def __init__(self, logger, auto_save=True):
        # We keep the transform that yields (NCHW) data for part of it, 
        # but also we might have additional "base" features from the transform.
        self.transform = NCHWTransform(logger, size=11)
        super().__init__("VGG5Plus", None, logger, self.transform, auto_save=auto_save)

        self.logger = logger
        self.model_ctx = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"VGGPlusModel using device {self.model_ctx}")

        # We define the actual PyTorch block:
        self.model = PyTorchVGGPlusBlock().to(self.model_ctx)

        # Basic loss and optimizer
        # You used HuberLoss in MXNet; let's do nn.SmoothL1Loss (PyTorch's name for Huber).
        self.loss_function = nn.SmoothL1Loss(beta=5.0)  # beta ~ 'rho' in MXNet
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

        # Some training hyperparams
        self.epochs = 2
        self.batch_size = 512
        self.progress_metric_df = pd.DataFrame(columns=['epoch','last_batch_loss','mse_train','mse_val'])

    # -------------- Utilities for dataset + loader --------------
    def xds_to_nf_and_nchw(self, rxds):
        """
        Given the xarray from FeatureSelectionTransformationNCHW,
        return (X_nf, X_nchw) as numpy arrays or torch Tensors.
        We'll keep them as numpy for now, then convert to torch later.
        """
        # For the "flat" features, exclude 'QQ' and any other columns you don't want
        # In the old code, you do something like:
        #   self.features = [f for f in base_transform.input_columns if not f.startswith('cav_')]
        #   self.channels = ...
        # You can replicate that logic or just keep it simple:

        # We might simply do:
        #  1) X_nf = rxds['base'].loc[dict(base_fields=<some set>)]
        #  2) X_nchw = rxds['cav'].loc[dict(channel=<some subset>)]
        #  3) shape => [N, C, H, W]
        # For demonstration, we do the same approach as your old code:

        features = [f for f in self.transform.base_transform.input_columns if f != 'QQ']
        channels = [c for c in cav.FeatureSelectionTransformationNCHW.channels if c != 'origin']

        X_nf = rxds['base'].loc[dict(base_fields=features)].astype(np.float32).values
        X_nchw = rxds['cav'].loc[dict(channel=channels)].values

        # X_nchw shape is (N, #channels, H, W)
        # X_nf shape is (N, F)
        return X_nf, X_nchw

    class MultiInputTensorDataset(Dataset):
        """
        A simple Dataset that holds three arrays: x_nf, x_nchw, y
        Each __getitem__ returns (nf_row, nchw_row, y_row).
        """
        def __init__(self, x_nf, x_nchw, y=None):
            self.x_nf = x_nf
            self.x_nchw = x_nchw
            self.y = y
            self.has_labels = (y is not None)
            assert len(x_nf) == len(x_nchw), "Mismatch in x_nf / x_nchw length!"

        def __len__(self):
            return len(self.x_nf)

        def __getitem__(self, idx):
            nf   = self.x_nf[idx]
            nchw = self.x_nchw[idx]
            if self.has_labels:
                return nf, nchw, self.y[idx]
            else:
                return nf, nchw

    # -------------- Training + Fitting --------------
    def fit_xds(self, rxds, eval_on_train=False, batch_size=256, epochs=2, verbose=1, validation_split=0.1, model_save_path=None):
        """
        Train on xarray dataset that includes 'QQ' as the label.
        This replaces your old fit_(...) approach in MXNet.
        """
        X_nf, X_nchw = self.xds_to_nf_and_nchw(rxds)
        y = rxds['base'].loc[dict(base_fields='QQ')].astype(np.float32).values

        return self._fit_arrays(X_nf, X_nchw, y,
                                eval_on_train=eval_on_train,
                                batch_size=batch_size,
                                epochs=epochs,
                                verbose=verbose,
                                validation_split=validation_split,
                                model_save_path=model_save_path)

    def _fit_arrays(self, X_nf, X_nchw, y, eval_on_train=False, batch_size=256, epochs=2, verbose=1, validation_split=0.1, model_save_path=None):
        # Possibly shuffle:
        n = len(X_nf)
        indices = np.arange(n)
        np.random.shuffle(indices)
        X_nf   = X_nf[indices]
        X_nchw = X_nchw[indices]
        y      = y[indices]

        # Train/Val split
        val_size = int(n * validation_split)
        train_size = n - val_size
        train_nf, val_nf = X_nf[:train_size], X_nf[train_size:]
        train_nchw, val_nchw = X_nchw[:train_size], X_nchw[train_size:]
        train_y, val_y = y[:train_size], y[train_size:]

        # Build Datasets
        train_ds = self.MultiInputTensorDataset(train_nf, train_nchw, train_y)
        val_ds   = self.MultiInputTensorDataset(val_nf, val_nchw, val_y)
        # DataLoaders
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        self.batch_size = batch_size
        self.epochs = epochs
        self.progress_metric_df = pd.DataFrame(columns=['epoch','last_batch_loss','mse_train','mse_val'])

        # If we want to save after each epoch and have not specified a path, generate one
        if self.auto_save and model_save_path is None:
            model_save_path = self.create_model_save_path()
            self.path = model_save_path
        self.logger.info(f"Training VGGPlusModel, save_path: {model_save_path}")

        for epoch_i in range(epochs):
            self.model.train()
            running_loss = 0.0
            steps = 0
            for (xb_nf, xb_nchw, yb) in train_loader:
                xb_nf   = xb_nf.to(self.model_ctx)
                xb_nchw = xb_nchw.to(self.model_ctx)
                yb      = yb.to(self.model_ctx).view(-1,1)

                self.optimizer.zero_grad()
                preds = self.model(xb_nf, xb_nchw)
                loss  = self.loss_function(preds, yb)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * xb_nf.size(0)
                steps += xb_nf.size(0)

            last_batch_loss = loss.item()
            mse_train = 0.0
            if eval_on_train:
                mse_train = self._score_arrays(train_nf, train_nchw, train_y)
            else:
                # average training loss in MSE terms
                # or we can just store the final batch’s value
                # or store the average SmoothL1 if you prefer
                preds_array = self.predict_(train_nf, train_nchw)
                mse_train   = sklearn.metrics.mean_squared_error(train_y, preds_array)

            mse_val = 0.0
            if val_size > 0:
                mse_val = self._score_arrays(val_nf, val_nchw, val_y)

            self.progress_metric_df.loc[len(self.progress_metric_df)] = [
                epoch_i, last_batch_loss, mse_train, mse_val
            ]

            if verbose:
                print(f"Epoch {epoch_i+1}/{epochs}, last_batch_loss={last_batch_loss:.4f}, mse_train={mse_train:.4f}, mse_val={mse_val:.4f}")

            # Auto-save
            if self.auto_save and model_save_path is not None:
                self.save(model_save_path)

        return self

    # -------------- Scoring + Prediction --------------
    def predict_(self, X_nf, X_nchw):
        """
        Low-level prediction that returns a numpy array of shape (N,).
        """
        self.model.eval()

        ds = self.MultiInputTensorDataset(X_nf, X_nchw, None)
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=False)

        out_list = []
        with torch.no_grad():
            for (xb_nf, xb_nchw) in loader:
                xb_nf   = xb_nf.to(self.model_ctx)
                xb_nchw = xb_nchw.to(self.model_ctx)
                preds   = self.model(xb_nf, xb_nchw)
                out_list.append(preds.cpu().numpy())

        out_cat = np.concatenate(out_list, axis=0)  # shape [N,1]
        return out_cat.ravel()

    def _score_arrays(self, X_nf, X_nchw, y):
        """
        Returns MSE between predictions and y.
        """
        preds = self.predict_(X_nf, X_nchw)
        return sklearn.metrics.mean_squared_error(y, preds)

    def predict_xds(self, xds):
        """
        Predict directly from an xarray dataset produced by NCHW transform.
        """
        X_nf, X_nchw = self.xds_to_nf_and_nchw(xds)
        return self.predict_(X_nf, X_nchw)

    def fit_file(self, id, augment_with_penalty_moves=False, eval_on_train=False,
                 batch_size=256, epochs=2, verbose=1, validation_split=0.1, model_save_path=None):
        """
        Load an xds from file using transform.batch_transform(),
        then call fit_xds.
        """
        rxds = self.transform.batch_transform(id, augment_with_penalty_moves=augment_with_penalty_moves)
        return self.fit_xds(rxds,
                            eval_on_train=eval_on_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=verbose,
                            validation_split=validation_split,
                            model_save_path=model_save_path)

    def predict(self, X):
        """
        If X is a DataFrame, transform to xarray, then run predict_xds.
        Otherwise, if X is already an xarray, pass directly.
        """
        if isinstance(X, pd.DataFrame):
            # Extract only the needed columns
            X = X[self.transform.base_transform.input_columns]
            t = cav.FeatureSelectionTransformationNCHW(X)
            rxds = t.transform()
            return self.predict_xds(rxds)
        else:
            # assume it's an xarray (rxds)
            return self.predict_xds(X)

    # -------------- Plot & Logging --------------
    def plot(self):
        """
        Simple method to plot MSE on train/val across epochs.
        """
        ldf = self.progress_metric_df
        plt.figure()
        plt.plot(ldf['mse_train'].values, label='train')
        plt.plot(ldf['mse_val'].values, label='val')
        plt.ylabel('MSE')
        plt.xlabel('epoch')
        plt.legend()
        plt.show()

    # -------------- Overriding BaseModel Save/Load --------------
    def save(self, file_name):
        """
        Save both the model’s state and (optionally) the optimizer state if desired.
        """
        torch.save({
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
        }, file_name)

    def load(self, id=None, suffix=None, latest=True):
        """
        Look for .pt file. If found, do torch.load(...) 
        Then load model + optimizer states. 
        """
        model_load_path = get_model_load_path(
            self.name,
            suffix=DEFAULT_MODEL_SUFFIX if suffix is None else suffix,
            selected_id=id,
            latest=latest
        )
        if model_load_path and os.path.isfile(model_load_path):
            self.logger.info(f"Loading VGGPlusModel from: {model_load_path}")
            self.load_(model_load_path)
        else:
            self.logger.info("No model found to load.")

    def load_(self, model_load_path):
        checkpoint = torch.load(model_load_path, map_location=self.model_ctx)
        self.model.load_state_dict(checkpoint['model_state'])
        if 'optimizer_state' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.model.eval()
        self.logger.info(f"Successfully loaded model from {model_load_path}")