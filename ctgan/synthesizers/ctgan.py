"""CTGAN module."""

import warnings
# -----
from ctgan import data
from ctgan.synthesizers.transformer_generator import TransformerGenerator
from ctgan.synthesizers.hybrid_generator import HybridGenerator
from ctgan.synthesizers.improved_transformer_generator import ImprovedTransformerGenerator


import numpy as np
import pandas as pd
import torch
from torch import optim
from torch.nn import BatchNorm1d, Dropout, LeakyReLU, Linear, Module, ReLU, Sequential, functional
from tqdm import tqdm

from ctgan.data_sampler import DataSampler
from ctgan.data_transformer import DataTransformer
from ctgan.errors import InvalidDataError
from ctgan.synthesizers.base import BaseSynthesizer, random_state
# -----

def sample_gumbel(shape, device='cpu', eps=1e-20):
    """Sample from Gumbel(0, 1) distribution."""
    U = torch.rand(shape, device=device)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, tau):
    """Sample from Gumbel-Softmax distribution."""
    g = sample_gumbel(logits.size(), device=logits.device)
    y = (logits + g) / tau
    return functional.softmax(y, dim=-1)


def gumbel_softmax(logits, tau, hard=False):
    """Gumbel-Softmax with optional straight-through hard sampling."""
    y = gumbel_softmax_sample(logits, tau)
    if hard:
        _, k = y.max(-1)
        y_hard = torch.zeros_like(y).scatter_(-1, k.unsqueeze(-1), 1.0)
        y = (y_hard - y).detach() + y
    return y


def compute_tau_per_column(cardinality, importance, step, total_steps,
                           tau_start=1.0, tau_final=0.1,
                           w_card=0.5, w_time=0.4, w_imp=0.1,
                           min_tau=0.05, max_tau=2.0, time_exp=1.0):
    """
    Compute adaptive temperature τ for each categorical column.
    
    Args:
        cardinality (list): Number of categories for each column
        importance (list): Importance weights (higher -> rarer/more important)
        step (int): Current training step
        total_steps (int): Total planned training steps
        tau_start (float): Initial temperature
        tau_final (float): Final temperature
        w_card (float): Weight for cardinality effect
        w_time (float): Weight for time annealing
        w_imp (float): Weight for importance
        min_tau (float): Minimum temperature
        max_tau (float): Maximum temperature
        time_exp (float): Time annealing exponent
    
    Returns:
        torch.Tensor: Temperature values for each column
    """
    if not cardinality:
        return torch.tensor([])
    
    card = torch.tensor(cardinality, dtype=torch.float32)
    imp = torch.tensor(importance, dtype=torch.float32)
    
    # -----
    max_card = card.max().clamp(min=1.0)
    tau_card = torch.log(card + 1.0) / torch.log(max_card + 1.0)
    tau_card = tau_card * (max_tau - min_tau) + min_tau
    
    # -----
    frac = (1.0 - float(step) / float(total_steps)) ** time_exp
    tau_time = tau_final + (tau_start - tau_final) * frac
    
    # -----
    tau_imp = (1.0 - imp) * (max_tau - min_tau) + min_tau
    
    # -----
    tau = w_card * tau_card + w_time * tau_time + w_imp * tau_imp
    tau = torch.clamp(tau, min=min_tau, max=max_tau)
    
    return tau


# -----
class Discriminator(Module):
    """Discriminator for the CTGAN."""

    def __init__(self, input_dim, discriminator_dim, pac=10):
        super(Discriminator, self).__init__()
        dim = input_dim * pac
        self.pac = pac
        self.pacdim = dim
        seq = []
        for item in list(discriminator_dim):
            seq += [Linear(dim, item), LeakyReLU(0.2), Dropout(0.5)]
            dim = item

        seq += [Linear(dim, 1)]
        self.seq = Sequential(*seq)

    def calc_gradient_penalty(self, real_data, fake_data, device='cpu', pac=10, lambda_=10):
        """Compute the gradient penalty."""
        alpha = torch.rand(real_data.size(0) // pac, 1, 1, device=device)
        alpha = alpha.repeat(1, pac, real_data.size(1))
        alpha = alpha.view(-1, real_data.size(1))

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        disc_interpolates = self(interpolates)

        gradients = torch.autograd.grad(
            outputs=disc_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones(disc_interpolates.size(), device=device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        gradients_view = gradients.view(-1, pac * real_data.size(1)).norm(2, dim=1) - 1
        gradient_penalty = ((gradients_view) ** 2).mean() * lambda_

        return gradient_penalty

    def forward(self, input_):
        """Apply the Discriminator to the `input_`."""
        assert input_.size()[0] % self.pac == 0
        return self.seq(input_.view(-1, self.pacdim))


class Residual(Module):
    """Residual layer for the CTGAN."""

    def __init__(self, i, o):
        super(Residual, self).__init__()
        self.fc = Linear(i, o)
        self.bn = BatchNorm1d(o)
        self.relu = ReLU()

    def forward(self, input_):
        """Apply the Residual layer to the `input_`."""
        out = self.fc(input_)
        out = self.bn(out)
        out = self.relu(out)
        return torch.cat([out, input_], dim=1)


class Generator(Module):
    """Generator for the CTGAN."""

    def __init__(self, embedding_dim, generator_dim, data_dim):
        super(Generator, self).__init__()
        dim = embedding_dim
        seq = []
        for item in list(generator_dim):
            seq += [Residual(dim, item)]
            dim += item
        seq.append(Linear(dim, data_dim))
        self.seq = Sequential(*seq)

    def forward(self, input_):
        """Apply the Generator to the `input_`."""
        data = self.seq(input_)
        return data


class CTGAN(BaseSynthesizer):
    """Conditional Table GAN Synthesizer.

    This is the core class of the CTGAN project, where the different components
    are orchestrated together.
    For more details about the process, please check the [Modeling Tabular data using
    Conditional GAN](https://arxiv.org/abs/1907.00503) paper.

    Args:
        embedding_dim (int):
            Size of the random sample passed to the Generator. Defaults to 128.
        generator_dim (tuple or list of ints):
            Size of the output samples for each one of the Residuals. A Residual Layer
            will be created for each one of the values provided. Defaults to (256, 256).
        discriminator_dim (tuple or list of ints):
            Size of the output samples for each one of the Discriminator Layers. A Linear Layer
            will be created for each one of the values provided. Defaults to (256, 256).
        generator_lr (float):
            Learning rate for the generator. Defaults to 2e-4.
        generator_decay (float):
            Generator weight decay for the Adam Optimizer. Defaults to 1e-6.
        discriminator_lr (float):
            Learning rate for the discriminator. Defaults to 2e-4.
        discriminator_decay (float):
            Discriminator weight decay for the Adam Optimizer. Defaults to 1e-6.
        batch_size (int):
            Number of data samples to process in each step.
        discriminator_steps (int):
            Number of discriminator updates to do for each generator update.
            From the WGAN paper: https://arxiv.org/abs/1701.07875. WGAN paper
            default is 5. Default used is 1 to match original CTGAN implementation.
        log_frequency (boolean):
            Whether to use log frequency of categorical levels in conditional
            sampling. Defaults to ``True``.
        verbose (boolean):
            Whether to have print statements for progress results. Defaults to ``False``.
        epochs (int):
            Number of training epochs. Defaults to 300.
        pac (int):
            Number of samples to group together when applying the discriminator.
            Defaults to 10.
        cuda (bool):
            Whether to attempt to use cuda for GPU computation.
            If this is False or CUDA is not available, CPU will be used.
            Defaults to ``True``.
    """

    def __init__(
        self,
        embedding_dim=128,
        generator_dim=(256, 256),
        discriminator_dim=(256, 256),
        generator_lr=2e-4,
        generator_decay=1e-6,
        discriminator_lr=2e-4,
        discriminator_decay=1e-6,
        batch_size=500,
        discriminator_steps=1,
        log_frequency=True,
        verbose=False,
        epochs=300,
        pac=10,
        cuda=True,
        #-----
        normalizer='vgm',
        normalizer_kwargs=None,
    ):
        assert batch_size % 2 == 0

        self._embedding_dim = embedding_dim
        self._generator_dim = generator_dim
        self._discriminator_dim = discriminator_dim

        self._generator_lr = generator_lr
        self._generator_decay = generator_decay
        self._discriminator_lr = discriminator_lr
        self._discriminator_decay = discriminator_decay

        self._batch_size = batch_size
        self._discriminator_steps = discriminator_steps
        self._log_frequency = log_frequency
        self._verbose = verbose
        self._epochs = epochs
        self.pac = pac

        if not cuda or not torch.cuda.is_available():
            device = 'cpu'
        elif isinstance(cuda, str):
            device = cuda
        else:
            device = 'cuda'

        self._device = torch.device(device)

        self._transformer = None
        self._data_sampler = None
        self._generator = None
        self.loss_values = None
        # -----
        self.normalizer = normalizer
        self.normalizer_kwargs = normalizer_kwargs or {}
        # -----

# -----

        
        # -----
        self._categorical_cardinality = []
        self._categorical_importance = []
        self._current_step = 0
        self._total_steps = 0


# -----
    @staticmethod
    def _gumbel_softmax(logits, tau=1, hard=False, eps=1e-10, dim=-1):
        """Deals with the instability of the gumbel_softmax for older versions of torch.

        For more details about the issue:
        https://drive.google.com/file/d/1AA5wPfZ1kquaRtVruCd6BiYZGcDeNxyP/view?usp=sharing

        Args:
            logits […, num_features]:
                Unnormalized log probabilities
            tau:
                Non-negative scalar temperature
            hard (bool):
                If True, the returned samples will be discretized as one-hot vectors,
                but will be differentiated as if it is the soft sample in autograd
            dim (int):
                A dimension along which softmax will be computed. Default: -1.

        Returns:
            Sampled tensor of same shape as logits from the Gumbel-Softmax distribution.
        """
        for _ in range(10):
            transformed = functional.gumbel_softmax(logits, tau=tau, hard=hard, eps=eps, dim=dim)
            if not torch.isnan(transformed).any():
                return transformed

        raise ValueError('gumbel_softmax returning NaN.')

    def _apply_activate(self, data):
        """Apply proper activation function to the output of the generator."""
        data_t = []
        st = 0
        # -----
        categorical_idx = 0

        # -----
        if self._categorical_cardinality and self._total_steps > 0:
            taus = compute_tau_per_column(
                self._categorical_cardinality,
                self._categorical_importance,
                self._current_step,
                self._total_steps
            ).to(self._device)
        else:
            # -----
            taus = torch.tensor([0.2] * len(self._categorical_cardinality)).to(self._device)

        # -----
        # -----
        def _span_dim(span):
            if isinstance(span, (tuple, list)):
                return int(span[0])
            if hasattr(span, "dim"):
                return int(span.dim)
            try:
                return int(span)
            except Exception:
                raise TypeError(f"Unexpected span format: {type(span)} - {span!r}")

        def _span_act(span):
            if isinstance(span, (tuple, list)):
                return span[1]
            if hasattr(span, "activation_fn"):
                return span.activation_fn
            return None
        # -----

        for column_info in self._transformer.output_info_list:
            for span_info in column_info:
                # -----
                span_act = _span_act(span_info)
                span_dim = _span_dim(span_info)

                if span_act == 'tanh':
                    ed = st + span_dim
                    data_t.append(torch.tanh(data[:, st:ed]))
                    st = ed
                elif span_act == 'softmax':
                    ed = st + span_dim
                    # -----
                    # transformed = self._gumbel_softmax(data[:, st:ed], tau=0.2) --- IGNORE ---
                    # -----
                    if categorical_idx < len(taus):
                        tau = taus[categorical_idx].item()
                    else:
                        tau = 0.2  # -----

                    transformed = gumbel_softmax(data[:, st:ed], tau=tau, hard=True)
                    data_t.append(transformed)
                    st = ed

                    # -----
                    categorical_idx += 1
                    # -----

                else:
                    # -----
                    raise ValueError(f'Unexpected activation function {span_act}.')

        return torch.cat(data_t, dim=1)
    
# #-----

#     def _initialize_categorical_info(self, train_data, discrete_columns):
#         """Initialize categorical column information for adaptive temperature scheduling."""
#         self._categorical_cardinality = []
#         self._categorical_importance = []
        
#         for column_transform_info in self._transformer._column_transform_info_list:
#             # Check if this is a discrete column
#             if column_transform_info.column_type == 'discrete':
#                 cardinality = column_transform_info.output_dimensions
#                 self._categorical_cardinality.append(cardinality)
                
#                 # Compute importance based on rarity (inverse frequency)
#                 column_name = column_transform_info.column_name
                
#                 if hasattr(train_data, 'columns') and column_name in train_data.columns:
#                     # Calculate category frequencies
#                     value_counts = train_data[column_name].value_counts()
#                     # Normalize to get inverse frequency as importance
#                     min_freq = value_counts.min()
#                     max_freq = value_counts.max()
#                     if max_freq > min_freq:
#                         avg_inv_freq = (max_freq - value_counts.mean()) / (max_freq - min_freq)
#                     else:
#                         avg_inv_freq = 0.5
#                     # Clip to [0, 1] range
#                     importance = np.clip(avg_inv_freq, 0.0, 1.0)
#                 else:
#                     # Default importance for unknown columns
#                     importance = 0.5
#                 
#                 self._categorical_importance.append(importance)

    
# #-----    

    # -----
    def _initialize_categorical_info(self, train_data, discrete_columns):
        """Initialize categorical info for adaptive temperature scheduling.

        Compatible with your DataTransformer._column_transform_info_list layout
        (list of dicts with keys: 'column_name', 'column_type', 'transform_info').
        """
        self._categorical_cardinality = []
        self._categorical_importance = []

        # -----
        if not hasattr(self._transformer, "_column_transform_info_list"):
            return

        # -----
        freq_importance = {}
        if hasattr(train_data, "columns"):
            for col in train_data.columns:
                # -----
                # -----
                pass

        # -----
        for info in self._transformer._column_transform_info_list:
            col_name = info.get("column_name")
            col_type = info.get("column_type")
            transform_info = info.get("transform_info", {})

            if col_type != "discrete":
                # -----
                continue

            # -----
            if "num_categories" in transform_info:
                cardinality = int(transform_info["num_categories"])
            elif "num_components" in transform_info:
                # -----
                cardinality = int(transform_info["num_components"])
            else:
                # -----
                cardinality = 1

            self._categorical_cardinality.append(cardinality)

            # -----
            importance = 0.5  # -----
            if hasattr(train_data, "columns") and col_name in train_data.columns:
                try:
                    vc = train_data[col_name].value_counts(dropna=True).astype(float)
                    # -----
                    if len(vc) > 0:
                        mean_freq = vc.mean()
                        max_freq = vc.max()
                        if max_freq > 0:
                            importance = float(1.0 - (mean_freq / max_freq))
                            # -----
                            importance = float(np.clip(importance, 0.0, 1.0))
                except Exception:
                    importance = 0.5

            self._categorical_importance.append(importance)

    # def _cond_loss(self, data, c, m):
    #     """Compute the cross entropy loss on the fixed discrete column."""
    #     loss = []
    #     st = 0
    #     st_c = 0
    #     for column_info in self._transformer.output_info_list:
    #         for span_info in column_info:
    #             if len(column_info) != 1 or span_info.activation_fn != 'softmax':
    #                 # not discrete column
    #                 st += span_info.dim
    #             else:
    #                 ed = st + span_info.dim
    #                 ed_c = st_c + span_info.dim
    #                 tmp = functional.cross_entropy(
    #                     data[:, st:ed], torch.argmax(c[:, st_c:ed_c], dim=1), reduction='none'
    #                 )
    #                 loss.append(tmp)
    #                 st = ed
    #                 st_c = ed_c

    #     loss = torch.stack(loss, dim=1)  # noqa: PD013

    #     return (loss * m).sum() / data.size()[0]
    def _cond_loss(self, data, c, m):
        """Compute the cross entropy loss on the fixed discrete column.

        This version reads output_info_list spans in both tuple and object forms.
        """
        loss = []
        st = 0
        st_c = 0

        def _span_dim(span):
            if isinstance(span, (tuple, list)):
                return int(span[0])
            if hasattr(span, "dim"):
                return int(span.dim)
            try:
                return int(span)
            except Exception:
                raise TypeError(f"Unexpected span format: {type(span)} - {span!r}")

        def _span_act(span):
            if isinstance(span, (tuple, list)):
                return span[1]
            if hasattr(span, "activation_fn"):
                return span.activation_fn
            return None

        for column_info in self._transformer.output_info_list:
            for span in column_info:
                act = _span_act(span)
                dim = _span_dim(span)
                if len(column_info) != 1 or act != 'softmax':
                    # -----
                    st += dim
                else:
                    ed = st + dim
                    ed_c = st_c + dim
                    tmp = functional.cross_entropy(
                        data[:, st:ed], torch.argmax(c[:, st_c:ed_c], dim=1), reduction='none'
                    )
                    loss.append(tmp)
                    st = ed
                    st_c = ed_c

        if not loss:
            # -----
            return torch.tensor(0.0, device=data.device)

        loss = torch.stack(loss, dim=1)  # -----
        return (loss * m).sum() / data.size()[0]
    def _validate_discrete_columns(self, train_data, discrete_columns):
        """Check whether ``discrete_columns`` exists in ``train_data``.

        Args:
            train_data (numpy.ndarray or pandas.DataFrame):
                Training Data. It must be a 2-dimensional numpy array or a pandas.DataFrame.
            discrete_columns (list-like):
                List of discrete columns to be used to generate the Conditional
                Vector. If ``train_data`` is a Numpy array, this list should
                contain the integer indices of the columns. Otherwise, if it is
                a ``pandas.DataFrame``, this list should contain the column names.
        """
        if isinstance(train_data, pd.DataFrame):
            invalid_columns = set(discrete_columns) - set(train_data.columns)
        elif isinstance(train_data, np.ndarray):
            invalid_columns = []
            for column in discrete_columns:
                if column < 0 or column >= train_data.shape[1]:
                    invalid_columns.append(column)
        else:
            raise TypeError('``train_data`` should be either pd.DataFrame or np.array.')

        if invalid_columns:
            raise ValueError(f'Invalid columns found: {invalid_columns}')

    def _validate_null_data(self, train_data, discrete_columns):
        """Check whether null values exist in continuous ``train_data``.

        Args:
            train_data (numpy.ndarray or pandas.DataFrame):
                Training Data. It must be a 2-dimensional numpy array or a pandas.DataFrame.
            discrete_columns (list-like):
                List of discrete columns to be used to generate the Conditional
                Vector. If ``train_data`` is a Numpy array, this list should
                contain the integer indices of the columns. Otherwise, if it is
                a ``pandas.DataFrame``, this list should contain the column names.
        """
        if isinstance(train_data, pd.DataFrame):
            continuous_cols = list(set(train_data.columns) - set(discrete_columns))
            any_nulls = train_data[continuous_cols].isna().any().any()
        else:
            continuous_cols = [i for i in range(train_data.shape[1]) if i not in discrete_columns]
            any_nulls = pd.DataFrame(train_data)[continuous_cols].isna().any().any()

        if any_nulls:
            raise InvalidDataError(
                'CTGAN does not support null values in the continuous training data. '
                'Please remove all null values from your continuous training data.'
            )

    @random_state
    # -----
    def fit(self, train_data, discrete_columns=(), epochs=None):
        """Fit the CTGAN Synthesizer models to the training data.

        Args:
            train_data (numpy.ndarray or pandas.DataFrame):
                Training Data. It must be a 2-dimensional numpy array or a pandas.DataFrame.
            discrete_columns (list-like):
                List of discrete columns to be used to generate the Conditional
                Vector. If ``train_data`` is a Numpy array, this list should
                contain the integer indices of the columns. Otherwise, if it is
                a ``pandas.DataFrame``, this list should contain the column names.
        """
        self._validate_discrete_columns(train_data, discrete_columns)
        self._validate_null_data(train_data, discrete_columns)

        if epochs is None:
            epochs = self._epochs
        else:
            warnings.warn(
                (
                    '`epochs` argument in `fit` method has been deprecated and will be removed '
                    'in a future version. Please pass `epochs` to the constructor instead'
                ),
                DeprecationWarning,
            )

        # self._transformer = DataTransformer()
        # self._transformer.fit(train_data, discrete_columns)
        # -----
        self._transformer = DataTransformer(
            normalizer_type=self.normalizer,
            normalizer_kwargs=self.normalizer_kwargs
        )
        self._transformer.fit(train_data, discrete_columns)

#-----


        # -----
        if isinstance(train_data, np.ndarray):
            if discrete_columns:
                # -----
                column_names = [f'col_{i}' for i in range(train_data.shape[1])]
                train_df = pd.DataFrame(train_data, columns=column_names)
                discrete_col_names = [column_names[i] for i in discrete_columns] if discrete_columns else []
            else:
                train_df = train_data
                discrete_col_names = []
        else:
            train_df = train_data
            discrete_col_names = discrete_columns

        # -----
        self._initialize_categorical_info(train_df, discrete_col_names)

       #-----
        train_data = self._transformer.transform(train_data)

        self._data_sampler = DataSampler(
            train_data, self._transformer.output_info_list, self._log_frequency
        )

        data_dim = self._transformer.output_dimensions

        # self._generator = Generator(
        #     self._embedding_dim + self._data_sampler.dim_cond_vec(), self._generator_dim, data_dim
        # ).to(self._device)
        
        # -----
        self._generator = HybridGenerator(
            embedding_dim=self._embedding_dim + self._data_sampler.dim_cond_vec(),
            generator_dim=self._generator_dim,
            data_dim=data_dim,
            nhead=2,
            num_layers=1
        ).to(self._device)
        discriminator = Discriminator(
            data_dim + self._data_sampler.dim_cond_vec(), self._discriminator_dim, pac=self.pac
        ).to(self._device)

        optimizerG = optim.Adam(
            self._generator.parameters(),
            lr=self._generator_lr,
            betas=(0.5, 0.9),
            weight_decay=self._generator_decay,
        )

        optimizerD = optim.Adam(
            discriminator.parameters(),
            lr=self._discriminator_lr,
            betas=(0.5, 0.9),
            weight_decay=self._discriminator_decay,
        )

        mean = torch.zeros(self._batch_size, self._embedding_dim, device=self._device)
        std = mean + 1

        self.loss_values = pd.DataFrame(columns=['Epoch', 'Generator Loss', 'Distriminator Loss'])

        epoch_iterator = tqdm(range(epochs), disable=(not self._verbose))
        if self._verbose:
            description = 'Gen. ({gen:.2f}) | Discrim. ({dis:.2f})'
            epoch_iterator.set_description(description.format(gen=0, dis=0))

        steps_per_epoch = max(len(train_data) // self._batch_size, 1)

        #-----
        
        # -----
        self._total_steps = epochs * steps_per_epoch
        self._current_step = 0
        
     #-----
        for i in epoch_iterator:
            for id_ in range(steps_per_epoch):
                #-----
                # -----
                self._current_step = i * steps_per_epoch + id_
                #-----
                for n in range(self._discriminator_steps):
                    fakez = torch.normal(mean=mean, std=std)

                    condvec = self._data_sampler.sample_condvec(self._batch_size)
                    if condvec is None:
                        c1, m1, col, opt = None, None, None, None
                        real = self._data_sampler.sample_data(
                            train_data, self._batch_size, col, opt
                        )
                    else:
                        c1, m1, col, opt = condvec
                        c1 = torch.from_numpy(c1).to(self._device)
                        m1 = torch.from_numpy(m1).to(self._device)
                        fakez = torch.cat([fakez, c1], dim=1)

                        perm = np.arange(self._batch_size)
                        np.random.shuffle(perm)
                        real = self._data_sampler.sample_data(
                            train_data, self._batch_size, col[perm], opt[perm]
                        )
                        c2 = c1[perm]

                    fake = self._generator(fakez)
                    fakeact = self._apply_activate(fake)

                    real = torch.from_numpy(real.astype('float32')).to(self._device)

                    if c1 is not None:
                        fake_cat = torch.cat([fakeact, c1], dim=1)
                        real_cat = torch.cat([real, c2], dim=1)
                    else:
                        real_cat = real
                        fake_cat = fakeact

                    y_fake = discriminator(fake_cat)
                    y_real = discriminator(real_cat)

                    pen = discriminator.calc_gradient_penalty(
                        real_cat, fake_cat, self._device, self.pac
                    )
                    loss_d = -(torch.mean(y_real) - torch.mean(y_fake))

                    optimizerD.zero_grad(set_to_none=False)
                    pen.backward(retain_graph=True)
                    loss_d.backward()
                    optimizerD.step()

                fakez = torch.normal(mean=mean, std=std)
                condvec = self._data_sampler.sample_condvec(self._batch_size)

                if condvec is None:
                    c1, m1, col, opt = None, None, None, None
                else:
                    c1, m1, col, opt = condvec
                    c1 = torch.from_numpy(c1).to(self._device)
                    m1 = torch.from_numpy(m1).to(self._device)
                    fakez = torch.cat([fakez, c1], dim=1)

                fake = self._generator(fakez)
                fakeact = self._apply_activate(fake)

                if c1 is not None:
                    y_fake = discriminator(torch.cat([fakeact, c1], dim=1))
                else:
                    y_fake = discriminator(fakeact)

                if condvec is None:
                    cross_entropy = 0
                else:
                    cross_entropy = self._cond_loss(fake, c1, m1)

                loss_g = -torch.mean(y_fake) + cross_entropy

                optimizerG.zero_grad(set_to_none=False)
                loss_g.backward()
                optimizerG.step()

            generator_loss = loss_g.detach().cpu().item()
            discriminator_loss = loss_d.detach().cpu().item()

            epoch_loss_df = pd.DataFrame({
                'Epoch': [i],
                'Generator Loss': [generator_loss],
                'Discriminator Loss': [discriminator_loss],
            })
            if not self.loss_values.empty:
                self.loss_values = pd.concat([self.loss_values, epoch_loss_df]).reset_index(
                    drop=True
                )
            else:
                self.loss_values = epoch_loss_df

            if self._verbose:
                epoch_iterator.set_description(
                    description.format(gen=generator_loss, dis=discriminator_loss)
                )

    @random_state
    def sample(self, n, condition_column=None, condition_value=None):
        """Sample data similar to the training data.

        Choosing a condition_column and condition_value will increase the probability of the
        discrete condition_value happening in the condition_column.

        Args:
            n (int):
                Number of rows to sample.
            condition_column (string):
                Name of a discrete column.
            condition_value (string):
                Name of the category in the condition_column which we wish to increase the
                probability of happening.

        Returns:
            numpy.ndarray or pandas.DataFrame
        """
        if condition_column is not None and condition_value is not None:
            condition_info = self._transformer.convert_column_name_value_to_id(
                condition_column, condition_value
            )
            global_condition_vec = self._data_sampler.generate_cond_from_condition_column_info(
                condition_info, self._batch_size
            )
        else:
            global_condition_vec = None

        steps = n // self._batch_size + 1
        data = []
        for i in range(steps):
            mean = torch.zeros(self._batch_size, self._embedding_dim)
            std = mean + 1
            fakez = torch.normal(mean=mean, std=std).to(self._device)

            if global_condition_vec is not None:
                condvec = global_condition_vec.copy()
            else:
                condvec = self._data_sampler.sample_original_condvec(self._batch_size)

            if condvec is None:
                pass
            else:
                c1 = condvec
                c1 = torch.from_numpy(c1).to(self._device)
                fakez = torch.cat([fakez, c1], dim=1)

            fake = self._generator(fakez)
            fakeact = self._apply_activate(fake)
            data.append(fakeact.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)
        data = data[:n]

        return self._transformer.inverse_transform(data)

    def set_device(self, device):
        """Set the `device` to be used ('GPU' or 'CPU)."""
        self._device = device
        if self._generator is not None:
            self._generator.to(self._device)
