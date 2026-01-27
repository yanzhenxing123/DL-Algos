# -*- encoding=utf-8 -*-
"""TokenMixer 与其依赖组件。

整体用途（结合 new_model.py 的使用方式）：
- 输入通常是把大特征向量 `deep_base` 切分成 token 序列后的张量 `x`，形状为 `[B, T, D]`。
- 先通过 `PerTokenDenseLayer` 将每个 token 映射到统一 `hidden_size`（例如 512）。
- 再通过 `TokenMixer` 堆叠多层：每层包含
  1) `mixup`：对 token 维与通道维做 reshape/transpose，改变 token 分桶方式（token mixing）。
  2) per-token FFN：对每个 token 做两层 MLP（channel mixing，且每个 token 可拥有独立参数）。
  3) 残差连接 + `LayerNormFP32`：稳定训练与数值。
- 最终仍输出 `[B, T', hidden_size]`，上游一般再对 token 维做 mean pooling 得到 `[B, hidden_size]`。

本文件的一个关键实现点：
- `BatchMatMulDense` 的权重形状是 `(S, D, D')`，其中 S 是 token 数。
  这意味着“每个 token 位置一套 Dense 权重”（token-wise parameterization），不同于共享权重的普通 Dense。
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sail import modules
from sail import initializers
from sail.initializers.base_initializer import Initializer
from sail import layers
from sail import optimizers
from sail.layers.base_layer import Layer
from sail.layers import Swish
from sail import tf
from sail.layers import Dense
from layers.utils import *
from layers.lhuc import *

SHARD_NUM = 2048


def mixup(inputs, new_tokens_num):
    """对 token 维做“重分桶/洗牌”。

    Args:
        inputs: `[B, T, D]`
        new_tokens_num: 新的 token 数 `T'`

    Returns:
        `[B, T', D']`，其中 `D' = D * T / T'`（要求 reshape 的整除关系成立）。

    直觉：把原来的 token×通道二维块重新 reshape+transpose，让信息在 token 维与通道维之间发生重排，
    从而实现 MLP-Mixer 风格的 token mixing。
    """
    _, token_nums, dims = inputs.get_shape().as_list()
    new_dims = dims * token_nums // new_tokens_num

    # 形状变化示意：
    # [B, T, D] -> [B, T, T', D/T'] -> transpose -> [B, T', T, D/T'] -> [B, T', D*T/T']
    output = tf.reshape(inputs, [-1, token_nums, new_tokens_num, dims // new_tokens_num])
    output = tf.transpose(output, [0, 2, 1, 3])
    output = tf.reshape(output, [-1, new_tokens_num, new_dims])  # [B, T', D*T/T']
    return output


class LayerNormFP32(Layer):
    # 该 Layer 显式在 FP32 下计算均值/方差与归一化，避免混合精度下 LN 数值不稳定。
    _support_mixed_precision_compute = False

    def __init__(self, name, out_type="stack", dtype=tf.float32, epsilon=1e-6,
                 mixed_precision=False, enable_batch_ln=False, **kwargs):
        if name is None or len(name) == 0:
            if "name" not in kwargs:
                kwargs["name"] = "layer_norm"
        else:
            kwargs["name"] = name
        super(LayerNormFP32, self).__init__(**kwargs)

        self.out_type = out_type
        self.beta = None
        self.gamma = None
        self.initializer = kwargs.get("initializer", initializers.Ones())
        self.dtype = dtype
        self.epsilon = epsilon
        self.mixed_precision = mixed_precision
        self.enable_batch_ln = enable_batch_ln

    def build(self, input_shape):
        assert len(input_shape) > 1
        token_dim = input_shape[-1]

        # enable_batch_ln=True 时，相当于“每个 token 位置单独一套 gamma/beta”，输入必须是 [S, B, D]
        # enable_batch_ln=False 时，gamma/beta 只与最后一维 D 相关（标准 LayerNorm）
        if self.enable_batch_ln:
            token_num = input_shape[0]
            self.beta = self.add_weight(
                name="beta",
                dtype=self.dtype,
                initial_value=initializers.Zeros()(shape=[token_num, 1, token_dim]),
            )
            self.gamma = self.add_weight(
                name="gamma",
                dtype=self.dtype,
                initial_value=self.initializer(shape=[token_num, 1, token_dim]),
            )
        else:
            self.beta = self.add_weight(
                name="beta",
                dtype=self.dtype,
                initial_value=initializers.Zeros()(shape=[token_dim]),
            )
            self.gamma = self.add_weight(
                name="gamma",
                dtype=self.dtype,
                initial_value=self.initializer(shape=[token_dim]),
            )

        self._snapshot_for_serving(self.beta, "beta")
        self._snapshot_for_serving(self.gamma, "gamma")

    def call(self, inputs, **kwargs):
        # 这里用 tf.nn.batch_normalization 实现 LayerNorm 的归一化公式：
        # - moments 统计的是最后一维 D 的 mean/var
        # - beta/gamma 分别对应偏移与缩放
        # 注意：包一层 serving_in_float32，确保统计与归一化在 fp32 下执行。
        with tf.name_scope(""):
            with tf.variable_scope(""):
                with S.serving_in_float32():
                    mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
                    output = tf.nn.batch_normalization(
                        inputs,
                        mean,
                        variance,
                        self.beta,
                        self.gamma,
                        variance_epsilon=self.epsilon,
                    )

                    if self.out_type == "stack":
                        ln_output = output
                    elif self.out_type == "concat":
                        ln_output = tf.concat(output, axis=1)
                    else:
                        ln_output = tf.unstack(output, axis=1)

        return ln_output


class PerTokenDenseLayer(Layer):
    """把每个 token 映射到统一 hidden_size。

    输入/输出形状：
    - 输入：`[B, T, D]`
    - 输出：`[B, T, hidden_size]`

    关键点：内部使用 `BatchMatMulDense`，其权重带 token 维 `(T, D, hidden_size)`，因此每个 token 位置
    都可以拥有不同的线性映射参数。
    """

    def __init__(self, hidden_size, dropout=0., mixed_precision=False, use_pre_ln=False, **xargs):
        super(PerTokenDenseLayer, self).__init__(**xargs)
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.mixed_precision = mixed_precision
        self.use_pre_ln = use_pre_ln

    def build(self, input_shape):
        self.fc = BatchMatMulDense(
            name=self.name + "bmm",
            hidden_size=self.hidden_size,
            kernel_initializer=VarianceScalingBatchMM(),
            bias_initializer=initializers.RandomNormal(),
            mixed_precision=self.mixed_precision,
            max_shard_num=SHARD_NUM
        )
        if self.use_pre_ln:
            self.ln = LayerNormFP32(name=self.name + "_pre_ln")
            print("use pre ln in PerTokenDenseLayer")

    def call(self, x):
        # `BatchMatMulDense` 期望输入是 [S, B, D]，这里把 token 维转到最前。
        # [B, T, D] -> [T, B, D]
        x = tf.transpose(x, perm=[1, 0, 2])

        # 可选：先做一次 pre-LN（同样用 fp32 计算）
        if self.use_pre_ln:
            x = self.ln(x)

        # token-wise Dense：权重形状 (T, D, hidden_size)
        x = self.fc(x)
        x = tf.nn.dropout(x, rate=self.dropout)

        # 转回 [B, T, hidden_size]
        x = tf.transpose(x, perm=[1, 0, 2])
        tf.summary.histogram(self.name + '_token_out', x)
        return x


class PerTokenFFN(Layer):
    def __init__(self, hidden_size, scale_ratio=4.0, dropout=0., mixed_precision=False, **xargs):
        super(PerTokenFFN, self).__init__(**xargs)
        self.hidden_size = hidden_size
        self.scale_ratio = scale_ratio
        self.dropout = dropout
        self.mixed_precision = mixed_precision

    def build(self, input_shape):
        self.fc1 = BatchMatMulDense(
            name=self.name + "bmm1",
            hidden_size=int(self.hidden_size * self.scale_ratio),
            kernel_initializer=VarianceScalingBatchMM(),
            bias_initializer=initializers.RandomNormal(),
            mixed_precision=self.mixed_precision,
            max_shard_num=SHARD_NUM
        )
        self.fc2 = BatchMatMulDense(
            name=self.name + "bmm2",
            hidden_size=self.hidden_size,
            kernel_initializer=VarianceScalingBatchMM(),
            bias_initializer=initializers.RandomNormal(),
            mixed_precision=self.mixed_precision,
            max_shard_num=SHARD_NUM
        )

    def call(self, x):
        # [B, T, D] -> [T, B, D], for token FFN
        x = tf.transpose(x, perm=[1, 0, 2])

        x = self.fc1(x)
        x = Swish()(x)
        x = tf.nn.dropout(x, rate=self.dropout)

        x = self.fc2(x)
        x = tf.nn.dropout(x, rate=self.dropout)

        # [T, B, D] -> [B, T, D]
        x = tf.transpose(x, perm=[1, 0, 2])
        tf.summary.histogram(self.name + '_token_out', x)
        return x


class PerTokenAFFN(Layer):
    def __init__(self, hidden_size, scale_ratio=4.0, lora_reduce_ratio=8.0, dropout=0., mixed_precision=False, **xargs):
        super(PerTokenAFFN, self).__init__(**xargs)
        self.hidden_size = hidden_size
        self.scale_ratio = scale_ratio
        self.lora_reduce_ratio = lora_reduce_ratio
        self.dropout = dropout
        self.mixed_precision = mixed_precision

    def build(self, input_shape):
        self.ffns = AdaptiveFFN(
            name=self.name + "_ffn",
            hidden_size=self.hidden_size,
            scale_ratio=self.scale_ratio,
            lora_reduce_ratio=self.lora_reduce_ratio,
            dropout=self.dropout,
            mixed_precision=self.mixed_precision
        )

    def call(self, x):
        # [B, T, D] -> [T, B, D], for token FFN
        x = tf.transpose(x, perm=[1, 0, 2])

        x = self.ffns(x)

        # [T, B, D] -> [B, T, D]
        x = tf.transpose(x, perm=[1, 0, 2])
        tf.summary.histogram(self.name + '_token_out', x)
        return x


class AdaptiveFFN(Layer):
    def __init__(self, hidden_size, scale_ratio=4.0, lora_reduce_ratio=8.0, dropout=0., mixed_precision=False, **xargs):
        super(AdaptiveFFN, self).__init__(**xargs)
        self.hidden_size = hidden_size
        self.scale_ratio = scale_ratio
        self.lora_reduce_ratio = lora_reduce_ratio
        self.dropout = dropout
        self.mixed_precision = mixed_precision

    def build(self, input_shape):
        self.fc1 = BatchMatMulDense(
            name=self.name + "bmm1",
            hidden_size=int(self.hidden_size * self.scale_ratio),
            kernel_initializer=VarianceScalingBatchMM(),
            bias_initializer=initializers.RandomNormal(),
            mixed_precision=self.mixed_precision,
            max_shard_num=SHARD_NUM
        )
        self.fc2 = BatchMatMulDense(
            name=self.name + "bmm2",
            hidden_size=self.hidden_size,
            kernel_initializer=VarianceScalingBatchMM(),
            bias_initializer=initializers.RandomNormal(),
            mixed_precision=self.mixed_precision,
            max_shard_num=SHARD_NUM
        )

        input_dim = input_shape[-1]

        self.adapt1_tower1 = BatchMatMulDense(
            name=self.name + "adpt11",
            hidden_size=int(input_dim / self.lora_reduce_ratio),
            kernel_initializer=VarianceScalingBatchMM(),
            bias_initializer=initializers.RandomNormal(),
            activation=Swish(),
            mixed_precision=self.mixed_precision,
            max_shard_num=SHARD_NUM
        )
        self.adapt1_tower2 = BatchMatMulDense(
            name=self.name + "adpt12",
            hidden_size=input_dim,
            kernel_initializer=VarianceScalingBatchMM(),
            bias_initializer=initializers.RandomNormal(),
            activation=None,
            mixed_precision=self.mixed_precision,
            max_shard_num=SHARD_NUM
        )

        self.adapt2_tower1 = BatchMatMulDense(
            name=self.name + "adpt21",
            hidden_size=int(self.hidden_size * self.scale_ratio / self.lora_reduce_ratio),
            kernel_initializer=VarianceScalingBatchMM(),
            bias_initializer=initializers.RandomNormal(),
            activation=Swish(),
            mixed_precision=self.mixed_precision,
            max_shard_num=SHARD_NUM
        )
        self.adapt2_tower2 = BatchMatMulDense(
            name=self.name + "adpt22",
            hidden_size=int(self.hidden_size * self.scale_ratio),
            kernel_initializer=VarianceScalingBatchMM(),
            bias_initializer=initializers.RandomNormal(),
            activation=None,
            mixed_precision=self.mixed_precision,
            max_shard_num=SHARD_NUM
        )

    def call(self, x):
        adaptive1 = self.adapt1_tower2(self.adapt1_tower1(x))
        adaptive2 = self.adapt2_tower2(self.adapt2_tower1(x))
        print("adaptive1 {}, adaptive2 {}".format(adaptive1, adaptive2))
        tf.summary.histogram('adaptive_ffn/adaptive1', adaptive1)
        tf.summary.histogram('adaptive_ffn/adaptive1', adaptive2)

        x = self.fc1(x * tf.tanh(adaptive1))
        x = Swish()(x)
        x = tf.nn.dropout(x, rate=self.dropout)

        x = self.fc2(x * tf.tanh(adaptive2))
        x = tf.nn.dropout(x, rate=self.dropout)
        return x


class PerTokenSwiGLU(Layer):
    def __init__(self, hidden_size, scale_ratio=4.0, lora_reduce_ratio=8.0, use_lora=False, dropout=0.,
                 mixed_precision=False, **xargs):
        super(PerTokenSwiGLU, self).__init__(**xargs)
        self.hidden_size = hidden_size
        self.scale_ratio = scale_ratio
        self.lora_reduce_ratio = lora_reduce_ratio
        self.dropout = dropout
        self.mixed_precision = mixed_precision
        self.use_lora = use_lora

    def build(self, input_shape):
        if self.use_lora:
            self.gate1 = BatchMatMulDense(
                name='{}_bmm_gate1'.format(self.name),
                hidden_size=int(self.hidden_size * self.scale_ratio / self.lora_reduce_ratio),
                kernel_initializer=VarianceScalingBatchMM(),
                bias_initializer=initializers.RandomNormal(),
                mixed_precision=self.mixed_precision,
                max_shard_num=SHARD_NUM
            )
            self.gate2 = BatchMatMulDense(
                name='{}_bmm_gate2'.format(self.name),
                hidden_size=int(self.hidden_size * self.scale_ratio),
                kernel_initializer=VarianceScalingBatchMM(),
                bias_initializer=initializers.RandomNormal(),
                mixed_precision=self.mixed_precision,
                max_shard_num=SHARD_NUM
            )
        else:
            self.gate = BatchMatMulDense(
                name='{}_bmm_gate'.format(self.name),
                hidden_size=int(self.hidden_size * self.scale_ratio),
                kernel_initializer=VarianceScalingBatchMM(),
                bias_initializer=initializers.RandomNormal(),
                mixed_precision=self.mixed_precision,
                max_shard_num=SHARD_NUM
            )

        self.up = BatchMatMulDense(
            name='{}_bmm_up'.format(self.name),
            hidden_size=int(self.hidden_size * self.scale_ratio),
            kernel_initializer=VarianceScalingBatchMM(),
            bias_initializer=initializers.RandomNormal(),
            mixed_precision=self.mixed_precision,
            max_shard_num=SHARD_NUM
        )
        self.down = BatchMatMulDense(
            name='{}_bmm_down'.format(self.name),
            hidden_size=self.hidden_size,
            kernel_initializer=VarianceScalingBatchMM(),
            bias_initializer=initializers.RandomNormal(),
            mixed_precision=self.mixed_precision,
            max_shard_num=SHARD_NUM
        )

    def call(self, x):
        x = tf.transpose(x, perm=[1, 0, 2])

        if self.use_lora:
            gate_proj = layers.Swish()(self.gate2(self.gate1(x)))
        else:
            gate_proj = layers.Swish()(self.gate(x))
        up_proj = self.up(x)
        gated_output = gate_proj * up_proj
        down_proj = self.down(gated_output)

        x = tf.transpose(down_proj, perm=[1, 0, 2])

        tf.summary.histogram('{}_swiglu_gate_proj'.format(self.name), gate_proj)
        tf.summary.histogram('{}_swiglu_up_proj'.format(self.name), up_proj)
        tf.summary.histogram('{}_swiglu_gated_output'.format(self.name), gated_output)
        tf.summary.histogram('{}_swiglu_down_proj'.format(self.name), down_proj)

        return x


class BatchMatMulDense(Layer):
    """Token-wise Dense（每个 token 一套权重）。

    与普通 Dense 的差异：
    - 普通 Dense：kernel 形状 `(D, D')`，所有 token 位置共享同一套权重。
    - 本实现：kernel 形状 `(S, D, D')`，S 为 token 数；第 s 个 token 使用 kernel[s]。

    输入形状约定：
    - build/call 期望输入为 `[S, B, D]`（token 维在最前）。
    """

    def __init__(
            self,
            hidden_size,
            activation=None,
            use_bias=True,
            kernel_initializer=initializers.RandomNormal(),
            bias_initializer=initializers.Zeros(),
            use_weight_norm=True,
            use_learnable_weight_norm=True,
            mixed_precision=False,
            optimizer=None,
            **xargs
    ):
        super(BatchMatMulDense, self).__init__(**xargs)
        self.hidden_size = hidden_size
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.use_weight_norm = use_weight_norm
        self.use_learnable_weight_norm = use_learnable_weight_norm
        self.mixed_precision = mixed_precision
        self.no_clip = xargs.get("no_clip", False)
        self.dtype = S.variable_dtype()

        if not S.is_compiling_training():
            self.use_weight_norm = False
            self.use_learnable_weight_norm = False

    def build(self, input_shape):
        # input_shape: [S, B, D]
        self.token_num, self.token_dim = input_shape[0], input_shape[-1]

        # kernel: (S, D, D')，表示每个 token 位置都有一套 (D->D') 的线性变换
        kernel_shape = (self.token_num, self.token_dim, self.hidden_size)
        print('batchmm kernel:', kernel_shape)
        init_kernel = self.kernel_initializer(kernel_shape, self.dtype)
        self.kernel = self.add_weight(initial_value=init_kernel, name="kernel", no_clip=self.no_clip)

        if self.use_weight_norm:
            self.kernel = tf.nn.l2_normalize(
                self.kernel, axis=1, epsilon=1e-4, name="normalized_kernel"
            )
            if self.use_learnable_weight_norm:
                init_trainable_kernel_norm = tf.norm(
                    init_kernel, axis=1, name="init_trainable_kernel_norm",
                    keepdims=True
                )
                self.trainable_kernel_norm = self.add_weight(
                    initial_value=init_trainable_kernel_norm,
                    name="trainable_kernel_norm",
                )
                self._snapshot_for_serving(None, "trainable_kernel_norm")
                self.kernel = tf.multiply(
                    self.kernel,
                    self.trainable_kernel_norm,
                    name="mul_of_kernel_and_trainable_norm",
                )
        self._snapshot_for_serving(self.kernel, "kernel")

        if self.use_bias:
            # (S, 1, D')
            bias_shape = (self.token_num, 1, self.hidden_size)
            init_bias = self.bias_initializer(bias_shape, self.dtype)
            self.bias = self.add_weight(initial_value=init_bias, name="bias", no_clip=self.no_clip)
            self._snapshot_for_serving(self.bias, "bias")
        else:
            self.bias = None

        if self.mixed_precision:
            self.kernel = tf.cast(self.kernel, tf.float16)
            if self.use_bias:
                self.bias = tf.cast(self.bias, tf.float16)

        self.built = True

    def call(self, inputs):
        # inputs: [S, B, D]
        # kernel: [S, D, D']
        # tf.matmul 会按最前维 S 做 batch-matmul，得到 [S, B, D']
        output = tf.matmul(inputs, self.kernel)
        if self.use_bias:
            output = output + self.bias

        if self.activation is not None:
            output = self.activation(output)

        self._register_for_debug("output_for_layer_{}".format(self.name), output)
        return output


class VarianceScalingBatchMM(Initializer):
    def __init__(self, scale=1.0, mode="fan_avg", distribution="normal", seed=None):
        if scale <= 0.0:
            raise SailValueError("`scale` must be a positive float. Got:", scale)
        mode = mode.lower()
        if mode not in {"fan_in", "fan_out", "fan_avg"}:
            raise SailValueError(
                "Invalid `mode` argument: "
                'expected on of {"fan_in", "fan_out", "fan_avg"}'
                "bug got",
                mode,
            )
        distribution = distribution.lower()
        if distribution not in {"normal", "uniform"}:
            raise SailValueError(
                "Invalid `distribution` argument: "
                'expected one of {"normal", "uniform"} '
                "but got",
                distribution,
            )

        self.scale = scale
        self.mode = mode
        self.distribution = distribution
        self.seed = seed

    def __call__(self, shape, dtype):
        # batchmm weights shape is (S, D, D')
        fan_in, fan_out = shape[-2], shape[-1]
        scale = self.scale
        if self.mode == "fan_in":
            scale /= max(1.0, fan_in)
        elif self.mode == "fan_out":
            scale /= max(1.0, fan_out)
        else:
            scale /= max(1.0, float(fan_in + fan_out) / 2)

        if self.distribution == "normal":
            # 0.879... = scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
            stddev = np.sqrt(scale) / 0.87962566103423978
            return tf.random.truncated_normal(shape, 0.0, stddev, dtype=dtype, seed=self.seed)
        else:
            limit = np.sqrt(3.0 * scale)
            return tf.random_uniform(shape, -limit, limit, dtype=dtype, seed=self.seed)

    def get_config(self):
        return {
            "scale": self.scale,
            "mode": self.mode,
            "distribution": self.distribution,
            "seed": self.seed,
        }


class TokenMixer(Layer):
    """TokenMixer 主体。

    Args:
        n_token_list: 每层 mixup 后的 token 数列表，例如 [32, 32, 32] 表示堆 3 层且每层都重分桶到 32 个 token。
        hidden_size: 每个 token 的通道维（通常与 PerTokenDenseLayer 输出一致）。
        ffn_type: per-token 子网络类型：'ffn'（两层 MLP）、'affn'（带自适应门控）、'swiglu'。
        use_res_map: 当 mixup 改变 token 数/通道数导致残差无法直接相加时，是否用一个映射把 residual 对齐。

    输入/输出：
        输入 `x`：`[B, T, hidden_size]`
        输出：`[B, T', hidden_size']`（由 mixup 与每层配置决定；在你们用法中通常保持 hidden_size 不变）。
    """

    def __init__(self, n_token_list, hidden_size, scale_ratio=4.0, lora_reduce_ratio=8.0, dropout=0.,
                 mixed_precision=False, ffn_type='ffn', use_res_map=False, **xargs):
        super(TokenMixer, self).__init__(**xargs)
        self.n_layer = len(n_token_list)
        self.n_token_list = n_token_list
        self.hidden_size = hidden_size
        self.mixed_precision = mixed_precision
        self.dropout = dropout
        self.ffn_type = ffn_type
        self.use_res_map = use_res_map
        self.scale_ratio = scale_ratio
        self.lora_reduce_ratio = lora_reduce_ratio
        self.perTokenFFNs = []
        self.ln = []
        self.res_map = []

    def build(self, input_shape):
        for i in range(self.n_layer):
            if self.ffn_type == 'affn':
                print("TokenMixer: using Adaptive FFN")
                tokenFFN = PerTokenAFFN(
                    name=self.name + "_adaffn_" + str(i),
                    hidden_size=self.hidden_size,
                    scale_ratio=self.scale_ratio,
                    lora_reduce_ratio=self.lora_reduce_ratio,
                    dropout=self.dropout,
                    mixed_precision=self.mixed_precision
                )
            elif self.ffn_type == 'ffn':
                print("TokenMixer: using FFN")
                tokenFFN = PerTokenFFN(
                    name=self.name + "_ffn_" + str(i),
                    hidden_size=self.hidden_size,
                    scale_ratio=self.scale_ratio,
                    dropout=self.dropout,
                    mixed_precision=self.mixed_precision
                )
            elif self.ffn_type == "swiglu":
                print("TokenMixer: using SwiGLU FFN")
                tokenFFN = PerTokenSwiGLU(
                    name=self.name + "_swigluffn_" + str(i),
                    hidden_size=self.hidden_size,
                    scale_ratio=self.scale_ratio,
                    use_lora=False,
                    dropout=self.dropout,
                    mixed_precision=self.mixed_precision
                )
            else:
                raise ValueError("TokenMixer: unknown ffn_type: {}".format(self.ffn_type))
            res = None
            if self.use_res_map:
                print("TokenMixer: using residual map")
                res = modules.DenseTower('{}_res_map_{}'.format(self.name, i),
                                         output_dims=[self.n_token_list[i]],
                                         initializers=initializers.GlorotNormal(mode='fan_in'),
                                         activations=Swish(),
                                         use_bias=True,
                                         )
            ln = LayerNormFP32(name=self.name + "_ln" + str(i))
            self.perTokenFFNs.append(tokenFFN)
            self.ln.append(ln)
            self.res_map.append(res)

    def call(self, x):
        # x: [B, T, D]
        tf.summary.histogram(self.name + "origin", x)
        total_gate_loss = 0.0

        for i in range(self.n_layer):
            # residual 分支（mixup 前）
            x_input = x  # [B, T, D]

            # 1) token mixing：改变 token 分桶方式，形状变为 [B, T_i, D_i]
            x = mixup(x, self.n_token_list[i])

            # 2) channel mixing：对每个 token 做 FFN（内部会转成 [T_i, B, D_i] 做 token-wise Dense）
            token_out = self.perTokenFFNs[i](x)
            tf.summary.histogram(self.name + "/pertokenFFNs" + str(i), token_out)

            # 3) 残差 + LN
            # - 若残差分支与 token_out 的最后一维一致，则直接相加
            # - 若不一致且开启 use_res_map，则把 residual 在 token 维上做对齐映射再相加
            if x.shape[-1] == token_out.shape[-1]:
                print("TokenMixer: directly res connection in layer {}".format(i))
                x = self.ln[i](x_input + tf.nn.dropout(token_out, rate=self.dropout))
            else:
                if self.use_res_map:
                    # res_map 期望输入形状为 [B, D, T]（把 token 当作“长度维”去映射到新的 token 数）
                    x_trans = tf.transpose(x_input, [0, 2, 1])
                    x_res = tf.transpose(self.res_map[i](x_trans), [0, 2, 1])
                    x = self.ln[i](x_res + tf.nn.dropout(token_out, rate=self.dropout))
                else:
                    x = self.ln[i](tf.nn.dropout(token_out, rate=self.dropout))

        return x
