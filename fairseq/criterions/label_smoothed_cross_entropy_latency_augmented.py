# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from fairseq.criterions import register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import (
    LabelSmoothedCrossEntropyCriterion,
)


@register_criterion("latency_augmented_label_smoothed_cross_entropy")
class LatencyAugmentedLabelSmoothedCrossEntropyCriterion(
    LabelSmoothedCrossEntropyCriterion
):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        ignore_prefix_size,
        report_accuracy,
        latency_weight_avg,
        latency_weight_avg_type,
        latency_weight_var,
        latency_weight_var_type,
        mass_preservation,
        average_method,
    ):
        super().__init__(
            task, sentence_avg, label_smoothing, ignore_prefix_size, report_accuracy
        )
        from examples.simultaneous_translation.utils.latency import LatencyTraining

        self.eps = label_smoothing
        self.latency_weight_avg = latency_weight_avg
        self.latency_weight_avg_type = latency_weight_avg_type
        self.latency_weight_var = latency_weight_var
        self.latency_weight_var_type = latency_weight_var_type
        self.mass_preservation = mass_preservation
        self.average_method = average_method
        self.latency_train = LatencyTraining(
            self.latency_weight_avg,
            self.latency_weight_var,
            self.latency_weight_avg_type,
            self.latency_weight_var_type,
            self.mass_preservation,
            self.average_method,
        )

    def forward(self, model, sample, update_num=None, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        # for speech-to-text simultaneous translation with fixed pre-decision
        # train_threshold = 0.8 + 0.2 * math.exp(-update_num / 60000)
        # if update_num < 4000:
        #     train_threshold = None
        # net_output = model(**sample["net_input"], train_threshold=train_threshold)
        net_output = model(**sample["net_input"])

        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        # pdb.set_trace()
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        return loss, sample_size, logging_output

    @staticmethod
    def add_args(parser):
        super(
            LatencyAugmentedLabelSmoothedCrossEntropyCriterion,
            LatencyAugmentedLabelSmoothedCrossEntropyCriterion,
        ).add_args(parser)
        # fmt: off

        """Add criterion-specific arguments to the parser."""
        parser.add_argument(
            "--label-smoothing",
            default=0.0,
            type=float,
            metavar="D",
            help="epsilon for label smoothing, 0 means no label smoothing",
        )
        parser.add_argument(
            "--ignore_prefix_size",
            default=0,
            type=int,
            help="ignore first N tokens",
        )
        parser.add_argument(
            "--report-accuracy",
            default=False,
            type=bool,
            help="report accuracy metric",
        )
        parser.add_argument("--latency-weight-avg", default=0., type=float, metavar='D',
                            help="Average loss weight")
        parser.add_argument("--latency-weight-var", default=0., type=float, metavar='D',
                            help="Variance loss weight")
        parser.add_argument("--latency-weight-avg-type", default="differentiable_average_lagging",
                            help="Statistics for Average loss type")
        parser.add_argument("--latency-weight-var-type", default="variance_delay",
                            help="Statistics for variance loss type")
        parser.add_argument("--average-method", default="weighted_average",
                            help="Average loss type")
        # fmt: on

    def compute_loss(self, model, net_output, sample, reduce=True):
        # Compute cross entropy loss first
        loss, nll_loss = super().compute_loss(model, net_output, sample, reduce)

        # Obtain the expected alignment
        attn_list = [item["alpha"] for item in net_output[-1]["attn_list"]]

        target_padding_mask = model.get_targets(sample, net_output).eq(self.padding_idx)

        source_padding_mask = net_output[-1].get("encoder_padding_mask", None)

        # Get latency loss
        latency_loss = self.latency_train.loss(
            attn_list, source_padding_mask, target_padding_mask
        )

        loss += latency_loss

        return loss, nll_loss
