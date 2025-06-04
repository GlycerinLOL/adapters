import torch
import torch.nn as nn

from transformers.modeling_outputs import CausalLMOutput, CausalLMOutputWithPast, MaskedLMOutput, Seq2SeqLMOutput
from typing import Optional, Tuple, Union

from ..methods.modeling import Activation_Function_Class
from ..composition import MoE
from .base import PredictionHead


class CausalLMHead(PredictionHead):
    _tied_weights_keys = []

    def __init__(
        self,
        model,
        head_name,
        vocab_size=None,
        embedding_size=None,
        layers=1,
        activation_function=None,
        layer_norm=False,
        bias=False,
        shift_labels=True,
        dropout_prob=None,
    ):
        super(CausalLMHead, self).__init__(head_name)
        self.config = {
            "head_type": "causal_lm",
            "vocab_size": vocab_size or model.config.vocab_size,
            "embedding_size": embedding_size or getattr(model.config, "embedding_size", model.config.hidden_size),
            "layers": layers,
            "activation_function": activation_function,
            "layer_norm": layer_norm,
            "bias": bias,
            "shift_labels": shift_labels,
            "label2id": None,
            "dropout_prob": dropout_prob,
        }
        self.build(model)

    def build(self, model):
        model_config = model.config
        # Additional FC layers
        pred_head = []
        with_layer_norm = self.config.get("layer_norm", False)
        embedding_size = self.config.get("embedding_size", model_config.hidden_size)

        for l_id in range(self.config["layers"] - 1):
            if l_id == 0:
                pred_head.append(nn.Linear(model_config.hidden_size, embedding_size))
            else:
                pred_head.append(nn.Linear(embedding_size, embedding_size))
            if self.config["activation_function"]:
                pred_head.append(Activation_Function_Class(self.config["activation_function"]))
            if with_layer_norm:
                eps = getattr(model_config, "layer_norm_eps", 1e-12)
                pred_head.append(nn.LayerNorm(embedding_size, eps=eps))

        for i, module in enumerate(pred_head):
            self.add_module(str(i), module)

        # Final embedding layer
        self.add_module(
            str(len(pred_head)),
            nn.Linear(
                embedding_size,
                self.config["vocab_size"],
                bias=self.config["bias"],
            ),
        )
        self._tied_weights_keys.append(f"{len(pred_head)}.*")

        self.apply(model._init_weights)
        self.train(model.training)  # make sure training mode is consistent

    def get_output_embeddings(self):
        # The last child is our embedding layer
        return self._modules[next(reversed(self._modules))]

    def set_output_embeddings(self, new_embeddings):
        # The last child is our embedding layer
        self._modules[next(reversed(self._modules))] = new_embeddings

    @staticmethod
    def _create_model_output(loss, logits, base_outputs):
        if "past_key_values" in base_outputs:
            return CausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                hidden_states=base_outputs.hidden_states,
                attentions=base_outputs.attentions,
                past_key_values=base_outputs.past_key_values,
            )
        else:
            return CausalLMOutput(
                loss=loss,
                logits=logits,
                hidden_states=base_outputs.hidden_states,
                attentions=base_outputs.attentions,
            )
            
    def load_balancing_loss_func(
        self, gate_logits: Union[torch.Tensor, Tuple[torch.Tensor]], num_experts: torch.Tensor = None, top_k=2, attention_mask: Optional[torch.Tensor] = None
    ) -> float:
        r"""
        Computes auxiliary load balancing loss as in Switch Transformer - implemented in Pytorch.

        See Switch Transformer (https://arxiv.org/abs/2101.03961) for more details. This function implements the loss
        function presented in equations (4) - (6) of the paper. It aims at penalizing cases where the routing between
        experts is too unbalanced.

        Args:
            gate_logits (Union[`torch.Tensor`, Tuple[torch.Tensor]):
                Logits from the `gate`, should be a tuple of model.config.num_hidden_layers tensors of
                shape [batch_size X sequence_length, num_experts].
            attention_mask (`torch.Tensor`, None):
                The attention_mask used in forward function
                shape [batch_size X sequence_length] if not None.
            num_experts (`int`, *optional*):
                Number of experts

        Returns:
            The auxiliary loss.
        """
        if gate_logits is None or not isinstance(gate_logits, tuple):
            return 0

        if isinstance(gate_logits, tuple):
            compute_device = gate_logits[0].device
            concatenated_gate_logits = torch.cat([layer_gate.to(compute_device) for layer_gate in gate_logits], dim=0)

        routing_weights = torch.nn.functional.softmax(concatenated_gate_logits, dim=-1)

        _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)

        expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)

        if attention_mask is None:
            # Compute the percentage of tokens routed to each experts
            tokens_per_expert = torch.mean(expert_mask.float(), dim=0)

            # Compute the average probability of routing to these experts
            router_prob_per_expert = torch.mean(routing_weights, dim=0)
        else:
            batch_size, sequence_length = attention_mask.shape
            num_hidden_layers = concatenated_gate_logits.shape[0] // (batch_size * sequence_length)

            # Compute the mask that masks all padding tokens as 0 with the same shape of expert_mask
            expert_attention_mask = (
                attention_mask[None, :, :, None, None]
                .expand((num_hidden_layers, batch_size, sequence_length, top_k, num_experts))
                .reshape(-1, top_k, num_experts)
                .to(compute_device)
            )

            # Compute the percentage of tokens routed to each experts
            tokens_per_expert = torch.sum(expert_mask.float() * expert_attention_mask, dim=0) / torch.sum(
                expert_attention_mask, dim=0
            )

            # Compute the mask that masks all padding tokens as 0 with the same shape of tokens_per_expert
            router_per_expert_attention_mask = (
                attention_mask[None, :, :, None]
                .expand((num_hidden_layers, batch_size, sequence_length, num_experts))
                .reshape(-1, num_experts)
                .to(compute_device)
            )

            # Compute the average probability of routing to these experts
            router_prob_per_expert = torch.sum(routing_weights * router_per_expert_attention_mask, dim=0) / torch.sum(
                router_per_expert_attention_mask, dim=0
            )

        overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
        return overall_loss * num_experts

    def forward(self, outputs, cls_output=None, attention_mask=None, return_dict=False, **kwargs):
        # First, pass through all layers except the last embedding layer
        seq_outputs = outputs[0]
        for i in range(len(self) - 1):
            seq_outputs = self[i](seq_outputs)

        # Now, pass through an invertible adapter if available
        inv_adapter = kwargs.pop("invertible_adapter", None)
        if inv_adapter is not None:
            seq_outputs = inv_adapter(seq_outputs, rev=True)

        # Finally, pass through the last embedding layer
        lm_logits = self[len(self) - 1](seq_outputs)

        loss = None
        labels = kwargs.pop("labels", None)
        moe_config = kwargs.pop("moe_config", None)
        router_logits = kwargs.pop("adapter_router_logits", None)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            if self.config["shift_labels"]:
                logits_for_loss = lm_logits[..., :-1, :].contiguous()
                labels = labels[..., 1:].contiguous()
            else:
                logits_for_loss = lm_logits

            # adjust labels for prompt tuning
            if kwargs.get("prompt_tokens_length", 0) > 0:
                prompt_length = kwargs.get("prompt_tokens_length")
                prompt_labels = torch.full(
                    (labels.shape[0], prompt_length), loss_fct.ignore_index, dtype=torch.long, device=labels.device
                )
                labels = torch.cat((prompt_labels, labels), dim=-1)

            loss = loss_fct(logits_for_loss.reshape(-1, self.config["vocab_size"]), labels.reshape(-1))

        if isinstance(moe_config, MoE):
            if moe_config.lb_loss:
                aux_loss = self.load_balancing_loss_func(
                    gate_logits=router_logits, num_experts=moe_config.num_experts, top_k=moe_config.top_k, attention_mask=None
                )
                if labels is not None:
                    loss += moe_config.lb_loss_weight * aux_loss.to(loss.device)

        if return_dict:
            return self._create_model_output(loss, lm_logits, outputs)
        else:
            outputs = (lm_logits,) + outputs[1:]
            if loss is not None:
                outputs = (loss,) + outputs
            return outputs


class Seq2SeqLMHead(CausalLMHead):
    def __init__(
        self,
        model,
        head_name,
        vocab_size=None,
        layers=1,
        activation_function=None,
        layer_norm=False,
        bias=False,
        shift_labels=False,
    ):
        super(CausalLMHead, self).__init__(head_name)
        self.config = {
            "head_type": "seq2seq_lm",
            "vocab_size": vocab_size or model.config.vocab_size,
            "layers": layers,
            "activation_function": activation_function,
            "layer_norm": layer_norm,
            "bias": bias,
            "shift_labels": shift_labels,
            "label2id": None,
        }
        self.build(model)

    @staticmethod
    def _create_model_output(loss, logits, base_outputs):
        return Seq2SeqLMOutput(
            loss=loss,
            logits=logits,
            past_key_values=base_outputs.past_key_values,
            decoder_hidden_states=base_outputs.decoder_hidden_states,
            decoder_attentions=base_outputs.decoder_attentions,
            cross_attentions=base_outputs.cross_attentions,
            encoder_last_hidden_state=base_outputs.encoder_last_hidden_state,
            encoder_hidden_states=base_outputs.encoder_hidden_states,
            encoder_attentions=base_outputs.encoder_attentions,
        )


class BertStyleMaskedLMHead(CausalLMHead):
    def __init__(
        self,
        model,
        head_name,
        vocab_size=None,
        embedding_size=None,
        layers=2,
        activation_function="gelu",
        layer_norm=True,
        bias=True,
        shift_labels=False,
    ):
        super(CausalLMHead, self).__init__(head_name)
        self.config = {
            "head_type": "masked_lm",
            "vocab_size": vocab_size or model.config.vocab_size,
            "embedding_size": embedding_size or getattr(model.config, "embedding_size", model.config.hidden_size),
            "layers": layers,
            "activation_function": activation_function,
            "layer_norm": layer_norm,
            "bias": bias,
            "shift_labels": shift_labels,
            "label2id": None,
        }
        self.build(model)

    @staticmethod
    def _create_model_output(loss, logits, base_outputs):
        return MaskedLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=base_outputs.hidden_states,
            attentions=base_outputs.attentions,
        )
