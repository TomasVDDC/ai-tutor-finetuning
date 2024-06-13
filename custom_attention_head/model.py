from .configuration_openelm import OpenELMConfig
from modeling_openelm import OpenELMForCausalLM


class OpenELMCustomHeadForCausalLM(OpenELMForCausalLM):
    def __init__(self, config: OpenELMConfig):
        super().__init__(config)

        self.layers = nn.ModuleList([CondensedLlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        # Initialize weights and apply final processing
        self.post_init()