"""AIFI-A2: use P5 for queries/keys and gated P4-enhanced P5 for values."""

import torch
import torch.nn.functional as F

from .aifi_s4_qk_gated_v import AIFI_S4QK_GatedV

__all__ = ("AIFI_S5QK_GatedV",)


class AIFI_S5QK_GatedV(AIFI_S4QK_GatedV):
    """Cross-scale AIFI with Q/K from P5 and V from P5 plus gated, downsampled P4."""

    def forward(self, x):
        """Fuse ``[P4, P5]`` and return a P5-resolution feature map."""
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise ValueError(f"{type(self).__name__} forward expects a two-item [P4, P5] sequence")

        p4, p5 = x
        p4 = self.p4_down(p4)
        p5 = self.p5_proj(p5)
        if p4.shape[-2:] != p5.shape[-2:]:
            p4 = F.interpolate(p4, size=p5.shape[-2:], mode="bilinear", align_corners=False)

        gate = torch.sigmoid(self.gate(torch.cat((p5, p4), dim=1)))
        value_map = p5 + self.gate_scale * gate * p4

        # A2 keeps the attention relation anchored to the semantically stable P5, so the
        # Q/K source and the residual stream are one and the same token tensor.
        residual = self._to_tokens(p5)
        qk = residual + self._pos_embed(p5)
        value = self._to_tokens(value_map)

        attended = self.attn(qk, qk, value=value, need_weights=False)[0]
        tokens = self.norm1(residual + self.dropout1(attended))
        ffn = self.fc2(self.dropout(self.act(self.fc1(tokens))))
        tokens = self.norm2(tokens + self.dropout2(ffn))
        return tokens.transpose(1, 2).reshape(p5.shape).contiguous()
