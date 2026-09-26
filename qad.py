"""
QAD: Quality-Aware Decoder for RT-DETR.
"""

import torch
import torch.nn as nn

from ..modules.head import RTDETRDecoder
from ..modules.transformer import (MLP, DeformableTransformerDecoder, DeformableTransformerDecoderLayer)
from ..modules.utils import inverse_sigmoid

__all__ = ("RTDETRDecoderQAD", "QueryRelationBias", "DeformableTransformerDecoderLayerRSA",
           "DeformableTransformerDecoderQAD")


def _pairwise_iou_cxcywh(boxes, eps=1e-7):
    """Pairwise IoU of ``(bs, n, 4)`` cxcywh boxes, returned as ``(bs, n, n)``."""
    cx, cy, w, h = boxes.unbind(-1)
    x1, y1 = cx - w * 0.5, cy - h * 0.5
    x2, y2 = cx + w * 0.5, cy + h * 0.5
    area = (w * h).clamp(min=0)

    lt_x = torch.maximum(x1[:, :, None], x1[:, None, :])
    lt_y = torch.maximum(y1[:, :, None], y1[:, None, :])
    rb_x = torch.minimum(x2[:, :, None], x2[:, None, :])
    rb_y = torch.minimum(y2[:, :, None], y2[:, None, :])
    inter = (rb_x - lt_x).clamp(min=0) * (rb_y - lt_y).clamp(min=0)
    return inter / (area[:, :, None] + area[:, None, :] - inter + eps)


class QueryRelationBias(nn.Module):
    """Per-head additive attention bias from the pairwise geometry of the reference boxes.

    Args:
        n_heads (int): number of attention heads the bias is produced for.
        hidden (int): hidden width of the bias MLP. Kept small because the intermediate
            tensor is ``(bs, n, n, hidden)``; at ``nq=300`` plus ~200 denoising queries and
            ``bs=4`` a hidden width of 16 costs ~64 MB per decoder layer.
        eps (float): numerical floor for the log-ratio features.
    """

    def __init__(self, n_heads=8, hidden=16, eps=1e-3):
        super().__init__()
        self.n_heads = n_heads
        self.eps = eps
        self.mlp = nn.Sequential(nn.Linear(5, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, n_heads))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)  # plain self-attention at init

    def forward(self, boxes):
        """Map ``(bs, n, 4)`` cxcywh boxes in [0, 1] to a ``(bs, n_heads, n, n)`` bias."""
        boxes = boxes.detach()  # the bias is a prior on attention, not a path into box regression
        cx, cy, w, h = boxes.unbind(-1)
        w = w.clamp(min=self.eps)
        h = h.clamp(min=self.eps)

        dx = (cx[:, :, None] - cx[:, None, :]).abs() / w[:, :, None]
        dy = (cy[:, :, None] - cy[:, None, :]).abs() / h[:, :, None]
        rw = w[:, :, None] / w[:, None, :]
        rh = h[:, :, None] / h[:, None, :]
        iou = _pairwise_iou_cxcywh(boxes)

        feat = torch.stack((torch.log(dx + self.eps), torch.log(dy + self.eps),
                            torch.log(rw), torch.log(rh), iou), dim=-1)
        return self.mlp(feat).permute(0, 3, 1, 2)  # (bs, n_heads, n, n)


class DeformableTransformerDecoderLayerRSA(DeformableTransformerDecoderLayer):
    """Decoder layer whose query self-attention carries a learned position-relation bias."""

    def __init__(self, d_model=256, n_heads=8, d_ffn=1024, dropout=0., act=nn.ReLU(), n_levels=4, n_points=4,
                 rsa_hidden=16):
        super().__init__(d_model, n_heads, d_ffn, dropout, act, n_levels, n_points)
        self.n_heads = n_heads
        self.relation = QueryRelationBias(n_heads, rsa_hidden)

    def forward(self, embed, refer_bbox, feats, shapes, padding_mask=None, attn_mask=None, query_pos=None):
        """Same as the stock layer, with the relation bias folded into the self-attention mask."""
        bs, nq = embed.shape[:2]

        # Self attention with the geometric relation prior.
        bias = self.relation(refer_bbox)  # (bs, nh, nq, nq)
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                bias = bias.masked_fill(attn_mask, float('-inf'))
            else:
                bias = bias + attn_mask
        sa_mask = bias.reshape(bs * self.n_heads, nq, nq)

        q = k = self.with_pos_embed(embed, query_pos)
        tgt = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), embed.transpose(0, 1),
                             attn_mask=sa_mask)[0].transpose(0, 1)
        embed = embed + self.dropout1(tgt)
        embed = self.norm1(embed)

        # Cross attention
        tgt = self.cross_attn(self.with_pos_embed(embed, query_pos), refer_bbox.unsqueeze(2), feats, shapes,
                              padding_mask)
        embed = embed + self.dropout2(tgt)
        embed = self.norm2(embed)

        # FFN
        return self.forward_ffn(embed)


class DeformableTransformerDecoderQAD(DeformableTransformerDecoder):
    """Deformable decoder that also emits a per-query localisation-quality logit."""

    def forward(
            self,
            embed,  # decoder embeddings
            refer_bbox,  # anchor
            feats,  # image features
            shapes,  # feature shapes
            bbox_head,
            score_head,
            qual_head,
            pos_mlp,
            attn_mask=None,
            padding_mask=None):
        """Perform the forward pass, returning boxes, class logits and quality logits."""
        output = embed
        dec_bboxes = []
        dec_cls = []
        dec_qual = []
        last_refined_bbox = None
        refer_bbox = refer_bbox.sigmoid()
        for i, layer in enumerate(self.layers):
            output = layer(output, refer_bbox, feats, shapes, padding_mask, attn_mask, pos_mlp(refer_bbox))

            bbox = bbox_head[i](output)
            refined_bbox = torch.sigmoid(bbox + inverse_sigmoid(refer_bbox))

            if self.training:
                dec_cls.append(score_head[i](output))
                dec_qual.append(qual_head(output))
                if i == 0:
                    dec_bboxes.append(refined_bbox)
                else:
                    dec_bboxes.append(torch.sigmoid(bbox + inverse_sigmoid(last_refined_bbox)))
            elif i == self.eval_idx:
                dec_cls.append(score_head[i](output))
                dec_qual.append(qual_head(output))
                dec_bboxes.append(refined_bbox)
                break

            last_refined_bbox = refined_bbox
            refer_bbox = refined_bbox.detach() if self.training else refined_bbox

        return torch.stack(dec_bboxes), torch.stack(dec_cls), torch.stack(dec_qual)


class RTDETRDecoderQAD(RTDETRDecoder):
    """RT-DETR decoder with a dense quality head, quality-fused scoring and relation attention.

    The QAD-specific arguments sit after ``ndl`` so a yaml can reach them positionally::

        [nc, hd, nq, ndp, nh, ndl,
         qual_beta, qual_gain, rsa, rsa_hidden,          # v1
         qual_tau, qual_theta, qual_lambda,              # v2  M1 + M2
         rank_gain, rank_topk, rank_delta,               # v2  M3
         nd,                                             # v2  M4
         match_iou_alpha, match_cls_gain,                # PMM
         match_iou_margin]                               # PMM-margin

    Every v2 argument defaults to the v1 behaviour, so an existing v1 yaml builds and trains
    exactly as before.

    Args:
        qual_beta (float): inference-time exponent on the quality score. ``0`` reproduces the
            stock scoring exactly, ``1`` ranks purely by predicted quality. Mutable after
            training -- set ``model.model[-1].qual_beta`` to re-run the sweep offline.
        qual_gain (float): weight of the dense quality loss.
        rsa (bool): enable the relation-aware self-attention bias.
        rsa_hidden (int): hidden width of the relation bias MLP.
        qual_tau (float): **M1** ownership-discount exponent. ``0`` = v1's plain
            ``max_j IoU`` target, which rewards duplicates because a near-copy of the winning
            box inherits nearly the winner's IoU. With ``tau > 0`` the target becomes
            ``max_j IoU_ij * (IoU_ij / max_k IoU_kj)**tau``: the winner is untouched and every
            duplicate is discounted by how far it trails the winner on that same object.
        qual_theta (float): **M2** IoU above which a query counts as "competitive". ``0``
            disables the re-weighting.
        qual_lambda (float): **M2** loss weight for non-competitive queries.
        rank_gain (float): **M3** weight of the top-k pairwise ranking loss. ``0`` disables it.
        rank_topk (int): **M3** how many queries per image enter the ranking loss.
        rank_delta (float): **M3** minimum target gap for a pair to be supervised. This also
            self-gates the loss early in training, when every box is still bad and no pair is
            separable -- no explicit warm-up is needed.
        nd (int): **M4** denoising budget. ``get_cdn_group`` builds
            ``num_dn = 2 * max_gt * (nd // max_gt)`` queries regardless of ``nq``, so lowering
            ``nq`` without lowering ``nd`` lets the denoising branch dominate the loss.
        match_iou_alpha (float): **PMM** exponent of Stable-DINO's position-modulated matching cost
            (Liu et al., ICCV'23). ``0`` keeps the stock Hungarian cost. The class probability entering
            the focal matching cost becomes ``p * ((GIoU + 1) / 2) ** match_iou_alpha``, so the positive
            label goes to the query that both scores high *and* localises well. Motivation: on this split
            the stock matcher picks the top-1 scorer 88-90% of the time and the best-IoU query only
            19-25%, with the best-IoU query at score rank ~15 -- the classification score is trained to
            rank whichever query already scores highest, not the best-localised one. Paper default 0.5.
            The head only stores the value; ``RTDETRDetectionModel.init_criterion`` hands it to the loss.
        match_cls_gain (float): **PMM** weight of the class term in the matching cost (stock 2.0).
            On this dataset the modulation alone is nearly inert: offline on the trained QADv2-PostBN3
            control, P(positive = best-IoU query) val / test goes 25.4 / 19.0% (stock) -> 26.2 / 19.8%
            (alpha 0.5, weight 2), because the score gap the stock matcher has entrenched between the
            matched query and its ~25 near-copies dwarfs a 3-5% modulation. Lowering the weight is what
            lets the box terms decide: weight 0.5 -> 49.2 / 40.9%, weight 0 (position only) ->
            79.0 / 73.4%. The PMM arm uses 0.5 so the box terms overturn the score only when the better
            box is >= ~0.05 IoU better; at 0 the positive would hop between near-identical copies.
            OUTCOME (run 2026-09-16, 300 ep): weight 0.5 COLLAPSED -- the positive hopped anyway, the class
            score flattened to a max of ~0.23/image with ~56 boxes above 0.25, mAP50 0.13 (control 0.89)
            while best-of-100 box IoU *improved* 0.83 -> 0.86. Keep this at 2.0.
        match_iou_margin (float): **PMM-margin** hysteresis re-assignment on top of the stock match: the
            positive moves to a free query only if its IoU beats the Hungarian pick's by at least this
            much (0.10 in the arm). Near-copies never clear the margin, so duplicates stay sticky; a clearly
            better box (31-38% of GTs have one >= 0.10 better) takes the label until its score overtakes.
    """

    def __init__(
            self,
            nc=80,
            ch=(512, 1024, 2048),
            hd=256,  # hidden dim
            nq=300,  # num queries
            ndp=4,  # num decoder points
            nh=8,  # num head
            ndl=6,  # num decoder layers
            qual_beta=0.5,
            qual_gain=2.0,
            rsa=True,
            rsa_hidden=16,
            qual_tau=0.0,  # M1: 0.0 == v1 (plain IoU target)
            qual_theta=0.0,  # M2: 0.0 == v1 (uniform weighting)
            qual_lambda=1.0,  # M2
            rank_gain=0.0,  # M3: 0.0 == v1 (no ranking loss)
            rank_topk=8,  # M3
            rank_delta=0.05,  # M3
            nd=100,  # M4: num denoising
            match_iou_alpha=0.0,  # PMM: 0.0 == stock Hungarian cost
            match_cls_gain=2.0,  # PMM: class-term weight in the matching cost
            match_iou_margin=0.0,  # PMM-margin: 0.0 == no re-assignment
            d_ffn=1024,  # dim of feedforward
            eval_idx=-1,
            dropout=0.,
            act=nn.ReLU(),
            # Training args
            label_noise_ratio=0.5,
            box_noise_scale=1.0,
            learnt_init_query=False):
        super().__init__(nc=nc, ch=ch, hd=hd, nq=nq, ndp=ndp, nh=nh, ndl=ndl, d_ffn=d_ffn, eval_idx=eval_idx,
                         dropout=dropout, act=act, nd=nd, label_noise_ratio=label_noise_ratio,
                         box_noise_scale=box_noise_scale, learnt_init_query=learnt_init_query)
        self.qual_beta = qual_beta
        self.qual_gain = qual_gain
        self.use_rsa = rsa
        self.qual_tau = qual_tau
        self.qual_theta = qual_theta
        self.qual_lambda = qual_lambda
        self.rank_gain = rank_gain
        self.rank_topk = rank_topk
        self.rank_delta = rank_delta
        self.match_iou_alpha = match_iou_alpha
        self.match_cls_gain = match_cls_gain
        self.match_iou_margin = match_iou_margin

        # Rebuild the decoder stack; `_reset_parameters` does not touch it, and both layer
        # types self-initialise their deformable attention, so no re-init is needed here.
        if rsa:
            decoder_layer = DeformableTransformerDecoderLayerRSA(hd, nh, d_ffn, dropout, act, self.nl, ndp, rsa_hidden)
        else:
            decoder_layer = DeformableTransformerDecoderLayer(hd, nh, d_ffn, dropout, act, self.nl, ndp)
        self.decoder = DeformableTransformerDecoderQAD(hd, decoder_layer, ndl, eval_idx)

        # Dense quality head, shared across decoder layers.
        self.dec_qual_head = MLP(hd, hd, 1, num_layers=2)

    def forward(self, x, batch=None):
        """Forward pass returning quality-fused scores at inference and a 6-tuple in training."""
        from ultralytics.models.utils.ops import get_cdn_group

        # Input projection and embedding
        feats, shapes = self._get_encoder_input(x)

        # Prepare denoising training
        dn_embed, dn_bbox, attn_mask, dn_meta = \
            get_cdn_group(batch,
                          self.nc,
                          self.num_queries,
                          self.denoising_class_embed.weight,
                          self.num_denoising,
                          self.label_noise_ratio,
                          self.box_noise_scale,
                          self.training)

        embed, refer_bbox, enc_bboxes, enc_scores = \
            self._get_decoder_input(feats, shapes, dn_embed, dn_bbox)

        # Decoder
        dec_bboxes, dec_scores, dec_qual = self.decoder(embed,
                                                        refer_bbox,
                                                        feats,
                                                        shapes,
                                                        self.dec_bbox_head,
                                                        self.dec_score_head,
                                                        self.dec_qual_head,
                                                        self.query_pos_head,
                                                        attn_mask=attn_mask)
        x = dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta, dec_qual
        if self.training:
            return x

        scores = dec_scores.squeeze(0).sigmoid()
        if self.qual_beta > 0:
            qual = dec_qual.squeeze(0).sigmoid().clamp(min=1e-6)  # (bs, nq, 1), broadcast over classes
            scores = scores.clamp(min=1e-6).pow(1 - self.qual_beta) * qual.pow(self.qual_beta)
        # (bs, 300, 4+nc)
        y = torch.cat((dec_bboxes.squeeze(0), scores), -1)
        return y if self.export else (y, x)
