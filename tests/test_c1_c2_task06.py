import torch

from models.context_affective_metric import (
    AffectiveMetricHead, MultiSourceAffectiveMetric, bound_symmetric, low_rank_metric_scores, random_basis,
)
from utils.pairwise_consistency import (
    apply_best_single_action, apply_multi_pair_actions, pairwise_residuals, utility_labels,
)
from utils.seediv_c1_c2_protocol import canonical_hash, oof_plan, resolve_c1_config, source_roles


def test_low_rank_matches_dense():
    torch.manual_seed(7)
    n, c, d, r = 5, 4, 11, 2
    z, p = torch.randn(n, d), torch.randn(c, d)
    z.requires_grad_()
    b = random_basis(d, r, 9)
    h = bound_symmetric(torch.randn(n, r, r))
    low, _, diag = low_rank_metric_scores(z, p, b, h, tau=1.0)
    dense = []
    for i in range(n):
        m = torch.eye(d) + b @ h[i] @ b.T
        num = z[i] @ m @ p.T
        den = torch.sqrt((z[i] @ m @ z[i]) * ((p @ m) * p).sum(-1))
        dense.append(num / den)
    assert torch.allclose(low, torch.stack(dense), atol=2e-5, rtol=2e-5)
    assert int(diag["denominator_clamp_count"]) == 0
    low.sum().backward()
    assert torch.isfinite(z.grad).all()


def test_metric_bounds_and_controls():
    torch.manual_seed(4)
    p, z, q = torch.randn(4, 12), torch.randn(6, 12), torch.randn(6, 8)
    b = random_basis(12, 2, 3)
    for variant in ("E0", "E1", "E2", "E3", "E4", "E5"):
        head = AffectiveMetricHead(variant, 8, p, b, tau=.07)
        out = head(q, z)
        assert out.logits.shape == (6, 4) and torch.isfinite(out.logits).all()
        if variant in ("E1", "E2", "E5"):
            assert out.diagnostics["metric_eigenvalues"].min() > .5
            assert out.diagnostics["metric_eigenvalues"].max() < 1.5
            assert torch.allclose(out.relation, out.relation.transpose(-1,-2), atol=1e-6)
            assert torch.allclose(out.relation.diagonal(dim1=-2,dim2=-1), torch.ones(6,4), atol=1e-5)
            assert torch.linalg.eigvalsh(out.relation).min() > -1e-5
    e1 = AffectiveMetricHead("E1", 8, p, b)
    assert torch.allclose(e1(q, z).logits, e1(q * 11, z).logits)
    base = AffectiveMetricHead("E0", 8, p, b)(q, z).logits
    assert torch.allclose(AffectiveMetricHead("E3", 8, p, b)(q, z).logits, base)
    assert torch.allclose(AffectiveMetricHead("E4", 8, p, b)(q, z).logits, base, atol=2e-6)
    dynamic=AffectiveMetricHead("E2",8,p,b)
    torch.nn.init.normal_(dynamic.metric_generator.net[-1].weight, std=.1)
    assert not torch.allclose(dynamic(q,z).logits,dynamic(q.flip(0),z).logits)
    clone=AffectiveMetricHead("E2",8,p,b); clone.load_state_dict(dynamic.state_dict())
    assert torch.equal(dynamic(q,z).logits,clone(q,z).logits)


def test_pairwise_utility_and_null_actions():
    ref = torch.tensor([[2., 1., 0., -1.], [0., 1., 2., 3.]])
    expert = torch.stack((ref + torch.tensor([1., -1., 0., 0.]), ref), dim=1)
    labels = torch.tensor([0, 3])
    residual = pairwise_residuals(ref, expert)
    utility = utility_labels(ref, expert, labels)
    assert residual.shape == utility.shape == (2, 2, 6)
    no_op, action = apply_best_single_action(ref, residual, torch.full_like(utility, -1.0))
    assert torch.allclose(no_op, ref - ref.mean(-1, keepdim=True)) and not action.any()
    multi, gates, _ = apply_multi_pair_actions(ref, residual, torch.full_like(utility, -1.0))
    assert torch.allclose(multi, ref - ref.mean(-1, keepdim=True)) and not gates.any()
    shifted=ref+17.0
    assert torch.allclose(pairwise_residuals(shifted,expert+17.0),residual)
    assert torch.allclose(utility_labels(shifted,expert+17.0,labels),utility)


def test_subject_isolation():
    roles = source_roles(1)
    assert [len(roles[x]) for x in ("T", "V", "U")] == [8, 3, 3]
    assert not (set(roles["T"]) & set(roles["V"]) | set(roles["T"]) & set(roles["U"]))
    for row in oof_plan(roles["T"]):
        assert row["held_out"] not in row["teacher_train"]


def test_branch_first_fusion_and_effective_config_hash():
    torch.manual_seed(11)
    prototypes=torch.randn(4,10); basis=random_basis(10,2,5)
    module=MultiSourceAffectiveMetric(2,variant="E0",context_dim=6,prototypes=prototypes,basis=basis)
    context=torch.randn(3,6); embeddings=[torch.randn(3,10),torch.randn(3,10)]
    weights=torch.tensor([[.8,.2,.5],[.2,.8,.5]])
    output=module.fused_output(context,embeddings,weights,"branch_logits")
    branches=module.source_outputs([context,context],embeddings)
    expected=(weights[...,None]*torch.stack([x.logits for x in branches])).sum(0)
    assert torch.allclose(output.logits,expected)
    shared=MultiSourceAffectiveMetric(2,head_sharing="shared",variant="E2",context_dim=6,prototypes=prototypes,basis=basis)
    shared_output=shared.fused_output(context,embeddings,weights,"shared_fused_embedding")
    assert shared_output.logits.shape==(3,4) and len(shared.heads)==1
    base={"epochs":200,"batch_size":64,"learning_rate":1e-3}
    first=resolve_c1_config(base,{"learning_rate":2e-3})
    second=resolve_c1_config(base,{"learning_rate":3e-3})
    assert first["learning_rate"]==2e-3 and canonical_hash(first)!=canonical_hash(second)
