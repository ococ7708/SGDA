"""C2 utility-router trainer for subject-level OOF evidence packages.

Input NPZ must contain reference_logits[N,C], expert_logits[N,K,C], labels[N],
prototype_pair_features[E,P], source_distances[N,K], and subject_ids[N].  The
producer must additionally provide provenance JSON proving that each row's
teacher excluded that row's subject; this program validates that invariant.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from models.pairwise_utility_router import PairwiseUtilityRouter, build_router_features
from utils.pairwise_consistency import utility_labels


def load_package(npz_path, provenance_path):
    data=np.load(npz_path); required={"reference_logits","expert_logits","labels","prototype_pair_features","source_distances","subject_ids"}
    missing=required-set(data.files)
    if missing: raise ValueError(f"OOF package missing {sorted(missing)}")
    prov=json.loads(Path(provenance_path).read_text(encoding="utf-8"))
    held={int(row["held_out"]):set(map(int,row["teacher_train"])) for row in prov["folds"]}
    for sid in np.unique(data["subject_ids"]):
        if int(sid) not in held or int(sid) in held[int(sid)]: raise ValueError(f"invalid OOF provenance for subject {sid}")
    return data,prov


def train(a):
    data,prov=load_package(a.input,a.provenance); device=torch.device(a.device if torch.cuda.is_available() else "cpu")
    tensors={k:torch.as_tensor(data[k],device=device) for k in data.files if k!="subject_ids"}
    ref=tensors["reference_logits"].float(); exp=tensors["expert_logits"].float(); labels=tensors["labels"].long()
    features=build_router_features(ref,exp,tensors["prototype_pair_features"].float(),tensors["source_distances"].float())
    targets=utility_labels(ref,exp,labels,alpha=a.alpha,clip_limit=a.residual_clip)
    model=PairwiseUtilityRouter(features.size(-1)).to(device); model.standardizer.fit(features)
    optimizer=torch.optim.Adam(model.parameters(),lr=a.lr); history=[]
    for epoch in range(1,a.epochs+1):
        optimizer.zero_grad(); pred=model(features); loss=model.loss(pred,targets); loss.backward(); optimizer.step()
        history.append({"epoch":epoch,"huber":float(loss.detach().cpu())})
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
    torch.save({"model":model.state_dict(),"optimizer":optimizer.state_dict(),"feature_dim":features.size(-1),"history":history,
        "input":str(a.input),"provenance":prov,"config":vars(a)},out)
    print(json.dumps({"status":"PASS","checkpoint":str(out),"samples":len(ref),"feature_shape":list(features.shape),"final_huber":history[-1]["huber"]},indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--input",required=True); p.add_argument("--provenance",required=True); p.add_argument("--output",required=True)
    p.add_argument("--epochs",type=int,default=100); p.add_argument("--lr",type=float,default=1e-3); p.add_argument("--alpha",type=float,default=.25)
    p.add_argument("--residual-clip",type=float,default=2.0); p.add_argument("--device",default="cuda:0"); train(p.parse_args())
