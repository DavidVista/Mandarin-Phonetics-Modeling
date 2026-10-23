"""Immutable, train-only-vocabulary BiLSTM character-to-pinyin labeller."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import random
import shutil
import tempfile
import unicodedata
from collections import Counter,defaultdict
from pathlib import Path

from ..artifacts import safe_artifact,sha256_file,write_json,write_jsonl
from ..data_contract import is_han
from ..pinyin import canonical_labels,decompose
from ..reproducibility import seed_everything

LABELLER_VERSION="1"
REQUIRED={"weights.pt","vocab.json","config.json","splits.json","metrics.json","manifest.json"}


def _model(config,vocab):
    import torch
    class Labeller(torch.nn.Module):
        def __init__(self):
            super().__init__(); a=config["architecture"]
            self.embedding=torch.nn.Embedding(len(vocab["characters"]),a["embedding_dim"],padding_idx=1)
            self.lstm=torch.nn.LSTM(a["embedding_dim"],a["hidden_dim"],a["layers"],batch_first=True,
                                    bidirectional=True,dropout=a["dropout"] if a["layers"]>1 else 0.0)
            self.output=torch.nn.Linear(2*a["hidden_dim"],len(vocab["pinyin"]))
        def forward(self,ids,lengths):
            packed=torch.nn.utils.rnn.pack_padded_sequence(self.embedding(ids),lengths.cpu(),batch_first=True,enforce_sorted=False)
            values,_=self.lstm(packed); padded,_=torch.nn.utils.rnn.pad_packed_sequence(values,batch_first=True,total_length=ids.shape[1])
            logits=self.output(padded); logits[...,0]=torch.finfo(logits.dtype).min; logits[...,1]=torch.finfo(logits.dtype).min
            return logits
    return Labeller()


def train_labeller(training_csv,output_dir,*,bundle_id,source_revision,source_sha256,
                   inventory_path,inventory_sha256,config,columns=None):
    import torch
    training_csv=Path(training_csv).resolve(); inventory_path=Path(inventory_path).resolve()
    output=Path(output_dir).resolve(); output.mkdir(parents=True,exist_ok=True); target=(output/bundle_id).resolve()
    if not bundle_id or target.parent!=output or target.exists(): raise FileExistsError(target)
    if not isinstance(source_revision,str) or not source_revision: raise ValueError("source_revision is required")
    if sha256_file(training_csv)!=source_sha256 or sha256_file(inventory_path)!=inventory_sha256: raise ValueError("source checksum mismatch")
    _validate_config(config); tokens=_inventory(inventory_path,inventory_sha256)
    rows,rejections=_training_rows(training_csv,config["tone_convention"],set(tokens),columns)
    groups={}
    provenance=defaultdict(list)
    for row in rows:
        if row["text"] in groups and groups[row["text"]]["labels"]!=row["labels"]: raise ValueError("conflicting labels for duplicate training text")
        groups.setdefault(row["text"],row); provenance[row["text"]].append(row["source_id"])
    unique=sorted(groups.values(),key=lambda r:_stable(r["text"],r["labels"])); rng=random.Random(config["seed"]); rng.shuffle(unique)
    n=len(unique); ratios=config["split_ratios"]; a=int(n*ratios[0]); b=a+int(n*ratios[1])
    parts={"train":unique[:a],"validation":unique[a:b],"test":unique[b:]}
    if any(not value for value in parts.values()): raise ValueError("all labeller splits must be non-empty")
    char_vocab=["[UNK]","[PAD]",*sorted({ch for row in parts["train"] for ch in row["text"]})]
    vocab={"characters":char_vocab,"pinyin":["[UNK]","[PAD]",*tokens]}; char_stoi={x:i for i,x in enumerate(char_vocab)}; py_stoi={x:i for i,x in enumerate(vocab["pinyin"])}
    polyphonic_characters=_train_polyphonic_characters(parts["train"])
    seed_everything(config["seed"]); device=torch.device(config["device"]); model=_model(config,vocab).to(device).float()
    optimizer=torch.optim.AdamW(model.parameters(),lr=config["training"]["learning_rate"]); best=None; best_loss=math.inf; history=[]
    for epoch in range(config["training"]["epochs"]):
        model.train(); shuffled=list(parts["train"]); random.Random(config["seed"]+epoch).shuffle(shuffled)
        for batch in _batches(shuffled,config["training"]["batch_size"],char_stoi,py_stoi,device,reject_oov=True):
            optimizer.zero_grad(); logits=model(batch[0],batch[2]); loss=torch.nn.functional.cross_entropy(logits.reshape(-1,len(vocab["pinyin"])),batch[1].reshape(-1),ignore_index=1); loss.backward(); optimizer.step()
        val=_evaluate(model,parts["validation"],char_stoi,py_stoi,device,config["training"]["batch_size"],polyphonic_characters); history.append({"epoch":epoch+1,**val})
        if val["loss"]<best_loss: best_loss=val["loss"]; best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(best); metrics={"synthetic_or_source_quality_only":True,"history":history,
        "group_definition":{"polyphonic_characters_train":polyphonic_characters,
                            "note":"neutral, polyphonic and OOV memberships are evaluated independently and may overlap"},
        "validation":_evaluate(model,parts["validation"],char_stoi,py_stoi,device,config["training"]["batch_size"],polyphonic_characters),
        "test":_evaluate(model,parts["test"],char_stoi,py_stoi,device,config["training"]["batch_size"],polyphonic_characters),"rejections":dict(Counter(x["reason"] for x in rejections))}
    temp=Path(tempfile.mkdtemp(prefix=f".{bundle_id}.",dir=output))
    try:
        write_json(temp/"vocab.json",vocab); write_json(temp/"config.json",config)
        split_ids={key:[{"sample_id":_stable(r["text"],r["labels"]),"source_ids":sorted(provenance[r["text"]])} for r in value] for key,value in parts.items()}; write_json(temp/"splits.json",split_ids); write_json(temp/"metrics.json",metrics)
        source={"path":str(training_csv),"revision":source_revision,"sha256":source_sha256,"columns":columns or {"text":"text","pinyin":"pinyin","source_id":"source_id"},
                "inventory":{"path":str(inventory_path),"sha256":inventory_sha256}}
        torch.save({"state_dict":best,"config_sha256":sha256_file(temp/"config.json"),"vocab_sha256":sha256_file(temp/"vocab.json"),"source_sha256":source_sha256},temp/"weights.pt")
        artifacts={p.name:sha256_file(p) for p in temp.iterdir() if p.is_file()}
        write_json(temp/"manifest.json",{"bundle_type":"pinyin_labeller","schema_version":1,"version":LABELLER_VERSION,"bundle_id":bundle_id,"source":source,"artifacts":artifacts,
            "code":{"labeller.py":sha256_file(Path(__file__))},"environment":{"python":platform.python_version(),"torch":torch.__version__,"numpy":__import__("numpy").__version__}})
        _load_bundle(temp); os.rename(temp,target); return target
    except Exception: shutil.rmtree(temp,ignore_errors=True); raise


def load_labeller(path,*,device="cpu"):
    bundle=_load_bundle(Path(path).resolve()); bundle["device"]=device; bundle["model"].to(device); return bundle


def _load_bundle(root):
    import torch
    manifest=json.loads((root/"manifest.json").read_text(encoding="utf-8"))
    if manifest.get("bundle_type")!="pinyin_labeller" or manifest.get("schema_version")!=1: raise ValueError("unsupported labeller bundle")
    if set(manifest.get("artifacts",{}))!=REQUIRED-{"manifest.json"}: raise ValueError("labeller artifacts incomplete")
    for relative,digest in manifest["artifacts"].items():
        path=safe_artifact(root,relative)
        if not path.is_file() or sha256_file(path)!=digest: raise ValueError("labeller artifact checksum mismatch")
    config=json.loads((root/"config.json").read_text(encoding="utf-8")); _validate_config(config); vocab=json.loads((root/"vocab.json").read_text(encoding="utf-8"))
    if (not isinstance(vocab.get("characters"),list) or not isinstance(vocab.get("pinyin"),list)
            or vocab["characters"][:2]!=["[UNK]","[PAD]"] or vocab["pinyin"][:2]!=["[UNK]","[PAD]"]
            or len(vocab["characters"])!=len(set(vocab["characters"])) or len(vocab["pinyin"])!=len(set(vocab["pinyin"]))): raise ValueError("invalid labeller vocab")
    splits=json.loads((root/"splits.json").read_text(encoding="utf-8")); ids=[row["sample_id"] for part in splits.values() for row in part]
    if set(splits)!={"train","validation","test"} or any(not x for x in splits.values()) or len(ids)!=len(set(ids)): raise ValueError("invalid labeller splits")
    checkpoint=torch.load(root/"weights.pt",map_location="cpu",weights_only=True)
    if checkpoint["config_sha256"]!=sha256_file(root/"config.json") or checkpoint["vocab_sha256"]!=sha256_file(root/"vocab.json") or checkpoint["source_sha256"]!=manifest["source"]["sha256"]: raise ValueError("checkpoint provenance mismatch")
    model=_model(config,vocab); model.load_state_dict(checkpoint["state_dict"],strict=True); model.eval()
    return {"root":root,"manifest":manifest,"config":config,"vocab":vocab,"splits":splits,"model":model,"device":"cpu"}


def predict_pinyin(labeller,texts):
    import torch
    single=isinstance(texts,str); values=[texts] if single else list(texts); stoi={x:i for i,x in enumerate(labeller["vocab"]["characters"])}
    if any(not isinstance(text,str) or not text for text in values): raise ValueError("prediction text must be non-empty")
    missing=[sorted({ch for ch in text if ch not in stoi}) for text in values]
    if any(missing): raise ValueError(f"prediction contains OOV characters: {missing}")
    lengths=torch.tensor([len(x) for x in values]); width=max(lengths.tolist()); ids=torch.full((len(values),width),1,dtype=torch.long)
    for i,text in enumerate(values): ids[i,:len(text)]=torch.tensor([stoi[ch] for ch in text])
    device=labeller.get("device","cpu"); model=labeller["model"].to(device); model.eval()
    with torch.inference_mode(): prediction=model(ids.to(device),lengths.to(device)).argmax(-1).cpu()
    labels=labeller["vocab"]["pinyin"]; result=[]
    for row,length in zip(prediction,lengths.tolist()):
        decoded=[labels[index] for index in row[:length].tolist()]
        if any(x in {"[UNK]","[PAD]"} for x in decoded): raise ValueError("model predicted a special token")
        result.append(decoded)
    return result[0] if single else result


def annotate_corpus(records,labeller,output_root,*,dataset_id,cleaning_metrics=None,cleaning_rejections=None):
    import torch
    parent=Path(output_root).resolve(); parent.mkdir(parents=True,exist_ok=True); root=(parent/dataset_id).resolve()
    if root.parent!=parent or root.exists(): raise FileExistsError(root)
    temp=Path(tempfile.mkdtemp(prefix=f".{dataset_id}.",dir=parent)); accepted=[]; rejected=[]; stoi={x:i for i,x in enumerate(labeller["vocab"]["characters"])}
    try:
        for row in records:
            if not row.get("text"): rejected.append({**row,"reason":"empty"}); continue
            missing=sorted({ch for ch in row["text"] if ch not in stoi})
            if missing: rejected.append({**row,"reason":"oov_character","missing_characters":missing}); continue
            try: prediction=predict_pinyin(labeller,row["text"])
            except ValueError as exc: rejected.append({**row,"reason":"special_prediction","message":str(exc)}); continue
            accepted.append({**row,"labels":prediction})
        with (temp/"labelled.csv").open("w",encoding="utf-8",newline="") as h:
            writer=csv.DictWriter(h,fieldnames=["text","pinyin","source_id"]); writer.writeheader(); writer.writerows({"text":r["text"],"pinyin":" ".join(r["labels"]),"source_id":r["source_id"]} for r in accepted)
        accepted_by_id={r["source_id"]:r for r in accepted}
        source_rows=[]
        for row in records:
            labelled=accepted_by_id.get(row["source_id"]); source_rows.append({**row,"status":"accepted" if labelled else "rejected",
                "labels":labelled.get("labels") if labelled else None})
        write_jsonl(temp/"source_rows.jsonl",source_rows)
        write_json(temp/"labeling_metrics.json",{"input":len(records),"accepted":len(accepted),"rejected":len(rejected),
            "reasons":dict(Counter(x["reason"] for x in rejected)),"cleaning":cleaning_metrics})
        write_jsonl(temp/"labeling_rejections.jsonl",rejected)
        write_jsonl(temp/"cleaning_rejections.jsonl",cleaning_rejections or [])
        lots={r["lot_id"]:{"sha256":r.get("source_sha256"),"revision":r.get("source_revision")} for r in records if r.get("lot_id")}
        write_json(temp/"labeling_provenance.json",{"labeller":{"bundle_id":labeller["manifest"]["bundle_id"],
            "manifest_sha256":sha256_file(labeller["root"]/"manifest.json"),"config_sha256":sha256_file(labeller["root"]/"config.json"),
            "vocab_sha256":sha256_file(labeller["root"]/"vocab.json")},"lots":lots})
        os.rename(temp,root); return root
    except Exception as exc:
        write_json(parent/f"{dataset_id}.failed.json",{"status":"failure","exception":type(exc).__name__,"message":str(exc)}); shutil.rmtree(temp,ignore_errors=True); raise


def _training_rows(path,tone_convention,inventory,columns=None):
    rows=[]; rejected=[]
    with path.open(encoding="utf-8-sig",newline="") as h:
        reader=csv.DictReader(h,strict=True); columns=columns or {"text":"text","pinyin":"pinyin","source_id":"source_id"}
        if not reader.fieldnames or not {columns["text"],columns["pinyin"]}.issubset(reader.fieldnames): raise ValueError("training CSV columns disagree with explicit mapping")
        for number,row in enumerate(reader,2):
            text=row.get(columns["text"]); raw=row.get(columns["pinyin"],"").split(); source=row.get(columns.get("source_id","source_id")) or f"row-{number}"; reason=None
            if not isinstance(text,str) or not text or len(text)!=len(raw): reason="alignment"
            elif any(ch.isdigit() for ch in text): reason="digit"
            else:
                kept=[(ch,label) for ch,label in zip(text,raw) if not ch.isspace() and unicodedata.category(ch)[0]!="P"]
                text="".join(x[0] for x in kept); raw=[x[1] for x in kept]
                if not text or any(not is_han(ch) for ch in text): reason="characters"
                else:
                    try: labels=canonical_labels(raw,tone_convention=tone_convention); [decompose(x) for x in labels]
                    except ValueError: reason="pinyin"
                    if reason is None and any(x not in inventory for x in labels): reason="pinyin_inventory"
            if reason: rejected.append({"source_id":source,"source_row":number,"reason":reason})
            else: rows.append({"text":text,"labels":labels,"source_id":source,"source_row":number})
    return rows,rejected


def _inventory(path,digest):
    if sha256_file(path)!=digest: raise ValueError("inventory checksum mismatch")
    tokens=[x.strip() for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    if not tokens: raise ValueError("pinyin inventory is empty")
    if len(tokens)!=len(set(tokens)): raise ValueError("duplicate pinyin inventory token")
    canonical=[]
    for token in tokens:
        normalized=canonical_labels([token],tone_convention="explicit")[0]
        if normalized!=token: raise ValueError("inventory token is not canonical explicit-tone pinyin")
        decompose(token); canonical.append(normalized)
    if len(canonical)!=len(set(canonical)): raise ValueError("duplicate canonical pinyin inventory token")
    return tokens


def _validate_config(config):
    required={"architecture","training","seed","split_ratios","device","tone_convention"}
    if set(config)!=required or not isinstance(config["seed"],int) or isinstance(config["seed"],bool): raise ValueError("invalid labeller config")
    a=config["architecture"]; t=config["training"]
    if any(not isinstance(a.get(k),int) or isinstance(a.get(k),bool) or a[k]<=0 for k in ("embedding_dim","hidden_dim","layers")): raise ValueError("invalid architecture")
    if not isinstance(a.get("dropout"),(int,float)) or isinstance(a.get("dropout"),bool) or not math.isfinite(a["dropout"]) or not 0<=a["dropout"]<1: raise ValueError("invalid dropout")
    if any(not isinstance(t.get(k),int) or isinstance(t.get(k),bool) or t[k]<=0 for k in ("epochs","batch_size")) or not isinstance(t.get("learning_rate"),(int,float)) or isinstance(t.get("learning_rate"),bool) or not math.isfinite(t["learning_rate"]) or t["learning_rate"]<=0: raise ValueError("invalid training config")
    ratios=config["split_ratios"]
    if len(ratios)!=3 or any(not isinstance(x,(int,float)) or isinstance(x,bool) or not math.isfinite(x) or x<=0 for x in ratios) or not math.isclose(sum(ratios),1): raise ValueError("invalid split ratios")
    if config["tone_convention"] not in {"explicit","neutral_unmarked"} or not config["device"]: raise ValueError("invalid device/tone convention")


def _batches(rows,size,char_stoi,py_stoi,device,reject_oov):
    import torch
    for start in range(0,len(rows),size):
        group=rows[start:start+size]; usable=[r for r in group if not reject_oov or all(ch in char_stoi for ch in r["text"])]
        if not usable: continue
        lengths=torch.tensor([len(r["text"]) for r in usable]); width=int(lengths.max()); ids=torch.full((len(usable),width),1,dtype=torch.long); targets=torch.full_like(ids,1)
        for i,row in enumerate(usable):
            ids[i,:len(row["text"])]=torch.tensor([char_stoi.get(ch,0) for ch in row["text"]]); targets[i,:len(row["labels"])]=torch.tensor([py_stoi.get(x,0) for x in row["labels"]])
        yield ids.to(device),targets.to(device),lengths.to(device)


def _evaluate(model,rows,char_stoi,py_stoi,device,batch_size,polyphonic_characters=()):
    import torch
    model.eval(); loss_sum=correct=sentences=sentence_correct=tokens=oov=0
    neutral_ids={index for label,index in py_stoi.items() if label.endswith("5")}
    polyphonic_ids={char_stoi[char] for char in polyphonic_characters if char in char_stoi}
    grouped={name:{"token_count":0,"correct_token_count":0} for name in ("neutral","polyphonic","oov")}
    with torch.inference_mode():
        for ids,target,lengths in _batches(rows,batch_size,char_stoi,py_stoi,device,reject_oov=False):
            logits=model(ids,lengths); flat=target!=1; losses=torch.nn.functional.cross_entropy(logits.reshape(-1,logits.shape[-1]),target.reshape(-1),ignore_index=1,reduction="sum")
            pred=logits.argmax(-1); correct_mask=(pred==target)&flat; loss_sum+=losses.item(); tokens+=int(flat.sum()); correct+=int(correct_mask.sum()); oov+=int(((ids==0)&flat).sum())
            masks={"neutral":flat & torch.isin(target,torch.tensor(sorted(neutral_ids),device=target.device,dtype=target.dtype)),
                   "polyphonic":flat & torch.isin(ids,torch.tensor(sorted(polyphonic_ids),device=ids.device,dtype=ids.dtype)),
                   "oov":flat & (ids==0)}
            for name,mask in masks.items(): grouped[name]["token_count"]+=int(mask.sum()); grouped[name]["correct_token_count"]+=int((correct_mask&mask).sum())
            for i,length in enumerate(lengths.tolist()): sentences+=1; sentence_correct+=int(torch.equal(pred[i,:length],target[i,:length]))
    for values in grouped.values(): values["token_accuracy"]=(values["correct_token_count"]/values["token_count"] if values["token_count"] else None)
    return {"loss":loss_sum/max(tokens,1),"token_accuracy":correct/max(tokens,1),"sentence_accuracy":sentence_correct/max(sentences,1),"tokens":tokens,"sentences":sentences,"oov_tokens":oov,"groups":grouped}


def _train_polyphonic_characters(rows):
    readings=defaultdict(set)
    for row in rows:
        for char,label in zip(row["text"],row["labels"]): readings[char].add(label)
    return sorted(char for char,labels in readings.items() if len(labels)>1)


def _stable(text,labels): return hashlib.sha256(json.dumps([text,labels],ensure_ascii=False,separators=(",",":")).encode()).hexdigest()
