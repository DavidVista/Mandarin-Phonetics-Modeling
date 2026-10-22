import csv
import json
import pickle
import sys
import tempfile
import unittest
import wave
import types
from unittest import mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from phonetics.artifacts import sha256_file
from phonetics.preparation import prepare_dataset, validate_prepared
from phonetics.preparation.config import validate_config
from phonetics.preparation.embeddings import TransformersBackend, cache_key
from phonetics.preparation.readers import audio_info


class Stub:
    def __call__(self, entry, config):
        import numpy as np
        return np.array([len(entry["label"]), 1.0], dtype=np.float32)


def wav(path):
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1); handle.setsampwidth(2); handle.setframerate(8000); handle.writeframes(b"\0\0" * 8)


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=Path(__file__).parent); self.root=Path(self.temp.name)
        sounds=self.root/"sounds"; sounds.mkdir()
        for label in ("ni3","hao3","ma5","wo3","ai4","zhong1","guo2"): wav(sounds/f"{label}.wav")
        self.wubi={"你":"wq","好":"vb","吗":"kc","我":"tr","爱":"ep","中":"kh","国":"lg","坏":"fg"}
        with (self.root/"wubi.pkl").open("wb") as h: pickle.dump(self.wubi,h)
        self.corpus=self.root/"corpus.csv"
        self.config={"dataset_id":"fixture","output_root":str(self.root/"out"),"cache_root":str(self.root/"cache"),
          "corpus":{"path":str(self.corpus),"source_id":"fixture","tone_convention":"explicit"},
          "sounds":{"path":str(sounds),"tone_convention":"explicit"},"wubi":{"path":str(self.root/"wubi.pkl")},
          "model":{"id":"unused","revision":"exact","layer":-1,"pooling":"mean_all","sample_rate":8000,"mono":True,"padding":"none","dtype":"float32","device":"cpu"},
          "sampling":{"size":None,"strategy":"uniform","seed":7}}
    def tearDown(self): self.temp.cleanup()
    def write_rows(self, rows):
        with self.corpus.open("w",encoding="utf-8",newline="") as h:
            w=csv.writer(h); w.writerow(["text","pinyin","source_id"]); w.writerows(rows)
    def good_bundle(self, dataset_id, size=None):
        self.write_rows([["你好","ni3 hao3","a"],["我爱","wo3 ai4","b"]])
        config={**self.config,"dataset_id":dataset_id,"sampling":{"size":size,"strategy":"uniform","seed":7}}
        return prepare_dataset(config,embedding_backend=Stub())
    def tamper_stats(self, path, mutate):
        stats_path=path/"reports/run_statistics.json"; stats=json.loads(stats_path.read_text(encoding="utf-8")); mutate(stats)
        stats_path.write_text(json.dumps(stats,ensure_ascii=False,sort_keys=True,indent=2)+"\n",encoding="utf-8")
        manifest_path=path/"manifest.json"; manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["artifacts"]["reports/run_statistics.json"]=sha256_file(stats_path)
        manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,sort_keys=True,indent=2)+"\n",encoding="utf-8")

    def test_exact_filter_losses_and_inventory_independent_of_sampling(self):
        self.write_rows([["","","a"],["你!","ni3 hao3","b"],["你好","ni3","c"],["你","ni","d"],
                         ["缺","que1","e"],["你","hei1","f"],["你好","ni3 hao3","g"],["我爱中国","wo3 ai4 zhong1 guo2","h"]])
        result=prepare_dataset(self.config,embedding_backend=Stub()); manifest=validate_prepared(result)
        stats=json.loads((result/"reports/run_statistics.json").read_text(encoding="utf-8"))
        filters=[s for s in stats["stages"] if s["name"] in ("nonempty","characters","alignment","pinyin","wubi","acoustic")]
        self.assertEqual([s["rejected"] for s in filters],[1,1,1,1,1,1])
        self.assertEqual(stats["outputs"],{"csv_rows":2,"embeddings":7,"wubi_entries":8})
        small={**self.config,"dataset_id":"small","sampling":{"size":1,"strategy":"uniform","seed":7}}
        small_path=prepare_dataset(small,embedding_backend=Stub())
        with (result/"embeddings/pinyin_embeddings.pkl").open("rb") as a,(small_path/"embeddings/pinyin_embeddings.pkl").open("rb") as b:
            self.assertEqual(set(pickle.load(a)),set(pickle.load(b)))
        self.assertEqual(manifest["data_contract"],"prepared-1")

    def test_conflict_bounds_overwrite_corruption_and_failure_report(self):
        self.write_rows([["你好","ni3 hao3","a"],["你好","ni3 ma5","b"],["我爱","wo3 ai4","c"]])
        expected=self.root/"conflict-config.json"; expected.write_text(json.dumps({"schema_version":1,**self.config},ensure_ascii=False,sort_keys=True,indent=2)+"\n",encoding="utf-8")
        with self.assertRaisesRegex(ValueError,"conflicting labels"): prepare_dataset(self.config,embedding_backend=Stub())
        failure=json.loads((self.root/"out/fixture.failed.json").read_text(encoding="utf-8")); dedup=next(s for s in failure["stages"] if s["name"]=="deduplicate")
        self.assertEqual((dedup["input_count"],dedup["output_count"],dedup["rejected"],dedup["blocked_count"]),(3,0,2,1))
        self.assertEqual((dedup["conflict_groups"],dedup["conflict_rows"]),(1,2)); self.assertTrue(dedup["counts_complete"])
        self.assertEqual(failure["config_sha256"],sha256_file(expected))
        self.assertTrue(all(s["status"]=="not-run" for s in failure["stages"] if s["name"] in ("sampling","embeddings","final_validation")))
        self.assertFalse((self.root/"out/fixture").exists())
        self.write_rows([["你好","ni3 hao3","a"],["我爱","wo3 ai4","b"]]); self.config["dataset_id"]="ok"
        path=prepare_dataset(self.config,embedding_backend=Stub())
        with self.assertRaises(FileExistsError): prepare_dataset(self.config,embedding_backend=Stub())
        (path/"corpus/train.csv").write_text("broken",encoding="utf-8")
        with self.assertRaisesRegex(ValueError,"checksum"): validate_prepared(path)
        too_many={**self.config,"dataset_id":"large","sampling":{"size":3,"strategy":"uniform","seed":7}}
        with self.assertRaisesRegex(ValueError,"exceeds"): prepare_dataset(too_many,embedding_backend=Stub())
        large=json.loads((self.root/"out/large.failed.json").read_text(encoding="utf-8")); sampling=next(s for s in large["stages"] if s["name"]=="sampling")
        self.assertEqual((sampling["input_count"],sampling["output_count"],sampling["rejected"],sampling["blocked_count"],sampling["requested_size"]),(2,0,0,2,3))

    def test_early_failure_has_stable_identity_without_fake_counts(self):
        config={**self.config,"dataset_id":"early","corpus":{**self.config["corpus"],"path":str(self.root/"missing.csv")}}
        with self.assertRaises(FileNotFoundError): prepare_dataset(config,embedding_backend=Stub())
        report=json.loads((self.root/"out/early.failed.json").read_text(encoding="utf-8")); load=report["stages"][0]
        self.assertIsNone(load["input_count"]); self.assertIsNone(load["output_count"]); self.assertFalse(load["counts_complete"])
        saved={"schema_version":1,**config}; expected=self.root/"expected-config.json"
        expected.write_text(json.dumps(saved,ensure_ascii=False,sort_keys=True,indent=2)+"\n",encoding="utf-8")
        self.assertEqual(report["config_sha256"],sha256_file(expected)); self.assertTrue(report["run_id"])

    def test_validator_rejects_rehashed_statistics_tampering(self):
        mutations=[
          lambda s:s["sampling"].update(eligible=0),
          lambda s:next(x for x in s["stages"] if x["name"]=="sampling").update(input_count=0),
          lambda s:next(x for x in s["stages"] if x["name"]=="sampling").update(output_count=999,rejected=-997),
          lambda s:next(x for x in s["stages"] if x["name"]=="sampling").update(status="not-run"),
          lambda s:s["stages"].pop(),
          lambda s:s["stages"].append(dict(s["stages"][-1])),
          lambda s:next(x for x in s["stages"] if x["name"]=="sampling").update(input_count=True),
          lambda s:next(x for x in s["stages"] if x["name"]=="sampling").update(rejected=1.0),
          lambda s:s["sampling"].update(size=1),
          lambda s:s["sampling"].update(seed=99),
          lambda s:s["sampling"].update(strategy="other"),
          lambda s:next(x for x in s["stages"] if x["name"]=="deduplicate").update(output_count=1),
          lambda s:s["sampling"].update(selected=2.0),
          lambda s:s["sampling"].update(seed=True),
          lambda s:s["outputs"].update(csv_rows=2.0),
          lambda s:s["inventory"].update(sound_files=True),
        ]
        for index,mutation in enumerate(mutations):
            with self.subTest(index=index):
                path=self.good_bundle(f"tamper-{index}",None); self.tamper_stats(path,mutation)
                with self.assertRaises(ValueError): validate_prepared(path)

    def test_exact_full_pool_sample_is_valid(self):
        path=self.good_bundle("exact-full",2); self.assertEqual(validate_prepared(path)["dataset_id"],"exact-full")

    def test_legacy_success_statistics_without_counts_complete_remain_valid(self):
        path=self.good_bundle("legacy-stats",None)
        self.tamper_stats(path,lambda stats:[stage.pop("counts_complete",None) for stage in stats["stages"]])
        self.assertEqual(validate_prepared(path)["dataset_id"],"legacy-stats")

    def test_config_cache_and_truncated_wav(self):
        bad={**self.config,"model":{**self.config["model"],"pooling":"mean_valid"}}
        with self.assertRaisesRegex(ValueError,"mean_valid"): validate_config(bad)
        for size in (True,0,-1):
            invalid={**self.config,"sampling":{"size":size,"strategy":"uniform","seed":7}}
            with self.assertRaisesRegex(ValueError,"positive"): validate_config(invalid)
        entry={"sha256":"a"*64}; self.assertNotEqual(cache_key(entry,self.config["model"],"stub-a"),cache_key(entry,self.config["model"],"transformers"))
        broken=self.root/"broken.wav"; wav(broken); data=broken.read_bytes(); broken.write_bytes(data[:-2])
        with self.assertRaises((ValueError,EOFError,wave.Error)): audio_info(broken)

    def test_sampling_is_independent_of_input_order(self):
        rows=[["你好","ni3 hao3","a"],["我爱","wo3 ai4","b"]]
        self.write_rows(rows); first={**self.config,"dataset_id":"order-a","sampling":{"size":1,"strategy":"uniform","seed":3}}
        a=prepare_dataset(first,embedding_backend=Stub())
        chosen_a=json.loads((a/"reports/selection.jsonl").read_text(encoding="utf-8"))
        self.write_rows(list(reversed(rows))); second={**first,"dataset_id":"order-b"}
        b=prepare_dataset(second,embedding_backend=Stub())
        chosen_b=json.loads((b/"reports/selection.jsonl").read_text(encoding="utf-8"))
        self.assertEqual(chosen_a["sample_id"],chosen_b["sample_id"])

    def test_real_backend_processing_pad_layer_pool_and_resample(self):
        import numpy as np
        import torch
        calls={}
        class Processor:
            def __call__(self,samples,**kwargs):
                calls["samples"]=np.asarray(samples); calls.update(kwargs)
                return {"input_values":torch.tensor([samples],dtype=torch.float32)}
        class Model:
            def __call__(self,**kwargs):
                calls["inference_mode"]=torch.is_inference_mode_enabled()
                return types.SimpleNamespace(hidden_states=(torch.zeros(1,2,2),torch.tensor([[[1.,3.],[5.,7.]]])))
        backend=TransformersBackend.__new__(TransformersBackend); backend.torch=torch; backend.processor=Processor(); backend.model=Model()
        config={**self.config["model"],"sample_rate":4,"padding":"legacy_one_second","layer":1}
        with mock.patch.dict(TransformersBackend.__call__.__globals__, {"_read_audio": mock.Mock(return_value=(np.array([.1,.2],dtype=np.float32),4))}):
            vector=backend({"path":"x.wav"},config)
        self.assertEqual(len(calls["samples"]),4); self.assertTrue(calls["inference_mode"])
        np.testing.assert_allclose(vector,[3.,5.])
        fake_librosa=types.SimpleNamespace(resample=mock.Mock(return_value=np.array([.4,.5],dtype=np.float32)))
        config={**config,"padding":"none"}
        with mock.patch.dict(TransformersBackend.__call__.__globals__, {"_read_audio": mock.Mock(return_value=(np.array([.1],dtype=np.float32),2))}), mock.patch.dict(sys.modules,{"librosa":fake_librosa}):
            backend({"path":"x.wav"},config)
        fake_librosa.resample.assert_called_once()
        self.assertEqual(fake_librosa.resample.call_args.kwargs,{"orig_sr":2,"target_sr":4})


if __name__ == "__main__": unittest.main()
