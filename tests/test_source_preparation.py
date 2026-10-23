import csv, hashlib, json, pickle, sys, tempfile, unittest, wave
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]/"src"))
from phonetics.artifacts import sha256_file
from phonetics.source_preparation.corpus import clean_aixblock_text,load_aixblock_lots
from phonetics.source_preparation.sounds import prepare_studycli_sounds
from phonetics.source_preparation.labeller import train_labeller,load_labeller,annotate_corpus,predict_pinyin,_inventory,_evaluate,_train_polyphonic_characters

def make_wav(path):
    with wave.open(str(path),"wb") as h: h.setnchannels(1);h.setsampwidth(2);h.setframerate(8000);h.writeframes(b"\0\0"*8)

class SourcePreparationTests(unittest.TestCase):
    def setUp(self): self.tmp=tempfile.TemporaryDirectory(dir=Path(__file__).parent);self.root=Path(self.tmp.name)
    def tearDown(self): self.tmp.cleanup()
    def test_cleaning_and_two_lot_accounting(self):
        self.assertEqual(clean_aixblock_text("编号10000")[1],"contains_digit")
        self.assertEqual(clean_aixblock_text("编号１２")[1],"contains_digit")
        self.assertEqual(clean_aixblock_text("，。 ")[1],"empty_after_cleaning")
        self.assertEqual(clean_aixblock_text(None)[1],"missing_or_empty")
        self.assertEqual(clean_aixblock_text("你好🙂")[1],"unsupported_character")
        specs=[]
        for lot,rows in (("lot1",[["1","你好！","Accepted","Simplified Chinese Mainland"],["2","坏1","Accepted","Simplified Chinese Mainland"]]),
                         ("lot2",[["1","中国","Rejected","Simplified Chinese Mainland"],["1","我爱","Accepted","Simplified Chinese Mainland"]])):
            path=self.root/f"{lot}.csv"
            with path.open("w",encoding="utf-8",newline="") as h:
                w=csv.writer(h);w.writerow(["Sr No","Utterance","Status","Language"]);w.writerows(rows)
            specs.append({"lot_id":lot,"path":str(path),"sha256":sha256_file(path),"revision":"r1"})
        records,rejections,counts=load_aixblock_lots(specs)
        self.assertEqual(counts,{"input":4,"source_eligible":3,"accepted":2,"rejected":2,"changed":1})
        self.assertEqual(len({r["source_id"] for r in records}),2);self.assertEqual({r["lot_id"] for r in records},{"lot1","lot2"})

    def test_sound_offline_publish_failure_and_cache(self):
        neutral=self.root/"neutral";neutral.mkdir();make_wav(neutral/"ma5.wav")
        inventory=self.root/"inventory.txt";inventory.write_text("ma5\nni3\n",encoding="utf-8")
        good=self.root/"good.wav";make_wav(good);calls=[]
        def fetch(url): calls.append(url);return (200,good.read_bytes())
        out=prepare_studycli_sounds(source_root=self.root/"source",tonal_inventory_path=inventory,inventory_sha256=sha256_file(inventory),neutral_dir=neutral,requested_urls={"ni3":"https://x/ni3.wav"},fetcher=fetch)
        self.assertEqual(out,self.root/"source/audio");self.assertEqual(len(calls),1)
        prepare_studycli_sounds(source_root=self.root/"source",tonal_inventory_path=inventory,inventory_sha256=sha256_file(inventory),neutral_dir=neutral,requested_urls={"ni3":"https://x/ni3.wav"},fetcher=fetch)
        self.assertEqual(len(calls),1)
        provenance=json.loads((out/"sounds_provenance.json").read_text());self.assertTrue(provenance["complete"]);self.assertEqual(provenance["fetch_mode"],"injected")
        manifest=out/"sounds_manifest.csv";original=manifest.read_text(encoding="utf-8");manifest.write_text(original.replace("ni3","ma5"),encoding="utf-8")
        with self.assertRaisesRegex(ValueError,"manifest"):prepare_studycli_sounds(source_root=self.root/"source",tonal_inventory_path=inventory,inventory_sha256=sha256_file(inventory),neutral_dir=neutral,requested_urls={"ni3":"https://x/ni3.wav"},fetcher=fetch)
        manifest.write_text(original,encoding="utf-8")
        with self.assertRaises(ValueError): prepare_studycli_sounds(source_root=self.root/"source",tonal_inventory_path=inventory,inventory_sha256=sha256_file(inventory),neutral_dir=neutral,requested_urls={"ni3":"https://different"},fetcher=fetch)
        failed=self.root/"failed-source"
        with self.assertRaises(ValueError): prepare_studycli_sounds(source_root=failed,tonal_inventory_path=inventory,inventory_sha256=sha256_file(inventory),neutral_dir=neutral,requested_urls={"ni3":"https://x/missing.wav"},fetcher=lambda url:(404,b"html"))
        report=json.loads((failed/"audio.failed.json").read_text());self.assertEqual(report["counts"]["failed"],1);self.assertFalse((failed/"audio").exists())

    def test_sound_rejects_html_timeout_truncation_and_collision(self):
        neutral=self.root/"neutral";neutral.mkdir();make_wav(neutral/"ma5.wav")
        inventory=self.root/"inventory.txt";inventory.write_text("ma5\nni3\n",encoding="utf-8");digest=sha256_file(inventory)
        cases=[("html",lambda url:(200,b"<html>no audio</html>")),("timeout",lambda url:(_ for _ in ()).throw(TimeoutError("timeout")))]
        for name,fetch in cases:
            with self.subTest(name=name):
                root=self.root/name
                with self.assertRaises(Exception):prepare_studycli_sounds(source_root=root,tonal_inventory_path=inventory,inventory_sha256=digest,neutral_dir=neutral,requested_urls={"ni3":"https://x/ni3.wav"},fetcher=fetch)
                self.assertTrue((root/"audio.failed.json").is_file())
        with self.assertRaisesRegex(ValueError,"collide"):prepare_studycli_sounds(source_root=self.root/"collision",tonal_inventory_path=inventory,inventory_sha256=digest,neutral_dir=neutral,requested_urls={"ma5":"https://x/ma5.wav"},fetcher=lambda u:b"")
        broken=self.root/"broken-neutral";broken.mkdir();make_wav(broken/"ma5.wav");data=(broken/"ma5.wav").read_bytes();(broken/"ma5.wav").write_bytes(data[:-2])
        with self.assertRaises(Exception):prepare_studycli_sounds(source_root=self.root/"truncated",tonal_inventory_path=inventory,inventory_sha256=digest,neutral_dir=broken,requested_urls={"ni3":"https://x/ni3.wav"},fetcher=lambda u:b"")

    def test_partial_audio_removes_failed_temporary_and_reuses_without_fetch(self):
        neutral=self.root/"neutral-partial";neutral.mkdir();make_wav(neutral/"ma5.wav");good=self.root/"ni3.wav";make_wav(good)
        inventory=self.root/"partial.txt";inventory.write_text("ma5\nni3\nhao3\n",encoding="utf-8");calls=[]
        def fetch(url): calls.append(url);return (200,good.read_bytes() if "ni3" in url else b"<html>bad</html>")
        kwargs={"source_root":self.root/"partial-source","tonal_inventory_path":inventory,"inventory_sha256":sha256_file(inventory),"neutral_dir":neutral,
                "requested_urls":{"ni3":"https://x/ni3.wav","hao3":"https://x/hao3.wav"},"fetcher":fetch,"require_complete":False}
        audio=prepare_studycli_sounds(**kwargs);self.assertEqual(len(calls),2);self.assertFalse(any(p.name.endswith(".tmp") for p in (audio/"sounds").iterdir()))
        provenance=json.loads((audio/"sounds_provenance.json").read_text());self.assertEqual(provenance["missing"],["hao3"]);self.assertTrue((audio/"sounds/ni3.wav").is_file())
        prepare_studycli_sounds(**kwargs);self.assertEqual(len(calls),2)

    def test_group_metrics_and_train_only_polyphony(self):
        import torch
        class Model:
            def eval(self): return self
            def __call__(self,ids,lengths):
                logits=torch.zeros(ids.shape[0],ids.shape[1],5)
                logits[...,3]=1; return logits
        rows=[{"text":"你你新","labels":["ma5","ni3","ma5"]}]
        result=_evaluate(Model(),rows,{"[UNK]":0,"[PAD]":1,"你":2},{"[UNK]":0,"[PAD]":1,"ni3":2,"ma5":3,"hao3":4},torch.device("cpu"),2,["你"])
        self.assertEqual(result["groups"]["neutral"],{"token_count":2,"correct_token_count":2,"token_accuracy":1.0})
        self.assertEqual(result["groups"]["polyphonic"],{"token_count":2,"correct_token_count":1,"token_accuracy":.5})
        self.assertEqual(result["groups"]["oov"],{"token_count":1,"correct_token_count":1,"token_accuracy":1.0});self.assertEqual(result["oov_tokens"],1)
        empty=_evaluate(Model(),[{"text":"好","labels":["hao3"]}],{"[UNK]":0,"[PAD]":1,"好":2},{"[UNK]":0,"[PAD]":1,"ni3":2,"ma5":3,"hao3":4},torch.device("cpu"),1,[])
        self.assertIsNone(empty["groups"]["neutral"]["token_accuracy"]);self.assertIsNone(empty["groups"]["polyphonic"]["token_accuracy"])
        train=[{"text":"妈","labels":["ma1"]}]; heldout=[{"text":"妈","labels":["ma3"]}]
        self.assertEqual(_train_polyphonic_characters(train),[]);self.assertEqual(_train_polyphonic_characters(train+heldout),["妈"])

    def test_inventory_requires_nonempty_explicit_canonical_tokens(self):
        for index,content in enumerate(("","de\n","ma5\nma5\n")):
            path=self.root/f"bad-{index}.txt";path.write_text(content,encoding="utf-8")
            with self.assertRaises(ValueError):_inventory(path,sha256_file(path))

    def test_tiny_labeller_bundle_padding_and_oov(self):
        inventory=self.root/"pinyin.txt";inventory.write_text("ni3\nhao3\nwo3\nai4\nzhong1\nguo2\nma5\n",encoding="utf-8")
        training=self.root/"duyu.csv";rows=[["你好","ni3 hao3"],["我爱","wo3 ai4"],["中国","zhong1 guo2"],["你吗","ni3 ma5"],["好马","hao3 ma5"],["爱国","ai4 guo2"]]
        with training.open("w",encoding="utf-8",newline="") as h:w=csv.writer(h);w.writerow(["sentence","labels"]);w.writerows(rows)
        config={"architecture":{"embedding_dim":4,"hidden_dim":4,"layers":1,"dropout":0.0},"training":{"epochs":1,"batch_size":2,"learning_rate":.01},"seed":3,"split_ratios":[.5,.25,.25],"device":"cpu","tone_convention":"explicit"}
        bundle=train_labeller(training,self.root/"models",bundle_id="tiny",source_revision="r1",source_sha256=sha256_file(training),inventory_path=inventory,inventory_sha256=sha256_file(inventory),config=config,columns={"text":"sentence","pinyin":"labels"})
        loaded=load_labeller(bundle);self.assertEqual(loaded["vocab"]["characters"][:2],["[UNK]","[PAD]"])
        split_ids=[row["sample_id"] for split in loaded["splits"].values() for row in split]
        self.assertEqual(len(split_ids),len(set(split_ids)));self.assertLess(len(loaded["vocab"]["characters"])-2,len({ch for text,_ in rows for ch in text}))
        known=loaded["vocab"]["characters"][2]
        single=predict_pinyin(loaded,known);batched=predict_pinyin(loaded,[known,known*2])[0];self.assertEqual(single,batched)
        output=annotate_corpus([{"text":known,"source_id":"a","raw_text":known,"source_row":1,"lot_id":"lot1"},{"text":"新","source_id":"b"}],loaded,self.root/"sources",dataset_id="labelled")
        metrics=json.loads((output/"labeling_metrics.json").read_text());self.assertEqual(metrics["input"],2);self.assertEqual(metrics["rejected"],1)
        vocab_path=bundle/"vocab.json";vocab=json.loads(vocab_path.read_text());vocab["characters"].append("篡");vocab_path.write_text(json.dumps(vocab,ensure_ascii=False,sort_keys=True,indent=2)+"\n",encoding="utf-8")
        manifest_path=bundle/"manifest.json";manifest=json.loads(manifest_path.read_text());manifest["artifacts"]["vocab.json"]=sha256_file(vocab_path);manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,sort_keys=True,indent=2)+"\n",encoding="utf-8")
        with self.assertRaisesRegex(ValueError,"provenance"):load_labeller(bundle)
        with self.assertRaises(FileExistsError): train_labeller(training,self.root/"models",bundle_id="tiny",source_revision="r1",source_sha256=sha256_file(training),inventory_path=inventory,inventory_sha256=sha256_file(inventory),config=config,columns={"text":"sentence","pinyin":"labels"})

if __name__=="__main__":unittest.main()
