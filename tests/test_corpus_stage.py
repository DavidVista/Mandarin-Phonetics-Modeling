import csv,json,pickle,shutil,sys,tempfile,unittest,wave
from pathlib import Path
from unittest import mock
import numpy as np

sys.path.insert(0,str(Path(__file__).parents[1]/"src"))
from phonetics import (assemble_dataset,prepare_acoustic_bundle,prepare_corpus,
                       prepare_experiment,validate_corpus_bundle,validate_prepared)
from phonetics.artifacts import sha256_file,write_json
from phonetics.preparation.corpus_processing import deduplicate_rows


class Backend:
    def __init__(self,name):self.cache_id=name
    def __call__(self,entry,config):return np.array([1.0,float(len(entry["label"]))],dtype=np.float32)


class CorpusStageTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=Path(__file__).parent);self.root=Path(self.tmp.name)
        sounds=self.root/"sounds";sounds.mkdir()
        for name in ("ma1","ma3","ba4"):
            with wave.open(str(sounds/f"{name}.wav"),"wb") as h:
                h.setnchannels(1);h.setsampwidth(2);h.setframerate(8000);h.writeframes(b"\1\0"*32)
        self.sound_manifest=self.root/"sounds.csv";self.sound_manifest.write_text("path,pinyin\nma1.wav,ma1\nma3.wav,ma3\nba4.wav,ba4\n",encoding="utf8")
        self.wubi=self.root/"wubi.pkl"
        with self.wubi.open("wb") as h:pickle.dump({"妈":"a","爸":"b"},h)
        self.model={"id":"fixture-a","revision":"r1","layer":-1,"pooling":"mean_all","sample_rate":8000,"mono":True,"padding":"none","dtype":"float32","device":"cpu"}
        self.bundle_a=self.acoustic("a","fixture-a")
    def tearDown(self):self.tmp.cleanup()
    def acoustic(self,bid,model_id):
        model={**self.model,"id":model_id}
        return prepare_acoustic_bundle({"bundle_id":bid,"output_root":str(self.root/"acoustic"),"cache_root":str(self.root/f"cache-{bid}"),
            "sounds":{"path":str(self.root/"sounds"),"manifest":str(self.sound_manifest),"sha256":sha256_file(self.sound_manifest),"tone_convention":"explicit"},"model":model},embedding_backend=Backend(bid))
    def raw(self,name,rows):
        path=self.root/name
        with path.open("w",encoding="utf8",newline="") as h:
            w=csv.writer(h,lineterminator="\n");w.writerow(["text","pinyin","source_id"]);w.writerows(rows)
        return path
    def corpus_config(self,path,cid="corpus",size=None,bundle=None):
        bundle=bundle or self.bundle_a
        return {"corpus_id":cid,"output_root":str(self.root/"processed"),"corpus":{"path":str(path),"sha256":sha256_file(path),"tone_convention":"explicit"},
            "wubi":{"path":str(self.wubi),"sha256":sha256_file(self.wubi)},"sampling":{"size":size,"strategy":"uniform","seed":7},
            "acoustic_bundle":{"path":str(bundle),"manifest_sha256":sha256_file(bundle/"manifest.json")}}
    def assembly_config(self,corpus,bundle=None,did="dataset"):
        bundle=bundle or self.bundle_a
        return {"dataset_id":did,"output_root":str(self.root/"prepared"),"processed_corpus":{"path":str(corpus),"manifest_sha256":sha256_file(corpus/"manifest.json")},
            "wubi":{"path":str(self.wubi),"sha256":sha256_file(self.wubi)},"acoustic_bundle":{"path":str(bundle),"manifest_sha256":sha256_file(bundle/"manifest.json")}}

    def test_filters_duplicate_sampling_and_assembly_is_byte_copy(self):
        raw=self.raw("raw.csv",[["妈","ma1","1"],["妈","ma1","2"],["爸","ba4","3"],["坏","huai4","4"],["","","5"]])
        corpus=prepare_corpus(self.corpus_config(raw,size=2));manifest=validate_corpus_bundle(corpus)
        self.assertEqual(json.loads((corpus/"reports/processing_statistics.json").read_text(encoding="utf8"))["selected"],2)
        self.assertEqual(len((corpus/"reports/rejections.jsonl").read_text(encoding="utf8").splitlines()),2)
        with mock.patch("phonetics.preparation.corpus_processing.normalize_rows",side_effect=AssertionError("assembly transformed rows")),mock.patch("phonetics.preparation.corpus_processing.filter_rows",side_effect=AssertionError("assembly transformed rows")):
            prepared=assemble_dataset(self.assembly_config(corpus))
        self.assertEqual((corpus/"corpus.csv").read_bytes(),(prepared/"corpus/train.csv").read_bytes())
        self.assertEqual(self.wubi.read_bytes(),(prepared/"wubi/chinese_to_wubi.pkl").read_bytes())
        self.assertEqual((self.bundle_a/"pinyin_embeddings.pkl").read_bytes(),(prepared/"embeddings/pinyin_embeddings.pkl").read_bytes())
        self.assertEqual((self.bundle_a/"sound_inventory.jsonl").read_bytes(),(prepared/"reports/sound_inventory.jsonl").read_bytes());validate_prepared(prepared)
        self.assertEqual(manifest["sources"]["sound_inventory_sha256"],json.loads((prepared/"manifest.json").read_text(encoding="utf8"))["pins"]["sound_inventory_sha256"])

    def test_conflicts_are_fully_excluded_and_legacy_default_still_errors(self):
        raw=self.raw("conflict.csv",[["妈","ma1","1"],["妈","ma2","2"],["爸","ba4","3"],["爸","ba4","4"],["坏","huai4","5"]])
        corpus=prepare_corpus(self.corpus_config(raw,cid="conflict"));selection=[json.loads(x) for x in (corpus/"reports/selection.jsonl").read_text(encoding="utf8").splitlines()]
        self.assertEqual([row["text"] for row in selection],["爸"]);self.assertEqual(selection[0]["source_ids"],["3","4"])
        rejected=[json.loads(x) for x in (corpus/"reports/rejections.jsonl").read_text(encoding="utf8").splitlines()]
        conflicts=[row for row in rejected if row["reason"]=="conflicting_pronunciation"]
        self.assertEqual(len(conflicts),2);self.assertEqual({row["source_id"] for row in conflicts},{"1","2"})
        stats=json.loads((corpus/"reports/processing_statistics.json").read_text(encoding="utf8"));stage=next(x for x in stats["stages"] if x["name"]=="conflicts")
        self.assertEqual((stage["input_count"],stage["output_count"],stage["rejected"]),(5,3,2));self.assertEqual(stats["rejections"]["conflicting_pronunciation"],2)
        rows=[{"text":"妈","labels":["ma1"],"source_id":"1","source_row":1},{"text":"妈","labels":["ma2"],"source_id":"2","source_row":2}]
        with self.assertRaisesRegex(ValueError,"conflicting labels"):deduplicate_rows(rows)

    def test_wholly_conflicting_corpus_fails_diagnostically(self):
        raw=self.raw("all-conflict.csv",[["妈","ma1","1"],["妈","ma2","2"]])
        with self.assertRaisesRegex(ValueError,"no rows remain"):prepare_corpus(self.corpus_config(raw,cid="all-conflict"))
        report=json.loads((self.root/"processed/all-conflict.failed.json").read_text(encoding="utf8"))
        self.assertEqual(report["details"]["conflict_rows"],2);self.assertEqual(len(report["details"]["rejections"]),2)
        self.assertEqual(report["details"]["reasons"],{"conflicting_pronunciation":2})
        conflict_stage=next(x for x in report["stages"] if x["name"]=="conflicts")
        self.assertEqual((conflict_stage["input_count"],conflict_stage["output_count"],conflict_stage["rejected"]),(2,0,2))
        self.assertFalse((self.root/"processed/all-conflict").exists())

    def test_invalid_pinyin_is_rejected_before_conflict_detection(self):
        raw=self.raw("invalid-before-conflict.csv",[["妈","ma1","1"],["妈","invalid","2"],["爸","ba4","3"]])
        corpus=prepare_corpus(self.corpus_config(raw,cid="invalid-first"));selection=[json.loads(x) for x in (corpus/"reports/selection.jsonl").read_text(encoding="utf8").splitlines()]
        self.assertEqual({row["text"] for row in selection},{"妈","爸"})
        stats=json.loads((corpus/"reports/processing_statistics.json").read_text(encoding="utf8"));self.assertEqual(stats["rejections"],{"pinyin":1})
        self.assertEqual(next(x for x in stats["stages"] if x["name"]=="conflicts")["rejected"],0)

    def test_inventory_compatibility_is_model_independent_but_exact(self):
        raw=self.raw("reuse.csv",[["妈","ma1","1"],["爸","ba4","2"]]);corpus=prepare_corpus(self.corpus_config(raw))
        bundle_b=self.acoustic("b","fixture-b");prepared=assemble_dataset(self.assembly_config(corpus,bundle_b,"reused"));validate_prepared(prepared)
        other=self.root/"other-sounds";other.mkdir()
        with wave.open(str(other/"ma1.wav"),"wb") as h:h.setnchannels(1);h.setsampwidth(2);h.setframerate(8000);h.writeframes(b"\2\0"*16)
        sm=self.root/"other.csv";sm.write_text("path,pinyin\nma1.wav,ma1\n",encoding="utf8")
        bad=prepare_acoustic_bundle({"bundle_id":"bad","output_root":str(self.root/"acoustic"),"cache_root":str(self.root/"cache-bad"),"sounds":{"path":str(other),"manifest":str(sm),"sha256":sha256_file(sm),"tone_convention":"explicit"},"model":self.model},embedding_backend=Backend("bad"))
        with self.assertRaisesRegex(ValueError,"inventory pin mismatch"):assemble_dataset(self.assembly_config(corpus,bad,"bad-inventory"))
        changed=self.root/"changed.pkl"
        with changed.open("wb") as h:pickle.dump({"妈":"z","爸":"b"},h)
        cfg=self.assembly_config(corpus,bundle_b,"bad-wubi");cfg["wubi"]={"path":str(changed),"sha256":sha256_file(changed)}
        with self.assertRaisesRegex(ValueError,"Wubi pin mismatch"):assemble_dataset(cfg)

    def test_assembled_bundle_is_portable_and_prepares_experiment(self):
        texts=["妈","爸","妈爸","爸妈","妈妈","爸爸","妈妈爸","妈爸爸","爸妈妈","爸爸妈","妈妈妈","爸爸爸"]
        rows=[[text," ".join("ma1" if ch=="妈" else "ba4" for ch in text),str(i)] for i,text in enumerate(texts,1)]
        corpus=prepare_corpus(self.corpus_config(self.raw("downstream.csv",rows)));prepared=assemble_dataset(self.assembly_config(corpus))
        shutil.rmtree(self.root/"sounds");shutil.rmtree(self.root/"acoustic");shutil.rmtree(self.root/"processed")
        validate_prepared(prepared)
        experiment=prepare_experiment("tiny",datasets_root=prepared,output_root=self.root/"experiments",config_overrides={"architecture":{"hidden_dim":1}})
        self.assertTrue((experiment/"manifest.json").is_file())

    def test_rehashed_semantic_tampering_is_rejected(self):
        corpus=prepare_corpus(self.corpus_config(self.raw("tamper.csv",[["妈","ma1","1"],["爸","ba4","2"]])))
        stats_path=corpus/"reports/processing_statistics.json";stats=json.loads(stats_path.read_text(encoding="utf8"));stats["selected"]+=1;write_json(stats_path,stats)
        manifest=json.loads((corpus/"manifest.json").read_text(encoding="utf8"));manifest["artifacts"]["reports/processing_statistics.json"]=sha256_file(stats_path);write_json(corpus/"manifest.json",manifest)
        with self.assertRaisesRegex(ValueError,"selection counts"):validate_corpus_bundle(corpus)

    def test_missing_run_id_and_invented_inventory_count_are_independently_rejected(self):
        for cid,mutation,pattern in (
            ("missing-run",lambda stats:stats.pop("run_id"),"statistics identity"),
            ("bad-inventory-count",lambda stats:next(x for x in stats["stages"] if x["name"]=="sound_inventory").update({"input_count":999,"output_count":999}),"inventory counts")):
            with self.subTest(cid=cid):
                corpus=prepare_corpus(self.corpus_config(self.raw(f"{cid}.csv",[["妈","ma1","1"],["爸","ba4","2"]]),cid=cid))
                stats_path=corpus/"reports/processing_statistics.json";stats=json.loads(stats_path.read_text(encoding="utf8"));mutation(stats);write_json(stats_path,stats)
                manifest_path=corpus/"manifest.json";manifest=json.loads(manifest_path.read_text(encoding="utf8"));manifest["artifacts"]["reports/processing_statistics.json"]=sha256_file(stats_path);write_json(manifest_path,manifest)
                with self.assertRaisesRegex(ValueError,pattern):validate_corpus_bundle(corpus)

    def test_empty_filter_results_report_actual_reasons(self):
        cases=(("empty",[["","","1"]],"nonempty"),("wubi",[["坏","huai4","1"]],"wubi"),("acoustic",[["妈","ma2","1"]],"acoustic"))
        for cid,rows,reason in cases:
            with self.subTest(reason=reason):
                with self.assertRaisesRegex(ValueError,"no rows remain after corpus filtering"):
                    prepare_corpus(self.corpus_config(self.raw(f"all-{cid}.csv",rows),cid=f"all-{cid}"))
                report=json.loads((self.root/f"processed/all-{cid}.failed.json").read_text(encoding="utf8"))
                self.assertEqual(report["details"]["reasons"],{reason:1});self.assertNotIn("conflicting_pronunciation",report["details"]["reasons"])
                failed=next(x for x in report["stages"] if x["status"]=="failure");self.assertEqual(failed["name"],"validate_nonempty")

    def test_custom_wubi_alphabet_survives_processing_and_assembly_validation(self):
        with self.wubi.open("wb") as h:pickle.dump({"妈":"A","爸":"B"},h)
        raw=self.raw("uppercase.csv",[["妈","ma1","1"],["爸","ba4","2"]]);cfg=self.corpus_config(raw,cid="uppercase");cfg["wubi"]["alphabet"]="AB"
        corpus=prepare_corpus(cfg);assembled=self.assembly_config(corpus,did="uppercase");assembled["wubi"]["alphabet"]="AB"
        prepared=assemble_dataset(assembled);validate_prepared(prepared)


if __name__=="__main__":unittest.main()
