import csv,json,stat,sys,tempfile,unittest,zipfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]/"src"))
from phonetics.artifacts import sha256_file
from phonetics.source_preparation import build_sound_manifest,export_duyu,pack_sounds,unpack_sounds

class SourcePreparationTests(unittest.TestCase):
    def setUp(self):self.temp=tempfile.TemporaryDirectory(dir=Path(__file__).parent);self.root=Path(self.temp.name)
    def tearDown(self):self.temp.cleanup()
    def test_duyu_injected_export_alignment_cleaning_and_provenance(self):
        calls=[];rows=[{"txt":"你好！","py":"ni3 hao3 ！","id":7},{"txt":"编号10000","py":"bian1 hao4","id":8},
          {"txt":"我 爱","py":["wo3"," ","ai4"],"id":None},{"txt":"坏🙂","py":"huai4 ma5","id":9},{"txt":"短","py":"duan3 extra","id":10},{"txt":"吗","py":"ma","id":11}]
        def loader(name,**kwargs):calls.append((name,kwargs));return rows
        output=export_duyu(self.root/"duyu",revision="commit-abc",split="train",columns={"text":"txt","pinyin":"py","id":"id"},tone_convention="neutral_unmarked",dataset_loader=loader)
        self.assertEqual(calls,[("Duyu/Pinyin-Hanzi",{"revision":"commit-abc","split":"train"})])
        with (output/"corpus.csv").open(encoding="utf-8",newline="") as h: accepted=list(csv.DictReader(h))
        self.assertEqual([(r["text"],r["pinyin"]) for r in accepted],[("你好","ni3 hao3"),("我爱","wo3 ai4"),("吗","ma5")])
        self.assertEqual(accepted[0]["source_id"],"7");self.assertEqual(accepted[1]["source_id"],"commit-abc:train:2")
        metrics=json.loads((output/"metrics.json").read_text(encoding="utf-8"));self.assertEqual(metrics["input"],6);self.assertEqual(metrics["accepted"]+metrics["rejected"],6)
        self.assertEqual(metrics["reasons"],{"alignment":1,"digit":1,"unsupported_character":1})
        raw=[json.loads(x) for x in (output/"source_rows.jsonl").read_text(encoding="utf-8").splitlines()];self.assertEqual(len(raw),6);self.assertEqual(raw[1]["raw_text"],rows[1]["txt"])
        provenance=json.loads((output/"provenance.json").read_text(encoding="utf-8"));self.assertEqual(provenance["revision"],"commit-abc");self.assertEqual(provenance["loader"],"injected")
        for name,digest in provenance["files"].items():self.assertEqual(sha256_file(output/name),digest)
        with self.assertRaises(FileExistsError):export_duyu(output,revision="commit-abc",split="train",columns={"text":"txt","pinyin":"py"},tone_convention="neutral_unmarked",dataset_loader=loader)

    def test_duyu_guards_placeholders_network_and_loader_failure(self):
        with self.assertRaises(ValueError):export_duyu(self.root/"x",revision="REPLACE_WITH_REVISION",split="train",columns={"text":"a","pinyin":"b"},tone_convention="explicit",dataset_loader=lambda *a,**k:[])
        with self.assertRaises(PermissionError):export_duyu(self.root/"x",revision="abc",split="train",columns={"text":"a","pinyin":"b"},tone_convention="explicit")
        def broken(*a,**k):raise RuntimeError("loader failed")
        with self.assertRaises(RuntimeError):export_duyu(self.root/"x",revision="abc",split="train",columns={"text":"a","pinyin":"b"},tone_convention="explicit",dataset_loader=broken)
        self.assertFalse((self.root/"x").exists())
        with self.assertRaisesRegex(ValueError,"missing mapped columns"):export_duyu(self.root/"schema",revision="abc",split="train",columns={"text":"a","pinyin":"b"},tone_convention="explicit",dataset_loader=lambda *a,**k:[{"a":"你"}])
        self.assertFalse((self.root/"schema").exists())

    def test_archive_roundtrip_and_determinism(self):
        source=self.root/"sounds";source.mkdir();(source/"ma.mp3").write_bytes(b"mp3");(source/"ni3.wav").write_bytes(b"wav");(source/"ignore.txt").write_text("x")
        first=pack_sounds(source,self.root/"a.zip");second=pack_sounds(source,self.root/"b.zip")
        self.assertEqual((self.root/"a.zip").read_bytes(),(self.root/"b.zip").read_bytes());self.assertEqual(first["files"],2)
        report=unpack_sounds(self.root/"a.zip",self.root/"out",expected_sha256=first["sha256"]);self.assertEqual(report["inventory"],first["inventory"])
        self.assertEqual({p.name:p.read_bytes() for p in (self.root/"out").iterdir()},{"ma.mp3":b"mp3","ni3.wav":b"wav"})
        with self.assertRaises(FileExistsError):unpack_sounds(self.root/"a.zip",self.root/"out",expected_sha256=first["sha256"])
        with self.assertRaises(ValueError):unpack_sounds(self.root/"a.zip",self.root/"other",expected_sha256="0"*64)

    def test_archive_rejects_unsafe_duplicate_case_symlink_and_empty(self):
        cases={"traversal":[("../x.wav",b"x",None)],"nested":[("a/x.wav",b"x",None)],"backslash":[("a\\x.wav",b"x",None)],
          "dot-alias":[("./x.wav",b"x",None)],"ads":[("x.wav:stream",b"x",None)],"device":[("CON.wav",b"x",None)],
          "trailing-dot":[("x.wav.",b"x",None)],"trailing-space":[("x.wav ",b"x",None)],
          "unsupported":[("x.txt",b"x",None)],"duplicate":[("x.wav",b"1",None),("x.wav",b"2",None)],
          "case":[("X.wav",b"1",None),("x.wav",b"2",None)],"symlink":[("x.wav",b"x",(stat.S_IFLNK|0o777)<<16)],"empty":[]}
        for name,entries in cases.items():
            with self.subTest(name=name):
                archive=self.root/f"{name}.zip"
                with zipfile.ZipFile(archive,"w") as z:
                    for filename,data,attr in entries:
                        info=zipfile.ZipInfo(filename)
                        if attr is not None:info.external_attr=attr
                        z.writestr(info,data)
                destination=self.root/f"out-{name}"
                with self.assertRaises(ValueError):unpack_sounds(archive,destination,expected_sha256=sha256_file(archive))
                self.assertFalse(destination.exists())

    def test_manifest_aliases_exclusions_provenance_and_immutability(self):
        source=self.root/"lessons";source.mkdir()
        for name in ("lue4.mp3","ma.mp3","on1.mp3"):(source/name).write_bytes(name.encode())
        before={p.name:p.read_bytes() for p in source.iterdir()}
        output=build_sound_manifest(source,self.root/"manifest",aliases={"lue4.mp3":"lve4"})
        with (output/"sounds_manifest.csv").open(encoding="utf-8",newline="") as handle:rows=list(csv.DictReader(handle))
        self.assertEqual(rows,[{"path":"lue4.mp3","pinyin":"lve4"},{"path":"ma.mp3","pinyin":"ma5"}])
        report=json.loads((output/"sound_manifest_report.json").read_text(encoding="utf-8"))
        self.assertEqual((report["input"],report["included"],report["excluded"]),(3,2,1))
        self.assertEqual(report["excluded_records"][0]["path"],"on1.mp3")
        provenance=json.loads((output/"sounds_provenance.json").read_text(encoding="utf-8"))
        self.assertEqual(provenance["manifest_sha256"],sha256_file(output/"sounds_manifest.csv"))
        self.assertEqual({p.name:p.read_bytes() for p in source.iterdir()},before)
        with self.assertRaises(FileExistsError):build_sound_manifest(source,output,aliases={"lue4.mp3":"lve4"})
        with self.assertRaises(ValueError):build_sound_manifest(source,self.root/"unused",aliases={"missing.mp3":"ma1"})

    def test_manifest_collision_and_pack_reject_bad_source_name(self):
        source=self.root/"collision";source.mkdir();(source/"ma.mp3").write_bytes(b"a");(source/"ma5.wav").write_bytes(b"b")
        with self.assertRaisesRegex(ValueError,"collision"):build_sound_manifest(source,self.root/"out",aliases={})
        bad=self.root/"bad";bad.mkdir();(bad/"CON.wav").write_bytes(b"x")
        with self.assertRaises(ValueError):pack_sounds(bad,self.root/"bad.zip")
        self.assertFalse((self.root/"bad.zip").exists())

if __name__=="__main__":unittest.main()
