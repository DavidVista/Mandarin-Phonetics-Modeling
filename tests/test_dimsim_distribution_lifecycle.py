import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest

PROJECT=Path(__file__).parents[1]

@unittest.skipUnless(importlib.util.find_spec("matplotlib") and importlib.util.find_spec("dimsim"),"matplotlib and pinned DIMSIM required")
class DimsimDistributionLifecycleTests(unittest.TestCase):
    def test_real_tiny_bundle_complete_nooverwrite_and_failed_manifest(self):
        parent=Path(os.environ.get("TEMP",PROJECT))
        with tempfile.TemporaryDirectory(dir=parent) as tmp:
            root=Path(tmp); (root/"src").mkdir(); shutil.copytree(PROJECT/"src/phonetics",root/"src/phonetics")
            script=textwrap.dedent("""
                import json
                from pathlib import Path
                from test_dimsim_distribution_research import write_bundle
                from phonetics.research.dimsim_distribution import DimsimDistributionConfig,run_dimsim_distribution
                import phonetics.research.dimsim_distribution as research
                root=Path(__import__('sys').argv[1]); evaluation=root/'evaluation'; evaluation.mkdir(); digest=write_bundle(evaluation)
                config=DimsimDistributionConfig(root,'tiny',evaluation,digest); output=run_dimsim_distribution(config)
                manifest=json.loads((output/'manifest.json').read_text()); assert manifest['status']=='complete' and len(manifest['artifacts'])==16 and len(manifest['inputs'])==1
                assert 'cache' not in json.dumps(manifest).lower()
                try: run_dimsim_distribution(config)
                except FileExistsError: pass
                else: raise AssertionError('no-overwrite guard did not fire')
                original=research.render_distribution
                def fail(*args,**kwargs): raise RuntimeError('injected render failure')
                research.render_distribution=fail
                try: run_dimsim_distribution(DimsimDistributionConfig(root,'failed',evaluation,digest))
                except RuntimeError: pass
                else: raise AssertionError('injected failure did not propagate')
                failed=json.loads((root/'outputs/dimsim-distribution/failed/manifest.json').read_text()); assert failed['status']=='failed'
            """)
            env=os.environ.copy(); env["PYTHONPATH"]=os.pathsep.join(x for x in (str(root/"src"),str(PROJECT/"tests"),env.get("PYTHONPATH","")) if x)
            subprocess.run([sys.executable,"-c",script,str(root)],env=env,check=True)

if __name__=="__main__": unittest.main()
