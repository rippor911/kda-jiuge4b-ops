import ast,hashlib,json,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
class RepositoryTests(unittest.TestCase):
    def test_manifest(self):
        names=(ROOT/'MANIFEST.txt').read_text().splitlines()
        self.assertEqual(len(names),len(set(names)))
        for n in names:
            f=ROOT/n;self.assertTrue(f.is_file(),n);self.assertLess(f.stat().st_size,1024*1024,n)
    def test_python(self):
        for f in ROOT.rglob('*.py'):
            if '.git' not in f.parts:ast.parse(f.read_text(),filename=str(f))
    def test_contract(self):
        c=json.loads((ROOT/'configs/tp8-cpt.json').read_text());self.assertEqual(c['parallel']['tp'],8)
        self.assertEqual((c['model']['layers'],c['task']['seq_length'],c['task']['micro_batch_size'],c['task']['global_batch_size']),(62,512,1,16))
        self.assertTrue(c['task']['distributed_optimizer'] and c['task']['recompute'] and c['task']['bf16'])
        self.assertEqual((c['task']['profile_start'],c['task']['profile_end']),(10,15))
    def test_generated_sources(self):
        c=json.loads((ROOT/'src/DISPATCH_CONTRACT.json').read_text())
        for v in c['sources'].values():self.assertEqual(hashlib.sha256((ROOT/'src'/v['file']).read_bytes()).hexdigest(),v['sha256'])
    def test_provenance(self):
        for v in json.loads((ROOT/'docs/SOURCE_PROVENANCE.json').read_text()):
            self.assertEqual(hashlib.sha256((ROOT/v['file']).read_bytes()).hexdigest(),v['export_sha256']);self.assertTrue(v['math_ast_unchanged'])
if __name__=='__main__':unittest.main()
