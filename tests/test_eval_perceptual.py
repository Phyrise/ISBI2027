"""CPU comparisons with original perceptual functions; full real-weight LPIPS is opt-in and uses cached weights."""
import ast
import contextlib
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


def original(path, names, **namespace):
    nodes=[n for n in ast.parse(Path(path).read_text()).body if isinstance(n,ast.FunctionDef) and n.name in names]
    exec(compile(ast.Module(nodes,[]),str(path),'exec'),namespace)
    return namespace


@contextlib.contextmanager
def cpu_cuda_calls():
    saved=torch.Tensor.cuda
    torch.Tensor.cuda=lambda x,*a,**k:x
    try: yield
    finally: torch.Tensor.cuda=saved


class TestPerceptualPorts(unittest.TestCase):
    def setUp(self):
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        torch.set_num_threads(2)

    @unittest.skipUnless(os.environ.get('ISBI_ORIG_MEDDINO'), 'set ISBI_ORIG_MEDDINO to the source metrics file')
    def test_meddino_preprocess_features(self):
        import nibabel as nib
        from torchvision import transforms
        from evaluation.metrics.perceptual import meddino as port
        ref=original(os.environ['ISBI_ORIG_MEDDINO'],['preprocess_volume','extract_features'],
                     np=np,nib=nib,torch=torch,transforms=transforms)
        rng=np.random.default_rng(0); v=rng.normal(size=(4,40,48)).astype(np.float32)
        with tempfile.TemporaryDirectory(dir=os.environ.get('ISBI_EVAL_TMP')) as tmp:
            f=Path(tmp)/'image.nii';port.nifti(f,v,{'affine_RAS':np.eye(4).tolist()})
            a=port.preprocess_volume(str(f)); b=ref['preprocess_volume'](str(f))
        gap=float(np.max(np.abs(a-b)));self.assertEqual(gap,0.)
        # Probe exposes the exact tensor passed into the pretrained encoder, including channels, resize and normalisation.
        probe=torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d((3,3)),torch.nn.Flatten())
        for size in (None,(32,32)):
            for norm in (False,True):
                a=port.extract_features(probe,v,size,2,torch.device('cpu'),norm)
                b=ref['extract_features'](probe,v,size,2,torch.device('cpu'),norm)
                gap=max(gap,float((a-b).abs().max()));self.assertTrue(torch.equal(a,b))
        print('GAP MedDINO preprocessing / encoder inputs max_abs_diff=',gap,flush=True)

    @unittest.skipUnless(os.environ.get('ISBI_ORIG_VIT'), 'set ISBI_ORIG_VIT to the source ViT reader')
    def test_vit_masks_canvas_token_weights(self):
        from evaluation.metrics.perceptual import nnf_vit as port
        ref=original(os.environ['ISBI_ORIG_VIT'],['dilate','canvas','token_weights'],np=np,WIN=192)
        rng=np.random.default_rng(1);v=rng.normal(size=(16,24,32)).astype(np.float32); b=v>0;box=(slice(None),)*3
        gap=0.
        for k in (0,2,5):
            self.assertTrue(np.array_equal(port.dilate(b,k),ref['dilate'](b,k)))
        for fill in (None,0.):
            a=port.canvas(v,b,-2.,box,[3,4,5],fill);c=ref['canvas'](v,b,-2.,box,[3,4,5],fill)
            gap=max(gap,float(np.max(np.abs(a-c))));self.assertTrue(np.array_equal(a,c))
        with cpu_cuda_calls():
            a=port.token_weights({'body':b},torch.device('cpu')); c=ref['token_weights']({'body':b})
        gap=max(gap,float((a['body']-c['body']).abs().max()));self.assertTrue(torch.equal(a['body'],c['body']))
        print('GAP nnF ViT dilation / canvas / token weights max_abs_diff=',gap,flush=True)

    @unittest.skipUnless(os.environ.get('ISBI_LPIPS_CHECK')=='1' and os.environ.get('ISBI_ORIG_TOOLS'),
                         'set ISBI_LPIPS_CHECK=1 and ISBI_ORIG_TOOLS; cached VGG weights required')
    def test_lpips_real_weights_original_loop(self):
        from evaluation.metrics.perceptual import lpips_slices as port
        cache=Path(torch.hub.get_dir())/'checkpoints'
        if not list(cache.glob('vgg16-*.pth')): self.skipTest('VGG16 backbone is not cached; no download')
        path=Path(os.environ['ISBI_ORIG_TOOLS'])/'mama_test91_20261029.py'
        fn=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='cmd_lpips')
        loop=next(n for n in ast.walk(fn) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='case')
        start=next(i for i,n in enumerate(loop.body) if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Tuple)
                   and [getattr(t,'id',None) for t in n.targets[0].elts]==['lo','hi'])
        stop=next(i for i,n in enumerate(loop.body) if isinstance(n,ast.Assign) and getattr(n.targets[0],'id',None)=='row')
        row=loop.body[stop].value
        metrics=row.values[next(i for i,k in enumerate(row.keys) if isinstance(k,ast.Constant) and k.value=='metrics')]
        code=ast.Module(loop.body[start:stop]+[ast.Assign([ast.Name('result',ast.Store())],metrics)],[])
        ast.fix_missing_locations(code)
        net,_=port.build(torch.device('cpu'));rng=np.random.default_rng(0)
        real=rng.normal(size=(4,32,32)).astype(np.float32);pred=real+rng.normal(0,.1,real.shape).astype(np.float32)
        body=np.ones_like(real,dtype=bool);les=np.zeros_like(body);les[:,8:16,8:16]=True
        ns=dict(np=np,torch=torch,E=SimpleNamespace(BODY_FRAC=.05),net=net,pred=pred,real=real,body=body,les=les)
        with cpu_cuda_calls(): exec(compile(code,str(path),'exec'),ns)
        got=port.lpips_slices(net,pred,real,body,les,torch.device('cpu'))['metrics']
        gap=max(abs(got[k]-ns['result'][k]) for k in got)
        self.assertEqual(gap,0.)
        print('GAP LPIPS VGG real-weight original loop max_abs_diff=',gap,flush=True)
