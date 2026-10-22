#!/usr/bin/env python3
"""Small adversarial fixtures for the validation diagnostic's matching semantics."""
import contextlib
import importlib.util
import io
from pathlib import Path
import unittest

from pycocotools.coco import COCO

path=Path(__file__).resolve().parents[1]/'analysis/analyze_light_val_thresholds.py'
spec=importlib.util.spec_from_file_location('light_analysis',path)
a=importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)


def fixture(boxes, predictions, crowds=None):
    c=COCO()
    crowds=crowds or [0]*len(boxes)
    c.dataset=dict(info={},images=[dict(id=1,file_name='x.png',width=200,height=200)],
                   categories=[dict(id=8,name='light')],
                   annotations=[dict(id=i+1,image_id=1,category_id=8,bbox=b,area=b[2]*b[3],iscrowd=cr)
                                for i,(b,cr) in enumerate(zip(boxes,crowds))])
    with contextlib.redirect_stdout(io.StringIO()):c.createIndex()
    prs=[dict(image_id=1,category_id=8,bbox=b,score=s) for b,s in predictions]
    return c,prs


class MatchingTests(unittest.TestCase):
    def test_low_score_localization_and_duplicate(self):
        c,p=fixture([[0,0,20,20],[50,50,20,20],[100,100,20,20]],
                    [([0,0,20,20],.9),([0,0,20,20],.8),([50,50,20,20],.3),([110,100,20,20],.7)])
        e=a.evaluate(c,p)
        r=next(r for r in a.scan(e,8) if r['threshold']==.48 and r['scope']=='light' and r['iou']==.5)
        self.assertEqual((r['tp'],r['fp'],r['fn'],r['duplicate_fp']),(1,2,2,1))
        self.assertEqual([r['bucket'] for r in a.diagnose(c,e,p,8)],['detected','low_score','localization'])
        self.assertEqual(len(a.nms(p)),3)

    def test_one_candidate_cannot_recover_two_gt(self):
        c,p=fixture([[0,0,20,20],[1,1,20,20]],[([0,0,20,20],.9)])
        diag=a.diagnose(c,a.evaluate(c,p),p,8)
        self.assertEqual(sum(r['bucket']=='detected' for r in diag),1)
        self.assertEqual(sum(r['bucket']=='matching_or_postprocess_loss' for r in diag),1)

    def test_crowd_is_ignored_not_fp(self):
        c,p=fixture([[0,0,20,20],[100,100,30,30]],
                    [([0,0,20,20],.9),([100,100,10,10],.8)],crowds=[0,1])
        r=next(r for r in a.scan(a.evaluate(c,p),8) if r['threshold']==.48 and r['scope']=='light' and r['iou']==.5)
        self.assertEqual((r['tp'],r['fp'],r['fn'],r['gt']),(1,0,0,1))


if __name__=='__main__':unittest.main()
