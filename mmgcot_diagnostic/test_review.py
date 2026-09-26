import json
import pytest
from mmgcot_diagnostic.review import reconcile


def save(path, rows):
    path.write_text("".join(json.dumps(r)+"\n" for r in rows))
    return path


def test_disagreement_remains_uncertain(tmp_path):
    k=save(tmp_path/'key',[dict(review_id='x',sample_id='s',image_id='1',trajectory_index=0)])
    a=save(tmp_path/'a',[dict(review_id='x',status='correct_unique',reason='a')])
    b=save(tmp_path/'b',[dict(review_id='x',status='ambiguous',reason='b')])
    out=tmp_path/'out'
    reconcile(k,a,b,out)
    row=json.loads(out.read_text())
    assert row['status']=='uncertain' and row['needs_adjudication']


def test_incomplete_independent_review_is_not_consensus(tmp_path):
    k=save(tmp_path/'key',[dict(review_id='x')])
    a=save(tmp_path/'a',[dict(review_id='x',status='correct_unique')])
    b=save(tmp_path/'b',[])
    with pytest.raises(ValueError,match='cover every'):
        reconcile(k,a,b,tmp_path/'out')
