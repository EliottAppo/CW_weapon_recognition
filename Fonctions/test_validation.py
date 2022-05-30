from prediction import prediction
from validation import validation

def test_validation():
    assert validation("../BDD",0.2)
    