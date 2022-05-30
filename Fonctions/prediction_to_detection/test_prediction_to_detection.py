from prediction_to_detection import detection_with_prediction_for_test,non_max_suppression
import os
import xml.etree.ElementTree as ET
import numpy as np

def file_to_box(imgfile,xmlfile):
    """Permet de passer d'une image avec une arme au découpage de cette arme qui a été fait manuellement """
    tree = ET.parse(xmlfile)
    root = tree.getroot()
    w = int(root[4][0].text)
    h = int(root[4][1].text)
    xmin = int(root[6][4][0].text)
    ymin = int(root[6][4][1].text)
    return [ymin,ymin+h,xmin,xmin+w]


def test_detection_with_prediction():
    """On a un dataset avec des images pour lesquelles on a découpé les zones où il y des armes.
    Pour on prend les zones données par le dataset et par detection_with_prediction.
    Ensuite on regarde si ces zones sont assez superposées avec non_max suppression """
    file_list = os.listdir("../../BDD/dataset/train")

    for i in range(len(file_list)//2):
        print("../../BDD/dataset/train/"+file_list[2*i],"../../BDD/dataset/train/"+file_list[2*i+1])
        box1 = file_to_box("../../BDD/dataset/train/"+file_list[2*i],"../../BDD/dataset/train/"+file_list[2*i+1])
        box2 = detection_with_prediction_for_test("../../BDD/dataset/train/"+file_list[2*i],0.2)
        print(box2,type(box2))
        assert box2 is not None
        boxes =np.array([box1,list(box2)])
        print(boxes)
        boxes,proba = non_max_suppression(boxes)
        assert len(boxes)==1
        print(boxes)

