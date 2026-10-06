"""Reproducible small appearance verifier, trained on manually labeled ball crops.

2 m and 5 m examples are training data. 11 m is a separate validation clip.
No frame coordinates or recording paths are used by the inference module.
"""
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from robokeeper.appearance import BallAppearanceVerifier

ROOT=Path(__file__).resolve().parents[1]
LABELS={
 '2m': {0:(550,357,49),70:(599,408,49),100:(644,409,49),110:(675,402,53),
        120:(764,386,67),130:(898,366,94),140:(1150,356,130),240:(1275,355,94)},
 '5m': {0:(550,390,20),80:(570,411,21),110:(567,424,22),125:(605,409,25),
        140:(705,394,35),160:(938,375,57),170:(1220,350,85),235:(321,366,86)},
 '11m': {0:(563,390,12),100:(567,395,13),130:(575,394,15),145:(596,384,18),
         160:(623,384,24),180:(695,378,37),200:(943,343,75),280:(947,339,61)}
}


def frames(distance):
    path=next((ROOT/'recordings').glob('test_'+distance+'_onground*.mp4'))
    cap=cv2.VideoCapture(str(path))
    try:
        for index,label in LABELS[distance].items():
            cap.set(cv2.CAP_PROP_POS_FRAMES,index)
            ok,frame=cap.read()
            if not ok:raise RuntimeError('Missing labeled frame')
            yield index,cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),label
    finally:cap.release()


# Manually reviewed false proposals from the training clips (never 11 m).
HARD_NEGATIVES = {
    '2m': {100: (790, 127, 16), 216: (755, 125, 21), 218: (757, 130, 21),
           263: (733, 409, 33), 275: (766, 423, 32)},
    '5m': {91: (602, 417, 15), 110: (612, 435, 13), 335: (580, 406, 14),
           360: (616, 257, 9), 390: (592, 409, 19), 410: (576, 148, 11)}
}


def additional_features(hog):
    rng = np.random.default_rng(123)
    features, classes = [], []
    def add(gray, x, y, radius, label):
        patch = BallAppearanceVerifier.patch(gray, x, y, radius)
        patch = np.clip(patch.astype(float)*rng.uniform(.7, 1.3)+rng.uniform(-20, 20), 0, 255).astype(np.uint8)
        features.append(hog.compute(patch).ravel())
        classes.append(label)
    for distance in ('2m', '5m'):
        for index, gray, (x, y, radius) in frames(distance):
            gray = cv2.GaussianBlur(gray, (3, 3), 0)
            partial = x + radius > gray.shape[1] or x - radius < 0
            # Preserve the physical clipping direction for border examples.
            for _ in range(240 if partial else 60):
                add(gray, x+rng.normal(0, radius*.025), y+rng.normal(0, radius*.025),
                    radius*rng.uniform(.85, 1.1), 1)
    for distance, examples in HARD_NEGATIVES.items():
        cap = cv2.VideoCapture(str(next((ROOT/'recordings').glob('test_'+distance+'_onground*.mp4'))))
        for index, (x, y, radius) in examples.items():
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            if not ok: raise RuntimeError('Missing labeled negative frame')
            gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (3, 3), 0)
            for _ in range(600):
                add(gray, x+rng.normal(0, radius*.25), y+rng.normal(0, radius*.25),
                    radius*rng.uniform(.6, 1.5), -1)
        cap.release()
    return np.asarray(features, np.float32), np.asarray(classes, np.int32)


def main():
    cv2.setNumThreads(2)
    cv2.setRNGSeed(512)
    rng=np.random.default_rng(512)
    hog=cv2.HOGDescriptor((32,32),(16,16),(8,8),(8,8),9)
    features=[];classes=[]
    def add(patch,label):
        features.append(hog.compute(patch).ravel());classes.append(label)
    for distance in ('2m','5m'):
        for index,gray,(x,y,r) in frames(distance):
            print("Preparing",distance,index,flush=True)
            for _ in range(160):
                patch=BallAppearanceVerifier.patch(gray,x+rng.normal(0,r*.05),
                                                     y+rng.normal(0,r*.05),r*rng.uniform(.9,1.1))
                angle=rng.uniform(0,360)
                patch=cv2.warpAffine(patch,cv2.getRotationMatrix2D((15.5,15.5),angle,1),(32,32),
                                     borderMode=cv2.BORDER_REFLECT)
                # Small-image blur/compression and lighting augmentation.
                if rng.random()<.6:
                    size=int(rng.integers(10,27));patch=cv2.resize(cv2.resize(patch,(size,size)),(32,32))
                patch=np.clip(patch.astype(float)*rng.uniform(.7,1.3)+rng.uniform(-20,20),0,255).astype(np.uint8)
                add(patch,1)
            circles=cv2.HoughCircles(cv2.GaussianBlur(gray,(3,3),0),cv2.HOUGH_GRADIENT,1.2,10,
                                     param1=80,param2=22,minRadius=3,maxRadius=200)
            negatives=[] if circles is None else list(circles[0])
            negatives.extend((rng.uniform(0,gray.shape[1]),rng.uniform(0,gray.shape[0]),
                              rng.uniform(4,100)) for _ in range(200))
            for nx,ny,nr in negatives:
                if np.hypot(nx-x,ny-y)<nr+r+10:continue
                add(BallAppearanceVerifier.patch(gray,nx,ny,nr),-1)
    # Additional negative views cover camera shake and moving clothing throughout
    # training clips; the upper band never contains the labeled ball in these clips.
    for distance in ('2m','5m'):
        cap=cv2.VideoCapture(str(next((ROOT/'recordings').glob('test_'+distance+'_onground*.mp4'))))
        count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for index in range(0,count,15):
            cap.set(cv2.CAP_PROP_POS_FRAMES,index)
            ok,frame=cap.read()
            if not ok:continue
            gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            for _ in range(150):
                nx=rng.uniform(0,gray.shape[1]);nr=rng.uniform(3,50)
                ny=rng.uniform(nr, max(nr+1,280-nr))
                add(BallAppearanceVerifier.patch(gray,nx,ny,nr),-1)
        cap.release()
    data=np.asarray(features,np.float32);labels=np.asarray(classes,np.int32)
    extra_data, extra_labels = additional_features(hog)
    fit_model(np.concatenate((data, extra_data)), np.concatenate((labels, extra_labels)))


def fit_model(data, labels):
    cv2.setRNGSeed(512)
    print("Training forest", len(data),flush=True)
    tree=cv2.ml.RTrees_create()
    tree.setMaxDepth(16);tree.setMinSampleCount(8);tree.setMaxCategories(2)
    tree.setActiveVarCount(24)
    tree.setPriors(np.array([1.,float((labels==-1).sum()/(labels==1).sum())],np.float32))
    tree.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER,80,0))
    tree.train(data,cv2.ml.ROW_SAMPLE,labels)
    tree.save(str(ROOT/'robokeeper/models/ball_hog_trees.xml.gz'))
    print("Saved forest",flush=True)
    verifier=BallAppearanceVerifier()
    print('Training samples',len(data),'positives',int((labels==1).sum()))
    validation=[]
    for index,gray,(x,y,r) in frames('11m'):
        gray=cv2.GaussianBlur(gray,(3,3),0)
        margin=verifier.score(gray,x,y,r)
        validation.append({'frame':index,'margin':round(margin,3)})
        print('11 m held-out ball',index,round(margin,3))
    manifest={'model':'HOG + OpenCV random forest, 80 trees, depth 16',
              'model_sha256':hashlib.sha256((ROOT/'robokeeper/models/ball_hog_trees.xml.gz').read_bytes()).hexdigest(),
              'opencv_version':cv2.__version__, 'training_distances':['2m','5m'],'validation_distance':'11m','labels':LABELS,'hard_negative_labels':HARD_NEGATIVES,
              'training_samples':len(data),'positive_samples':int((labels==1).sum()),
              'held_out_positive_margins':validation,
              'preprocessing':'Grayscale Gaussian blur 3x3 before inference',
              'validation_note':'11m excluded from model training, but used during algorithm development; not an untouched test set.',
              'negative_augmentation':{'position_std_fraction':.25,'radius_scale':[.6,1.5]},
              'limitation':'Same environment/ball; not independent-environment validation.'}
    (ROOT/'robokeeper/models/training_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')

if __name__=='__main__':main()
