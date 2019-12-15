# Pattern-Recognition
## BOVW(Bag of Visual word) in Colab

### Detail



#### Data Loader (from kaggle)

```
! kaggle competitions download -c 2019-ml-finalproject
! unzip 2019-ml-finalproject.zip
```
#### Dense sift 

```
for img in img_list:
    image = cv2.imread(DATA_ROOT_TRAIN+'/'+cls+'/'+img)
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (256,256))
      
    kp = [cv2.KeyPoint(x,y,8) for y in range(0,gray.shape[0],8) for x in range(0,gray.shape[1], 8)]
    kp,des = sift.compute(gray,kp)

    train_des.append(des)
    train_images.append(gray)
    train_labels.append(label)
```
#### Codebook(Vocabulary)

```
(생략)

seeding = kmc2.kmc2(np.array(train_des).reshape(-1,128),600) 
Kmeans = MiniBatchKMeans(600, init=seeding).fit(np.array(train_des).reshape(-1,128))
codebook = Kmeans.cluster_centers_

```

#### histogram

```
train_h = []
for i in tqdm(range(len(train_des))):
  code, _ = vq.vq(train_des[i], codebook)
  word_hist, bin_edges = np.histogram(code, bins=range(codebook.shape[0] + 1)) 
  train_h.append(word_hist)
```


#### Build Spatial Pyramid  
```
def build_spatial_pyramid(image, descriptor, level):
  
    
    step_size = 8
    height = int(image.shape[0]/step_size)
    width = int(image.shape[1]/step_size)
    idx_crop = np.array(range(len(descriptor))).reshape(height,width)
    
    size = idx_crop.itemsize
    bh, bw = 2**(5-level), 2**(5-level)
    shape = (int(height/bh), int(width/bw), bh, bw)
    strides = size * np.array([width*bh, bw, width, 1])
    crops = np.lib.stride_tricks.as_strided(idx_crop, shape=shape, strides=strides)
    des_idxs = [col_block.flatten().tolist() for row_block in crops
                for col_block in row_block]
    pyramid = []
    for idxs in des_idxs:
        pyramid.append(np.asarray([descriptor[idx] for idx in idxs]))
    return pyramid
    

```
#### Spatial Pyramid Matching
```
def spatial_pyramid_matching(image, descriptor, codebook, level):
    pyramid = []
    if level == 0:
        pyramid += build_spatial_pyramid(image, descriptor, level=0)
        code = [input_vector_encoder(crop, codebook) for crop in pyramid]
        return np.asarray(code).flatten()
    if level == 1:
        pyramid += build_spatial_pyramid(image, descriptor, level=0)
        pyramid += build_spatial_pyramid(image, descriptor, level=1)
        code = [input_vector_encoder(crop, codebook) for crop in pyramid]
        code_level_0 = 0.5 * np.asarray(code[0]).flatten()
        code_level_1 = 0.5 * np.asarray(code[1:]).flatten()
        return np.concatenate((code_level_0, code_level_1))
    if level == 2:
        pyramid += build_spatial_pyramid(image, descriptor, level=0)
        pyramid += build_spatial_pyramid(image, descriptor, level=1)
        pyramid += build_spatial_pyramid(image, descriptor, level=2)
        code = [input_vector_encoder(crop, codebook) for crop in pyramid]
        code_level_0 = 0.25 * np.asarray(code[0]).flatten()
        code_level_1 = 0.25 * np.asarray(code[1:5]).flatten()
        code_level_2 = 0.5 * np.asarray(code[5:]).flatten()
        return np.concatenate((code_level_0, code_level_1, code_level_2))


```



#### 

-------------------------------------
### Paper 

[Beyond Bags of Features: Spatial Pyramid Matching
for Recognizing Natural Scene Categories](https://inc.ucsd.edu/~marni/Igert/Lazebnik_06.pdf)


-------------------------------------
### Report

#### BOVW

| Level | codebook_size | step_size | img_size | histogram_intersection | scaler |accuracy |
|:--------: |:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|
| 0 | 200 | 8 | 256x256 | - | O | 0.37529|
| 0 | 600 | 8 | 256x256 | - | O | 0.40543|
| 2 | 600 | 8 | 256x256 | - | O | 0.58037 |




-------------------------------------
### Reference

##### (https://github.com/CyrusChiu/Image-recognition)</br>
##### (https://github.com/bikz05/bag-of-words)</br>
-------------------------------------
