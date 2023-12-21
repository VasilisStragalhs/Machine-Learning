import os
from PIL import Image
import numpy as np
from sklearn import decomposition
from sklearn import cluster
from sklearn import metrics

train_photos, test_photos = [], []

files = os.listdir('../input/images/train_data')
print('total files = ', len(files))

file1 = os.listdir('../input/images/train_data/280')
print('file1 contains ', len(file1), 'images')

file2 = os.listdir('../input/images/train_data/295')
print('file2 contains ', len(file2), 'images')

file3 = os.listdir('../input/images/train_data/472')
print('file3 contains ', len(file3), 'images')

file4 = os.listdir('../input/images/train_data/936')
print('file4 contains ', len(file4), 'images')

file5 = os.listdir('../input/images/train_data/2103')
print('file5 contains ', len(file5), 'images')

file6 = os.listdir('../input/images/train_data/2237')
print('file6 contains ', len(file6), 'images')

file7 = os.listdir('../input/images/train_data/3998')
print('file7 contains ', len(file7), 'images')

file8 = os.listdir('../input/images/train_data/18')
print('file8 contains ', len(file8), 'images')

file9 = os.listdir('../input/images/train_data/3479')
print('file9 contains ', len(file9), 'images')

file10 = os.listdir('../input/images/train_data/1234')
print('file10 contains ', len(file10), 'images')


# file 280
photo1 = np.asarray(Image.open('../input/images/train_data/280/0184_03.jpg')) / 255
photo2 = np.asarray(Image.open('../input/images/train_data/280/0068_01.jpg')) / 255
photo3 = np.asarray(Image.open('../input/images/train_data/280/0094_01.jpg')) / 255
photo4 = np.asarray(Image.open('../input/images/train_data/280/0035_03.jpg')) / 255
photo5 = np.asarray(Image.open('../input/images/train_data/280/0171_01.jpg')) / 255
photo6 = np.asarray(Image.open('../input/images/train_data/280/0157_04.jpg')) / 255
photo7 = np.asarray(Image.open('../input/images/train_data/280/0064_01.jpg')) / 255
photo8 = np.asarray(Image.open('../input/images/train_data/280/0082_01.jpg')) / 255
photo9 = np.asarray(Image.open('../input/images/train_data/280/0116_01.jpg')) / 255
photo10 = np.asarray(Image.open('../input/images/train_data/280/0254_07.jpg')) / 255

train_photos.append(photo1)
train_photos.append(photo2)
train_photos.append(photo3)
train_photos.append(photo4)
train_photos.append(photo5)
train_photos.append(photo6)
train_photos.append(photo7)
train_photos.append(photo8)
train_photos.append(photo9)
train_photos.append(photo10)


photo11 = np.asarray(Image.open('../input/images/train_data/280/0169_03.jpg')) / 255
photo12 = np.asarray(Image.open('../input/images/train_data/280/0031_01.jpg')) / 255
photo13 = np.asarray(Image.open('../input/images/train_data/280/0539_01.jpg')) / 255
photo14 = np.asarray(Image.open('../input/images/train_data/280/0076_01.jpg')) / 255
photo15 = np.asarray(Image.open('../input/images/train_data/280/0052_01.jpg')) / 255
photo16 = np.asarray(Image.open('../input/images/train_data/280/0151_02.jpg')) / 255
photo17 = np.asarray(Image.open('../input/images/train_data/280/0222_04.jpg')) / 255
photo18 = np.asarray(Image.open('../input/images/train_data/280/0206_01.jpg')) / 255
photo19 = np.asarray(Image.open('../input/images/train_data/280/0072_01.jpg')) / 255
photo20 = np.asarray(Image.open('../input/images/train_data/280/0224_02.jpg')) / 255

train_photos.append(photo11)
train_photos.append(photo12)
train_photos.append(photo13)
train_photos.append(photo14)
train_photos.append(photo15)
train_photos.append(photo16)
train_photos.append(photo17)
train_photos.append(photo18)
train_photos.append(photo19)
train_photos.append(photo20)


photo21 = np.asarray(Image.open('../input/images/train_data/280/0006_01.jpg')) / 255
photo22 = np.asarray(Image.open('../input/images/train_data/280/0024_01.jpg')) / 255
photo23 = np.asarray(Image.open('../input/images/train_data/280/0067_01.jpg')) / 255
photo24 = np.asarray(Image.open('../input/images/train_data/280/0538_02.jpg')) / 255
photo25 = np.asarray(Image.open('../input/images/train_data/280/0137_01.jpg')) / 255
photo26 = np.asarray(Image.open('../input/images/train_data/280/0019_02.jpg')) / 255
photo27 = np.asarray(Image.open('../input/images/train_data/280/0242_01.jpg')) / 255
photo28 = np.asarray(Image.open('../input/images/train_data/280/0130_01.jpg')) / 255
photo29 = np.asarray(Image.open('../input/images/train_data/280/0196_01.jpg')) / 255
photo30 = np.asarray(Image.open('../input/images/train_data/280/0200_03.jpg')) / 255

train_photos.append(photo21)
train_photos.append(photo22)
train_photos.append(photo23)
train_photos.append(photo24)
train_photos.append(photo25)
train_photos.append(photo26)
train_photos.append(photo27)
train_photos.append(photo28)
train_photos.append(photo29)
train_photos.append(photo30)


photo31 = np.asarray(Image.open('../input/images/train_data/280/0193_01.jpg')) / 255
photo32 = np.asarray(Image.open('../input/images/train_data/280/0030_05.jpg')) / 255
photo33 = np.asarray(Image.open('../input/images/train_data/280/0147_01.jpg')) / 255
photo34 = np.asarray(Image.open('../input/images/train_data/280/0130_01.jpg')) / 255
photo35 = np.asarray(Image.open('../input/images/train_data/280/0111_01.jpg')) / 255
photo36 = np.asarray(Image.open('../input/images/train_data/280/0069_02.jpg')) / 255
photo37 = np.asarray(Image.open('../input/images/train_data/280/0108_02.jpg')) / 255
photo38 = np.asarray(Image.open('../input/images/train_data/280/0110_01.jpg')) / 255
photo39 = np.asarray(Image.open('../input/images/train_data/280/0271_01.jpg')) / 255
photo40 = np.asarray(Image.open('../input/images/train_data/280/0563_01.jpg')) / 255

train_photos.append(photo31)
train_photos.append(photo32)
train_photos.append(photo33)
train_photos.append(photo34)
train_photos.append(photo35)
train_photos.append(photo36)
train_photos.append(photo37)
train_photos.append(photo38)
train_photos.append(photo39)
train_photos.append(photo40)


photo41 = np.asarray(Image.open('../input/images/train_data/280/0211_01.jpg')) / 255
photo42 = np.asarray(Image.open('../input/images/train_data/280/0158_01.jpg')) / 255
photo43 = np.asarray(Image.open('../input/images/train_data/280/0264_02.jpg')) / 255
photo44 = np.asarray(Image.open('../input/images/train_data/280/0252_05.jpg')) / 255
photo45 = np.asarray(Image.open('../input/images/train_data/280/0029_01.jpg')) / 255
photo46 = np.asarray(Image.open('../input/images/train_data/280/0001_01.jpg')) / 255
photo47 = np.asarray(Image.open('../input/images/train_data/280/0009_02.jpg')) / 255
photo48 = np.asarray(Image.open('../input/images/train_data/280/0005_01.jpg')) / 255
photo49 = np.asarray(Image.open('../input/images/train_data/280/0099_02.jpg')) / 255
photo50 = np.asarray(Image.open('../input/images/train_data/280/0038_01.jpg')) / 255

train_photos.append(photo41)
train_photos.append(photo42)
train_photos.append(photo43)
train_photos.append(photo44)
train_photos.append(photo45)
train_photos.append(photo46)
train_photos.append(photo47)
train_photos.append(photo48)
train_photos.append(photo49)
train_photos.append(photo50)
###################################+###########################################################
# file 295
photo1 = np.asarray(Image.open('../input/images/train_data/295/0059_01.jpg')) / 255
photo2 = np.asarray(Image.open('../input/images/train_data/295/0205_01.jpg')) / 255
photo3 = np.asarray(Image.open('../input/images/train_data/295/0164_01.jpg')) / 255
photo4 = np.asarray(Image.open('../input/images/train_data/295/0223_01.jpg')) / 255
photo5 = np.asarray(Image.open('../input/images/train_data/295/0034_01.jpg')) / 255
photo6 = np.asarray(Image.open('../input/images/train_data/295/0036_01.jpg')) / 255
photo7 = np.asarray(Image.open('../input/images/train_data/295/0495_01.jpg')) / 255
photo8 = np.asarray(Image.open('../input/images/train_data/295/0082_01.jpg')) / 255
photo9 = np.asarray(Image.open('../input/images/train_data/295/0469_01.jpg')) / 255
photo10 = np.asarray(Image.open('../input/images/train_data/295/0188_01.jpg')) / 255

train_photos.append(photo1)
train_photos.append(photo2)
train_photos.append(photo3)
train_photos.append(photo4)
train_photos.append(photo5)
train_photos.append(photo6)
train_photos.append(photo7)
train_photos.append(photo8)
train_photos.append(photo9)
train_photos.append(photo10)


photo11 = np.asarray(Image.open('../input/images/train_data/295/0141_01.jpg')) / 255
photo12 = np.asarray(Image.open('../input/images/train_data/295/0155_01.jpg')) / 255
photo13 = np.asarray(Image.open('../input/images/train_data/295/0017_01.jpg')) / 255
photo14 = np.asarray(Image.open('../input/images/train_data/295/0171_01.jpg')) / 255
photo15 = np.asarray(Image.open('../input/images/train_data/295/0102_01.jpg')) / 255
photo16 = np.asarray(Image.open('../input/images/train_data/295/0206_01.jpg')) / 255
photo17 = np.asarray(Image.open('../input/images/train_data/295/0028_01.jpg')) / 255
photo18 = np.asarray(Image.open('../input/images/train_data/295/0117_01.jpg')) / 255
photo19 = np.asarray(Image.open('../input/images/train_data/295/0127_01.jpg')) / 255
photo20 = np.asarray(Image.open('../input/images/train_data/295/0062_01.jpg')) / 255

train_photos.append(photo11)
train_photos.append(photo12)
train_photos.append(photo13)
train_photos.append(photo14)
train_photos.append(photo15)
train_photos.append(photo16)
train_photos.append(photo17)
train_photos.append(photo18)
train_photos.append(photo19)
train_photos.append(photo20)


photo21 = np.asarray(Image.open('../input/images/train_data/295/0197_01.jpg')) / 255
photo22 = np.asarray(Image.open('../input/images/train_data/295/0505_01.jpg')) / 255
photo23 = np.asarray(Image.open('../input/images/train_data/295/0183_01.jpg')) / 255
photo24 = np.asarray(Image.open('../input/images/train_data/295/0176_02.jpg')) / 255
photo25 = np.asarray(Image.open('../input/images/train_data/295/0160_01.jpg')) / 255
photo26 = np.asarray(Image.open('../input/images/train_data/295/0091_01.jpg')) / 255
photo27 = np.asarray(Image.open('../input/images/train_data/295/0220_02.jpg')) / 255
photo28 = np.asarray(Image.open('../input/images/train_data/295/0482_01.jpg')) / 255
photo29 = np.asarray(Image.open('../input/images/train_data/295/0120_01.jpg')) / 255
photo30 = np.asarray(Image.open('../input/images/train_data/295/0235_01.jpg')) / 255

train_photos.append(photo21)
train_photos.append(photo22)
train_photos.append(photo23)
train_photos.append(photo24)
train_photos.append(photo25)
train_photos.append(photo26)
train_photos.append(photo27)
train_photos.append(photo28)
train_photos.append(photo29)
train_photos.append(photo30)


photo31 = np.asarray(Image.open('../input/images/train_data/295/0067_01.jpg')) / 255
photo32 = np.asarray(Image.open('../input/images/train_data/295/0134_01.jpg')) / 255
photo33 = np.asarray(Image.open('../input/images/train_data/295/0179_01.jpg')) / 255
photo34 = np.asarray(Image.open('../input/images/train_data/295/0107_01.jpg')) / 255
photo35 = np.asarray(Image.open('../input/images/train_data/295/0111_01.jpg')) / 255
photo36 = np.asarray(Image.open('../input/images/train_data/295/0126_01.jpg')) / 255
photo37 = np.asarray(Image.open('../input/images/train_data/295/0065_01.jpg')) / 255
photo38 = np.asarray(Image.open('../input/images/train_data/295/0161_01.jpg')) / 255
photo39 = np.asarray(Image.open('../input/images/train_data/295/0353_03.jpg')) / 255
photo40 = np.asarray(Image.open('../input/images/train_data/295/0086_01.jpg')) / 255

train_photos.append(photo31)
train_photos.append(photo32)
train_photos.append(photo33)
train_photos.append(photo34)
train_photos.append(photo35)
train_photos.append(photo36)
train_photos.append(photo37)
train_photos.append(photo38)
train_photos.append(photo39)
train_photos.append(photo40)


photo41 = np.asarray(Image.open('../input/images/train_data/295/0210_01.jpg')) / 255
photo42 = np.asarray(Image.open('../input/images/train_data/295/0021_01.jpg')) / 255
photo43 = np.asarray(Image.open('../input/images/train_data/295/0104_01.jpg')) / 255
photo44 = np.asarray(Image.open('../input/images/train_data/295/0014_01.jpg')) / 255
photo45 = np.asarray(Image.open('../input/images/train_data/295/0477_01.jpg')) / 255
photo46 = np.asarray(Image.open('../input/images/train_data/295/0023_01.jpg')) / 255
photo47 = np.asarray(Image.open('../input/images/train_data/295/0058_02.jpg')) / 255
photo48 = np.asarray(Image.open('../input/images/train_data/295/0025_02.jpg')) / 255
photo49 = np.asarray(Image.open('../input/images/train_data/295/0087_01.jpg')) / 255
photo50 = np.asarray(Image.open('../input/images/train_data/295/0184_02.jpg')) / 255

train_photos.append(photo41)
train_photos.append(photo42)
train_photos.append(photo43)
train_photos.append(photo44)
train_photos.append(photo45)
train_photos.append(photo46)
train_photos.append(photo47)
train_photos.append(photo48)
train_photos.append(photo49)
train_photos.append(photo50)
###################################+###########################################################
# file 472
photo1 = np.asarray(Image.open('../input/images/train_data/472/0138_01.jpg')) / 255
photo2 = np.asarray(Image.open('../input/images/train_data/472/0219_01.jpg')) / 255
photo3 = np.asarray(Image.open('../input/images/train_data/472/0335_01.jpg')) / 255
photo4 = np.asarray(Image.open('../input/images/train_data/472/0261_01.jpg')) / 255
photo5 = np.asarray(Image.open('../input/images/train_data/472/0203_01.jpg')) / 255
photo6 = np.asarray(Image.open('../input/images/train_data/472/0095_02.jpg')) / 255
photo7 = np.asarray(Image.open('../input/images/train_data/472/0201_01.jpg')) / 255
photo8 = np.asarray(Image.open('../input/images/train_data/472/0176_01.jpg')) / 255
photo9 = np.asarray(Image.open('../input/images/train_data/472/0071_01.jpg')) / 255
photo10 = np.asarray(Image.open('../input/images/train_data/472/0206_01.jpg')) / 255

train_photos.append(photo1)
train_photos.append(photo2)
train_photos.append(photo3)
train_photos.append(photo4)
train_photos.append(photo5)
train_photos.append(photo6)
train_photos.append(photo7)
train_photos.append(photo8)
train_photos.append(photo9)
train_photos.append(photo10)


photo11 = np.asarray(Image.open('../input/images/train_data/472/0293_02.jpg')) / 255
photo12 = np.asarray(Image.open('../input/images/train_data/472/0053_01.jpg')) / 255
photo13 = np.asarray(Image.open('../input/images/train_data/472/0069_01.jpg')) / 255
photo14 = np.asarray(Image.open('../input/images/train_data/472/0076_01.jpg')) / 255
photo15 = np.asarray(Image.open('../input/images/train_data/472/0415_01.jpg')) / 255
photo16 = np.asarray(Image.open('../input/images/train_data/472/0393_02.jpg')) / 255
photo17 = np.asarray(Image.open('../input/images/train_data/472/0274_01.jpg')) / 255
photo18 = np.asarray(Image.open('../input/images/train_data/472/0254_01.jpg')) / 255
photo19 = np.asarray(Image.open('../input/images/train_data/472/0074_01.jpg')) / 255
photo20 = np.asarray(Image.open('../input/images/train_data/472/0412_01.jpg')) / 255

train_photos.append(photo11)
train_photos.append(photo12)
train_photos.append(photo13)
train_photos.append(photo14)
train_photos.append(photo15)
train_photos.append(photo16)
train_photos.append(photo17)
train_photos.append(photo18)
train_photos.append(photo19)
train_photos.append(photo20)


photo21 = np.asarray(Image.open('../input/images/train_data/472/0209_01.jpg')) / 255
photo22 = np.asarray(Image.open('../input/images/train_data/472/0330_02.jpg')) / 255
photo23 = np.asarray(Image.open('../input/images/train_data/472/0006_01.jpg')) / 255
photo24 = np.asarray(Image.open('../input/images/train_data/472/0168_01.jpg')) / 255
photo25 = np.asarray(Image.open('../input/images/train_data/472/0173_01.jpg')) / 255
photo26 = np.asarray(Image.open('../input/images/train_data/472/0392_01.jpg')) / 255
photo27 = np.asarray(Image.open('../input/images/train_data/472/0124_01.jpg')) / 255
photo28 = np.asarray(Image.open('../input/images/train_data/472/0130_01.jpg')) / 255
photo29 = np.asarray(Image.open('../input/images/train_data/472/0396_01.jpg')) / 255
photo30 = np.asarray(Image.open('../input/images/train_data/472/0147_01.jpg')) / 255

train_photos.append(photo21)
train_photos.append(photo22)
train_photos.append(photo23)
train_photos.append(photo24)
train_photos.append(photo25)
train_photos.append(photo26)
train_photos.append(photo27)
train_photos.append(photo28)
train_photos.append(photo29)
train_photos.append(photo30)


photo31 = np.asarray(Image.open('../input/images/train_data/472/0008_01.jpg')) / 255
photo32 = np.asarray(Image.open('../input/images/train_data/472/0181_01.jpg')) / 255
photo33 = np.asarray(Image.open('../input/images/train_data/472/0408_01.jpg')) / 255
photo34 = np.asarray(Image.open('../input/images/train_data/472/0037_01.jpg')) / 255
photo35 = np.asarray(Image.open('../input/images/train_data/472/0040_01.jpg')) / 255
photo36 = np.asarray(Image.open('../input/images/train_data/472/0073_01.jpg')) / 255
photo37 = np.asarray(Image.open('../input/images/train_data/472/0022_01.jpg')) / 255
photo38 = np.asarray(Image.open('../input/images/train_data/472/0343_02.jpg')) / 255
photo39 = np.asarray(Image.open('../input/images/train_data/472/0193_01.jpg')) / 255
photo40 = np.asarray(Image.open('../input/images/train_data/472/0119_01.jpg')) / 255

train_photos.append(photo31)
train_photos.append(photo32)
train_photos.append(photo33)
train_photos.append(photo34)
train_photos.append(photo35)
train_photos.append(photo36)
train_photos.append(photo37)
train_photos.append(photo38)
train_photos.append(photo39)
train_photos.append(photo40)


photo41 = np.asarray(Image.open('../input/images/train_data/472/0270_02.jpg')) / 255
photo42 = np.asarray(Image.open('../input/images/train_data/472/0179_02.jpg')) / 255
photo43 = np.asarray(Image.open('../input/images/train_data/472/0240_01.jpg')) / 255
photo44 = np.asarray(Image.open('../input/images/train_data/472/0298_02.jpg')) / 255
photo45 = np.asarray(Image.open('../input/images/train_data/472/0187_02.jpg')) / 255
photo46 = np.asarray(Image.open('../input/images/train_data/472/0139_01.jpg')) / 255
photo47 = np.asarray(Image.open('../input/images/train_data/472/0247_01.jpg')) / 255
photo48 = np.asarray(Image.open('../input/images/train_data/472/0217_03.jpg')) / 255
photo49 = np.asarray(Image.open('../input/images/train_data/472/0085_01.jpg')) / 255
photo50 = np.asarray(Image.open('../input/images/train_data/472/0208_03.jpg')) / 255

train_photos.append(photo41)
train_photos.append(photo42)
train_photos.append(photo43)
train_photos.append(photo44)
train_photos.append(photo45)
train_photos.append(photo46)
train_photos.append(photo47)
train_photos.append(photo48)
train_photos.append(photo49)
train_photos.append(photo50)
###################################+###########################################################
# file 936
photo1 = np.asarray(Image.open('../input/images/train_data/936/0331_01.jpg')) / 255
photo2 = np.asarray(Image.open('../input/images/train_data/936/0371_01.jpg')) / 255
photo3 = np.asarray(Image.open('../input/images/train_data/936/0209_02.jpg')) / 255
photo4 = np.asarray(Image.open('../input/images/train_data/936/0031_01.jpg')) / 255
photo5 = np.asarray(Image.open('../input/images/train_data/936/0389_02.jpg')) / 255
photo6 = np.asarray(Image.open('../input/images/train_data/936/0351_03.jpg')) / 255
photo7 = np.asarray(Image.open('../input/images/train_data/936/0149_02.jpg')) / 255
photo8 = np.asarray(Image.open('../input/images/train_data/936/0083_01.jpg')) / 255
photo9 = np.asarray(Image.open('../input/images/train_data/936/0019_02.jpg')) / 255
photo10 = np.asarray(Image.open('../input/images/train_data/936/0137_01.jpg')) / 255

train_photos.append(photo1)
train_photos.append(photo2)
train_photos.append(photo3)
train_photos.append(photo4)
train_photos.append(photo5)
train_photos.append(photo6)
train_photos.append(photo7)
train_photos.append(photo8)
train_photos.append(photo9)
train_photos.append(photo10)


photo11 = np.asarray(Image.open('../input/images/train_data/936/0235_01.jpg')) / 255
photo12 = np.asarray(Image.open('../input/images/train_data/936/0090_01.jpg')) / 255
photo13 = np.asarray(Image.open('../input/images/train_data/936/0238_01.jpg')) / 255
photo14 = np.asarray(Image.open('../input/images/train_data/936/0097_02.jpg')) / 255
photo15 = np.asarray(Image.open('../input/images/train_data/936/0063_01.jpg')) / 255
photo16 = np.asarray(Image.open('../input/images/train_data/936/0144_01.jpg')) / 255
photo17 = np.asarray(Image.open('../input/images/train_data/936/0362_01.jpg')) / 255
photo18 = np.asarray(Image.open('../input/images/train_data/936/0098_01.jpg')) / 255
photo19 = np.asarray(Image.open('../input/images/train_data/936/0065_02.jpg')) / 255
photo20 = np.asarray(Image.open('../input/images/train_data/936/0110_01.jpg')) / 255

train_photos.append(photo11)
train_photos.append(photo12)
train_photos.append(photo13)
train_photos.append(photo14)
train_photos.append(photo15)
train_photos.append(photo16)
train_photos.append(photo17)
train_photos.append(photo18)
train_photos.append(photo19)
train_photos.append(photo20)


photo21 = np.asarray(Image.open('../input/images/train_data/936/0143_05.jpg')) / 255
photo22 = np.asarray(Image.open('../input/images/train_data/936/0173_01.jpg')) / 255
photo23 = np.asarray(Image.open('../input/images/train_data/936/0111_01.jpg')) / 255
photo24 = np.asarray(Image.open('../input/images/train_data/936/0044_01.jpg')) / 255
photo25 = np.asarray(Image.open('../input/images/train_data/936/0130_01.jpg')) / 255
photo26 = np.asarray(Image.open('../input/images/train_data/936/0089_02.jpg')) / 255
photo27 = np.asarray(Image.open('../input/images/train_data/936/0147_01.jpg')) / 255
photo28 = np.asarray(Image.open('../input/images/train_data/936/0296_03.jpg')) / 255
photo29 = np.asarray(Image.open('../input/images/train_data/936/0086_01.jpg')) / 255
photo30 = np.asarray(Image.open('../input/images/train_data/936/0341_01.jpg')) / 255

train_photos.append(photo21)
train_photos.append(photo22)
train_photos.append(photo23)
train_photos.append(photo24)
train_photos.append(photo25)
train_photos.append(photo26)
train_photos.append(photo27)
train_photos.append(photo28)
train_photos.append(photo29)
train_photos.append(photo30)


photo31 = np.asarray(Image.open('../input/images/train_data/936/0211_01.jpg')) / 255
photo32 = np.asarray(Image.open('../input/images/train_data/936/0038_02.jpg')) / 255
photo33 = np.asarray(Image.open('../input/images/train_data/936/0073_01.jpg')) / 255
photo34 = np.asarray(Image.open('../input/images/train_data/936/0118_03.jpg')) / 255
photo35 = np.asarray(Image.open('../input/images/train_data/936/0025_01.jpg')) / 255
photo36 = np.asarray(Image.open('../input/images/train_data/936/0200_01.jpg')) / 255
photo37 = np.asarray(Image.open('../input/images/train_data/936/0021_01.jpg')) / 255
photo38 = np.asarray(Image.open('../input/images/train_data/936/0051_01.jpg')) / 255
photo39 = np.asarray(Image.open('../input/images/train_data/936/0305_01.jpg')) / 255
photo40 = np.asarray(Image.open('../input/images/train_data/936/0088_02.jpg')) / 255

train_photos.append(photo31)
train_photos.append(photo32)
train_photos.append(photo33)
train_photos.append(photo34)
train_photos.append(photo35)
train_photos.append(photo36)
train_photos.append(photo37)
train_photos.append(photo38)
train_photos.append(photo39)
train_photos.append(photo40)


photo41 = np.asarray(Image.open('../input/images/train_data/936/0172_02.jpg')) / 255
photo42 = np.asarray(Image.open('../input/images/train_data/936/0014_01.jpg')) / 255
photo43 = np.asarray(Image.open('../input/images/train_data/936/0177_01.jpg')) / 255
photo44 = np.asarray(Image.open('../input/images/train_data/936/0005_01.jpg')) / 255
photo45 = np.asarray(Image.open('../input/images/train_data/936/0148_02.jpg')) / 255
photo46 = np.asarray(Image.open('../input/images/train_data/936/0056_01.jpg')) / 255
photo47 = np.asarray(Image.open('../input/images/train_data/936/0179_04.jpg')) / 255
photo48 = np.asarray(Image.open('../input/images/train_data/936/0087_01.jpg')) / 255
photo49 = np.asarray(Image.open('../input/images/train_data/936/0030_01.jpg')) / 255
photo50 = np.asarray(Image.open('../input/images/train_data/936/0208_03.jpg')) / 255

train_photos.append(photo41)
train_photos.append(photo42)
train_photos.append(photo43)
train_photos.append(photo44)
train_photos.append(photo45)
train_photos.append(photo46)
train_photos.append(photo47)
train_photos.append(photo48)
train_photos.append(photo49)
train_photos.append(photo50)
###################################+###########################################################
# file 2103
photo1 = np.asarray(Image.open('../input/images/train_data/2103/0173_03.jpg')) / 255
photo2 = np.asarray(Image.open('../input/images/train_data/2103/0277_01.jpg')) / 255
photo3 = np.asarray(Image.open('../input/images/train_data/2103/0042_01.jpg')) / 255
photo4 = np.asarray(Image.open('../input/images/train_data/2103/0256_03.jpg')) / 255
photo5 = np.asarray(Image.open('../input/images/train_data/2103/0004_01.jpg')) / 255
photo6 = np.asarray(Image.open('../input/images/train_data/2103/0070_01.jpg')) / 255
photo7 = np.asarray(Image.open('../input/images/train_data/2103/0221_02.jpg')) / 255
photo8 = np.asarray(Image.open('../input/images/train_data/2103/0083_01.jpg')) / 255
photo9 = np.asarray(Image.open('../input/images/train_data/2103/0061_01.jpg')) / 255
photo10 = np.asarray(Image.open('../input/images/train_data/2103/0120_01.jpg')) / 255

train_photos.append(photo1)
train_photos.append(photo2)
train_photos.append(photo3)
train_photos.append(photo4)
train_photos.append(photo5)
train_photos.append(photo6)
train_photos.append(photo7)
train_photos.append(photo8)
train_photos.append(photo9)
train_photos.append(photo10)


photo11 = np.asarray(Image.open('../input/images/train_data/2103/0235_01.jpg')) / 255
photo12 = np.asarray(Image.open('../input/images/train_data/2103/0090_01.jpg')) / 255
photo13 = np.asarray(Image.open('../input/images/train_data/2103/0238_01.jpg')) / 255
photo14 = np.asarray(Image.open('../input/images/train_data/2103/0063_01.jpg')) / 255
photo15 = np.asarray(Image.open('../input/images/train_data/2103/0293_01.jpg')) / 255
photo16 = np.asarray(Image.open('../input/images/train_data/2103/0123_01.jpg')) / 255
photo17 = np.asarray(Image.open('../input/images/train_data/2103/0250_05.jpg')) / 255
photo18 = np.asarray(Image.open('../input/images/train_data/2103/0236_01.jpg')) / 255
photo19 = np.asarray(Image.open('../input/images/train_data/2103/0327_01.jpg')) / 255
photo20 = np.asarray(Image.open('../input/images/train_data/2103/0098_01.jpg')) / 255

train_photos.append(photo11)
train_photos.append(photo12)
train_photos.append(photo13)
train_photos.append(photo14)
train_photos.append(photo15)
train_photos.append(photo16)
train_photos.append(photo17)
train_photos.append(photo18)
train_photos.append(photo19)
train_photos.append(photo20)


photo21 = np.asarray(Image.open('../input/images/train_data/2103/0020_01.jpg')) / 255
photo22 = np.asarray(Image.open('../input/images/train_data/2103/0069_02.jpg')) / 255
photo23 = np.asarray(Image.open('../input/images/train_data/2103/0111_01.jpg')) / 255
photo24 = np.asarray(Image.open('../input/images/train_data/2103/0118_01.jpg')) / 255
photo25 = np.asarray(Image.open('../input/images/train_data/2103/0126_01.jpg')) / 255
photo26 = np.asarray(Image.open('../input/images/train_data/2103/0167_01.jpg')) / 255
photo27 = np.asarray(Image.open('../input/images/train_data/2103/0065_01.jpg')) / 255
photo28 = np.asarray(Image.open('../input/images/train_data/2103/0132_01.jpg')) / 255
photo29 = np.asarray(Image.open('../input/images/train_data/2103/0119_01.jpg')) / 255
photo30 = np.asarray(Image.open('../input/images/train_data/2103/0193_01.jpg')) / 255

train_photos.append(photo21)
train_photos.append(photo22)
train_photos.append(photo23)
train_photos.append(photo24)
train_photos.append(photo25)
train_photos.append(photo26)
train_photos.append(photo27)
train_photos.append(photo28)
train_photos.append(photo29)
train_photos.append(photo30)


photo31 = np.asarray(Image.open('../input/images/train_data/2103/0211_01.jpg')) / 255
photo32 = np.asarray(Image.open('../input/images/train_data/2103/0210_01.jpg')) / 255
photo33 = np.asarray(Image.open('../input/images/train_data/2103/0158_01.jpg')) / 255
photo34 = np.asarray(Image.open('../input/images/train_data/2103/0251_02.jpg')) / 255
photo35 = np.asarray(Image.open('../input/images/train_data/2103/0033_01.jpg')) / 255
photo36 = np.asarray(Image.open('../input/images/train_data/2103/0040_01.jpg')) / 255
photo37 = np.asarray(Image.open('../input/images/train_data/2103/0169_01.jpg')) / 255
photo38 = np.asarray(Image.open('../input/images/train_data/2103/0200_01.jpg')) / 255
photo39 = np.asarray(Image.open('../input/images/train_data/2103/0037_01.jpg')) / 255
photo40 = np.asarray(Image.open('../input/images/train_data/2103/0296_01.jpg')) / 255

train_photos.append(photo31)
train_photos.append(photo32)
train_photos.append(photo33)
train_photos.append(photo34)
train_photos.append(photo35)
train_photos.append(photo36)
train_photos.append(photo37)
train_photos.append(photo38)
train_photos.append(photo39)
train_photos.append(photo40)


photo41 = np.asarray(Image.open('../input/images/train_data/2103/0015_01.jpg')) / 255
photo42 = np.asarray(Image.open('../input/images/train_data/2103/0253_01.jpg')) / 255
photo43 = np.asarray(Image.open('../input/images/train_data/2103/0181_01.jpg')) / 255
photo44 = np.asarray(Image.open('../input/images/train_data/2103/0041_01.jpg')) / 255
photo45 = np.asarray(Image.open('../input/images/train_data/2103/0001_01.jpg')) / 255
photo46 = np.asarray(Image.open('../input/images/train_data/2103/0273_01.jpg')) / 255
photo47 = np.asarray(Image.open('../input/images/train_data/2103/0046_02.jpg')) / 255
photo48 = np.asarray(Image.open('../input/images/train_data/2103/0287_01.jpg')) / 255
photo49 = np.asarray(Image.open('../input/images/train_data/2103/0131_01.jpg')) / 255
photo50 = np.asarray(Image.open('../input/images/train_data/2103/0087_01.jpg')) / 255

train_photos.append(photo41)
train_photos.append(photo42)
train_photos.append(photo43)
train_photos.append(photo44)
train_photos.append(photo45)
train_photos.append(photo46)
train_photos.append(photo47)
train_photos.append(photo48)
train_photos.append(photo49)
train_photos.append(photo50)
###################################+###########################################################
# file 2237
photo1 = np.asarray(Image.open('../input/images/train_data/2237/0029_02.jpg')) / 255
photo2 = np.asarray(Image.open('../input/images/train_data/2237/0097_01.jpg')) / 255
photo3 = np.asarray(Image.open('../input/images/train_data/2237/0254_01.jpg')) / 255
photo4 = np.asarray(Image.open('../input/images/train_data/2237/0160_01.jpg')) / 255
photo5 = np.asarray(Image.open('../input/images/train_data/2237/0277_01.jpg')) / 255
photo6 = np.asarray(Image.open('../input/images/train_data/2237/0224_01.jpg')) / 255
photo7 = np.asarray(Image.open('../input/images/train_data/2237/0268_01.jpg')) / 255
photo8 = np.asarray(Image.open('../input/images/train_data/2237/0196_01.jpg')) / 255
photo9 = np.asarray(Image.open('../input/images/train_data/2237/0269_01.jpg')) / 255
photo10 = np.asarray(Image.open('../input/images/train_data/2237/0091_01.jpg')) / 255

train_photos.append(photo1)
train_photos.append(photo2)
train_photos.append(photo3)
train_photos.append(photo4)
train_photos.append(photo5)
train_photos.append(photo6)
train_photos.append(photo7)
train_photos.append(photo8)
train_photos.append(photo9)
train_photos.append(photo10)


photo11 = np.asarray(Image.open('../input/images/train_data/2237/0083_01.jpg')) / 255
photo12 = np.asarray(Image.open('../input/images/train_data/2237/0359_01.jpg')) / 255
photo13 = np.asarray(Image.open('../input/images/train_data/2237/0137_01.jpg')) / 255
photo14 = np.asarray(Image.open('../input/images/train_data/2237/0068_02.jpg')) / 255
photo15 = np.asarray(Image.open('../input/images/train_data/2237/0075_02.jpg')) / 255
photo16 = np.asarray(Image.open('../input/images/train_data/2237/0217_01.jpg')) / 255
photo17 = np.asarray(Image.open('../input/images/train_data/2237/0090_01.jpg')) / 255
photo18 = np.asarray(Image.open('../input/images/train_data/2237/0238_01.jpg')) / 255
photo19 = np.asarray(Image.open('../input/images/train_data/2237/0283_01.jpg')) / 255
photo20 = np.asarray(Image.open('../input/images/train_data/2237/0180_01.jpg')) / 255

train_photos.append(photo11)
train_photos.append(photo12)
train_photos.append(photo13)
train_photos.append(photo14)
train_photos.append(photo15)
train_photos.append(photo16)
train_photos.append(photo17)
train_photos.append(photo18)
train_photos.append(photo19)
train_photos.append(photo20)


photo21 = np.asarray(Image.open('../input/images/train_data/2237/0215_01.jpg')) / 255
photo22 = np.asarray(Image.open('../input/images/train_data/2237/0364_01.jpg')) / 255
photo23 = np.asarray(Image.open('../input/images/train_data/2237/0098_01.jpg')) / 255
photo24 = np.asarray(Image.open('../input/images/train_data/2237/0289_01.jpg')) / 255
photo25 = np.asarray(Image.open('../input/images/train_data/2237/0216_02.jpg')) / 255
photo26 = np.asarray(Image.open('../input/images/train_data/2237/0168_01.jpg')) / 255
photo27 = np.asarray(Image.open('../input/images/train_data/2237/0297_02.jpg')) / 255
photo28 = np.asarray(Image.open('../input/images/train_data/2237/0149_01.jpg')) / 255
photo29 = np.asarray(Image.open('../input/images/train_data/2237/0147_01.jpg')) / 255
photo30 = np.asarray(Image.open('../input/images/train_data/2237/0161_01.jpg')) / 255

train_photos.append(photo21)
train_photos.append(photo22)
train_photos.append(photo23)
train_photos.append(photo24)
train_photos.append(photo25)
train_photos.append(photo26)
train_photos.append(photo27)
train_photos.append(photo28)
train_photos.append(photo29)
train_photos.append(photo30)


photo31 = np.asarray(Image.open('../input/images/train_data/2237/0227_01.jpg')) / 255
photo32 = np.asarray(Image.open('../input/images/train_data/2237/0163_01.jpg')) / 255
photo33 = np.asarray(Image.open('../input/images/train_data/2237/0228_01.jpg')) / 255
photo34 = np.asarray(Image.open('../input/images/train_data/2237/0022_01.jpg')) / 255
photo35 = np.asarray(Image.open('../input/images/train_data/2237/0210_01.jpg')) / 255
photo36 = np.asarray(Image.open('../input/images/train_data/2237/0123_02.jpg')) / 255
photo37 = np.asarray(Image.open('../input/images/train_data/2237/0115_02.jpg')) / 255
photo38 = np.asarray(Image.open('../input/images/train_data/2237/0037_01.jpg')) / 255
photo39 = np.asarray(Image.open('../input/images/train_data/2237/0308_01.jpg')) / 255
photo40 = np.asarray(Image.open('../input/images/train_data/2237/0041_01.jpg')) / 255

train_photos.append(photo31)
train_photos.append(photo32)
train_photos.append(photo33)
train_photos.append(photo34)
train_photos.append(photo35)
train_photos.append(photo36)
train_photos.append(photo37)
train_photos.append(photo38)
train_photos.append(photo39)
train_photos.append(photo40)


photo41 = np.asarray(Image.open('../input/images/train_data/2237/0079_02.jpg')) / 255
photo42 = np.asarray(Image.open('../input/images/train_data/2237/0014_01.jpg')) / 255
photo43 = np.asarray(Image.open('../input/images/train_data/2237/0179_02.jpg')) / 255
photo44 = np.asarray(Image.open('../input/images/train_data/2237/0041_01.jpg')) / 255
photo45 = np.asarray(Image.open('../input/images/train_data/2237/0023_01.jpg')) / 255
photo46 = np.asarray(Image.open('../input/images/train_data/2237/0401_01.jpg')) / 255
photo47 = np.asarray(Image.open('../input/images/train_data/2237/0173_02.jpg')) / 255
photo48 = np.asarray(Image.open('../input/images/train_data/2237/0058_02.jpg')) / 255
photo49 = np.asarray(Image.open('../input/images/train_data/2237/0085_01.jpg')) / 255
photo50 = np.asarray(Image.open('../input/images/train_data/2237/0053_02.jpg')) / 255

train_photos.append(photo41)
train_photos.append(photo42)
train_photos.append(photo43)
train_photos.append(photo44)
train_photos.append(photo45)
train_photos.append(photo46)
train_photos.append(photo47)
train_photos.append(photo48)
train_photos.append(photo49)
train_photos.append(photo50)
###################################+###########################################################
# file 3998
photo1 = np.asarray(Image.open('../input/images/train_data/3998/0034_01.jpg')) / 255
photo2 = np.asarray(Image.open('../input/images/train_data/3998/0011_01.jpg')) / 255
photo3 = np.asarray(Image.open('../input/images/train_data/3998/0036_01.jpg')) / 255
photo4 = np.asarray(Image.open('../input/images/train_data/3998/0100_01.jpg')) / 255
photo5 = np.asarray(Image.open('../input/images/train_data/3998/0109_01.jpg')) / 255
photo6 = np.asarray(Image.open('../input/images/train_data/3998/0079_01.jpg')) / 255
photo7 = np.asarray(Image.open('../input/images/train_data/3998/0013_01.jpg')) / 255
photo8 = np.asarray(Image.open('../input/images/train_data/3998/0143_01.jpg')) / 255
photo9 = np.asarray(Image.open('../input/images/train_data/3998/0012_01.jpg')) / 255
photo10 = np.asarray(Image.open('../input/images/train_data/3998/0080_01.jpg')) / 255

train_photos.append(photo1)
train_photos.append(photo2)
train_photos.append(photo3)
train_photos.append(photo4)
train_photos.append(photo5)
train_photos.append(photo6)
train_photos.append(photo7)
train_photos.append(photo8)
train_photos.append(photo9)
train_photos.append(photo10)


photo11 = np.asarray(Image.open('../input/images/train_data/3998/0141_01.jpg')) / 255
photo12 = np.asarray(Image.open('../input/images/train_data/3998/0017_01.jpg')) / 255
photo13 = np.asarray(Image.open('../input/images/train_data/3998/0121_01.jpg')) / 255
photo14 = np.asarray(Image.open('../input/images/train_data/3998/0138_01.jpg')) / 255
photo15 = np.asarray(Image.open('../input/images/train_data/3998/0068_01.jpg')) / 255
photo16 = np.asarray(Image.open('../input/images/train_data/3998/0225_01.jpg')) / 255
photo17 = np.asarray(Image.open('../input/images/train_data/3998/0106_01.jpg')) / 255
photo18 = np.asarray(Image.open('../input/images/train_data/3998/0072_01.jpg')) / 255
photo19 = np.asarray(Image.open('../input/images/train_data/3998/0028_01.jpg')) / 255
photo20 = np.asarray(Image.open('../input/images/train_data/3998/0127_01.jpg')) / 255

train_photos.append(photo11)
train_photos.append(photo12)
train_photos.append(photo13)
train_photos.append(photo14)
train_photos.append(photo15)
train_photos.append(photo16)
train_photos.append(photo17)
train_photos.append(photo18)
train_photos.append(photo19)
train_photos.append(photo20)


photo21 = np.asarray(Image.open('../input/images/train_data/3998/0075_01.jpg')) / 255
photo22 = np.asarray(Image.open('../input/images/train_data/3998/0052_01.jpg')) / 255
photo23 = np.asarray(Image.open('../input/images/train_data/3998/0220_01.jpg')) / 255
photo24 = np.asarray(Image.open('../input/images/train_data/3998/0113_01.jpg')) / 255
photo25 = np.asarray(Image.open('../input/images/train_data/3998/0096_01.jpg')) / 255
photo26 = np.asarray(Image.open('../input/images/train_data/3998/0097_01.jpg')) / 255
photo27 = np.asarray(Image.open('../input/images/train_data/3998/0046_01.jpg')) / 255
photo28 = np.asarray(Image.open('../input/images/train_data/3998/0155_02.jpg')) / 255
photo29 = np.asarray(Image.open('../input/images/train_data/3998/0120_01.jpg')) / 255
photo30 = np.asarray(Image.open('../input/images/train_data/3998/0054_01.jpg')) / 255

train_photos.append(photo21)
train_photos.append(photo22)
train_photos.append(photo23)
train_photos.append(photo24)
train_photos.append(photo25)
train_photos.append(photo26)
train_photos.append(photo27)
train_photos.append(photo28)
train_photos.append(photo29)
train_photos.append(photo30)


photo31 = np.asarray(Image.open('../input/images/train_data/3998/0063_01.jpg')) / 255
photo32 = np.asarray(Image.open('../input/images/train_data/3998/0024_01.jpg')) / 255
photo33 = np.asarray(Image.open('../input/images/train_data/3998/0144_01.jpg')) / 255
photo34 = np.asarray(Image.open('../input/images/train_data/3998/0115_01.jpg')) / 255
photo35 = np.asarray(Image.open('../input/images/train_data/3998/0110_01.jpg')) / 255
photo36 = np.asarray(Image.open('../input/images/train_data/3998/0107_01.jpg')) / 255
photo37 = np.asarray(Image.open('../input/images/train_data/3998/0044_01.jpg')) / 255
photo38 = np.asarray(Image.open('../input/images/train_data/3998/0149_01.jpg')) / 255
photo39 = np.asarray(Image.open('../input/images/train_data/3998/0130_01.jpg')) / 255
photo40 = np.asarray(Image.open('../input/images/train_data/3998/0147_01.jpg')) / 255

train_photos.append(photo31)
train_photos.append(photo32)
train_photos.append(photo33)
train_photos.append(photo34)
train_photos.append(photo35)
train_photos.append(photo36)
train_photos.append(photo37)
train_photos.append(photo38)
train_photos.append(photo39)
train_photos.append(photo40)


photo41 = np.asarray(Image.open('../input/images/train_data/3998/0119_01.jpg')) / 255
photo42 = np.asarray(Image.open('../input/images/train_data/3998/0163_01.jpg')) / 255
photo43 = np.asarray(Image.open('../input/images/train_data/3998/0035_01.jpg')) / 255
photo44 = np.asarray(Image.open('../input/images/train_data/3998/0158_01.jpg')) / 255
photo45 = np.asarray(Image.open('../input/images/train_data/3998/0010_01.jpg')) / 255
photo46 = np.asarray(Image.open('../input/images/train_data/3998/0084_01.jpg')) / 255
photo47 = np.asarray(Image.open('../input/images/train_data/3998/0039_01.jpg')) / 255
photo48 = np.asarray(Image.open('../input/images/train_data/3998/0005_01.jpg')) / 255
photo49 = np.asarray(Image.open('../input/images/train_data/3998/0131_01.jpg')) / 255
photo50 = np.asarray(Image.open('../input/images/train_data/3998/0152_01.jpg')) / 255

train_photos.append(photo41)
train_photos.append(photo42)
train_photos.append(photo43)
train_photos.append(photo44)
train_photos.append(photo45)
train_photos.append(photo46)
train_photos.append(photo47)
train_photos.append(photo48)
train_photos.append(photo49)
train_photos.append(photo50)
###################################+###########################################################
# file 18
photo1 = np.asarray(Image.open('../input/images/train_data/18/0089_01.jpg')) / 255
photo2 = np.asarray(Image.open('../input/images/train_data/18/0356_02.jpg')) / 255
photo3 = np.asarray(Image.open('../input/images/train_data/18/0309_03.jpg')) / 255
photo4 = np.asarray(Image.open('../input/images/train_data/18/0378_02.jpg')) / 255
photo5 = np.asarray(Image.open('../input/images/train_data/18/0117_01.jpg')) / 255
photo6 = np.asarray(Image.open('../input/images/train_data/18/0069_01.jpg')) / 255
photo7 = np.asarray(Image.open('../input/images/train_data/18/0076_01.jpg')) / 255
photo8 = np.asarray(Image.open('../input/images/train_data/18/0159_01.jpg')) / 255
photo9 = np.asarray(Image.open('../input/images/train_data/18/0415_01.jpg')) / 255
photo10 = np.asarray(Image.open('../input/images/train_data/18/0183_01.jpg')) / 255

train_photos.append(photo1)
train_photos.append(photo2)
train_photos.append(photo3)
train_photos.append(photo4)
train_photos.append(photo5)
train_photos.append(photo6)
train_photos.append(photo7)
train_photos.append(photo8)
train_photos.append(photo9)
train_photos.append(photo10)


photo11 = np.asarray(Image.open('../input/images/train_data/18/0310_03.jpg')) / 255
photo12 = np.asarray(Image.open('../input/images/train_data/18/0229_03.jpg')) / 255
photo13 = np.asarray(Image.open('../input/images/train_data/18/0430_01.jpg')) / 255
photo14 = np.asarray(Image.open('../input/images/train_data/18/0151_01.jpg')) / 255
photo15 = np.asarray(Image.open('../input/images/train_data/18/0160_01.jpg')) / 255
photo16 = np.asarray(Image.open('../input/images/train_data/18/0043_03.jpg')) / 255
photo17 = np.asarray(Image.open('../input/images/train_data/18/0074_01.jpg')) / 255
photo18 = np.asarray(Image.open('../input/images/train_data/18/0136_01.jpg')) / 255
photo19 = np.asarray(Image.open('../input/images/train_data/18/0061_01.jpg')) / 255
photo20 = np.asarray(Image.open('../input/images/train_data/18/0068_02.jpg')) / 255

train_photos.append(photo11)
train_photos.append(photo12)
train_photos.append(photo13)
train_photos.append(photo14)
train_photos.append(photo15)
train_photos.append(photo16)
train_photos.append(photo17)
train_photos.append(photo18)
train_photos.append(photo19)
train_photos.append(photo20)


photo21 = np.asarray(Image.open('../input/images/train_data/18/0067_01.jpg')) / 255
photo22 = np.asarray(Image.open('../input/images/train_data/18/0090_01.jpg')) / 255
photo23 = np.asarray(Image.open('../input/images/train_data/18/0403_03.jpg')) / 255
photo24 = np.asarray(Image.open('../input/images/train_data/18/0230_02.jpg')) / 255
photo25 = np.asarray(Image.open('../input/images/train_data/18/0024_01.jpg')) / 255
photo26 = np.asarray(Image.open('../input/images/train_data/18/0144_01.jpg')) / 255
photo27 = np.asarray(Image.open('../input/images/train_data/18/0437_01.jpg')) / 255
photo28 = np.asarray(Image.open('../input/images/train_data/18/0056_04.jpg')) / 255
photo29 = np.asarray(Image.open('../input/images/train_data/18/0118_05.jpg')) / 255
photo30 = np.asarray(Image.open('../input/images/train_data/18/0098_01.jpg')) / 255

train_photos.append(photo21)
train_photos.append(photo22)
train_photos.append(photo23)
train_photos.append(photo24)
train_photos.append(photo25)
train_photos.append(photo26)
train_photos.append(photo27)
train_photos.append(photo28)
train_photos.append(photo29)
train_photos.append(photo30)


photo31 = np.asarray(Image.open('../input/images/train_data/18/0110_01.jpg')) / 255
photo32 = np.asarray(Image.open('../input/images/train_data/18/0111_01.jpg')) / 255
photo33 = np.asarray(Image.open('../input/images/train_data/18/0126_01.jpg')) / 255
photo34 = np.asarray(Image.open('../input/images/train_data/18/0010_02.jpg')) / 255
photo35 = np.asarray(Image.open('../input/images/train_data/18/0132_01.jpg')) / 255
photo36 = np.asarray(Image.open('../input/images/train_data/18/0163_01.jpg')) / 255
photo37 = np.asarray(Image.open('../input/images/train_data/18/0228_01.jpg')) / 255
photo38 = np.asarray(Image.open('../input/images/train_data/18/0022_01.jpg')) / 255
photo39 = np.asarray(Image.open('../input/images/train_data/18/0004_05.jpg')) / 255
photo40 = np.asarray(Image.open('../input/images/train_data/18/0115_02.jpg')) / 255

train_photos.append(photo31)
train_photos.append(photo32)
train_photos.append(photo33)
train_photos.append(photo34)
train_photos.append(photo35)
train_photos.append(photo36)
train_photos.append(photo37)
train_photos.append(photo38)
train_photos.append(photo39)
train_photos.append(photo40)


photo41 = np.asarray(Image.open('../input/images/train_data/18/0021_01.jpg')) / 255
photo42 = np.asarray(Image.open('../input/images/train_data/18/0051_01.jpg')) / 255
photo43 = np.asarray(Image.open('../input/images/train_data/18/0198_04.jpg')) / 255
photo44 = np.asarray(Image.open('../input/images/train_data/18/0014_01.jpg')) / 255
photo45 = np.asarray(Image.open('../input/images/train_data/18/0005_01.jpg')) / 255
photo46 = np.asarray(Image.open('../input/images/train_data/18/0424_01.jpg')) / 255
photo47 = np.asarray(Image.open('../input/images/train_data/18/0040_02.jpg')) / 255
photo48 = np.asarray(Image.open('../input/images/train_data/18/0019_03.jpg')) / 255
photo49 = np.asarray(Image.open('../input/images/train_data/18/0152_01.jpg')) / 255
photo50 = np.asarray(Image.open('../input/images/train_data/18/0432_01.jpg')) / 255

train_photos.append(photo41)
train_photos.append(photo42)
train_photos.append(photo43)
train_photos.append(photo44)
train_photos.append(photo45)
train_photos.append(photo46)
train_photos.append(photo47)
train_photos.append(photo48)
train_photos.append(photo49)
train_photos.append(photo50)
###################################+###########################################################
# file 3479
photo1 = np.asarray(Image.open('../input/images/train_data/3479/0269_01.jpg')) / 255
photo2 = np.asarray(Image.open('../input/images/train_data/3479/0091_01.jpg')) / 255
photo3 = np.asarray(Image.open('../input/images/train_data/3479/0306_01.jpg')) / 255
photo4 = np.asarray(Image.open('../input/images/train_data/3479/0406_01.jpg')) / 255
photo5 = np.asarray(Image.open('../input/images/train_data/3479/0136_01.jpg')) / 255
photo6 = np.asarray(Image.open('../input/images/train_data/3479/0061_01.jpg')) / 255
photo7 = np.asarray(Image.open('../input/images/train_data/3479/0019_02.jpg')) / 255
photo8 = np.asarray(Image.open('../input/images/train_data/3479/0137_01.jpg')) / 255
photo9 = np.asarray(Image.open('../input/images/train_data/3479/0235_01.jpg')) / 255
photo10 = np.asarray(Image.open('../input/images/train_data/3479/0238_01.jpg')) / 255

train_photos.append(photo1)
train_photos.append(photo2)
train_photos.append(photo3)
train_photos.append(photo4)
train_photos.append(photo5)
train_photos.append(photo6)
train_photos.append(photo7)
train_photos.append(photo8)
train_photos.append(photo9)
train_photos.append(photo10)


photo11 = np.asarray(Image.open('../input/images/train_data/3479/0283_01.jpg')) / 255
photo12 = np.asarray(Image.open('../input/images/train_data/3479/0024_01.jpg')) / 255
photo13 = np.asarray(Image.open('../input/images/train_data/3479/0180_01.jpg')) / 255
photo14 = np.asarray(Image.open('../input/images/train_data/3479/0316_01.jpg')) / 255
photo15 = np.asarray(Image.open('../input/images/train_data/3479/0134_01.jpg')) / 255
photo16 = np.asarray(Image.open('../input/images/train_data/3479/0245_02.jpg')) / 255
photo17 = np.asarray(Image.open('../input/images/train_data/3479/0065_02.jpg')) / 255
photo18 = np.asarray(Image.open('../input/images/train_data/3479/0110_01.jpg')) / 255
photo19 = np.asarray(Image.open('../input/images/train_data/3479/0282_01.jpg')) / 255
photo20 = np.asarray(Image.open('../input/images/train_data/3479/0241_01.jpg')) / 255

train_photos.append(photo11)
train_photos.append(photo12)
train_photos.append(photo13)
train_photos.append(photo14)
train_photos.append(photo15)
train_photos.append(photo16)
train_photos.append(photo17)
train_photos.append(photo18)
train_photos.append(photo19)
train_photos.append(photo20)


photo21 = np.asarray(Image.open('../input/images/train_data/3479/0270_01.jpg')) / 255
photo22 = np.asarray(Image.open('../input/images/train_data/3479/0149_01.jpg')) / 255
photo23 = np.asarray(Image.open('../input/images/train_data/3479/0276_01.jpg')) / 255
photo24 = np.asarray(Image.open('../input/images/train_data/3479/0126_01.jpg')) / 255
photo25 = np.asarray(Image.open('../input/images/train_data/3479/0396_01.jpg')) / 255
photo26 = np.asarray(Image.open('../input/images/train_data/3479/0161_01.jpg')) / 255
photo27 = np.asarray(Image.open('../input/images/train_data/3479/0227_01.jpg')) / 255
photo28 = np.asarray(Image.open('../input/images/train_data/3479/0048_01.jpg')) / 255
photo29 = np.asarray(Image.open('../input/images/train_data/3479/0086_01.jpg')) / 255
photo30 = np.asarray(Image.open('../input/images/train_data/3479/0342_02.jpg')) / 255

train_photos.append(photo21)
train_photos.append(photo22)
train_photos.append(photo23)
train_photos.append(photo24)
train_photos.append(photo25)
train_photos.append(photo26)
train_photos.append(photo27)
train_photos.append(photo28)
train_photos.append(photo29)
train_photos.append(photo30)


photo31 = np.asarray(Image.open('../input/images/train_data/3479/0158_01.jpg')) / 255
photo32 = np.asarray(Image.open('../input/images/train_data/3479/0343_01.jpg')) / 255
photo33 = np.asarray(Image.open('../input/images/train_data/3479/0073_01.jpg')) / 255
photo34 = np.asarray(Image.open('../input/images/train_data/3479/0182_01.jpg')) / 255
photo35 = np.asarray(Image.open('../input/images/train_data/3479/0029_01.jpg')) / 255
photo36 = np.asarray(Image.open('../input/images/train_data/3479/0051_01.jpg')) / 255
photo37 = np.asarray(Image.open('../input/images/train_data/3479/0104_01.jpg')) / 255
photo38 = np.asarray(Image.open('../input/images/train_data/3479/0253_01.jpg')) / 255
photo39 = np.asarray(Image.open('../input/images/train_data/3479/0181_01.jpg')) / 255
photo40 = np.asarray(Image.open('../input/images/train_data/3479/0200_02.jpg')) / 255

train_photos.append(photo31)
train_photos.append(photo32)
train_photos.append(photo33)
train_photos.append(photo34)
train_photos.append(photo35)
train_photos.append(photo36)
train_photos.append(photo37)
train_photos.append(photo38)
train_photos.append(photo39)
train_photos.append(photo40)


photo41 = np.asarray(Image.open('../input/images/train_data/3479/0041_01.jpg')) / 255
photo42 = np.asarray(Image.open('../input/images/train_data/3479/0008_01.jpg')) / 255
photo43 = np.asarray(Image.open('../input/images/train_data/3479/0273_01.jpg')) / 255
photo44 = np.asarray(Image.open('../input/images/train_data/3479/0240_01.jpg')) / 255
photo45 = np.asarray(Image.open('../input/images/train_data/3479/0177_01.jpg')) / 255
photo46 = np.asarray(Image.open('../input/images/train_data/3479/0005_01.jpg')) / 255
photo47 = np.asarray(Image.open('../input/images/train_data/3479/0139_01.jpg')) / 255
photo48 = np.asarray(Image.open('../input/images/train_data/3479/0247_01.jpg')) / 255
photo49 = np.asarray(Image.open('../input/images/train_data/3479/0173_02.jpg')) / 255
photo50 = np.asarray(Image.open('../input/images/train_data/3479/0038_01.jpg')) / 255

train_photos.append(photo41)
train_photos.append(photo42)
train_photos.append(photo43)
train_photos.append(photo44)
train_photos.append(photo45)
train_photos.append(photo46)
train_photos.append(photo47)
train_photos.append(photo48)
train_photos.append(photo49)
train_photos.append(photo50)
###################################+###########################################################
# file 1234
photo1 = np.asarray(Image.open('../input/images/train_data/1234/0101_01.jpg')) / 255
photo2 = np.asarray(Image.open('../input/images/train_data/1234/0081_01.jpg')) / 255
photo3 = np.asarray(Image.open('../input/images/train_data/1234/0304_01.jpg')) / 255
photo4 = np.asarray(Image.open('../input/images/train_data/1234/0291_02.jpg')) / 255
photo5 = np.asarray(Image.open('../input/images/train_data/1234/0188_01.jpg')) / 255
photo6 = np.asarray(Image.open('../input/images/train_data/1234/0141_01.jpg')) / 255
photo7 = np.asarray(Image.open('../input/images/train_data/1234/0262_01.jpg')) / 255
photo8 = np.asarray(Image.open('../input/images/train_data/1234/0346_01.jpg')) / 255
photo9 = np.asarray(Image.open('../input/images/train_data/1234/0310_01.jpg')) / 255
photo10 = np.asarray(Image.open('../input/images/train_data/1234/0155_01.jpg')) / 255

train_photos.append(photo1)
train_photos.append(photo2)
train_photos.append(photo3)
train_photos.append(photo4)
train_photos.append(photo5)
train_photos.append(photo6)
train_photos.append(photo7)
train_photos.append(photo8)
train_photos.append(photo9)
train_photos.append(photo10)


photo11 = np.asarray(Image.open('../input/images/train_data/1234/0066_01.jpg')) / 255
photo12 = np.asarray(Image.open('../input/images/train_data/1234/0171_01.jpg')) / 255
photo13 = np.asarray(Image.open('../input/images/train_data/1234/0335_01.jpg')) / 255
photo14 = np.asarray(Image.open('../input/images/train_data/1234/0117_01.jpg')) / 255
photo15 = np.asarray(Image.open('../input/images/train_data/1234/0075_01.jpg')) / 255
photo16 = np.asarray(Image.open('../input/images/train_data/1234/0369_01.jpg')) / 255
photo17 = np.asarray(Image.open('../input/images/train_data/1234/0078_01.jpg')) / 255
photo18 = np.asarray(Image.open('../input/images/train_data/1234/0076_01.jpg')) / 255
photo19 = np.asarray(Image.open('../input/images/train_data/1234/0113_01.jpg')) / 255
photo20 = np.asarray(Image.open('../input/images/train_data/1234/0004_01.jpg')) / 255

train_photos.append(photo11)
train_photos.append(photo12)
train_photos.append(photo13)
train_photos.append(photo14)
train_photos.append(photo15)
train_photos.append(photo16)
train_photos.append(photo17)
train_photos.append(photo18)
train_photos.append(photo19)
train_photos.append(photo20)


photo21 = np.asarray(Image.open('../input/images/train_data/1234/0070_01.jpg')) / 255
photo22 = np.asarray(Image.open('../input/images/train_data/1234/0091_01.jpg')) / 255
photo23 = np.asarray(Image.open('../input/images/train_data/1234/0306_01.jpg')) / 255
photo24 = np.asarray(Image.open('../input/images/train_data/1234/0136_01.jpg')) / 255
photo25 = np.asarray(Image.open('../input/images/train_data/1234/0061_01.jpg')) / 255
photo26 = np.asarray(Image.open('../input/images/train_data/1234/0217_01.jpg')) / 255
photo27 = np.asarray(Image.open('../input/images/train_data/1234/0090_01.jpg')) / 255
photo28 = np.asarray(Image.open('../input/images/train_data/1234/0238_01.jpg')) / 255
photo29 = np.asarray(Image.open('../input/images/train_data/1234/0008_02.jpg')) / 255
photo30 = np.asarray(Image.open('../input/images/train_data/1234/0164_03.jpg')) / 255

train_photos.append(photo21)
train_photos.append(photo22)
train_photos.append(photo23)
train_photos.append(photo24)
train_photos.append(photo25)
train_photos.append(photo26)
train_photos.append(photo27)
train_photos.append(photo28)
train_photos.append(photo29)
train_photos.append(photo30)


photo31 = np.asarray(Image.open('../input/images/train_data/1234/0316_01.jpg')) / 255
photo32 = np.asarray(Image.open('../input/images/train_data/1234/0271_01.jpg')) / 255
photo33 = np.asarray(Image.open('../input/images/train_data/1234/0174_01.jpg')) / 255
photo34 = np.asarray(Image.open('../input/images/train_data/1234/0110_01.jpg')) / 255
photo35 = np.asarray(Image.open('../input/images/train_data/1234/0100_02.jpg')) / 255
photo36 = np.asarray(Image.open('../input/images/train_data/1234/0173_01.jpg')) / 255
photo37 = np.asarray(Image.open('../input/images/train_data/1234/0270_01.jpg')) / 255
photo38 = np.asarray(Image.open('../input/images/train_data/1234/0118_01.jpg')) / 255
photo39 = np.asarray(Image.open('../input/images/train_data/1234/0167_01.jpg')) / 255
photo40 = np.asarray(Image.open('../input/images/train_data/1234/0163_01.jpg')) / 255

train_photos.append(photo31)
train_photos.append(photo32)
train_photos.append(photo33)
train_photos.append(photo34)
train_photos.append(photo35)
train_photos.append(photo36)
train_photos.append(photo37)
train_photos.append(photo38)
train_photos.append(photo39)
train_photos.append(photo40)


photo41 = np.asarray(Image.open('../input/images/train_data/1234/0048_01.jpg')) / 255
photo42 = np.asarray(Image.open('../input/images/train_data/1234/0158_01.jpg')) / 255
photo43 = np.asarray(Image.open('../input/images/train_data/1234/0343_01.jpg')) / 255
photo44 = np.asarray(Image.open('../input/images/train_data/1234/0033_01.jpg')) / 255
photo45 = np.asarray(Image.open('../input/images/train_data/1234/0329_01.jpg')) / 255
photo46 = np.asarray(Image.open('../input/images/train_data/1234/0177_01.jpg')) / 255
photo47 = np.asarray(Image.open('../input/images/train_data/1234/0333_01.jpg')) / 255
photo48 = np.asarray(Image.open('../input/images/train_data/1234/0039_01.jpg')) / 255
photo49 = np.asarray(Image.open('../input/images/train_data/1234/0130_02.jpg')) / 255
photo50 = np.asarray(Image.open('../input/images/train_data/1234/0247_01.jpg')) / 255

train_photos.append(photo41)
train_photos.append(photo42)
train_photos.append(photo43)
train_photos.append(photo44)
train_photos.append(photo45)
train_photos.append(photo46)
train_photos.append(photo47)
train_photos.append(photo48)
train_photos.append(photo49)
train_photos.append(photo50)

# red-green-blue photos to gray photos
i = 0
while i < len(train_photos):
    red = train_photos[i][:, :, 0]
    green = train_photos[i][:, :, 1]
    blue = train_photos[i][:, :, 2]
    
    train_photos[i] = (
        0.299 * red
        + 0.587 * green
        + 0.114 * blue)
    i = i + 1

# numpy array
train_photos = np.array(train_photos)
train_photos = train_photos.reshape(train_photos.shape[0], train_photos.shape[1] * train_photos.shape[2])

# Principal Component Analysis
M = 100
train_photos = decomposition.PCA(n_components = M).fit_transform(train_photos)

# test
zero = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
one = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
two = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, ]
three = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
         3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
         3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
four = [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
        4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
        4, 4, 4, 4, 4, 4, 4, 4, 4, 4]
five = [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
        5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
        5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
six = [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
       6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
       6, 6, 6, 6, 6, 6, 6, 6, 6, 6]
seven = [7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
         7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
         7, 7, 7, 7, 7, 7, 7, 7, 7, 7]
eight = [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
         8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
         8, 8, 8, 8, 8, 8, 8, 8, 8, 8]
nine = [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
        9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
        9, 9, 9, 9, 9, 9, 9, 9, 9, 9]

test_photos = np.append(test_photos, (zero, one, two, three, four, five, six, seven, eight, nine)).astype(int)


# agglomerative hierarchical clustering
num = 10
cluster = cluster.AgglomerativeClustering(linkage = 'ward', n_clusters = num).fit_predict(train_photos)

# confusion matrix
cm = metrics.cluster.contingency_matrix(test_photos, cluster)

print("\npurity: ", np.sum(np.amax(cm, axis=0)) / np.sum(cm))
print("f measure: ", metrics.f1_score(test_photos, cluster, average='micro'))