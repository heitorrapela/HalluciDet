import os
import cv2
import torch
import imageio
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image
from utils.utils import Utils
import glob


class ToTensor(object):
    def __init__(self, modality):
        self.modality = modality

    def __call__(self, sample):

        if(self.modality == 'both'):
            image_rgb, image_ir = sample["img_rgb"], sample["img_ir"]
            bbox_rgb, bbox_ir = sample["annot_rgb"]["bboxes"], sample["annot_ir"]["bboxes"]
            
        else:
            image, bbox = sample["img"], sample["annot"]["bboxes"]

        if self.modality == "both":

            image_rgb = self.check_range(image_rgb)
            image_ir = self.check_range(image_ir)

            image_rgb = torch.from_numpy(image_rgb.transpose((2, 0, 1)))
            image_ir = torch.from_numpy(image_ir).float().unsqueeze_(0)
            
            target_rgb = {
                "boxes" : torch.from_numpy(bbox_rgb).squeeze().view(-1, 4),
                "labels": torch.from_numpy(sample["annot_rgb"]["labels"]).view(-1)
            }
            target_ir = {
                "boxes" : torch.from_numpy(bbox_ir).squeeze().view(-1, 4),
                "labels": torch.from_numpy(sample["annot_ir"]["labels"]).view(-1)
            }

            return image_rgb, target_rgb, image_ir, target_ir

        elif self.modality == "rgb":
            image = self.check_range(image)
            image = torch.from_numpy(image.transpose((2, 0, 1)))

        elif self.modality == "ir":

            image = self.check_range(image)
            image = torch.from_numpy(image)
            image.unsqueeze_(0)

        target = {
            "boxes" : torch.from_numpy(bbox).squeeze().view(-1, 4),
            "labels": torch.from_numpy(sample["annot"]["labels"]).view(-1)
        }

        return image, target

    def check_range(self, image):

        if image.dtype == np.uint8 or (
            image.min() >= 0 and image.max() > 1 and image.max() <= 255
        ):
            image = image.astype("float") / 255.0

        assert (
            image.dtype == np.float and image.min() >= 0 and image.max() <= 1
        ), "Please, verify your images are either uint8 (0-255) or float (0.0-1.0)"

        return image


# Pytorch dataset creation for loading both RGB and IR images with bbox annotations
class SingleModalDetectionDataset(torch.utils.data.Dataset):
    """
    SingleModalDetectionDataset is a PyTorch dataset implementation for loading
    RGB or IR annotated data. This implementation uses ``imageio``
    for reading the images, ``numpy`` for reading the raw IR data and ``xml``
    for parsing the annotations files.
    """
    def __init__(self, dataset, path_images, modality=None, transforms=None, ext=".png", train=True):
        """
        The constructor receives the path to RGB images and the path to IR npy.
        The folders are assumed to contain both images/raw IR and ``xml`` annotations
        with matching names.
        """
        self.modality = modality

        self.ext = ext

        self.dataset = dataset

        self.indices = None

        self.path_images = path_images

        self.train = train
        
        ## Kaist multispectral dataset (clean empty bbox from annotation)
        if(self.dataset == 'kaist'):

            if(train == True):
                self.indices = [49, 50, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 74, 75, 76, 77, 80, 81, 82, 83, 84, 86, 87, 88, 89, 90, 107, 108, 109, 135, 136, 137, 138, 163, 164, 165, 166, 167, 188, 189, 192, 193, 194, 195, 196, 197, 198, 208, 209, 210, 211, 212, 213, 222, 223, 224, 225, 229, 230, 232, 233, 234, 235, 236, 237, 244, 245, 246, 247, 249, 250, 258, 259, 260, 261, 268, 269, 270, 271, 272, 273, 274, 309, 310, 311, 312, 313, 357, 358, 359, 367, 368, 369, 370, 376, 377, 378, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 420, 421, 427, 428, 429, 430, 431, 432, 433, 440, 492, 493, 494, 495, 496, 497, 509, 510, 511, 512, 513, 514, 515, 516, 517, 518, 519, 520, 521, 522, 523, 529, 530, 531, 532, 540, 541, 542, 543, 545, 546, 547, 553, 554, 555, 556, 557, 558, 559, 560, 597, 598, 599, 600, 601, 611, 612, 613, 614, 615, 616, 617, 618, 640, 641, 642, 643, 644, 645, 646, 647, 648, 649, 650, 651, 652, 653, 654, 655, 656, 657, 658, 698, 699, 700, 701, 702, 703, 704, 705, 706, 707, 708, 709, 710, 711, 712, 735, 736, 737, 738, 739, 740, 741, 742, 760, 761, 766, 767, 768, 771, 772, 773, 774, 775, 776, 777, 778, 779, 780, 781, 782, 783, 784, 785, 786, 787, 788, 789, 790, 791, 792, 793, 794, 795, 796, 797, 798, 799, 800, 801, 802, 803, 804, 805, 806, 814, 815, 816, 817, 818, 819, 820, 852, 853, 854, 855, 856, 857, 858, 876, 877, 883, 932, 933, 938, 939, 945, 946, 947, 948, 949, 950, 951, 952, 953, 954, 955, 956, 957, 958, 959, 960, 961, 962, 963, 964, 965, 966, 967, 968, 969, 970, 971, 972, 973, 974, 975, 976, 977, 978, 979, 980, 981, 982, 983, 984, 985, 986, 987, 988, 989, 990, 991, 992, 993, 994, 996, 997, 998, 999, 1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023, 1024, 1025, 1026, 1033, 1034, 1037, 1038, 1049, 1050, 1052, 1053, 1057, 1058, 1059, 1060, 1066, 1067, 1068, 1069, 1070, 1071, 1096, 1100, 1101, 1117, 1118, 1125, 1126, 1131, 1137, 1138, 1139, 1140, 1141, 1142, 1143, 1144, 1145, 1146, 1147, 1148, 1149, 1150, 1151, 1152, 1153, 1154, 1155, 1156, 1157, 1158, 1159, 1183, 1184, 1185, 1232, 1233, 1234, 1235, 1236, 1237, 1238, 1239, 1258, 1262, 1263, 1264, 1265, 1269, 1270, 1271, 1272, 1273, 1274, 1275, 1276, 1287, 1288, 1289, 1290, 1291, 1292, 1293, 1294, 1295, 1296, 1297, 1298, 1299, 1300, 1301, 1302, 1303, 1304, 1305, 1306, 1307, 1308, 1309, 1310, 1311, 1312, 1313, 1331, 1332, 1333, 1334, 1335, 1336, 1337, 1338, 1339, 1340, 1341, 1342, 1343, 1344, 1345, 1346, 1347, 1357, 1358, 1362, 1363, 1364, 1365, 1366, 1367, 1368, 1369, 1370, 1371, 1372, 1373, 1374, 1375, 1376, 1377, 1378, 1379, 1385, 1386, 1387, 1388, 1389, 1390, 1391, 1392, 1393, 1394, 1395, 1396, 1397, 1398, 1399, 1400, 1401, 1402, 1403, 1404, 1405, 1406, 1407, 1408, 1409, 1410, 1411, 1412, 1434, 1435, 1436, 1443, 1444, 1445, 1446, 1447, 1448, 1449, 1450, 1451, 1453, 1454, 1455, 1456, 1457, 1458, 1459, 1460, 1461, 1462, 1463, 1464, 1465, 1466, 1467, 1468, 1469, 1470, 1471, 1472, 1473, 1474, 1475, 1476, 1478, 1479, 1480, 1481, 1482, 1484, 1485, 1486, 1487, 1488, 1489, 1490, 1491, 1492, 1493, 1494, 1495, 1502, 1503, 1504, 1505, 1506, 1507, 1508, 1509, 1511, 1512, 1513, 1514, 1517, 1518, 1519, 1520, 1521, 1522, 1523, 1524, 1525, 1526, 1527, 1528, 1529, 1530, 1531, 1532, 1533, 1534, 1535, 1536, 1537, 1538, 1539, 1540, 1541, 1542, 1544, 1545, 1547, 1548, 1549, 1550, 1551, 1552, 1554, 1555, 1556, 1557, 1558, 1559, 1560, 1562, 1563, 1564, 1566, 1567, 1569, 1570, 1572, 1573, 1574, 1575, 1576, 1580, 1584, 1585, 1586, 1591, 1592, 1593, 1594, 1595, 1596, 1597, 1598, 1599, 1600, 1601, 1602, 1606, 1607, 1611, 1612, 1613, 1614, 1615, 1616, 1617, 1618, 1619, 1622, 1623, 1624, 1625, 1626, 1627, 1628, 1629, 1630, 1631, 1637, 1651, 1663, 1664, 1665, 1666, 1667, 1668, 1669, 1670, 1671, 1672, 1673, 1674, 1675, 1676, 1677, 1690, 1691, 1692, 1695, 1696, 1697, 1698, 1699, 1700, 1701, 1702, 1703, 1704, 1705, 1706, 1707, 1708, 1709, 1710, 1711, 1712, 1713, 1714, 1715, 1716, 1717, 1718, 1719, 1720, 1724, 1725, 1726, 1727, 1728, 1729, 1730, 1731, 1732, 1735, 1736, 1737, 1744, 1745, 1750, 1751, 1752, 1820, 1821, 1822, 1823, 1824, 1825, 1841, 1842, 1843, 1852, 1853, 1871, 1872, 1873, 1874, 1875, 1876, 1877, 1882, 1883, 1884, 1885, 1886, 1887, 1888, 1889, 1890, 1891, 1892, 1893, 1894, 1895, 1896, 1897, 1898, 1899, 1900, 1901, 1902, 1903, 1904, 1905, 1906, 1907, 1908, 1909, 1910, 1911, 1912, 1913, 1914, 1917, 1918, 1919, 1920, 1921, 1922, 1923, 1924, 1925, 1926, 1927, 1928, 1929, 1930, 1931, 1932, 1933, 1934, 1935, 1939, 1940, 1941, 1942, 1943, 1944, 1945, 1946, 1947, 1948, 1968, 1969, 1970, 1971, 1972, 1973, 1974, 1975, 1976, 1977, 1978, 2012, 2020, 2036, 2041, 2042, 2049, 2050, 2051, 2052, 2053, 2054, 2055, 2056, 2057, 2058, 2059, 2060, 2061, 2062, 2063, 2064, 2065, 2066, 2067, 2068, 2069, 2070, 2071, 2072, 2073, 2074, 2075, 2076, 2077, 2078, 2079, 2080, 2081, 2082, 2083, 2084, 2085, 2086, 2087, 2088, 2089, 2090, 2091, 2092, 2093, 2094, 2095, 2096, 2097, 2098, 2099, 2100, 2101, 2102, 2103, 2104, 2105, 2106, 2107, 2108, 2109, 2110, 2111, 2112, 2113, 2114, 2115, 2116, 2117, 2118, 2127, 2128, 2129, 2130, 2133, 2134, 2150, 2153, 2154, 2155, 2156, 2160, 2161, 2162, 2163, 2178, 2179, 2183, 2197, 2198, 2199, 2207, 2208, 2209, 2215, 2220, 2223, 2224, 2226, 2227, 2228, 2235, 2240, 2241, 2250, 2251, 2252, 2253, 2254, 2255, 2256, 2257, 2258, 2259, 2260, 2261, 2262, 2263, 2264, 2265, 2266, 2267, 2268, 2269, 2270, 2271, 2272, 2273, 2274, 2275, 2276, 2277, 2278, 2279, 2280, 2281, 2282, 2283, 2284, 2285, 2286, 2287, 2288, 2297, 2298, 2321, 2322, 2323, 2324, 2332, 2333, 2338, 2339, 2340, 2341, 2343, 2344, 2345, 2346, 2367, 2368, 2369, 2370, 2371, 2372, 2373, 2374, 2375, 2376, 2377, 2378, 2379, 2380, 2381, 2382, 2383, 2384, 2385, 2386, 2396, 2397, 2398, 2399, 2400, 2401, 2402, 2403, 2404, 2405, 2408, 2409, 2410, 2411, 2412, 2413, 2414, 2415, 2416, 2417, 2418, 2419, 2420, 2421, 2422, 2423, 2424, 2425, 2426, 2427, 2428, 2429, 2430, 2431, 2432, 2433, 2434, 2435, 2436, 2437, 2438, 2439, 2440, 2441, 2442, 2446, 2447, 2448, 2459, 2462, 2463, 2464, 2465, 2466, 2467, 2468, 2469, 2470, 2471, 2472, 2473, 2476, 2477, 2490, 2491, 2492, 2498, 2499]

            self.list_names = sorted(Utils().open_txt_file(
                                Path(self.path_images + '/' + ('train-all-20-rgb.txt' if (self.modality == 'rgb' or self.modality == 'both') 
                                                               else 'train-all-20-ir.txt')) if train == True else
                                Path(self.path_images + '/' + ('test-all-20-rgb.txt' if (self.modality == 'rgb' or self.modality == 'both') 
                                                               else 'test-all-20-ir.txt')), 
                                self.path_images)
                            )
            
        elif(self.dataset == 'llvip'):

            self.list_names = [x.split('.jpg')[0] for x in sorted(glob.glob(os.path.join(self.path_images, 
                                     'visible' if (self.modality == 'rgb' or self.modality == 'both') else 'infrared',
                                     'train' if train == True else 'test',
                                     '*.jpg'
                                    )))]

        elif(self.dataset == 'flir'):

            self.list_names = sorted(Utils().open_txt_file(
                    Path(self.path_images + '/' + 'align_train.txt' if train == True else
                    Path(self.path_images + '/' + 'align_validation.txt')), 
                    self.path_images)
                )

            self.list_names = [os.path.join(
                                self.path_images,
                                'JPEGImages',
                                x.split(self.path_images)[-1] if self.modality == 'infrared' 
                                else x.split(self.path_images)[-1].split('PreviewData')[0] + 'RGB',
                            ) for x in self.list_names]

        self.transforms = transforms


    def __len__(self):
        """
        Utility to get the length of the dataset according to the number of images.
        """
        return len(self.indices) if self.indices is not None else len(self.list_names)
    

    def __getitem__(self, index):
        """
        getitem is the most important function of this dataset. The function is
        called by PyTorch's DataLoader for creating the minibatchs. The image at
        position ``index`` will be returned by matching RGB, IR, and annotation
        files names. Data augmentation is also applied if the argument ``transforms``
        was provided.
        The function returns a dictionary with the keys ``image``, ``image_bbox``,
        ``ir``, and ``ir_bbox``.
        """
        index = self.indices[index] if self.indices is not None else index

        # take the name prefix
        _name = self.list_names[index]

        # obtain the full path for every modality and annotation file
        path_images = _name + self.ext

        img = (Image.open(path_images).convert("RGB") if self.modality == 'rgb' 
                                                    else Image.open(path_images).convert('L')
        )

        annot = Utils.get_bbox(_name + ".xml", self.dataset, self.train)
        bboxes = annot["bboxes"]
        labels = annot["labels"]

        target = {
            "boxes" : torch.from_numpy(bboxes).squeeze().view(-1, 4), 
            "labels": torch.from_numpy(labels).view(-1),
            "path_image": path_images, 
        }

        # apply data augmentation if provided
        if self.transforms is not None:
            img = self.transforms((img))

        # return the dict
        return img, target
    

# Pytorch dataset creation for loading both RGB and IR images with bbox annotations
class MultiModalDetectionDataset(SingleModalDetectionDataset):
    """
    MultiModalDetectionDataset is a PyTorch dataset implementation for loading
    RGB or IR annotated data. This implementation uses ``imageio``
    for reading the images, ``numpy`` for reading the raw IR data and ``xml``
    for parsing the annotations files. ps: First, we consider that the bbox are aligned (same bbox for ir and rbg).
    """
    def __init__(self, dataset, path_images_rgb, path_images_ir, modality=None, transforms_rgb=None, transforms_ir=None,
                ext=".png", train=True):
        """
        The constructor receives the path to RGB images and the path to IR npy.
        The folders are assumed to contain both images/raw IR and ``xml`` annotations
        with matching names.
        """
        ## path_images = any of the modalities, just to bypass the constructor
        super().__init__(dataset=dataset, path_images=path_images_rgb, modality=modality, 
                                                    transforms=None, ext=ext, train=train)

        self.list_names_rgb = self.list_names

        ## Kaist multispectral dataset (clean empty bbox from annotation)
        if(self.dataset == 'kaist'):

                self.list_names_ir = sorted(Utils().open_txt_file(
                        Path(path_images_ir + '/' + ('train-all-20-ir.txt')) if train == True else
                        Path(path_images_ir + '/' + 'test-all-20-ir.txt'), 
                        path_images_ir)
                    )
                
        elif(self.dataset == 'llvip'):

                self.list_names_ir = [x.split('.jpg')[0] for x in sorted(glob.glob(os.path.join(self.path_images, 
                                    'infrared',
                                    'train' if train == True else 'test',
                                    '*.jpg'
                                )))]

        elif(self.dataset == 'flir'):

            self.list_names = sorted(Utils().open_txt_file(
                    Path(self.path_images + '/' + 'align_train.txt' if train == True else
                    Path(self.path_images + '/' + 'align_validation.txt')), 
                    self.path_images)
                )

            self.list_names_ir = [os.path.join(
                                self.path_images,
                                'JPEGImages',
                                x.split(self.path_images)[-1]
                            ) for x in self.list_names]
            
        self.transforms_rgb = transforms_rgb 
        self.transforms_rgb = transforms_ir 


    def __getitem__(self, index):
        """
        getitem is the most important function of this dataset. The function is
        called by PyTorch's DataLoader for creating the minibatchs. The image at
        position ``index`` will be returned by matching RGB, IR, and annotation
        files names. Data augmentation is also applied if the argument ``transforms``
        was provided.
        The function returns a dictionary with the keys ``image``, ``image_bbox``,
        ``ir``, and ``ir_bbox``.
        """
        index = self.indices[index] if self.indices is not None else index

        # take the name prefix
        _name_rgb = self.list_names_rgb[index]
        _name_ir = self.list_names_ir[index]

        # read image/ir and their corresponding annotations
        item = {
            "img_rgb": imageio.imread(_name_rgb + self.ext), 
            "img_ir": np.asarray(Image.open(_name_ir + ('.jpeg' if self.dataset == 'flir' else self.ext)).convert('L')), 
            "annot_rgb": Utils.get_bbox((_name_ir if self.dataset == 'flir' else _name_rgb) + ".xml", self.dataset, self.train), 
            "annot_ir": Utils.get_bbox(_name_ir + ".xml", self.dataset, self.train), 
            "name_rgb": _name_rgb, 
            "name_ir": _name_ir
        }

        # return the dict
        return ToTensor(modality=self.modality)(item)
    
    
    def get_name(self, index):
        return self.list_names_rgb[index], self.list_names_ir[index]