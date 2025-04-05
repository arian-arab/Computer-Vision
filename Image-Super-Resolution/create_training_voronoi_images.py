import numpy as np
np.random.seed(42)

########### create Voronoi images ###########
colors = np.random.randint(0, 255, size=(32, 3), dtype=np.int32)
colors = [tuple(i) for i in colors]

y_train_regions = [np.random.randint(10,20) for i in range(800)]
y_tune_regions = [np.random.randint(10,20) for i in range(200)]
y_test_regions = [np.random.randint(10,20) for i in range(100)]

from voronoi import *
y_train = [generate(width = 256, height = 256, regions = y_train_regions[i], colors = colors, color_algorithm = ColorAlgorithm.no_adjacent_same,border_size = 5) for i in range(800)]
y_tune = [generate(width = 256, height = 256, regions = y_tune_regions[i], colors = colors, color_algorithm = ColorAlgorithm.no_adjacent_same,border_size = 5) for i in range(200)]
y_test = [generate(width = 256, height = 256, regions = y_test_regions[i], colors = colors, color_algorithm = ColorAlgorithm.no_adjacent_same,border_size = 5) for i in range(100)]
 
x_train = np.array([cv2.resize(i, (16, 16), interpolation=cv2.INTER_AREA) for i in y_train])
x_tune = np.array([cv2.resize(i, (16, 16), interpolation=cv2.INTER_AREA) for i in y_tune])
x_test = np.array([cv2.resize(i, (16, 16), interpolation=cv2.INTER_AREA) for i in y_test])

y_train = np.array(y_train)
y_tune = np.array(y_tune)
y_test = np.array(y_test)

y_train_regions = np.array(y_train_regions)
y_tune_regions = np.array(y_tune_regions)
y_test_regions = np.array(y_test_regions)

np.save('voronoi_imgs',[x_train,x_tune,x_test,
                        y_train,y_tune,y_test,
                        y_train_regions,y_tune_regions,y_test_regions])

